# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import io
import uuid
import json
import zipfile
from pathlib import Path
from PIL import Image
import tempfile
import requests


class OCDRNET:

    def __init__(self, api_key, url="https://ai.api.nvidia.com/v1/cv/nvidia/ocdrnet"):

        self.api_key = api_key
        self.url = url
        self.header_auth = f"Bearer {self.api_key}"

    def _upload_asset(self, image_path, description):
        """
        Uploads an asset to the NVCF API.
        :param image_path: The image path
        :param description: A description of the asset

        """

        assets_url = "https://api.nvcf.nvidia.com/v2/nvcf/assets"

        headers = {
            "Authorization": self.header_auth,
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        s3_headers = {
            "x-amz-meta-nvcf-asset-description": description,
            "content-type": f"image/jpeg",
        }

        payload = {"contentType": f"image/jpeg", "description": description}

        response = requests.post(assets_url, headers=headers, json=payload, timeout=30)

        response.raise_for_status()

        asset_url = response.json()["uploadUrl"]
        asset_id = response.json()["assetId"]

        # Convert image to jpeg before uploading
        image = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()  # temporary buffer to save image
        image.save(buf, format="JPEG")

        # upload image
        response = requests.put(
            asset_url,
            data=buf.getvalue(),
            headers=s3_headers,
            timeout=300,
        )

        response.raise_for_status()
        return uuid.UUID(asset_id)

    def __call__(self, image_path, output_folder=None):
        """Extract characters from input image. Response will be returned in JSON format."""

        # optional folder to save outputs. otherwise will go to temp directory
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)

        asset_id = self._upload_asset(image_path, "Input Image")

        inputs = {"image": f"{asset_id}", "render_label": False}
        asset_list = f"{asset_id}"

        headers = {
            "Content-Type": "application/json",
            "NVCF-INPUT-ASSET-REFERENCES": asset_list,
            "NVCF-FUNCTION-ASSET-IDS": asset_list,
            "Authorization": self.header_auth,
        }

        response = requests.post(self.url, headers=headers, json=inputs)

        if output_folder:
            zip_path = Path(output_folder) / (Path(image_path).stem + ".zip")
        else:
            temp_dir = tempfile.mkdtemp()
            zip_path = Path(temp_dir) / (Path(image_path).with_suffix(".zip").name)

        # Results are returned as a zip. Write to disk and extract.
        with open(zip_path, "wb") as out:
            out.write(response.content)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(zip_path.parent / zip_path.stem)

        zip_path.unlink()  # delete temp zip
        os.sync()
        result_path = zip_path.with_suffix("")
        return self.parse_output(result_path)

    def _calculate_centroid(self, polygon):
        """Used to sort detections"""
        x_coords = [polygon[key] for key in polygon.keys() if key.startswith("x")]
        y_coords = [polygon[key] for key in polygon.keys() if key.startswith("y")]
        centroid_x = sum(x_coords) / len(x_coords)
        centroid_y = sum(y_coords) / len(y_coords)
        return centroid_x, centroid_y

    def parse_output(self, results_path, sort=True):
        """Get the ocd results and sort results."""
        ocd_response_file = list(Path(results_path).glob("*.response"))[
            0
        ]  # get response file
        with ocd_response_file.open("r") as file:
            data = json.load(file)

        if sort:
            """Sort the results based on the polygons from topleft to bottoms right."""
            for entry in data["metadata"]:  # calculate centroid for each polygon
                polygon = entry["polygon"]
                centroid = self._calculate_centroid(polygon)
                entry["centroid"] = centroid

            data["metadata"].sort(
                key=lambda x: (x["centroid"][1], x["centroid"][0])
            )  # sort based on centroid
            for entry in data["metadata"]:  # remove centroid
                del entry["centroid"]

        return data
