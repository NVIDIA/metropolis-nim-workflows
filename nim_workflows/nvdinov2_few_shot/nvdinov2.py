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

import io
import requests
import uuid
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from tqdm import tqdm


class NVDINOv2:

    def __init__(
        self, api_key, base_url="https://ai.api.nvidia.com/v1/cv/nvidia/nv-dinov2"
    ):
        """Initialize with NVCLIP url and API key"""
        self.base_url = base_url
        self.api_key = api_key
        self.header_auth = f"Bearer {self.api_key}"

    def _combine_responses(self, responses):
        pass

    def _upload_asset(self, image_path, description="input image"):
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
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image_path, Image.Image):
            image = image_path.convert("RGB")

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

    def _embed(self, image_path):
        asset_id = self._upload_asset(image_path)
        payload = {"messages": []}
        asset_list = f"{asset_id}"

        headers = {
            "Content-Type": "application/json",
            "NVCF-INPUT-ASSET-REFERENCES": asset_list,
            "NVCF-FUNCTION-ASSET-IDS": asset_list,
            "Authorization": self.header_auth,
        }

        response = requests.post(self.base_url, headers=headers, json=payload)

        return response.json()

    def __call__(self, image_paths, workers=16, return_meta=False):
        """Embeds images provided as a list of file paths or PIL images. Will use multiple worker threads to submit simultaneous requests. Returns full metadata or just a list of embeddings"""

        if isinstance(image_paths, str):
            image_paths = [image_paths]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            responses = []
            futures = []
            print("submitting requests")
            for i in tqdm(range(0, len(image_paths))):
                future = executor.submit(self._embed, image_paths[i])
                futures.append(future)

            print("collecting responses")
            for future in tqdm(futures):
                if return_meta:
                    responses.append(future.result())
                else:
                    responses.append(future.result()["metadata"][0]["embedding"])

        return responses
