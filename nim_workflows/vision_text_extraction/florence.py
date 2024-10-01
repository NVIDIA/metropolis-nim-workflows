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

# The model can perform 14 different vision language model and computer vision tasks. The input ```content``` field should be formatted as ```"<TASK_PROMPT><text_prompt (only when needed)><img>"```.
# Users need to specify the task type at the beginning. Image supports both base64 and NvCF asset id. Some tasks require a text prompt, and users need to provide that after image. Below are the examples for each task.
# For <CAPTION_TO_PHRASE_GROUNDING>, <REFERRING_EXPRESSION_SEGMENTATION>, <OPEN_VOCABULARY_DETECTION>, users can change the text prompt as other descriptions.
# For <REGION_TO_SEGMENTATION>, <REGION_TO_CATEGORY>, <REGION_TO_DESCRIPTION>, the text prompt is formatted as <loc_x1><loc_y1><loc_x2><loc_y2>, which is the normalized coordinates from region of interest bbox. x1=int(top_left_x_coor/width*999), y1=int(top_left_y_coor/height*999), x2=int(bottom_right_x_coor/width*999), y2=int(bottom_right_y_coor/height*999).
import os
import sys
import zipfile
import requests
import io
from PIL import Image
import uuid
from pathlib import Path
import json
import tempfile


class Florence:

    def __init__(
        self, api_key, base_url="https://ai.api.nvidia.com/v1/vlm/microsoft/florence-2"
    ):
        self.api_key = api_key
        self.base_url = base_url
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

    def __call__(self, task_id, image, prompt=""):
        asset_id = self._upload_asset(image, "Input Image")
        prompts = [
            "<CAPTION>",
            "<DETAILED_CAPTION>",
            "<MORE_DETAILED_CAPTION>",
            "<OD>",
            "<DENSE_REGION_CAPTION>",
            "<REGION_PROPOSAL>",
            f"<CAPTION_TO_PHRASE_GROUNDING>{prompt}",
            f"<REFERRING_EXPRESSION_SEGMENTATION>{prompt}",
            f"<REGION_TO_SEGMENTATION>{prompt}",
            f"<OPEN_VOCABULARY_DETECTION>{prompt}",
            f"<REGION_TO_CATEGORY>{prompt}",
            f"<REGION_TO_DESCRIPTION>{prompt}",
            "<OCR>",
            "<OCR_WITH_REGION>",
        ]

        content = f'{prompts[task_id]}<img src="data:image/jpeg;asset_id,{asset_id}" />'

        payload = {"messages": [{"role": "user", "content": content}]}

        headers = {
            "Content-Type": "application/json",
            "NVCF-INPUT-ASSET-REFERENCES": str(asset_id),
            "NVCF-FUNCTION-ASSET-IDS": str(asset_id),
            "Authorization": self.header_auth,
            "Accept": "application/json",
        }

        # Send the request to the NIM API.
        response = requests.post(self.base_url, headers=headers, json=payload)

        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = Path(temp_dir) / "result.zip"

            with open(zip_path, "wb") as out:
                out.write(response.content)

            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(zip_path.parent / zip_path.stem)

            result_path = zip_path.with_suffix("")

            ocd_response_file = list(Path(result_path).glob("*.response"))[
                0
            ]  # get response file
            with ocd_response_file.open("r") as file:
                data = json.load(file)

        return data


if __name__ == "__main__":
    image_path = "license.jpg"
    florence = Florence("nvapi-***")
    resp = florence(12, image_path)
    print(resp)
