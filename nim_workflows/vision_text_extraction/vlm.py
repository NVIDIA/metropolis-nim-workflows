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
import base64
import uuid
from PIL import Image
import requests


class VLM:
    def __init__(self, url, api_key):
        """Provide NIM API URL and an API key"""
        self.api_key = api_key
        self.model = url.split("/")[-2:]
        self.model = "/".join(self.model)
        # llama URLs are slightly different from other VLMs
        if url in [
            "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct",
            "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct",
        ]:
            url = url + "/chat/completions"

        self.url = url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    def _upload_asset(self, image_path, description="image"):
        """
        Uploads an asset to the NVCF API.
        :param image_path: The image path
        :param description: A description of the asset

        """

        assets_url = "https://api.nvcf.nvidia.com/v2/nvcf/assets"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
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

    def __call__(self, prompt, image_path, system_prompt=None):
        """Call VLM object with the prompt and path to image"""

        asset_id = self._upload_asset(image_path, "Input Image")
        asset_list = f"{asset_id}"
        headers = {
            "Content-Type": "application/json",
            "NVCF-INPUT-ASSET-REFERENCES": asset_list,
            "NVCF-FUNCTION-ASSET-IDS": asset_list,
            "Authorization": f"Bearer {self.api_key}",
        }

        # For simplicity, the image will be appended to the end of the prompt.
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f'{prompt} Here is the image: <img src="data:image/jpeg;asset_id,{asset_id}"/>"',
                },
            ],
            "max_tokens": 1024,
            "temperature": 0.20,
            "top_p": 0.70,
            "stream": False,
            "model": self.model,
        }
        if system_prompt:
            payload["messages"].insert(0, {"role": "system", "content": system_prompt})
        response = requests.post(self.url, headers=headers, json=payload)
        print(response)
        print(response.text)
        response = response.json()
        reply = response["choices"][0]["message"]["content"]
        return reply  # return reply and the full response
