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

import numpy as np
import cv2
from threading import Thread
from PIL import Image
import io
import requests, base64


class VLM:

    def __init__(self, url, api_key, callback):
        self.model = url.split("/")[-2:]
        self.model = "/".join(self.model)

        # llama URLs are slightly different from other VLMs
        if url in [
            "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct",
            "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct",
        ]:
            url = url + "/chat/completions"

        self.url = url

        self.busy = False
        self.reply = ""
        self.api_key = api_key
        self.callback = callback

    def _encode_image(self, image):
        """Resize image, encode as jpeg to shrink size then convert to b64 for upload"""

        if isinstance(image, str):  # file path
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):  # pil image
            image = image.convert("RGB")
        elif isinstance(image, np.ndarray):  # cv2 / np array image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        else:
            print(f"Unsupported image input: {type(image)}")
            return None

        image = image.resize(
            (336, 336)
        )  # centercrop or pad square then resize are other strategies
        buf = io.BytesIO()  # temporary buffer to save processed image
        image.save(buf, format="JPEG")
        image = buf.getvalue()
        image_b64 = base64.b64encode(image).decode()
        assert len(image_b64) < 180_000, "Image too large to upload."
        return image_b64

    def _call(self, message, image=None, callback_args={}):

        try:
            image_b64 = self._encode_image(image)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            }

            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": f'{message} Here is the image: <img src="data:image/jpeg;base64,{image_b64}" />',
                    }
                ],
                "max_tokens": 128,
                "temperature": 0.20,
                "top_p": 0.70,
                "stream": False,
                "model": self.model,
            }

            response = requests.post(self.url, headers=headers, json=payload)
            self.reply = response.json()["choices"][0]["message"]["content"]
        finally:
            self.busy = False

        self.callback(message, self.reply, **callback_args)

    def __call__(self, message, image=None, **kwargs):
        if self.busy:
            print("VLM is busy")
            return None

        else:
            self.busy = True
            Thread(target=self._call, args=(message, image, kwargs)).start()
