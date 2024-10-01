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
import requests, base64
import os 
from concurrent.futures import ThreadPoolExecutor
from time import time 

import numpy as np 
import cv2
from PIL import Image 
from tqdm import tqdm 

class NVCLIP:

    def __init__(self, api_key, base_url="https://integrate.api.nvidia.com/v1/embeddings"):
        """Initialize with NVCLIP url and API key"""
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        
    def _encode_image(self, image, resize=True):
        """ Resize image, encode as jpeg to shrink size then convert to b64 for upload """

        if isinstance(image, str): #file path
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image): #pil image 
            image = image.convert("RGB")
        elif isinstance(image, np.ndarray): #cv2 / np array image 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        else:
            print(f"Unsupported image input: {type(image)}")
            return None 
            
        if resize:
            image = image.resize((336,336)) #could also centercrop or pad square and resize 

        buf = io.BytesIO() #temporary buffer to save processed image 
        image.save(buf, format="JPEG")
        image = buf.getvalue()
        image_b64 = base64.b64encode(image).decode()
        return image_b64


    def _combine_responses(self, responses):
        """Combine multiple NVCLIP request responses."""
        
        if len(responses) == 0:
            raise Exception("Received no responses to combine.")

        usage = {"num_images": 0, "prompt_tokens": 0, "total_tokens": 0}
        model = responses[0]["model"]
        object = responses[0]["object"]

        combined_responses = {"object":object, "data":[], "usage":usage, "model":model}

        for resp in responses:
            """sum the usage stats and combine data lists """
            usage = {key: (usage[key] + resp["usage"][key]) for key in usage.keys()} #"sum up usage"
            combined_responses["data"].extend(resp["data"])

        for i in range(len(combined_responses["data"])):
            """reindex the embeddings"""
            combined_responses["data"][i]["index"] = i

        return combined_responses

    def __call__(self, items, chunk=64, workers=16, resize=True):
        """Embed images or text. Items should be a list of string or local filepaths to images. The items are chunked and spread across N worker threads. NVCLIP will accept upto 64 items in one request. """
    
        with ThreadPoolExecutor(max_workers=workers) as executor:

            responses = []
            futures = []
            print("Submitting Requests")
            for i in tqdm(range(0,len(items),chunk)):
                item_chunk = items[i:min(len(items), i+chunk)] #each request will send chunk number of items 
                embed_items = []
                for item in item_chunk:
                    if os.path.isfile(item):
                        embed_items.append(f"data:image/jpeg;base64,{self._encode_image(item, resize=resize)}") #image 
                    else:
                        embed_items.append(item) #string 
                payload = {"input": embed_items, "model":"nvidia/nvclip"}
                future = executor.submit(requests.post, self.base_url, headers=self.headers, json=payload)
                futures.append(future)
                
            print("Collecting Responses")
            for future in tqdm(futures):
                responses.append(future.result().json())

        return self._combine_responses(responses) #combine all responses and return 


if __name__ == "__main__":
    """Example Usage"""
    nvclip = NVCLIP("nvapi-***")
    input = ["test"] * 256
    start = time()
    response = nvclip(input, workers=4)
    print(f"Time: {time() - start}")
    print(response)
