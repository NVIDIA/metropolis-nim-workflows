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

from vlm import VLM
from florence import Florence
from ocdrnet import OCDRNET
from openai import OpenAI

import re
import json


class TextExtraction:

    def __init__(self, api_key, vlm=None, llm=None, ocd=None, **kwargs):
        self.api_key = api_key
        self.vlm = vlm  # ["nvidia/neva-22b"]
        self.llm = llm  # ["nv-mistralai/mistral-nemo-12b-instruct"]
        self.ocd = ocd  # ["nvidia/ocdrnet", "microsoft/florence-2"]

        # VLM or LLM and OCD required
        if not self.vlm:
            if not (self.ocd and self.llm):
                raise Exception("VLM or OCD and LLM required.")

        self.vlm_system_prompt = kwargs.get(
            "vlm_system_prompt",
            "Your job is to inspect an image and fill out a form provided by the user. This form will be provided in JSON format and will include a list of fields and field descriptions. Inspect the image and do your best to fill out the fields in JSON format based on image. The JSON output should be in a JSON code block.",
        )
        self.llm_system_prompt = kwargs.get(
            "llm_system_prompt",
            "You are an AI assistant whose job is to inspect a string that may have json formatted output. This json format may not be correct so you must extract the json and make it properly formatted in a JSON block. You will be provided a list of keys that you must find in the input string. If you cannot find the associated value then put an empty string.",
        )

    def __call__(self, image, field_names, field_descriptions=None):
        """image - PIL image or file path"""

        field_descriptions = (
            [""] * len(field_names)
            if field_descriptions is None
            else field_descriptions
        )

        # Get Field Names and Descriptions in dict
        fields = {}
        for x in range(len(field_names)):
            fields[field_names[x]] = field_descriptions[x]

        # Stage 1: OCDR with OCDRNet or Florence
        if self.ocd is not None:
            # Setup OCD
            if self.ocd == "microsoft/florence-2":
                florence = Florence(self.api_key)
                ocd_response = florence(12, image)
                ocd_response = ocd_response["choices"][0]["message"]["content"]
            elif self.ocd == "nvidia/ocdrnet":
                ocdrnet = OCDRNET(self.api_key)
                ocd_response = ocdrnet(image)
                ocd_response = [x["label"] for x in ocd_response["metadata"]]
                ocd_response = " ".join(ocd_response)
        else:
            ocd_response = None

        # Stage 2: VLM Field Extraction
        if self.vlm is not None:
            # setup VLM
            vlm = VLM(self.vlm, self.api_key)

            # Form Prompt
            user_prompt = f"Here are the fields: {fields}. Fill out each field based on the image and respond in JSON format."
            if ocd_response:
                user_prompt = (
                    user_prompt
                    + f"To assist you with filling out the fields. The following text has been extract from the image: {ocd_response}"  # add OCDR output if available
                )
            vlm_response = vlm(user_prompt, image, system_prompt=self.vlm_system_prompt)
        else:
            vlm_response = None

        # Stage 3: LLM Post Processing
        if self.llm:
            llm_input = vlm_response if vlm_response else ocd_response
            # LLM call for fixing json formatting
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1", api_key=self.api_key
            )
            messages = [
                {"role": "system", "content": self.llm_system_prompt},
                {
                    "role": "user",
                    "content": f"I have a text string that may or may not be formatted in proper json. The response should have the following keys: [{[x for x in fields.keys()]}]. Please parse the response and match the key pair values and format it in JSON. Here is the response: {llm_input}",
                },
            ]
            completion = client.chat.completions.create(
                model=self.llm,
                messages=messages,
                temperature=0.2,
                top_p=0.7,
                max_tokens=1024,
                stream=False,
            )

            llm_response = completion.choices[0].message.content
        else:
            llm_response = None

        final_response = llm_response if llm_response else vlm_response

        # Extract the JSON part from the code block
        try:
            # Try to find json code block
            re_search = re.search(r"```json\n(.*?)\n```", final_response, re.DOTALL)
            if re_search:
                json_string = re_search.group(1)
            # If no code block then find curly braces
            else:
                left_index = final_response.find("{")
                right_index = final_response.rfind("}")
                json_string = final_response[left_index : right_index + 1]

            json_object = json.loads(json_string)
        except Exception as e:
            print(f"JSON Parsing Error: {e}")
            return {
                key: None for key in field_names
            }  # return empty expected dict with no values

        return json_object
