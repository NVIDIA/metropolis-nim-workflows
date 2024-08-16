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

import gradio as gr
import requests


def send_prompt(prompt, loop=False):
    # Send prompt
    params = {"query": prompt, "alert": loop}
    url = f"http://localhost:5432/query"
    response = requests.get(url, params=params)
    return response.text


def gradio_interface(prompt, is_alert):
    if is_alert is None:
        is_alert = "No"
    loop = True if "yes" in is_alert.lower() else False
    try:
        return send_prompt(prompt, loop=loop)
    except Exception as e:
        return f"Could not send prompt. Ensure demo is running. Error: {e}"


# Create Gradio interface
interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Radio(["Yes", "No"], label="Is this an Alert?"),
    ],
    outputs="text",
    title="NVIDIA NIM AI Agent",
    description="Enter a prompt and specify if it's an alert.",
)

# Launch the interface
interface.launch()
