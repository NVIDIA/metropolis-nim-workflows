# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

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
        gr.Radio(["Yes", "No"], label="Is this an Alert?")
    ],
    outputs="text",
    title="NVIDIA NIM AI Agent",
    description="Enter a prompt and specify if it's an alert."
)

# Launch the interface
interface.launch()
