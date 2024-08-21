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
import argparse
from textextraction import TextExtraction

# Values for dropdowns
AVAILABLE_OCDR = ["nvidia/ocdrnet", "microsoft/florence-2", None]
AVAILABLE_VLMS = {
    "nvidia/neva-22b": "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b",
    "microsoft/kosmos-2": "https://ai.api.nvidia.com/v1/vlm/microsoft/kosmos-2",
    "adept/fuyu-8b": "https://ai.api.nvidia.com/v1/vlm/adept/fuyu-8b",
    "google/paligemma": "https://ai.api.nvidia.com/v1/vlm/google/paligemma",
    "microsoft/phi-3-vision-128k-instruct": "https://ai.api.nvidia.com/v1/vlm/microsoft/phi-3-vision-128k-instruct",
    "None": None,
}
SUGGESTED_LLMS = [
    "nv-mistralai/mistral-nemo-12b-instruct",
    "mistralai/mixtral-8x22b-instruct-v0.1",
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.1-70b-instruct",
    None,
]

# global state
api_key_g = None


def extract_text(vlm, llm, ocd, structured_fields, image_input):
    field_names = [x[0] for x in structured_fields]
    field_descriptions = [x[1] for x in structured_fields]

    pipeline = TextExtraction(api_key_g, vlm, llm, ocd)
    response = pipeline(image_input, field_names, field_descriptions)
    response_table = [[key, value] for key, value in response.items()]
    return response_table


def main(port):
    with gr.Blocks() as demo:
        gr.HTML(
            '<h1 style="color: #6aa84f; font-size: 250%;">Vision NIMs for Structured Text Extraction</h1>'
        )
        with gr.Row():

            with gr.Column():
                gr.Markdown("## Model Options")
                gr.Markdown(
                    "Select a combination of VLMs, OCDR Models and LLMs to build the text extraction pipeline. OCDR and LLM are optional."
                )
                vlm_selection = gr.Dropdown(
                    choices=[x for x in AVAILABLE_VLMS.keys()],
                    label="VLM Selection",
                    info="VLM to extract user defined fields from Image.",
                    value=list(AVAILABLE_VLMS.keys())[0],
                )
                ocd_selection = gr.Dropdown(
                    choices=AVAILABLE_OCDR,
                    label="OCDR Selection",
                    info="OCDR model to accurately extract all text from the image.",
                    value=AVAILABLE_OCDR[0],
                )
                llm_selection = gr.Dropdown(
                    choices=SUGGESTED_LLMS,
                    label="LLM Selection",
                    info="LLM to post process the output and ensure it is in JSON format.",
                    value=SUGGESTED_LLMS[0],
                )

            with gr.Column():
                gr.Markdown("## User Defined Fields")
                gr.Markdown(
                    "Supply the fields to extract from the image. Optionally a field description can be added to give the models more context about what the field name means."
                )
                structured_fields = gr.DataFrame(
                    interactive=True,
                    headers=["Field Name", "Field Description"],
                    col_count=(2, "fixed"),
                    type="array",
                )
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Image Input")
                gr.Markdown("Supply an Input Image to extract fields from.")
                image_input = gr.Image(type="filepath")
            with gr.Column():
                gr.Markdown("## Structured Output")
                gr.Markdown("View the fields the Vision NIMs extracted from the image.")
                text_output = gr.DataFrame(
                    interactive=False,
                    type="array",
                    headers=["Field Name", "Extracted Value"],
                )

        submit_btn = gr.Button()

        submit_btn.click(
            fn=extract_text,
            inputs=[
                vlm_selection,
                llm_selection,
                ocd_selection,
                structured_fields,
                image_input,
            ],
            outputs=text_output,
        )

    demo.launch(server_port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structured Text Extraction")
    parser.add_argument("api_key", type=str, help="NVIDIA NIM API Key")
    parser.add_argument(
        "--gradio_port", type=int, default=7860, help="Port to run Gradio UI"
    )
    args = parser.parse_args()

    api_key_g = args.api_key

    main(args.gradio_port)
