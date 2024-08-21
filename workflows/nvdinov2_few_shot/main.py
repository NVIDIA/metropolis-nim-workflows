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

from pathlib import Path
from collections import Counter
import argparse

import gradio as gr
from pymilvus import MilvusClient, DataType
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from sklearn.manifold import TSNE
import numpy as np

from bokeh.palettes import Dark2_5 as palette
import itertools


from nvclip import NVCLIP
from nvdinov2 import NVDINOv2

# global state
embedding_model_g = None
client_g = None
classes_g = []


def _update_plot(perplexity):
    """Generate plot based on latest vectors from db"""
    global client_g

    # get all embeddings vectors from milvus
    res = client_g.query(
        collection_name="few_shot", output_fields=["vector", "class_label"], limit=1000
    )
    vectors = np.array([x["vector"] for x in res])
    class_labels = np.array([x["class_label"] for x in res])

    # Need atleast 2 vectors to plot
    if len(vectors) < 2:
        return None

    # labels = np.array([1,2,3] * 30)
    tsne = TSNE(
        n_components=2,
        perplexity=min(len(vectors) - 1, perplexity),
        learning_rate=200,
        early_exaggeration=30,
        n_iter=2000,
        random_state=42,
        metric="cosine",
    )
    embedding_2d = tsne.fit_transform(vectors)

    p = figure(
        title="NVCLIP Embedding Visualization",
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
    )
    colors = itertools.cycle(palette)
    for n in np.unique(class_labels):
        indices = np.where(class_labels == n)
        n_vectors = embedding_2d[indices]
        x = n_vectors[:, 0]
        y = n_vectors[:, 1]

        source = ColumnDataSource(dict(x=x, y=y))

        p.scatter("x", "y", source=source, size=8, color=next(colors), legend_label=n)
    return p


def add_sample(sample, class_label):
    """Embed and store few shot examples in vector db"""
    global client_g
    # embed
    print(class_label)
    data = []
    response = embedding_model_g(sample)
    for i, vector in enumerate(response):
        data_sample = {
            "vector": vector,
            "class_label": class_label,
            "image_path": sample[i],
        }
        data.append(data_sample)
    client_g.insert(collection_name="few_shot", data=data)

    return _update_plot(7), None, None


def add_class(class_name):
    """Add new user defined classes"""
    global classes_g
    classes_g.append(class_name)
    print(classes_g)
    return [[x] for x in classes_g], ""


def update_plot(perplexity):
    return _update_plot(perplexity)


def update_class_dropdown(evt: gr.EventData):
    """Update Drop Down for class selection"""
    global classes_g
    print("updating dropdown")
    return gr.Dropdown(choices=classes_g, interactive=True)


def classify(sample, neighbors):
    """Use KNN to classify new images"""
    global client_g

    # embed sample
    embedding = embedding_model_g([sample])[0]

    # get nearest neighbors
    results = client_g.search(
        collection_name="few_shot",
        data=[embedding],
        limit=neighbors,
        output_fields=["class_label", "image_path"],
    )

    # determine majority class
    labels = [x["entity"]["class_label"] for x in results[0]]
    neighor_images = [x["entity"]["image_path"] for x in results[0]]
    label_counter = Counter(labels)

    # get label
    label = label_counter.most_common()[0][0]
    return label, neighor_images


def main(port):
    """Build  and launch Gradio UI"""
    with gr.Blocks() as demo:
        # title
        gr.HTML(
            '<h1 style="color: #6aa84f; font-size: 250%;">NVDINOv2 Few Shot Classification</h1>'
        )

        with gr.Row():
            gr.Markdown("## Step 1) Add Classes")
        with gr.Row():
            with gr.Column():

                class_name_tb = gr.Textbox(label="Class Name")

            with gr.Column():
                added_classes = gr.Dataframe(
                    datatype="str", col_count=1, headers=["Classes"]
                )
        with gr.Row():
            add_class_btn = gr.Button("Add Class")

        # Add dataset samples
        with gr.Row():
            gr.Markdown(
                """
                ## Step 2) Add Sample Images
                """
            )
        with gr.Row():
            sample_image_upload = gr.File(
                file_count="multiple", label="Upload Sample Images"
            )
            class_dropdown = gr.Dropdown(classes_g, label="Select Class")
        with gr.Row():
            add_sample_btn = gr.Button("Add Samples")

        # Plot data
        with gr.Row():
            gr.Markdown("## Step 3) Plot Embeddings")
        with gr.Row():
            gr.Markdown(
                "*A minimum of 2 sample images must be uploaded before the plot can be generated."
            )

        with gr.Row():
            with gr.Column():
                # graph output
                database_plot = gr.Plot()
                perplexity_slider = gr.Slider(
                    1,
                    100,
                    value=15,
                    step=1,
                    label="Perplexity",
                    info="Adjusts the clustering",
                )
        with gr.Row():
            update_plot_btn = gr.Button("Update Plot")

        # Inference
        with gr.Row():
            gr.Markdown("## Step 4) Classify New Image")
        with gr.Row():
            with gr.Column():
                # classify options
                inference_image_upload = gr.Image(type="filepath", label="Upload Image")
                gr.Markdown("### KNN Classification Options")
                neighbor_slider = gr.Slider(
                    1,
                    50,
                    value=1,
                    label="Neighbors",
                    step=1,
                    info="Maximum number of neighbors for classification",
                )

            with gr.Column():
                classification_result = gr.Textbox(label="Class Label Prediction")
                neighbor_gallery = gr.Gallery(
                    label="Nearest Neighbors", object_fit="scale-down", preview=False
                )
        with gr.Row():
            classify_btn = gr.Button("Classify Image")

        # button callbacks
        add_class_btn.click(
            fn=add_class, inputs=class_name_tb, outputs=[added_classes, class_name_tb]
        )
        add_sample_btn.click(
            fn=add_sample,
            inputs=[sample_image_upload, class_dropdown],
            outputs=[database_plot, sample_image_upload, class_dropdown],
        )
        update_plot_btn.click(
            fn=update_plot, inputs=[perplexity_slider], outputs=database_plot
        )
        classify_btn.click(
            fn=classify,
            inputs=[inference_image_upload, neighbor_slider],
            outputs=[classification_result, neighbor_gallery],
        )

        # dropdown updates
        class_dropdown.focus(fn=update_class_dropdown, outputs=class_dropdown)

        demo.launch(server_port=port)


if __name__ == "__main__":
    """Launches the Few Shot Classification Application. Starts by setting up local Milvus Vector DB"""
    parser = argparse.ArgumentParser(description="NVDINOv2 Few Shot Classification")
    parser.add_argument("api_key", type=str, help="NVIDIA NIM API Key")
    parser.add_argument(
        "model",
        choices=["nvclip", "nvdinov2"],
        help="Embedding model to use for few shot classification.",
    )
    parser.add_argument(
        "--gradio_port", type=int, default=7860, help="Port to run Gradio UI"
    )
    args = parser.parse_args()

    # setup embedding model. NVCLIP or NVDINOv2
    if args.model == "nvclip":
        embedding_model_g = NVCLIP(args.api_key)
    elif args.model == "nvdinov2":
        embedding_model_g = NVDINOv2(args.api_key)
    else:
        raise Exception(f"Unsupported Embedding Model: {args.model}")

    # remove db if it exists
    Path.unlink("./localdb.db", missing_ok=True)

    # create database
    client_g = MilvusClient("localdb.db")

    # setup db schema
    schema = MilvusClient.create_schema(auto_id=True)
    schema.add_field(field_name="sample_id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(
        field_name="class_label", datatype=DataType.VARCHAR, max_length=200
    )
    if args.model == "nvclip":
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024
        )  # nvclip 1024d
    else:
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1536
        )  # NVDINO 1536d
    schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=500)

    # setup index
    index_params = client_g.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")

    if not client_g.has_collection(collection_name="few_shot"):
        # create collection in database. This will associate a vector with the metadata
        client_g.create_collection(
            collection_name="few_shot", schema=schema, index_params=index_params
        )

    main(args.gradio_port)
