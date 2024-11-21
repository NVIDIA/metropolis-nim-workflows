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

import argparse
import os
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
import gradio as gr
from pymilvus import MilvusClient

from nvclip import NVCLIP

milvus_client_g = None
nvclip_g = None
embeddings_2d_g = None
image_paths_g = None


def highlighted_plot(vector_ids=None):
    """Plot 2D embeddings and highlight specific vector_ids in red."""

    global embeddings_2d_g
    global image_paths_g

    if vector_ids is None:
        vector_ids = []

    # Create plot
    p = figure(
        title="NV-CLIP Embedding Visualization",
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        tooltips="""
            <div>
                <div>
                    <img src="@image_path" height="100" alt="file_name" style="float: left; margin: 0px 15px 15px 0px;"/>
                </div>
            </div>
        """,
    )

    highlight_embeddings = embeddings_2d_g[vector_ids]
    highlight_image_paths = [image_paths_g[i] for i in vector_ids]
    source = ColumnDataSource(
        dict(x=embeddings_2d_g[:, 0], y=embeddings_2d_g[:, 1], image_path=image_paths_g)
    )
    highlight_source = ColumnDataSource(
        dict(
            x=highlight_embeddings[:, 0],
            y=highlight_embeddings[:, 1],
            image_path=highlight_image_paths,
        )
    )
    p.scatter("x", "y", source=source, size=8, legend_label="Image Embedding")
    if vector_ids != 0:
        p.scatter(
            "x",
            "y",
            source=highlight_source,
            size=10,
            color="red",
            legend_label="Similar Images",
        )
    return p


def query_callback(query):
    """Callback for text and image search. Returns closest images based on vector similarity and updated plot."""
    global milvus_client_g
    global nvclip_g
    global image_paths_g

    if query == "":
        return [], highlighted_plot()

    resp = nvclip_g([query])
    query_vector = resp["data"][0]["embedding"]
    results = milvus_client_g.search(
        collection_name="collection",
        data=[query_vector],
        limit=20,
        output_fields=["file_name", "id"],
    )
    image_paths = [x["entity"]["file_name"] for x in results[0]]
    vector_ids = [x["entity"]["id"] for x in results[0]]

    image_paths = [image_paths_g[i] for i in vector_ids]

    return image_paths, highlighted_plot(vector_ids)


def main(image_folder, gradio_port):
    """Create file server and launch Gradio UI"""

    global image_paths_g

    # create file server for gradio ui. Allows images to be displayed quickly.
    print("creating file server")
    app = FastAPI()
    app.mount("/images", StaticFiles(directory=image_folder), name="images")

    # create gradio UI
    print("creating gradio ui")
    with gr.Blocks(theme=gr.themes.Monochrome()) as blocks:
        gr.HTML(
            '<h1 style="color: #6aa84f; font-size: 250%;">NV-CLIP Multimodal Search</h1>'
        )

        gr.HTML('<h2">Search by text or image.</h1>')
        with gr.Row():
            text_query = gr.Textbox(placeholder="Search Query", show_label=False)
            image_upload = gr.Image(type="filepath")

        gr.HTML(
            '<h2">The most similar images to the search query will populate the gallery and be highlighted in the embedding plot.</h1>'
        )
        with gr.Row():
            gallery = gr.Gallery(
                value=[image_paths_g[i] for i in range(20)],
                columns=4,
                height="750px",
                object_fit="scale-down",
                preview=False,
            )
            embedding_plot = gr.Plot(value=highlighted_plot())

        text_query.change(
            query_callback, text_query, [gallery, embedding_plot], show_progress=False
        )
        image_upload.upload(
            query_callback, image_upload, [gallery, embedding_plot], show_progress=False
        )

    # mount gradio UI to fastapi server
    app = gr.mount_gradio_app(app, blocks, path="/")

    print("launching server")
    print(f"Access Gradio UI at http://localhost:{gradio_port}")
    uvicorn.run(app, host="localhost", port=gradio_port, reload=False, log_level="info")


if __name__ == "__main__":
    """Launches the Semantic Search Application. Starts by generating embeddings and storing to local vector DB with Milvus."""
    parser = argparse.ArgumentParser(description="NV-CLIP Multimodal Search")
    parser.add_argument(
        "image_folder",
        type=str,
        help="Path to folder of images to embed and search over",
    )
    parser.add_argument("api_key", type=str, help="NVIDIA NIM API Key")
    parser.add_argument(
        "--nvclip_url",
        type=str,
        default="https://integrate.api.nvidia.com/v1/embeddings",
        help="URL to NV-CLIP NIM",
    )
    parser.add_argument(
        "--gradio_port", type=int, default=7860, help="Port to run Gradio UI"
    )
    args = parser.parse_args()

    # connect to NVCLIP NIM
    nvclip_g = NVCLIP(args.api_key)

    # Setup Database
    print("creating database client")
    db_name = str(Path(args.image_folder).name) + ".db"
    milvus_client_g = MilvusClient(db_name)

    # try to create database collection. If collection already exists it will use the previously saved embeddings.
    print("creating database collection")
    if not milvus_client_g.has_collection(collection_name="collection"):
        # create collection in database. This will associate a vector with image path
        milvus_client_g.create_collection(
            collection_name="collection", dimension=1024  # NVCLIP output dimension
        )

        # embed images
        print("embedding images")
        image_names = os.listdir(args.image_folder)
        image_paths_g = [str(Path(args.image_folder) / x) for x in image_names]
        resp = nvclip_g(image_paths_g)

        # parse results
        print("parsing embedding results")
        image_embedding_data = []
        for i, data in enumerate(resp["data"]):
            data = {"id": i, "vector": data["embedding"], "file_name": image_paths_g[i]}
            image_embedding_data.append(data)

        # insert embeddings
        res = milvus_client_g.insert(
            collection_name="collection", data=image_embedding_data
        )
        print("database setup complete")

    # Project vector db embeddings to 2D for plotting
    print("generating 2D projection")
    res = milvus_client_g.query(
        collection_name="collection",
        output_fields=["vector", "file_name"],
        limit=len(os.listdir(args.image_folder)),
    )
    vectors = np.array([x["vector"] for x in res])
    file_names = [x["file_name"] for x in res]
    image_paths_g = [
        f"http://localhost:{args.gradio_port}/images/{Path(x).name}" for x in file_names
    ]
    # use TSNE to project embeddings to 2D
    tsne = TSNE(
        n_components=2,
        perplexity=20,
        learning_rate=200,
        early_exaggeration=30,
        max_iter=2000,
        random_state=42,
        metric="cosine",
    )
    embeddings_2d_g = tsne.fit_transform(vectors)
    print("launching gradio UI")

    # Start Gradio UI
    main(args.image_folder, args.gradio_port)
