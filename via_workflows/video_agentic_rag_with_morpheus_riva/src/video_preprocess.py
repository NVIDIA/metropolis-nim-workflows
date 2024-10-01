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

import requests
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import jsonschema
import json
from tqdm import tqdm
import nbformat
import os
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_community.document_loaders.blob_loaders.schema import Blob
from langchain_core.embeddings import Embeddings


class Handler:
    def __init__(self, name) -> None:
        self.name = name


def validate_config(parsed_yaml, schema_json_filepath):
    try:
        with open(schema_json_filepath) as f:
            spec_schema = json.load(f)
        jsonschema.validate(parsed_yaml, spec_schema)
    except jsonschema.ValidationError as e:
        # print(f"{'.'.join([str(p) for p in e.absolute_path])}: {e.message}")
        raise ValueError(
            f"Invalid config file: {'.'.join([str(p) for p in e.absolute_path])}: {e.message}"
        )


class StorageHandler(Handler):
    def __init__(self, name) -> None:
        super().__init__(name)

    def add_summary(self, summary, metadata):
        pass

    def get_text_data(self):
        pass

    def search(self, search_query):
        pass


class MilvusDBHandler(StorageHandler):
    """Handler for Milvus DB which stores the video embeddings mapped using
    the summary text embeddings which can be used for retrieval.

    Implements StorageHandler class
    """

    def __init__(
        self,
        collection_name="text_summaries",
        host="127.0.0.1",
        port="31014",
        embedding_model_name="all-MiniLM-L6-v2",
    ) -> None:
        super().__init__("milvus_db_handler")
        self.connection = {"host": host, "port": port}
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vector_db = Milvus(
            embedding_function=self.embeddings,
            connection_args=self.connection,
            collection_name=self.collection_name,
            auto_id=True,
            drop_old=False,
        )

    def add_summary(self, summary: str, metadata: dict):
        doc = Document(page_content=summary, metadata=metadata)
        self.vector_db.add_documents([doc])

    def get_text_data(self, fields=["*"], filter="pk > 0"):
        if self.vector_db.col:
            results = self.vector_db.col.query(expr=filter, output_fields=fields)
            return [
                {k: v for k, v in result.items() if k != "pk"} for result in results
            ]
        else:
            return []

    def search(self, search_query, top_k=1):
        search_results = self.vector_db.similarity_search(search_query, k=top_k)
        return [result.metadata for result in search_results]

    def drop_data(self):
        if self.vector_db.col:
            self.vector_db.col.delete(expr="pk > 0")

    def drop_collection(self):
        self.vector_db = Milvus(
            embedding_function=self.embeddings,
            connection_args=self.connection,
            collection_name=self.collection_name,
            auto_id=True,
            drop_old=True,
        )

    def get_embedding_field_pairs(self, fields=["vector", "text"], filter="pk > 0"):
        if self.vector_db.col:
            vector_list = self.get_text_data(fields)
            tuples = []

            for v in vector_list:
                text = v["text"]
                embedding = v["vector"]
                print(text)
                tuples.append((text, embedding))

            return tuples

        else:
            return []


class VideoPreprocessor:

    def __init__(self, host="host.docker.internal", vlm_port=31012, milvus_port=31014):
        self.host = host
        self.vlm_port = vlm_port
        self.milvus_port = milvus_port
        self.summarize_endpoint = f"http://{self.host}:{str(self.vlm_port)}/summarize"
        self.upload_endpoint = f"http://{self.host}:{str(self.vlm_port)}/files"
        self.milvus_handler = MilvusDBHandler(port=self.milvus_port, host=self.host)
        self.file_id = None

    def upload_video(self, file_path):
        files = {
            "file": open(file_path, "rb"),
        }
        data = {
            "purpose": "vision",
            "media_type": "video",
        }

        response = requests.post(self.upload_endpoint, files=files, data=data)
        response.raise_for_status()

        file_info = response.json()
        self.file_id = file_info["id"]
        return self.file_id

    def perform_initial_summarization(self, chunk_duration=60, chunk_overlap=0):
        assert self.file_id is not None, "Must call the upload_video function first."

        data = {
            "id": self.file_id,
            "model": "gpt-4o",  # Specify the model to use
            "response_format": {"type": "text"},
            "chunk_duration": chunk_duration,
            "chunk_overlap_duration": chunk_overlap,
            "max_tokens": 512,
            "prompt": f"""
                    You are an expert at world understanding and description. Your task is to capture, in as much detail as possible, the events in a provided ego-centric video.
                    Be sure to capture as much description as possible about the environment, people, objects, and actions performed in the video. Please be explicit about what kinds of objects 
                    Also note shifts in direction of the camera and the relative change in location of the objects in the environment that result. 
                    """,
        }

        response = requests.post(self.summarize_endpoint, json=data)
        response.raise_for_status()

    def cache_embeddings_to_faiss(self, faiss_dir="./vdb"):
        cache_tuples = self.milvus_handler.get_embedding_field_pairs()
        # Ensure the output directory exists
        os.makedirs(faiss_dir, exist_ok=True)

        embeddings = [t[0] for t in cache_tuples]

        # Create FAISS index
        db = FAISS.from_embeddings(cache_tuples, embeddings)

        # Save to directory
        db.save_local(faiss_dir)

    def preprocess(
        self, file_path, chunk_duration=60, chunk_overlap=0, faiss_dir="./vdb"
    ):
        print("*Pre-processing video.")
        print("\t**Uploading video")
        file_id = self.upload_video(file_path=file_path)
        print("\t**Creating dense captions. This could take a few minutes.")
        self.perform_initial_summarization(
            chunk_duration=chunk_duration, chunk_overlap=chunk_overlap
        )
        print("\t**Caching captions in FAISS.")
        self.cache_embeddings_to_faiss(faiss_dir=faiss_dir)

        return file_id
