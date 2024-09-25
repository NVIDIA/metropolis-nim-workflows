<!--
SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# Agentic RAG-Enabled Video Interaction Pipeline with Morpheus, RIVA and VIA Microservices

This directory provides a step-by-step guide to setting up an Agentic RAG workflow using Morpheus, RIVA NIM, and VIA Microservices. Additionally, it includes a sample use case demonstrating how this workflow can enhance accessibility for the visually impaired.

<div align="center">
  <img src="notebooks/images/Video%20Agentic%20RAG%20with%20VIA.png" width="900">
</div>

## Prerequisites

### 1. Clone repository
```bash
git clone https://github.com/NVIDIA/metropolis-nim-workflows
cd metropolis-nim-workflows/via_workflows/video_agentic_rag_with_morpheus_riva
```
### 2. API Keys

To run the workflow, you will need access to the following keys:
- NVIDIA API Key from `build.nvidia.com`. These are necessary to access hosted NIMs like RIVA ASR, TTS and LLMs which are a key component of this workflow.
- Serp API Key. To allow the agent to perform internet searches.
- OpenAI API Key to call GPT-4o which is used by VIA Microservices. (Not required if you are using a different VLM)

Add the first two keys to the default .env file in the ```video_agentic_rag_with_morpheus_riva``` folder.

### 3. NVIDIA Containers

#### 3.1 Morpheus

You will need to have a `Morpheus 24.06` docker container built and present in the environment. 

This notebook has originally been designed to run with the NVIDIA AI Enterprise Morpheus container from NGC:

```bash
docker pull nvcr.io/nvidia/morpheus/morpheus:24.06-runtime
```

If you do not have access to NVIDIA AI Enterprise containers, you can follow instructions to build from source at the [Morpheus Repository](https://github.com/nv-morpheus/Morpheus/tree/branch-24.03).

If you are using a Morpheus version that is not `v24.06-runtime`, please update the version argument in the `docker-compose.yml` file as follows:

```bash
      args:
        - MORPHEUS_CONTAINER=${MORPHEUS_CONTAINER:-nvcr.io/nvidia/morpheus/morpheus}
        - MORPHEUS_CONTAINER_VERSION=${MORPHEUS_CONTAINER_VERSION:-v24.06-runtime}
```


#### 3.2 VIA Microservices

* Apply for [VIA Microservices Developer Preview](https://developer.nvidia.com/visual-insight-agent-early-access) to get access to VIA container image.
* Pull the docker image

```bash
docker pull nvcr.io/metropolis/via-dp/via-engine:2.0-dp
```

### 4. RIVA Python Client

Download Python client code by cloning Python [Client Repository](https://github.com/nvidia-riva/python-clients). This is required to call the [RIVA NIM](https://build.nvidia.com/nvidia/parakeet-ctc-1_1b-asr/api).

```bash
git clone https://github.com/nvidia-riva/python-clients.git
```


## Deploying the VIA Microservice

First, open the `vlm-container/run_via.sh` file in a text editor. Then, set the following API keys in the file:

```bash
export NVIDIA_API_KEY=<KEY> #build.nvidia.com
export OPENAI_API_KEY=<KEY> #OpenAI for GPT-4o
 ```

Then, run the following commands in a new terminal window:

```bash
cd vlm-container
bash run_via.sh
```

Once the initialization is complete, you are ready to run the demo. 


## Running the Jupyter Notebook

Open a new terminal and run the following commands to launch the notebook:

```bash
cd metropolis-nim-workflows/via_workflows/video_agentic_rag_with_morpheus_riva
docker compose up jupyter
```

Once launched, you should see a link in the output to connect to the JupyterLab server. Open this link in your web browser to access the content. For example:
```  
vlm-jupyter-jupyter-1  |     To access the server, open this file in a browser:
vlm-jupyter-jupyter-1  |         file:///root/.local/share/jupyter/runtime/jpserver-1-open.html
vlm-jupyter-jupyter-1  |     Or copy and paste one of these URLs:
vlm-jupyter-jupyter-1  |         http://localhost:8888/lab?token=a5aa264bd4cc18d6311ec6cba589be2060c86a9cb0d715f7
vlm-jupyter-jupyter-1  |         http://127.0.0.1:8888/lab?token=a5aa264bd4cc18d6311ec6cba589be2060c86a9cb0d715f7
```

Once connected to the JupyterLab server, you can navigate to the `notebooks` directory and open the `vlm-agent-demo.ipynb` Notebook. The notebook contains the instructions and all of the necessary content to run the workflow.

### Stopping the Container

To stop the container, use the following command:

```bash
docker compose down
```

Also kill the VIA container. 