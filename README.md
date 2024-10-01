# NVIDIA Metropolis Generative AI Workflows

Generative AI such as Large Language Models (LLMs) and Vision Language Models (VLMs) are used to build powerful workflows and agents to solve critical business problems. This repository hosts training materials, reference applications and workflows to build Visual AI Agents using NVIDIA NIM microservices. **You can get started for free with no GPU required!**

If you find this repository helpful, **leave a star** and **share the link** 🙂. 

## Table of Contents
- [News](#news-newspaper)
- [Getting Started](#getting-started-rocket) 
  - [Prerequisites](#prerequisites)
  - [NIM Workflows](#nim-microservice-workflows)
    - [VLM Alert System](nim_workflows/vlm_alerts/README.md)
    - [NV-CLIP Multimodal Search](nim_workflows/nvclip_multimodal_search/README.md)
    - [Vision Structured Text Extraction](nim_workflows/vision_text_extraction/README.md)
    - [NVDINOv2 Few Shot Classification](nim_workflows/nvdinov2_few_shot/README.md) 
  - [VIA Workflows](#via-microservice-workflows) 
    - [VIA Setup](via_workflows/README.md)
    - [Summarize videos with VIA Microservice](via_workflows/summarization_examples/)
    - [Agentic RAG with Morpheus, RIVA and VIA Microservices](via_workflows/video_agentic_rag_with_morpheus_riva/)
- [Related Resources](#related-resources) 
- [Contributors](#contributors-star) 


## News :newspaper:
- **9/25** - Llama 3.2 Vision NIM added to [build.nvidia.com](http://build.nvidia.com). Workflows updated to support Llama 3.2 Vision. 
- **9/24** - Added VIA workflows 
- **9/9** - VILA VLM NIM added to [build.nvidia.com](http://build.nvidia.com) 
- **8/21** - Added NVDINOv2 Few Shot Classification workflow
- **8/16** - Added Structured Text Extraction workflow
- **8/7** - Added NV-CLIP Semantic Search workflow
- **6/26** - Added VLM Alert workflow


## Getting Started :rocket:
To access NIM Microservices, visit [build.nvidia.com](https://build.nvidia.com) to create an account and generate an API key. Each new account can receive up to **5,000 free credits**. These credits will give you free access to preview APIs that allow you to use state of the art generative AI models such as Llama3.2 Vision, Nemotron, Mistral, VILA and much more with no GPU required! 

### Prerequisites 
1) Go to [build.nvidia.com](http://build.nvidia.com) and use your email to sign up. 

2) After making an account, you can get an API Key by selecting any of the available NIMs then in the example code section, click on "Get API Key" then "Generate Key". 

<div align="center">
  <img src="readme_assets/generate_api_key.png" width="700">
</div>

3) You will then see your API Key that will look something like "nvapi-xxx-xxxxxxxx_xxxxxxxxxxxxxxxx_xxxxxxxxxxx-xxxxxxx-xxxxxxxxxxxxxx". This full key is what will be used in the example notebooks and scripts. 

You can now continue to explore the workflows in the next sections. 

### NIM microservice workflows 

[NVIDIA NIMs](https://developer.nvidia.com/nim) are GPU-accelerated AI models wrapped in an easy to use REST API interface. To help developers get started, each NIM has a hosted preview API that is accessible after generating an API token from [build.nvidia.com](https://build.nvidia.com). The preview APIs can be used for **free** to develop and experiment with state of the art AI models including LLMs, VLMs, Embedding and CV models. 

The workflows listed under this section make use of the preview APIs and do **not** require a local GPU! You can run these workflows on nearly any computer and each one is quick and easy to launch. Each workflow includes a Jupyter notebook workshop that walks through how to build with NIM Preview APIs in Python. 

Follow the links below to start running the workflows: 

- [Learn how to use VLMs to automatically monitor a video stream for custom events.](nim_workflows/vlm_alerts/README.md)
- [Learn how to search images with natural language using NV-CLIP.](nim_workflows/nvclip_multimodal_search/README.md)
- [Learn how to combine VLMs, LLMs and CV models to build a robust text extraction pipeline.](nim_workflows/vision_text_extraction/README.md)
- [Learn how to use embeddings with NVDINOv2 and a Milvus VectorDB to build a few shot classification model.](nim_workflows/nvdinov2_few_shot/README.md)


### VIA microservice workflows 

[NVIDIA VIA Microservices](https://developer.nvidia.com/visual-insight-agent-early-access) are cloud-native building blocks to build AI agents capable of processing large amounts of live or archived videos and images with Vision-Language Models (VLM).

At a minimum VIA requires a NIM API Key and a local graphics card. A consumer RTX card is enough to get started.

First follow the VIA specific setup steps
- [VIA Setup](via_workflows/README.md)

Then you can explore the following VIA workflows from this repository: 
- [Summarize videos with VIA Microservice](via_workflows/summarization_examples/)
- [Agentic RAG with Morpheus, RIVA and VIA Microservices](via_workflows/video_agentic_rag_with_morpheus_riva/)

## Changelog :memo:
- rel-2.1: Add support for Llama 3.2 Vision 
- rel-2.0: Reorganize repository. Add VIA Workflows. Update READMEs.
- rel-1.3.1: Update VLM Alert workflow with VILA 
- rel-1.3: Add NVDINOv2 Few Shot Classification workflow.
- rel-1.2: Add Structured Text Extraction Workflow.
- rel-1.1: Add Websocket server output for VLM Alert workflow. Add NV-CLIP Semantic Search workflow.
- rel-1.0: Add VLM Alert workflow 

## Questions, Discussion, and Bugs :grey_question:
If you find any bugs, have questions or want to start a discussion around the workflows feel free to [file an issue](https://github.com/NVIDIA/metropolis-nim-workflows/issues). Any suggestions, feedback and new ideas are also welcome 🙂. 

## Related Resources :link:
Relevant technical blogs that explore building AI Agents. 
**Technical Blogs**      
- [Visual AI Agents for Jetson](https://developer.nvidia.com/blog/develop-generative-ai-powered-visual-ai-agents-for-the-edge/)  
- [VIA Deep Dive](https://developer.nvidia.com/blog/build-vlm-powered-visual-ai-agents-using-nvidia-nim-and-nvidia-via-microservices/)  

**Web Pages**    
Relevant web pages to find more information about NIMs, Metropolis and VIA. 
- [NIM microservices](https://build.nvidia.com)   
- [Metropolis Visual AI Agents](https://www.nvidia.com/en-us/use-cases/visual-ai-agents/)  
- [VIA EA Registration](https://developer.nvidia.com/visual-insight-agent-early-access)  

**GitHub Repositories**     
The following GitHub repositories include more examples of how to build with NIM microservices. 
- [NVIDIA Generative AI Examples (RAG and more NIM workflows)](https://github.com/NVIDIA/GenerativeAIExamples)  
- [NVIDIA Blueprints - Enterprise ready NIM based workflows](https://github.com/NVIDIA-NIM-Agent-Blueprints) 

**NVIDIA Developer Forums**
For questions and discussions, feel free to post under the appropriate topic on our developer forums. 
- [Visual AI Agent Topic](https://forums.developer.nvidia.com/c/accelerated-computing/intelligent-video-analytics/visual-ai-agent/680)  
- [NIM Topic](https://forums.developer.nvidia.com/c/ai-data-science/nvidia-nim/678)  

## Contributors :star:
Thanks to the following people for contributing to our workflows:
- [Samuel Ochoa](https://github.com/ssmmoo1)
- [Shubham Agrawal](https://github.com/shubham050300)
- [Dhruv Nandakumar](https://github.com/dnandakumar-nv)
- [Adeola Adesoba](https://github.com/Adeola-Adesoba)


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=NVIDIA/metropolis-nim-workflows&type=Date)](https://star-history.com/#NVIDIA/metropolis-nim-workflows&Date)
