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

import logging

from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents.agent import AgentExecutor
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS

from morpheus.llm import LLMEngine
from morpheus.llm.nodes.extracter_node import ExtracterNode
from morpheus.llm.nodes.langchain_agent_node import LangChainAgentNode
from src.llm_service import LLMService
from src.langchain_llm_client_wrapper import LangchainLLMClientWrapper
from morpheus.llm.task_handlers.simple_task_handler import SimpleTaskHandler

from src.checklist_node import ChecklistNode
from src.config import EngineAgentConfig
from src.config import EngineConfig
from src.tools import VideoEventQuery
from src.summary_node import VideoSummaryNode
from src.serp_api_wrapper import MorpheusSerpAPIWrapper


def build_agent_executor(
    config: EngineAgentConfig, handle_parsing_errors=False
) -> AgentExecutor:

    llm_service = LLMService.create(
        config.model.service.type, **config.model.service.model_dump(exclude={"type"})
    )

    llm_client = llm_service.get_client(**config.model.model_dump(exclude={"service"}))

    # Wrap the Morpheus client in a LangChain compatible wrapper
    langchain_llm = LangchainLLMClientWrapper(client=llm_client)

    # Initialize a SerpAPIWrapper object to perform internet searches.
    search = MorpheusSerpAPIWrapper(max_retries=10)

    # tools = load_tools(["serpapi", "llm-math"], llm=llm)
    tools: list[Tool] = [
        Tool(
            name="Internet Search",
            func=search.run,  # Synchronous function for running searches.
            coroutine=search.arun,  # Asynchronous coroutine for running searches.
            description="Useful for when you want to search external facts or state of the world.",
        ),
    ]

    if config.video.file_id is not None:

        # Load the SBOM
        video_util = VideoEventQuery(
            file_id=config.video.file_id,
            start=config.video.start_timestamp,
            end=config.video.end_timestamp,
            port=config.video.vlm_port,
        )

        tools.append(
            Tool(
                name="FPV Video QA System",
                func=video_util.video_search,
                description=(
                    "Useful for when you want a summary of all the times a certain event occurred in a video or a description of a certain item in a video."
                    "The input should be the description of the event you want to search for or the item you want to describe, only."
                    'For example, searches can be "Describe the contents of the refridgerator in the video", or '
                    '"One person shaking another\'s hand". The string you provide will be passed to a vision language model for '
                    " summarization of a video. If the returned response does not contain information about what you asked for, assume it is not visible currently and use Text QA instead."
                ),
            )
        )

    if config.text_db is not None:
        embeddings = HuggingFaceEmbeddings(
            model_name=config.text_db.embedding_model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": False},
        )

        # load code vector DB
        code_vector_db = FAISS.load_local(
            folder_path=config.text_db.faiss_dir,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        code_qa_tool = RetrievalQA.from_chain_type(
            llm=langchain_llm,
            chain_type="stuff",
            retriever=code_vector_db.as_retriever(),
        )
        tools.append(
            Tool(
                name="Environment Text QA System",
                func=code_qa_tool.run,
                description=(
                    "Useful for when you want to search for things about your environment or past actions "
                    "that are not found using the FPV Video QA System. If objects are not in view in the video, use this tool to search past views."
                    "Input should be a description of the environment/action you want to search for or semantically similar questions. "
                ),
            )
        )

    sys_prompt = (
        "You are a very powerful assistant who visually impaired users navigate and understand the world around them. "
        " You will be given a checklist of action items to investigate from a first person video feed emulating the users fiels of view."
        "Your role is to walk through a provided checklist and answer each item in the checklist. "
        " Information about the objects in the view or field of view is made available to you via a Video QA tool and Text QA tool."
    )

    if handle_parsing_errors:
        agent_executor = initialize_agent(
            tools,
            langchain_llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=config.verbose,
            handle_parsing_errors="Check your output. Make sure you're using the right Action/Action input syntax.",
        )
    else:
        agent_executor = initialize_agent(
            tools,
            langchain_llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=config.verbose,
        )

    agent_executor.agent.llm_chain.prompt.template = (
        sys_prompt
        + " "
        + agent_executor.agent.llm_chain.prompt.template.replace(
            "Answer the following questions as best you can.",
            (
                "If the input is not a question, formulate it into a question first. "
                "Include intermediate thought in the final answer."
            ),
        ).replace(
            "Use the following format:",
            (
                "Use the following format (start each response with one of the following prefixes): "
                "Question, Thought, Action, Action Input, Final Answer). "
                "If you are making an action, wait for a response to the action input before making an observation."
                "Every response must contain at least one action (and thoughts and observations if you have them), but you cannot have both a final answer and an action in a response."
                "Actions must always be only the name of the toosl available to you ('FPV Video QA System,' 'Environment Text QA System', or 'Internet Search'). Action input must only contain the exact input, do not provide any text following that in your response. Its generally recommended to search both QA tools available to you instead of sticking with just one. If needed, try using the internet search to validate things like the date, time, weather, or facts about the world."
                "Always end your response with either an action, or a final answer. Do not include an paranthesis around your prefixes."
            ),
        )
    )

    return agent_executor


def build_acc_llm_engine(config: EngineConfig, handle_parsing_errors=True) -> LLMEngine:

    summary_service = LLMService.create(
        config.agent.model.service.type,
        **config.agent.model.service.model_dump(exclude={"type"})
    )

    engine = LLMEngine()

    engine.add_node("extracter", node=ExtracterNode())

    engine.add_node(
        "checklist", inputs=["/extracter"], node=ChecklistNode(config=config.checklist)
    )

    engine.add_node(
        "agent",
        inputs=[("/checklist")],
        node=LangChainAgentNode(
            agent_executor=build_agent_executor(
                config=config.agent, handle_parsing_errors=handle_parsing_errors
            )
        ),
    )

    engine.add_node(
        "summary",
        inputs=[
            ("/checklist", "checklist_inputs"),
            ("/agent", "checklist_responses"),
        ],
        node=VideoSummaryNode(
            llm_client=summary_service.get_client(
                **config.agent.model.model_dump(exclude={"service", "type"})
            )
        ),
    )

    handler_inputs = ["/summary/checklist", "/summary/summary"]
    handler_outputs = ["checklist", "summary"]

    # Add our task handler
    engine.add_task_handler(
        inputs=handler_inputs, handler=SimpleTaskHandler(output_columns=handler_outputs)
    )

    return engine
