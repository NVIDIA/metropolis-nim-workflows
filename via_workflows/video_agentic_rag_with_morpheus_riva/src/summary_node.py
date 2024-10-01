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
import re

from morpheus.llm import LLMLambdaNode
from morpheus.llm import LLMNode
from morpheus.llm.nodes.llm_generate_node import LLMGenerateNode
from morpheus.llm.nodes.prompt_template_node import PromptTemplateNode
from morpheus.llm.services.llm_service import LLMClient


logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """Given the following results from search in a user's field of view, summarize the findings into clear, actionable paragraph that can be read to a visually impaired user as a description of their surroundings. 

Checklist and Findings:
{{response}}
"""


def get_checklist_item_string(item_num: int, item: dict) -> str:
    """
    Formats and returns a string containing the checklist item number, question, and answer.
    """
    question = remove_number_prefix(item['question'])
    answer = remove_number_prefix(item['response'])
    return f"- Checklist Item {item_num}: {question}\n  - Answer: {answer}"

def remove_number_prefix(text: str) -> str:
    """
    Removes number prefix, e.g. '1.' from a string and returns the modified string.
    """
    # Regular expression pattern to match 'number.' at the beginning of the string
    pattern = r'^\d+\.'

    # Strip any leading whitespace that could interfere with regex
    text = text.lstrip()
    # Remove the matching pattern (if found)
    text = re.sub(pattern, '', text)
    # Strip any remaining whitespace
    text = text.strip()
    return text


class VideoSummaryNode(LLMNode):
    """
    A node to summarize the results of the CVE tools checklist responses.
    """

    def __init__(self, *, llm_client: LLMClient):
        """
        Initialize the VideoSummaryNode with a selected model.

        Parameters
        ----------
        llm_client : LLMClient
            The LLM client to use for generating the summary.
        """
        super().__init__()

        async def build_summary_output(checklist_inputs: list[list[str]],
                                       checklist_responses: list[list[str]]) -> list[list[dict]]:

            combined_checklist = []

            for check_in, check_out in zip(checklist_inputs, checklist_responses):
                combined_checklist.append(
                    [{
                        "question": q, "response": r
                    } for q, r in zip(check_in, check_out) if q and r])

            return combined_checklist

        # Build the output for the checklist as a list of dictionaries
        self.add_node("checklist",
                      inputs=["checklist_inputs", "checklist_responses"],
                      node=LLMLambdaNode(build_summary_output),
                      is_output=True)

        # Concatenate the output of the checklist responses into a single string
        async def concat_checklist_responses(agent_q_and_a: list[list[dict]]) -> list[str]:
            concatted_responses = []

            for checklist in agent_q_and_a:
                checklist_str = '\n'.join([get_checklist_item_string(i + 1, item) for i, item in enumerate(checklist)])
                concatted_responses.append(checklist_str)

            return concatted_responses

        self.add_node('results', inputs=['/checklist'], node=LLMLambdaNode(concat_checklist_responses))

        self.add_node('summary_prompt',
                      inputs=[('/results', 'response')],
                      node=PromptTemplateNode(template=SUMMARY_PROMPT, template_format='jinja'))

        # Generate a summary from the combined checklist responses
        self.add_node("summary",
                      inputs=["/summary_prompt"],
                      node=LLMGenerateNode(llm_client=llm_client),
                      is_output=True)
