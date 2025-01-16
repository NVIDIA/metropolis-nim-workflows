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

import ast
import logging
import re
from textwrap import dedent

from morpheus.llm import LLMLambdaNode
from morpheus.llm import LLMNode
from morpheus.llm.nodes.llm_generate_node import LLMGenerateNode
from morpheus.llm.nodes.prompt_template_node import PromptTemplateNode
from src.llm_service import LLMService

from .config import EngineChecklistConfig

logger = logging.getLogger(__name__)

checklist_prompt_template = dedent(
    """You are an expert assistant to the visually imparied, helping to navigate, plan, and complex questions about the environment around you given a first person view video of the environment. Your objective is to add a "Checklist" section containing steps to use for a downstream agent to follow to achieve the result of a given query.\
For each checklist item, start with an action verb, making it clear and actionable

**Context**:
Understand the world around you and planning or decribing certain scenarios are a complex task which could require multiple steps. Creating a checklist of to-dos to achieve a certain query helps reduce complexity while maintaining throoughness. 

**Example Format**:
Below is a format for an example that illustrate transforming a query into an actionable checklist, 

Example Query:
Where are the dirty dishes on the counter I'm looking at?

Example Scene Evaluation Checklist:
[
"Verify your environment: Check if the user is looking at a counter. Verify if there are dishes on the counter.",
"Produce absolute directions: Identifying where in the video (field of view) the object is helps provide direction (e.g. to the left)",
"Produce relative landmarks: Identifying the location of the dishes in the video is a mixutre of absolute directions like "to the left", but also relative to other objects in the field of view."
"Plan: Idenitfying if anything in the field of view needs to be done in order to access the object in the query such as moving things out of the way, etc."
]

**Criteria**:
- Checklists must relate to the information in the specific query.
- Checklists must include checks for mitigating conditions or queries for objects not present in the field of view.
- Avoid repetitive objectives between checklist items. The more concise your list, the better.
- Please be explicit about the objects you reference in the checklist. For example, avoid ambiguities like referencing "table" which could indicate "coffee table" or "dining table".

**Procedure**:
[
"Understand what the query is asking of you, and what common sense checks need to be done to validate.",
"Produce a checklist of action items or queries.",
"Format the checklist as comma separated list surrounded by square braces.",
"Output the checklist."
]

**Query Details:**
{{cve_details}}

**Checklist**: 

Please only provide the comma separated, python formatted list, no other text. """
).strip("\n")

parselist_prompt_template = dedent(
    """
Parse the following numbered checklist's contents into a python list in the format ['x', 'y', 'z'], a comma separated list surrounded by square braces. For example, the following checklist:

1. Check for notable vulnerable software vendors
2. Consider the network exposure of your Docker container

Should generate: ["Check for notable vulnerable software vendors", "Consider the network exposure of your Docker container"]

Checklist:
{{template}}

Limit the length of your list to at most 4 items, and avoid duplication of any tasks as much as possible. Please only provide the comma separated, python formatted, list."""
).strip("\n")

# Find all substrings that start and end with quotes, allowing for spaces before a comma or closing bracket
re_quote_capture = re.compile(
    r"""
        (['"])                    # Opening quote
        (                         # Start capturing the quoted content
            (?:\\.|[^\\])*?       # Non-greedy match for any escaped character or non-backslash character
        )                         # End capturing the quoted content
        \1                        # Matching closing quote
        (?=\s*,|\s*\])            # Lookahead for whitespace followed by a comma or closing bracket, without including it in the match
    """,
    re.VERBOSE,
)


def attempt_fix_list_string(s: str) -> str:
    """
    Attempt to fix unescaped quotes in a string that represents a list to make it parsable.

    Parameters
    ----------
    s : str
        A string representation of a list that potentially contains unescaped quotes.

    Returns
    -------
    str
        The corrected string where internal quotes are properly escaped, ensuring it can be parsed as a list.

    Notes
    -----
    This function is useful for preparing strings to be parsed by `ast.literal_eval` by ensuring that quotes inside
    the string elements of the list are properly escaped. It adds brackets at the beginning and end if they are missing.
    """
    # Check if the input starts with '[' and ends with ']'
    s = s.strip()
    if not s.startswith("["):
        s = "[" + s
    if not s.endswith("]"):
        s = s + "]"

    def fix_quotes(match):
        # Extract the captured groups
        quote_char, content = match.group(1), match.group(2)
        # Escape quotes inside the string content
        fixed_content = re.sub(r"(?<!\\)(%s)" % re.escape(quote_char), r"\\\1", content)
        # Reconstruct the string with escaped quotes and the same quote type as the delimiters
        return f"{quote_char}{fixed_content}{quote_char}"

    # Fix the quotes inside the strings
    fixed_s = re_quote_capture.sub(fix_quotes, s)

    return fixed_s


async def _parse_list(text: list[str]) -> list[list[str]]:
    """
    Asynchronously parse a list of strings, each representing a list, into a list of lists.

    Parameters
    ----------
    text : list of str
        A list of strings, each intended to be parsed into a list.

    Returns
    -------
    list of lists of str
        A list of lists, parsed from the input strings.

    Raises
    ------
    ValueError
        If the string cannot be parsed into a list or if the parsed object is not a list.

    Notes
    -----
    This function tries to fix strings that represent lists with unescaped quotes by calling
    `attempt_fix_list_string` and then uses `ast.literal_eval` for safe parsing of the string into a list.
    It ensures that each element of the parsed list is actually a list and will raise an error if not.
    """
    return_val = []

    for x in text:
        try:
            # Try to do some very basic string cleanup to fix unescaped quotes
            x = attempt_fix_list_string(x)

            # Only proceed if the input is a valid Python literal
            # This isn't really dangerous, literal_eval only evaluates a small subset of python
            current = ast.literal_eval(x)

            # Ensure that the parsed data is a list
            if not isinstance(current, list):
                raise ValueError(f"Input is not a list: {x}")

            # Process the list items
            for i in range(len(current)):
                if isinstance(current[i], list) and len(current[i]) == 1:
                    current[i] = current[i][0]

            return_val.append(current)
        except (ValueError, SyntaxError) as e:
            # Handle the error, log it, or re-raise it with additional context
            raise ValueError(f"Failed to parse input {x}: {e}")

    return return_val


class ChecklistNode(LLMNode):
    """
    A node that orchestrates the process of generating a checklist items.
    It integrates various nodes that handle lookup, prompting, generation, and parsing to produce an actionable checklist.
    """

    def __init__(self, *, config: EngineChecklistConfig):
        """
        Initialize the ChecklistNode with optional caching and a vulnerability endpoint retriever.

        Parameters
        ----------
        model_name : str, optional
            The name of the language model to be used for generating text, by default "gpt-3.5-turbo".
        cache_dir : str, optional
            The directory where the node's cache should be stored. If None, caching is not used.
        """
        super().__init__()

        self._config = config

        llm_service = LLMService.create(
            config.model.service.type,
            **config.model.service.model_dump(exclude={"type"}),
        )

        # Add a node to create a prompt for CVE checklist generation based on the CVE details obtained from the lookup
        # node
        self.add_node(
            "checklist_prompt",
            inputs=[("*", "*")],
            node=PromptTemplateNode(
                template=checklist_prompt_template, template_format="jinja"
            ),
        )

        # Instantiate a chat service and configure a client for generating responses to the checklist prompt
        llm_client_1 = llm_service.get_client(
            **config.model.model_dump(exclude={"service"})
        )
        self.add_node(
            "generate_checklist",
            inputs=["/checklist_prompt"],
            node=LLMGenerateNode(llm_client=llm_client_1),
        )

        # Add an output parser node to process the final generated checklist into a structured list
        self.add_node(
            "output_parser",
            inputs=["/generate_checklist"],
            node=LLMLambdaNode(_parse_list),
            is_output=True,
        )
