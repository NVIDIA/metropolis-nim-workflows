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

import aiohttp
from langchain.pydantic_v1 import root_validator
from langchain.utilities.serpapi import SerpAPIWrapper
from langchain.utils.env import get_from_env

from src.async_http_utils import retry_async
from src.url_utils import url_join


class MorpheusSerpAPIWrapper(SerpAPIWrapper):

    base_url: str = "https://serpapi.com"
    max_retries: int = 10

    @root_validator()
    def validate_environment(cls, values: dict) -> dict:
        """Validate the base URL from the environment."""

        values["base_url"] = get_from_env(key="base_url", env_key="SERPAPI_BASE_URL", default=values["base_url"])

        # Build from the base class
        values = super().validate_environment(values)

        # Update the base URL for search_engine
        values["search_engine"].BACKEND = values["base_url"]

        return values

    @retry_async()
    async def _session_get_with_retry(self, session: aiohttp.ClientSession, url: str, params: dict) -> dict:

        async with session.get(url, params=params) as response:
            res = await response.json()
            return res

    # Override the method with hardcoded URL
    async def aresults(self, query: str) -> dict:
        """Use aiohttp to run query through SerpAPI and return the results async."""

        def construct_url_and_params() -> tuple[str, dict[str, str]]:
            params = self.get_params(query)
            params["source"] = "python"
            if self.serpapi_api_key:
                params["serp_api_key"] = self.serpapi_api_key
            params["output"] = "json"

            # Use the base path for the URL (add a "/" to ensure they get joined)
            url = url_join(self.base_url, "search")
            return url, params

        url, params = construct_url_and_params()
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                res = await self._session_get_with_retry(session, url, params)
        else:
            res = await self._session_get_with_retry(self.aiosession, url, params)

        return res

    @staticmethod
    def _process_response(res: dict) -> str:
        """Catch the ValueError and return a message if no good search result found."""
        try:
            return SerpAPIWrapper._process_response(res)
        except ValueError:
            return "No good search result found"