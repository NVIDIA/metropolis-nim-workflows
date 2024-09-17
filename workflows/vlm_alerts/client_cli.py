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


class bcolors:
    # Basic colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    ENDC = "\033[0m"


def send_prompt(prompt, loop=False):
    # Send prompt
    params = {"query": prompt, "alert": loop}
    url = f"http://localhost:5432/query"
    response = requests.get(url, params=params)
    return response.text


if __name__ == "__main__":
    while True:
        prompt = input(f"{bcolors.GREEN}Prompt: {bcolors.ENDC}")
        loop = input(f"{bcolors.GREEN}Is this an Alert? (y/n): {bcolors.ENDC}")
        loop = True if "y" in loop.lower() else False
        print("")

        try:
            print(f"{bcolors.RED}{send_prompt(prompt, loop=loop)}{bcolors.ENDC}")
            print("")
        except Exception as e:
            print(e)
            print(
                f"{bcolors.FAIL}Could not send prompt. Ensure demo is running.{bcolors.ENDC}"
            )
            print("")
