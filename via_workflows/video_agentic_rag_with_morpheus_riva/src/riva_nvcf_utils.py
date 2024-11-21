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

import subprocess
import os


def transcribe_file(input_file):
    # Define the command as a list of arguments
    api_key = os.getenv("NVIDIA_API_KEY")
    command = [
        "python",
        "/workspace/python-clients/scripts/asr/transcribe_file.py",
        "--server",
        "grpc.nvcf.nvidia.com:443",
        "--use-ssl",
        "--metadata",
        "function-id",
        "1598d209-5e27-4d3c-8079-4751568b1081",
        "--metadata",
        "authorization",
        f"Bearer {api_key}",
        "--language-code",
        "en-US",
        "--input-file",
        input_file,
    ]

    try:
        # Run the command
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        return result.stdout

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")


def generate_audio(text, output_file="output.wav"):
    # Define the command as a list of arguments
    api_key = os.getenv("NVIDIA_API_KEY")
    command = [
        "python",
        "/workspace/python-clients/scripts/tts/talk.py",
        "--server",
        "grpc.nvcf.nvidia.com:443",
        "--use-ssl",
        "--metadata",
        "function-id",
        "0149dedb-2be8-4195-b9a0-e57e0e14f972",
        "--metadata",
        "authorization",
        f"Bearer {api_key}",
        "--text",
        text,
        "--voice",
        "English-US.Female-1",
        "--output",
        output_file,
    ]

    try:
        # Run the command
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")
