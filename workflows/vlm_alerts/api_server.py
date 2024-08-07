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

# Setup endpoint that can be used to update the prompt
from threading import Thread
from flask import Flask, request
from uuid import uuid4
from dataclasses import dataclass
from time import time, sleep


@dataclass
class APIMessage:
    type: str
    data: str
    id: int


class FlaskServer:

    def __init__(self, cmd_q, resp_d, port=5432):
        self.cmd_q = cmd_q
        self.resp_d = resp_d

        self.app = Flask(__name__)

        self.app.add_url_rule("/query", "query", self.query)
        self.port = port

    def get_command_response(self, uuid_str, timeout=10):
        start_time = time()

        # wait for reseponse or timeout
        while True:
            if time() - start_time >= timeout:
                return None

            if uuid_str in self.resp_d:
                return self.resp_d.pop(uuid_str)
            else:
                sleep(0.1)

    def query(self):
        print(request.args)
        type = "alert" if request.args.get("alert", False) == "True" else "query"
        print(type)
        queue_message = APIMessage(
            type=type,
            data=request.args.get("query", "Describe the scene."),
            id=str(uuid4()),
        )
        self.cmd_q.put(queue_message)
        response = self.get_command_response(queue_message.id)
        if response:
            return response
        else:
            return "Server timed out processing the request"

    def _start_flask(self):
        self.app.run(use_reloader=False, host="0.0.0.0", port=self.port)

    def start_flask(self):
        self.flask_thread = Thread(target=self._start_flask, daemon=True)
        self.flask_thread.start()
