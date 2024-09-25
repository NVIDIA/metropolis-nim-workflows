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

import asyncio
import websockets
import json
import threading


class WebSocketServer:
    def __init__(self, host="localhost", port=5433):
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = asyncio.new_event_loop()
        self.start_server = websockets.serve(
            self.handler, self.host, self.port, loop=self.loop
        )

    async def handler(self, websocket, path):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                # Handle incoming messages here if needed
                pass
        finally:
            self.clients.remove(websocket)

    async def send_message(self, message):
        if self.clients:
            await asyncio.gather(
                *[client.send(json.dumps(message)) for client in self.clients]
            )

    def __call__(self, message):
        asyncio.run_coroutine_threadsafe(self.send_message(message), self.loop)

    def run(self):
        def start_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.start_server)
            loop.run_forever()

        threading.Thread(target=start_loop, args=(self.loop,)).start()
