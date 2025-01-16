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

from websockets.sync.server import serve
from threading import Thread
from queue import Queue
from time import sleep
import json


class WebSocketServer:
    def __init__(self, host="localhost", port=5433):
        self.host = host
        self.port = port
        self.ws_thread = None
        self.connection_qs = {}

    def _manage_connection(self, connection):
        self.connection_qs[connection.id] = Queue()
        try:
            while True:
                message = self.connection_qs[connection.id].get()
                connection.send(message)
        except Exception as e:
            print(f"Closed connection id: {connection.id}\n Exception {e}")
            del self.connection_qs[connection.id]

    def _start_server(self):
        with serve(self._manage_connection, self.host, self.port) as server:
            server.serve_forever()

    def run(self):
        self.ws_thread = Thread(target=self._start_server, daemon=True)
        self.ws_thread.start()

    def __call__(self, message):
        if isinstance(message, (dict,)):
            message = json.dumps(message)
        connections = self.connection_qs.keys()
        for connection in connections:
            self.connection_qs[connection].put(message)


if __name__ == "__main__":
    ws_server = WebSocketServer()
    ws_server.run()
    for x in range(10):
        print("sending message")
        ws_server("Hello there")
        sleep(2)
    print("done")
