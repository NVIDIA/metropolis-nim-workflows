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
import sys

morpheus_logger = logging.getLogger("morpheus")

if not getattr(morpheus_logger, "_configured_by_morpheus", False):

    # Set the morpheus logger to propagate upstream
    morpheus_logger.propagate = False

    # Add a default handler to the morpheus logger to print to screen
    morpheus_logger.addHandler(logging.StreamHandler(stream=sys.stdout))

    # Set a flag to indicate that the logger has been configured by Morpheus
    setattr(morpheus_logger, "_configured_by_morpheus", True)

logger = logging.getLogger(__name__)

# Set the parent logger for the entire package to use morpheus so we can take advantage of configure_logging
logger.parent = morpheus_logger
