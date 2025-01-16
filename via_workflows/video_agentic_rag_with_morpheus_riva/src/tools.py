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
import warnings
from textwrap import dedent
import requests

from packaging.version import InvalidVersion
from packaging.version import parse as parse_version

logger = logging.getLogger(__name__)


def range_version_comparator(
    software_version: str,
    vulnerability_lower_range: str,
    vulnerability_upper_range: str,
):
    """
    Compare a software's version to a range of vulnerable versions to determine vulnerability.

    Parameters
    ----------
    software_version : str
        The version of the software currently in use.
    vulnerability_lower_range : str
        The lower bound of the vulnerable version range.
    vulnerability_upper_range : str
        The upper bound of the vulnerable version range.

    Returns
    -------
    bool
        Returns True if the software version is within the range of vulnerable versions,
        indicating potential vulnerability.

    Raises
    ------
    InvalidVersion
        If the version strings are not in a valid format, a warning is issued and alphabetic
        comparison is used instead.

    Notes
    -----
    This function assumes that the software is vulnerable if its version falls inclusively
    between the lower and upper bounds of the vulnerability range. It uses the `parse_version`
    function to interpret the versions and compares them accordingly. If `parse_version` fails,
    Debian version parsing is attempted. Finally, if both of these fail, it falls
    back to a simple string comparison.
    """
    try:
        sv = parse_version(str(software_version))
        lvv = parse_version(str(vulnerability_lower_range))
        uvv = parse_version(str(vulnerability_upper_range))
        return sv <= uvv and sv >= lvv
    except InvalidVersion:
        # Failed PEP440 versioning; moving on to Debian
        pass

    try:
        return (
            Dpkg.compare_versions(str(software_version), str(vulnerability_lower_range))
            != -1
            and Dpkg.compare_versions(
                str(software_version), str(vulnerability_upper_range)
            )
            != 1
        )
    except DpkgVersionError:
        warnings.warn(
            "Unable to parse provided versions. Using alpha sorting.", stacklevel=2
        )
    # Fallback to alphabetic comparison if version parsing fails
    return str(software_version) <= str(vulnerability_upper_range) and str(
        software_version
    ) >= str(vulnerability_lower_range)


def single_version_comparator(software_version: str, vulnerability_version: str):
    """
    Compare a software's version to a known vulnerable version.

    Parameters
    ----------
    software_version : str
        The version of the software currently in use.
    vulnerability_version : str
        The version of the software that is known to be vulnerable.

    Returns
    -------
    bool
        Returns True if the software version is less than or equal to the vulnerability version,
        indicating potential vulnerability.

    Raises
    ------
    InvalidVersion
        If the version strings are not in a valid format, a warning is issued and alphabetic
        comparison is used instead.
    """
    try:
        sv = parse_version(str(software_version))
        vv = parse_version(str(vulnerability_version))
        return sv <= vv
    except InvalidVersion:
        # Failed PEP440 versioning; moving on to Debian
        pass
    try:
        return (
            Dpkg.compare_versions(str(software_version), str(vulnerability_version))
            != 1
        )
    except DpkgVersionError:
        warnings.warn(
            "Unable to parse provided versions. Using alpha sorting.", stacklevel=2
        )
    return str(software_version) <= str(vulnerability_version)


def version_comparison(software_version: str):
    """
    Compare a software's version to multiple known vulnerable versions.

    Parameters
    ----------
    software_version : str
        A string containing the software version to compare, and the vulnerable versions,
        separated by commas. A single vulnerable version, a vulnerable range (two versions),
        or multiple specific vulnerable versions can be provided.

    Returns
    -------
    bool or str
        Returns True if the software version matches any of the vulnerable versions,
        or is within the vulnerable range. Returns a string message if the input doesn't
        contain enough information for a comparison.

    Notes
    -----
    This function can compare against a single vulnerable version, a range of versions,
    or a list of specific versions. It uses the `single_version_comparator` for single comparisons,
    and `range_version_comparator` for range comparisons.
    """
    v = software_version.split(",")
    if len(v) == 2:
        return single_version_comparator(v[0], v[1])
    elif len(v) == 3:
        return range_version_comparator(v[0], v[1], v[2])
    elif len(v) > 3:
        return any([v[0] == v_ for v_ in v[1:]])
    else:
        return "Couldn't able compare the software version, not enough input"


class VideoEventQuery:
    tool_description = dedent(
        """
        Useful for when you want a summary of all the times a certain event occurred in a video or  
        a description of a certain item in a video. 
        The input should be the description of the event you want to search for or the item you want to describe, only. 
        For example, searches can be "Describe the contents of the refridgerator in the video", or 
        "One person shaking another's hand". The string you provide will be passed to a vision language model for 
        summarization of a video.
    """
    ).replace("\n", "")

    def __init__(self, file_id: str, start: int, end: int, port: str):

        self.file_id = file_id
        self.base_url = f"http://localhost:{port}"
        self.upload_endpoint = f"{self.base_url}/files"
        self.summarize_endpoint = f"{self.base_url}/summarize"
        self.start = start
        self.end = end

    def video_search(self, search_prompt):
        data = {
            "id": self.file_id,
            "model": "gpt-4o",  # Specify the model to use
            "response_format": {"type": "text"},
            "chunk_duration": 600,
            "chunk_overlap_duration": 3,
            "max_tokens": 512,
            "media_info": {
                "type": "offset",
                "start_offset": self.start,
                "end_offset": self.end,
            },
            "prompt": f"""
            You are an expert world understanding model. You are part of a complex machine learning pipeline that takes video of the worl and is helps visually impaired users understand and manipulate the world around them. The videos are in first person view. Given the following description or query, your task is to search your video for that query and return a detailed response of all of what is requested. Note relative locations in the video of objects, color, and anything else that may be helpful for someone to understand their surroundings. Search the video for the following
            
            {search_prompt}
            """,
        }

        response = requests.post(self.summarize_endpoint, json=data)
        if response.status_code == 200:
            summary_response = response.json()
            return summary_response["choices"][0]["message"]["content"]
        else:
            return f"Failed to get summary: {response.text}"
