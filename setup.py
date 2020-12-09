# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
    name="world_models",
    version="1.0.0",
    author="Google LLC",
    author_email="no-reply@google.com",
    description="World Models Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google-research/tree/master/world_models",
    packages=["world_models"],
    package_dir={"world_models": ""},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "absl-py",
        "gin-config",
        "numpy",
        "tensorflow==1.15",
        "tensorflow-probability==0.7",
        "gym",
        "dm_control",
        "mujoco-py==2.0.2.8",
    ],
    python_requires="<3.8",
)