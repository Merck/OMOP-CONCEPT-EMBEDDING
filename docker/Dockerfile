from nvidia/cuda:11.4.0-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip openjdk-16-jdk-headless build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev file libfontconfig1-dev libcairo2-dev r-base-dev postgresql-client
RUN ln -s java-16-openjdk-amd64 /usr/lib/jvm/default-java
ADD drivers /drivers
ENV DATABASECONNECTOR_JAR_FOLDER=/drivers
RUN pip3 install scikit-learn==0.24.2 pandas==1.2.2
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
ADD install.R /install.R
RUN Rscript /install.R
RUN pip3 install rpy2==3.4.5 tqdm==4.62.0 sqlalchemy==1.4.22 psycopg2_binary==2.9.1


# Copyright © 2022 Merck & Co., Inc., Rahway, NJ, USA and its affiliates. All rights reserved.
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