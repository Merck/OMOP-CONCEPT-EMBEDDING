#Sys.setenv(DATABASECONNECTOR_JAR_FOLDER="../drivers")
options(fftempdir = "ff_temp")
dbms <- "postgresql"
user <- ''
pw <- ''
server <- 'localhost/omop'  # host/databash
port <- '5432'
schema <- 'synpuf'
cdmDatabaseSchema <- schema
cdmDatabaseName <- 'omop'
oracleTempSchema <- 'public'
cohortDatabaseSchema <- 'apps'

if (file.exists('local_config.R')) {
    source('local_config.R')
}

connectionDetails <- DatabaseConnector::createConnectionDetails(dbms = dbms,
                                                                server = server,
                                                                user = user,
                                                                password = pw,
                                                                port = port
)


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