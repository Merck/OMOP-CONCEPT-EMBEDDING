#!/usr/bin/env Rscript

source("./config.R")

tasks = commandArgs(trailingOnly=TRUE)

for (task in tasks) {

    outputFolder <- paste("./plp/", task, sep="")
    json <- paste("predictionAnalysisList_", task, ".json", sep="")
    cohortTable = paste("cohort_", task, sep="")

    EHDENRAPrediction::execute(connectionDetails = connectionDetails,
            cdmDatabaseSchema = cdmDatabaseSchema,
            cdmDatabaseName = cdmDatabaseName,
            cohortDatabaseSchema = cohortDatabaseSchema,
            cohortTable = cohortTable,
            outputFolder = outputFolder,
            createProtocol = F,
            createCohorts = F,
            runAnalyses = T,
            createResultsDoc = F,
            packageResults = F,
            createValidationPackage = F,
            minCellCount= 5,
            createShiny = F,
            createJournalDocument = F,
            analysisIdDocument = 1,
       	predictionAnalysisListJson = json
    	)
} 



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
