#!/usr/bin/env Rscript
#devtools::load_all()
source("./config.R")

cohortTable <- 'cohort_mdd'
outputFolder <- './create_cohorts/mdd'

MddAfterBbValidation::execute(connectionDetails = connectionDetails,
                              databaseName = cdmDatabaseName,
                              cdmDatabaseSchema = cdmDatabaseSchema,
                              cohortDatabaseSchema = cohortDatabaseSchema,
                              oracleTempSchema = oracleTempSchema,
                              cohortTable = cohortTable,
                              outputFolder = outputFolder,
                              createCohorts = T,
                              runValidation = F,
                              packageResults = F,
                              minCellCount = 5,
                              sampleSize = NULL)
