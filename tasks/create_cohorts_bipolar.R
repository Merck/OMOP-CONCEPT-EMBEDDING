#!/usr/bin/env Rscript
#devtools::load_all()
source("./config.R")

cohortTable <- 'cohort_bipolar'
outputFolder <- './create_cohorts/bipolar'
restrictToAdults <- FALSE

BipolarMisclassificationValidation::execute(connectionDetails = connectionDetails,
                                            cdmDatabaseSchema = cdmDatabaseSchema,
                                            cohortDatabaseSchema = cohortDatabaseSchema,
                                            cohortTable = cohortTable,
                                            outputFolder = outputFolder,
                                            databaseName = cdmDatabaseName,
                                            oracleTempSchema = oracleTempSchema,
                                            viewModel = F,
                                            createCohorts = T,
                                            runValidation = F,
                                            packageResults = F,
                                            minCellCount = minCellCount,
                                            sampleSize = NULL,
                                            restrictToAdults = restrictToAdults)
