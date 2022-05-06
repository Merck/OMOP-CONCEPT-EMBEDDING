#!/usr/bin/env Rscript
#devtools::load_all()
source("./config.R")

cohortTable <- 'cohort_bc'
outputFolder <- "./create_cohorts/bc"

finalWoo::execute(connectionDetails = connectionDetails,
        cdmDatabaseSchema = cdmDatabaseSchema,
        cdmDatabaseName = cdmDatabaseName,
        cohortDatabaseSchema = cohortDatabaseSchema,
        cohortTable = cohortTable,
        outputFolder = outputFolder,
        createProtocol = F,
        createCohorts = T,
        runAnalyses = F,
        createResultsDoc = F,
        packageResults = F,
        createValidationPackage = F,
        minCellCount= 5,
        createShiny = F,
        createJournalDocument = F,
        analysisIdDocument = 1
	)

