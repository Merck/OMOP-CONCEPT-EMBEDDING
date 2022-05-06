#!/usr/bin/env Rscript
#devtools::load_all()
source("./config.R")

cohortTable <- 'cohort_ra'
outputFolder <- "./create_cohorts/ra"

EHDENRAPrediction::execute(connectionDetails = connectionDetails,
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
        analysisIdDocument = 1,
	cohortsToCreateCsv = 'cohorts_ra.csv',
	predictionAnalysisListJson = 'predictionAnalysisList_ra.json'
	)

