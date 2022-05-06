#!/usr/bin/env Rscript
source("./config.R")

cohortTable <- 'cohort_stroke'
outputLocation <- './output_create_cohorts_stroke'

ExistingStrokeRiskExternalValidation::main(
  connectionDetails=connectionDetails,
  oracleTempSchema = NULL,
  databaseName=cdmDatabaseName,
  cdmDatabaseSchema=cdmDatabaseSchema,
  cohortDatabaseSchema=cohortDatabaseSchema,
  outputLocation=outputLocation,
  cohortTable=cohortTable,
  createCohorts = T,
  runAtria = F,
  runFramingham = F,
  runChads2 = F,
  runChads2Vas = F,
  runQstroke = F,
  summariseResults = F,
  packageResults = F,
  N=10)

