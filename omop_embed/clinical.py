# https://ohdsi.github.io/CommonDataModel/cdm531.html#Clinical_Data_Tables

DOMAIN = {
        'condition_occurrence': 1,
        'procedure_occurrence': 2,
        'drug_exposure': 3,
}


# We do not use all fields in ETL, this allows us to
#   - achieve some simplicity in model.
#   - achieve compatibility between different CDM models

TABLE_COLUMNS = {
  "person": [
    "person_id",
    "gender_concept_id",
    "year_of_birth",
    "month_of_birth",
    "day_of_birth",
    "birth_datetime",
    #"race_concept_id",
    #"ethnicity_concept_id",
    #"location_id",
    #"provider_id",
    #"care_site_id",
    #"person_source_value",
    #"gender_source_value",
    #"gender_source_concept_id",
    #"race_source_value",
    #"race_source_concept_id",
    #"ethnicity_source_value",
    #"ethnicity_source_concept_id"
  ],
  #"observation_period": [
  #  "observation_period_id",
  #  "person_id",
  #  "observation_period_start_date",
  #  "observation_period_end_date",
  #  "period_type_concept_id"
  #],
  "visit_occurrence": [
    "person_id",
    "visit_concept_id",
    "visit_start_date",
    #"visit_start_datetime",
    "visit_end_date",
    #"visit_end_datetime",
    "visit_type_concept_id",
    #"visit_occurrence_id",
    #"provider_id",
    #"care_site_id",
    #"visit_source_value",
    #"visit_source_concept_id",
    #"admitting_source_concept_id",
    #"admitting_source_value",
    #"discharge_to_concept_id",
    #"discharge_to_source_value",
    #"preceding_visit_occurrence_id"
  ],
  "condition_occurrence": [
    #"condition_occurrence_id",
    "person_id",
    "condition_concept_id",
    "condition_start_date",
    #"condition_start_datetime",
    "condition_end_date",
    #"condition_end_datetime",
    "condition_type_concept_id",
    #"stop_reason",
    #"provider_id",
    #"visit_occurrence_id",
    #"condition_source_value",
    #"condition_source_concept_id",
    #"condition_status_source_value",
    #"condition_status_concept_id"
  ],
  "drug_exposure": [
    #"drug_exposure_id",
    "person_id",
    "drug_concept_id",
    "drug_exposure_start_date",
    #"drug_exposure_start_datetime",
    "drug_exposure_end_date",
    #"drug_exposure_end_datetime",
    #"verbatim_end_date",
    "drug_type_concept_id",
    #"stop_reason",
    #"refills",
    #"quantity",
    #"days_supply",
    #"sig",
    #"route_concept_id",
    #"lot_number",
    #"provider_id",
    #"visit_occurrence_id",
    #"drug_source_value",
    #"drug_source_concept_id",
    #"route_source_value",
    #"dose_unit_source_value"
  ],
  "procedure_occurrence": [
    #"procedure_occurrence_id",
    "person_id",
    "procedure_concept_id",
    "procedure_date",
    #"procedure_datetime",
    "procedure_type_concept_id",
    #"modifier_concept_id",
    #"quantity",
    #"provider_id",
    #"visit_occurrence_id",
    #"procedure_source_value",
    #"procedure_source_concept_id",
    #"qualifier_source_value",
    #"modifier_source_value"
  ],
  "device_exposure": [
    #"device_exposure_id",
    "person_id",
    "device_concept_id",
    "device_exposure_start_date",
    #"device_exposure_start_datetime",
    "device_exposure_end_date",
    #"device_exposure_end_datetime",
    "device_type_concept_id",
    #"unique_device_id",
    #"quantity",
    #"provider_id",
    #"visit_occurrence_id",
    #"device_source_value",
    #"device_source_concept_id"
  ],
#  "measurement": [
#    "measurement_id",
#    "person_id",
#    "measurement_concept_id",
#    "measurement_date",
#    "measurement_datetime",
#    "measurement_type_concept_id",
#    "operator_concept_id",
#    "value_as_number",
#    "value_as_concept_id",
#    "unit_concept_id",
#    "range_low",
#    "range_high",
#    "provider_id",
#    "visit_occurrence_id",
#    "measurement_source_value",
#    "measurement_source_concept_id",
#    "unit_source_value",
#    "value_source_value"
#  ],
#  "observation": [
#    "observation_id",
#    "person_id",
#    "observation_concept_id",
#    "observation_date",
#    "observation_datetime",
#    "observation_type_concept_id",
#    "value_as_number",
#    "value_as_string",
#    "value_as_concept_id",
#    "qualifier_concept_id",
#    "unit_concept_id",
#    "provider_id",
#    "visit_occurrence_id",
#    "observation_source_value",
#    "observation_source_concept_id",
#    "unit_source_value",
#    "qualifier_source_value"
#  ],
#  "death": [
#    "person_id",
#    "death_date",
#    "death_datetime",
#    "death_type_concept_id",
#    "cause_concept_id",
#    "cause_source_value",
#    "cause_source_concept_id"
#  ],
#  "note": [
#    "note_id",
#    "person_id",
#    "note_date",
#    "note_datetime",
#    "note_type_concept_id",
#    "note_class_concept_id",
#    "note_title",
#    "note_text",
#    "encoding_concept_id",
#    "language_concept_id",
#    "provider_id",
#    "visit_occurrence_id",
#    "note_source_value"
#  ],
#  "note_nlp": [
#    "note_nlp_id",
#    "note_id",
#    "section_concept_id",
#    "snippet",
#    "offset",
#    "lexical_variant",
#    "note_nlp_concept_id",
#    "note_nlp_source_concept_id",
#    "nlp_system",
#    "nlp_date",
#    "nlp_datetime",
#    "term_exists",
#    "term_temporal",
#    "term_modifiers"
#  ],
#  "specimen": [
#    "specimen_id",
#    "person_id",
#    "specimen_concept_id",
#    "specimen_type_concept_id",
#    "specimen_date",
#    "specimen_datetime",
#    "quantity",
#    "unit_concept_id",
#    "anatomic_site_concept_id",
#    "disease_status_concept_id",
#    "specimen_source_id",
#    "specimen_source_value",
#    "unit_source_value",
#    "anatomic_site_source_value",
#    "disease_status_source_value"
#  ]
}

COHORT_MERGE_CONFIG = {
    #'visit_occurrence': ['visit_concept_id', 'visit_start_date', 'visit_end_date'],
    'condition_occurrence': ['condition_concept_id', 'condition_start_date', 'condition_end_date'],
    'drug_exposure': ['drug_concept_id', 'drug_exposure_start_date', 'drug_exposure_end_date'],
    'procedure_occurrence': ['procedure_concept_id', 'procedure_date', 'procedure_date']
}


'''
Copyright © 2022 Merck & Co., Inc., Rahway, NJ, USA and its affiliates. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
