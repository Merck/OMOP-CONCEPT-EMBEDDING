#!/usr/bin/env Rscript
install.packages(c('devtools', 'DatabaseConnector', 'reticulate'))
reticulate::install_miniconda()
reticulate::py_install('sklearn', pip=T)
devtools::install_github('OHDSI/OhdsiSharing', ref='master')
devtools::install_github('aaalgo/FeatureExtraction', ref='d656135')
devtools::install_github('aaalgo/PatientLevelPrediction', ref='master')
devtools::install_github('aaalgo/EhdenRaPrediction', ref='master')
devtools::install_github('ohdsi-studies/ExistingStrokeRiskExternalValidation', ref='8857f63')
devtools::install_github('ohdsi-studies/MddAfterBbValidation', ref='dc5e20c')
devtools::install_github('aaalgo/BreastCancerBetweenScreen', ref='f36705c')
devtools::install_github('ohdsi-studies/BipolarMisclassificationValidation', ref='441fae0')


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