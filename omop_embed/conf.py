#!/usr/bin/env python3
import sys
import os
import subprocess as sp
import json

def find_binary (prog):
    path = sp.check_output('which %s' % prog, shell=True)
    if path is None:
        return None
    path = path.decode('ascii').strip()
    if len(path) == 0 or not os.path.exists(path):
        return None
    return path


HOME = os.path.abspath(os.path.dirname(__file__))
EHASH_PATH = find_binary('ehash')
PCAT_PATH = find_binary('pcat')
PSQL_PATH = find_binary('psql')

PARTITIONS = 1000
DATA_DIR = 'data'

SPLIT_TEST_RATIO = 0.2
SPLIT_SEED = 2019

DB = {
    'default': {
        "PGUSER": "",
        "PGPASSWORD": "",
        "PGHOST": "",
        "PGPORT": "",
        "PGDATABASE": "",
        "SCHEMA": "",
        "COHORT_SCHEMA": "",
        "BATCH": 1000
    }
}


if os.path.exists('conf.py'):
    with open('conf.py', 'r') as f:
        exec(f.read())

if not os.path.exists(EHASH_PATH):
    print("Please download latest ehash binary from https://github.com/aaalgo/ehash/releases/ and put in %s/bin/" % HOME)
    sys.exit(1)
    pass


#assert len(dict(CLINICAL_TABLES)) == len(CLINICAL_TABLES), "Two letter clinical table symbol must be unique"


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
