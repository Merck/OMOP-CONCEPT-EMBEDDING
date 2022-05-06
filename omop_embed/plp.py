#!/usr/bin/env python3
import os
from glob import glob
import random
import pandas as pd
from sqlalchemy import create_engine
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from omop_embed import clinical, etl, conf

def patch_pop (pop):
    if 'index' in pop.columns:
        return
    n = pop.shape[0]
    test = round(n * conf.SPLIT_TEST_RATIO)
    train = n - test
    index = [-1] * test + [1] * train
    random.seed(conf.SPLIT_SEED)
    random.shuffle(index)
    pop['index'] = index


def readRDS (path):
    rds = ro.r.readRDS(path)
    if rds is None:
        return None
    with localconverter(ro.default_converter + pandas2ri.converter):
        rds = pandas2ri.rpy2py(rds)
    return rds

def saveRDS (df, path):
    with localconverter(ro.default_converter + pandas2ri.converter):
        df = ro.conversion.py2rpy(df)
    ro.r.saveRDS(df, path)

def read_analysis_metrics (path):   # input dir is .../Analysis_XXX
    rds_path = os.path.join(path, 'plpResult/performanceEvaluation.rds')
    if not os.path.exists(rds_path):
        return None
    a = readRDS(rds_path)
    try:
        a = a[0]
    except:
        print(path)
        return None
    metrics = {}
    for i in range(a.nrow):
        v = {}
        for j in range(a.ncol):
            e = a.rx(i+1, j+1)
            v[a.colnames[j]] = a.rx(i+1, j+1)[0]
            pass
        if v['Eval'] == 'test':
            metrics[v['Metric']] = float(v['Value'])
    return metrics

def generate_blacklist (black_path):
    files = glob(f'plp/*/StudyPop*.rds')
    black = set()
    for i, path in enumerate(files):
        #print("Loading ", path, file=sys.stderr)
        rds = readRDS(path)
        x = rds[['subjectId', 'globalIndexes']]
        black1 = set(x[x['globalIndexes'] < 0]['subjectId'].tolist())
        #print("Loaded %d items from %s" % (len(black1), path))
        black = black | black1
    print("Found %d black list items." % len(black))
    if len(black) > 0:
        with open(black_path, 'w') as f:
            for x in black:
                f.write("0\t%d\n" % x)
        return True
    return False

def read_pop_file (task, target, outcome, pop, cov = 1):
    settings = pd.read_csv(f'plp/%s/settings.csv' % task)
    ones = settings[(settings['cohortId'] == target) & (settings['outcomeId'] == outcome) & (settings['populationSettingId'] == pop) & (settings['covariateSettingId'] == cov)]
    a = ones.studyPopFile.unique()
    if len(a) > 1:
        print(a)
    assert len(a) == 1
    rds = readRDS(os.path.join(HOME, 'tasks', a[0]))
    return rds

MAX_LONG_INT = 2**53

def sanity_check (task, cohorts):
    engine = create_engine(study.DB_URI)
    schema = conf['COHORT_SCHEMA']
    for cohort in cohorts:
        print('sanity checking cohort', cohort)
        c1 = pd.read_sql(f"select count(subject_id) from {schema}.cohort_{task} where cohort_definition_id={cohort}", engine)
        c2 = pd.read_sql(f"select count(distinct subject_id) from {schema}.cohort_{task} where cohort_definition_id={cohort}", engine)
        assert c1.iloc[0,0] == c2.iloc[0,0]

def create_extract_table (db, task, update):
    db = etl.DB(db)
    engine = create_engine(db.uri())
    schema = db.conf['COHORT_SCHEMA']
    info = pd.read_sql_query(f"select * from information_schema.tables where table_schema = '{schema}' and table_name = 'cohort_{task}_extract';", engine)
    n_tables = info.shape[0]
    if n_tables > 1:
        assert False
    if n_tables == 1 and not update:
        print("Not updating extract table.")
        return

    # the list of patients to be extracted
    plp_dir = os.path.join('plp', task)
    settings = pd.read_csv(os.path.join(plp_dir, 'settings.csv'))
    patients = set()
    for _, setting in settings.iterrows():
        analysisId = setting['analysisId']
        rds_path = os.path.join(plp_dir, 'Analysis_%d' % analysisId, 'plpResult', 'prediction.rds')
        df = study.readRDS(rds_path)
        patients = patients | set(df['subjectId'])
        pass
    print("loaded %d patients from plp of %s" % (len(patients), task))
    patients = pd.DataFrame({'subject_id': list(patients)})

    patients.to_sql(f"cohort_{task}_extract", engine, schema=schema, index=False, method='multi', chunksize=10000, if_exists='replace')


def create_partition (workdir, source, task, end_day, cohorts):

    #cov_dfs = []
    #cov_ref = None

    loader = etl.Loader(source=source)

    cohort_schema = conf['COHORT_SCHEMA']

    for cohort in cohorts:
        cohort_dir = os.path.join(workdir, 'cohort_%d' % cohort)
        partition_path = os.path.join(cohort_dir, 'partition')
        os.makedirs(cohort_dir, exist_ok=True)
        for table in clinical.DOMAIN.keys():
            loader.load_cohort(table, end_day, cohort_schema, f"cohort_{task}", f"cohort_{task}_extract", cohort, os.path.join(cohort_dir, table))
        etl.merge_partition(cohort_dir, partition_path)

#        conf = {}
#        conf.update(xtor.conf)
#        del conf['black_list']
#        conf.update({
#            'paths': [partition_path],
#             'shuffle': False,
#             'loop': False,
#             'inference': True,
#             'min_tokens': 0,
#             'cohort_dir': cohort_dir
#            })
#        print(conf)
#
#        patients, fts = xtor.extract_stream(conf)
#        with open(os.path.join(cohort_dir, 'features.pkl'), 'wb') as f:
#            pickle.dump((patients, fts), f)
#
#        if not R:
#            continue
#        records = fts.shape[0]
#        assert records == patients.shape[0]
#        dims = fts.shape[1]
#        n = records * dims
#        patients = patients.repeat(dims).reshape((-1, 1))
#        assert patients.shape[0] == n
#        #patients = patients.astype(np.int64)
#        assert (patients <= MAX_LONG_INT).all()
#        patients = patients.astype(np.float64)
#        patients = pd.DataFrame(patients, columns=['subjectId'])
#        vids = np.array(list(range(1, dims+1)), dtype=np.int32)
#        vids = np.tile(vids, records).reshape((-1,1))
#        assert vids.shape[0] ==  n
#        vids = pd.DataFrame(vids, columns=['covariateId'])
#        values = fts.reshape((-1,1)).astype(np.float64)
#        assert values.shape[0] ==  n
#        values = pd.DataFrame(values, columns=['covariateValue'])
#
#        cov = pd.concat([patients, vids, values], axis=1)
#        cov['cohortDefinitionId'] = cohort
#        print("df_created for chort %d: " % cohort, cov.shape)
#        cov_dfs.append(cov)
#
#        if cov_ref is None:
#            cov_ref = pd.DataFrame([[i+1, 'em%d' % (i+1)] for i in range(dims)],
#                                columns = ['covariateId', 'covariateName'])
#            cov_ref['covariateId'] = cov_ref['covariateId'].astype(np.float64)
#
#


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
