#!/usr/bin/env python3
import sys
import os
import json
import subprocess as sp
from glob import glob
from collections import defaultdict
import omop_core
from omop_embed import clinical, conf

class DB:
    def __init__ (self, name = 'default'):
        self.conf = conf.DB[name]

    def schema (self):
        return self.conf['SCHEMA']

    def uri (self):
        return 'postgresql://%s:%s@%s:%s/%s' % (
                self.conf['PGUSER'],
                self.conf['PGPASSWORD'],
                self.conf['PGHOST'],
                self.conf['PGPORT'],
                self.conf['PGDATABASE'])

    def env (self):
        env = os.environ.copy()
        for key in ['PGUSER', 'PGPASSWORD', 'PGHOST', 'PGPORT', 'PGDATABASE']:
            env[key] = self.conf[key]
        batch = self.conf.get("BATCH", 1000)
        env["PGBATCH"] = str(batch)
        return env


class Loader:

    def __init__ (self, db='example'):
        self.db = DB(db)
        self.env = self.db.env()


    def load_sql (self, SQL, partition_key=1, table=None, total_lines=None, parallel=None):
        assert not '"' in SQL
        # ehash loader
        L = '%s "{}"' % conf.PCAT_PATH
        #L = '%s -c "\COPY ({}) TO STDOUT WITH CSV"' % omop.PSQL_PATH

        # load target configuration
        args = [conf.EHASH_PATH]
        args.extend(['-p', str(conf.PARTITIONS)])
        args.extend(['-d', ','])
        args.extend(['-o', os.path.join(conf.DATA_DIR, 'ehash')])
        args.extend(['-k', str(partition_key)])
        args.extend(['-l', L])
        args.extend(['-i', '1000000'])
        args.extend(['-c', 'gzip - > {}'])
        if not table is None:
            args.extend(['-b', table])
        if not total_lines is None:
            args.extend(['-n', str(total_lines)])
        if parallel is None:
            args.extend(['-t', '1'])
            args.extend([SQL])
        else:
            args.extend(['-t', str(parallel)])
            args.extend([SQL + f' where person_id % {parallel} = {mod}' for mod in range(parallel)])
        sp.check_call(args, env=self.env)

    def load (self, table, columns, limit=None, parallel=None, probe_only=False):
        key = columns.index('person_id')
        schema = self.db.schema()
        CNT_SQL = f"select count(1) from {schema}.{table}"
        lines = sp.check_output([conf.PCAT_PATH, CNT_SQL], env=self.env)
        lines = int(lines.strip())
        if probe_only:
            return lines
        SQL = f"select " + ','.join(columns) + f" from {schema}.{table}"
        if not limit is None:
            assert parallel is None
            SQL += ' limit %d' % limit
        self.load_sql(SQL, key, table, lines, parallel)
        pass

    def load_cohort (self, table, end_day, cohort_schema, cohort_table, extract_table, cohort, output_path):
        schema = self.db.schema()
        concept_column, domain_start_date, domain_end_date = clinical.COHORT_MERGE_CONFIG[table]
        cols = ','.join(['src.%s' % x for x in clinical.TABLE_COLUMNS[table]])
        # TODO fix end date
        SQL = f"""select distinct {cols} from {schema}.{table} as src, {cohort_schema}.{cohort_table} as cohort, {cohort_schema}.{extract_table} as extract
        where src.person_id = cohort.subject_id
        AND src.person_id = extract.subject_id
        AND cohort.cohort_definition_id = {cohort}
        AND src.{domain_start_date} <= cohort.cohort_start_date + interval '1 day' * {end_day}
        """
        args = [conf.PCAT_PATH, SQL]
        with open(output_path, 'w') as f:
            sp.check_call(args, env=self.env, stdout=f)
        pass

    def load_all (self, limit=None, parallel=None):
        for table, columns in clinical.TABLE_COLUMNS.items():
            print("Loading %s ..." % table)
            self.load(table, columns, limit, parallel)
            pass

    def estimate_all (self, build_model):
        if build_model:
            sizes = defaultdict(lambda: 0)
            model = {}
            for path in glob('%s/ehash/part-*/*' % conf.DATA_DIR):
                sz = os.path.getsize(path)
                table = os.path.basename(path)
                sizes[table] += sz
            for k, v in sizes.items():
                model[k] = [v, 0]
        else:
            with open('estimate_model.json', 'r') as f:
                model = json.load(f)

        total = 0
        for table, columns in clinical.TABLE_COLUMNS.items():
            lines = self.load(table, columns, probe_only=True)
            #sz = sizes[table]
            if build_model:
                model[table][1] = lines
                sz = model[table][0]
            else:
                m_sz, m_l = model[table]
                sz = m_sz * lines / m_l
            total += sz
            pass
            print("table %s %g GB" % (table, sz/1024/1024/1024))

        print("total %g GB" % (total/1024/1024/1024))
        if build_model:
            with open('estimate_model.json', 'w') as f:
                json.dump(model, f)

def merge_partition (dir, output, src=0):
    loader = omop_core.PartitionLoader(src)
    #load_partition_table(loader, dir, 'visit_occurrence', 'visit_concept_id', 'visit_start_date', 'visit_end_date')
    for table, (concept_column, start_date_column, end_date_column) in clinical.COHORT_MERGE_CONFIG.items():
        print('merge', table, concept_column, start_date_column, end_date_column)
        #load_partition_table(loader, dir, key, *columns)
        domain = clinical.DOMAIN[table]  # TODO: we can do something about domain
        columns = clinical.TABLE_COLUMNS[table]
        loader.load_file(os.path.join(dir, table),
                domain,
                len(columns),
                columns.index('person_id'),
                columns.index(concept_column),
                columns.index(start_date_column),
                columns.index(end_date_column))

    loader.save(output)
    pass

#def unique_time ():
#    return datetime.now().strftime('%Y%m%d%H%M%S')


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
