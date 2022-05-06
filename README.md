OMOP CONCEPT EMBEDDING
======================

# 1. Prerequisites and Setup

## 1.1 Dependencies

The software depends on the following:

- Typical Python3 environment.
- C++ building environment.
- ~psql command~.

## 1.2 SageMaker


```
$ yum install -y postgresql96 postgresql96-devel
```

Add to `~/.bashrc`:

```
conda activate pytorch_p36
export PATH=$PATH:$HOME/anaconda3/envs/R/bin:$HOME/SageMaker/tools/bin
```

Logout and back in to make `.bashrc` effective.


## 1.3 Installation From Source

After git clone or extracting the software, cd into the directory and
run

```
$ pip3 install -r requirements.txt
$ ./setup.py build
$ ln -s build/*/*.so ./
```

Add the following to `~/.bashrc`, substituting DIR with current directory
```
export PYTHONPATH=DIR
export PATH=$PATH:DIR/tools:DIR/scripts
```

# 2. Working Directory

The software always assumes that the working directory is the current
directory.  The working directory follows the structure below.

```
[WORKING DIRECTORY]
├── conf.py				# local configuration file
├── settings.txt		# experimental settings
├── data
│   ├── dict			# vocabulary
│   ├── ehash			# intermediate data
│   │   ├── part-00000	# can be removed
│   │   ├── part-00001	# after calling omop_merge
│   │   ├── part-00002	# and partition files are produced
│   │   ......
│   ├── partitions		# preprocessed OMOP data
│   │   ├── part-00000	# this is what we use to train
│   │   ├── part-00001	# deep learning models.
│   │   ├── part-00002
│   │   ......
│   │ 
│   └── tasks
│       └── bc			# partition files of down stream tasks
│           ├── cohort_10631	# produced by omop_load_plp
│           └── cohort_10845
├── labels				# label data
│   └── bc				# results directory produced by PLP
│       ├── labels-183-257-1.csv
│       ├── labels-183-257-2.csv
│ 		......
│ 
├── plp					#
│   └── bc				# results directory produced by PLP
│       ├── plplog.txt
│       ├── settings.csv
│       ├── StudyPop_L1_T10631_O10082_P1.rds
│       ├── StudyPop_L1_T10845_O10082_P1.rds
│       ......
│       ├── Analysis_1
│       │   ├── plplog.txt
│       │   └── plpResult
│       ├── Analysis_10
│       │   ├── plplog.txt
│       │   └── plpResult

│       ......
└── results				# local experimental results
    ├── lgbm			# each sub directory is a TAG
    └── plp

```

Resules produced by `omop_eval_...` are written to the `results`
subdirectory and are organized by tags which can be specified
on command line.  The evaluation scripts needs to read feature
data for each task stored in `data/tasks/{task}` directory.
It also needs the original output of PLP for cohort and train-test
splitting information.

## 2.1 Dictionary File

A dictionary file (`data/dict`) lists all concepts that should be recognized
by the machine learning part of the software.  The file is generated
during the ETL process (Section 4 below).  Experiments in Section 3 can
be done without ETL and the user will have to prepare a dictionary file.

The format of the file is like below:
```
2414397	198
2414398	653
2108115	582
320128	312
2314284	222
```

The file should container two columns separated by tab, without a
header.  The first column is the concept ID and the second is the
frequency.  The file should be sorted by frequency in descending order.

Suppose one has a directory INPUT containing three files as below
```
INPUT
├── condition_occurrence
├── drug_exposure
└── procedure_occurrence
```
The following command can be used to generated the dictionary file:

```
cut -f 2 -d , INPUT/* | sort | uniq -c | awk 'BEGIN{OFS="\t";}{print $2,$1;}' | sort -k 2 -gr > dict
```

# 3. Basic Experiments

## 3.1 Running Experiments

The following have to be prepared and put into the working directory.

- A `settings.txt` file which describes all experiments.
- A partition file for each experiment containing all EMR data.  This
  may be reused across different experiments.
- A label file for each experiment.
- A dictionary file, see Section 2.1 above.


The settings.txt file is a tab-separated text file of three columns
without header.  Each row specifies the configuration of an experiment,
called a task.
The first columm is the task label.
second column the path to label file and third column the path to
partition file.

To run an experiment, run for example

```
omop_eval_sklearn --task TASK --tag TAG
```
or
```
omop_eval_dl --dict data/dict --task LABEL --tag TAG  --gpu  --learning_rate 0.001 --weight_decay 0.001 --whitelist_threshold 50 --seq_len 768 --net baseline --dim 1024 --drop_rate 0.5 --epochs 100
```

In the above two commands, `--task` specifies the row in `settings.txt`.
The label and EMR data are loaded from the settings of the  corresponding line.
The TAG names the method, or experiment setting.

The results will be samed in the `results` directory, first by `TAG` and
then by `TASK`.  Uses the following command to plot results

One might want to run through each task for each TAG.

```
omop_plot_results -o PREFIX TAG1 TAG2 TAG3 ...
```

This command will generate a series files whose names starts with
PREFIX.  The primary output is PREFIX.png which is a bar plot comparing
methods specified by different TAGS on all tasks found in
`settings.txt`.


## 3.2 Prepare label file.

A label file is a CSV file (comma-separated with header) containing sample labels. 
The file should contain the following columns:

- subjectId: patient ID.
- outcomeCount: label, 0 or 1.
- cohortStartDate: predicting date, with format YYYY-MM-DD e.g.
  1970-01-01.
- indexes (optional): > 0 for training, -1 for testing.

If indexing is missing, then training and testing splitting is randomly
created.

## 3.3 Creating Partition File.

The partition file must be generated from a set of text files containing
EMR data.  Suppose we have the following director:

```
data/tasks/some_task
└── some_cohort
    ├── condition_occurrence
    ├── drug_exposure
    └── procedure_occurrence
```

The following command is used to generate a partition file within the
directory.

```
omop_merge_dirs data/tasks/some_task/some_cohort
```

After finishing the script the file `partition` will appear in the same
directory as the text files and this partition file can be used in the
third column of `settings.txt`.

The three text files are CSV files without header.  The values should
not be quoted. The columns of each
file are listed below:

- `condition_occurrence`: subjectId, conceptId, startDate, endDate,
  TypeConceptId.
- `drug_exposure`: subjectId, conceptId, startDate, endDate,
  TypeConceptId.
- `procedure_exposure`: subjectId, conceptId, date,
  TypeConceptId.

Dates are of the format YYYY-MM-DD, eg 1970-01-01.  End dates could be
missing and can be filled with 0.  TypeConceptId are currently not used,
but must be present.


## 3.4 Finetuning

To fine-tune a pre-trained model, one needs to run `omop_eval_dl` with the following additional parameters:

* --net   A network architecture that is written to work with pretrain, e.g. "bottom" (without using time) or "bottom_decay" (using time decay).

* --pre path_to_a_pretrained_model    (A model you can use is in ~/SageMaker/pretrained/pre.ep29 on the wdong5 instance)

* --freeze K      This will freeze the pretrained portion of model in the first K iterations and only train the rest part.

* --tune 0.1      After the first K iterations, training rate will be multiplied by this number.


A complete example is below:

```
omop_eval_dl --dict data/dict --task LABEL --tag TAG --gpu  --learning_rate 0.001 --weight_decay 0.001 --whitelist_threshold 50 --seq_len 768 --dim 1024 --drop_rate 0.5 --epochs 100  --net bottom --pre path/to/model.ep29 --freeze 5 --tune 0.1
```

## 3.5 Pretraining

One needs to follow Section 4.1-4.3 to get data loaded in the working
directory.  The working directory must contain the following items:

- `conf.py`: local configuration file.
- `data/dict`
- `data/partitions/part-*****`: the partition files.
- A black list file listing forbidden concepts, one on each line.  Use
  `/dev/null` if a black list is not needed. (If not provided the
  program will try to scan PLP files and behavior is undocumented.)

The pretrain command is like below

```
omop_pretrain  --seq_len 768 --dim 1024 -v 10000 -t 0.01 --net pretrain --output output --gpu --batch 128 --epoch 10000 --epochs 50 --save 1  --black /dev/null
```

- `-v`: vocabulary size, the top this number of concepts in the
  dictionary will be used.
- `-t`: validation split.  For example, if there are 200 files in
  `data/partitions` and `-t 0.01` is set, then 200x0.01 = 2 partitions
  will be used for validation.
- `--output`: output directory.
- `--save`: save model every this number of epochs.
- `--epochs`: maximal number of epochs.
- `--epoch`: train this number of batches each epoch.

One needs to make sure at least one partition is used for validation.
So multiple files must be present in `data/partitions`, and `-t` must
not be too small.


# 4. Data ETL

## 4.1 Overview

Usually one maintains multiple OMOP data sources and want to train
models with different combinations of subsets of the sources.  Our
ETL process is to load sources into "targets", each target is a
dataset use to train one machine learning model.

## 4.2 Configuration



The user can specify configuration of database credentials in `conf.py`
in the working directory.

```
DB.update({'default': {
  "PGDATABASE": "...",
  "PGUSER": "...",
  "PGPASSWORD": "...",
  "PGHOST": "...",
  "PGPORT": "5439",
  "SCHEMA": "...",
  "COHORT_SCHEMA": "...",
}})

PARTITIONS = 1000

```

`DB` is a dictionary and the user can specify multiple databases.  For
those scripts that need to read from database, the database name can be
specified using `--db database`.  If unspecified, "default" is used.


Because the whole dataset is unlikely to fit in main memory, we use
map-reduce (via ehash) for preprocessing.  The dataset are randomly
partitioned into multiple files, with all data related to one person
hashed into a single partition file.  These partition files are to
be processed in parallel, so number of partitions are configured such
that:

- Typical parition size must be small enough so that each processor core
  can load one into memory for processing.
- Typicall partition size must be small enough so each one can be
  comfortably saved in S3.
- There must not be too many partitions.
- In pretrain, the training and validation are splitted by partitions
  and the minimal validation set can be specified is at least ONE
  partition.  So if we do not want validation to take too much time,
  the partition cannot be too big.

## 4.3 Loading Big Data

After specifying data source information in `conf.py`, run the
following:

```
$ omop_load [--db database] [-l limit] [-p processes]
# This may take a long time to run.
# It loads OMOP data from database into text files in 
# data/ehash/part-...
#	-l limit: maximal number of lines to read from each table
#			  for testing connectivity.
#	-p processes: how many parallel processes to use.
$ omop_merge [-p processes]
# This may take a long time to run.
# This is the reduce step.  It processes the partition directories
# and produce partition files that can be loaded by the C++ library.
```

The resulting dataset is very big and is used for pretraining
the embedding model.



## 4.4 Loading PLP Task Data

The feature data used by each individual task is also saved as
partition files so we don't need to access database each time we
run evaluation.  If the corresponding PLP results of task `task`
have be copied to `plp/task`, the partition file can be constructed
with

```
$ omop_load_plp task
```


# 4. Downstream Tasks with PLP

## 4.1 Overview

Downstream tasks are initialized by running R scripts based on
the PLP framework.  We need a special version of PLP slightly
modified to ensure that no test data is leaked to pretrain and
in cross-validation process.

Please see `docker/Dockerfile` for package installation instructions.

After PLP environment is setup, the downstream tasks can be run
within the `tasks` directory of this repository.

```
# Modify local_config.R to specify database credentials
create_cohort_{task}.R
plp.R {task}
```

After results are produced, they should be copied to the `plp`
directory under the working directory.

## 4.2 Redshift

On redshift, use the following for `local_config.R`

```
dbms <- "redshift"
```

Download drivers with

```
downloadJdbcDrivers(
  dbms,
  pathToDriver = Sys.getenv("DATABASECONNECTOR_JAR_FOLDER"),
  method = "auto",
  ...
)
```

The JDBC Driver directory can be specified using the following
environment variable.

```
export DATABASECONNECTOR_JAR_FOLDER=...chosen...directory...
```

## 5 License


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

