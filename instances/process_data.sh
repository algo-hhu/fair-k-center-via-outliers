#!/usr/bin/env bash

python3 acs_dataset_getter.py

python3 process_bank.py
python3 process_bank.py --normalize

python3 process_census.py
python3 process_census.py --normalize

python3 process_diabetes.py
python3 process_diabetes.py --normalize

python3 process_income.py
