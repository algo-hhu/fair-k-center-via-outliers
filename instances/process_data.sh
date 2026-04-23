#!/usr/bin/env bash

if [ ! -f "acs_income_2018_1-Year_person_CA_features.csv" ]; then
    python3 acs_dataset_getter.py
fi

python3 process_bank.py
python3 process_bank.py --normalize

python3 process_census.py
python3 process_census.py --normalize

python3 process_diabetes.py
python3 process_diabetes.py --normalize

python3 process_income.py
python3 process_income.py --normalize
