#!/usr/bin/env bash

python process_bank.py
python process_bank.py --normalize

python process_census.py
python process_census.py --normalize

python process_diabetes.py
python process_diabetes.py --normalize
