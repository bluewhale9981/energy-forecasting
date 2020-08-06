#!/bin/bash
mkdir -p data/raw

wget https://s3.amazonaws.com/drivendata/data/55/public/cold_start_test.csv
wget https://s3.amazonaws.com/drivendata/data/55/public/meta.csv
wget https://s3.amazonaws.com/drivendata/data/55/public/consumption_train.csv
wget https://s3.amazonaws.com/drivendata/data/55/public/submission_format.csv
mv cold_start_test.csv data/raw/cold_start_test.csv
mv meta.csv data/raw/meta.csv
mv consumption_train.csv data/raw/consumption_train.csv
mv submission_format.csv data/raw/submission_format.csv
