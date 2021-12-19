#!/bin/bash

input_file=$1
output_file=${input_file%.*}.csv

python xml_to_csv.py $input_file > $output_file
