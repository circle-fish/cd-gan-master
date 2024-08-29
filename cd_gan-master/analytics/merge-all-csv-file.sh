#!/usr/bin/bash

# use merge all csv to merge all csv file in ship data and provide one single output. then use insert_data to insert
# data into click house data analytics.

INPUT_FORLDER=$1
OUTPUT=$2

touch $OUTPUT

for filename in $INPUT_FORLDER/*.csv; do
    echo $filename
    cat $filename >> $OUTPUT
done
