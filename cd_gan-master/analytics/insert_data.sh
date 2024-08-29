#!/usr/bin/bash

# You may update the file name for your references.

cat ~/datasets/Ship/select-data.csv | clickhouse-client --query="INSERT INTO ship.data FORMAT CSV" --input_format_allow_errors_ratio=0.001


cat ~/datasets/Ship/ship_type_mapping.csv | clickhouse-client --query="INSERT INTO ship.type FORMAT CSV" --input_format_allow_errors_ratio=0.001
