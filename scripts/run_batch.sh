#!/bin/bash

NUM_JOBS=20

for ((i=0; i<$NUM_JOBS; i++)); do
    echo "Starting batch $i..."
    nohup python process_batch.py $i > log_$i.txt 2>&1 &
done

echo "All $NUM_JOBS jobs launched in background."