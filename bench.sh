#!/usr/bin/env bash

trap "rm -f radix radix.c data.in data.out" EXIT

N_values=(100000 1000000 5000000 10000000)

for N in "${N_values[@]}"; do
    echo "Testing with N = $N"

    futhark dataset --i32-bounds=-9999:9999 -g "[$N]i32" > data.in

    # futhark cuda radix.fut
    # ./radix < data.in > data.out

    futhark bench radix.fut --backend=cuda -r 400
done