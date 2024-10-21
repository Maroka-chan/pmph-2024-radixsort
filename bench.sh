#!/usr/bin/env bash

futhark dataset --i32-bounds=-9999:9999 -g "[1000000]i32" > data.in
futhark c radix.fut
./radix < data.in > data.out
futhark bench radix.fut
