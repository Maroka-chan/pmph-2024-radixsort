#!/usr/bin/env bash

rsync -ae 'ssh -F ../PMPH/config' ./cub-code-radixsort hendrix:
ssh -F ../PMPH/config hendrix "pushd cub-code-radixsort && make"
