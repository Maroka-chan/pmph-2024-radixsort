#!/usr/bin/env -S nix shell nixpkgs#sshpass --command bash

rsync -ae 'sshpass -f /home/maroka/hendrix_pass ssh -F ../PMPH/config' ./cub-code-radixsort hendrix:
sshpass -f /home/maroka/hendrix_pass ssh -F ../PMPH/config hendrix "pushd cub-code-radixsort && make"
