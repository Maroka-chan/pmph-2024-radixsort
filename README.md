# Futhark benchmark
```bash
futhark dataset --i32-bounds=-9999:9999 -g "[1000000]i32" > data.in
futhark c radix.fut && cat data.in | ./radix > data.out
futhark bench
```
