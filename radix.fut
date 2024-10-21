import "lib/github.com/diku-dk/sorts/radix_sort"

-- Primes: Flat-Parallel Version
-- ==
-- input @ data.in
-- output @ data.out
def main (numbers: []i32) = radix_sort_int i32.num_bits i32.get_bit numbers
