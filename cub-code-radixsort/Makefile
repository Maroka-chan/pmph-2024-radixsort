CUB=cub-1.8.0

all: bench run-bench-script

bench: sorting_test.cu helper.cu.h
	nvcc -I$(CUB)/cub -o test-sort sorting_test.cu
	- ./test-sort 100000 0

run-bench-script:
	- pushd .. && ./bench.sh

clean:
	rm -f test-sort
