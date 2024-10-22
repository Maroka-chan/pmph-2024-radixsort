CUB=cub-1.8.0

all: bench

bench: sorting_test.cu helper.cu.h
	nvcc -I$(CUB)/cub -o test-sort sorting_test.cu
	./test-sort 10000000 0 

clean:
	rm -f test-sort