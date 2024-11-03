CUB=cub-1.8.0

all: bench

bench: sorting_test.cu helper.cu.h
	nvcc -I$(CUB)/cub -o test-sort sorting_test.cu
	./test-sort 16 0
	./test-sort 1000 0
	./test-sort 1100 0
	./test-sort 2500 0
	./test-sort 5000 0

clean:
	rm -f test-sort
