.PHONY: clean run go debug release batch

PROG=PrimeOpenMP
CC ?= gcc
FLAGS_OPENMP ?= -fopenmp

release:
	$(CC) -Wall -g $(FLAGS_OPENMP) -O3 ${PROG}.c -o ${PROG}.exe -lm

batch:
	$(CC) -Wall -g $(FLAGS_OPENMP) -O3 -DNO_PROGRESS ${PROG}.c -o ${PROG}.exe -lm

debug:
	$(CC) -Wall -g $(FLAGS_OPENMP) -O0 ${PROG}.c -o ${PROG}.exe -lm

run go:
	OMP_DYNAMIC=false ./${PROG}.exe 0 10000000

clean:
	rm -f ${PROG}.exe ${PROG}.o *~
