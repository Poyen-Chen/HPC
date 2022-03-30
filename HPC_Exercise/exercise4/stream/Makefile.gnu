PROG=stream
DIM=40000000

debug:
	gcc -Wall -g -fopenmp -DN=${DIM} ${PROG}.c -o ${PROG}.exe -lm

release:
	gcc -Wall -g -fopenmp -O3 -DN=${DIM} ${PROG}.c -o ${PROG}.exe -lm

run go: ${PROG}.exe
	./${PROG}.exe

clean:
	rm -rf *.o *~ core* *.er ${PROG}.exe

${PROG}.exe: debug
