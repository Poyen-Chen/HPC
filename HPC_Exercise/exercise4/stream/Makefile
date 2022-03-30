PROG=stream
DIM=40000000

debug:
	icc -Wall -g -qopenmp -DN=${DIM} ${PROG}.c -o ${PROG}.exe

release:
	icc -Wall -g -qopenmp -O3 -DN=${DIM} ${PROG}.c -o ${PROG}.exe

run go: ${PROG}.exe
	./${PROG}.exe

clean:
	rm -rf *.o *~ core* *.er ${PROG}.exe

${PROG}.exe: debug
