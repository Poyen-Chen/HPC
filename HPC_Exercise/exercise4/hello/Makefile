PROG=hello

debug:
	icc -Wall -g -qopenmp ${PROG}.c -o ${PROG}.exe

release:
	icc -Wall -g -qopenmp -O3 ${PROG}.c -o ${PROG}.exe

run go:
	./${PROG}.exe

clean:
	rm -f ${PROG}.exe ${PROG}.o *~
