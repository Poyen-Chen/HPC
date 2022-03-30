PROG=for

debug:
	icpc -Wall -g -qopenmp ${PROG}.c -o ${PROG}.exe

release:
	icpc -Wall -g -qopenmp -O3 ${PROG}.c -o ${PROG}.exe

gprof:
	icpc -Wall -p -g -qopenmp -O2 ${PROG}.c -o ${PROG}.exe

run go: {PROG}.exe
	./${PROG}.exe

clean:
	rm -rf ${PROG}.exe ${PROG}.o *~ *.er gmon.out gprof.txt

{PROG}.exe: debug

