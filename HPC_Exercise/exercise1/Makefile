
FF = $(F77) 
ifeq ($(FC),ifort)
        FFLAGS = -qopt-zmm-usage=high -O3 -xHost -qopt-streaming-stores never #-qopt-report=5 
endif
ifeq ($(FC),gfortran)
        FFLAGS = -O3 -march=native -ffixed-line-length-90 -mprefer-vector-width=512 #-fopt-info-vec-all 
endif
ifeq ($(FC),pgf90)
	FF=pgfortran
        FFLAGS = -O3 -tp=skylake -Mvect=simd:512 -fast
endif


default: serial
serial: stream2.exe


stream2.exe: stream2.f mysecond.o
	$(FF) $(FFLAGS) -c stream2.f
	$(FF) $(FFLAGS) stream2.o mysecond.o -o stream2.exe $(LDFLAGS)

clean:
	rm -f stream2.exe *.o
