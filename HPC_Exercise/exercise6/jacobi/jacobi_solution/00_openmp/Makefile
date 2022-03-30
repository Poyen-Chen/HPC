# Set CC or FC and FLAGS_FAST if your are not in the Cluster Environment
# CC       = pgcc
# FC       = pgf90
# FLAGS_FAST = -O3 
 
CCAP     = 60
CUDA     = 9.0
CCFLAGS  = $(FLAGS_FAST)
FCFLAGS  = $(FLAGS_FAST) -Mpreprocess
ACCFLAGS = -Minfo=accel -acc -ta=nvidia,cc$(CCAP),$(CUDA)

CCSRC   = laplace.c
FCSRC   = realtime.f90 checkResults.f90 laplace.f90
BIN     = laplace_openacc
ENVVARS = 

# OpenMP options
threads=24
ENVVARS+= OMP_PROC_BIND=true
ifneq ($(threads), )
ENVVARS+= OMP_NUM_THREADS=$(threads)
endif


ifneq ($(dev), )
DEVID=$(dev)
endif

ifeq ($(time),1)
ENVVARS+=PGI_ACC_TIME=1
endif

ifneq ($(notify), )
ENVVARS+=ACC_NOTIFY=$(notify)
endif

default: clean openmp_c

fc: $(FCSRC)
	$(FC) $(FCFLAGS) $(ACCFLAGS) -o $(BIN) $(FCSRC)

c: $(CCSRC)
	$(CC) $(CCFLAGS) $(ACCFLAGS) -o $(BIN) $(CCSRC)

serial_c: $(CCSRC)
	$(CC) $(CCFLAGS) -o $(BIN) $(CCSRC)

openmp_c: $(CCSRC)
	$(CC) $(CCFLAGS) $(FLAGS_OPENMP) -o $(BIN) $(CCSRC)

run:
	$(ENVVARS) ./$(BIN) $(DEVID)

clean:
	$(RM) $(BIN)
	$(RM) ./*.o
	$(RM) ./*.mod

help:
	@echo "USAGE: make [ c | fc | run | clean | help ]"
	@echo
	@echo "OPTIONS:"
	@echo "    make run dev=<GPU device ID>   Runs the program explicitly on the device specified (ID>=0)."
	@echo "    make [c|fc] time=1             Enables collecting simple timing information. Default: false."
	
