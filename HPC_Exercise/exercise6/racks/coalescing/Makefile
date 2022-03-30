#SHELL=zsh
NVCC=nvcc
SRCDIR=.
SRC=racks.cu
EXE=racks.exe

ccap=60
FLAGS_GPU=-arch=sm_$(ccap)
FLAGS_FAST=-O3
FLAGS_DEBUG=-g -lineinfo

ifneq ( $(dev), )
DEVID=$(dev)
endif

all: clean release run

debug:
	$(NVCC) $(FLAGS_DEBUG) $(FLAGS_GPU) $(SRCDIR)/$(SRC) -o $(EXE) -lm

release:
	$(NVCC) $(FLAGS_FAST) $(FLAGS_GPU)  $(SRCDIR)/$(SRC) -o $(EXE) -lm

run:
	$(EXE) $(DEVID)

clean:
	@rm -rf *.exe core* *.o a.out 2> /dev/null

help:
	@echo "USAGE: make [ release | debug | run | clean | help ]"
	@echo
	@echo "OPTIONS:"
	@echo "    make [run] [dev=<GPU device ID>]                      Runs the program explicitly on the device specified (ID>=0)."
	@echo "    make [release | debug] [ccap=<compute capability>]    Sets the compute capability to compile for. Default: 20 (=2.0)."
