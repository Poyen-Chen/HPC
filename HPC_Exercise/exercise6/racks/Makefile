#SHELL=zsh
SRCDIR=.
SRC=racks.c
EXE=racks.exe

FLAGS_DEBUG=-g -O0

ifndef dev
override dev = 0
endif

all: clean release run

debug:
	$(CC) $(FLAGS_OFFLOAD_OPENMP) $(FLAGS_DEBUG) $(SRCDIR)/$(SRC) -o $(EXE) -lm

release:
	$(CC) $(FLAGS_OFFLOAD_OPENMP) $(SRCDIR)/$(SRC) -o $(EXE) -lm

run:
	CUDA_VISIBLE_DEVICES=$(dev) $(EXE)

clean:
	@rm -rf *.exe core* *.o a.out 2> /dev/null

help:
	@echo "USAGE: make [ release | debug | run | clean | help ]"
	@echo
	@echo "OPTIONS:"
	@echo "    make [run] [dev=<GPU device ID>]                      Runs the program explicitly on the device specified (ID>=0)."
