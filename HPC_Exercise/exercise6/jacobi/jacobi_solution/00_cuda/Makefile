NVCC       = nvcc
CCFLAGS  = -O3 
NVCCFLAGS = -arch=sm_60 -m64
BIN =  laplace2d_cudac


default: release

release: laplace2d.cu
	$(NVCC) $(CCFLAGS) $(NVCCFLAGS) -o $(BIN) $<

run:
	./$(BIN)

clean:
	$(RM) $(BIN)

help:
	@echo "USAGE: make [ release | run | clean | help ]"
	@echo

debug: laplace2d.cu
	$(NVCC) $(CCFLAGS) -g -G $(NVCCFLAGS) -o $(BIN) $<
