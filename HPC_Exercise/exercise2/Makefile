norm: norm.exe 
	@echo "type <make run> for starting the program"

norm_block: norm_block.exe 
	@echo "type <make run> for starting the program"
	
single: norm_max.exe norm_1_col.exe norm_1_row.exe
	@echo "single executable for each algorithm"

single_block: norm_max.exe norm_1_col.exe norm_1_row.exe norm_1_block.exe
	@echo "single executable for each algorithm"

gprof: norm.gprof.exe
	./norm.gprof.exe
	gprof norm.gprof.exe | less

gprof2: norm.gprof2.exe
	./norm.gprof2.exe
	gprof norm.gprof2.exe | less

gprof3: norm.gprof3.exe
	./norm.gprof3.exe
	gprof norm.gprof3.exe | less

all: norm.exe norm_max.exe norm_1_col.exe norm_1_row.exe norm_1_block.exe norm.gprof.exe norm.gprof2.exe

norm.exe: norm.cpp
	${CXX} -o $@ $< ${FLAGS_FAST} -DNORM_MAX -DNORM_1_COL -DNORM_1_ROW

norm_block.exe: norm.cpp
	${CXX} -o norm.exe $< ${FLAGS_FAST} -DNORM_MAX -DNORM_1_COL -DNORM_1_ROW -DNORM_1_BLOCK

norm_max.exe: norm.cpp
	${CXX} -o $@ $< ${FLAGS_FAST} -DNORM_MAX 

norm_1_col.exe: norm.cpp
	${CXX} -o $@ $< ${FLAGS_FAST} -DNORM_1_COL

norm_1_row.exe: norm.cpp
	${CXX} -o $@ $< ${FLAGS_FAST} -DNORM_1_ROW

norm_1_block.exe: norm.cpp
	${CXX} -o $@ $< ${FLAGS_FAST} -DNORM_1_BLOCK

norm.gprof.exe: norm.cpp
	${CXX} -pg -o $@ $< ${FLAGS_FAST} -DNORM_MAX -DNORM_1_COL -DNORM_1_ROW -DNORM_1_BLOCK

norm.gprof2.exe: norm.cpp
	${CXX} -pg -fno-inline -o $@ $< ${FLAGS_FAST} -DNORM_MAX -DNORM_1_COL -DNORM_1_ROW -DNORM_1_BLOCK

norm.gprof3.exe: norm.cpp
	${CXX} -pg -O0 -o $@ $< -DNORM_MAX -DNORM_1_COL -DNORM_1_ROW -DNORM_1_BLOCK

run: norm.exe
	./norm.exe

help:
	@echo "possible targets: norm single gprof gprof2 run doc clean"

#-----------------------------------------------------------------------
# Command to generate documentation
#-----------------------------------------------------------------------
doc:
	doxygen doxy.conf

clean:
	rm -f norm.exe
	rm -f norm_max.exe
	rm -f norm_1_col.exe
	rm -f norm_1_row.exe
	rm -f norm_1_block.exe
	rm -f norm.gprof*.exe
	rm -f *.o

distclean: clean
	rm -rf doc

