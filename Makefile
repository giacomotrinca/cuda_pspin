# p-Spin 2+4 -- CUDA 9.1 / gcc 7.5 / V100S (sm_70)

NVCC    = nvcc
CXX     = g++

# shared library objects (everything except the three main programs)
LIB_OBJ = obj/config.o obj/disorder.o obj/hamiltonian.o obj/mc.o obj/spins.o

NVFLAGS = -std=c++11 -arch=sm_70 -O3 -Iinclude -DNDEBUG
CXFLAGS = -std=c++11 -O3 -Wall -DNDEBUG
LIBS    = -lcurand -lm

.PHONY: all clean

all: dirs bin/pspin24 bin/simulated_annealing bin/parallel_tempering \
     bin/analysis bin/analysis_sa bin/analysis_pt
	@echo "Done."

dirs:
	@mkdir -p bin obj

# --- CUDA executables ---

bin/pspin24: obj/montecarlo.o $(LIB_OBJ) | dirs
	$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

bin/simulated_annealing: obj/simulated_annealing.o $(LIB_OBJ) | dirs
	$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

bin/parallel_tempering: obj/parallel_tempering.o $(LIB_OBJ) | dirs
	$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

# --- C++ analysis ---

bin/analysis: src/analysis.cpp | dirs
	$(CXX) $(CXFLAGS) -o $@ $< -lm

bin/analysis_sa: src/analysis_sa.cpp | dirs
	$(CXX) $(CXFLAGS) -o $@ $< -lm

bin/analysis_pt: src/analysis_pt.cpp | dirs
	$(CXX) $(CXFLAGS) -o $@ $< -lm

# --- pattern rule for .cu -> .o ---

obj/%.o: src/%.cu | dirs
	$(NVCC) $(NVFLAGS) -c -o $@ $<

clean:
	rm -rf obj bin
