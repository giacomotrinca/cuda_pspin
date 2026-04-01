# p-Spin 2+4 -- CUDA build
# Default: V100S (sm_70).   Use  make ... visnu=1  for GTX 680 (sm_30).

NVCC    = nvcc
CXX     = g++

# shared library objects (everything except the three main programs)
LIB_OBJ = obj/config.o obj/disorder.o obj/hamiltonian.o obj/mc.o obj/spins.o

# sparse variant: smoothed-cube constraint + sparse H4
LIB_SPARSE_OBJ = obj/config.o obj/disorder.o obj/mc_sparse.o obj/spins_sparse.o

ifdef visnu
  ARCH = sm_30
else
  ARCH = sm_70
endif

NVFLAGS = -std=c++11 -arch=$(ARCH) -O3 -Iinclude -DNDEBUG
CXFLAGS = -std=c++17 -O3 -Wall -DNDEBUG -Iinclude/sciplot
LIBS    = -lcurand -lm

.PHONY: all clean mc sa pt pts analysis_mc analysis_sa analysis_pt bench bench_plot

all: dirs bin/pspin24 bin/simulated_annealing bin/parallel_tempering \
     bin/analysis bin/analysis_sa bin/analysis_pt
	@echo "Done."

mc:           dirs bin/pspin24
sa:           dirs bin/simulated_annealing
pt:           dirs bin/parallel_tempering
pts:          dirs bin/parallel_tempering_sparse
analysis_mc:  dirs bin/analysis
analysis_sa:  dirs bin/analysis_sa
analysis_pt:  dirs bin/analysis_pt
bench:        dirs bin/benchmark
bench_plot:   dirs bin/plot_benchmark

dirs:
	@mkdir -p bin obj

# --- CUDA executables ---

bin/pspin24: obj/montecarlo.o $(LIB_OBJ) | dirs
	$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

bin/simulated_annealing: obj/simulated_annealing.o $(LIB_OBJ) | dirs
	$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

bin/parallel_tempering: obj/parallel_tempering.o $(LIB_OBJ) | dirs
	$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

bin/parallel_tempering_sparse: obj/parallel_tempering_sparse.o $(LIB_SPARSE_OBJ) | dirs
	$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

bin/benchmark: obj/benchmark.o $(LIB_OBJ) | dirs
	$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

# --- C++ analysis ---

bin/analysis: src/analysis.cpp | dirs
	$(CXX) $(CXFLAGS) -o $@ $< -lm

bin/analysis_sa: src/analysis_sa.cpp | dirs
	$(CXX) $(CXFLAGS) -o $@ $< -lm

bin/analysis_pt: src/analysis_pt.cpp | dirs
	$(CXX) $(CXFLAGS) -o $@ $< -lm

bin/plot_benchmark: src/plot_benchmark.cpp | dirs
	$(CXX) $(CXFLAGS) -o $@ $< -lm

# --- pattern rule for .cu -> .o ---

obj/%.o: src/%.cu | dirs
	$(NVCC) $(NVFLAGS) -c -o $@ $<

clean:
	rm -rf obj bin
