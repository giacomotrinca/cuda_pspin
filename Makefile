# ╔══════════════════════════════════════════════════════════════╗
# ║  p-Spin 2+4  ──  CUDA build                                ║
# ║  Default: V100S (sm_70)                                     ║
# ║    make ... visnu=1   →  GTX 680   (sm_30)                  ║
# ║    make ... kraken=1  →  Tesla K20c (sm_35)                 ║
# ╚══════════════════════════════════════════════════════════════╝

NVCC    = nvcc

# ── pretty output ────────────────────────────────────────────
# Use  make V=1  to see raw compiler commands
V ?= 0
ifeq ($(V),0)
  Q = @
else
  Q =
endif

# colors
C_RST   = \033[0m
C_BOLD  = \033[1m
C_CYN   = \033[36m
C_GRN   = \033[32m
C_YEL   = \033[33m
C_MAG   = \033[35m
C_RED   = \033[31m
C_DIM   = \033[2m
C_BLU   = \033[34m

# status prefixes  (BMP-only symbols — render in any terminal)
P_CU    = @printf '  $(C_CYN)$(C_BOLD)⚡ NVCC$(C_RST)     %s\n'
P_CXX   = @printf '  $(C_MAG)$(C_BOLD)◆  C++$(C_RST)      %s\n'
P_LINK  = @printf '  $(C_GRN)$(C_BOLD)►  LINK$(C_RST)     %s\n'
P_TEST  = @printf '  $(C_YEL)$(C_BOLD)✦  TEST$(C_RST)     %s\n'
P_CLEAN = @printf '  $(C_RED)$(C_BOLD)✕  CLEAN$(C_RST)    %s\n'

define BANNER
	@printf '\n'
	@printf '  $(C_BLU)$(C_BOLD)╔══════════════════════════════════════════════════╗$(C_RST)\n'
	@printf '  $(C_BLU)$(C_BOLD)║$(C_RST)  $(C_CYN)$(C_BOLD)%-48s$(C_RST)$(C_BLU)$(C_BOLD)║$(C_RST)\n' 'p-Spin 2+4  ·  Random Laser CUDA toolkit'
	@printf '  $(C_BLU)$(C_BOLD)║$(C_RST)  %-48s$(C_BLU)$(C_BOLD)║$(C_RST)\n' 'arch=$(ARCH)  nvcc=$(NVCC)'
	@printf '  $(C_BLU)$(C_BOLD)╚══════════════════════════════════════════════════╝$(C_RST)\n'
	@printf '\n'
endef

define DONE_MSG
	@printf '\n'
	@printf '  $(C_GRN)$(C_BOLD)>> Build complete!$(C_RST)\n'
	@printf '  $(C_DIM)Binaries in ./bin/$(C_RST)\n'
	@printf '\n'
endef

define TESTS_DONE_MSG
	@printf '\n'
	@printf '  $(C_GRN)$(C_BOLD)>> All test binaries built.$(C_RST)\n'
	@printf '\n'
endef

# ── object sets ──────────────────────────────────────────────

# shared library objects (everything except the three main programs)
LIB_OBJ = obj/config.o obj/disorder.o obj/hamiltonian.o obj/mc.o obj/spins.o

# sparse variant: smoothed-cube constraint + sparse H4
LIB_SPARSE_OBJ = obj/config.o obj/disorder.o obj/mc_sparse.o obj/spins_sparse.o

# combined (dense + sparse) for benchmarks that need both
LIB_ALL_OBJ = obj/config.o obj/disorder.o obj/hamiltonian.o obj/mc.o obj/spins.o \
              obj/mc_sparse.o obj/spins_sparse.o

ifdef visnu
  ARCH = sm_30
else ifdef kraken
  ARCH = sm_35
else ifdef dariah
  ARCH = sm_70
  # dariah modules: load CUDA toolkit + gcc via environment-modules
  _ := $(shell bash -c 'source /usr/share/Modules/init/bash 2>/dev/null; module use /apps/CNR/modulefiles/nvidia; module load nvhpc/21.5' >/dev/null 2>&1)
  NVCC_PATH := $(shell bash -c 'source /usr/share/Modules/init/bash 2>/dev/null; module use /apps/CNR/modulefiles/nvidia; module load nvhpc/21.5; which nvcc 2>/dev/null')
  CXX_PATH  := $(shell bash -c 'source /usr/share/Modules/init/bash 2>/dev/null; module use /apps/CNR/modulefiles/nvidia; module load nvhpc/21.5; which g++ 2>/dev/null')
  ifneq ($(NVCC_PATH),)
    NVCC = $(NVCC_PATH)
  endif
  ifneq ($(CXX_PATH),)
    CXX = $(CXX_PATH)
  endif
else
  ARCH = sm_70
endif

ifdef kraken
  CXX = $(HOME)/gcc7/bin/g++
else
  CXX = g++
endif

NVFLAGS = -std=c++11 -arch=$(ARCH) -O3 -Iinclude -DNDEBUG -Wno-deprecated-gpu-targets
CXFLAGS = -std=c++17 -O3 -Wall -DNDEBUG -Iinclude/sciplot
ifdef dariah
  # static link: compute nodes don't mount /apps/ so .so files are unavailable
  LIBS  = -lcurand_static -lculibos -lpthread -ldl -lm
else
  LIBS  = -lcurand -lm
endif

.PHONY: all clean mc sa pt pts analysis_mc analysis_sa analysis_pt bench bench_plot bench_pt smcu_test \
       test_quartet test_spherical test_delta_e test_inf_temp test_detailed_balance \
       test_fmc_mask test_replica_exchange test_sparse_dense test_mean_shift test_fmc_survivors tests

# ── top-level targets ────────────────────────────────────────

all: dirs bin/pspin24 bin/simulated_annealing bin/parallel_tempering \
     bin/analysis bin/analysis_sa bin/analysis_pt
	$(DONE_MSG)

mc:           dirs bin/pspin24
sa:           dirs bin/simulated_annealing
pt:           dirs bin/parallel_tempering
pts:          dirs bin/parallel_tempering_sparse
analysis_mc:  dirs bin/analysis
analysis_sa:  dirs bin/analysis_sa
analysis_pt:  dirs bin/analysis_pt
bench:        dirs bin/benchmark
bench_plot:   dirs bin/plot_benchmark
bench_pt:     dirs bin/bench_pt_scaling
smcu_test:    dirs bin/smoothed_cube

# --- test suite ---
test_quartet:          dirs bin/test_quartet_index
test_spherical:        dirs bin/test_spherical
test_delta_e:          dirs bin/test_delta_e
test_inf_temp:         dirs bin/test_inf_temp
test_detailed_balance: dirs bin/test_detailed_balance
test_fmc_mask:         dirs bin/test_fmc_mask
test_replica_exchange: dirs bin/test_replica_exchange
test_sparse_dense:     dirs bin/test_sparse_dense
test_mean_shift:       dirs bin/test_mean_shift
test_fmc_survivors:    dirs bin/test_fmc_survivors

tests: test_quartet test_spherical test_delta_e test_inf_temp test_detailed_balance \
       test_fmc_mask test_replica_exchange test_sparse_dense test_mean_shift test_fmc_survivors
	$(TESTS_DONE_MSG)

dirs:
	@mkdir -p bin obj
	$(BANNER)

# ── CUDA executables ─────────────────────────────────────────

bin/pspin24: obj/montecarlo.o $(LIB_OBJ) | dirs
	$(P_LINK) '$@'
	$(Q)$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

bin/simulated_annealing: obj/simulated_annealing.o $(LIB_OBJ) | dirs
	$(P_LINK) '$@'
	$(Q)$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

bin/parallel_tempering: obj/parallel_tempering.o $(LIB_OBJ) | dirs
	$(P_LINK) '$@'
	$(Q)$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

bin/parallel_tempering_sparse: obj/parallel_tempering_sparse.o $(LIB_SPARSE_OBJ) | dirs
	$(P_LINK) '$@'
	$(Q)$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

bin/benchmark: obj/benchmark.o $(LIB_OBJ) | dirs
	$(P_LINK) '$@'
	$(Q)$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

bin/bench_pt_scaling: obj/bench_pt_scaling.o $(LIB_ALL_OBJ) | dirs
	$(P_LINK) '$@'
	$(Q)$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

bin/smoothed_cube: obj/smoothed_cube.o obj/spins_sparse.o | dirs
	$(P_LINK) '$@'
	$(Q)$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

# ── test suite binaries ──────────────────────────────────────

bin/test_quartet_index: obj/test_quartet_index.o | dirs
	$(P_TEST) '$@'
	$(Q)$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

bin/test_spherical: obj/test_spherical.o $(LIB_OBJ) | dirs
	$(P_TEST) '$@'
	$(Q)$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

bin/test_delta_e: obj/test_delta_e.o $(LIB_OBJ) | dirs
	$(P_TEST) '$@'
	$(Q)$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

bin/test_inf_temp: obj/test_inf_temp.o $(LIB_OBJ) | dirs
	$(P_TEST) '$@'
	$(Q)$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

bin/test_detailed_balance: obj/test_detailed_balance.o $(LIB_OBJ) | dirs
	$(P_TEST) '$@'
	$(Q)$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

bin/test_fmc_mask: obj/test_fmc_mask.o obj/disorder.o | dirs
	$(P_TEST) '$@'
	$(Q)$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

bin/test_replica_exchange: obj/test_replica_exchange.o $(LIB_OBJ) | dirs
	$(P_TEST) '$@'
	$(Q)$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

bin/test_sparse_dense: obj/test_sparse_dense.o $(LIB_SPARSE_OBJ) | dirs
	$(P_TEST) '$@'
	$(Q)$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

bin/test_mean_shift: obj/test_mean_shift.o obj/disorder.o | dirs
	$(P_TEST) '$@'
	$(Q)$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

bin/test_fmc_survivors: obj/test_fmc_survivors.o obj/disorder.o | dirs
	$(P_TEST) '$@'
	$(Q)$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

# ── C++ analysis ─────────────────────────────────────────────

bin/analysis: src/analysis.cpp | dirs
	$(P_CXX) '$@'
	$(Q)$(CXX) $(CXFLAGS) -o $@ $< -lm

bin/analysis_sa: src/analysis_sa.cpp | dirs
	$(P_CXX) '$@'
	$(Q)$(CXX) $(CXFLAGS) -o $@ $< -lm

bin/analysis_pt: src/analysis_pt.cpp | dirs
	$(P_CXX) '$@'
	$(Q)$(CXX) $(CXFLAGS) -o $@ $< -lm -lpthread

bin/plot_benchmark: src/plot_benchmark.cpp | dirs
	$(P_CXX) '$@'
	$(Q)$(CXX) $(CXFLAGS) -o $@ $< -lm

# ── pattern rule for .cu → .o ────────────────────────────────

obj/%.o: src/%.cu | dirs
	$(P_CU) '$<'
	$(Q)$(NVCC) $(NVFLAGS) -c -o $@ $<

# ── clean ─────────────────────────────────────────────────────

clean:
	$(P_CLEAN) 'obj/ bin/'
	$(Q)rm -rf obj bin
	@printf '  $(C_DIM)All clean.$(C_RST)\n'
