# Makefile for p-Spin 2+4 Complex Spherical MC (CUDA) + Analysis (C++)
# Target: Tesla V100S (sm_70), CUDA 9.1, gcc 7.5

# Compilers
NVCC       := nvcc
CXX        := g++

# Directories
SRCDIR     := src
INCDIR     := include
LIBDIR     := lib
BINDIR     := bin
OBJDIR     := obj

# Targets
MC_TARGET  := $(BINDIR)/pspin24
SA_TARGET  := $(BINDIR)/simulated_annealing
ANA_TARGET := $(BINDIR)/analysis
ANA_SA_TARGET := $(BINDIR)/analysis_sa

# CUDA architecture: V100S = sm_70
# sm_70 = native SASS, compute_70 = embedded PTX for forward compat
CUDA_ARCH  := -gencode arch=compute_70,code=sm_70 \
               -gencode arch=compute_70,code=compute_70

# Host compiler optimisation (passed to gcc via -Xcompiler)
HOST_OPT   := -O3 -march=native -funroll-loops -ffast-math

# Flags
NVCCFLAGS  := -std=c++14 $(CUDA_ARCH) -O3 --use_fast_math \
               -I$(INCDIR) --expt-relaxed-constexpr \
               -Xcompiler "$(HOST_OPT)" \
               --ptxas-options=-O3
CXXFLAGS   := -std=c++14 $(HOST_OPT) -Wall -flto
LDFLAGS    := -lcurand -lm
LDFLAGS_CXX := -lm -flto

# Debug build
ifdef DEBUG
NVCCFLAGS  = -std=c++14 $(CUDA_ARCH) -O0 -g -G -DDEBUG -lineinfo \
              -I$(INCDIR) --expt-relaxed-constexpr -Xcompiler "-O0 -g"
CXXFLAGS   = -std=c++14 -O0 -g -DDEBUG -Wall
LDFLAGS_CXX = -lm
else
NVCCFLAGS  += -DNDEBUG
CXXFLAGS   += -DNDEBUG
endif

# Library sources (shared between mc and sa)
LIB_SOURCES := $(filter-out $(SRCDIR)/montecarlo.cu $(SRCDIR)/simulated_annealing.cu $(SRCDIR)/analysis.cpp, $(wildcard $(SRCDIR)/*.cu))
LIB_OBJECTS := $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(LIB_SOURCES))

# Rules
.PHONY: all mc sa analysis analysis_sa clean directories

all: mc sa analysis analysis_sa

mc: directories $(MC_TARGET)

sa: directories $(SA_TARGET)

analysis: directories $(ANA_TARGET)

analysis_sa: directories $(ANA_SA_TARGET)

directories:
	@mkdir -p $(BINDIR) $(OBJDIR)

$(MC_TARGET): $(OBJDIR)/montecarlo.o $(LIB_OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

$(SA_TARGET): $(OBJDIR)/simulated_annealing.o $(LIB_OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

$(ANA_TARGET): $(SRCDIR)/analysis.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS_CXX)

$(ANA_SA_TARGET): $(SRCDIR)/analysis_sa.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS_CXX)

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Dependencies (manual for now)
$(OBJDIR)/montecarlo.o:           $(SRCDIR)/montecarlo.cu $(INCDIR)/config.h $(INCDIR)/mc.h $(INCDIR)/hamiltonian.h $(INCDIR)/disorder.h
$(OBJDIR)/simulated_annealing.o:  $(SRCDIR)/simulated_annealing.cu $(INCDIR)/config.h $(INCDIR)/mc.h $(INCDIR)/disorder.h
$(OBJDIR)/config.o:      $(SRCDIR)/config.cu $(INCDIR)/config.h
$(OBJDIR)/spins.o:       $(SRCDIR)/spins.cu $(INCDIR)/spins.h
$(OBJDIR)/disorder.o:    $(SRCDIR)/disorder.cu $(INCDIR)/disorder.h
$(OBJDIR)/hamiltonian.o: $(SRCDIR)/hamiltonian.cu $(INCDIR)/hamiltonian.h $(INCDIR)/disorder.h
$(OBJDIR)/mc.o:          $(SRCDIR)/mc.cu $(INCDIR)/mc.h $(INCDIR)/spins.h $(INCDIR)/disorder.h $(INCDIR)/hamiltonian.h $(INCDIR)/config.h
