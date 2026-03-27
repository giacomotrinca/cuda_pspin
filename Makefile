# Makefile for p-Spin 2+4 Complex Spherical MC (CUDA) + Analysis (C++)
# Target: Tesla V100S, CUDA 11.4

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
ANA_TARGET := $(BINDIR)/analysis

# CUDA architecture: V100S = sm_70
CUDA_ARCH  := -gencode arch=compute_70,code=sm_70

# Flags
NVCCFLAGS  := -std=c++14 $(CUDA_ARCH) -O3 --use_fast_math -I$(INCDIR) --expt-relaxed-constexpr
CXXFLAGS   := -std=c++14 -O3 -Wall
LDFLAGS    := -lcurand -lm

# Debug build
ifdef DEBUG
NVCCFLAGS  += -g -G -DDEBUG -lineinfo
CXXFLAGS   += -g -DDEBUG
else
NVCCFLAGS  += -DNDEBUG
CXXFLAGS   += -DNDEBUG
endif

# MC sources and objects (CUDA)
MC_SOURCES := $(filter-out $(SRCDIR)/analysis.cpp, $(wildcard $(SRCDIR)/*.cu))
MC_OBJECTS := $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(MC_SOURCES))

# Rules
.PHONY: all mc analysis clean directories

all: mc analysis

mc: directories $(MC_TARGET)

analysis: directories $(ANA_TARGET)

directories:
	@mkdir -p $(BINDIR) $(OBJDIR)

$(MC_TARGET): $(MC_OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

$(ANA_TARGET): $(SRCDIR)/analysis.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Dependencies (manual for now)
$(OBJDIR)/main.o:        $(SRCDIR)/main.cu $(INCDIR)/config.h $(INCDIR)/mc.h $(INCDIR)/hamiltonian.h $(INCDIR)/disorder.h
$(OBJDIR)/config.o:      $(SRCDIR)/config.cu $(INCDIR)/config.h
$(OBJDIR)/spins.o:       $(SRCDIR)/spins.cu $(INCDIR)/spins.h
$(OBJDIR)/disorder.o:    $(SRCDIR)/disorder.cu $(INCDIR)/disorder.h
$(OBJDIR)/hamiltonian.o: $(SRCDIR)/hamiltonian.cu $(INCDIR)/hamiltonian.h $(INCDIR)/disorder.h
$(OBJDIR)/mc.o:          $(SRCDIR)/mc.cu $(INCDIR)/mc.h $(INCDIR)/spins.h $(INCDIR)/disorder.h $(INCDIR)/hamiltonian.h $(INCDIR)/config.h
