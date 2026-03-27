# Makefile for p-Spin 2+4 Complex Spherical MC (CUDA)
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

# Target
TARGET     := $(BINDIR)/pspin24

# CUDA architecture: V100S = sm_70
CUDA_ARCH  := -gencode arch=compute_70,code=sm_70

# Flags
NVCCFLAGS  := -std=c++14 $(CUDA_ARCH) -O2 -I$(INCDIR) --expt-relaxed-constexpr
LDFLAGS    := -lcurand -lm

# Debug build
ifdef DEBUG
NVCCFLAGS  += -g -G -DDEBUG -lineinfo
else
NVCCFLAGS  += -DNDEBUG
endif

# Sources and objects
SOURCES    := $(wildcard $(SRCDIR)/*.cu)
OBJECTS    := $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(SOURCES))

# Rules
.PHONY: all clean directories

all: directories $(TARGET)

directories:
	@mkdir -p $(BINDIR) $(OBJDIR)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

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
