.DEFAULT_GOAL := help

TRITON_IR_DIR := kernels_ir/triton
CUTILE_IR_DIR := kernels_ir/cutile
BUILD_DIR := build
TRITON_BUILD_DIR := $(BUILD_DIR)/triton
CUTILE_BUILD_DIR := $(BUILD_DIR)/cutile
TRITON_LOWERED_DIR := $(TRITON_BUILD_DIR)/lowered
CUTILE_LOWERED_DIR := $(CUTILE_BUILD_DIR)/lowered
TRITON_SRCS := $(wildcard $(TRITON_IR_DIR)/*.mlir)
CUTILE_SRCS := $(wildcard $(CUTILE_IR_DIR)/*.mlir)
TRITON_OBJS := $(patsubst $(TRITON_IR_DIR)/%.mlir,$(TRITON_BUILD_DIR)/%.o,$(TRITON_SRCS))
CUTILE_OBJS := $(patsubst $(CUTILE_IR_DIR)/%.mlir,$(CUTILE_BUILD_DIR)/%.o,$(CUTILE_SRCS))
TRITON_LOWERED := $(patsubst $(TRITON_IR_DIR)/%.mlir,$(TRITON_LOWERED_DIR)/%.mlir,$(TRITON_SRCS))
CUTILE_LOWERED := $(patsubst $(CUTILE_IR_DIR)/%.mlir,$(CUTILE_LOWERED_DIR)/%.mlir,$(CUTILE_SRCS))

LAUNCHER_BUILD_DIR := $(BUILD_DIR)/launchers
LAUNCHER_KERNELS := layernorm resize sharpen softmax swiglu warp
LAUNCHER_CPP_BINS := $(addprefix $(LAUNCHER_BUILD_DIR)/,$(addsuffix _launch_cpp,$(LAUNCHER_KERNELS)))
LAUNCHER_TRITON_BINS := $(addprefix $(LAUNCHER_BUILD_DIR)/,$(addsuffix _launch_triton,$(LAUNCHER_KERNELS)))
LAUNCHER_CUTILE_BINS := $(addprefix $(LAUNCHER_BUILD_DIR)/,$(addsuffix _launch_cutile,$(LAUNCHER_KERNELS)))
LAUNCHER_BINS := $(LAUNCHER_CPP_BINS) $(LAUNCHER_TRITON_BINS) $(LAUNCHER_CUTILE_BINS)
LAUNCHER_OBJS := $(addsuffix .o,$(LAUNCHER_TRITON_BINS) $(LAUNCHER_CUTILE_BINS))

CXX_CPP := g++
CXX_LLVM := clang++
LAUNCHER_CXXFLAGS := -fopenmp -O3 -march=native

TRITON_OBJ_layernorm := $(TRITON_BUILD_DIR)/layernorm_fwd.o
TRITON_OBJ_resize := $(TRITON_BUILD_DIR)/resize.o
TRITON_OBJ_sharpen := $(TRITON_BUILD_DIR)/sharpen_3x3.o
TRITON_OBJ_softmax := $(TRITON_BUILD_DIR)/softmax_per_row.o
TRITON_OBJ_swiglu := $(TRITON_BUILD_DIR)/swiglu.o
TRITON_OBJ_warp := $(TRITON_BUILD_DIR)/warp.o

CUTILE_OBJ_layernorm := $(CUTILE_BUILD_DIR)/layernorm_fwd.o
CUTILE_OBJ_resize := $(CUTILE_BUILD_DIR)/resize.o
CUTILE_OBJ_sharpen := $(CUTILE_BUILD_DIR)/sharpen_3x3.o
CUTILE_OBJ_softmax := $(CUTILE_BUILD_DIR)/softmax_per_row.o
CUTILE_OBJ_swiglu := $(CUTILE_BUILD_DIR)/swiglu_fwd.o
CUTILE_OBJ_warp := $(CUTILE_BUILD_DIR)/warp.o

.PHONY: help triton cutile launchers clean
.SECONDEXPANSION:
.SECONDARY: $(TRITON_LOWERED) $(CUTILE_LOWERED) $(LAUNCHER_OBJS)

help:
	@echo "Run 'make triton', 'make cutile', or 'make launchers'."

triton: $(TRITON_OBJS)

cutile: $(CUTILE_OBJS)

launchers: $(LAUNCHER_BINS)

$(TRITON_BUILD_DIR)/%.o: $(TRITON_LOWERED_DIR)/%.mlir llvm_to_obj.sh
	@mkdir -p $(TRITON_BUILD_DIR)
	./llvm_to_obj.sh $< $@

$(TRITON_LOWERED_DIR)/%.mlir: $(TRITON_IR_DIR)/%.mlir lower_triton.sh
	@mkdir -p $(TRITON_LOWERED_DIR)
	./lower_triton.sh $< $@

$(CUTILE_BUILD_DIR)/%.o: $(CUTILE_LOWERED_DIR)/%.mlir llvm_to_obj.sh
	@mkdir -p $(CUTILE_BUILD_DIR)
	./llvm_to_obj.sh $< $@

$(CUTILE_LOWERED_DIR)/%.mlir: $(CUTILE_IR_DIR)/%.mlir lower_tileir.sh
	@mkdir -p $(CUTILE_LOWERED_DIR)
	./lower_tileir.sh $< $@

$(LAUNCHER_BUILD_DIR)/%_launch_cpp: kernels/%/launcher.cpp support/support.hpp
	@mkdir -p $(LAUNCHER_BUILD_DIR)
	$(CXX_CPP) -DLAUNCH_CPP $(LAUNCHER_CXXFLAGS) $< -o $@

$(LAUNCHER_BUILD_DIR)/%_launch_triton.o: kernels/%/launcher.cpp support/support.hpp
	@mkdir -p $(LAUNCHER_BUILD_DIR)
	$(CXX_LLVM) -DLAUNCH_TRITON $(LAUNCHER_CXXFLAGS) -c $< -o $@

$(LAUNCHER_BUILD_DIR)/%_launch_triton: $(LAUNCHER_BUILD_DIR)/%_launch_triton.o
	$(CXX_LLVM) $(LAUNCHER_CXXFLAGS) $< $(TRITON_OBJ_$*) -o $@

$(LAUNCHER_BUILD_DIR)/%_launch_cutile.o: kernels/%/launcher.cpp support/support.hpp
	@mkdir -p $(LAUNCHER_BUILD_DIR)
	$(CXX_LLVM) -DLAUNCH_CUTILE $(LAUNCHER_CXXFLAGS) -c $< -o $@

$(LAUNCHER_BUILD_DIR)/%_launch_cutile: $(LAUNCHER_BUILD_DIR)/%_launch_cutile.o
	$(CXX_LLVM) $(LAUNCHER_CXXFLAGS) $< $(CUTILE_OBJ_$*) -o $@

clean:
	rm -rf $(BUILD_DIR)
