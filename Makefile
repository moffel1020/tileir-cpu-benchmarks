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

.PHONY: help triton cutile clean
.SECONDARY: $(TRITON_LOWERED) $(CUTILE_LOWERED)

help:
	@echo "Run 'make triton' or 'make cutile' to compile IR objects into $(BUILD_DIR)/."

triton: $(TRITON_OBJS)

cutile: $(CUTILE_OBJS)

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

clean:
	rm -rf $(BUILD_DIR)
