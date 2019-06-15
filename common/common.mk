# common.mk
# Scott Grauer-Gray
# 'common' makefile for compiling in case where there are multiple .cu files to be compiled into object files and then
# used for the file compilation

# define the path where CUDA is installed and location of the nvcc compiler
CUDA_INSTALL_PATH := /opt/nvidia/cuda
NVCC       := $(CUDA_INSTALL_PATH)/bin/nvcc 

# Basic directory setup for SDK
# (override directories only if they are not already defined)
NVIDIA_GPU_COMPUTING_SDK_DIR := /home/ec2-user/NVIDIA_GPU_Computing_SDK
ROOTDIR    := $(NVIDIA_GPU_COMPUTING_SDK_DIR)/C/common
LIBDIR     := $(ROOTDIR)/../lib
COMMONDIR  := $(ROOTDIR)/../common
SHAREDDIR  := $(ROOTDIR)/../../shared/


# define the directory for running belief propagation
BELIEF_PROP_ROOT := /home/ec2-user/cudaBeliefPropMotionPatRecJournalVims/cudaBeliefPropagation

# need to include for cutil to work...
INCLUDE_DIRS := -I. -I$(CUDA_INSTALL_PATH)/include -I$(COMMONDIR)/inc -I$(BELIEF_PROP_ROOT)/include -I$(SHAREDDIR)/inc

# have include directory as flag
COMPILE_FLAGS += $(INCLUDE_DIRS) -DUNIX 

# include the optimization level
COMPILE_FLAGS += -O2 -Xptxas -v

# rule for compilation of .cu file into object file
$(OUTPUT_OBJECT_FILE_NAME) : $(CU_FILES_DEPEND)
	$(NVCC) -c -keep -arch=compute_20 -code=sm_20 -maxrregcount=$(MAX_NUM_REGISTERS) $(OUTPUT_OBJECT_FILE_NAME) $(MAIN_CU_FILE) $(COMPILE_FLAGS)

# rule for clean-up of object and linkinfo files
clean : 
	@rm -rf *.o *.linkinfo
