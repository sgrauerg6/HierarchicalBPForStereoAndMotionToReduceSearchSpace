# Makefile
# Scott Grauer-Gray
# Makefile to compile the executable to run belief propagation on the GPU

# object files that are only on the host and don't use nvcc to compile
HOST_OBJECTS := inputParams.o runSmoothImage.o runBeliefProp.o resultMovement.o bpImage.o cudaBeliefPropagation.o runBeliefPropMultImages.o inputParamsMultInputImages.o inputParamsGivenExpMotion.o

# all objects files, including the ones compiled via nvcc that use the GPU
ALL_OBJECTS := inputParams.o runSmoothImage.o runBeliefProp.o resultMovement.o bpImage.o cudaBeliefPropagation.o runBeliefPropMultImages.o inputParamsMultInputImages.o inputParamsGivenExpMotion.o \
		src/smoothImageCuda/runSmoothImageHostFuncts.o src/runBeliefPropCuda/runBeliefPropHostFuncts.o \
		src/genCudaFuncts/genCudaFuncts.o

# define the path where CUDA is installed and location of the nvcc compiler
CUDA_INSTALL_PATH := /opt/nvidia/cuda
NVCC       := $(CUDA_INSTALL_PATH)/bin/nvcc 

# Basic directory setup for SDK
NVIDIA_GPU_COMPUTING_SDK_DIR := /home/ec2-user/NVIDIA_GPU_Computing_SDK
ROOTDIR    := $(NVIDIA_GPU_COMPUTING_SDK_DIR)/C/common
LIBDIR     := $(ROOTDIR)/../lib
COMMONDIR  := $(ROOTDIR)/../common
SHAREDDIR  := $(ROOTDIR)/../../shared
OSLOWER := $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# define the directory for running belief propagation
BELIEF_PROP_ROOT := /home/ec2-user/cudaBeliefPropMotionPatRecJournalVims/cudaBeliefPropagation

# contains the directories to include...
INCLUDE_DIRS := -I. -I$(CUDA_INSTALL_PATH)/include -I$(COMMONDIR)/inc -I$(BELIEF_PROP_ROOT)/include -I$(SHAREDDIR)/inc

# contains the library files needed for linking
LIB := -L$(CUDA_INSTALL_PATH)/lib64 -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -lcuda -lcudart -lcutil_x86_64
	
# compile all the cu files to an object file output, then use gcc to compile the resulting files
executables : inputParams smoothImageCuda runBeliefPropCuda genCudaFuncts runSmoothImage runBeliefProp resultMovement bpImage cudaBeliefPropagation runBeliefPropMultImages inputParamsMultInputImages inputParamsGivenExpMotion 
	g++ -o execBpMotionCuda $(ALL_OBJECTS) $(LIB)
smoothImageCuda : 
	make -C src/smoothImageCuda
runBeliefPropCuda :
	make -C src/runBeliefPropCuda
genCudaFuncts : 
	make -C src/genCudaFuncts
inputParams : 
	g++ -c src/inputParams.cpp $(INCLUDE_DIRS)
inputParamsMultInputImages : 
	g++ -c src/inputParamsMultInputImages.cpp $(INCLUDE_DIRS)
inputParamsGivenExpMotion : 
	g++ -c src/inputParamsGivenExpMotion.cpp $(INCLUDE_DIRS)
runSmoothImage : 
	g++ -c src/runSmoothImage.cpp $(INCLUDE_DIRS)
runBeliefProp :
	g++ -c src/runBeliefProp.cpp $(INCLUDE_DIRS)
runBeliefPropMultImages : 
	g++ -c src/runBeliefPropMultImages.cpp $(INCLUDE_DIRS)
resultMovement : 
	g++ -c src/resultMovement.cpp $(INCLUDE_DIRS)
bpImage : 
	g++ -c src/bpImage.cpp $(INCLUDE_DIRS)
cudaBeliefPropagation :
	g++ -c src/cudaBeliefPropagation.cpp $(INCLUDE_DIRS)
clean : cleanSmoothImageCuda cleanRunBeliefPropCuda cleanGenCudaFuncts
	@rm -rf $(HOST_OBJECTS)
cleanSmoothImageCuda : 
	make clean -C src/smoothImageCuda
cleanRunBeliefPropCuda :
	make clean -C src/runBeliefPropCuda
cleanGenCudaFuncts : 
	make clean -C src/genCudaFuncts

