# setupCudaBeliefProp.py
# Scott Grauer-Gray
# September 7, 2010
# Functions for setting up CUDA belief propagation...

# needed to run functions that mess with the settings on the os...
import os

# needed for 'general constants' used in setting up CUDA belief propagation
import genCudaConsts

# function to initialize such that cuda belief propagation can be run
def initToRunCudaBeliefProp():

	# set the path and library path environment variables such that cuda belief can be run
	os.environ[genCudaConsts.CONST_PATH_ENVIRON_VAR_NAME_CUDA_GEN_CONSTS] += os.pathsep + genCudaConsts.PATH_VAR_ADD_CUDA_GEN_CONSTS	
	os.environ[genCudaConsts.CONST_LD_LIBRARY_PATH_ENVIRON_VAR_NAME_CUDA_GEN_CONSTS] = genCudaConsts.LIBRARY_PATH_VAR_ADD_CUDA_GEN_CONSTS


# function to set up Cuda profiling
def setUpCudaProfiling():

	# make the changes to the environment variables to set up Cuda profiling with the output in csv format
	# (basically, set them to on...)
	os.environ[genCudaConsts.CUDA_PROFILING_ENVIR_VAR_CUDA_GEN_CONSTS] = str(genCudaConsts.ON_STATE_CUDA_VARS_CUDA_GEN_CONSTS)
	os.environ[genCudaConsts.CSV_OUTPUT_CUDA_PROFILING_ENVIR_VAR_CUDA_GEN_CONSTS] = str(genCudaConsts.ON_STATE_CUDA_VARS_CUDA_GEN_CONSTS)

	# set the file name of the configuration file
	os.environ[genCudaConsts.FILE_NAME_OUTPUT_CONFIG_CUDA_GEN_CONSTS] = genCudaConsts.NAME_PROFILER_FILE_CUDA_GEN_CONSTS
