# Compile the runBpStereoDiffMemModes host and kernel into an object file

# list the 'main' cu file here
MAIN_CU_FILE	:= runSmoothImageHostFuncts.cu

# list the cu files depended on here
CU_FILES_DEPEND := kernelSmoothImage.cu 

# give the name of the output object file here
OUTPUT_OBJECT_FILE_NAME := runSmoothImageHostFuncts.o

# define the maximum number of registers
MAX_NUM_REGISTERS := 256

################################################################################
# Rules and targets

include ../../common/common.mk

