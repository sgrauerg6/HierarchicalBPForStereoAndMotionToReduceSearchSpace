# runMotionBeliefPropConsts.py
# Scott Grauer-Gray
# August 30, 2010
# Constants for running "motion" belief propagation

# needed for the outer folder of the application
import genCudaConsts

# directory of the generated executable...
DIR_GENERATED_EXECUTABLE = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST

# full path of the executable
PATH_MOTION_BELIEF_PROP_EXECUTABLE = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST + '/execBpMotionCuda'

# directory of the Makefile...
DIR_MAKEFILE = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST

# command to 'make' a program via the Makefile...
MAKE_COMMAND = 'make'

# command to 'clean' a 'made' program
MAKE_CLEAN_COMMAND = 'make clean'

# name of the executable generated that is run...
EXECUTABLE_GENERATED_IS_RUN = ' '

# constant representing a "space"
SPACE_CONST = ' '

# constant representing a "newline"
NEW_LINE_CONST = '\n'




