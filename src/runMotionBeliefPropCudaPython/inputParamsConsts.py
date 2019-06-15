# inputParamsConsts.py
# Scott Grauer-Gray
# September 2, 2010
# Constants corresponding to the input parameters 

# needed for the path of the outer folder of the application
import genCudaConsts

# constant file path of the header file
FILE_PATH_HEADER_FILE = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST + '/include'

# constant name of the header file...
FILE_NAME_HEADER_FILE = 'genBeliefPropMotionConsts.h'

# constant for the 'start' of the header file for the input params...
START_HEADER_FILE_INPUT_PARAMS = '#ifndef GEN_BELIEF_PROP_MOTION_CONSTS_H\n#define GEN_BELIEF_PROP_MOTION_CONSTS_H\n'

# constant for the 'end' of the header file for the input params...
END_HEADER_FILE_INPUT_PARAMS = '#endif //GEN_BELIEF_PROP_MOTION_CONSTS_H\n'

# constant for the #define directive
DEFINE_DIR_HEADER_FILE = '#define'

# constants for the name of the maximum number of movements in the x and y directions
NAME_MAX_NUM_MOVES_X_DIR = 'GEN_MAX_NUM_MOVES_X_DIR' 
NAME_MAX_NUM_MOVES_Y_DIR = 'GEN_MAX_NUM_MOVES_Y_DIR' 

