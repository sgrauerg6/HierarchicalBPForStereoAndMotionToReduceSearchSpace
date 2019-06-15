# genCudaConsts.py
# Scott Grauer-Gray
# September 7, 2010
# Python script with the 'general' CUDA constants...

# needed for the config file constants...
#import genProfileConfigFile

# path of CUDA on the system
CUDA_PATH_SYSTEM = '/opt/nvidia/cuda'

# path of the outer folder for application
OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST = '/home/ec2-user/cudaBeliefPropMotionPatRecJournalVims/cudaBeliefPropagation'

# constants defining the extensions used for the generation of .ptx and .cubin files...
PTX_EXT_RUN_CUDA_GEN_CONSTS = '.ptx'
CUBIN_EXT_RUN_CUDA_GEN_CONSTS = '.cubin'

# constant with the extension defining the architecture used; this 'comes before' the '.cubin' in the cubin file name
ARCH_USED_CUDA_GEN_CONSTS = ''

# constant defining a 'space' character for cuda belief propagation
SPACE_CHAR_CUDA_GEN_CONSTS = ' '

# constant defining a 'newline' character for cuda belief propagation
NEWLINE_CHAR_CUDA_GEN_CONSTS = '\n'

# constant defining the symbol to use for piping output
SYMBOL_PIPE_OUTPUT_CUDA_GEN_CONSTS = '>'

# constant defining the name of the file to pipe the output to...
FILE_NAME_PIPE_OUTPUT_CUDA_GEN_CONSTS = 'outputPipe.txt'

# keyword of max registers to adjust
MAX_REGISTERS_KEYWORD_MAKEFILE_CUDA_GEN_CONSTS = 'MAX_NUM_REGISTERS'

# constant defining the 'separator' between directories
CONST_SEPARATOR_DIRECTORIES_CUDA_CONSTS_GEN_CONSTS = '/'

# constant representing the 'PATH' environment variable name
CONST_PATH_ENVIRON_VAR_NAME_CUDA_GEN_CONSTS = 'PATH'

# constant representing the 'LD_LIBRARY_PATH' environment variable name
CONST_LD_LIBRARY_PATH_ENVIRON_VAR_NAME_CUDA_GEN_CONSTS = 'LD_LIBRARY_PATH'

# PATH environment variable to add in order to run Cuda
PATH_VAR_ADD_CUDA_GEN_CONSTS = CUDA_PATH_SYSTEM + '/bin'

# export of the library path environment variable in order to run Cuda
LIBRARY_PATH_VAR_ADD_CUDA_GEN_CONSTS = CUDA_PATH_SYSTEM + '/lib:' + CUDA_PATH_SYSTEM + '/lib64'

# constant representing the 'on' state of cuda profiling and also returning in csv format
ON_STATE_CUDA_VARS_CUDA_GEN_CONSTS = 1

# environment variable for Cuda profiling...
CUDA_PROFILING_ENVIR_VAR_CUDA_GEN_CONSTS = 'CUDA_PROFILE'

# environment variable for csv output of Cuda profiling...
CSV_OUTPUT_CUDA_PROFILING_ENVIR_VAR_CUDA_GEN_CONSTS = 'CUDA_PROFILE_CSV'

# constant to define the variable name representing the output configuration
FILE_NAME_OUTPUT_CONFIG_CUDA_GEN_CONSTS = 'CUDA_PROFILE_CONFIG'

# constant used to set the name of the log file
SET_PROFILER_LOG_FILE_NAME_CUDA_GEN_CONSTS = 'CUDA_PROFILE_LOG'

# constant used to define the name of the profiler file (this doesn't need to change...)
NAME_PROFILER_FILE_CUDA_GEN_CONSTS = 'signalsToProfile.config'

# constants defining the start, end, and file names
START_PARAM_CONFIG_FILE_CUDA_GEN_CONSTS = 0
END_PARAM_CONFIG_FILE_CUDA_GEN_CONSTS = 1 
FILE_NAME_OUTPUT_LOG_FILE_CUDA_GEN_CONSTS = 2

# constant representing a 'starting index'
STARTING_INDEX_VAL_RUN_CUDA_GEN_CONSTS = 0

# constant path representing the current directory
PATH_REP_CURRENT_DIR_CUDA_GEN_CONSTS = './'

# constant when the substring is not found in the given string
RESULT_SUBSTRING_NOT_FOUND_IN_STRING = -1

# constant for the number of bytes in a float val
NUM_BYTES_FLOAT_VAL = 4

# define the locations of the thread block width/height
THREAD_BLOCK_WIDTH_LOC = 0
THREAD_BLOCK_HEIGHT_LOC = 1

