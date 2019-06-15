# runMotionBeliefPropGivenExpMotionInputs.py
# Scott Grauer-Gray
# November 29, 2010
# Python script for the inputs to run motion belief propagation given a flo file with the expected motion

# needed for outer folder of application
import genCudaConsts

# file to save the output text with the input params/output results
FILE_PATH_OUTPUT_TEXT = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST + '/other-data-gray/Grove2/outInfo.txt'

# path for the input image 1
FILE_PATH_INPUT_IMAGE_1 = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST + '/other-data-gray/Grove2/frame10.pgm'

# path for the input image 2
FILE_PATH_INPUT_IMAGE_2 = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST + '/other-data-gray/Grove2/frame11.pgm'

FILE_PATH_SAVE_RESULTING_MOTION_IMAGE = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST + '/other-data-gray/Grove2/grove2Frame10To11RefinedResults.ppm'

FILE_PATH_EXPECTED_MOTION = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST + '/other-gt-flow/Grove2/flowGrove2MultImages.flo'

WIDTH_INPUT_IMAGE_BP_MOTION_TEST = 640 
HEIGHT_INPUT_IMAGE_BP_MOTION_TEST = 480 

NUM_BELIEF_PROP_LEVELS_BP_MOTION_TEST = 7
NUM_BELIEF_PROP_ITERS_PER_LEVEL_BP_MOTION_TEST = 1000

DATA_COST_CAP_BP_MOTION_TEST = 15.0
SMOOTHNESS_COST_CAP_BP_MOTION_TEST = 3.4
DATA_COST_WEIGHT_BP_MOTION_TEST = 0.07

CURRENT_MOVE_INCREMENT_X_BP_MOTION_TEST = 30.0
CURRENT_MOVE_INCREMENT_Y_BP_MOTION_TEST = 30.0
SAMPLING_LEVEL_BP_MOTION_TEST = 1.0

X_MOVE_MIN_BP_MOTION_TEST = -60.0
X_MOVE_MAX_BP_MOTION_TEST = 60.0

Y_MOVE_MIN_BP_MOTION_TEST = -60.0
Y_MOVE_MAX_BP_MOTION_TEST = 60.0

RANGE_DISPLAY_MOTION_TEST = 7.5 

# file path of the ground truth movement in flo format
GROUND_TRUTH_MOVEMENT_FLO_FORMAT = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST + '/other-gt-flow/Grove2/flow10.flo'

# file path of the saved movement data in flo format
FILE_PATH_SAVED_MOVEMENT_DATA_FLO_FORMAT = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST + '/other-gt-flow/Grove2/grove2Frame10to11RefinedResults.flo'

PROP_CHANGE_MOVE_NEXT_LEVEL_BP_MOTION_TEST = 0.5

SMOOTHING_SIGMA_BP_MOTION_TEST = 0.0

EST_MOVEMENT_CAP_BP_MOTION_TEST = 3.4
EST_MOVEMENT_WEIGHT_BP_MOTION_TEST = 0.03

# constant saying that belief propagation is not processed as a pyramid hierarchy...
NOT_PROCESSED_IN_PYR_HIERARCHY_CONST_BP_MOTION_TEST = -999

# number of `pyramid levels' to be processed in each level of the hierarchy...
NUM_PYR_LEVELS_EACH_LEV_HIERARCHY_BP_MOTION_TEST = 5

