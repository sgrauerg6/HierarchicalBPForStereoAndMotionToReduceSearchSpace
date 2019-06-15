# runMotionBeliefPropMultImagesInputs.py
# Scott Grauer-Gray
# September 28, 2010
# Python script for the inputs to run motion belief propagation on multiple images

# needed for outer folder path
import genCudaConsts

# file to save the output text with the input params/output results
FILE_PATH_OUTPUT_TEXT = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST + '/other-data-gray/Grove2/outInfo.txt'

# 'base' path for the input image
FILE_PATH_INPUT_IMAGE_FILE_BASE = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST + '/other-data-gray/Grove2/frame'

# 'starting' image number...
FILE_PATH_START_IMAGE_NUM = 7

# 'ending' image number...
FILE_PATH_END_IMAGE_NUM = 11

# 'end' extention to the image name
FILE_PATH_END_EXTENSION = '.pgm'

FILE_PATH_SAVE_RESULTING_MOTION_IMAGE = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST + '/other-data-gray/Grove2/outMotionGrove2MultImages.ppm'

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
FILE_PATH_SAVED_MOVEMENT_DATA_FLO_FORMAT = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST + '/other-gt-flow/Grove2/flowGrove2MultImages.flo'

PROP_CHANGE_MOVE_NEXT_LEVEL_BP_MOTION_TEST = 0.5

SMOOTHING_SIGMA_BP_MOTION_TEST = 0.0

EST_MOVEMENT_CAP_BP_MOTION_TEST = 3.4
EST_MOVEMENT_WEIGHT_BP_MOTION_TEST = 0.03

# constant saying that belief propagation is not processed as a pyramid hierarchy...
NOT_PROCESSED_IN_PYR_HIERARCHY_CONST_BP_MOTION_TEST = -999

# number of `pyramid levels' to be processed in each level of the hierarchy...
NUM_PYR_LEVELS_EACH_LEV_HIERARCHY_BP_MOTION_TEST = 5

