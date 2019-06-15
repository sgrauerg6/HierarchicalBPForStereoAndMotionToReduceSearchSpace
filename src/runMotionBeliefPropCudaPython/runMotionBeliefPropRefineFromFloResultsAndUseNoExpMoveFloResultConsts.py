# runMotionBeliefPropRefineFromFloResultsAndUseNoExpMoveFloResultConsts.py
# Scott Grauer-Gray
# November 30, 2010
# Constants for running the python script for running motion belief propagation taking into account the previously generated motion but also taking a result without using the expected motion

# needed for outer folder of application
import genCudaConsts

# constant for the path of the flo file of the result that doesn't take expected motion into account...
PATH_FLO_FILE_EXP_MOTION_NOT_USED = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST + '/other-data-gray/Grove2/floPathResultNoExpMotion.flo'

# weight of expected motion when it's not taken into account (it's 0.0, duh...)
WEIGHT_EXP_MOTION_WHEN_NOT_USED = 0.00

# smoothing sigma when expected motion not used
SMOOTHING_SIGMA_WHEN_EXP_MOTION_NOT_USED = 0.0

# constant for the path of the flo file of the 'merged result' that takes both motions into account
PATH_FLO_FILE_MERGED_RESULT_ACCOUNT_BOTH_MOTIONS = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST + '/other-data-gray/Grove2/outMotionGrove2MotionsMergedResult.flo'

# file path for the 'output' image that doesn't take expected motion into account...
FILE_PATH_OUTPUT_IMAGE_NO_ACCOUNT_EXP_MOTION = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST + '/other-data-gray/Grove2/outMotionGrove2NoExpMotion.ppm'

