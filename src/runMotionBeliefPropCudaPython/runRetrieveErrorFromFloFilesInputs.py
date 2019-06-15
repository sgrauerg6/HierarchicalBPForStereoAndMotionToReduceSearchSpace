# runRetrieveErrorFromFloFilesInputs.py
# Scott Grauer-Gray
# December 1, 2010
# Python script with the constants for retrieving the error from the flo file inputs...

# needed for outer folder of application
import genCudaConsts

GROUND_TRUTH_MOVEMENT_FILE_PATH = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST + '/other-gt-flow/Grove2/flow10.flo'
ORIG_MOVEMENT_FILE_PATH = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST + '/other-data-gray/Grove2/flow10OrigResult.flo'
ROUGH_MULT_IMAGE_MOVEMENT_FILE_PATH = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST + '/other-data-gray/Grove2/flow10RoughResults.flo'
ADJUST_ORIG_MOVEMENT_FILE_PATH = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST + '/other-data-gray/Grove2/floPathNoExpMotion.flo'
REFINED_MOVEMENT_FILE_PATH = genCudaConsts.OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST + '/other-data-gray/Grove2/flow10RefinedResults.flo'
