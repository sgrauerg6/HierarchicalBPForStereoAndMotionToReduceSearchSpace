//inputParamsConsts.h
//Scott Grauer-Gray
//September 2, 2010
//Constants for the input parameters...

#ifndef INPUT_PARAMS_CONSTS_H
#define INPUT_PARAMS_CONSTS_H

//give the 'index' of each parameter in the input...
const int IMAGE_1_FILE_INDEX = 1;
const int IMAGE_2_FILE_INDEX = 2;

const int MOTION_IMAGE_FILE_SAVE_INDEX = 3;
const int WIDTH_INPUT_IMAGES_INDEX = 4;
const int HEIGHT_INPUT_IMAGES_INDEX = 5;
const int NUM_BELIEF_PROP_LEVELS_INDEX = 6;
const int NUM_ITERS_PER_BELIEF_PROP_LEVEL_INDEX = 7;
const int DATA_COST_CAP_INDEX = 8;
const int SMOOTHNESS_COST_CAP_INDEX = 9;

const int DATA_COST_WEIGHT_INDEX = 10;
const int CURR_MOVE_INCR_X_INDEX = 11;
const int CURR_MOVE_INCR_Y_INDEX = 12;
const int SAMPLING_LEVEL_INDEX = 13;
const int X_MOVE_MIN_INDEX = 14;
const int X_MOVE_MAX_INDEX = 15;
const int Y_MOVE_MIN_INDEX = 16;
const int Y_MOVE_MAX_INDEX = 17;

const int MOTION_DISPLAY_RANGE_INDEX = 18;
const int GROUND_TRUTH_MOVE_FLO_INDEX = 19;
const int MOTION_MOVE_FLO_INDEX = 20;
const int PROP_CHANGE_MOVE_INDEX = 21;
const int SMOOTHING_SIGMA_INDEX = 22;

const int NUM_LEVEL_WITHIN_LEV_PYR_HIERARCH_INDEX = 23;

//constant representing 'null'
const char* NULL_CONST = "null";

#endif //INPUT_PARAMS_CONSTS_H

