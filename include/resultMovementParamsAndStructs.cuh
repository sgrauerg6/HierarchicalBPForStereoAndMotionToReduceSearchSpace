//resultMovementParamsAndStructs.cuh
//Scott Grauer-Gray
//June 30, 2009
//Declares the resulting movement parameters and structs

#ifndef RESULT_MOVEMENT_PARAMS_AND_STRUCTS_CUH
#define RESULT_MOVEMENT_PARAMS_AND_STRUCTS_CUH

//needed for general parameters and structs
#include "genParamsAndStructs.cuh"

//define where the resulting movement is by default
#define DEFAULT_LOCATION_RESULTING_MOVEMENT DATA_ON_DEVICE

//enum to define whether manhattan or euclidean distance is used for the difference between two movements
typedef enum
{
	MANHATTAN_DIST_FOR_MOVE_DIFF,
	EUCLIDEAN_DIST_FOR_MOVE_DIFF
} moveDiffCalc;

//define the int and float values indicating an error
const float ERROR_FLOAT_VAL = -1.0f;
const int ERROR_INT_VAL = -1;

//define the default movement value
const float DEFAULT_RESULT_MOVEMENT_VAL = 0.0f;

//define the location of the resulting movement given the disparity image
const dataLocation LOCATION_MOVEMENT_LOAD_FROM_DISP_IMAGE = DATA_ON_HOST;

//define the movement direction given a disparity image
const posMoveDirection MOVEMENT_DIR_GIVEN_DISP_IMAGE = POS_X_MOTION_LEFT_POS_Y_MOTION_UP;

//define the movement location and direction given a flo image
const dataLocation LOCATION_MOVEMENT_LOAD_FROM_FLO_IMAGE = DATA_ON_HOST;
const posMoveDirection MOVEMENT_DIR_GIVEN_FLO_IMAGE = POS_X_MOTION_LEFT_POS_Y_MOTION_UP;

//define the threshold at which a movement is considered "unknown"
const float THRESHOLD_MOVEMENT_UNKNOWN = 10000.0f;

//set the color representing "unknown" motion
const unsigned char COLOR_UNKNOWN_MOTION = 0;

#endif //RESULT_MOVEMENT_PARAMS_AND_STRUCTS_CUH
