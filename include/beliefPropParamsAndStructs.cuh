//beliefPropParamsAndStructs.cuh
//Scott Grauer-Gray
//June 24, 2009
//Define the parameters and structs, including the default parameters, for running belief propagation

#ifndef BELIEF_PROP_PARAMS_AND_STRUCTS_CUH
#define BELIEF_PROP_PARAMS_AND_STRUCTS_CUH

//needed for the inputs from the python script...
#include "genBeliefPropMotionConsts.h"

//needed for the general parameters and structs
#include "genParamsAndStructs.cuh"

//needed for the "float2" and possibly other CUDA utility stuff
#include <cutil.h>
#include <vector_types.h>

//structure for the parameter offsets and size of each level
typedef struct
{
	int offset;
	int levelWidthCheckerboard;
	int levelHeightCheckerboard;
} paramOffsetsSizes;


//enum to determine whether or not to use the `pyramid hierarchy'...
typedef enum
{
	pyrHierarch, //represents a `pyramid hierarchy' where each 'higher' level is half the width/height of the previous
	constHierarch, //represents a `constant hierarchy' where each level has the same width/height...
	constHierarchPyrWithin //represents a `constant hierarchy' where a pyramid is used within...
} currMethodProcessingHierarch;

//set the default method used for processing
#define DEFAULT_PROCESSING_METHOD constHierarch

//enum to determine whether or not to use the `previous movement'
typedef enum
{
	yesUsePrevMovement,
	noDontUsePrevMovement
} usePrevMovement;

//set the default method used as to whether or not to use previous movement...
#define DEFAULT_USE_PREV_MOVEMENT_METHOD noDontUsePrevMovement

//define the default location of the output movement
#define DEFAULT_LOCATION_OUTPUT_MOVEMENT_DATA DATA_ON_DEVICE

//define the default starting message value
#define DEFAULT_INITIAL_MESSAGE_VAL 0.0f

//defie the width and height of the current thread block
#define BLOCK_SIZE_WIDTH_BP 32
#define BLOCK_SIZE_HEIGHT_BP 4

//define the default float array output value in case the value cannot be calculated
#define DEFAULT_OUTPUT_FLOAT_ARRAY_VAL 0.0f

//define "infinity"...AKA a very large value
#define INF_BP INF_VALUE_FLOAT

//define the possible movement range in the x and y directions
#define TOTAL_POSS_MOVE_RANGE_X GEN_MAX_NUM_MOVES_X_DIR
#define TOTAL_POSS_MOVE_RANGE_Y GEN_MAX_NUM_MOVES_Y_DIR

//define the default data cost
#define DEFAULT_DATA_COST_VALUE 0.0f

//define the default movement cost
#define DEFAULT_EST_MOVEMENT_COST_VALUE 0.0f

//define the constant that there is 'no movement data' at a particular point
#define NO_MOVEMENT_DATA -999.0f

//defines the current starting x and y movement parameters at the given level for the current pixel
typedef float2 currentStartMoveParamsPixelAtLevel;


//define a structure with pointers to each of the four message values
//within the checkerboard
typedef struct
{
	float* messageUDevice;
	float* messageDDevice;
	float* messageLDevice;
	float* messageRDevice;
} checkerboardMessagesDeviceStruct;

//define a structure with the data costs, message values, and motion estimations for
//bp on each "checkerboard"
typedef struct
{
	currentStartMoveParamsPixelAtLevel* paramsCurrentLevel;
	checkerboardMessagesDeviceStruct checkerboardMessVals;
	float* dataCostsVals;
	float* estimatedXMovement;
	float* estimatedYMovement;
} bpCurrentValsCheckerboard;

//enum defining the discontinuity cost type
typedef enum
{
	USE_APPROX_EUCLIDEAN_DIST_FOR_DISCONT,
	USE_EXACT_EUCLIDEAN_DIST_FOR_DISCONT,
	USE_MANHATTAN_DIST_FOR_DISCONT,
	USE_MANHATTEN_DIST_USING_BRUTE_FORCE
} discontCostType;

//define the structure to store the current
//belief propagation parameters on the device
//(most likely in constant memory)
typedef struct
{
	//define the number of BP levels and iterations per level
	int numBpLevels;
	int numBpIterations;
	
	//define the current level number
	int currentLevelNum;
	
	//define the width and height of the current set of images
	int widthImages;
	int heightImages;
	
	//define the width and height of the current level of the hierarchy
	int widthCurrentLevel;
	int heightCurrentLevel;
	
	//define the weighing and cost caps
	//for the data and smoothness costs
	float dataCostCap;
	float smoothnessCostCap;
	float dataCostWeight;

	//define the current increment of possible movement in the x and y directions
	//float currentMoveIncrement;
	float currentMoveIncrementX;
	float currentMoveIncrementY;

	//define the change in the movement between levels
	float propChangeMoveNextLevel;

	//define the level of sampling for the output movement
	//1.0 means each input pixel sample; less than 1.0 gives more samples, greater than 1.0 gives less samples
	float samplingLevel;
	
	//define the starting possible move at each pixel
	float startPossMoveX;
	float startPossMoveY;

	//define the total number of possible movements in the x and y directions
	int totalNumMovesXDir;
	int totalNumMovesYDir;
	
	//define the current discontinuity cost type
	discontCostType currDiscCostType;
	
	//define the current direction for positive movement
	posMoveDirection directionPosMovement;
	
	//define the motion increment at the bottom level
	//float motionIncBotLev;
	float motionIncBotLevX;
	float motionIncBotLevY;

	//define the weight and cost cap of 'estimated movement'
	float estMovementCostWeight;
	float estMovementCostCap;

	//define the number of levels in the `pyramid hierarchy' used for computation...
	int numPyrHierarchLevels;

	//define whether or not to use the `previous movement'
	usePrevMovement usePrevMovementSetting;

} currBeliefPropParams;



//defines the default output float value
#define DEFAULT_OUTPUT_FLOAT_ARRAY_VAL 0.0f

//define whether or not to use a checkerboard border
#define USE_CHECKERBOARD_BORDER 1
#define DONT_USE_CHECKERBOARD_BORDER 0

#define USE_CHECKERBOARD_BORDER_SETTING DONT_USE_CHECKERBOARD_BORDER

#define SIZE_CHECKERBOARD_BORDER 1

//define the data cost and message values in the checkerboard border region
#define DATA_COST_VAL_CHECKERBOARD_BORDER 0.0f
#define MESSAGE_VAL_CHECKERBOARD_BORDER 0.0f
#define X_MOVEMENT_VAL_CHECKERBOARD_BORDER 0.0f
#define Y_MOVEMENT_VAL_CHECKERBOARD_BORDER 0.0f

//enum defining the current checkerboard portion
typedef enum
{
	CHECKERBOARD_PART_1_ENUM,
	CHECKERBOARD_PART_2_ENUM
} checkerboardPortionEnum;

//define the method used to compute the data costs
#define ADD_COSTS_METHOD 0
#define SAMP_INVARIENT_SAMPLING 1
#define DATA_COST_METHOD SAMP_INVARIENT_SAMPLING

//define whether to use the minimum message value or the interpolated value when passing the message value
#define USE_MIN_MESSAGE_VAL 0
#define USE_INTERPOLATED_MESSAGE_VAL 1
#define USE_MAX_MESSAGE_VAL 2
#define RESET_MESSAGE_VAL_TO_DEFAULT 3
#define USE_IMMEDIATE_MESSAGE_VAL 4

#define MESSAGE_VAL_PASS_SCHEME USE_MIN_MESSAGE_VAL

//define the manner to retrieve the message value in "next" level
#define COPY_MESS_VAL_PASS_SCHEME USE_MIN_MESSAGE_VAL 

//define the possible methods of retrieving the movement range of the next "level"
#define USE_MID_VAL_RET_MOVE_RANGE 0
#define ROUND_MID_VAL_UP_RET_MOVE_RANGE 1
#define ROUND_MID_VAL_DOWN_RET_MOVE_RANGE 2

//define whether or not to adjust the range at the boundaries such that the estimated value is part of the "range"
#define DONT_ADJUST_RANGE_BOUND_FOR_EST_VAL 0
#define ADJUST_RANGE_BOUND_FOR_EST_VAL 1

//define the methods of retrieving the movement of the "next" range
#define RET_MOVE_RANGE_METHOD USE_MID_VAL_RET_MOVE_RANGE
#define RANGE_BOUNDARIES_SETTING ADJUST_RANGE_BOUND_FOR_EST_VAL

//define whether or not to clamp the edges in the range of movements when retrieving the data cost
#define CLAMP_EDGE_MOVES_DATA_COST 0
#define DONT_CLAMP_EDGE_MOVES_DATA_COST 1

#define CLAMP_EDGE_MOVES_DATA_COST_SETTING CLAMP_EDGE_MOVES_DATA_COST

//define whether or not the round the resulting movement values
#define ROUND_RESULTING_MOVE_VALS 0
#define DONT_ROUND_RESULTING_MOVE_VALS 1
#define ROUND_RESULTING_MOVE_VALS_SETTING DONT_ROUND_RESULTING_MOVE_VALS

#define DO_USE_MOVEMENT_DATA_FROM_PREV_IMAGE_SET 0
#define DONT_USE_MOVEMENT_DATA_FROM_PREV_IMAGE_SET 1

#define USE_MOVEMENT_DATA_FROM_PREV_IMAGE_SETTING DONT_USE_MOVEMENT_DATA_FROM_PREV_IMAGE_SET

//define the defaults for the 'movement data'
#define DEFAULT_MOVEMENT_COST_CAP 30.0f
#define DEFAULT_MOVEMENT_COST_WEIGHT 0.10f


//default value for the number of pyramid levels in the `pyramid hierarchy'
#define DEFAULT_NUM_PYR_LEVELS_PYRAMID_HIERARCHY -999


#endif //BELIEF_PROP_PARAMS_AND_STRUCTS_CUH
