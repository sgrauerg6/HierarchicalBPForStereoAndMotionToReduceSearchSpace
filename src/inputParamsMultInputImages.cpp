//inputParamsMultInputImages.cpp
//Scott Grauer-Gray
//September 29, 2010
//Definitions file for the input parameters for multiple input images...

//needed as the header file to the function
#include "inputParamsMultInputImages.h"

//constructor for the input parameters...
inputParamsMultInputImages::inputParamsMultInputImages(char* inParams[])
{
	imagesFileBase = inParams[IMAGES_FILE_PATH_BASE_INDEX_MULT_INPUT_IMAGES];

	startImageNum = atoi(inParams[START_IMAGE_NUM_INDEX_MULT_INPUT_IMAGES]);
	endImageNum = atoi(inParams[END_IMAGE_NUM_INDEX_MULT_INPUT_IMAGES]);

	imageFilePathExtension = inParams[FILE_EXTENSION_INDEX_MULT_INPUT_IMAGES];

	motionImageFileSave = inParams[MOTION_IMAGE_FILE_SAVE_INDEX_MULT_INPUT_IMAGES];
	
	widthInputImages = atoi(inParams[WIDTH_INPUT_IMAGES_INDEX_MULT_INPUT_IMAGES]);
	heightInputImages = atoi(inParams[HEIGHT_INPUT_IMAGES_INDEX_MULT_INPUT_IMAGES]);

	numBeliefPropLevels = atoi(inParams[NUM_BELIEF_PROP_LEVELS_INDEX_MULT_INPUT_IMAGES]);
	numItersInBeliefPropLevel = atoi(inParams[NUM_ITERS_PER_BELIEF_PROP_LEVEL_INDEX_MULT_INPUT_IMAGES]);

	dataCostCap = atof(inParams[DATA_COST_CAP_INDEX_MULT_INPUT_IMAGES]);
	smoothnessCostCap = atof(inParams[SMOOTHNESS_COST_CAP_INDEX_MULT_INPUT_IMAGES]);
	dataCostWeight = atof(inParams[DATA_COST_WEIGHT_INDEX_MULT_INPUT_IMAGES]);

	currMoveIncrX = atof(inParams[CURR_MOVE_INCR_X_INDEX_MULT_INPUT_IMAGES]);
	currMoveIncrY = atof(inParams[CURR_MOVE_INCR_Y_INDEX_MULT_INPUT_IMAGES]);

	samplingLevel = atof(inParams[SAMPLING_LEVEL_INDEX_MULT_INPUT_IMAGES]);

	xMoveMin = atof(inParams[X_MOVE_MIN_INDEX_MULT_INPUT_IMAGES]);
	xMoveMax = atof(inParams[X_MOVE_MAX_INDEX_MULT_INPUT_IMAGES]);

	yMoveMin = atof(inParams[Y_MOVE_MIN_INDEX_MULT_INPUT_IMAGES]);
	yMoveMax = atof(inParams[Y_MOVE_MAX_INDEX_MULT_INPUT_IMAGES]);

	motionDisplayRange = atof(inParams[MOTION_DISPLAY_RANGE_INDEX_MULT_INPUT_IMAGES]);

	groundTruthMoveFlo = inParams[GROUND_TRUTH_MOVE_FLO_INDEX_MULT_INPUT_IMAGES];
	motionFloFileSave = inParams[MOTION_MOVE_FLO_INDEX_MULT_INPUT_IMAGES];
	propChangeMove = atof(inParams[PROP_CHANGE_MOVE_INDEX_MULT_INPUT_IMAGES]);

	smoothingSigma = atof(inParams[SMOOTHING_SIGMA_INDEX_MULT_INPUT_IMAGES]);

	//retrieve the movement info...
	estMoveCap = atof(inParams[EST_MOVE_CAP_INDEX_MULT_INPUT_IMAGES]);
	estMoveWeight = atof(inParams[EST_MOVE_WEIGHT_INDEX_MULT_INPUT_IMAGES]);

	numLevelsWithinLevPyrHierarch = atoi(inParams[NUM_LEVEL_WITHIN_LEV_PYR_HIERARCH_INDEX_MULT_INPUT_IMAGES]);
}

//function to print out each of the parameters
void inputParamsMultInputImages::printInputParams()
{
	cout << "Image file path base: " << imagesFileBase << endl;
	cout << "Start image num: " << startImageNum << endl;
	cout << "End image num: " << endImageNum << endl;
	cout << "Image file path extension: " << imageFilePathExtension << endl;
	cout << "motionImageFileSave: " << motionImageFileSave << endl;
	cout << "widthInputImages: " << widthInputImages << endl;
	cout << "heightInputImages: " << heightInputImages << endl;
	cout << "numBeliefPropLevels: " << numBeliefPropLevels << endl;
	cout << "numItersInBeliefPropLevel: " << numItersInBeliefPropLevel << endl;
	cout << "dataCostCap: " << dataCostCap << endl;
	cout << "smoothnessCostCap: " << smoothnessCostCap << endl;
	cout << "dataCostWeight: " << dataCostWeight << endl;
	cout << "currMoveIncrX: " << currMoveIncrX << endl;
	cout << "currMoveIncrY: " << currMoveIncrY << endl;
	cout << "samplingLevel: " << samplingLevel << endl;
	cout << "xMoveMin: " << xMoveMin << endl;
	cout << "xMoveMax: " << xMoveMax << endl;
	cout << "yMoveMin: " << yMoveMin << endl;
	cout << "yMoveMax: " << yMoveMax << endl;
	cout << "motionDisplayRange: " << motionDisplayRange << endl;
	cout << "groundTruthMoveFlo: " << groundTruthMoveFlo << endl;
	cout << "motionFloFileSave: " << motionFloFileSave << endl;
	cout << "propChangeMove: " << propChangeMove << endl;
	cout << "smoothingSigma: " << smoothingSigma << endl;
	cout << "estMoveCap: " << estMoveCap << endl;
	cout << "estMoveWeight: " << estMoveWeight << endl;
	cout << "numLevelsWithinLevPyrHierarch: " << numLevelsWithinLevPyrHierarch << endl;
}
