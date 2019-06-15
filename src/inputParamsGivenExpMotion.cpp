//inputParamsGivenExpMotion.cpp
//Scott Grauer-Gray
//November 29, 2010
//Definitions file for the input parameters given the expected motion...

#include "inputParamsGivenExpMotion.h"
#include "inputParamsGivenExpMotionConsts.h"

//constructor for the input parameters...
inputParamsGivenExpMotion::inputParamsGivenExpMotion(char* inParams[])
{
	image1FilePath = inParams[IMAGE_1_FILE_GIVEN_EXP_MOTION_INDEX];
	image2FilePath = inParams[IMAGE_2_FILE_GIVEN_EXP_MOTION_INDEX];

	motionImageFileSave = inParams[MOTION_IMAGE_FILE_SAVE_GIVEN_EXP_MOTION_INDEX];
	expectedMotionFloFilePath = inParams[EXPECTED_MOTION_FLO_FILE_GIVEN_EXP_MOTION_INDEX];
	
	widthInputImages = atoi(inParams[WIDTH_INPUT_IMAGES_GIVEN_EXP_MOTION_INDEX]);
	heightInputImages = atoi(inParams[HEIGHT_INPUT_IMAGES_GIVEN_EXP_MOTION_INDEX]);

	numBeliefPropLevels = atoi(inParams[NUM_BELIEF_PROP_LEVELS_GIVEN_EXP_MOTION_INDEX]);
	numItersInBeliefPropLevel = atoi(inParams[NUM_ITERS_PER_BELIEF_PROP_LEVEL_GIVEN_EXP_MOTION_INDEX]);

	dataCostCap = atof(inParams[DATA_COST_CAP_GIVEN_EXP_MOTION_INDEX]);
	smoothnessCostCap = atof(inParams[SMOOTHNESS_COST_CAP_GIVEN_EXP_MOTION_INDEX]);
	dataCostWeight = atof(inParams[DATA_COST_WEIGHT_GIVEN_EXP_MOTION_INDEX]);

	currMoveIncrX = atof(inParams[CURR_MOVE_INCR_X_GIVEN_EXP_MOTION_INDEX]);
	currMoveIncrY = atof(inParams[CURR_MOVE_INCR_Y_GIVEN_EXP_MOTION_INDEX]);
	samplingLevel = atof(inParams[SAMPLING_LEVEL_GIVEN_EXP_MOTION_INDEX]);

	xMoveMin = atof(inParams[X_MOVE_MIN_GIVEN_EXP_MOTION_INDEX]);
	xMoveMax = atof(inParams[X_MOVE_MAX_GIVEN_EXP_MOTION_INDEX]);

	yMoveMin = atof(inParams[Y_MOVE_MIN_GIVEN_EXP_MOTION_INDEX]);
	yMoveMax = atof(inParams[Y_MOVE_MAX_GIVEN_EXP_MOTION_INDEX]);

	motionDisplayRange = atof(inParams[MOTION_DISPLAY_RANGE_GIVEN_EXP_MOTION_INDEX]);

	groundTruthMoveFlo = inParams[GROUND_TRUTH_MOVE_FLO_GIVEN_EXP_MOTION_INDEX];
	motionFloFileSave = inParams[MOTION_MOVE_FLO_GIVEN_EXP_MOTION_INDEX];
	propChangeMove = atof(inParams[PROP_CHANGE_MOVE_GIVEN_EXP_MOTION_INDEX]);

	smoothingSigma = atof(inParams[SMOOTHING_SIGMA_GIVEN_EXP_MOTION_INDEX]);
	estMoveCap = atof(inParams[EST_MOVE_CAP_GIVEN_EXP_MOTION_INDEX]);
	estMoveWeight = atof(inParams[EST_MOVE_WEIGHT_GIVEN_EXP_MOTION_INDEX]);

	numLevelsWithinLevPyrHierarch = atoi(inParams[NUM_LEVEL_WITHIN_LEV_PYR_HIERARCH_GIVEN_EXP_MOTION_INDEX]);
}

//function to print out each of the parameters
void inputParamsGivenExpMotion::printInputParams()
{
	cout << "Image 1 file: " << image1FilePath << endl;
	cout << "Image 2 file: " << image2FilePath << endl;
	cout << "motionImageFileSave: " << motionImageFileSave << endl;
	cout << "expectedMotionFloFilePath: " << expectedMotionFloFilePath << endl;
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
