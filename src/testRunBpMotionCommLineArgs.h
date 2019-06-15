//testRunBpMotionCommLineArgs.h
//Scott Grauer-Gray
//September 7, 2010
//Function to run cuda belief propagation motion using command line arguments

#ifndef TEST_RUN_BP_MOTION_COMM_LINE_ARGS_H
#define TEST_RUN_BP_MOTION_COMM_LINE_ARGS_H

#include "resultMovement.h"
#include "runBeliefProp.h"
#include "runSmoothImage.h"
#include "bpImage.h"
#include "inputParams.h"


const dataLocation DATA_LOCATION_OUTPUT_SMOOTHED_IMAGE_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = DATA_ON_DEVICE;
const currentDataType DATA_TYPE_OUTPUT_SMOOTHED_IMAGE_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = FLOAT_DATA;

const discontCostType DISC_TYPE_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = USE_MANHATTAN_DIST_FOR_DISCONT;
const posMoveDirection DIRECTION_POS_MOTION_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = POS_X_MOTION_RIGHT_POS_Y_MOTION_DOWN;

const int borderDiffCompWithCommLineArgs = 0;

const float thresh1MoveDiffWithCommLineArgs = 0.10f;
const float thresh2MoveDiffWithCommLineArgs = 0.25f;
const float thresh3MoveDiffWithCommLineArgs = 0.50f;
const float thresh4MoveDiffWithCommLineArgs = 1.0f;
const float thresh5MoveDiffWithCommLineArgs = 1.5f;
const float thresh6MoveDiffWithCommLineArgs = 2.0f;

//define the method of movement difference to use
const moveDiffCalc MOVE_DIFF_CALC_TO_USE_MOTION_WITH_COMM_LINE_ARGS = EUCLIDEAN_DIST_FOR_MOVE_DIFF;


//function to run cuda belief propagation motion given a string of input arguments
void testRunBpMotionWithCommLineArgs(char* inArgs[])
{
	//retrieve the input parameters from the command line arguments
	inputParams* currInputParams = new inputParams(inArgs);

	//print the input parameters...
	currInputParams->printInputParams();
		
	//retrieve each of the inputs from the input params class...
	char* IMAGE_1_FILE_PATH_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = currInputParams->getImage1FilePath();
	char* IMAGE_2_FILE_PATH_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = currInputParams->getImage2FilePath();

	char* MOTION_IMAGE_FILE_SAVE_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = currInputParams->getMotionImageFileSave();

	int WIDTH_INPUT_IMAGE_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = currInputParams->getWidthInputImages();
	int HEIGHT_INPUT_IMAGE_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = currInputParams->getHeightInputImages();

	int NUM_BELIEF_PROP_LEVELS_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = currInputParams->getNumBeliefPropLevels();
	int NUM_BELIEF_PROP_ITERS_PER_LEVEL_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = currInputParams->getNumItersInBeliefPropLevel();

	float DATA_COST_CAP_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = currInputParams->getDataCostCap();
	float SMOOTHNESS_COST_CAP_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = currInputParams->getSmoothnessCostCap();
	float DATA_COST_WEIGHT_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = currInputParams->getDataCostWeight();

	float CURRENT_MOVE_INCREMENT_X_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = currInputParams->getCurrMoveIncrX();
	float CURRENT_MOVE_INCREMENT_Y_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = currInputParams->getCurrMoveIncrY();
	float SAMPLING_LEVEL_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = currInputParams->getSamplingLevel();

	float X_MOVE_MIN_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = currInputParams->getXMoveMin();
	float X_MOVE_MAX_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = currInputParams->getXMoveMax();

	float Y_MOVE_MIN_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = currInputParams->getYMoveMin();
	float Y_MOVE_MAX_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = currInputParams->getYMoveMax();

	float RANGE_DISPLAY_MOTION_TEST_WITH_COMM_LINE_ARGS = currInputParams->getMotionDisplayRange();

	//file path of the ground truth movement in flo format
	char* GROUND_TRUTH_MOVEMENT_FLO_FORMAT_WITH_COMM_LINE_ARGS = currInputParams->getGroundTruthMoveFlo();

	//file path of the saved movement data in flo format
	char* FILE_PATH_SAVED_MOVEMENT_DATA_FLO_FORMAT_WITH_COMM_LINE_ARGS = currInputParams->getMotionFloFileSave();

	float PROP_CHANGE_MOVE_NEXT_LEVEL_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = currInputParams->getPropChangeMove();

	float SMOOTHING_SIGMA_BP_MOTION_TEST_WITH_COMM_LINE_ARGS = currInputParams->getSmoothingSigma();

	int NUM_LEVELS_WITHIN_LEV_PYR_HIERARCH = currInputParams->getNumLevelsWithinLevPyrHierarch();

	//read the ground truth flow
	resultMovement* groundTruthFlow = new resultMovement(GROUND_TRUTH_MOVEMENT_FLO_FORMAT_WITH_COMM_LINE_ARGS);
	int numValsMoveKnown = groundTruthFlow->numValsKnownMovement();

	bpImage* image1 = new bpImage(IMAGE_1_FILE_PATH_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, WIDTH_INPUT_IMAGE_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, HEIGHT_INPUT_IMAGE_BP_MOTION_TEST_WITH_COMM_LINE_ARGS);
	bpImage* image2 = new bpImage(IMAGE_2_FILE_PATH_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, WIDTH_INPUT_IMAGE_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, HEIGHT_INPUT_IMAGE_BP_MOTION_TEST_WITH_COMM_LINE_ARGS);

	//smooth both the reference and test images
	runSmoothImage* runSmoothImageFunct = new runSmoothImage(SMOOTHING_SIGMA_BP_MOTION_TEST_WITH_COMM_LINE_ARGS,
		DATA_LOCATION_OUTPUT_SMOOTHED_IMAGE_BP_MOTION_TEST_WITH_COMM_LINE_ARGS,
		DATA_TYPE_OUTPUT_SMOOTHED_IMAGE_BP_MOTION_TEST_WITH_COMM_LINE_ARGS);
	bpImage* smoothedImage1 = runSmoothImageFunct->operator()(image1);
	bpImage* smoothedImage2 = runSmoothImageFunct->operator()(image2);

	//clean up the original reference and test images
	delete image1;
	delete image2;

	//set the run belief propagation function object
	runBeliefProp* currRunBeliefPropObject =
		new runBeliefProp(NUM_BELIEF_PROP_LEVELS_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, NUM_BELIEF_PROP_ITERS_PER_LEVEL_BP_MOTION_TEST_WITH_COMM_LINE_ARGS,
			DATA_COST_CAP_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, SMOOTHNESS_COST_CAP_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, DATA_COST_WEIGHT_BP_MOTION_TEST_WITH_COMM_LINE_ARGS,
			CURRENT_MOVE_INCREMENT_X_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, CURRENT_MOVE_INCREMENT_Y_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, SAMPLING_LEVEL_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, X_MOVE_MIN_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, X_MOVE_MAX_BP_MOTION_TEST_WITH_COMM_LINE_ARGS,
			Y_MOVE_MIN_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, Y_MOVE_MAX_BP_MOTION_TEST_WITH_COMM_LINE_ARGS,
			PROP_CHANGE_MOVE_NEXT_LEVEL_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, DISC_TYPE_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, DIRECTION_POS_MOTION_BP_MOTION_TEST_WITH_COMM_LINE_ARGS,
			NUM_LEVELS_WITHIN_LEV_PYR_HIERARCH);

	resultMovement* resultingMovementWithCommLineArgs = currRunBeliefPropObject->operator ()(smoothedImage1, smoothedImage2);

	resultingMovementWithCommLineArgs->transferMovementDataDeviceToHost();

	resultingMovementWithCommLineArgs->writeMovementDataInFloFormat(FILE_PATH_SAVED_MOVEMENT_DATA_FLO_FORMAT_WITH_COMM_LINE_ARGS);

	if (resultingMovementWithCommLineArgs != NULL)
	{
		//save the resulting movement
		resultingMovementWithCommLineArgs->saveMovementImage(RANGE_DISPLAY_MOTION_TEST_WITH_COMM_LINE_ARGS, MOTION_IMAGE_FILE_SAVE_BP_MOTION_TEST_WITH_COMM_LINE_ARGS);
	}

	//compare the computed movement with the ground truth
	int numPixDiffThresh1 = groundTruthFlow->retrieveNumPixMoveDiff(resultingMovementWithCommLineArgs, borderDiffCompWithCommLineArgs, MOVE_DIFF_CALC_TO_USE_MOTION_WITH_COMM_LINE_ARGS, thresh1MoveDiffWithCommLineArgs);
	int numPixDiffThresh2 = groundTruthFlow->retrieveNumPixMoveDiff(resultingMovementWithCommLineArgs, borderDiffCompWithCommLineArgs, MOVE_DIFF_CALC_TO_USE_MOTION_WITH_COMM_LINE_ARGS, thresh2MoveDiffWithCommLineArgs);
	int numPixDiffThresh3 = groundTruthFlow->retrieveNumPixMoveDiff(resultingMovementWithCommLineArgs, borderDiffCompWithCommLineArgs, MOVE_DIFF_CALC_TO_USE_MOTION_WITH_COMM_LINE_ARGS, thresh3MoveDiffWithCommLineArgs);
	int numPixDiffThresh4 = groundTruthFlow->retrieveNumPixMoveDiff(resultingMovementWithCommLineArgs, borderDiffCompWithCommLineArgs, MOVE_DIFF_CALC_TO_USE_MOTION_WITH_COMM_LINE_ARGS, thresh4MoveDiffWithCommLineArgs);
	int numPixDiffThresh5 = groundTruthFlow->retrieveNumPixMoveDiff(resultingMovementWithCommLineArgs, borderDiffCompWithCommLineArgs, MOVE_DIFF_CALC_TO_USE_MOTION_WITH_COMM_LINE_ARGS, thresh5MoveDiffWithCommLineArgs);
	int numPixDiffThresh6 = groundTruthFlow->retrieveNumPixMoveDiff(resultingMovementWithCommLineArgs, borderDiffCompWithCommLineArgs, MOVE_DIFF_CALC_TO_USE_MOTION_WITH_COMM_LINE_ARGS, thresh6MoveDiffWithCommLineArgs);

	//print the percent of "bad pixels"
	printf("Percent 'bad pixels'\n");
	printf("Threshold 1: %f\n", static_cast<float>(numPixDiffThresh1) / (static_cast<float>(numValsMoveKnown)));
	printf("Threshold 2: %f\n", static_cast<float>(numPixDiffThresh2) / (static_cast<float>(numValsMoveKnown)));
	printf("Threshold 3: %f\n", static_cast<float>(numPixDiffThresh3) / (static_cast<float>(numValsMoveKnown)));
	printf("Threshold 4: %f\n", static_cast<float>(numPixDiffThresh4) / (static_cast<float>(numValsMoveKnown)));
	printf("Threshold 5: %f\n", static_cast<float>(numPixDiffThresh5) / (static_cast<float>(numValsMoveKnown)));
	printf("Threshold 6: %f\n\n", static_cast<float>(numPixDiffThresh6) / (static_cast<float>(numValsMoveKnown)));

	//retrieve the total "movement diff"
	float totalMoveDiff = groundTruthFlow->retrieveMovementDiff(resultingMovementWithCommLineArgs, borderDiffCompWithCommLineArgs, MOVE_DIFF_CALC_TO_USE_MOTION_WITH_COMM_LINE_ARGS);

	printf("Total movement diff: %f\n", totalMoveDiff);
	printf("Average movement diff: %f\n", totalMoveDiff / (static_cast<float>(numValsMoveKnown)));

	//clean up the smoothed image 1 and image 2
	delete smoothedImage1;
	delete smoothedImage2;
}

#endif //TEST_RUN_BP_MOTION_COMM_LINE_ARGS_H

