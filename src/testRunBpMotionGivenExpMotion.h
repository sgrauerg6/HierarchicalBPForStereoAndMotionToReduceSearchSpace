//testRunBpMotionGivenExpMotion.h
//Scott Grauer-Gray
//November 29, 2010
//Function for running bp motion on images with the given expected motion...

#ifndef TEST_RUN_BP_MOTION_GIVEN_EXP_MOTION_H
#define TEST_RUN_BP_MOTION_GIVEN_EXP_MOTION_H

#include "resultMovement.h"
#include "runBeliefProp.h"
#include "runSmoothImage.h"
#include "bpImage.h"

//needed for running belief propagation on multiple images...
#include "runBeliefPropMultImages.h"

//needed for the input parameters for multiple images...
#include "inputParamsGivenExpMotion.h"

const dataLocation DATA_LOCATION_OUTPUT_SMOOTHED_IMAGE_BP_MOTION_GIVEN_EXP_MOTION_TEST = DATA_ON_DEVICE;
const currentDataType DATA_TYPE_OUTPUT_SMOOTHED_IMAGE_BP_MOTION_GIVEN_EXP_MOTION_TEST = FLOAT_DATA;

const discontCostType DISC_TYPE_BP_MOTION_GIVEN_EXP_MOTION_TEST = USE_MANHATTAN_DIST_FOR_DISCONT; //USE_APPROX_EUCLIDEAN_DIST_FOR_DISCONT;// 
const posMoveDirection DIRECTION_POS_MOTION_BP_MOTION_GIVEN_EXP_MOTION_TEST = POS_X_MOTION_RIGHT_POS_Y_MOTION_DOWN;

const int borderDiffComp_givenExpectedMotionTest = 0;

const float thresh1MoveDiff_givenExpectedMotionTest = 0.10f;
const float thresh2MoveDiff_givenExpectedMotionTest = 0.25f;
const float thresh3MoveDiff_givenExpectedMotionTest = 0.50f;
const float thresh4MoveDiff_givenExpectedMotionTest = 1.0f;
const float thresh5MoveDiff_givenExpectedMotionTest = 1.5f;
const float thresh6MoveDiff_givenExpectedMotionTest = 2.0f;

//define the method of movement difference to use
const moveDiffCalc MOVE_DIFF_CALC_TO_USE_MOTION_GIVEN_EXP_MOTION_TEST = EUCLIDEAN_DIST_FOR_MOVE_DIFF;

//constant for the number of characters in the image file path
#define NUM_CHARS_IMAGE_FILE_PATH 250

char* itoaGivenExpectedMotion(int val, int base){
	
	static char buf[32] = {0};
	
	int i = 30;
	
	for(; val && i ; --i, val /= base)
	
		buf[i] = "0123456789abcdef"[val % base];
	
	return &buf[i+1];
	
}

void testRunBpMotionGivenExpectedMotion(char** inArgs)
{
	//retrieve the input parameters for multiple images from the command line arguments
	inputParamsGivenExpMotion* currInputParams = new inputParamsGivenExpMotion(inArgs);

	//print the input parameters...
	currInputParams->printInputParams();

	//extension of the file name
	char* IMAGE_1_FILE_NAME = currInputParams->getImage1FilePath();
	char* IMAGE_2_FILE_NAME = currInputParams->getImage2FilePath();

	char* MOTION_IMAGE_FILE_SAVE_BP_MOTION_TEST_GIVEN_EXP_MOTION = currInputParams->getMotionImageFileSave();

	char* EXPECTED_MOTION_FLO_FILE_BP_MOTION_TEST_GIVEN_EXP_MOTION = currInputParams->getExpectedMotionFloFilePath();

	int WIDTH_INPUT_IMAGE_BP_MOTION_TEST_GIVEN_EXP_MOTION = currInputParams->getWidthInputImages();
	int HEIGHT_INPUT_IMAGE_BP_MOTION_TEST_GIVEN_EXP_MOTION = currInputParams->getHeightInputImages();

	int NUM_BELIEF_PROP_LEVELS_BP_MOTION_TEST_GIVEN_EXP_MOTION = currInputParams->getNumBeliefPropLevels();
	int NUM_BELIEF_PROP_ITERS_PER_LEVEL_BP_MOTION_TEST_GIVEN_EXP_MOTION = currInputParams->getNumItersInBeliefPropLevel();

	float DATA_COST_CAP_BP_MOTION_TEST_GIVEN_EXP_MOTION = currInputParams->getDataCostCap();
	float SMOOTHNESS_COST_CAP_BP_MOTION_TEST_GIVEN_EXP_MOTION = currInputParams->getSmoothnessCostCap();
	float DATA_COST_WEIGHT_BP_MOTION_TEST_GIVEN_EXP_MOTION = currInputParams->getDataCostWeight();

	float CURRENT_MOVE_INCREMENT_X_BP_MOTION_TEST_GIVEN_EXP_MOTION = currInputParams->getCurrMoveIncrX();
	float CURRENT_MOVE_INCREMENT_Y_BP_MOTION_TEST_GIVEN_EXP_MOTION = currInputParams->getCurrMoveIncrY();
	float SAMPLING_LEVEL_BP_MOTION_TEST_GIVEN_EXP_MOTION = currInputParams->getSamplingLevel();

	float X_MOVE_MIN_BP_MOTION_TEST_GIVEN_EXP_MOTION = currInputParams->getXMoveMin();
	float X_MOVE_MAX_BP_MOTION_TEST_GIVEN_EXP_MOTION = currInputParams->getXMoveMax();

	float Y_MOVE_MIN_BP_MOTION_TEST_GIVEN_EXP_MOTION = currInputParams->getYMoveMin();
	float Y_MOVE_MAX_BP_MOTION_TEST_GIVEN_EXP_MOTION = currInputParams->getYMoveMax();

	float RANGE_DISPLAY_MOTION_TEST_GIVEN_EXP_MOTION = currInputParams->getMotionDisplayRange();

	//file path of the ground truth movement in flo format
	char* GROUND_TRUTH_MOVEMENT_FLO_FORMAT_GIVEN_EXP_MOTION = currInputParams->getGroundTruthMoveFlo();

	//file path of the saved movement data in flo format
	char* FILE_PATH_SAVED_MOVEMENT_DATA_FLO_FORMAT_GIVEN_EXP_MOTION = currInputParams->getMotionFloFileSave();

	float PROP_CHANGE_MOVE_NEXT_LEVEL_BP_MOTION_TEST_GIVEN_EXP_MOTION = currInputParams->getPropChangeMove();

	float SMOOTHING_SIGMA_BP_MOTION_TEST_GIVEN_EXP_MOTION = currInputParams->getSmoothingSigma();

	//retrieve the estimated motion info
	float EST_MOVE_CAP_BP_MOTION_GIVEN_EXP_MOTION_TEST = currInputParams->getEstMovementCap();
	float EST_MOVE_WEIGHT_BP_MOTION_GIVEN_EXP_MOTION_TEST = currInputParams->getEstMovementWeight();

	//read the ground truth flow
	resultMovement* groundTruthFlow = new resultMovement(GROUND_TRUTH_MOVEMENT_FLO_FORMAT_GIVEN_EXP_MOTION);

	//retrieve the number of values where the movement is "known"
	int numValsMoveKnown = groundTruthFlow->numValsKnownMovement();
	
	//smooth the input images...
	//retrieve each of the images in the directory...
	bpImage* image1 = new bpImage(IMAGE_1_FILE_NAME, WIDTH_INPUT_IMAGE_BP_MOTION_TEST_GIVEN_EXP_MOTION, HEIGHT_INPUT_IMAGE_BP_MOTION_TEST_GIVEN_EXP_MOTION);
	bpImage* image2 = new bpImage(IMAGE_2_FILE_NAME, WIDTH_INPUT_IMAGE_BP_MOTION_TEST_GIVEN_EXP_MOTION, HEIGHT_INPUT_IMAGE_BP_MOTION_TEST_GIVEN_EXP_MOTION);

	//smooth both the reference and test images
	runSmoothImage* runSmoothImageFunct = new runSmoothImage(SMOOTHING_SIGMA_BP_MOTION_TEST_GIVEN_EXP_MOTION,
		DATA_LOCATION_OUTPUT_SMOOTHED_IMAGE_BP_MOTION_GIVEN_EXP_MOTION_TEST,
		DATA_TYPE_OUTPUT_SMOOTHED_IMAGE_BP_MOTION_GIVEN_EXP_MOTION_TEST);
		
	bpImage* smoothedImage1 = runSmoothImageFunct->operator()(image1);
	bpImage* smoothedImage2 = runSmoothImageFunct->operator()(image2);
		
	//clean up the original input images
	delete image1;
	delete image2;

	//retrieve the estimated motion from the flo file...
	resultMovement* expectedMovement = new resultMovement(EXPECTED_MOTION_FLO_FILE_BP_MOTION_TEST_GIVEN_EXP_MOTION);
	

	//set the run belief propagation function object
	runBeliefPropMultImages* currRunBeliefPropObject =
		new runBeliefPropMultImages(NUM_BELIEF_PROP_LEVELS_BP_MOTION_TEST_GIVEN_EXP_MOTION, NUM_BELIEF_PROP_ITERS_PER_LEVEL_BP_MOTION_TEST_GIVEN_EXP_MOTION,
			DATA_COST_CAP_BP_MOTION_TEST_GIVEN_EXP_MOTION, SMOOTHNESS_COST_CAP_BP_MOTION_TEST_GIVEN_EXP_MOTION, DATA_COST_WEIGHT_BP_MOTION_TEST_GIVEN_EXP_MOTION,
			CURRENT_MOVE_INCREMENT_X_BP_MOTION_TEST_GIVEN_EXP_MOTION, CURRENT_MOVE_INCREMENT_Y_BP_MOTION_TEST_GIVEN_EXP_MOTION, SAMPLING_LEVEL_BP_MOTION_TEST_GIVEN_EXP_MOTION, X_MOVE_MIN_BP_MOTION_TEST_GIVEN_EXP_MOTION, X_MOVE_MAX_BP_MOTION_TEST_GIVEN_EXP_MOTION,
			Y_MOVE_MIN_BP_MOTION_TEST_GIVEN_EXP_MOTION, Y_MOVE_MAX_BP_MOTION_TEST_GIVEN_EXP_MOTION,
			PROP_CHANGE_MOVE_NEXT_LEVEL_BP_MOTION_TEST_GIVEN_EXP_MOTION, DISC_TYPE_BP_MOTION_GIVEN_EXP_MOTION_TEST, DIRECTION_POS_MOTION_BP_MOTION_GIVEN_EXP_MOTION_TEST,
			EST_MOVE_CAP_BP_MOTION_GIVEN_EXP_MOTION_TEST, EST_MOVE_WEIGHT_BP_MOTION_GIVEN_EXP_MOTION_TEST);

	resultMovement* resultingMovement = currRunBeliefPropObject->runBeliefPropWithEstMovement(smoothedImage1, smoothedImage2, expectedMovement->getXMovement(), expectedMovement->getYMovement());

	resultingMovement->transferMovementDataDeviceToHost();

	resultingMovement->writeMovementDataInFloFormat(FILE_PATH_SAVED_MOVEMENT_DATA_FLO_FORMAT_GIVEN_EXP_MOTION);


	if (resultingMovement != NULL)
	{
		//save the resulting movement
		resultingMovement->saveMovementImage(RANGE_DISPLAY_MOTION_TEST_GIVEN_EXP_MOTION, MOTION_IMAGE_FILE_SAVE_BP_MOTION_TEST_GIVEN_EXP_MOTION);
	}

	//compare the computed movement with the ground truth
	int numPixDiffThresh1 = groundTruthFlow->retrieveNumPixMoveDiff(resultingMovement, borderDiffComp_givenExpectedMotionTest, MOVE_DIFF_CALC_TO_USE_MOTION_GIVEN_EXP_MOTION_TEST, thresh1MoveDiff_givenExpectedMotionTest);
	int numPixDiffThresh2 = groundTruthFlow->retrieveNumPixMoveDiff(resultingMovement, borderDiffComp_givenExpectedMotionTest, MOVE_DIFF_CALC_TO_USE_MOTION_GIVEN_EXP_MOTION_TEST, thresh2MoveDiff_givenExpectedMotionTest);
	int numPixDiffThresh3 = groundTruthFlow->retrieveNumPixMoveDiff(resultingMovement, borderDiffComp_givenExpectedMotionTest, MOVE_DIFF_CALC_TO_USE_MOTION_GIVEN_EXP_MOTION_TEST, thresh3MoveDiff_givenExpectedMotionTest);
	int numPixDiffThresh4 = groundTruthFlow->retrieveNumPixMoveDiff(resultingMovement, borderDiffComp_givenExpectedMotionTest, MOVE_DIFF_CALC_TO_USE_MOTION_GIVEN_EXP_MOTION_TEST, thresh4MoveDiff_givenExpectedMotionTest);
	int numPixDiffThresh5 = groundTruthFlow->retrieveNumPixMoveDiff(resultingMovement, borderDiffComp_givenExpectedMotionTest, MOVE_DIFF_CALC_TO_USE_MOTION_GIVEN_EXP_MOTION_TEST, thresh5MoveDiff_givenExpectedMotionTest);
	int numPixDiffThresh6 = groundTruthFlow->retrieveNumPixMoveDiff(resultingMovement, borderDiffComp_givenExpectedMotionTest, MOVE_DIFF_CALC_TO_USE_MOTION_GIVEN_EXP_MOTION_TEST, thresh6MoveDiff_givenExpectedMotionTest);

	//print the percent of "bad pixels"
	printf("Percent 'bad pixels'\n");
	printf("Threshold 1: %f\n", static_cast<float>(numPixDiffThresh1) / (static_cast<float>(numValsMoveKnown)));
	printf("Threshold 2: %f\n", static_cast<float>(numPixDiffThresh2) / (static_cast<float>(numValsMoveKnown)));
	printf("Threshold 3: %f\n", static_cast<float>(numPixDiffThresh3) / (static_cast<float>(numValsMoveKnown)));
	printf("Threshold 4: %f\n", static_cast<float>(numPixDiffThresh4) / (static_cast<float>(numValsMoveKnown)));
	printf("Threshold 5: %f\n", static_cast<float>(numPixDiffThresh5) / (static_cast<float>(numValsMoveKnown)));
	printf("Threshold 6: %f\n\n", static_cast<float>(numPixDiffThresh6) / (static_cast<float>(numValsMoveKnown)));

	//retrieve the total "movement diff"
	float totalMoveDiff = groundTruthFlow->retrieveMovementDiff(resultingMovement, borderDiffComp_givenExpectedMotionTest, MOVE_DIFF_CALC_TO_USE_MOTION_GIVEN_EXP_MOTION_TEST);

	printf("Total movement diff: %f\n", totalMoveDiff);
	printf("Average movement diff: %f\n", totalMoveDiff / (static_cast<float>(numValsMoveKnown)));

	//clean up the expected movement
	delete expectedMovement;

	//clean up the ground truth movement...
	delete groundTruthFlow;

	//clean up the smoothed images
	delete smoothedImage1;
	delete smoothedImage2;
}

#endif //TEST_RUN_BP_MOTION_GIVEN_EXP_MOTION_H
