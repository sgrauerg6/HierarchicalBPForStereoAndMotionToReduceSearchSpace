//testRunBpMotionMultImages.h
//Scott Grauer-Gray
//September 27, 2010
//Function for running bp motion on multiple images with the goal of retrieving accurate motion for a set of images...

#ifndef TEST_RUN_BP_MOTION_MULT_IMAGES_H
#define TEST_RUN_BP_MOTION_MULT_IMAGES_H

#include "resultMovement.h"
#include "runBeliefProp.h"
#include "runSmoothImage.h"
#include "bpImage.h"

//needed for running belief propagation on multiple images...
#include "runBeliefPropMultImages.h"

//needed for the input parameters for multiple images...
#include "inputParamsMultInputImages.h"

const dataLocation DATA_LOCATION_OUTPUT_SMOOTHED_IMAGE_BP_MOTION_MULT_IMAGES_TEST = DATA_ON_DEVICE;
const currentDataType DATA_TYPE_OUTPUT_SMOOTHED_IMAGE_BP_MOTION_MULT_IMAGES_TEST = FLOAT_DATA;

const discontCostType DISC_TYPE_BP_MOTION_MULT_IMAGES_TEST = USE_MANHATTAN_DIST_FOR_DISCONT; //USE_APPROX_EUCLIDEAN_DIST_FOR_DISCONT;// 
const posMoveDirection DIRECTION_POS_MOTION_BP_MOTION_MULT_IMAGES_TEST = POS_X_MOTION_RIGHT_POS_Y_MOTION_DOWN;

const int borderDiffComp_multImagesTest = 0;

const float thresh1MoveDiff_multImagesTest = 0.10f;
const float thresh2MoveDiff_multImagesTest = 0.25f;
const float thresh3MoveDiff_multImagesTest = 0.50f;
const float thresh4MoveDiff_multImagesTest = 1.0f;
const float thresh5MoveDiff_multImagesTest = 1.5f;
const float thresh6MoveDiff_multImagesTest = 2.0f;

//define the method of movement difference to use
const moveDiffCalc MOVE_DIFF_CALC_TO_USE_MOTION_MULT_IMAGES_TEST = EUCLIDEAN_DIST_FOR_MOVE_DIFF;

//constant for the number of characters in the image file path
#define NUM_CHARS_IMAGE_FILE_PATH 250

char* itoa(int val, int base){
	
	static char buf[32] = {0};
	
	int i = 30;
	
	for(; val && i ; --i, val /= base)
	
		buf[i] = "0123456789abcdef"[val % base];
	
	return &buf[i+1];
	
}

void testRunBpMotionMultImages(char** inArgs)
{
	//retrieve the input parameters for multiple images from the command line arguments
	inputParamsMultInputImages* currInputParams = new inputParamsMultInputImages(inArgs);

	//print the input parameters...
	currInputParams->printInputParams();

	//retrieve the `base' of the file name...
	char* FILE_PATH_BASE_BP_MOTION_MULT_IMAGES_TEST = currInputParams->getImageFilePathBase();

	//starting `number' of image in series...
	int START_NUM_IMAGE = currInputParams->getNumStartImage();

	//ending `number' of image in series...
	int ENDING_NUM_IMAGE = currInputParams->getNumEndImage();
		
	//extension of the file name
	char* IMAGE_FILE_NAME_EXTENSION = currInputParams->getFileExtension();

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

	//retrieve the estimated motion info
	float EST_MOVE_CAP_BP_MOTION_MULT_IMAGES_TEST = currInputParams->getEstMovementCap();
	float EST_MOVE_WEIGHT_BP_MOTION_MULT_IMAGES_TEST = currInputParams->getEstMovementWeight();

	//read the ground truth flow
	resultMovement* groundTruthFlow = new resultMovement(GROUND_TRUTH_MOVEMENT_FLO_FORMAT_WITH_COMM_LINE_ARGS);

	//retrieve the number of values where the movement is "known"
	int numValsMoveKnown = groundTruthFlow->numValsKnownMovement();

	//allocate space for the input images as an array
	bpImage** inImageSet = new bpImage*[(ENDING_NUM_IMAGE-START_NUM_IMAGE) + 1];

	
	//retrieve and smooth the sequence of images
	for (int numImage = START_NUM_IMAGE; numImage <= ENDING_NUM_IMAGE; numImage++)
	{
		//declare a string for the image file path...
		char currImageFilePath[NUM_CHARS_IMAGE_FILE_PATH];

		sprintf ( currImageFilePath,"%s%d%s", FILE_PATH_BASE_BP_MOTION_MULT_IMAGES_TEST, numImage, IMAGE_FILE_NAME_EXTENSION);

		//use concatenation to put together each component of the image file path...
		//strcat(currImageFilePath, FILE_PATH_BASE_BP_MOTION_MULT_IMAGES_TEST);
		//strcat(currImageFilePath, inToStringString);
		//strcat(currImageFilePath, IMAGE_FILE_NAME_EXTENSION);

		//print the image file path
		printf("Image File Path: %s\n", currImageFilePath);

		//retrieve each of the images in the directory...
		bpImage* currImage = new bpImage(currImageFilePath, WIDTH_INPUT_IMAGE_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, HEIGHT_INPUT_IMAGE_BP_MOTION_TEST_WITH_COMM_LINE_ARGS);

		//smooth both the reference and test images
		runSmoothImage* runSmoothImageFunct = new runSmoothImage(SMOOTHING_SIGMA_BP_MOTION_TEST_WITH_COMM_LINE_ARGS,
			DATA_LOCATION_OUTPUT_SMOOTHED_IMAGE_BP_MOTION_MULT_IMAGES_TEST,
			DATA_TYPE_OUTPUT_SMOOTHED_IMAGE_BP_MOTION_MULT_IMAGES_TEST);
		bpImage* currSmoothedImage = runSmoothImageFunct->operator()(currImage);

		//now store the `smoothed image' in the image set with the current number...
		inImageSet[(numImage-START_NUM_IMAGE)] = currSmoothedImage;
	
		//clean up the `current image'
		delete currImage;
	}

	

	//set the run belief propagation function object
	runBeliefPropMultImages* currRunBeliefPropObject =
		new runBeliefPropMultImages(NUM_BELIEF_PROP_LEVELS_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, NUM_BELIEF_PROP_ITERS_PER_LEVEL_BP_MOTION_TEST_WITH_COMM_LINE_ARGS,
			DATA_COST_CAP_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, SMOOTHNESS_COST_CAP_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, DATA_COST_WEIGHT_BP_MOTION_TEST_WITH_COMM_LINE_ARGS,
			CURRENT_MOVE_INCREMENT_X_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, CURRENT_MOVE_INCREMENT_Y_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, SAMPLING_LEVEL_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, X_MOVE_MIN_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, X_MOVE_MAX_BP_MOTION_TEST_WITH_COMM_LINE_ARGS,
			Y_MOVE_MIN_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, Y_MOVE_MAX_BP_MOTION_TEST_WITH_COMM_LINE_ARGS,
			PROP_CHANGE_MOVE_NEXT_LEVEL_BP_MOTION_TEST_WITH_COMM_LINE_ARGS, DISC_TYPE_BP_MOTION_MULT_IMAGES_TEST, DIRECTION_POS_MOTION_BP_MOTION_TEST_WITH_COMM_LINE_ARGS,
			EST_MOVE_CAP_BP_MOTION_MULT_IMAGES_TEST, EST_MOVE_WEIGHT_BP_MOTION_MULT_IMAGES_TEST);

	resultMovement* resultingMovement = currRunBeliefPropObject->operator ()(inImageSet, (ENDING_NUM_IMAGE-START_NUM_IMAGE) + 1, START_NUM_IMAGE, ENDING_NUM_IMAGE);

	resultingMovement->transferMovementDataDeviceToHost();

	resultingMovement->writeMovementDataInFloFormat(FILE_PATH_SAVED_MOVEMENT_DATA_FLO_FORMAT_WITH_COMM_LINE_ARGS);


	if (resultingMovement != NULL)
	{
		//save the resulting movement
		resultingMovement->saveMovementImage(RANGE_DISPLAY_MOTION_TEST_WITH_COMM_LINE_ARGS, MOTION_IMAGE_FILE_SAVE_BP_MOTION_TEST_WITH_COMM_LINE_ARGS);
	}

	//compare the computed movement with the ground truth
	int numPixDiffThresh1 = groundTruthFlow->retrieveNumPixMoveDiff(resultingMovement, borderDiffComp_multImagesTest, MOVE_DIFF_CALC_TO_USE_MOTION_MULT_IMAGES_TEST, thresh1MoveDiff_multImagesTest);
	int numPixDiffThresh2 = groundTruthFlow->retrieveNumPixMoveDiff(resultingMovement, borderDiffComp_multImagesTest, MOVE_DIFF_CALC_TO_USE_MOTION_MULT_IMAGES_TEST, thresh2MoveDiff_multImagesTest);
	int numPixDiffThresh3 = groundTruthFlow->retrieveNumPixMoveDiff(resultingMovement, borderDiffComp_multImagesTest, MOVE_DIFF_CALC_TO_USE_MOTION_MULT_IMAGES_TEST, thresh3MoveDiff_multImagesTest);
	int numPixDiffThresh4 = groundTruthFlow->retrieveNumPixMoveDiff(resultingMovement, borderDiffComp_multImagesTest, MOVE_DIFF_CALC_TO_USE_MOTION_MULT_IMAGES_TEST, thresh4MoveDiff_multImagesTest);
	int numPixDiffThresh5 = groundTruthFlow->retrieveNumPixMoveDiff(resultingMovement, borderDiffComp_multImagesTest, MOVE_DIFF_CALC_TO_USE_MOTION_MULT_IMAGES_TEST, thresh5MoveDiff_multImagesTest);
	int numPixDiffThresh6 = groundTruthFlow->retrieveNumPixMoveDiff(resultingMovement, borderDiffComp_multImagesTest, MOVE_DIFF_CALC_TO_USE_MOTION_MULT_IMAGES_TEST, thresh6MoveDiff_multImagesTest);

	//print the percent of "bad pixels"
	printf("Percent 'bad pixels'\n");
	printf("Threshold 1: %f\n", static_cast<float>(numPixDiffThresh1) / (static_cast<float>(numValsMoveKnown)));
	printf("Threshold 2: %f\n", static_cast<float>(numPixDiffThresh2) / (static_cast<float>(numValsMoveKnown)));
	printf("Threshold 3: %f\n", static_cast<float>(numPixDiffThresh3) / (static_cast<float>(numValsMoveKnown)));
	printf("Threshold 4: %f\n", static_cast<float>(numPixDiffThresh4) / (static_cast<float>(numValsMoveKnown)));
	printf("Threshold 5: %f\n", static_cast<float>(numPixDiffThresh5) / (static_cast<float>(numValsMoveKnown)));
	printf("Threshold 6: %f\n\n", static_cast<float>(numPixDiffThresh6) / (static_cast<float>(numValsMoveKnown)));

	//retrieve the total "movement diff"
	float totalMoveDiff = groundTruthFlow->retrieveMovementDiff(resultingMovement, borderDiffComp_multImagesTest, MOVE_DIFF_CALC_TO_USE_MOTION_MULT_IMAGES_TEST);

	printf("Total movement diff: %f\n", totalMoveDiff);
	printf("Average movement diff: %f\n", totalMoveDiff / (static_cast<float>(numValsMoveKnown)));


	//clean up the smoothed images
	for (int currInImageNum = 0; currInImageNum < ((ENDING_NUM_IMAGE-START_NUM_IMAGE) + 1); currInImageNum++)
	{
		delete inImageSet[currInImageNum];
	}
}

#endif //TEST_RUN_BP_MOTION_MULT_IMAGES_H
