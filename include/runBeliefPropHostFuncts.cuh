//runBeliefPropHostFuncts.cuh
//Scott Grauer-Gray
//June 25, 2009
//Defines the host functions used to run belief propagation

#ifndef RUN_BELIEF_PROP_HOST_FUNCTS_CUH
#define RUN_BELIEF_PROP_HOST_FUNCTS_CUH

//needed for the current belief propagation parameters and structures
#include "beliefPropParamsAndStructs.cuh"

//needed for standard input/output
#include <stdio.h>


extern "C"
{
	

	//set the current BP settings in the host in constant memory on the device
	void setCurrBeliefPropParamsInConstMem(currBeliefPropParams& currentBeliefPropParams);

	//run the kernel function to round the set of values on the device
	void roundDeviceVals(float* inputDeviceVals, float* outputDeviceVals, int widthVals, int heightVals);

	//run the given number of iterations of BP at the current level using the given message values in global device memory
	void runBPAtCurrentLevel(
		checkerboardMessagesDeviceStruct messageValsCheckerboard1, checkerboardMessagesDeviceStruct messageValsCheckerboard2,
		float* dataCostDeviceCheckerboard1, float* dataCostDeviceCheckerboard2,
		currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1, currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
		int numIterationsAtLevel, int widthLevelActualIntegerSize, int heightLevelActualIntegerSize, 
		size_t numBytesDataAndMessageSetInCheckerboardAtLevel, size_t paramsOffsetLevel = 0);

	//run the given number of iterations of BP at the current level using the given message values in global device memory...no textures...
	void runBPAtCurrentLevelNoTextures(
		checkerboardMessagesDeviceStruct messageValsCheckerboard1, checkerboardMessagesDeviceStruct messageValsCheckerboard2,
		float* dataCostDeviceCheckerboard1, float* dataCostDeviceCheckerboard2,
		currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1, currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
		int numIterationsAtLevel, int widthLevelActualIntegerSize, int heightLevelActualIntegerSize, 
		size_t numBytesDataAndMessageSetInCheckerboardAtLevel, size_t paramsOffsetLevel);

	//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
	//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
	//in the next level down
	//need two different "sets" of message values to avoid read-write conflicts
	//this step requires adjustments from the estimated value at one level of the hierarchy to the next level down depending on the
	//estimated value at each pixel
	void copyMessageValuesToNextLevelDown(
		checkerboardMessagesDeviceStruct messageDeviceCheckerboard1CopyFrom, 
		checkerboardMessagesDeviceStruct messageDeviceCheckerboard2CopyFrom, 
		currentStartMoveParamsPixelAtLevel* paramsPrevLevelDeviceCheckerboard1, currentStartMoveParamsPixelAtLevel* paramsPrevLevelDeviceCheckerboard2,
		currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboard1, currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboard2,
		float* estimatedMovementXCheckerboard1Device, float* estimatedMovementYCheckerboard1Device,
		float* estimatedMovementXCheckerboard2Device, float* estimatedMovementYCheckerboard2Device,
		checkerboardMessagesDeviceStruct messageDeviceCheckerboard1CopyTo,
		checkerboardMessagesDeviceStruct messageDeviceCheckerboard2CopyTo,
		int widthLevelActualIntegerSize, int heightLevelActualIntegerSize,
		int widthNextLevelActualIntegerSize, int heightNextLevelActualIntegerSize,
		size_t numBytesDataAndMessageSetInCheckerboardAtLevel);

	//initialize the data cost at each pixel at the current level
	//assume that motion parameters for current level are set
	void initializeDataCostsCurrentLevel(float* image1PixelsDevice, float* image2PixelsDevice,
										float* dataCostDeviceCheckerboard1, float* dataCostDeviceCheckerboard2,
										currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1,
										currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
										currBeliefPropParams& currentBeliefPropParams,
										int widthCurrentLevel, int heightCurrentLevel);

	//initialize the data cost at each pixel at the current level
	//assume that motion parameters for current level are set
	//no textures are used for the parameters here...
	void initializeDataCostsCurrentLevelNoTexParams(float* image1PixelsDevice, float* image2PixelsDevice,
							float* dataCostDeviceCheckerboard1, float* dataCostDeviceCheckerboard2,
							currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1,
							currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
							currBeliefPropParams& currentBeliefPropParams,
							int widthCurrentLevel, int heightCurrentLevel, int paramsOffset);

	//initialize the estimated movement costs and add it to the data costs in the overall computation...
	void initEstMovementInDataCosts(float* estMovesXDir, float* estMovesYDir, 
				float* dataCostDeviceCheckerboard1, float* dataCostDeviceCheckerboard2,
				currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1,
				currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
				currBeliefPropParams& currentBeliefPropParams,
				int widthOfCheckerboard, int heightOfCheckerboard,
				int widthCurrentLevel, int heightCurrentLevel,
				int numBytesDataAndMessagesCurrLevel);

	//initialize the estimated movement costs and add it to the data costs in the overall computation...without using textures...
	void initEstMovementInDataCostsNoTextures(float* estMovesXDir, float* estMovesYDir, 
				float* dataCostDeviceCheckerboard1, float* dataCostDeviceCheckerboard2,
				currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1,
				currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
				currBeliefPropParams& currentBeliefPropParams,
				int widthOfCheckerboard, int heightOfCheckerboard,
				int widthCurrentLevel, int heightCurrentLevel,
				int numBytesDataAndMessagesCurrLevel,
				int paramOffset);


	//initialize the parameters defining the starting movement at the current level on the device
	void initializeStartMovementParams(currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1, currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
													int widthLevel, int heightLevel, currentStartMoveParamsPixelAtLevel startMoveParams);


	//initialize the message values with no previous message values...all message values are set to DEFAULT_INITIAL_MESSAGE_VAL
	void initializeMessageVals(checkerboardMessagesDeviceStruct messageDeviceCheckerboard1,
										checkerboardMessagesDeviceStruct messageDeviceCheckerboard2,
										int widthCurrentLevel, int heightCurrentLevel);

	//run the kernel function to retrieve the "best" estimate for each pixel at the current level on the device
	//this is used to initialize each level of the hierarchy as to what the motion range is "allowed" at the next level
	void retrieveBestMotionEstLevel(float* movementXBetweenImagesCheckerboard1Device, float* movementYBetweenImagesCheckerboard1Device,
									float* movementXBetweenImagesCheckerboard2Device, float* movementYBetweenImagesCheckerboard2Device,
									checkerboardMessagesDeviceStruct messageDeviceCheckerboard1,
									checkerboardMessagesDeviceStruct messageDeviceCheckerboard2,
									float* dataCostsDeviceCheckerboard1, float* dataCostsDeviceCheckerboard2,
									currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1,
									currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
									size_t numBytesDataAndMessageSetInCheckerboardAtLevel, int widthLevel, int heightLevel);

	//host function to set the initial movement parameters at the "top" level for each individual pixel and also the parameters for "all" pixels
	//at the top level
	void initializeParamsAndMovementOnDevice(currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1,
													currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
													int widthLevel, int heightLevel, 
													currBeliefPropParams& currentBeliefPropParams);

	//host function to set the initial movement parameters at the "top" level for each individual pixel in each level of the computation hierarchy and also the parameters for "all" pixels
	//at the top level
	void initializeParamsAndMovementOnDeviceHierarchImp(currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1, currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
								paramOffsetsSizes* hierarchParams, int numHierarchLevels, currBeliefPropParams& currentBeliefPropParams);

	//adjust the parameters for the currBeliefPropParams and update in constant memory
	void adjustMovementAllPixParams(currBeliefPropParams& currentBeliefPropParams, int widthCurrentLevel, int heightCurrentLevel);

	//host function that takes the x and y movements in each checkerboard and combines them
	void combineXYCheckerboardMovements(float* movementXBetweenImagesCheckerboard1Device, float* movementYBetweenImagesCheckerboard1Device,
										float* movementXBetweenImagesCheckerboard2Device, float* movementYBetweenImagesCheckerboard2Device,
										float* movementXFromImage1To2Device, float* movementYFromImage1To2Device, int widthLevel, int heightLevel);


	//run the belief propagation algorithm on CUDA for motion estimation where the x and the y movements are computed where the two portions of the "checkerboard" are split
	//the BP implementation is run such that the range is adjusted in each level in the hierarchy
	void runBeliefPropMotionEstimationCUDA(float* image1PixelsDevice, float* image2PixelsDevice, float* movementXFromImage1To2Device, float* movementYFromImage1To2Device, currBeliefPropParams currentBeliefPropParams);

	//run the belief propagation algorithm on CUDA for motion estimation where the x and the y movements are computed where the two portions of the "checkerboard" are split
	//the BP implementation is run such that the range is adjusted in each level in the hierarchy
	//the width and height of each level are the same
	void runBeliefPropMotionEstimationCUDAUseConstLevelSize(float* image1PixelsDevice, float* image2PixelsDevice, float* movementXFromImage1To2Device, float* movementYFromImage1To2Device, currBeliefPropParams currentBeliefPropParams);

	//run the belief propagation algorithm on CUDA for motion estimation where the x and the y movements are computed where the two portions of the "checkerboard" are split
	//the BP implementation is run such that the range is adjusted in each level in the hierarchy
	//the width and height of each level are the same
	//this function uses `estimated movement', likely computed from the movement in the previous set of images...
	void runBeliefPropMotionEstimationCUDAUseConstLevelSizeUseEstMovement(float* image1PixelsDevice, float* image2PixelsDevice, float* prevAndCurrMovementXFromImage1To2Device, float* prevAndCurrMovementYFromImage1To2Device, currBeliefPropParams currentBeliefPropParams);

	//run the belief propagation algorithm on CUDA for motion estimation where the x and the y movements are computed where the two portions of the "checkerboard" are split
	//the BP implementation is run such that the range is adjusted in each level in the hierarchy
	//the width and height of each level are the same
	//prevAndCurrMovementXFromImage1To2Device and prevAndCurrMovementYFromImage1To2Device represents both the `input movement' from the previous iteration and the `current movement' in the output of the current iteration...
	//run using a hierarchy within each level to reduce the number of iterations...
	void runBeliefPropMotionEstimationCUDAUseConstLevelSizeHierarchInLevUseEstMovement(float* image1PixelsDevice, float* image2PixelsDevice, float* prevAndCurrMovementXFromImage1To2Device, float* prevAndCurrMovementYFromImage1To2Device, currBeliefPropParams currentBeliefPropParams);

	//function to retrieve the total number of 'values' needed for the parameter hierarchy given the starting `checkerboard' width and height and the number of levels
	int numValsParamHierarchy(int imageCheckerboardWidth, int imageCheckerboardHeight, int numLevelsInHierarch);

	//function to retrieve the `offsets' and `size' at each level of the `parameters hierarchy'
	paramOffsetsSizes* getOffsetsParamsHierarch(int imageCheckerboardWidth, int imageCheckerboardHeight, int numLevelsInHierarch);

	//function to retrieve the message values at the `next' level given the values at the previous level...
	void getMessValsNextLevel(checkerboardMessagesDeviceStruct messagesDevicePrevCurrCheckerboard1,
				checkerboardMessagesDeviceStruct messagesDevicePrevCurrCheckerboard2,
				checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboard1,
				checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboard2,
				int widthPrevLevel, int heightPrevLevel, int widthNextLevel, int heightNextLevel,
				currentStartMoveParamsPixelAtLevel* paramsDeviceCheckerboard1,
				currentStartMoveParamsPixelAtLevel* paramsDeviceCheckerboard2, 
				size_t paramsOffsetPrevLev, size_t paramsOffsetCurrLevel);

	//function for retrieving the starting parameters for the 'next level' at every pixel in the bottom level for each `checkerboard'...
	void retStartParamsNextLevel(currentStartMoveParamsPixelAtLevel* startParamsCheckerboard1, currentStartMoveParamsPixelAtLevel* startParamsCheckerboard2, float* estMovementXCheckerboard1,
				float* estMovementYCheckerboard1, float* estMovementXCheckerboard2, float* estMovementYCheckerboard2, int widthCheckerboard, int heightCheckerboard);

	//initialize the combined data and estimated movement costs at each pixel...assuming method is sampling invarient...
	void initializeDataAndEstMoveCostsCurrentLevel(float* image1PixelsDevice, float* image2PixelsDevice,
														float* dataCostDeviceCheckerboard1, float* dataCostDeviceCheckerboard2,
														currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1,
														currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
														currBeliefPropParams& currentBeliefPropParams,
														int widthCurrentLevel, int heightCurrentLevel, float* estMoveXDevice, float* estMoveYDevice);

	//run the belief propagation algorithm on CUDA for motion estimation where the x and the y movements are computed where the two portions of the "checkerboard" are split
	//the BP implementation is run such that the range is adjusted in each level in the hierarchy
	//the width and height of each level are the same
	//the estimated movement between the two images is given as an input parameter...
	void runBeliefPropMotionEstimationCUDAUseConstLevelSizeGivenEstMovement(float* image1PixelsDevice, float* image2PixelsDevice, float* currMovementXFromImage1To2Device, float* currMovementYFromImage1To2Device, currBeliefPropParams currentBeliefPropParams, float* expectedMovementXDirHost, float* expectedMovementYDirHost);

}

#endif //RUN_BELIEF_PROP_HOST_FUNCTS_CUH
