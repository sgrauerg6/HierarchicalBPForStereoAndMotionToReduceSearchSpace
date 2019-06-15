//kernelRunBeliefProp.cuh
//Scott Grauer-Gray
//June 25, 2009
//Declares the functions for the kernel used for running belief propagation

#ifndef KERNEL_RUN_BELIEF_PROP_CUH
#define KERNEL_RUN_BELIEF_PROP_CUH

//needed for structures and #defines
#include "beliefPropParamsAndStructs.cuh"

//needed for CUDA utility functions
//#include <cutil.h>

//constant memory holding the current movement parameters
__device__ __constant__ currBeliefPropParams currentBeliefPropParamsConstMem;

//declare the textures that the CUDA array of the images are bound to
texture<float, 2, cudaReadModeElementType> image1PixelsTexture;
texture<float, 2, cudaReadModeElementType> image2PixelsTexture;

//declare the textures that the current starting movement parameters are bound to
texture<float2, 1, cudaReadModeElementType> currentPixParamsTexCurrentCheckerboard;
texture<float2, 1, cudaReadModeElementType> currentPixParamsTexNeighCheckerboard;

//declare the textures for the current message values
texture<float, 1, cudaReadModeElementType> messageUTexCurrReadCheckerboard;
texture<float, 1, cudaReadModeElementType> messageDTexCurrReadCheckerboard;
texture<float, 1, cudaReadModeElementType> messageLTexCurrReadCheckerboard;
texture<float, 1, cudaReadModeElementType> messageRTexCurrReadCheckerboard;

//declare the textures for the data cost values
texture<float, 1, cudaReadModeElementType> dataCostsCurrCheckerboard;

//declare the texture for the current estimated movement in the x and y directions
texture<float, 1, cudaReadModeElementType> estimatedXMovementTexDeviceCurrentCheckerboard;
texture<float, 1, cudaReadModeElementType> estimatedYMovementTexDeviceCurrentCheckerboard;


//equation to retrieve the index of the message
#define RETRIEVE_INDEX_IN_DATA_OR_MESSAGE_ARRAY_EQUATION(xVal, yVal, widthDataLevel, heightDataLevel, totalNumMoveVals, currentMoveNum, offset) \
		(yVal*widthDataLevel*totalNumMoveVals + widthDataLevel*currentMoveNum + xVal + offset)

//retrieve the current 1-D index value of the given point at the given movement in the data cost and message data
__device__ int retrieveIndexInDataAndMessage(int xVal, int yVal, int widthDataLevel, int heightDataLevel, int currentMoveNum, int totalNumMoveVals, int offsetData = 0);

//retrieve the index value of the current point in a 2-D array where the index is across by column and then by row
__device__ int retrieveIndexCurrentPixel2DGrid(int xVal, int yVal, int width, int height, int offsetData = 0);

//checks if the current point is within the image bounds
__device__ bool withinImageBounds(int xVal, int yVal, int width, int height);

__device__ float getEuclideanDist(float xVal1, float yVal1, float xVal2, float yVal2);

__device__ float getManhattanDist(float xVal1, float yVal1, float xVal2, float yVal2);

//function to set the 'estimated movement' for each pixel in the 'next' image...
__global__ void setEstMoveAtPixels(float* resultantXMovePrevIm, float* resultantYMovePrevIm, float* expectedMovementX,
				float* expectedMovementY, int widthImageAtLevel, int heightImageAtLevel);

//function to add two sets of input data...
__global__ void addInputData(float* inOutData, float* inData, int widthImageAtLevel, int heightImageAtLevel);


//function to retrieve the 'deviation from estimated movement' cost
__global__ void getDevFromEstMoveCosts(float* inputEstMoveXVals, float* inputEstMoveYVals,
					float* outputEstValCosts, int widthImageAtLevel, int heightImageAtLevel,
					checkerboardPortionEnum checkerboardPart);

//function to retrieve the 'deviation from estimated movement' cost without using textures...
__global__ void getDevFromEstMoveCostsNoTextures(float* inputEstMoveXVals, float* inputEstMoveYVals,
					float* outputEstValCosts, int widthImageAtLevel, int heightImageAtLevel,
					checkerboardPortionEnum checkerboardPart, currentStartMoveParamsPixelAtLevel* currentPixParamsCurrentCheckerboard);

//computes the "distance" between two movements
__device__ float computeDistanceBetweenMovements(float movement1Xcomp, float movement1Ycomp, float movement2Xcomp, float movement2Ycomp);

//global function to set params defining the starting possible x and y motionto a particular level
__global__ void setParamsToGivenParamKernel(currentStartMoveParamsPixelAtLevel currentPixelParams, currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1,
											currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2, int widthLevel, int heightLevel);

//initialize the message values at each pixel of the current level to the default value for each checkerboard
__global__ void initMessageValsToDefaultCurrCheckerboard(checkerboardMessagesDeviceStruct messageValsDeviceCheckerboard1,
														 checkerboardMessagesDeviceStruct messageValsDeviceCheckerboard2,
														int widthLevel, int heightLevel);

//initialize the "data cost" for each possible movement at a particular level for the desired set of motions within one of the
//checkerboards
//the image data is stored in the CUDA arrays image1PixelsTextureAtLevel and 
//image2PixelsTextureAtLevel
//this method adds together all the results in a given range according to the current level
//in the "bottom" level, it is a single pixel, then the width/height are doubled at each level
//assuming that the input is full images
__global__ void initialDataCostsAtCurrentLevelAddCostsCheckerboard(float* dataCostsCurrentLevelCurrentCheckerboard,
																	int widthImageAtLevel, int heightImageAtLevel,
																	checkerboardPortionEnum checkerboardPart);
																	
__global__ void initialDataCostsAtCurrentLevelAddCostsCheckerboardSampInvarient(float* dataCostsCurrentLevelCurrentCheckerboard,
																	int widthImageAtLevel, int heightImageAtLevel,
																	checkerboardPortionEnum checkerboardPart);

//initialize the "data cost" for each possible movement at a particular level for the desired set of motions within one of the
//checkerboards
//the image data is stored in the CUDA arrays image1PixelsTextureAtLevel and 
//image2PixelsTextureAtLevel
//this method adds together all the results in a given range according to the current level
//in the "bottom" level, it is a single pixel, then the width/height are doubled at each level
//assuming that the input is full images
//no textures used for parameters, though they still are for the image data...
__global__ void initialDataCostsAtCurrentLevelAddCostsCheckerboardSampInvarientNoTextures(float* dataCostsCurrentLevelCurrentCheckerboard,
											int widthImageAtLevel, int heightImageAtLevel,
											checkerboardPortionEnum checkerboardPart, 
											currentStartMoveParamsPixelAtLevel* currentPixParamsCurrentCheckerboard);


//determine the message value using the brute force method of checking every possibility
__device__ void dtMotionEstimationUseBruteForce(float f[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y]);

//dt of 2d function; first minimize by row then by column; the manhatten distance is used for the distance function when the movement can
//be in both and x and y directions
__device__ void dtMotionEstimation(float f[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y]);

// compute BP message to pass to neighboring pixels
__device__ void computeBpMsgMotionEstimation(float s1[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float s2[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], 
	 float s3[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float s4[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y],
	 float dst[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y]);



//retrieve the message value of the given pixel at the given disparity level, this value is estimated if the message value is not within the desired "range"
//assuming that pixel actually does exist for now...
__device__ float retMessageValPixAtDispLevel(currentStartMoveParamsPixelAtLevel currentStartMovePixelParams, float currentXMove, float currentYMove, float desiredMessVals[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y]);


//device portion of the kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the 
//"checkerboard" scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
__device__ void runBPIterationUsingCheckerboardUpdatesDevice(checkerboardMessagesDeviceStruct messageValsDeviceCurrCheckOut,
														float dataMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], 
														int widthLevelCheckerboardPart, int heightLevelCheckerboard, int checkerboardAdjustment,
														int xVal, int yVal, int paramsTexOffset = 0);

//device portion of the kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the 
//"checkerboard" scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
__device__ void runBPIterationUsingCheckerboardUpdatesDeviceNoTextures(checkerboardMessagesDeviceStruct messageValsDeviceCurrCheckOut, 
									checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboardIn, 
									float dataMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], 
									int widthLevelCheckerboardPart, int heightLevelCheckerboard, int checkerboardAdjustment,
									int xVal, int yVal, 
									currentStartMoveParamsPixelAtLevel* currentPixParamsCurrentCheckerboard, 
									currentStartMoveParamsPixelAtLevel* currentPixParamsNeighCheckerboard);


//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard" 
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
__global__ void runBPIterationUsingCheckerboardUpdates( checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboardOut,
													int widthLevel, int heightLevel, checkerboardPortionEnum currentCheckerboardUpdating, int dataTexOffset = 0, int paramsTexOffset = 0);


//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard" 
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
//same as previous function but without textures...
__global__ void runBPIterationUsingCheckerboardUpdatesNoTextures( checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboardOut, checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboardIn,
													int widthLevel, int heightLevel, checkerboardPortionEnum currentCheckerboardUpdating,
				currentStartMoveParamsPixelAtLevel* currentPixParamsCurrentCheckerboard, 
									currentStartMoveParamsPixelAtLevel* currentPixParamsNeighCheckerboard,
									float* dataCostsCurrCheckerboard);

//kernal to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//note that since the motions at the next level "down" are of a different range, adjustments need to be made when copying values
//to the pixels at the next level down
//also copy the paramsCurrentPixelAtLevel to the next level down
__global__ void copyPrevLevelToNextLevelBPCheckerboard(checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboard1,
															checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboard2,
															int widthPrevLevel, int heightPrevLevel, int widthNextLevel, int heightNextLevel,
															currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboard1,
															currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboard2,
															checkerboardPortionEnum checkerboardPart);

//kernal to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//note that since the motions at the next level "down" are of a different range, adjustments need to be made when copying values
//to the pixels at the next level down
//also copy the paramsCurrentPixelAtLevel to the next level down
//the size of each level remains constant here
__global__ void copyPrevLevelToNextLevelBPCheckerboardLevelSizeConst(checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboardCopyTo,
															int widthLevel, int heightLevel,
															currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboardCopyTo);

//use the retrieved output movement to initialize the x and y movement range at the next level
//for now, perform this in steps of "two", where the movement range is halved each time
__device__ currentStartMoveParamsPixelAtLevel retrieveMovementRangeAtNextLevel(currentStartMoveParamsPixelAtLevel inputStartPossMovePixelParam, float estXDirMotion, float estYDirMotion);


//use the floating-point 2-D indices and the current 2-D array to retrieve the current value
//using bilinear interpolation
__device__ float retrieve2DArrayValFromIndex(float xIndex, float yIndex, int width2DArray, int height2DArray, float current2DArray[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y]);


//retrieve the "best movement/disparity" from image 1 to image 2 for each pixel in parallel
//the message values and data values are all stored in textures
//use message values for current nodes
__global__ void retrieveOutputMovementCheckerboard(float* movementXBetweenImagesCurrentCheckerboardDevice, float* movementYBetweenImagesCurrentCheckerboardDevice, int widthLevel, int heightLevel);


//device function to join together two float 2D array checkerboard portions for output as a single 2D array
__global__ void joinCheckerboardPortions(float* checkerboardPortion1Device, float* checkerboardPortion2Device, int widthLevel, int heightLevel, float* outputFloatArray);

//device function to round each floating point value in an array to an whole number
__global__ void roundDeviceValsKernel(float* inputValsDevice, int widthVals, int heightVals, float* outputValsDevice);


//global function to retrieve the parameters at the 'higher' level of the pyramid...
//this is performed from the perspective of the `current' pixel, retrieving parameters from the `lower level'...
__global__ void retParamsHigherLevPyramid(currentStartMoveParamsPixelAtLevel* paramsCheckerboard1EachHierarchLevel, currentStartMoveParamsPixelAtLevel* paramsCheckerboard2EachHierarchLevel, int widthPrevLevCheckerboard, int heightPrevLevel, int widthCurrLevCheckerboard, int heightCurrLevel, size_t paramsOffsetPrevLev, size_t paramsOffsetCurrLevel, checkerboardPortionEnum checkerboardPart);

//global function to retrieve the data costs for the current level of the computation pyramid...
__global__ void retDataCostsCurrLevCompPyramid(float* dataCostsCurrentLevelCurrentCheckerboard, currentStartMoveParamsPixelAtLevel* paramsLevelCurrCheckerboard, int widthImageAtLevel, int heightImageAtLevel, int offsetIntoParams, checkerboardPortionEnum checkerboardPart);

//device function to compute the values in the next level of the computation pyramid
__device__ void compValsNextLevelCompPyramid(float prevMessValsU[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float prevMessValsD[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float prevMessValsL[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float prevMessValsR[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], currentStartMoveParamsPixelAtLevel currentMovementStartPrevLevPix, currentStartMoveParamsPixelAtLevel currentMovementStartNextLevPix, float currMessValsU[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float currMessValsD[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float currMessValsL[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float currMessValsR[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y]);


//global function to copy the values `down' in the next level of the computation pyramid...
__global__ void copyMessValsDownCompPyramid(checkerboardMessagesDeviceStruct messagesDevicePrevCurrCheckerboard,
						checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboard1,
						checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboard2,
						int widthPrevLevel, int heightPrevLevel, 
						int widthNextLevel, int heightNextLevel,
						currentStartMoveParamsPixelAtLevel* paramsDeviceCheckerboard1,
						currentStartMoveParamsPixelAtLevel* paramsDeviceCheckerboard2, 
						size_t paramsOffsetPrevLev, size_t paramsOffsetCurrLevel, 
						checkerboardPortionEnum checkerboardPart);

//kernel for retrieving the `next' set of parameters given the current estimated values and the movement increment
__global__ void getNextParamSet(currentStartMoveParamsPixelAtLevel* paramsCurrCheckerboard, float* estXMotionCheckerboard, float* estYMotionCheckerboard, int widthCheckerboard, int heightCheckerboard);

//initialize the "data cost" for each possible movement at a particular level for the desired set of motions within one of the
//checkerboards
//the image data is stored in the CUDA arrays image1PixelsTextureAtLevel and 
//image2PixelsTextureAtLevel
//this method adds together all the results in a given range according to the current level
//in the "bottom" level, it is a single pixel, then the width/height are doubled at each level
//assuming that the input is full images
__global__ void initialDataAndEstMoveCostsAtCurrentLevelAddCostsCheckerboardSampInvarient(float* dataCostsCurrentLevelCurrentCheckerboard,
																	int widthImageAtLevel, int heightImageAtLevel,
																	checkerboardPortionEnum checkerboardPart, float* inputEstMoveXVals, float* inputEstMoveYVals);



#endif //KERNEL_RUN_BELIEF_PROP_CUH

