//runBeliefProp.cpp
//Scott Grauer-Gray
//June 24, 2009
//Define the functions for running belief propagation

#include "runBeliefProp.h"



//initialize running belief propagation with the desired parameters
runBeliefProp::runBeliefProp(int numBpLevels, int numBpItersPerLevel, float bpDataCostCap, float bpSmoothnessCostCap, float bpDataCostWeight,
		float bpCurrentMoveIncX, float bpCurrentMoveIncY, float bpSamplingLevel, float bpXMoveMin, float bpXMoveMax, float bpYMoveMin, float bpYMoveMax,
		float bpPropChangeMoveNextLevel, discontCostType bpCurrDiscCostType, posMoveDirection bpCurrPosMoveDir,
		int inputNumLevelsWithinLevPyrHierarch,
		currMethodProcessingHierarch inputCurrProcessingMethSetting, 
		usePrevMovement inputUsePrevMovementSetting) :
		numBeliefPropLevels(numBpLevels), numBeliefPropItersPerLevel(numBpItersPerLevel), dataCostCap(bpDataCostCap),
		smoothnessCostCap(bpSmoothnessCostCap), dataCostWeight(bpDataCostWeight), currentMoveIncrementX(bpCurrentMoveIncX), currentMoveIncrementY(bpCurrentMoveIncY),
		samplingLevel(bpSamplingLevel), xMoveMin(bpXMoveMin), xMoveMax(bpXMoveMax), yMoveMin(bpYMoveMin), yMoveMax(bpYMoveMax),
		propChangeMoveNextLevel(bpPropChangeMoveNextLevel), currDiscCostType(bpCurrDiscCostType), currPosMoveDir(bpCurrPosMoveDir), 
		numLevelsWithinLevPyrHierarch(inputNumLevelsWithinLevPyrHierarch),
		currProcessingMethSetting(inputCurrProcessingMethSetting),
		usePrevMovementSetting(inputUsePrevMovementSetting)
		{ }

runBeliefProp::~runBeliefProp(void)
{
}

//defines the operator to call to run belief propagation
resultMovement* runBeliefProp::operator()(bpImage* bpImage1, bpImage* bpImage2)
{
	//make sure that the width and height of the input images match
	if ((bpImage1->getImageWidth() != bpImage2->getImageWidth()) ||
		(bpImage1->getImageHeight() != bpImage2->getImageHeight()))
	{
		printf("Error, image dimensions do not match\n");
		return NULL;
	}

	//make sure that both images are in float form
	if ((bpImage1->getPixDataType() != FLOAT_DATA) || (bpImage2->getPixDataType() != FLOAT_DATA))
	{
		printf("Error, at least one input image not in float format\n");
		return NULL;
	}
	
	//make sure that both images are on device
	if ((bpImage1->getPixDataLocation() != DATA_ON_DEVICE) || 
		(bpImage2->getPixDataLocation() != DATA_ON_DEVICE))
	{
		printf("Error, at least one input image does not have data on device\n");
		return NULL;
	}

	//declares the resultant output from running belief propagation using the current parameters
	//generate the parameters for resultMovement
	resultMovement* resultantBeliefPropOutput = genResultMovementObject();

	//retrieve the width and height of the output movements using the width/height of the input and the sampling level
	int widthOutputMovement = (int)ceil(((float)bpImage1->getImageWidth()) / (samplingLevel));
	int heightOutputMovement = (int)ceil(((float)bpImage1->getImageHeight()) / (samplingLevel));

	//use the current width and height and the sampling level to define the width and height of the movement
	resultantBeliefPropOutput->setWidthMovements(widthOutputMovement);
	resultantBeliefPropOutput->setHeightMovements(heightOutputMovement);

	//generate the currBeliefPropParams
	currBeliefPropParams generatedCurrBeliefPropParams = genCurrBeliefPropParamsStruct();

	//input in the current width and height of the input images
	generatedCurrBeliefPropParams.widthImages = bpImage1->getImageWidth();
	generatedCurrBeliefPropParams.heightImages = bpImage1->getImageHeight();

	//declare and set the space for the output movement in the default location (which is on the device)
	float* outMoveXDevice;
	float* outMoveYDevice;

	allocateArray((void**)&outMoveXDevice, widthOutputMovement*heightOutputMovement*sizeof(float));
	allocateArray((void**)&outMoveYDevice, widthOutputMovement*heightOutputMovement*sizeof(float));

	//check desired hierarchical scheme to use
	if (currProcessingMethSetting == pyrHierarch)
	{
		runBeliefPropMotionEstimationCUDA(
			bpImage1->getImagePix().floatDataPointer,
			bpImage2->getImagePix().floatDataPointer,
			outMoveXDevice,
			outMoveYDevice,
			generatedCurrBeliefPropParams);
	}
	else if (currProcessingMethSetting == constHierarch)
	{
		runBeliefPropMotionEstimationCUDAUseConstLevelSize(
			bpImage1->getImagePix().floatDataPointer,
			bpImage2->getImagePix().floatDataPointer,
			outMoveXDevice,
			outMoveYDevice,
			generatedCurrBeliefPropParams);
	}
	else if (currProcessingMethSetting == constHierarchPyrWithin)
	{
		runBeliefPropMotionEstimationCUDAUseConstLevelSizeHierarchInLevUseEstMovement(
			bpImage1->getImagePix().floatDataPointer,
			bpImage2->getImagePix().floatDataPointer,
			outMoveXDevice,
			outMoveYDevice,
			generatedCurrBeliefPropParams);
	}

	//set the pointers to the output movement in resultantBeliefPropOutput to the generated output movement
	resultantBeliefPropOutput->setMovementXDir(outMoveXDevice);
	resultantBeliefPropOutput->setMovementYDir(outMoveYDevice);

	//return the output from running belief propagation
	return resultantBeliefPropOutput;
}


//defines the operator to call to run belief propagation with estimated movement...
resultMovement* runBeliefProp::runBeliefPropWithEstMovement(bpImage* bpImage1, bpImage* bpImage2, float* estMotionX, float* estMotionY)
{
	//make sure that the width and height of the input images match
	if ((bpImage1->getImageWidth() != bpImage2->getImageWidth()) ||
		(bpImage1->getImageHeight() != bpImage2->getImageHeight()))
	{
		printf("Error, image dimensions do not match\n");
		return NULL;
	}

	//make sure that both images are in float form
	if ((bpImage1->getPixDataType() != FLOAT_DATA) || (bpImage2->getPixDataType() != FLOAT_DATA))
	{
		printf("Error, at least one input image not in float format\n");
		return NULL;
	}
	
	//make sure that both images are on device
	if ((bpImage1->getPixDataLocation() != DATA_ON_DEVICE) || 
		(bpImage2->getPixDataLocation() != DATA_ON_DEVICE))
	{
		printf("Error, at least one input image does not have data on device\n");
		return NULL;
	}

	//declares the resultant output from running belief propagation using the current parameters
	//generate the parameters for resultMovement
	resultMovement* resultantBeliefPropOutput = genResultMovementObject();

	//retrieve the width and height of the output movements using the width/height of the input and the sampling level
	int widthOutputMovement = (int)ceil(((float)bpImage1->getImageWidth()) / (samplingLevel));
	int heightOutputMovement = (int)ceil(((float)bpImage1->getImageHeight()) / (samplingLevel));

	//use the current width and height and the sampling level to define the width and height of the movement
	resultantBeliefPropOutput->setWidthMovements(widthOutputMovement);
	resultantBeliefPropOutput->setHeightMovements(heightOutputMovement);

	//generate the currBeliefPropParams
	currBeliefPropParams generatedCurrBeliefPropParams = genCurrBeliefPropParamsStruct();

	//input in the current width and height of the input images
	generatedCurrBeliefPropParams.widthImages = bpImage1->getImageWidth();
	generatedCurrBeliefPropParams.heightImages = bpImage1->getImageHeight();

	//declare and set the space for the output movement in the default location (which is on the device)
	float* outMoveXDevice;
	float* outMoveYDevice;

	allocateArray((void**)&outMoveXDevice, widthOutputMovement*heightOutputMovement*sizeof(float));
	allocateArray((void**)&outMoveYDevice, widthOutputMovement*heightOutputMovement*sizeof(float));

	//right now running using a "const" hierarchy...
	runBeliefPropMotionEstimationCUDAUseConstLevelSizeGivenEstMovement(bpImage1->getImagePix().floatDataPointer, bpImage2->getImagePix().floatDataPointer, outMoveXDevice, outMoveYDevice, generatedCurrBeliefPropParams, estMotionX, estMotionY);

	//set the pointers to the output movement in resultantBeliefPropOutput to the generated output movement
	resultantBeliefPropOutput->setMovementXDir(outMoveXDevice);
	resultantBeliefPropOutput->setMovementYDir(outMoveYDevice);

	//return the output from running belief propagation
	return resultantBeliefPropOutput;
}

//private helper function to convert the current runBeliefProp into a currBeliefPropParams struct
currBeliefPropParams runBeliefProp::genCurrBeliefPropParamsStruct()
{
	//declare the currBeliefPropParams struct
	currBeliefPropParams generatedCurrBeliefPropParams;

	generatedCurrBeliefPropParams.numBpLevels = numBeliefPropLevels;
	generatedCurrBeliefPropParams.numBpIterations = numBeliefPropItersPerLevel;
	
	generatedCurrBeliefPropParams.dataCostCap = dataCostCap;
	generatedCurrBeliefPropParams.smoothnessCostCap = smoothnessCostCap;
	generatedCurrBeliefPropParams.dataCostWeight = dataCostWeight;

	generatedCurrBeliefPropParams.currentMoveIncrementX = currentMoveIncrementX;
	generatedCurrBeliefPropParams.currentMoveIncrementY = currentMoveIncrementY;					
	generatedCurrBeliefPropParams.propChangeMoveNextLevel = propChangeMoveNextLevel;
	generatedCurrBeliefPropParams.samplingLevel = samplingLevel;
	generatedCurrBeliefPropParams.startPossMoveX = xMoveMin;
	generatedCurrBeliefPropParams.startPossMoveY = yMoveMin;

	//need to compute the number of moves in the x and y directions using the
	//min and max possible movements as well as the increment
	generatedCurrBeliefPropParams.totalNumMovesXDir = (int)ceil((xMoveMax-xMoveMin)/currentMoveIncrementX) + 1;
	generatedCurrBeliefPropParams.totalNumMovesYDir = (int)ceil((yMoveMax-yMoveMin)/currentMoveIncrementY) + 1;

	//write in the current discontinuity cost type in the object...
	generatedCurrBeliefPropParams.currDiscCostType = currDiscCostType;

	//write in the current positive movement direction
	generatedCurrBeliefPropParams.directionPosMovement = currPosMoveDir;

	//write the number of 'levels' within the pyramid hierarchy...
	generatedCurrBeliefPropParams.numPyrHierarchLevels = numLevelsWithinLevPyrHierarch;

	//set whether or not to use the `previous movement'...
	generatedCurrBeliefPropParams.usePrevMovementSetting = usePrevMovementSetting;

	//return the generated currBeliefPropParams object
	return generatedCurrBeliefPropParams;
}

//private helper function to plug in the current parameters into the resultMovement
resultMovement* runBeliefProp::genResultMovementObject()
{
	resultMovement* resultingMove = new resultMovement();

	//define the direction of positive movement
	resultingMove->setPosMoveDirection(currPosMoveDir);

	//place the resulting location data in the default location
	resultingMove->setLocationMovementData(DEFAULT_LOCATION_OUTPUT_MOVEMENT_DATA);

	//define the starting movement in the x and y directions
	resultingMove->setStartMoveXDir(xMoveMin);
	resultingMove->setStartMoveYDir(yMoveMin);

	return resultingMove;
}
