//runBeliefPropMultImages.cpp
//Scott Grauer-Gray
//September 27, 2010
//Define the functions for running belief propagation on multiple images, with a particular set of images as the `target'

#include "runBeliefPropMultImages.h"



//initialize running belief propagation with the desired parameters
runBeliefPropMultImages::runBeliefPropMultImages(int numBpLevels, int numBpItersPerLevel, float bpDataCostCap, float bpSmoothnessCostCap, float bpDataCostWeight,
		float bpCurrentMoveIncX, float bpCurrentMoveIncY, float bpSamplingLevel, float bpXMoveMin, float bpXMoveMax, float bpYMoveMin, float bpYMoveMax,
		float bpPropChangeMoveNextLevel, discontCostType bpCurrDiscCostType, posMoveDirection bpCurrPosMoveDir, float bpMovementCostCap, float bpMovementCostWeight) :
		numBeliefPropLevels(numBpLevels), numBeliefPropItersPerLevel(numBpItersPerLevel), dataCostCap(bpDataCostCap),
		smoothnessCostCap(bpSmoothnessCostCap), dataCostWeight(bpDataCostWeight), currentMoveIncrementX(bpCurrentMoveIncX), currentMoveIncrementY(bpCurrentMoveIncY),
		samplingLevel(bpSamplingLevel), xMoveMin(bpXMoveMin), xMoveMax(bpXMoveMax), yMoveMin(bpYMoveMin), yMoveMax(bpYMoveMax),
		propChangeMoveNextLevel(bpPropChangeMoveNextLevel), currDiscCostType(bpCurrDiscCostType), currPosMoveDir(bpCurrPosMoveDir),
		movementCostCap(bpMovementCostCap), movementCostWeight(bpMovementCostWeight)
		{ }

runBeliefPropMultImages::~runBeliefPropMultImages(void)
{
}

//defines the operator to call to run belief propagation
//run belief propagation on each set of images and return the movement on the desired set of images...
resultMovement* runBeliefPropMultImages::operator()(bpImage** bpImageSet, int numImages, int startImageDesire, int endImageDesire)
{
	//retrieve the first image stuff for the width/height...
	bpImage* bpImageSamp = bpImageSet[0];

	//retrieve the width and height of the output movements using the width/height of the input and the sampling level
	int widthOutputMovement = (int)ceil(((float)bpImageSamp->getImageWidth()) / (samplingLevel));
	int heightOutputMovement = (int)ceil(((float)bpImageSamp->getImageHeight()) / (samplingLevel));

	//declare and set the space for the output movement in the default location (which is on the device)
	float* prevAndCurrMovementXFromImage1To2Device;
	float* prevAndCurrMovementYFromImage1To2Device;

	allocateArray((void**)&prevAndCurrMovementXFromImage1To2Device, widthOutputMovement*heightOutputMovement*sizeof(float));
	allocateArray((void**)&prevAndCurrMovementYFromImage1To2Device, widthOutputMovement*heightOutputMovement*sizeof(float));

	//declares the resultant output from running belief propagation using the current parameters
	//generate the parameters for resultMovement
	resultMovement* resultantBeliefPropOutput = genResultMovementObject();

	//go through each set of images and retrieve the desired movement between images...
	for (int imageNum = 0; imageNum < numImages-1; imageNum++)
	{

		//retrieve 'image 1' and 'image 2' of the current set...
		bpImage* bpImage1 = bpImageSet[imageNum];
		bpImage* bpImage2 = bpImageSet[imageNum+1];

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


		//use the current width and height and the sampling level to define the width and height of the movement
		resultantBeliefPropOutput->setWidthMovements(widthOutputMovement);
		resultantBeliefPropOutput->setHeightMovements(heightOutputMovement);

		//generate the currBeliefPropParams
		currBeliefPropParams generatedCurrBeliefPropParams = genCurrBeliefPropParamsStruct();

		//input in the current width and height of the input images
		generatedCurrBeliefPropParams.widthImages = bpImage1->getImageWidth();
		generatedCurrBeliefPropParams.heightImages = bpImage1->getImageHeight();


		//assume using cuboid scheme for now...

		//if the first two images, then obviously there is no previous movement...
		if (imageNum == 0)
		{
			runBeliefPropMotionEstimationCUDAUseConstLevelSize(
				bpImage1->getImagePix().floatDataPointer,
				bpImage2->getImagePix().floatDataPointer,
				prevAndCurrMovementXFromImage1To2Device,
				prevAndCurrMovementYFromImage1To2Device,
				generatedCurrBeliefPropParams);
		}
		else
		{

			runBeliefPropMotionEstimationCUDAUseConstLevelSizeUseEstMovement(
				bpImage1->getImagePix().floatDataPointer,
				bpImage2->getImagePix().floatDataPointer,
				prevAndCurrMovementXFromImage1To2Device, 
				prevAndCurrMovementYFromImage1To2Device, 
				generatedCurrBeliefPropParams);

		}

	}

	//set the pointers to the output movement in resultantBeliefPropOutput to the generated output movement
	resultantBeliefPropOutput->setMovementXDir(prevAndCurrMovementXFromImage1To2Device);
	resultantBeliefPropOutput->setMovementYDir(prevAndCurrMovementYFromImage1To2Device);

	//return the output from running belief propagation
	return resultantBeliefPropOutput;
}

//defines the operator to call to run belief propagation with estimated movement...
resultMovement* runBeliefPropMultImages::runBeliefPropWithEstMovement(bpImage* bpImage1, bpImage* bpImage2, float* estMotionX, float* estMotionY)
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
currBeliefPropParams runBeliefPropMultImages::genCurrBeliefPropParamsStruct()
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

	//write the previous movement info
	generatedCurrBeliefPropParams.estMovementCostCap = movementCostCap;
	generatedCurrBeliefPropParams.estMovementCostWeight = movementCostWeight;

	//return the generated currBeliefPropParams object
	return generatedCurrBeliefPropParams;
}


//private helper function to plug in the current parameters into the resultMovement
resultMovement* runBeliefPropMultImages::genResultMovementObject()
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

