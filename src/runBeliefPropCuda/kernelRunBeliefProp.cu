//kernelRunBeliefProp.cu
//Scott Grauer-Gray
//June 25, 2009
//Defines the kernel functions used for running belief propagation

#include "kernelRunBeliefProp.cuh"


//retrieve the current 1-D index value of the given point at the given movement in the data cost and message data
__device__ int retrieveIndexInDataAndMessage(int xVal, int yVal, int widthDataLevel, int heightDataLevel, int currentMoveNum, int totalNumMoveVals, int offsetData)
{
	//use the function defined in the pre-processor directive to retrieve the desired index value
	return RETRIEVE_INDEX_IN_DATA_OR_MESSAGE_ARRAY_EQUATION(xVal, yVal, widthDataLevel, heightDataLevel, totalNumMoveVals, currentMoveNum, offsetData);
}

//retrieve the index value of the current point in a 2-D array where the index is across by column and then by row
__device__ int retrieveIndexCurrentPixel2DGrid(int xVal, int yVal, int width, int height, int offsetData)
{
	return (yVal*width + xVal);
}

//checks if the current point is within the image bounds
__device__ bool withinImageBounds(int xVal, int yVal, int width, int height)
{
	return ((xVal >= 0) && (xVal < width) && (yVal >= 0) && (yVal < height));
}

__device__ float getEuclideanDist(float xVal1, float yVal1, float xVal2, float yVal2)
{
	return sqrt((xVal1 - xVal2) * (xVal1 - xVal2) + (yVal1 - yVal2) * (yVal1 - yVal2));
}

__device__ float getManhattanDist(float xVal1, float yVal1, float xVal2, float yVal2)
{
	return (abs(xVal1 - xVal2) + abs(yVal1 - yVal2));
}

//function to set 'no movement data' at each pixel (for default...)
__global__ void setNoMoveDataAtPixels(float* estMovementDataX, float* estMovementDataY, int widthImageAtLevel, int heightImageAtLevel)
{
	int xValThread = blockIdx.x * blockDim.x + threadIdx.x;
	int yValThread = blockIdx.y * blockDim.y + threadIdx.y;

	//check to make sure that x and y are within the "checkerboard"
	if (withinImageBounds(xValThread, yValThread, widthImageAtLevel, heightImageAtLevel))
	{
		estMovementDataX[yValThread*widthImageAtLevel + xValThread] = NO_MOVEMENT_DATA;
		estMovementDataY[yValThread*widthImageAtLevel + xValThread] = NO_MOVEMENT_DATA;
	}
}

//function to set the 'estimated movement' for each pixel in the 'next' image...
__global__ void setEstMoveAtPixels(float* resultantXMovePrevIm, float* resultantYMovePrevIm, float* expectedMovementX,
				float* expectedMovementY, int widthImageAtLevel, int heightImageAtLevel)
{
	int xValThread = blockIdx.x * blockDim.x + threadIdx.x;
	int yValThread = blockIdx.y * blockDim.y + threadIdx.y;

	//retrieve the x and y 'locations' where are the same as the thread here...
	int xLocation = xValThread;
	int yLocation = yValThread;

	//check to make sure that x and y are within the "checkerboard"
	if (withinImageBounds(xLocation, yLocation, widthImageAtLevel, heightImageAtLevel))
	{
		//retrieve the movement data in the x and y directions, then place it in the appropriated 'expectedMovement' slot...
		float retMoveOnPixX = resultantXMovePrevIm[yLocation*widthImageAtLevel + xLocation];		
		float retMoveOnPixY = resultantYMovePrevIm[yLocation*widthImageAtLevel + xLocation];

		//if the spot the move is 'to' is within the 'image', then set it as expected movement...
		
		//round where the movement is 'to' to the nearest integer...
		int xMoveTo = (int)floor(retMoveOnPixX + 0.5f) + xLocation;
		int yMoveTo = (int)floor(retMoveOnPixY + 0.5f) + yLocation;

		//check if the 'move to' is within image bounds; if so, then set in expected movement...
		if (withinImageBounds(xMoveTo, yMoveTo, widthImageAtLevel, heightImageAtLevel))
		{
			expectedMovementX[yMoveTo*widthImageAtLevel + xMoveTo] = retMoveOnPixX;
			expectedMovementY[yMoveTo*widthImageAtLevel + xMoveTo] = retMoveOnPixY;
		}
	}
}

//function to add two sets of input data...
__global__ void addInputData(float* inOutData, float* inData, int widthImageAtLevel, int heightImageAtLevel)
{
	int xValThread = blockIdx.x * blockDim.x + threadIdx.x;
	int yValThread = blockIdx.y * blockDim.y + threadIdx.y;

	//retrieve the width and height of the checkerboard by dividing the width
	//and setting the height of the checkerboard to the image height
	int widthCheckerboard = widthImageAtLevel / 2;
	int heightCheckerboard = heightImageAtLevel;

	//check to make sure that x and y are within the "checkerboard"
	if (withinImageBounds(xValThread, yValThread, widthCheckerboard, heightCheckerboard))
	{

		//go through each possible 'movement value' and add the input data...
		for (int movementXYVal = 0; movementXYVal < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementXYVal++)
		{

			int indexVal = retrieveIndexInDataAndMessage(xValThread, yValThread, widthCheckerboard, heightCheckerboard, movementXYVal, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));

			inOutData[indexVal] += inData[indexVal];
		}
	}
}


//function to retrieve the 'deviation from estimated movement' cost
__global__ void getDevFromEstMoveCosts(float* inputEstMoveXVals, float* inputEstMoveYVals,
					float* outputEstValCosts, int widthImageAtLevel, int heightImageAtLevel,
					checkerboardPortionEnum checkerboardPart)
{
	int xValThread = blockIdx.x * blockDim.x + threadIdx.x;
	int yValThread = blockIdx.y * blockDim.y + threadIdx.y;

	//retrieve the width and height of the checkerboard by dividing the width
	//and setting the height of the checkerboard to the image height
	int widthCheckerboard = widthImageAtLevel / 2;
	int heightCheckerboard = heightImageAtLevel;

	//check to make sure that x and y are within the "checkerboard"
	if (withinImageBounds(xValThread, yValThread, widthCheckerboard, heightCheckerboard))
	{
		//used to adjust the pixel value based on the checkerboard portion
		int checkerboardPartAdjustment;

		//retrieve the index of the current pixel using the thread and current checkerboard portion
		if (checkerboardPart == CHECKERBOARD_PART_1_ENUM)
		{
			checkerboardPartAdjustment = (yValThread%2);
		}
		else if (checkerboardPart == CHECKERBOARD_PART_2_ENUM)
		{
			checkerboardPartAdjustment = ((yValThread+1)%2);
		}

		//declare and define the x and y indices of the current pixel within the "level image"
		int xIndexPixel = xValThread*2 + checkerboardPartAdjustment;
		int yIndexPixel = yValThread;

		//retrieve the width and the height of the full image via the BPSettings in constant memory
		int widthFullImage = currentBeliefPropParamsConstMem.widthImages;
		int heightFullImage = currentBeliefPropParamsConstMem.heightImages;

		//retrieve the movement estimation in the x and y directions...
		float moveEstXDir = inputEstMoveXVals[(yIndexPixel*widthFullImage) + xIndexPixel];
		float moveEstYDir = inputEstMoveYVals[(yIndexPixel*widthFullImage) + xIndexPixel];

		//if the estimated movement is non-existant, then set to default values...
		if ((abs(moveEstXDir - NO_MOVEMENT_DATA) < SMALL_VALUE) || (abs(moveEstYDir - NO_MOVEMENT_DATA) < SMALL_VALUE))
		{
			//go through entire range of movements in the x and y directions and set estimation cost to default value
			for (int movementXYVal = 0; movementXYVal < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementXYVal++)
			{
				int indexVal = retrieveIndexInDataAndMessage(xValThread, yValThread, widthCheckerboard, heightCheckerboard, movementXYVal, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));

				//set each data cost to a default value if outside of range
				outputEstValCosts[indexVal] = DEFAULT_EST_MOVEMENT_COST_VALUE;
			}
		}

		//otherwise, use the movement data as desired...
		else
		{ 

			//retrieve the sampling at the current level using the width of the full images and the width of the images at the current level
			//TODO: look into using "smaller" versions of the images (like mipmapping...)
			float samplingXDir = (float)widthFullImage / (float)widthImageAtLevel;
			float samplingYDir = (float)heightFullImage / (float)heightImageAtLevel;

			//retrieve the middle x-value and y-value in the image using the thread x and y values and also the sampling in each direction
			float midXValImage = ((float)xIndexPixel) * (samplingXDir) + samplingXDir / 2.0f;
			float midYValImage = ((float)yIndexPixel) * (samplingYDir) + samplingYDir / 2.0f;

			//use the sampling increment in the x and y directions for the extent of the data costs from the midpoint
			//subtract each by 1 since going from mid-point to mid-point
			float extentMoveEstCostsSumAcross = floor(samplingXDir + 0.5f) - 1.0f;
			float extentMoveEstCostsSumVertical = floor(samplingYDir + 0.5f) - 1.0f;

			int indexVal;

			//retrieve the current min and max movements in the x and y directions using the current paramsCurrentPixelAtLevel
			//at the current pixel as well as the global currentParamsAllPixAtLevelConstMem which applies to all pixels at the
			//current level
			//retrieve the current paramsCurrentPixelAtLevel for the desired pixel

			currentStartMoveParamsPixelAtLevel currentStartMovePixelParams;

			//check the current "checkerboard" and retrieve the current parameters from the texture assumed to be bound to the
			//appropriate values
			currentStartMovePixelParams = tex1Dfetch(currentPixParamsTexCurrentCheckerboard, retrieveIndexCurrentPixel2DGrid(xValThread, yValThread,
						widthCheckerboard, heightCheckerboard));
			


			//set the current index of the movement to 0
			int movementXYVal = 0;

			float currentXMove = currentStartMovePixelParams.x;
			float currentYMove = currentStartMovePixelParams.y;

			//go through entire range of movements in the x and y directions
			for (int numMoveYInRange = 0; numMoveYInRange < (currentBeliefPropParamsConstMem.totalNumMovesYDir); numMoveYInRange++)
			{
				//reset the current x movement to the minimum x movement
				currentXMove = currentStartMovePixelParams.x;

				for (int numMoveXInRange = 0; numMoveXInRange < (currentBeliefPropParamsConstMem.totalNumMovesXDir); numMoveXInRange++)
				{

					//use the thread indices for the index value
					indexVal = retrieveIndexInDataAndMessage(xValThread, yValThread, widthCheckerboard, heightCheckerboard, movementXYVal, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));

					//set the movement cost to 0...
					outputEstValCosts[indexVal] = 0.0f;


					//loop through all the pixels in the current range where the movement costs are evaluated
					for (float xPixLocation = (midXValImage - extentMoveEstCostsSumAcross/2.0f); xPixLocation <= (midXValImage + extentMoveEstCostsSumAcross/2.0f + SMALL_VALUE); xPixLocation += 1.0f)
					{
						for (float yPixLocation = (midYValImage - extentMoveEstCostsSumVertical/2.0f); yPixLocation <= (midYValImage + extentMoveEstCostsSumVertical/2.0f + SMALL_VALUE); yPixLocation += 1.0f)
						{
							float currMinEstMoveCost = INF_BP;

							//declare the variable for the x and y moves to be tested
							float xMoveTest;
							float yMoveTest;

							//declare the variables which declare the "range" to check around the current move and initialize it to 0
							float xMoveRange = 0.0f;
							float yMoveRange = 0.0f;

							#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
								//declare the variables for the bounds of the change in the min and max moves in the x/y range
								float xMoveChangeBounds = 0.0f;
								float yMoveChangeBounds = 0.0f;
							#endif //CLAMP_EDGE_MOVES_DATA_COST_SETTING


							//if current move increment is beyond a certain point, then check various possible movements
							//perform sampling-invarient data costs beyond a certain move increment
							if ((currentBeliefPropParamsConstMem.currentMoveIncrementX > (currentBeliefPropParamsConstMem.motionIncBotLevX)) || (currentBeliefPropParamsConstMem.currentMoveIncrementY > (currentBeliefPropParamsConstMem.motionIncBotLevY)))
							{
								//go through the rand in increments of 0.5f for each pixel and take the minimum cost
								//if y range is greater than 1, then check values "around" for sampling-invarient data costs
								if (currentBeliefPropParamsConstMem.totalNumMovesXDir > 1)
								{
									//xMoveRange = currentBeliefPropParamsConstMem.currentMoveIncrement / 2.0f;
									float numXMoves = ceil((currentBeliefPropParamsConstMem.currentMoveIncrementX / 2.0f) / currentBeliefPropParamsConstMem.motionIncBotLevX);
									xMoveRange = numXMoves * currentBeliefPropParamsConstMem.motionIncBotLevX;

									#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
										xMoveChangeBounds = currentBeliefPropParamsConstMem.currentMoveIncrementX / 2.0f;
									#endif //CLAMP_EDGE_MOVES_DATA_COST_SETTING
								}
								if (currentBeliefPropParamsConstMem.totalNumMovesYDir > 1)
								{
									//yMoveRange = currentBeliefPropParamsConstMem.currentMoveIncrement / 2.0f;
									float numYMoves = ceil((currentBeliefPropParamsConstMem.currentMoveIncrementY / 2.0f) / currentBeliefPropParamsConstMem.motionIncBotLevY);
									yMoveRange = numYMoves * currentBeliefPropParamsConstMem.motionIncBotLevY;
									
									#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
										yMoveChangeBounds = currentBeliefPropParamsConstMem.currentMoveIncrementY / 2.0f;
									#endif //CLAMP_EDGE_MOVES_DATA_COST_SETTING
								}
								

								//go through and retrieve the estimated movement costs of each move in the given range
								for (yMoveTest = (-1.0f * yMoveRange); yMoveTest <= (yMoveRange + SMALL_VALUE); yMoveTest += currentBeliefPropParamsConstMem.motionIncBotLevY)
								{
									for (xMoveTest = (-1.0f * xMoveRange); xMoveTest <= (xMoveRange + SMALL_VALUE); xMoveTest += currentBeliefPropParamsConstMem.motionIncBotLevX)
									{
										//declare the variables for the current x and y moves and initialize to the current move (clamped if that's the setting...)
										#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == DONT_CLAMP_EDGE_MOVES_DATA_COST)
											float currXMove = currentXMove + xMoveTest;
											float currYMove = currentYMove + yMoveTest;
										#elif (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
											float currXMove = currentXMove + max(-1.0f * xMoveChangeBounds, min(xMoveTest, xMoveChangeBounds));
											float currYMove = currentYMove + max(-1.0f * yMoveChangeBounds, min(yMoveTest, yMoveChangeBounds));
										#endif//CLAMP_EDGE_MOVES_DATA_COST_SETTING
			

										currMinEstMoveCost = min(currMinEstMoveCost, currentBeliefPropParamsConstMem.estMovementCostWeight * min(getEuclideanDist(currXMove, currYMove, moveEstXDir, moveEstYDir), currentBeliefPropParamsConstMem.estMovementCostCap));
										
									}
								}

								//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
								//save the data cost for the current pixel on the current "checkerboard"
								outputEstValCosts[indexVal] += currMinEstMoveCost;
							}
						}

						movementXYVal++;


						
					}

					//increment the x-movement by current movement increment
					currentXMove += currentBeliefPropParamsConstMem.currentMoveIncrementX;

					
				}

				//increment the y-movement by current movement increment
				currentYMove += currentBeliefPropParamsConstMem.currentMoveIncrementY;	
			}
		}
	}
}


//function to retrieve the 'deviation from estimated movement' cost without using textures...
__global__ void getDevFromEstMoveCostsNoTextures(float* inputEstMoveXVals, float* inputEstMoveYVals,
					float* outputEstValCosts, int widthImageAtLevel, int heightImageAtLevel,
					checkerboardPortionEnum checkerboardPart, currentStartMoveParamsPixelAtLevel* currentPixParamsCurrentCheckerboard)
{
	int xValThread = blockIdx.x * blockDim.x + threadIdx.x;
	int yValThread = blockIdx.y * blockDim.y + threadIdx.y;

	//retrieve the width and height of the checkerboard by dividing the width
	//and setting the height of the checkerboard to the image height
	int widthCheckerboard = widthImageAtLevel / 2;
	int heightCheckerboard = heightImageAtLevel;

	//check to make sure that x and y are within the "checkerboard"
	if (withinImageBounds(xValThread, yValThread, widthCheckerboard, heightCheckerboard))
	{
		//used to adjust the pixel value based on the checkerboard portion
		int checkerboardPartAdjustment;

		//retrieve the index of the current pixel using the thread and current checkerboard portion
		if (checkerboardPart == CHECKERBOARD_PART_1_ENUM)
		{
			checkerboardPartAdjustment = (yValThread%2);
		}
		else if (checkerboardPart == CHECKERBOARD_PART_2_ENUM)
		{
			checkerboardPartAdjustment = ((yValThread+1)%2);
		}

		//declare and define the x and y indices of the current pixel within the "level image"
		int xIndexPixel = xValThread*2 + checkerboardPartAdjustment;
		int yIndexPixel = yValThread;

		//retrieve the width and the height of the full image via the BPSettings in constant memory
		int widthFullImage = currentBeliefPropParamsConstMem.widthImages;
		int heightFullImage = currentBeliefPropParamsConstMem.heightImages;

		//retrieve the movement estimation in the x and y directions...
		float moveEstXDir = inputEstMoveXVals[(yIndexPixel*widthFullImage) + xIndexPixel];
		float moveEstYDir = inputEstMoveYVals[(yIndexPixel*widthFullImage) + xIndexPixel];

		//if the estimated movement is non-existant, then set to default values...
		if ((abs(moveEstXDir - NO_MOVEMENT_DATA) < SMALL_VALUE) || (abs(moveEstYDir - NO_MOVEMENT_DATA) < SMALL_VALUE))
		{
			//go through entire range of movements in the x and y directions and set estimation cost to default value
			for (int movementXYVal = 0; movementXYVal < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementXYVal++)
			{
				int indexVal = retrieveIndexInDataAndMessage(xValThread, yValThread, widthCheckerboard, heightCheckerboard, movementXYVal, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));

				//set each data cost to a default value if outside of range
				outputEstValCosts[indexVal] = DEFAULT_EST_MOVEMENT_COST_VALUE;
			}
		}

		//otherwise, use the movement data as desired...
		else
		{ 

			//retrieve the sampling at the current level using the width of the full images and the width of the images at the current level
			//TODO: look into using "smaller" versions of the images (like mipmapping...)
			float samplingXDir = (float)widthFullImage / (float)widthImageAtLevel;
			float samplingYDir = (float)heightFullImage / (float)heightImageAtLevel;

			//retrieve the middle x-value and y-value in the image using the thread x and y values and also the sampling in each direction
			float midXValImage = ((float)xIndexPixel) * (samplingXDir) + samplingXDir / 2.0f;
			float midYValImage = ((float)yIndexPixel) * (samplingYDir) + samplingYDir / 2.0f;

			//use the sampling increment in the x and y directions for the extent of the data costs from the midpoint
			//subtract each by 1 since going from mid-point to mid-point
			float extentMoveEstCostsSumAcross = floor(samplingXDir + 0.5f) - 1.0f;
			float extentMoveEstCostsSumVertical = floor(samplingYDir + 0.5f) - 1.0f;

			int indexVal;

			//retrieve the current min and max movements in the x and y directions using the current paramsCurrentPixelAtLevel
			//at the current pixel as well as the global currentParamsAllPixAtLevelConstMem which applies to all pixels at the
			//current level
			//retrieve the current paramsCurrentPixelAtLevel for the desired pixel

			currentStartMoveParamsPixelAtLevel currentStartMovePixelParams;

			//check the current "checkerboard" and retrieve the current parameters from the texture assumed to be bound to the
			//appropriate values
			currentStartMovePixelParams = currentPixParamsCurrentCheckerboard[retrieveIndexCurrentPixel2DGrid(xValThread, yValThread,
						widthCheckerboard, heightCheckerboard)];
			

			//set the current index of the movement to 0
			int movementXYVal = 0;

			float currentXMove = currentStartMovePixelParams.x;
			float currentYMove = currentStartMovePixelParams.y;

			//go through entire range of movements in the x and y directions
			for (int numMoveYInRange = 0; numMoveYInRange < (currentBeliefPropParamsConstMem.totalNumMovesYDir); numMoveYInRange++)
			{
				//reset the current x movement to the minimum x movement
				currentXMove = currentStartMovePixelParams.x;

				for (int numMoveXInRange = 0; numMoveXInRange < (currentBeliefPropParamsConstMem.totalNumMovesXDir); numMoveXInRange++)
				{
						
					//use the thread indices for the index value
					indexVal = retrieveIndexInDataAndMessage(xValThread, yValThread, widthCheckerboard, heightCheckerboard, movementXYVal, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));

					//set the movement cost to 0...
					outputEstValCosts[indexVal] = 0.0f;


					//loop through all the pixels in the current range where the movement costs are evaluated
					for (float xPixLocation = (midXValImage - extentMoveEstCostsSumAcross/2.0f); xPixLocation <= (midXValImage + extentMoveEstCostsSumAcross/2.0f + SMALL_VALUE); xPixLocation += 1.0f)
					{
						for (float yPixLocation = (midYValImage - extentMoveEstCostsSumVertical/2.0f); yPixLocation <= (midYValImage + extentMoveEstCostsSumVertical/2.0f + SMALL_VALUE); yPixLocation += 1.0f)
						{
							float currMinEstMoveCost = INF_BP;

							//declare the variable for the x and y moves to be tested
							float xMoveTest;
							float yMoveTest;

							//declare the variables which declare the "range" to check around the current move and initialize it to 0
							float xMoveRange = 0.0f;
							float yMoveRange = 0.0f;

							#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
								//declare the variables for the bounds of the change in the min and max moves in the x/y range
								float xMoveChangeBounds = 0.0f;
								float yMoveChangeBounds = 0.0f;
							#endif //CLAMP_EDGE_MOVES_DATA_COST_SETTING


							//if current move increment is beyond a certain point, then check various possible movements
							//perform sampling-invarient data costs beyond a certain move increment
							if ((currentBeliefPropParamsConstMem.currentMoveIncrementX > (currentBeliefPropParamsConstMem.motionIncBotLevX)) || (currentBeliefPropParamsConstMem.currentMoveIncrementY > (currentBeliefPropParamsConstMem.motionIncBotLevY)))
							{
								//go through the rand in increments of 0.5f for each pixel and take the minimum cost
								//if y range is greater than 1, then check values "around" for sampling-invarient data costs
								if (currentBeliefPropParamsConstMem.totalNumMovesXDir > 1)
								{
									//xMoveRange = currentBeliefPropParamsConstMem.currentMoveIncrement / 2.0f;
									float numXMoves = ceil((currentBeliefPropParamsConstMem.currentMoveIncrementX / 2.0f) / currentBeliefPropParamsConstMem.motionIncBotLevX);
									xMoveRange = numXMoves * currentBeliefPropParamsConstMem.motionIncBotLevX;

									#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
										xMoveChangeBounds = currentBeliefPropParamsConstMem.currentMoveIncrementX / 2.0f;
									#endif //CLAMP_EDGE_MOVES_DATA_COST_SETTING
								}
								if (currentBeliefPropParamsConstMem.totalNumMovesYDir > 1)
								{
									//yMoveRange = currentBeliefPropParamsConstMem.currentMoveIncrement / 2.0f;
									float numYMoves = ceil((currentBeliefPropParamsConstMem.currentMoveIncrementY / 2.0f) / currentBeliefPropParamsConstMem.motionIncBotLevY);
									yMoveRange = numYMoves * currentBeliefPropParamsConstMem.motionIncBotLevY;
									
									#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
										yMoveChangeBounds = currentBeliefPropParamsConstMem.currentMoveIncrementY / 2.0f;
									#endif //CLAMP_EDGE_MOVES_DATA_COST_SETTING
								}
								

								//go through and retrieve the estimated movement costs of each move in the given range
								for (yMoveTest = (-1.0f * yMoveRange); yMoveTest <= (yMoveRange + SMALL_VALUE); yMoveTest += currentBeliefPropParamsConstMem.motionIncBotLevY)
								{
									for (xMoveTest = (-1.0f * xMoveRange); xMoveTest <= (xMoveRange + SMALL_VALUE); xMoveTest += currentBeliefPropParamsConstMem.motionIncBotLevX)
									{
										//declare the variables for the current x and y moves and initialize to the current move (clamped if that's the setting...)
										#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == DONT_CLAMP_EDGE_MOVES_DATA_COST)
											float currXMove = currentXMove + xMoveTest;
											float currYMove = currentYMove + yMoveTest;
										#elif (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
											float currXMove = currentXMove + max(-1.0f * xMoveChangeBounds, min(xMoveTest, xMoveChangeBounds));
											float currYMove = currentYMove + max(-1.0f * yMoveChangeBounds, min(yMoveTest, yMoveChangeBounds));
										#endif//CLAMP_EDGE_MOVES_DATA_COST_SETTING
			

										currMinEstMoveCost = min(currMinEstMoveCost, currentBeliefPropParamsConstMem.estMovementCostWeight * min(getEuclideanDist(currXMove, currYMove, moveEstXDir, moveEstYDir), currentBeliefPropParamsConstMem.estMovementCostCap));
										
									}
								}

								//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
								//save the data cost for the current pixel on the current "checkerboard"
								outputEstValCosts[indexVal] += currMinEstMoveCost;
							}
						}

						movementXYVal++;


						
					}

					//increment the x-movement by current movement increment
					currentXMove += currentBeliefPropParamsConstMem.currentMoveIncrementX;

					
				}

				//increment the y-movement by current movement increment
				currentYMove += currentBeliefPropParamsConstMem.currentMoveIncrementY;	
			}
		}
	}
}

//computes the "distance" between two movements
__device__ float computeDistanceBetweenMovements(float movement1Xcomp, float movement1Ycomp, float movement2Xcomp, float movement2Ycomp)
{
	if ((currentBeliefPropParamsConstMem.currDiscCostType == USE_EXACT_EUCLIDEAN_DIST_FOR_DISCONT) || (currentBeliefPropParamsConstMem.currDiscCostType == USE_APPROX_EUCLIDEAN_DIST_FOR_DISCONT))
	{
		return getEuclideanDist(movement1Xcomp, movement1Ycomp, movement2Xcomp, movement2Ycomp);
	}
	else if ((currentBeliefPropParamsConstMem.currDiscCostType == USE_MANHATTAN_DIST_FOR_DISCONT) || (currentBeliefPropParamsConstMem.currDiscCostType == USE_MANHATTEN_DIST_USING_BRUTE_FORCE))
	{	
		return getManhattanDist(movement1Xcomp, movement1Ycomp, movement2Xcomp, movement2Ycomp);
	}
	else
	{
		return 0.0f;
	}
}

//global function to set params defining the starting possible x and y motion to a particular level
__global__ void setParamsToGivenParamKernel(currentStartMoveParamsPixelAtLevel currentPixelParams, currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1,
											currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2, int widthLevel, int heightLevel)
{
	int xVal = blockIdx.x*blockDim.x + threadIdx.x;
	int yVal = blockIdx.y*blockDim.y + threadIdx.y;

	//set the width of the checkerboard at the current level, which is half the width of the level
	//and the height which is the same as the height at the level
	int widthCheckerboardAtLevel = widthLevel / 2;
	int heightCheckerboard = heightLevel;

	//set each pixel within the current bounds to the given parameter
	if (withinImageBounds(xVal, yVal, widthCheckerboardAtLevel, heightCheckerboard))
	{
		paramsCurrentLevelDeviceCheckerboard1[retrieveIndexCurrentPixel2DGrid(xVal, yVal, widthCheckerboardAtLevel, heightCheckerboard)] = currentPixelParams;
		paramsCurrentLevelDeviceCheckerboard2[retrieveIndexCurrentPixel2DGrid(xVal, yVal, widthCheckerboardAtLevel, heightCheckerboard)] = currentPixelParams;
	}
}

//initialize the message values at each pixel of the current level to the default value for each checkerboard
__global__ void initMessageValsToDefaultCurrCheckerboard(checkerboardMessagesDeviceStruct messageValsDeviceCheckerboard1,
														 checkerboardMessagesDeviceStruct messageValsDeviceCheckerboard2,
														int widthLevel, int heightLevel)
{
	int xVal = blockIdx.x * blockDim.x + threadIdx.x;
	int yVal = blockIdx.y * blockDim.y + threadIdx.y;

	int widthCheckerboardAtLevel = widthLevel / 2;
	int heightCheckerboardAtLevel = heightLevel;

	if (withinImageBounds(xVal, yVal, widthCheckerboardAtLevel, heightCheckerboardAtLevel))
	{
		//set the message value at each pixel for each movement to a default value
		for (int movementXY = 0; movementXY < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementXY++)
		{
			messageValsDeviceCheckerboard1.messageUDevice[retrieveIndexInDataAndMessage(xVal, yVal, widthCheckerboardAtLevel, heightCheckerboardAtLevel, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir))] = DEFAULT_INITIAL_MESSAGE_VAL;
			messageValsDeviceCheckerboard1.messageDDevice[retrieveIndexInDataAndMessage(xVal, yVal, widthCheckerboardAtLevel, heightCheckerboardAtLevel, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir))] = DEFAULT_INITIAL_MESSAGE_VAL;
			messageValsDeviceCheckerboard1.messageLDevice[retrieveIndexInDataAndMessage(xVal, yVal, widthCheckerboardAtLevel, heightCheckerboardAtLevel, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir))] = DEFAULT_INITIAL_MESSAGE_VAL;
			messageValsDeviceCheckerboard1.messageRDevice[retrieveIndexInDataAndMessage(xVal, yVal, widthCheckerboardAtLevel, heightCheckerboardAtLevel, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir))] = DEFAULT_INITIAL_MESSAGE_VAL;

			messageValsDeviceCheckerboard2.messageUDevice[retrieveIndexInDataAndMessage(xVal, yVal, widthCheckerboardAtLevel, heightCheckerboardAtLevel, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir))] = DEFAULT_INITIAL_MESSAGE_VAL;
			messageValsDeviceCheckerboard2.messageDDevice[retrieveIndexInDataAndMessage(xVal, yVal, widthCheckerboardAtLevel, heightCheckerboardAtLevel, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir))] = DEFAULT_INITIAL_MESSAGE_VAL;
			messageValsDeviceCheckerboard2.messageLDevice[retrieveIndexInDataAndMessage(xVal, yVal, widthCheckerboardAtLevel, heightCheckerboardAtLevel, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir))] = DEFAULT_INITIAL_MESSAGE_VAL;
			messageValsDeviceCheckerboard2.messageRDevice[retrieveIndexInDataAndMessage(xVal, yVal, widthCheckerboardAtLevel, heightCheckerboardAtLevel, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir))] = DEFAULT_INITIAL_MESSAGE_VAL;
		}
	}
}

//initialize the "data cost" for each possible movement at a particular level for the desired set of motions within one of the
//checkerboards
//the image data is stored in the CUDA arrays image1PixelsTextureAtLevel and 
//image2PixelsTextureAtLevel
//this method adds together all the results in a given range according to the current level
//in the "bottom" level, it is a single pixel, then the width/height are doubled at each level
//assuming that the input is full images
__global__ void initialDataCostsAtCurrentLevelAddCostsCheckerboard(float* dataCostsCurrentLevelCurrentCheckerboard,
																	int widthImageAtLevel, int heightImageAtLevel,
																	checkerboardPortionEnum checkerboardPart)
{
	int xValThread = blockIdx.x * blockDim.x + threadIdx.x;
	int yValThread = blockIdx.y * blockDim.y + threadIdx.y;


	//retrieve the width and height of the checkerboard by dividing the width
	//and setting the height of the checkerboard to the image height
	int widthCheckerboard = widthImageAtLevel / 2;
	int heightCheckerboard = heightImageAtLevel;

	//check to make sure that x and y are within the "checkerboard"
	if (withinImageBounds(xValThread, yValThread, widthCheckerboard, heightCheckerboard))
	{
		//used to adjust the pixel value based on the checkerboard portion
		int checkerboardPartAdjustment;

		//retrieve the index of the current pixel using the thread and current checkerboard portion
		if (checkerboardPart == CHECKERBOARD_PART_1_ENUM)
		{
			checkerboardPartAdjustment = (yValThread%2);
		}
		else if (checkerboardPart == CHECKERBOARD_PART_2_ENUM)
		{
			checkerboardPartAdjustment = ((yValThread+1)%2);
		}

		//declare and define the x and y indices of the current pixel within the "level image"
		int xIndexPixel = xValThread*2 + checkerboardPartAdjustment;
		int yIndexPixel = yValThread;


		//retrieve the width and the height of the full image via the BPSettings in constant memory
		int widthFullImage = currentBeliefPropParamsConstMem.widthImages;
		int heightFullImage = currentBeliefPropParamsConstMem.heightImages;

		//retrieve the sampling at the current level using the width of the full images and the width of the images at the current level
		//TODO: look into using "smaller" versions of the images (like mipmapping...)
		float samplingXDir = (float)widthFullImage / (float)widthImageAtLevel;
		float samplingYDir = (float)heightFullImage / (float)heightImageAtLevel;

		//retrieve the middle x-value and y-value in the image using the thread x and y values and also the sampling in each direction
		float midXValImage = ((float)xIndexPixel) * (samplingXDir) + samplingXDir / 2.0f;
		float midYValImage = ((float)yIndexPixel) * (samplingYDir) + samplingYDir / 2.0f;

		//use the sampling increment in the x and y directions for the extent of the data costs from the midpoint
		//subtract each by 1 since going from mid-point to mid-point
		float extentDataCostsSumAcross = floor(samplingXDir + 0.5f) - 1.0f;
		float extentDataCostsSumVertical = floor(samplingYDir + 0.5f) - 1.0f;

		int indexVal;

		if (withinImageBounds(midXValImage, midYValImage, widthFullImage, heightFullImage))
		{
			//retrieve the current min and max movements in the x and y directions using the current paramsCurrentPixelAtLevel
			//at the current pixel as well as the global currentParamsAllPixAtLevelConstMem which applies to all pixels at the
			//current level
			//retrieve the current paramsCurrentPixelAtLevel for the desired pixel
			
			currentStartMoveParamsPixelAtLevel currentStartMovePixelParams;

			//check the current "checkerboard" and retrieve the current parameters from the texture assumed to be bound to the
			//appropriate values
			currentStartMovePixelParams = tex1Dfetch(currentPixParamsTexCurrentCheckerboard, retrieveIndexCurrentPixel2DGrid(xValThread, yValThread,
					widthCheckerboard, heightCheckerboard));
			

			//now use the given parameters to retrieve the min and max movements
			float maxMovementLeft;
			float maxMovementRight;
			float maxMovementUp;
			float maxMovementDown;

			if (currentBeliefPropParamsConstMem.directionPosMovement == POS_X_MOTION_RIGHT_POS_Y_MOTION_DOWN)
			{
				maxMovementLeft = -1.0f*(currentStartMovePixelParams.x);
				maxMovementRight = (currentStartMovePixelParams.x + currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.currentMoveIncrementX);
				maxMovementUp = -1.0f*(currentStartMovePixelParams.y);
				maxMovementDown = (currentStartMovePixelParams.y + currentBeliefPropParamsConstMem.totalNumMovesYDir * currentBeliefPropParamsConstMem.currentMoveIncrementY);
			}
			else if (currentBeliefPropParamsConstMem.directionPosMovement == POS_X_MOTION_LEFT_POS_Y_MOTION_UP)
			{
				maxMovementLeft = (currentStartMovePixelParams.x + currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.currentMoveIncrementX);
				maxMovementRight = -1.0f*(currentStartMovePixelParams.x);
				maxMovementUp = (currentStartMovePixelParams.y + currentBeliefPropParamsConstMem.totalNumMovesYDir * currentBeliefPropParamsConstMem.currentMoveIncrementY);
				maxMovementDown = -1.0f*(currentStartMovePixelParams.y);
			}
			else
			{
				maxMovementLeft = 0.0f;
				maxMovementRight = 0.0f;
				maxMovementUp = 0.0f;
				maxMovementDown = 0.0f;
			}

			//set the current index of the movement to 0
			int movementXYVal = 0;

			if ( (((midXValImage - extentDataCostsSumAcross/2.0f) - maxMovementLeft) >= 0) && (((midXValImage + extentDataCostsSumAcross/2.0f) + maxMovementRight) < (currentBeliefPropParamsConstMem.widthImages)) &&
				(((midYValImage - extentDataCostsSumVertical/2.0f) - maxMovementUp) >= 0) && (((midYValImage + extentDataCostsSumVertical/2.0f) + maxMovementDown) < (currentBeliefPropParamsConstMem.heightImages)))
			{

				//set current movement to the min movement in the x and y directions and then increase by current move increment in the loop
				float currentXMove = currentStartMovePixelParams.x;
				float currentYMove = currentStartMovePixelParams.y;


				//go through entire range of movements in the x and y directions
				for (int numMoveYInRange = 0; numMoveYInRange < (currentBeliefPropParamsConstMem.totalNumMovesYDir); numMoveYInRange++)
				{
					//reset the current x movement to the minimum x movement
					currentXMove = currentStartMovePixelParams.x;

					for (int numMoveXInRange = 0; numMoveXInRange < (currentBeliefPropParamsConstMem.totalNumMovesXDir); numMoveXInRange++)
					{
						float currentPixelImage1;
						float currentPixelImage2;

						//use the thread indices for the index value
						indexVal = retrieveIndexInDataAndMessage(xValThread, yValThread, widthCheckerboard, heightCheckerboard, movementXYVal, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));

						//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
						//save the data cost for the current pixel on the current "checkerboard"
						//initialize each data cost to 0 then add each value within the current range of data values to add up
						dataCostsCurrentLevelCurrentCheckerboard[indexVal] = 0.0f;

						//loop through all the pixels in the current range where the data costs are being summed
						for (float xPixLocation = (midXValImage - extentDataCostsSumAcross/2.0f); xPixLocation <= (midXValImage + extentDataCostsSumAcross/2.0f + SMALL_VALUE); xPixLocation += 1.0f)
						{
							for (float yPixLocation = (midYValImage - extentDataCostsSumVertical/2.0f); yPixLocation <= (midYValImage + extentDataCostsSumVertical/2.0f + SMALL_VALUE); yPixLocation += 1.0f)
							{
								if (currentBeliefPropParamsConstMem.directionPosMovement == POS_X_MOTION_RIGHT_POS_Y_MOTION_DOWN)
								{
									//texture access is using normalized coordinate system
									currentPixelImage1 = tex2D(image1PixelsTexture, (((float)xPixLocation) / (float)widthFullImage), (((float)yPixLocation) / (float)heightFullImage));
									currentPixelImage2 = tex2D(image2PixelsTexture, (((float)(xPixLocation + currentXMove))) / ((float)widthFullImage), (((float)(yPixLocation + currentYMove))) / ((float)heightFullImage));
								}
								else if (currentBeliefPropParamsConstMem.directionPosMovement == POS_X_MOTION_LEFT_POS_Y_MOTION_UP)
								{
									currentPixelImage1 = tex2D(image1PixelsTexture, (((float)xPixLocation) / (float)widthFullImage), (((float)yPixLocation) / (float)heightFullImage));
									currentPixelImage2 = tex2D(image2PixelsTexture, (((float)(xPixLocation - currentXMove))) / ((float)widthFullImage), ((((float)(yPixLocation - currentYMove))) / ((float)heightFullImage)));
								}
								else
								{
									currentPixelImage1 = 0.0f;


									currentPixelImage2 = 0.0f;
								}

								//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
								//save the data cost for the current pixel on the current "checkerboard"
								dataCostsCurrentLevelCurrentCheckerboard[indexVal] += currentBeliefPropParamsConstMem.dataCostWeight * min(abs(currentPixelImage1 - currentPixelImage2), currentBeliefPropParamsConstMem.dataCostCap);

							}
						}

						movementXYVal++;

						//increment the x-movement by current movement increment
						currentXMove += currentBeliefPropParamsConstMem.currentMoveIncrementX;
					}

					//increment the y-movement by current movement increment
					currentYMove += currentBeliefPropParamsConstMem.currentMoveIncrementY;
				}

			}
			else
			{
				//go through entire range of movements in the x and y directions and set data cost to default value
				for (int movementXYVal = 0; movementXYVal < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementXYVal++)
				{
					indexVal = retrieveIndexInDataAndMessage(xValThread, yValThread, widthCheckerboard, heightCheckerboard, movementXYVal, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));

					//set each data cost to a default value if outside of range
					dataCostsCurrentLevelCurrentCheckerboard[indexVal] = currentBeliefPropParamsConstMem.dataCostWeight * DEFAULT_DATA_COST_VALUE;
				}
			}
		}
		else
		{
			//go through entire range of movements in the x and y directions and set data cost to default value
			for (int movementXYVal = 0; movementXYVal < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementXYVal++)
			{
				indexVal = retrieveIndexInDataAndMessage(xValThread, yValThread, widthCheckerboard, heightCheckerboard, movementXYVal, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));

				//set each data cost to a default value if outside of range
				dataCostsCurrentLevelCurrentCheckerboard[indexVal] = currentBeliefPropParamsConstMem.dataCostWeight * DEFAULT_DATA_COST_VALUE;
			}
		}
	}
}

//initialize the "data cost" for each possible movement at a particular level for the desired set of motions within one of the
//checkerboards
//the image data is stored in the CUDA arrays image1PixelsTextureAtLevel and 
//image2PixelsTextureAtLevel
//this method adds together all the results in a given range according to the current level
//in the "bottom" level, it is a single pixel, then the width/height are doubled at each level
//assuming that the input is full images
__global__ void initialDataCostsAtCurrentLevelAddCostsCheckerboardSampInvarient(float* dataCostsCurrentLevelCurrentCheckerboard,
																	int widthImageAtLevel, int heightImageAtLevel,
																	checkerboardPortionEnum checkerboardPart)
{
	int xValThread = blockIdx.x * blockDim.x + threadIdx.x;
	int yValThread = blockIdx.y * blockDim.y + threadIdx.y;

	//retrieve the width and height of the checkerboard by dividing the width
	//and setting the height of the checkerboard to the image height
	int widthCheckerboard = widthImageAtLevel / 2;
	int heightCheckerboard = heightImageAtLevel;


	//check to make sure that x and y are within the "checkerboard"
	if (withinImageBounds(xValThread, yValThread, widthCheckerboard, heightCheckerboard))
	{

		//used to adjust the pixel value based on the checkerboard portion
		int checkerboardPartAdjustment;

		//retrieve the index of the current pixel using the thread and current checkerboard portion
		if (checkerboardPart == CHECKERBOARD_PART_1_ENUM)
		{
			checkerboardPartAdjustment = (yValThread%2);
		}
		else if (checkerboardPart == CHECKERBOARD_PART_2_ENUM)
		{
			checkerboardPartAdjustment = ((yValThread+1)%2);
		}

		//declare and define the x and y indices of the current pixel within the "level image"
		int xIndexPixel = xValThread*2 + checkerboardPartAdjustment;
		int yIndexPixel = yValThread;


		//retrieve the width and the height of the full image via the BPSettings in constant memory
		int widthFullImage = currentBeliefPropParamsConstMem.widthImages;
		int heightFullImage = currentBeliefPropParamsConstMem.heightImages;

		//retrieve the sampling at the current level using the width of the full images and the width of the images at the current level
		//TODO: look into using "smaller" versions of the images (like mipmapping...)
		float samplingXDir = (float)widthFullImage / (float)widthImageAtLevel;
		float samplingYDir = (float)heightFullImage / (float)heightImageAtLevel;

		//retrieve the middle x-value and y-value in the image using the thread x and y values and also the sampling in each direction
		float midXValImage = ((float)xIndexPixel) * (samplingXDir) + samplingXDir / 2.0f;
		float midYValImage = ((float)yIndexPixel) * (samplingYDir) + samplingYDir / 2.0f;

		//use the sampling increment in the x and y directions for the extent of the data costs from the midpoint
		//subtract each by 1 since going from mid-point to mid-point
		float extentDataCostsSumAcross = floor(samplingXDir + 0.5f) - 1.0f;
		float extentDataCostsSumVertical = floor(samplingYDir + 0.5f) - 1.0f;

		int indexVal;

		if (withinImageBounds(midXValImage, midYValImage, widthFullImage, heightFullImage))
		{
			//retrieve the current min and max movements in the x and y directions using the current paramsCurrentPixelAtLevel
			//at the current pixel as well as the global currentParamsAllPixAtLevelConstMem which applies to all pixels at the
			//current level
			//retrieve the current paramsCurrentPixelAtLevel for the desired pixel

			currentStartMoveParamsPixelAtLevel currentStartMovePixelParams;

			//check the current "checkerboard" and retrieve the current parameters from the texture assumed to be bound to the
			//appropriate values
			currentStartMovePixelParams = tex1Dfetch(currentPixParamsTexCurrentCheckerboard, retrieveIndexCurrentPixel2DGrid(xValThread, yValThread,
					widthCheckerboard, heightCheckerboard));
			

			//now use the given parameters to retrieve the min and max movements
			float maxMovementLeft;
			float maxMovementRight;
			float maxMovementUp;
			float maxMovementDown;

			if (currentBeliefPropParamsConstMem.directionPosMovement == POS_X_MOTION_RIGHT_POS_Y_MOTION_DOWN)
			{
				maxMovementLeft = -1.0f*(currentStartMovePixelParams.x);
				maxMovementRight = (currentStartMovePixelParams.x + currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.currentMoveIncrementX);
				maxMovementUp = -1.0f*(currentStartMovePixelParams.y);
				maxMovementDown = (currentStartMovePixelParams.y + currentBeliefPropParamsConstMem.totalNumMovesYDir * currentBeliefPropParamsConstMem.currentMoveIncrementY);
			}
			else if (currentBeliefPropParamsConstMem.directionPosMovement == POS_X_MOTION_LEFT_POS_Y_MOTION_UP)
			{
				maxMovementLeft = (currentStartMovePixelParams.x + currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.currentMoveIncrementX);
				maxMovementRight = -1.0f*(currentStartMovePixelParams.x);
				maxMovementUp = (currentStartMovePixelParams.y + currentBeliefPropParamsConstMem.totalNumMovesYDir * currentBeliefPropParamsConstMem.currentMoveIncrementY);
				maxMovementDown = -1.0f*(currentStartMovePixelParams.y);
			}
			else
			{
				maxMovementLeft = 0.0f;
				maxMovementRight = 0.0f;
				maxMovementUp = 0.0f;
				maxMovementDown = 0.0f;
			}

			//set the current index of the movement to 0
			int movementXYVal = 0;

			if ( (((midXValImage - extentDataCostsSumAcross/2.0f) - maxMovementLeft) >= 0) && (((midXValImage + extentDataCostsSumAcross/2.0f) + maxMovementRight) < (currentBeliefPropParamsConstMem.widthImages)) &&
				(((midYValImage - extentDataCostsSumVertical/2.0f) - maxMovementUp) >= 0) && (((midYValImage + extentDataCostsSumVertical/2.0f) + maxMovementDown) < (currentBeliefPropParamsConstMem.heightImages)))
			{
				//set current movement to the min movement in the x and y directions and then increase by current move increment in the loop
				float currentXMove = currentStartMovePixelParams.x;
				float currentYMove = currentStartMovePixelParams.y;


				//go through entire range of movements in the x and y directions
				for (int numMoveYInRange = 0; numMoveYInRange < (currentBeliefPropParamsConstMem.totalNumMovesYDir); numMoveYInRange++)
				{
					//reset the current x movement to the minimum x movement
					currentXMove = currentStartMovePixelParams.x;

					for (int numMoveXInRange = 0; numMoveXInRange < (currentBeliefPropParamsConstMem.totalNumMovesXDir); numMoveXInRange++)
					{
						
						float currentPixelImage1;
						float currentPixelImage2;

						//use the thread indices for the index value
						indexVal = retrieveIndexInDataAndMessage(xValThread, yValThread, widthCheckerboard, heightCheckerboard, movementXYVal, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));


						//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
						//save the data cost for the current pixel on the current "checkerboard"
						//initialize each data cost to 0 then add each value within the current range of data values to add up
						dataCostsCurrentLevelCurrentCheckerboard[indexVal] = 0.0f;


						//loop through all the pixels in the current range where the data costs are being summed
						for (float xPixLocation = (midXValImage - extentDataCostsSumAcross/2.0f); xPixLocation <= (midXValImage + extentDataCostsSumAcross/2.0f + SMALL_VALUE); xPixLocation += 1.0f)
						{
							for (float yPixLocation = (midYValImage - extentDataCostsSumVertical/2.0f); yPixLocation <= (midYValImage + extentDataCostsSumVertical/2.0f + SMALL_VALUE); yPixLocation += 1.0f)
							{
								float currMinDataCost = INF_BP;

								//declare the variable for the x and y moves to be tested
								float xMoveTest;
								float yMoveTest;

								//declare the variables which declare the "range" to check around the current move and initialize it to 0
								float xMoveRange = 0.0f;
								float yMoveRange = 0.0f;

								#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
									//declare the variables for the bounds of the change in the min and max moves in the x/y range
									float xMoveChangeBounds = 0.0f;
									float yMoveChangeBounds = 0.0f;
								#endif //CLAMP_EDGE_MOVES_DATA_COST_SETTING


								//if current move increment is beyond a certain point, then check various possible movements
								//perform sampling-invarient data costs beyond a certain move increment
								if ((currentBeliefPropParamsConstMem.currentMoveIncrementX > (currentBeliefPropParamsConstMem.motionIncBotLevX)) || (currentBeliefPropParamsConstMem.currentMoveIncrementY > (currentBeliefPropParamsConstMem.motionIncBotLevY)))
								{
									//go through the rand in increments of 0.5f for each pixel and take the minimum cost
									//if y range is greater than 1, then check values "around" for sampling-invarient data costs
									if (currentBeliefPropParamsConstMem.totalNumMovesXDir > 1)
									{
										//xMoveRange = currentBeliefPropParamsConstMem.currentMoveIncrement / 2.0f;
										float numXMoves = ceil((currentBeliefPropParamsConstMem.currentMoveIncrementX / 2.0f) / currentBeliefPropParamsConstMem.motionIncBotLevX);
										xMoveRange = numXMoves * currentBeliefPropParamsConstMem.motionIncBotLevX;

										#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
											xMoveChangeBounds = currentBeliefPropParamsConstMem.currentMoveIncrementX / 2.0f;
										#endif //CLAMP_EDGE_MOVES_DATA_COST_SETTING
									}

									if (currentBeliefPropParamsConstMem.totalNumMovesYDir > 1)
									{
										//yMoveRange = currentBeliefPropParamsConstMem.currentMoveIncrement / 2.0f;
										float numYMoves = ceil((currentBeliefPropParamsConstMem.currentMoveIncrementY / 2.0f) / currentBeliefPropParamsConstMem.motionIncBotLevY);
										yMoveRange = numYMoves * currentBeliefPropParamsConstMem.motionIncBotLevY;
										
										#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
											yMoveChangeBounds = currentBeliefPropParamsConstMem.currentMoveIncrementY / 2.0f;
										#endif //CLAMP_EDGE_MOVES_DATA_COST_SETTING
									}
								}

								//go through and retrieve the data costs of each move in the given range
								for (yMoveTest = (-1.0f * yMoveRange); yMoveTest <= (yMoveRange + SMALL_VALUE); yMoveTest += currentBeliefPropParamsConstMem.motionIncBotLevY)
								{
									for (xMoveTest = (-1.0f * xMoveRange); xMoveTest <= (xMoveRange + SMALL_VALUE); xMoveTest += currentBeliefPropParamsConstMem.motionIncBotLevX)
									{
										//declare the variables for the current x and y moves and initialize to the current move (clamped if that's the setting...)
										#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == DONT_CLAMP_EDGE_MOVES_DATA_COST)
											float currXMove = currentXMove + xMoveTest;
											float currYMove = currentYMove + yMoveTest;
										#elif (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
											float currXMove = currentXMove + max(-1.0f * xMoveChangeBounds, min(xMoveTest, xMoveChangeBounds));
											float currYMove = currentYMove + max(-1.0f * yMoveChangeBounds, min(yMoveTest, yMoveChangeBounds));
										#endif//CLAMP_EDGE_MOVES_DATA_COST_SETTING


										if (currentBeliefPropParamsConstMem.directionPosMovement == POS_X_MOTION_RIGHT_POS_Y_MOTION_DOWN)
										{
											//texture access is using normalized coordinate system
											currentPixelImage1 = tex2D(image1PixelsTexture, (((float)xPixLocation) / (float)widthFullImage), (((float)yPixLocation) / (float)heightFullImage));
											currentPixelImage2 = tex2D(image2PixelsTexture, (((float)(xPixLocation + currXMove))) / ((float)widthFullImage), (((float)(yPixLocation + currYMove))) / ((float)heightFullImage));
										}
										else if (currentBeliefPropParamsConstMem.directionPosMovement == POS_X_MOTION_LEFT_POS_Y_MOTION_UP)
										{
											currentPixelImage1 = tex2D(image1PixelsTexture, (((float)xPixLocation) / (float)widthFullImage), (((float)yPixLocation) / (float)heightFullImage));
											currentPixelImage2 = tex2D(image2PixelsTexture, (((float)(xPixLocation - currXMove))) / ((float)widthFullImage), ((((float)(yPixLocation - currYMove))) / ((float)heightFullImage)));
										}
										else
										{
											currentPixelImage1 = 0.0f;
											currentPixelImage2 = 0.0f;
										}

										currMinDataCost = min(currMinDataCost, currentBeliefPropParamsConstMem.dataCostWeight * min(abs(currentPixelImage1 - currentPixelImage2), currentBeliefPropParamsConstMem.dataCostCap));
									}
								}

								//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
								//save the data cost for the current pixel on the current "checkerboard"
								dataCostsCurrentLevelCurrentCheckerboard[indexVal] += currMinDataCost;
							}
						}

						movementXYVal++;

						//increment the x-movement by current movement increment
						currentXMove += currentBeliefPropParamsConstMem.currentMoveIncrementX;
					}

					//increment the y-movement by current movement increment
					currentYMove += currentBeliefPropParamsConstMem.currentMoveIncrementY;
				}

			}
			else
			{
				//go through entire range of movements in the x and y directions and set data cost to default value
				for (int movementXYVal = 0; movementXYVal < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementXYVal++)
				{
					indexVal = retrieveIndexInDataAndMessage(xValThread, yValThread, widthCheckerboard, heightCheckerboard, movementXYVal, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));

					//set each data cost to a default value if outside of range
					dataCostsCurrentLevelCurrentCheckerboard[indexVal] = currentBeliefPropParamsConstMem.dataCostWeight * DEFAULT_DATA_COST_VALUE;
				}
			}
		}
		else
		{
			//go through entire range of movements in the x and y directions and set data cost to default value
			for (int movementXYVal = 0; movementXYVal < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementXYVal++)
			{
				indexVal = retrieveIndexInDataAndMessage(xValThread, yValThread, widthCheckerboard, heightCheckerboard, movementXYVal, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));

				//set each data cost to a default value if outside of range
				dataCostsCurrentLevelCurrentCheckerboard[indexVal] = currentBeliefPropParamsConstMem.dataCostWeight * DEFAULT_DATA_COST_VALUE;
			}
		}
	}
}


//initialize the "data cost" for each possible movement at a particular level for the desired set of motions within one of the
//checkerboards
//the image data is stored in the CUDA arrays image1PixelsTextureAtLevel and 
//image2PixelsTextureAtLevel
//this method adds together all the results in a given range according to the current level
//in the "bottom" level, it is a single pixel, then the width/height are doubled at each level
//assuming that the input is full images
__global__ void initialDataAndEstMoveCostsAtCurrentLevelAddCostsCheckerboardSampInvarient(float* dataCostsCurrentLevelCurrentCheckerboard,
																	int widthImageAtLevel, int heightImageAtLevel,
																	checkerboardPortionEnum checkerboardPart, float* inputEstMoveXVals, float* inputEstMoveYVals)
{
	int xValThread = blockIdx.x * blockDim.x + threadIdx.x;
	int yValThread = blockIdx.y * blockDim.y + threadIdx.y;

	//retrieve the width and height of the checkerboard by dividing the width
	//and setting the height of the checkerboard to the image height
	int widthCheckerboard = widthImageAtLevel / 2;
	int heightCheckerboard = heightImageAtLevel;


	//check to make sure that x and y are within the "checkerboard"
	if (withinImageBounds(xValThread, yValThread, widthCheckerboard, heightCheckerboard))
	{

		//used to adjust the pixel value based on the checkerboard portion
		int checkerboardPartAdjustment;

		//retrieve the index of the current pixel using the thread and current checkerboard portion
		if (checkerboardPart == CHECKERBOARD_PART_1_ENUM)
		{
			checkerboardPartAdjustment = (yValThread%2);
		}
		else if (checkerboardPart == CHECKERBOARD_PART_2_ENUM)
		{
			checkerboardPartAdjustment = ((yValThread+1)%2);
		}

		//declare and define the x and y indices of the current pixel within the "level image"
		int xIndexPixel = xValThread*2 + checkerboardPartAdjustment;
		int yIndexPixel = yValThread;


		//retrieve the width and the height of the full image via the BPSettings in constant memory
		int widthFullImage = currentBeliefPropParamsConstMem.widthImages;
		int heightFullImage = currentBeliefPropParamsConstMem.heightImages;

		//retrieve the sampling at the current level using the width of the full images and the width of the images at the current level
		//TODO: look into using "smaller" versions of the images (like mipmapping...)
		float samplingXDir = (float)widthFullImage / (float)widthImageAtLevel;
		float samplingYDir = (float)heightFullImage / (float)heightImageAtLevel;

		//retrieve the middle x-value and y-value in the image using the thread x and y values and also the sampling in each direction
		float midXValImage = ((float)xIndexPixel) * (samplingXDir) + samplingXDir / 2.0f;
		float midYValImage = ((float)yIndexPixel) * (samplingYDir) + samplingYDir / 2.0f;

		//use the sampling increment in the x and y directions for the extent of the data costs from the midpoint
		//subtract each by 1 since going from mid-point to mid-point
		float extentDataCostsSumAcross = floor(samplingXDir + 0.5f) - 1.0f;
		float extentDataCostsSumVertical = floor(samplingYDir + 0.5f) - 1.0f;

		//retrieve the movement estimation in the x and y directions...
		float moveEstXDir = inputEstMoveXVals[(yIndexPixel*widthFullImage) + xIndexPixel];
		float moveEstYDir = inputEstMoveYVals[(yIndexPixel*widthFullImage) + xIndexPixel];

		int indexVal;

		if (withinImageBounds(midXValImage, midYValImage, widthFullImage, heightFullImage))
		{
			//retrieve the current min and max movements in the x and y directions using the current paramsCurrentPixelAtLevel
			//at the current pixel as well as the global currentParamsAllPixAtLevelConstMem which applies to all pixels at the
			//current level
			//retrieve the current paramsCurrentPixelAtLevel for the desired pixel

			currentStartMoveParamsPixelAtLevel currentStartMovePixelParams;

			//check the current "checkerboard" and retrieve the current parameters from the texture assumed to be bound to the
			//appropriate values
			currentStartMovePixelParams = tex1Dfetch(currentPixParamsTexCurrentCheckerboard, retrieveIndexCurrentPixel2DGrid(xValThread, yValThread,
					widthCheckerboard, heightCheckerboard));
			

			//now use the given parameters to retrieve the min and max movements
			float maxMovementLeft;
			float maxMovementRight;
			float maxMovementUp;
			float maxMovementDown;

			if (currentBeliefPropParamsConstMem.directionPosMovement == POS_X_MOTION_RIGHT_POS_Y_MOTION_DOWN)
			{
				maxMovementLeft = -1.0f*(currentStartMovePixelParams.x);
				maxMovementRight = (currentStartMovePixelParams.x + currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.currentMoveIncrementX);
				maxMovementUp = -1.0f*(currentStartMovePixelParams.y);
				maxMovementDown = (currentStartMovePixelParams.y + currentBeliefPropParamsConstMem.totalNumMovesYDir * currentBeliefPropParamsConstMem.currentMoveIncrementY);
			}
			else if (currentBeliefPropParamsConstMem.directionPosMovement == POS_X_MOTION_LEFT_POS_Y_MOTION_UP)
			{
				maxMovementLeft = (currentStartMovePixelParams.x + currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.currentMoveIncrementX);
				maxMovementRight = -1.0f*(currentStartMovePixelParams.x);
				maxMovementUp = (currentStartMovePixelParams.y + currentBeliefPropParamsConstMem.totalNumMovesYDir * currentBeliefPropParamsConstMem.currentMoveIncrementY);
				maxMovementDown = -1.0f*(currentStartMovePixelParams.y);
			}
			else
			{
				maxMovementLeft = 0.0f;
				maxMovementRight = 0.0f;
				maxMovementUp = 0.0f;
				maxMovementDown = 0.0f;
			}

			//set the current index of the movement to 0
			int movementXYVal = 0;

			if ( (((midXValImage - extentDataCostsSumAcross/2.0f) - maxMovementLeft) >= 0) && (((midXValImage + extentDataCostsSumAcross/2.0f) + maxMovementRight) < (currentBeliefPropParamsConstMem.widthImages)) &&
				(((midYValImage - extentDataCostsSumVertical/2.0f) - maxMovementUp) >= 0) && (((midYValImage + extentDataCostsSumVertical/2.0f) + maxMovementDown) < (currentBeliefPropParamsConstMem.heightImages)))
			{
				//set current movement to the min movement in the x and y directions and then increase by current move increment in the loop
				float currentXMove = currentStartMovePixelParams.x;
				float currentYMove = currentStartMovePixelParams.y;


				//go through entire range of movements in the x and y directions
				for (int numMoveYInRange = 0; numMoveYInRange < (currentBeliefPropParamsConstMem.totalNumMovesYDir); numMoveYInRange++)
				{
					//reset the current x movement to the minimum x movement
					currentXMove = currentStartMovePixelParams.x;

					for (int numMoveXInRange = 0; numMoveXInRange < (currentBeliefPropParamsConstMem.totalNumMovesXDir); numMoveXInRange++)
					{
						
						float currentPixelImage1;
						float currentPixelImage2;

						//use the thread indices for the index value
						indexVal = retrieveIndexInDataAndMessage(xValThread, yValThread, widthCheckerboard, heightCheckerboard, movementXYVal, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));


						//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
						//save the data cost for the current pixel on the current "checkerboard"
						//initialize each data cost to 0 then add each value within the current range of data values to add up
						dataCostsCurrentLevelCurrentCheckerboard[indexVal] = 0.0f;


						//loop through all the pixels in the current range where the data costs are being summed
						for (float xPixLocation = (midXValImage - extentDataCostsSumAcross/2.0f); xPixLocation <= (midXValImage + extentDataCostsSumAcross/2.0f + SMALL_VALUE); xPixLocation += 1.0f)
						{
							for (float yPixLocation = (midYValImage - extentDataCostsSumVertical/2.0f); yPixLocation <= (midYValImage + extentDataCostsSumVertical/2.0f + SMALL_VALUE); yPixLocation += 1.0f)
							{
								float currMinDataAndEstMoveCost = INF_BP;

								//declare the variable for the x and y moves to be tested
								float xMoveTest;
								float yMoveTest;

								//declare the variables which declare the "range" to check around the current move and initialize it to 0
								float xMoveRange = 0.0f;
								float yMoveRange = 0.0f;

								#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
									//declare the variables for the bounds of the change in the min and max moves in the x/y range
									float xMoveChangeBounds = 0.0f;
									float yMoveChangeBounds = 0.0f;
								#endif //CLAMP_EDGE_MOVES_DATA_COST_SETTING


								//if current move increment is beyond a certain point, then check various possible movements
								//perform sampling-invarient data costs beyond a certain move increment
								if ((currentBeliefPropParamsConstMem.currentMoveIncrementX > (currentBeliefPropParamsConstMem.motionIncBotLevX)) || (currentBeliefPropParamsConstMem.currentMoveIncrementY > (currentBeliefPropParamsConstMem.motionIncBotLevY)))
								{
									//go through the rand in increments of 0.5f for each pixel and take the minimum cost
									//if y range is greater than 1, then check values "around" for sampling-invarient data costs
									if (currentBeliefPropParamsConstMem.totalNumMovesXDir > 1)
									{
										//xMoveRange = currentBeliefPropParamsConstMem.currentMoveIncrement / 2.0f;
										float numXMoves = ceil((currentBeliefPropParamsConstMem.currentMoveIncrementX / 2.0f) / currentBeliefPropParamsConstMem.motionIncBotLevX);
										xMoveRange = numXMoves * currentBeliefPropParamsConstMem.motionIncBotLevX;

										#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
											xMoveChangeBounds = currentBeliefPropParamsConstMem.currentMoveIncrementX / 2.0f;
										#endif //CLAMP_EDGE_MOVES_DATA_COST_SETTING
									}

									if (currentBeliefPropParamsConstMem.totalNumMovesYDir > 1)
									{
										//yMoveRange = currentBeliefPropParamsConstMem.currentMoveIncrement / 2.0f;
										float numYMoves = ceil((currentBeliefPropParamsConstMem.currentMoveIncrementY / 2.0f) / currentBeliefPropParamsConstMem.motionIncBotLevY);
										yMoveRange = numYMoves * currentBeliefPropParamsConstMem.motionIncBotLevY;
										
										#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
											yMoveChangeBounds = currentBeliefPropParamsConstMem.currentMoveIncrementY / 2.0f;
										#endif //CLAMP_EDGE_MOVES_DATA_COST_SETTING
									}
								}

								//go through and retrieve the data costs of each move in the given range
								for (yMoveTest = (-1.0f * yMoveRange); yMoveTest <= (yMoveRange + SMALL_VALUE); yMoveTest += currentBeliefPropParamsConstMem.motionIncBotLevY)
								{
									for (xMoveTest = (-1.0f * xMoveRange); xMoveTest <= (xMoveRange + SMALL_VALUE); xMoveTest += currentBeliefPropParamsConstMem.motionIncBotLevX)
									{
										//declare the variables for the current x and y moves and initialize to the current move (clamped if that's the setting...)
										#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == DONT_CLAMP_EDGE_MOVES_DATA_COST)
											float currXMove = currentXMove + xMoveTest;
											float currYMove = currentYMove + yMoveTest;
										#elif (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
											float currXMove = currentXMove + max(-1.0f * xMoveChangeBounds, min(xMoveTest, xMoveChangeBounds));
											float currYMove = currentYMove + max(-1.0f * yMoveChangeBounds, min(yMoveTest, yMoveChangeBounds));
										#endif//CLAMP_EDGE_MOVES_DATA_COST_SETTING


										if (currentBeliefPropParamsConstMem.directionPosMovement == POS_X_MOTION_RIGHT_POS_Y_MOTION_DOWN)
										{
											//texture access is using normalized coordinate system
											currentPixelImage1 = tex2D(image1PixelsTexture, (((float)xPixLocation) / (float)widthFullImage), (((float)yPixLocation) / (float)heightFullImage));
											currentPixelImage2 = tex2D(image2PixelsTexture, (((float)(xPixLocation + currXMove))) / ((float)widthFullImage), (((float)(yPixLocation + currYMove))) / ((float)heightFullImage));
										}
										else if (currentBeliefPropParamsConstMem.directionPosMovement == POS_X_MOTION_LEFT_POS_Y_MOTION_UP)
										{
											currentPixelImage1 = tex2D(image1PixelsTexture, (((float)xPixLocation) / (float)widthFullImage), (((float)yPixLocation) / (float)heightFullImage));
											currentPixelImage2 = tex2D(image2PixelsTexture, (((float)(xPixLocation - currXMove))) / ((float)widthFullImage), ((((float)(yPixLocation - currYMove))) / ((float)heightFullImage)));
										}
										else
										{
											currentPixelImage1 = 0.0f;
											currentPixelImage2 = 0.0f;
										}

										float currEstMoveCost;

										//if the estimated move in the x or y direction is less than -500, the cost is 0...
										if ((moveEstXDir < -500.0f) || (moveEstYDir < -500.0f)) 
										{
											currEstMoveCost = 0.0f;
										}
										else
										{
											currEstMoveCost = currentBeliefPropParamsConstMem.estMovementCostWeight * min(getEuclideanDist(currXMove, currYMove, moveEstXDir, moveEstYDir), currentBeliefPropParamsConstMem.estMovementCostCap);
										}
			
										float currDataCost = currentBeliefPropParamsConstMem.dataCostWeight * min(abs(currentPixelImage1 - currentPixelImage2), currentBeliefPropParamsConstMem.dataCostCap);

										currMinDataAndEstMoveCost = min(currMinDataAndEstMoveCost, currEstMoveCost + currDataCost);
									}
								}

								//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
								//save the data cost for the current pixel on the current "checkerboard"
								dataCostsCurrentLevelCurrentCheckerboard[indexVal] += currMinDataAndEstMoveCost;
							}
						}

						movementXYVal++;

						//increment the x-movement by current movement increment
						currentXMove += currentBeliefPropParamsConstMem.currentMoveIncrementX;
					}

					//increment the y-movement by current movement increment
					currentYMove += currentBeliefPropParamsConstMem.currentMoveIncrementY;
				}

			}
			else
			{
				//go through entire range of movements in the x and y directions and set data cost to default value
				for (int movementXYVal = 0; movementXYVal < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementXYVal++)
				{
					indexVal = retrieveIndexInDataAndMessage(xValThread, yValThread, widthCheckerboard, heightCheckerboard, movementXYVal, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));

					//set each data cost to a default value if outside of range
					dataCostsCurrentLevelCurrentCheckerboard[indexVal] = currentBeliefPropParamsConstMem.dataCostWeight * DEFAULT_DATA_COST_VALUE;
				}
			}
		}
		else
		{
			//go through entire range of movements in the x and y directions and set data cost to default value
			for (int movementXYVal = 0; movementXYVal < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementXYVal++)
			{
				indexVal = retrieveIndexInDataAndMessage(xValThread, yValThread, widthCheckerboard, heightCheckerboard, movementXYVal, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));

				//set each data cost to a default value if outside of range
				dataCostsCurrentLevelCurrentCheckerboard[indexVal] = currentBeliefPropParamsConstMem.dataCostWeight * DEFAULT_DATA_COST_VALUE;
			}
		}
	}
}


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
											currentStartMoveParamsPixelAtLevel* currentPixParamsCurrentCheckerboard)
{
	int xValThread = blockIdx.x * blockDim.x + threadIdx.x;
	int yValThread = blockIdx.y * blockDim.y + threadIdx.y;

	//retrieve the width and height of the checkerboard by dividing the width
	//and setting the height of the checkerboard to the image height
	int widthCheckerboard = widthImageAtLevel / 2;
	int heightCheckerboard = heightImageAtLevel;


	//check to make sure that x and y are within the "checkerboard"
	if (withinImageBounds(xValThread, yValThread, widthCheckerboard, heightCheckerboard))
	{

		//used to adjust the pixel value based on the checkerboard portion
		int checkerboardPartAdjustment;

		//retrieve the index of the current pixel using the thread and current checkerboard portion
		if (checkerboardPart == CHECKERBOARD_PART_1_ENUM)
		{
			checkerboardPartAdjustment = (yValThread%2);
		}
		else if (checkerboardPart == CHECKERBOARD_PART_2_ENUM)
		{
			checkerboardPartAdjustment = ((yValThread+1)%2);
		}

		//declare and define the x and y indices of the current pixel within the "level image"
		int xIndexPixel = xValThread*2 + checkerboardPartAdjustment;
		int yIndexPixel = yValThread;


		//retrieve the width and the height of the full image via the BPSettings in constant memory
		int widthFullImage = currentBeliefPropParamsConstMem.widthImages;
		int heightFullImage = currentBeliefPropParamsConstMem.heightImages;

		//retrieve the sampling at the current level using the width of the full images and the width of the images at the current level
		//TODO: look into using "smaller" versions of the images (like mipmapping...)
		float samplingXDir = (float)widthFullImage / (float)widthImageAtLevel;
		float samplingYDir = (float)heightFullImage / (float)heightImageAtLevel;

		//retrieve the middle x-value and y-value in the image using the thread x and y values and also the sampling in each direction
		float midXValImage = ((float)xIndexPixel) * (samplingXDir) + samplingXDir / 2.0f;
		float midYValImage = ((float)yIndexPixel) * (samplingYDir) + samplingYDir / 2.0f;

		//use the sampling increment in the x and y directions for the extent of the data costs from the midpoint
		//subtract each by 1 since going from mid-point to mid-point
		float extentDataCostsSumAcross = floor(samplingXDir + 0.5f) - 1.0f;
		float extentDataCostsSumVertical = floor(samplingYDir + 0.5f) - 1.0f;

		int indexVal;

		if (withinImageBounds(midXValImage, midYValImage, widthFullImage, heightFullImage))
		{
			//retrieve the current min and max movements in the x and y directions using the current paramsCurrentPixelAtLevel
			//at the current pixel as well as the global currentParamsAllPixAtLevelConstMem which applies to all pixels at the
			//current level
			//retrieve the current paramsCurrentPixelAtLevel for the desired pixel

			currentStartMoveParamsPixelAtLevel currentStartMovePixelParams;

			//check the current "checkerboard" and retrieve the current parameters from the texture assumed to be bound to the
			//appropriate values
			currentStartMovePixelParams = currentPixParamsCurrentCheckerboard[retrieveIndexCurrentPixel2DGrid(xValThread, yValThread,
					widthCheckerboard, heightCheckerboard)];
			

			//now use the given parameters to retrieve the min and max movements
			float maxMovementLeft;
			float maxMovementRight;
			float maxMovementUp;
			float maxMovementDown;

			if (currentBeliefPropParamsConstMem.directionPosMovement == POS_X_MOTION_RIGHT_POS_Y_MOTION_DOWN)
			{
				maxMovementLeft = -1.0f*(currentStartMovePixelParams.x);
				maxMovementRight = (currentStartMovePixelParams.x + currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.currentMoveIncrementX);
				maxMovementUp = -1.0f*(currentStartMovePixelParams.y);
				maxMovementDown = (currentStartMovePixelParams.y + currentBeliefPropParamsConstMem.totalNumMovesYDir * currentBeliefPropParamsConstMem.currentMoveIncrementY);
			}
			else if (currentBeliefPropParamsConstMem.directionPosMovement == POS_X_MOTION_LEFT_POS_Y_MOTION_UP)
			{
				maxMovementLeft = (currentStartMovePixelParams.x + currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.currentMoveIncrementX);
				maxMovementRight = -1.0f*(currentStartMovePixelParams.x);
				maxMovementUp = (currentStartMovePixelParams.y + currentBeliefPropParamsConstMem.totalNumMovesYDir * currentBeliefPropParamsConstMem.currentMoveIncrementY);
				maxMovementDown = -1.0f*(currentStartMovePixelParams.y);
			}
			else
			{
				maxMovementLeft = 0.0f;
				maxMovementRight = 0.0f;
				maxMovementUp = 0.0f;
				maxMovementDown = 0.0f;
			}

			//set the current index of the movement to 0
			int movementXYVal = 0;

			if ( (((midXValImage - extentDataCostsSumAcross/2.0f) - maxMovementLeft) >= 0) && (((midXValImage + extentDataCostsSumAcross/2.0f) + maxMovementRight) < (currentBeliefPropParamsConstMem.widthImages)) &&
				(((midYValImage - extentDataCostsSumVertical/2.0f) - maxMovementUp) >= 0) && (((midYValImage + extentDataCostsSumVertical/2.0f) + maxMovementDown) < (currentBeliefPropParamsConstMem.heightImages)))
			{
				//set current movement to the min movement in the x and y directions and then increase by current move increment in the loop
				float currentXMove = currentStartMovePixelParams.x;
				float currentYMove = currentStartMovePixelParams.y;

				//go through entire range of movements in the x and y directions
				for (int numMoveYInRange = 0; numMoveYInRange < (currentBeliefPropParamsConstMem.totalNumMovesYDir); numMoveYInRange++)
				{
					//reset the current x movement to the minimum x movement
					currentXMove = currentStartMovePixelParams.x;

					for (int numMoveXInRange = 0; numMoveXInRange < (currentBeliefPropParamsConstMem.totalNumMovesXDir); numMoveXInRange++)
					{
						float currentPixelImage1;
						float currentPixelImage2;

						//use the thread indices for the index value
						indexVal = retrieveIndexInDataAndMessage(xValThread, yValThread, widthCheckerboard, heightCheckerboard, movementXYVal, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));


						//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
						//save the data cost for the current pixel on the current "checkerboard"
						//initialize each data cost to 0 then add each value within the current range of data values to add up
						dataCostsCurrentLevelCurrentCheckerboard[indexVal] = 0.0f;


						//loop through all the pixels in the current range where the data costs are being summed
						for (float xPixLocation = (midXValImage - extentDataCostsSumAcross/2.0f); xPixLocation <= (midXValImage + extentDataCostsSumAcross/2.0f + SMALL_VALUE); xPixLocation += 1.0f)
						{
							for (float yPixLocation = (midYValImage - extentDataCostsSumVertical/2.0f); yPixLocation <= (midYValImage + extentDataCostsSumVertical/2.0f + SMALL_VALUE); yPixLocation += 1.0f)
							{
								float currMinDataCost = INF_BP;

								//declare the variable for the x and y moves to be tested
								float xMoveTest;
								float yMoveTest;

								//declare the variables which declare the "range" to check around the current move and initialize it to 0
								float xMoveRange = 0.0f;
								float yMoveRange = 0.0f;

								#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
									//declare the variables for the bounds of the change in the min and max moves in the x/y range
									float xMoveChangeBounds = 0.0f;
									float yMoveChangeBounds = 0.0f;
								#endif //CLAMP_EDGE_MOVES_DATA_COST_SETTING


								//if current move increment is beyond a certain point, then check various possible movements
								//perform sampling-invarient data costs beyond a certain move increment
								if ((currentBeliefPropParamsConstMem.currentMoveIncrementX > (currentBeliefPropParamsConstMem.motionIncBotLevX)) || (currentBeliefPropParamsConstMem.currentMoveIncrementY > (currentBeliefPropParamsConstMem.motionIncBotLevY)))
								{
									//go through the rand in increments of 0.5f for each pixel and take the minimum cost
									//if y range is greater than 1, then check values "around" for sampling-invarient data costs
									if (currentBeliefPropParamsConstMem.totalNumMovesXDir > 1)
									{
										//xMoveRange = currentBeliefPropParamsConstMem.currentMoveIncrement / 2.0f;
										float numXMoves = ceil((currentBeliefPropParamsConstMem.currentMoveIncrementX / 2.0f) / currentBeliefPropParamsConstMem.motionIncBotLevX);
										xMoveRange = numXMoves * currentBeliefPropParamsConstMem.motionIncBotLevX;

										#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
											xMoveChangeBounds = currentBeliefPropParamsConstMem.currentMoveIncrementX / 2.0f;
										#endif //CLAMP_EDGE_MOVES_DATA_COST_SETTING
									}

									if (currentBeliefPropParamsConstMem.totalNumMovesYDir > 1)
									{
										//yMoveRange = currentBeliefPropParamsConstMem.currentMoveIncrement / 2.0f;
										float numYMoves = ceil((currentBeliefPropParamsConstMem.currentMoveIncrementY / 2.0f) / currentBeliefPropParamsConstMem.motionIncBotLevY);
										yMoveRange = numYMoves * currentBeliefPropParamsConstMem.motionIncBotLevY;
										
										#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
											yMoveChangeBounds = currentBeliefPropParamsConstMem.currentMoveIncrementY / 2.0f;
										#endif //CLAMP_EDGE_MOVES_DATA_COST_SETTING
									}
								}

								//go through and retrieve the data costs of each move in the given range
								for (yMoveTest = (-1.0f * yMoveRange); yMoveTest <= (yMoveRange + SMALL_VALUE); yMoveTest += currentBeliefPropParamsConstMem.motionIncBotLevY)
								{
									for (xMoveTest = (-1.0f * xMoveRange); xMoveTest <= (xMoveRange + SMALL_VALUE); xMoveTest += currentBeliefPropParamsConstMem.motionIncBotLevX)
									{
										//declare the variables for the current x and y moves and initialize to the current move (clamped if that's the setting...)
										#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == DONT_CLAMP_EDGE_MOVES_DATA_COST)
											float currXMove = currentXMove + xMoveTest;
											float currYMove = currentYMove + yMoveTest;
										#elif (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
											float currXMove = currentXMove + max(-1.0f * xMoveChangeBounds, min(xMoveTest, xMoveChangeBounds));
											float currYMove = currentYMove + max(-1.0f * yMoveChangeBounds, min(yMoveTest, yMoveChangeBounds));
										#endif//CLAMP_EDGE_MOVES_DATA_COST_SETTING


										if (currentBeliefPropParamsConstMem.directionPosMovement == POS_X_MOTION_RIGHT_POS_Y_MOTION_DOWN)
										{
											//texture access is using normalized coordinate system
											currentPixelImage1 = tex2D(image1PixelsTexture, (((float)xPixLocation) / (float)widthFullImage), (((float)yPixLocation) / (float)heightFullImage));
											currentPixelImage2 = tex2D(image2PixelsTexture, (((float)(xPixLocation + currXMove))) / ((float)widthFullImage), (((float)(yPixLocation + currYMove))) / ((float)heightFullImage));
										}
										else if (currentBeliefPropParamsConstMem.directionPosMovement == POS_X_MOTION_LEFT_POS_Y_MOTION_UP)
										{
											currentPixelImage1 = tex2D(image1PixelsTexture, (((float)xPixLocation) / (float)widthFullImage), (((float)yPixLocation) / (float)heightFullImage));
											currentPixelImage2 = tex2D(image2PixelsTexture, (((float)(xPixLocation - currXMove))) / ((float)widthFullImage), ((((float)(yPixLocation - currYMove))) / ((float)heightFullImage)));
										}
										else
										{
											currentPixelImage1 = 0.0f;
											currentPixelImage2 = 0.0f;
										}

										currMinDataCost = min(currMinDataCost, currentBeliefPropParamsConstMem.dataCostWeight * min(abs(currentPixelImage1 - currentPixelImage2), currentBeliefPropParamsConstMem.dataCostCap));
									}
								}

								//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
								//save the data cost for the current pixel on the current "checkerboard"
								dataCostsCurrentLevelCurrentCheckerboard[indexVal] += currMinDataCost;
							}
						}

						movementXYVal++;

						//increment the x-movement by current movement increment
						currentXMove += currentBeliefPropParamsConstMem.currentMoveIncrementX;
					}

					//increment the y-movement by current movement increment
					currentYMove += currentBeliefPropParamsConstMem.currentMoveIncrementY;
				}

			}
			else
			{
				//go through entire range of movements in the x and y directions and set data cost to default value
				for (int movementXYVal = 0; movementXYVal < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementXYVal++)
				{
					indexVal = retrieveIndexInDataAndMessage(xValThread, yValThread, widthCheckerboard, heightCheckerboard, movementXYVal, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));

					//set each data cost to a default value if outside of range
					dataCostsCurrentLevelCurrentCheckerboard[indexVal] = currentBeliefPropParamsConstMem.dataCostWeight * DEFAULT_DATA_COST_VALUE;
				}
			}
		}
		else
		{
			//go through entire range of movements in the x and y directions and set data cost to default value
			for (int movementXYVal = 0; movementXYVal < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementXYVal++)
			{
				indexVal = retrieveIndexInDataAndMessage(xValThread, yValThread, widthCheckerboard, heightCheckerboard, movementXYVal, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));

				//set each data cost to a default value if outside of range
				dataCostsCurrentLevelCurrentCheckerboard[indexVal] = currentBeliefPropParamsConstMem.dataCostWeight * DEFAULT_DATA_COST_VALUE;
			}
		}
	}
}


//determine the message value using the brute force method of checking every possibility
__device__ void dtMotionEstimationUseBruteForce(float f[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y])
{
	//stores the calculated message value for each distance
	float calculatedMessageVals[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];


	//retrieve the x and y positions for "current point" with respect to the increment of the move
	float xMoveCurrentPoint = 0.0f;
	float yMoveCurrentPoint = 0.0f;

	//go through every possible movement index and compute the message value
	for (int yMoveCurrentIndex = 0; yMoveCurrentIndex < currentBeliefPropParamsConstMem.totalNumMovesYDir; yMoveCurrentIndex++)
	{
		xMoveCurrentPoint = 0.0f;
		for (int xMoveCurrentIndex = 0; xMoveCurrentIndex < currentBeliefPropParamsConstMem.totalNumMovesXDir; xMoveCurrentIndex++)
		{
			int indexCurrentMovement = yMoveCurrentIndex * currentBeliefPropParamsConstMem.totalNumMovesXDir + xMoveCurrentIndex;
			float currentBestMessageVal = f[indexCurrentMovement];

			//retrieve the x and y positions for "move to" with respect to the increment
			float xMoveToPoint = 0.0f;
			float yMoveToPoint = 0.0f;

			//go through each move to index
			for (int yMoveToIndex = 0; yMoveToIndex < currentBeliefPropParamsConstMem.totalNumMovesYDir; yMoveToIndex++)
			{
				xMoveToPoint = 0.0f;
				for (int xMoveToIndex = 0; xMoveToIndex < currentBeliefPropParamsConstMem.totalNumMovesXDir; xMoveToIndex++)
				{
					int indexMovementTo = yMoveToIndex * currentBeliefPropParamsConstMem.totalNumMovesXDir + xMoveToIndex;
					float currentMessageValMovementTo = f[indexMovementTo];
					float currentDiscCost = computeDistanceBetweenMovements(xMoveCurrentPoint, yMoveCurrentPoint, xMoveToPoint, yMoveToPoint);

					if ((currentMessageValMovementTo + currentDiscCost) < currentBestMessageVal)
					{
						currentBestMessageVal = (currentMessageValMovementTo + currentDiscCost);
					}

					xMoveToPoint += currentBeliefPropParamsConstMem.currentMoveIncrementX;
				}
				yMoveToPoint += currentBeliefPropParamsConstMem.currentMoveIncrementY;
			}
			calculatedMessageVals[indexCurrentMovement] = currentBestMessageVal;
			xMoveCurrentPoint += currentBeliefPropParamsConstMem.currentMoveIncrementX;
		}
		yMoveCurrentPoint += currentBeliefPropParamsConstMem.currentMoveIncrementY;
	}

	//now copy the calculated message vals to f to return them
	for (int movementNum = 0; movementNum < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementNum++)
	{
		f[movementNum] = calculatedMessageVals[movementNum];
	}
}

//dt of 2d function; first minimize by row then by column; the manhatten distance is used for the distance function when the movement can
//be in both and x and y directions
__device__ void dtMotionEstimation(float f[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y])
{
	int currentIndex;

	//current "range of movement" is from [enhancedBpMotionSettingsConstMemMotionEstimationEnhancedBpMotion.minMoveX, enhancedBpMotionSettingsConstMemMotionEstimationEnhancedBpMotion.minMoveX + enhancedBpMotionSettingsConstMemMotionEstimationEnhancedBpMotion.numValsInRangeX] * [enhancedBpMotionSettingsConstMemMotionEstimationEnhancedBpMotion.minMoveY, enhancedBpMotionSettingsConstMemMotionEstimationEnhancedBpMotion.minMoveY + enhancedBpMotionSettingsConstMemMotionEstimationEnhancedBpMotion.numValsInRangeY] 

	//using the current movement increment for computation of discontinuity cost
	for (int row = 0; row < currentBeliefPropParamsConstMem.totalNumMovesYDir; row++)
	{
		for (int col = 1; col < (currentBeliefPropParamsConstMem.totalNumMovesXDir); col++) 
		{
			currentIndex = row*currentBeliefPropParamsConstMem.totalNumMovesXDir + col;
			float prev = f[currentIndex -1] + currentBeliefPropParamsConstMem.currentMoveIncrementX;
			if (prev < f[currentIndex])
				f[currentIndex] = prev;
		}
		for (int col = (currentBeliefPropParamsConstMem.totalNumMovesXDir) - 2; col >= 0; col--) 
		{
			currentIndex = row*currentBeliefPropParamsConstMem.totalNumMovesXDir + col;
			float prev = f[currentIndex+1] + currentBeliefPropParamsConstMem.currentMoveIncrementX;
			if (prev < f[currentIndex])
				f[currentIndex] = prev;
		}
	}

	for (int col = 0; col < currentBeliefPropParamsConstMem.totalNumMovesXDir; col++)
	{
		for (int row = 1; row < (currentBeliefPropParamsConstMem.totalNumMovesYDir); row++) 
		{
			currentIndex = row*currentBeliefPropParamsConstMem.totalNumMovesXDir + col;
			float prev = f[currentIndex - currentBeliefPropParamsConstMem.totalNumMovesXDir] + currentBeliefPropParamsConstMem.currentMoveIncrementY;
			if (prev < f[currentIndex])
				f[currentIndex] = prev;
		}
		for (int row = (currentBeliefPropParamsConstMem.totalNumMovesYDir) - 2; row >= 0; row--) 
		{
			currentIndex = row*currentBeliefPropParamsConstMem.totalNumMovesXDir + col;
			float prev = f[currentIndex + currentBeliefPropParamsConstMem.totalNumMovesXDir] + currentBeliefPropParamsConstMem.currentMoveIncrementY;
			if (prev < f[currentIndex])
				f[currentIndex] = prev;
		}
	}


	//now try diagonals going across if calculating using the euclidean distance approximation
	if (currentBeliefPropParamsConstMem.currDiscCostType == USE_APPROX_EUCLIDEAN_DIST_FOR_DISCONT)
	{
		for (int row = 1; row < currentBeliefPropParamsConstMem.totalNumMovesYDir; row++)
		{
			for (int col = 1; col < (currentBeliefPropParamsConstMem.totalNumMovesXDir - 1); col++) 
			{
				currentIndex = row*currentBeliefPropParamsConstMem.totalNumMovesXDir + col;

				//check upper left
				int indexDiagonal = (row - 1) * currentBeliefPropParamsConstMem.totalNumMovesXDir + (col - 1);
				float prev = f[indexDiagonal] + sqrt(2.0f*currentBeliefPropParamsConstMem.currentMoveIncrementX);
				if (prev < f[currentIndex])
					f[currentIndex] = prev;

				//check upper right
				indexDiagonal = (row - 1) * currentBeliefPropParamsConstMem.totalNumMovesXDir + (col + 1);
				prev = f[indexDiagonal] + sqrt(2.0f*currentBeliefPropParamsConstMem.currentMoveIncrementX);
				if (prev < f[currentIndex])
					f[currentIndex] = prev;
			}
		}

		for (int row = currentBeliefPropParamsConstMem.totalNumMovesYDir - 2; row >= 0; row--)
		{
			for (int col = 1; col < (currentBeliefPropParamsConstMem.totalNumMovesXDir - 1); col++) 
			{
				currentIndex = row*currentBeliefPropParamsConstMem.totalNumMovesXDir + col;

				//check lower left
				int indexDiagonal = (row + 1) * currentBeliefPropParamsConstMem.totalNumMovesXDir + (col - 1);
				float prev = f[indexDiagonal] + sqrt(2.0f*currentBeliefPropParamsConstMem.currentMoveIncrementX);
				if (prev < f[currentIndex])
					f[currentIndex] = prev;

				//check lower right
				indexDiagonal = (row + 1) * currentBeliefPropParamsConstMem.totalNumMovesXDir + (col + 1);
				prev = f[indexDiagonal] + sqrt(2.0f*currentBeliefPropParamsConstMem.currentMoveIncrementX);
				if (prev < f[currentIndex])
					f[currentIndex] = prev;
			}
		}
	}
}

// compute BP message to pass to neighboring pixels
__device__ void computeBpMsgMotionEstimation(float s1[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float s2[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], 
	 float s3[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float s4[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y],
	 float dst[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y])
{
	float val;

	// aggregate and find min
	float minimum = INF_BP;
	for (int value = 0; value < currentBeliefPropParamsConstMem.totalNumMovesXDir*currentBeliefPropParamsConstMem.totalNumMovesYDir; value++) 
	{
		dst[value] = s1[value] + s2[value] + s3[value] + s4[value];
		if (dst[value] < minimum)
			minimum = dst[value];
	}

	// dt
	//check discontinuity cost option to determine what method to use to determine message values
	if ((currentBeliefPropParamsConstMem.currDiscCostType == USE_MANHATTAN_DIST_FOR_DISCONT) || (currentBeliefPropParamsConstMem.currDiscCostType == USE_APPROX_EUCLIDEAN_DIST_FOR_DISCONT))
	{
		dtMotionEstimation(dst);
	}
	else if ((currentBeliefPropParamsConstMem.currDiscCostType == USE_EXACT_EUCLIDEAN_DIST_FOR_DISCONT) || (currentBeliefPropParamsConstMem.currDiscCostType == USE_MANHATTEN_DIST_USING_BRUTE_FORCE))
	{
		dtMotionEstimationUseBruteForce(dst);
	}

	// truncate 
	minimum += currentBeliefPropParamsConstMem.smoothnessCostCap;
	for (int value = 0; value < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); value++)
	{
		if (minimum < dst[value])
		{
			dst[value] = minimum;
		}
	}

	// normalize
	val = 0;
	for (int value = 0; value < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); value++) 
		val += dst[value];

	val /= ((float)(currentBeliefPropParamsConstMem.totalNumMovesXDir*currentBeliefPropParamsConstMem.totalNumMovesYDir));
	for (int value = 0; value < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); value++) 
		dst[value] -= val;
}


//retrieve the message value of the given pixel at the given disparity level, this value is estimated if the message value is not within the desired "range"
//assuming that pixel actually does exist for now...
__device__ float retMessageValPixAtDispLevel(currentStartMoveParamsPixelAtLevel currentStartMovePixelParams, float currentXMove, float currentYMove, float desiredMessVals[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y])
{
	//retrieve the  x and y indices
	float xIndex = (((currentXMove - currentStartMovePixelParams.x) / currentBeliefPropParamsConstMem.currentMoveIncrementX));
	float yIndex = (((currentYMove - currentStartMovePixelParams.y) / currentBeliefPropParamsConstMem.currentMoveIncrementY));

	//retrieve the x and y indices representing the values closest to the current x and y moves that are stored in the current set of message values
	float closestXIndex = min((float)currentBeliefPropParamsConstMem.totalNumMovesXDir - 1.0f , max(0.0f, xIndex));
	float closestYIndex = min((float)currentBeliefPropParamsConstMem.totalNumMovesYDir - 1.0f , max(0.0f, yIndex));

#if (MESSAGE_VAL_PASS_SCHEME == USE_IMMEDIATE_MESSAGE_VAL)
{
	return (desiredMessVals[retrieveIndexCurrentPixel2DGrid((int)closestXIndex, (int)closestYIndex, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir)]);
}
#else //otherwise, retrieve the ceiling and the floor of the values
	//retrieve the interpolated message value if it's between different values
	//retrieve the "floor" and "ceiling" of the x and y indices in order to retrieve the whole-number indices to use
	int xIndexFloor = (int)floor(closestXIndex);
	int yIndexFloor = (int)floor(closestYIndex);

	int xIndexCeil = (int)ceil(closestXIndex);
	int yIndexCeil = (int)ceil(closestYIndex);

	//retrieve the four points of the set of points used for interpolation in order to retrieve the desired value
	float floorXFloorYPoint = desiredMessVals[retrieveIndexCurrentPixel2DGrid(xIndexFloor, yIndexFloor, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir)];
	float floorXCeilYPoint = desiredMessVals[retrieveIndexCurrentPixel2DGrid(xIndexFloor, yIndexCeil, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir)];
	float ceilXFloorYPoint = desiredMessVals[retrieveIndexCurrentPixel2DGrid(xIndexCeil, yIndexFloor, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir)];
	float ceilXCeilYPoint = desiredMessVals[retrieveIndexCurrentPixel2DGrid(xIndexCeil, yIndexCeil, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir)];

	/*if ((xIndex < 0.0f) || (yIndex < 0.0f) || (xIndex > ((float)currentBeliefPropParamsConstMem.totalNumMovesXDir - 1.0f + 0.0001f)) || (yIndex > ((float)currentBeliefPropParamsConstMem.totalNumMovesYDir - 1.0f + 0.0001f)))
	{
		floorXFloorYPoint = 0.0f;//desiredMessVals[retrieveIndexCurrentPixel2DGrid(xIndexFloor, yIndexFloor, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir)];
		floorXCeilYPoint = 0.0f;//desiredMessVals[retrieveIndexCurrentPixel2DGrid(xIndexFloor, yIndexCeil, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir)];
		ceilXFloorYPoint = 0.0f;//desiredMessVals[retrieveIndexCurrentPixel2DGrid(xIndexCeil, yIndexFloor, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir)];
		ceilXCeilYPoint = 0.0f;//desiredMessVals[retrieveIndexCurrentPixel2DGrid(xIndexCeil, yIndexCeil, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir)];

	}*/
#endif //MESSAGE_VAL_PASS_SCHEME

#if (MESSAGE_VAL_PASS_SCHEME == USE_INTERPOLATED_MESSAGE_VAL)

	//declare the average "bottom" and "top" values
	float avgYFloor;
	float avgYCeil;

	//set ceil and floor x values equal if they are the same
	if (xIndexFloor == xIndexCeil)
	{
		avgYFloor = floorXFloorYPoint;
		avgYCeil = floorXCeilYPoint;
	}
	else
	{
		//now retrieve the weighted values in the y floor and ceiling regions using the x index
		avgYFloor = (ceil(closestXIndex) - closestXIndex)*floorXFloorYPoint + (closestXIndex - floor(closestXIndex))*ceilXFloorYPoint;
		avgYCeil = (ceil(closestXIndex) - closestXIndex)*floorXCeilYPoint + (closestXIndex - floor(closestXIndex))*ceilXCeilYPoint;
	}

	//now retrieve the weighted average the two computed averages to retrieve the final interpolated value
	float averageOfYFloorCeilAverages;

	if (yIndexFloor == yIndexCeil)
	{
		averageOfYFloorCeilAverages = avgYFloor;
	}
	else
	{
		averageOfYFloorCeilAverages = (ceil(closestYIndex) - closestYIndex)*avgYFloor + (closestYIndex - floor(closestYIndex))*avgYCeil;
	}

	//now return the final interpolated value
	return averageOfYFloorCeilAverages;

#elif (MESSAGE_VAL_PASS_SCHEME == USE_MIN_MESSAGE_VAL)

	return (min(floorXFloorYPoint, min(floorXCeilYPoint, min(ceilXFloorYPoint, ceilXCeilYPoint))));

	
#elif (MESSAGE_VAL_PASS_SCHEME == USE_MAX_MESSAGE_VAL)

	return (max(floorXFloorYPoint, max(floorXCeilYPoint, max(ceilXFloorYPoint, ceilXCeilYPoint))));


#endif //MESSAGE_VAL_PASS_SCHEME*/
}

//device portion of the kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the 
//"checkerboard" scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
__device__ void runBPIterationUsingCheckerboardUpdatesDevice(checkerboardMessagesDeviceStruct messageValsDeviceCurrCheckOut,
														float dataMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], 
														int widthLevelCheckerboardPart, int heightLevelCheckerboard, int checkerboardAdjustment,
														int xVal, int yVal, int paramsTexOffset)
{
	int indexWriteTo;

	//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
	//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
	//declare a "default" start movement
	currentStartMoveParamsPixelAtLevel defStartMove;
	defStartMove.x = 0.0f;
	defStartMove.y = 0.0f;


	//declare the starting movement for the current pixel and each neighbor
	currentStartMoveParamsPixelAtLevel currentPixMoveStart;

	currentStartMoveParamsPixelAtLevel currentUMoveStart;
	currentStartMoveParamsPixelAtLevel currentDMoveStart;
	currentStartMoveParamsPixelAtLevel currentLMoveStart;
	currentStartMoveParamsPixelAtLevel currentRMoveStart;


	//used to retrieve the starting x and y movements at the current pixel for the current checkerboard and also
	//each of the neighbors
	currentPixMoveStart = tex1Dfetch(currentPixParamsTexCurrentCheckerboard, retrieveIndexCurrentPixel2DGrid(xVal, yVal, widthLevelCheckerboardPart, heightLevelCheckerboard, paramsTexOffset));

	//each neighbor is on "opposite" checkerboard of one being updated
	//check to make sure desired neighbor exists in each case
	if (yVal < (heightLevelCheckerboard-1))
	{
		currentUMoveStart = tex1Dfetch(currentPixParamsTexNeighCheckerboard, retrieveIndexCurrentPixel2DGrid(xVal, (yVal+1), widthLevelCheckerboardPart, heightLevelCheckerboard, paramsTexOffset));
	}
	else
	{
		currentUMoveStart = defStartMove;

	}

	if (yVal > 0)
	{
		currentDMoveStart = tex1Dfetch(currentPixParamsTexNeighCheckerboard, retrieveIndexCurrentPixel2DGrid(xVal, (yVal-1), widthLevelCheckerboardPart, heightLevelCheckerboard, paramsTexOffset));
	}
	else
	{
		currentDMoveStart = defStartMove;
	}

	if ((xVal) < (widthLevelCheckerboardPart - 1))
	{
		currentLMoveStart = tex1Dfetch(currentPixParamsTexNeighCheckerboard, retrieveIndexCurrentPixel2DGrid(xVal + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevelCheckerboard, paramsTexOffset));
	}
	else
	{
		currentLMoveStart = defStartMove;
	}

	if ((xVal) > 0)
	{
		currentRMoveStart = tex1Dfetch(currentPixParamsTexNeighCheckerboard, retrieveIndexCurrentPixel2DGrid(xVal - 1 + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevelCheckerboard, paramsTexOffset));
	}
	else
	{
		currentRMoveStart = defStartMove;
	}

	float prevUMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
	float prevDMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
	float prevLMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
	float prevRMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];

	//go through all the movements in the current range
	for (int movementXY = 0; movementXY < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementXY++)
	{
		//make sure that neighboring pixel exists...need to check boundaries
		if (yVal < (heightLevelCheckerboard-1))
		{
			prevUMessage[movementXY] = tex1Dfetch(messageUTexCurrReadCheckerboard, retrieveIndexInDataAndMessage(xVal, (yVal+1), widthLevelCheckerboardPart, heightLevelCheckerboard, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir)));
		}
		else
		{
			prevUMessage[movementXY] = 0.0f;
		}

		if (yVal > 0)
		{
			prevDMessage[movementXY] = tex1Dfetch(messageDTexCurrReadCheckerboard, retrieveIndexInDataAndMessage(xVal, (yVal-1), widthLevelCheckerboardPart, heightLevelCheckerboard, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir)));
		}
		else
		{
			prevDMessage[movementXY] = 0.0f;
		}

		if ((xVal) < (widthLevelCheckerboardPart - 1))
		{
			prevLMessage[movementXY] = tex1Dfetch(messageLTexCurrReadCheckerboard, retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), yVal, widthLevelCheckerboardPart, heightLevelCheckerboard, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir)));
		}
		else
		{
			prevLMessage[movementXY] = 0.0f;
		}

		if ((xVal) > 0)
		{
			prevRMessage[movementXY] = tex1Dfetch(messageRTexCurrReadCheckerboard, retrieveIndexInDataAndMessage((xVal - 1 + checkerboardAdjustment), yVal, widthLevelCheckerboardPart, heightLevelCheckerboard, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir)));
		}
		else
		{
			prevRMessage[movementXY] = 0.0f;
		}
	}

	float currentUMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
	float currentDMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
	float currentLMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
	float currentRMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];

	//retrieve the message values corresponding to each possible movement within the range
	float prevUMessValRangeAdjusted[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
	float prevDMessValRangeAdjusted[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
	float prevLMessValRangeAdjusted[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
	float prevRMessValRangeAdjusted[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];

	//go through all the movements in the current range
	float currentXMove = currentPixMoveStart.x;
	float currentYMove = currentPixMoveStart.y;
	int currentMoveNum = 0;
	for (int yMoveNum = 0; yMoveNum < (currentBeliefPropParamsConstMem.totalNumMovesYDir); yMoveNum++)
	{
		currentXMove = currentPixMoveStart.x;
		for (int xMoveNum = 0; xMoveNum < (currentBeliefPropParamsConstMem.totalNumMovesXDir); xMoveNum++)
		{
			if (yVal < (heightLevelCheckerboard-1))
			{
				prevUMessValRangeAdjusted[currentMoveNum] = retMessageValPixAtDispLevel(currentUMoveStart, currentXMove, currentYMove, prevUMessage);
			}
			else
			{
				prevUMessValRangeAdjusted[currentMoveNum] = 0.0f;
			}

			if (yVal > 0)
			{
				prevDMessValRangeAdjusted[currentMoveNum] = retMessageValPixAtDispLevel(currentDMoveStart, currentXMove, currentYMove, prevDMessage);
			}
			else
			{
				prevDMessValRangeAdjusted[currentMoveNum] = 0.0f;
			}

			if ((xVal) < (widthLevelCheckerboardPart - 1))
			{
				prevLMessValRangeAdjusted[currentMoveNum] = retMessageValPixAtDispLevel(currentLMoveStart, currentXMove, currentYMove, prevLMessage);
			}
			else
			{
				prevLMessValRangeAdjusted[currentMoveNum] = 0.0f;
			}

			if ((xVal) > 0)
			{
				prevRMessValRangeAdjusted[currentMoveNum] = retMessageValPixAtDispLevel(currentRMoveStart, currentXMove, currentYMove, prevRMessage);
			}
			else
			{
				prevRMessValRangeAdjusted[currentMoveNum] = 0.0f;
			}

			currentXMove += currentBeliefPropParamsConstMem.currentMoveIncrementX;
			currentMoveNum++;
		}
		currentYMove += currentBeliefPropParamsConstMem.currentMoveIncrementY;
	}

	computeBpMsgMotionEstimation(prevUMessValRangeAdjusted, prevLMessValRangeAdjusted, prevRMessValRangeAdjusted,
		dataMessage, currentUMessage);

	computeBpMsgMotionEstimation(prevDMessValRangeAdjusted, prevLMessValRangeAdjusted, prevRMessValRangeAdjusted,
		dataMessage, currentDMessage);

	computeBpMsgMotionEstimation(prevUMessValRangeAdjusted, prevDMessValRangeAdjusted, prevLMessValRangeAdjusted,
		dataMessage, currentLMessage);

	computeBpMsgMotionEstimation(prevUMessValRangeAdjusted, prevDMessValRangeAdjusted, prevRMessValRangeAdjusted,
		dataMessage, currentRMessage);


	for (int movementXY = 0; movementXY < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementXY++)
	{
		indexWriteTo = retrieveIndexInDataAndMessage(xVal, yVal, widthLevelCheckerboardPart, heightLevelCheckerboard, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));
		messageValsDeviceCurrCheckOut.messageUDevice[indexWriteTo] = currentUMessage[movementXY];
		messageValsDeviceCurrCheckOut.messageDDevice[indexWriteTo] = currentDMessage[movementXY];
		messageValsDeviceCurrCheckOut.messageLDevice[indexWriteTo] = currentLMessage[movementXY];
		messageValsDeviceCurrCheckOut.messageRDevice[indexWriteTo] = currentRMessage[movementXY];
	}
}



//device portion of the kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the 
//"checkerboard" scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
__device__ void runBPIterationUsingCheckerboardUpdatesDeviceNoTextures(checkerboardMessagesDeviceStruct messageValsDeviceCurrCheckOut, 
									checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboardIn, 
									float dataMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], 
									int widthLevelCheckerboardPart, int heightLevelCheckerboard, int checkerboardAdjustment,
									int xVal, int yVal, 
									currentStartMoveParamsPixelAtLevel* currentPixParamsCurrentCheckerboard, 
									currentStartMoveParamsPixelAtLevel* currentPixParamsNeighCheckerboard)
{
	int indexWriteTo;

	//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
	//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
	//declare a "default" start movement
	currentStartMoveParamsPixelAtLevel defStartMove;
	defStartMove.x = 0.0f;
	defStartMove.y = 0.0f;


	//declare the starting movement for the current pixel and each neighbor
	currentStartMoveParamsPixelAtLevel currentPixMoveStart;

	currentStartMoveParamsPixelAtLevel currentUMoveStart;
	currentStartMoveParamsPixelAtLevel currentDMoveStart;
	currentStartMoveParamsPixelAtLevel currentLMoveStart;
	currentStartMoveParamsPixelAtLevel currentRMoveStart;


	//used to retrieve the starting x and y movements at the current pixel for the current checkerboard and also
	//each of the neighbors
	currentPixMoveStart = currentPixParamsCurrentCheckerboard[retrieveIndexCurrentPixel2DGrid(xVal, yVal, widthLevelCheckerboardPart, heightLevelCheckerboard)];

	//each neighbor is on "opposite" checkerboard of one being updated
	//check to make sure desired neighbor exists in each case
	if (yVal < (heightLevelCheckerboard-1))
	{
		currentUMoveStart = currentPixParamsNeighCheckerboard[retrieveIndexCurrentPixel2DGrid(xVal, (yVal+1), widthLevelCheckerboardPart, heightLevelCheckerboard)];
	}
	else
	{
		currentUMoveStart = defStartMove;

	}

	if (yVal > 0)
	{
		currentDMoveStart = currentPixParamsNeighCheckerboard[retrieveIndexCurrentPixel2DGrid(xVal, (yVal-1), widthLevelCheckerboardPart, heightLevelCheckerboard)];
	}
	else
	{
		currentDMoveStart = defStartMove;
	}

	if ((xVal) < (widthLevelCheckerboardPart - 1))
	{
		currentLMoveStart = currentPixParamsNeighCheckerboard[retrieveIndexCurrentPixel2DGrid(xVal + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevelCheckerboard)];
	}
	else
	{
		currentLMoveStart = defStartMove;
	}
	if ((xVal) > 0)
	{
		currentRMoveStart = currentPixParamsNeighCheckerboard[retrieveIndexCurrentPixel2DGrid(xVal - 1 + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevelCheckerboard)];
	}
	else
	{
		currentRMoveStart = defStartMove;
	}

	float prevUMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
	float prevDMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
	float prevLMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
	float prevRMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];

	//go through all the movements in the current range
	for (int movementXY = 0; movementXY < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementXY++)
	{
		//make sure that neighboring pixel exists...need to check boundaries
		if (yVal < (heightLevelCheckerboard-1))
		{
			prevUMessage[movementXY] = messagesDeviceCurrentCheckerboardIn.messageUDevice[retrieveIndexInDataAndMessage(xVal, (yVal+1), widthLevelCheckerboardPart, heightLevelCheckerboard, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir))];
		}
		else
		{
			prevUMessage[movementXY] = 0.0f;
		}

		if (yVal > 0)
		{
			prevDMessage[movementXY] = messagesDeviceCurrentCheckerboardIn.messageDDevice[retrieveIndexInDataAndMessage(xVal, (yVal-1), widthLevelCheckerboardPart, heightLevelCheckerboard, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir))];
		}
		else
		{
			prevDMessage[movementXY] = 0.0f;
		}

		if ((xVal) < (widthLevelCheckerboardPart - 1))
		{
			prevLMessage[movementXY] = messagesDeviceCurrentCheckerboardIn.messageLDevice[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), yVal, widthLevelCheckerboardPart, heightLevelCheckerboard, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir))];
		}
		else
		{
			prevLMessage[movementXY] = 0.0f;
		}

		if ((xVal) > 0)
		{
			prevRMessage[movementXY] = messagesDeviceCurrentCheckerboardIn.messageRDevice[retrieveIndexInDataAndMessage((xVal - 1 + checkerboardAdjustment), yVal, widthLevelCheckerboardPart, heightLevelCheckerboard, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir))];
		}
		else
		{
			prevRMessage[movementXY] = 0.0f;
		}
	}

	float currentUMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
	float currentDMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
	float currentLMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
	float currentRMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];

	//retrieve the message values corresponding to each possible movement within the range
	float prevUMessValRangeAdjusted[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
	float prevDMessValRangeAdjusted[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
	float prevLMessValRangeAdjusted[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
	float prevRMessValRangeAdjusted[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];

	//go through all the movements in the current range
	float currentXMove = currentPixMoveStart.x;
	float currentYMove = currentPixMoveStart.y;
	int currentMoveNum = 0;
	for (int yMoveNum = 0; yMoveNum < (currentBeliefPropParamsConstMem.totalNumMovesYDir); yMoveNum++)
	{
		currentXMove = currentPixMoveStart.x;
		for (int xMoveNum = 0; xMoveNum < (currentBeliefPropParamsConstMem.totalNumMovesXDir); xMoveNum++)
		{
			if (yVal < (heightLevelCheckerboard-1))
			{
				prevUMessValRangeAdjusted[currentMoveNum] = retMessageValPixAtDispLevel(currentUMoveStart, currentXMove, currentYMove, prevUMessage);
			}
			else
			{
				prevUMessValRangeAdjusted[currentMoveNum] = 0.0f;
			}

			if (yVal > 0)
			{
				prevDMessValRangeAdjusted[currentMoveNum] = retMessageValPixAtDispLevel(currentDMoveStart, currentXMove, currentYMove, prevDMessage);
			}
			else
			{
				prevDMessValRangeAdjusted[currentMoveNum] = 0.0f;
			}

			if ((xVal) < (widthLevelCheckerboardPart - 1))
			{
				prevLMessValRangeAdjusted[currentMoveNum] = retMessageValPixAtDispLevel(currentLMoveStart, currentXMove, currentYMove, prevLMessage);
			}
			else
			{
				prevLMessValRangeAdjusted[currentMoveNum] = 0.0f;
			}

			if ((xVal) > 0)
			{
				prevRMessValRangeAdjusted[currentMoveNum] = retMessageValPixAtDispLevel(currentRMoveStart, currentXMove, currentYMove, prevRMessage);
			}
			else
			{
				prevRMessValRangeAdjusted[currentMoveNum] = 0.0f;
			}

			currentXMove += currentBeliefPropParamsConstMem.currentMoveIncrementX;
			currentMoveNum++;
		}
		currentYMove += currentBeliefPropParamsConstMem.currentMoveIncrementY;
	}

	computeBpMsgMotionEstimation(prevUMessValRangeAdjusted, prevLMessValRangeAdjusted, prevRMessValRangeAdjusted,
		dataMessage, currentUMessage);

	computeBpMsgMotionEstimation(prevDMessValRangeAdjusted, prevLMessValRangeAdjusted, prevRMessValRangeAdjusted,
		dataMessage, currentDMessage);

	computeBpMsgMotionEstimation(prevUMessValRangeAdjusted, prevDMessValRangeAdjusted, prevLMessValRangeAdjusted,
		dataMessage, currentLMessage);

	computeBpMsgMotionEstimation(prevUMessValRangeAdjusted, prevDMessValRangeAdjusted, prevRMessValRangeAdjusted,
		dataMessage, currentRMessage);


	for (int movementXY = 0; movementXY < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementXY++)
	{
		indexWriteTo = retrieveIndexInDataAndMessage(xVal, yVal, widthLevelCheckerboardPart, heightLevelCheckerboard, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));
		messageValsDeviceCurrCheckOut.messageUDevice[indexWriteTo] = currentUMessage[movementXY];
		messageValsDeviceCurrCheckOut.messageDDevice[indexWriteTo] = currentDMessage[movementXY];
		messageValsDeviceCurrCheckOut.messageLDevice[indexWriteTo] = currentLMessage[movementXY];
		messageValsDeviceCurrCheckOut.messageRDevice[indexWriteTo] = currentRMessage[movementXY];
	}
}



//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard" 
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
__global__ void runBPIterationUsingCheckerboardUpdates( checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboardOut,
													int widthLevel, int heightLevel, checkerboardPortionEnum currentCheckerboardUpdating, int dataTexOffset, int paramsTexOffset)
{
	int xVal = blockIdx.x * blockDim.x + threadIdx.x;
	int yVal = blockIdx.y * blockDim.y + threadIdx.y;

	//set the width of the "checkerboard" to half the width of the level and the height of the checkerboard
	//to the same height as the level
	int widthLevelCheckerboardPart = widthLevel / 2;
	int heightCheckerboard = heightLevel;

	//need to pad since using neighbors...
	if (withinImageBounds(xVal, yVal, widthLevelCheckerboardPart, heightCheckerboard))
	{
		int checkerboardAdjustment;

		//check if updating part 1 or 2 of the checkerboard
		if (currentCheckerboardUpdating == CHECKERBOARD_PART_1_ENUM)
		{
			checkerboardAdjustment = ((yVal)%2);
		}
		else //updating part 2 of checkerboard
		{
			checkerboardAdjustment = ((yVal+1)%2);
		}

		float dataMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];

		for (int movementXY = 0; movementXY < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementXY++)
		{
			dataMessage[movementXY] = tex1Dfetch(dataCostsCurrCheckerboard, retrieveIndexInDataAndMessage(xVal, yVal, widthLevelCheckerboardPart, heightCheckerboard, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir), dataTexOffset));
		}

		runBPIterationUsingCheckerboardUpdatesDevice(messagesDeviceCurrentCheckerboardOut, dataMessage, widthLevelCheckerboardPart, 
			heightCheckerboard, checkerboardAdjustment, xVal, yVal, paramsTexOffset);
	}
}



//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard" 
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
//same as previous function but without textures...
//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard" 
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
//same as previous function but without textures...
__global__ void runBPIterationUsingCheckerboardUpdatesNoTextures( checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboardOut, checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboardIn,
													int widthLevel, int heightLevel, checkerboardPortionEnum currentCheckerboardUpdating,
				currentStartMoveParamsPixelAtLevel* currentPixParamsCurrentCheckerboard, 
									currentStartMoveParamsPixelAtLevel* currentPixParamsNeighCheckerboard,
									float* dataCostsCurrCheckerboard)
{
	int xVal = blockIdx.x * blockDim.x + threadIdx.x;
	int yVal = blockIdx.y * blockDim.y + threadIdx.y;

	//set the width of the "checkerboard" to half the width of the level and the height of the checkerboard
	//to the same height as the level
	int widthLevelCheckerboardPart = widthLevel / 2;
	int heightCheckerboard = heightLevel;

	//need to pad since using neighbors...
	if (withinImageBounds(xVal, yVal, widthLevelCheckerboardPart, heightCheckerboard))
	{
		int checkerboardAdjustment;

		//check if updating part 1 or 2 of the checkerboard
		if (currentCheckerboardUpdating == CHECKERBOARD_PART_1_ENUM)
		{
			checkerboardAdjustment = ((yVal)%2);
		}
		else //updating part 2 of checkerboard
		{
			checkerboardAdjustment = ((yVal+1)%2);
		}

		float dataMessage[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];

		for (int movementXY = 0; movementXY < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementXY++)
		{
			dataMessage[movementXY] =  dataCostsCurrCheckerboard[retrieveIndexInDataAndMessage(xVal, yVal, widthLevelCheckerboardPart, heightCheckerboard, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir))];
		}

		runBPIterationUsingCheckerboardUpdatesDeviceNoTextures(messagesDeviceCurrentCheckerboardOut, messagesDeviceCurrentCheckerboardIn, dataMessage, widthLevelCheckerboardPart, 
			heightCheckerboard, checkerboardAdjustment, xVal, yVal, currentPixParamsCurrentCheckerboard, currentPixParamsNeighCheckerboard);
	}
}


//kernal to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//note that since the motions at the next level "down" are of a different range, adjustments need to be made when copying values
//to the pixels at the next level down
//also copy the paramsCurrentPixelAtLevel to the next level down
__global__ void copyPrevLevelToNextLevelBPCheckerboard(checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboard1,
															checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboard2,
															int widthPrevLevel, int heightPrevLevel, int widthNextLevel, int heightNextLevel,
															currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboard1,
															currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboard2,
															checkerboardPortionEnum checkerboardPart)
{
	//this is from the perspective of the "previous" level
	//so there are widthPrevLevel*heightPrevLevel threads within the bounds
	int xValInPrevCheck = blockIdx.x * blockDim.x + threadIdx.x;
	int yValInPrevCheck = blockIdx.y * blockDim.y + threadIdx.y;

	//width of checkerboard at the previous level is half the width of the level
	//if width of level is odd, the then last column is "skipped"
	//and then should be initialized with the same values as the "neighboring" values
	int widthPrevLevelCheckerboard = widthPrevLevel/2;
	int heightPrevLevelCheckerboard = heightPrevLevel;

	//first retrieve the estimated motion in the x and y direction at this level using the current message and data cost values
	if (withinImageBounds(xValInPrevCheck, yValInPrevCheck, widthPrevLevelCheckerboard, heightPrevLevelCheckerboard))
	{
		//declare and retrieve the current movement parameters in the x and y directions in the "previous" level
		currentStartMoveParamsPixelAtLevel currentMovementStartCurrentPix = tex1Dfetch(currentPixParamsTexCurrentCheckerboard, retrieveIndexCurrentPixel2DGrid(xValInPrevCheck, yValInPrevCheck,
				widthPrevLevelCheckerboard, heightPrevLevelCheckerboard));

		//width of checkerboard must be even
		int widthCheckerboardNextLevel = widthNextLevel / 2;
		int heightCheckerboardNextLevel = heightNextLevel;

		int indexCopyTo;
		int indexCopyFrom;

		int checkerboardPartAdjustment;

		//retrieve the current move parameters and the adjustment for the current checkerboard portion
		if (checkerboardPart == CHECKERBOARD_PART_1_ENUM)
		{
			checkerboardPartAdjustment = (yValInPrevCheck%2);
		}
		else //if (checkerboardPart == CHECKERBOARD_PART_2_ENUM)
		{
			checkerboardPartAdjustment = ((yValInPrevCheck+1)%2);
		}

		//retrieve the estimated x and y movements for the pixel at the "previous" level
		float estXMovementPixAtLevel = tex1Dfetch(estimatedXMovementTexDeviceCurrentCheckerboard, retrieveIndexCurrentPixel2DGrid(xValInPrevCheck, yValInPrevCheck, widthPrevLevelCheckerboard, heightPrevLevelCheckerboard));
		float estYMovementPixAtLevel = tex1Dfetch(estimatedYMovementTexDeviceCurrentCheckerboard, retrieveIndexCurrentPixel2DGrid(xValInPrevCheck, yValInPrevCheck, widthPrevLevelCheckerboard, heightPrevLevelCheckerboard));

		//retrieve the new movement range using the estimated movement values and the previous movement range
		currentStartMoveParamsPixelAtLevel newParamMoveRange = retrieveMovementRangeAtNextLevel(currentMovementStartCurrentPix, estXMovementPixAtLevel, estYMovementPixAtLevel);

		int indexCopyToMovement = retrieveIndexCurrentPixel2DGrid(xValInPrevCheck*2 + checkerboardPartAdjustment, (yValInPrevCheck*2), widthCheckerboardNextLevel, heightCheckerboardNextLevel);
		paramsNextLevelDeviceCheckerboard1[indexCopyToMovement] = newParamMoveRange;
		paramsNextLevelDeviceCheckerboard2[indexCopyToMovement] = newParamMoveRange;

		indexCopyToMovement = retrieveIndexCurrentPixel2DGrid(xValInPrevCheck*2 + checkerboardPartAdjustment, (yValInPrevCheck*2 + 1), widthCheckerboardNextLevel, heightCheckerboardNextLevel);
		paramsNextLevelDeviceCheckerboard1[indexCopyToMovement] = newParamMoveRange;
		paramsNextLevelDeviceCheckerboard2[indexCopyToMovement] = newParamMoveRange;

		//first retrieve an array with all the "previous" message values at the "previous" level
		float prevMessValsU[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
		float prevMessValsD[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
		float prevMessValsL[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
		float prevMessValsR[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];

		//go through all the possible movements and store then to prevMessVals[U/D/L/R]
		for (int movementXY = 0; movementXY < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir);
			movementXY++)
		{
			indexCopyFrom = retrieveIndexInDataAndMessage(xValInPrevCheck, yValInPrevCheck, widthPrevLevelCheckerboard, heightPrevLevelCheckerboard, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir)); 

			//retrieve the previous message values bound to the appropriate texture (either from checkerboard 1 or 2 depending on the current checkerboard being processed...)
			prevMessValsU[movementXY] = tex1Dfetch(messageUTexCurrReadCheckerboard, indexCopyFrom);
			prevMessValsD[movementXY] = tex1Dfetch(messageDTexCurrReadCheckerboard, indexCopyFrom);
			prevMessValsL[movementXY] = tex1Dfetch(messageLTexCurrReadCheckerboard, indexCopyFrom);
			prevMessValsR[movementXY] = tex1Dfetch(messageRTexCurrReadCheckerboard, indexCopyFrom);
		}

		//declare the current movement "number" which starts at 0
		int currentMoveNum = 0;

		//assuming the number of movements in each hierarchical level is constant for now...
		
		//retrieve the starting index of the new starting movement in the x and y directions within the previous
		//movement range
		float startXMovePrevMessLev = (newParamMoveRange.x - currentMovementStartCurrentPix.x) /
			(currentBeliefPropParamsConstMem.currentMoveIncrementX);
		float startYMovePrevMessLev = (newParamMoveRange.y - currentMovementStartCurrentPix.y) /
			(currentBeliefPropParamsConstMem.currentMoveIncrementY);

		//declare the current x and y indices that represent the location in the "previous message" level to copy
		//into the current message level
		float currentXIndexPrevMessLev = startXMovePrevMessLev;
		float currentYIndexPrevMessLev = startYMovePrevMessLev;

		for (int newMoveYNum = 0; newMoveYNum < currentBeliefPropParamsConstMem.totalNumMovesYDir; newMoveYNum++)
		{
			//reset the current x within the previous message with each row
			currentXIndexPrevMessLev = startXMovePrevMessLev;

			for (int newMoveXNum = 0; newMoveXNum < currentBeliefPropParamsConstMem.totalNumMovesXDir; newMoveXNum++)
			{
				//retrieve the interpolated message values from the previous message values to copy into the message values for the "next" level
				float interpolatedPrevValU = retrieve2DArrayValFromIndex(currentXIndexPrevMessLev, currentYIndexPrevMessLev, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir, prevMessValsU);
				float interpolatedPrevValD = retrieve2DArrayValFromIndex(currentXIndexPrevMessLev, currentYIndexPrevMessLev, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir, prevMessValsD);
				float interpolatedPrevValL = retrieve2DArrayValFromIndex(currentXIndexPrevMessLev, currentYIndexPrevMessLev, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir, prevMessValsL);
				float interpolatedPrevValR = retrieve2DArrayValFromIndex(currentXIndexPrevMessLev, currentYIndexPrevMessLev, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir, prevMessValsR);

				indexCopyTo = retrieveIndexInDataAndMessage((xValInPrevCheck*2 + checkerboardPartAdjustment), (yValInPrevCheck*2), widthCheckerboardNextLevel, heightCheckerboardNextLevel, currentMoveNum, ((currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir)));

				messagesDeviceCurrentCheckerboard1.messageUDevice[indexCopyTo] = interpolatedPrevValU;
				messagesDeviceCurrentCheckerboard1.messageDDevice[indexCopyTo] = interpolatedPrevValD;
				messagesDeviceCurrentCheckerboard1.messageLDevice[indexCopyTo] = interpolatedPrevValL;
				messagesDeviceCurrentCheckerboard1.messageRDevice[indexCopyTo] = interpolatedPrevValR;

				messagesDeviceCurrentCheckerboard2.messageUDevice[indexCopyTo] = interpolatedPrevValU;
				messagesDeviceCurrentCheckerboard2.messageDDevice[indexCopyTo] = interpolatedPrevValD;
				messagesDeviceCurrentCheckerboard2.messageLDevice[indexCopyTo] = interpolatedPrevValL;
				messagesDeviceCurrentCheckerboard2.messageRDevice[indexCopyTo] = interpolatedPrevValR;

				indexCopyTo = retrieveIndexInDataAndMessage((xValInPrevCheck*2 + checkerboardPartAdjustment), (yValInPrevCheck*2 + 1), widthCheckerboardNextLevel, heightCheckerboardNextLevel, currentMoveNum, ((currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir))); 

				messagesDeviceCurrentCheckerboard1.messageUDevice[indexCopyTo] = interpolatedPrevValU;

				messagesDeviceCurrentCheckerboard1.messageDDevice[indexCopyTo] = interpolatedPrevValD;
				messagesDeviceCurrentCheckerboard1.messageLDevice[indexCopyTo] = interpolatedPrevValL;
				messagesDeviceCurrentCheckerboard1.messageRDevice[indexCopyTo] = interpolatedPrevValR;

				messagesDeviceCurrentCheckerboard2.messageUDevice[indexCopyTo] = interpolatedPrevValU;
				messagesDeviceCurrentCheckerboard2.messageDDevice[indexCopyTo] = interpolatedPrevValD;
				messagesDeviceCurrentCheckerboard2.messageLDevice[indexCopyTo] = interpolatedPrevValL;
				messagesDeviceCurrentCheckerboard2.messageRDevice[indexCopyTo] = interpolatedPrevValR;

				//increment the current "move number" which increases across then vertically
				currentMoveNum++;

				//use the defined proportion that the movement is adjusted between the current and next level to adjust the current
				//x and y index
				currentXIndexPrevMessLev += currentBeliefPropParamsConstMem.propChangeMoveNextLevel;
			}

			//use the defined proportion that the movement is adjusted between the current and next level to adjust the current
			//x and y index
			currentYIndexPrevMessLev += currentBeliefPropParamsConstMem.propChangeMoveNextLevel;
		}
	}
}

//kernel for retrieving the `next' set of parameters given the current estimated values and the movement increment
__global__ void getNextParamSet(currentStartMoveParamsPixelAtLevel* paramsCurrCheckerboard, float* estXMotionCheckerboard, float* estYMotionCheckerboard, int widthCheckerboard, int heightCheckerboard) 
{
	//retrieve the x and y values of the thread...
	int xValThread = blockIdx.x * blockDim.x + threadIdx.x;
	int yValThread = blockIdx.y * blockDim.y + threadIdx.y;

	if (withinImageBounds(xValThread, yValThread, widthCheckerboard, heightCheckerboard))
	{
		//retrieve the estimated x and y motions...
		float estXMotion = estXMotionCheckerboard[retrieveIndexCurrentPixel2DGrid(xValThread, yValThread,
				widthCheckerboard, heightCheckerboard)];
		float estYMotion = estYMotionCheckerboard[retrieveIndexCurrentPixel2DGrid(xValThread, yValThread,
				widthCheckerboard, heightCheckerboard)];

		//retrieve the current parameter
		currentStartMoveParamsPixelAtLevel currStartMoveParam = paramsCurrCheckerboard[retrieveIndexCurrentPixel2DGrid(xValThread, yValThread,
				widthCheckerboard, heightCheckerboard)];

		//compute the movement parameter for the next level...
		paramsCurrCheckerboard[retrieveIndexCurrentPixel2DGrid(xValThread, yValThread,
				widthCheckerboard, heightCheckerboard)] = retrieveMovementRangeAtNextLevel(currStartMoveParam, estXMotion, estYMotion);
	}
}






//kernal to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//note that since the motions at the next level "down" are of a different range, adjustments need to be made when copying values
//to the pixels at the next level down
//also copy the paramsCurrentPixelAtLevel to the next level down
//the size of each level remains constant here
__global__ void copyPrevLevelToNextLevelBPCheckerboardLevelSizeConst(checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboardCopyTo,
															int widthLevel, int heightLevel,
															currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboardCopyTo)
{
	//this is from the perspective of the "previous" level
	//so there are widthLevel*heightLevel threads within the bounds
	int xValInPrevCheck = blockIdx.x * blockDim.x + threadIdx.x;
	int yValInPrevCheck = blockIdx.y * blockDim.y + threadIdx.y;

	//width of checkerboard is half the width of the level
	//if width of level is odd, the then last column is "skipped"
	//and then should be initialized with the same values as the "neighboring" values
	int widthCheckerboardEachLevel = widthLevel/2;
	int heightCheckerboardEachLevel = heightLevel;

	//first retrieve the estimated motion in the x and y direction at this level using the current message and data cost values
	if (withinImageBounds(xValInPrevCheck, yValInPrevCheck, widthCheckerboardEachLevel, heightCheckerboardEachLevel))
	{
		//declare and retrieve the current movement parameters in the x and y directions in the "previous" level
		currentStartMoveParamsPixelAtLevel currentMovementStartCurrentPix = tex1Dfetch(currentPixParamsTexCurrentCheckerboard, retrieveIndexCurrentPixel2DGrid(xValInPrevCheck, yValInPrevCheck,
				widthCheckerboardEachLevel, heightCheckerboardEachLevel));


		//retrieve the estimated x and y movements for the pixel at the "previous" level
		float estXMovementPixAtLevel = tex1Dfetch(estimatedXMovementTexDeviceCurrentCheckerboard, retrieveIndexCurrentPixel2DGrid(xValInPrevCheck, yValInPrevCheck, widthCheckerboardEachLevel, heightCheckerboardEachLevel));
		float estYMovementPixAtLevel = tex1Dfetch(estimatedYMovementTexDeviceCurrentCheckerboard, retrieveIndexCurrentPixel2DGrid(xValInPrevCheck, yValInPrevCheck, widthCheckerboardEachLevel, heightCheckerboardEachLevel));

		//retrieve the new movement range using the estimated movement values and the previous movement range
		currentStartMoveParamsPixelAtLevel newParamMoveRange = retrieveMovementRangeAtNextLevel(currentMovementStartCurrentPix, estXMovementPixAtLevel, estYMovementPixAtLevel);

		int indexCopyToMovement = retrieveIndexCurrentPixel2DGrid(xValInPrevCheck, yValInPrevCheck, widthCheckerboardEachLevel, heightCheckerboardEachLevel);
		paramsNextLevelDeviceCheckerboardCopyTo[indexCopyToMovement] = newParamMoveRange;

		//first retrieve an array with all the "previous" message values at the "previous" level
		float prevMessValsU[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
		float prevMessValsD[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
		float prevMessValsL[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
		float prevMessValsR[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];

		//go through all the possible movements and store then to prevMessVals[U/D/L/R]
		for (int movementXY = 0; movementXY < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir);
			movementXY++)
		{
			int indexCopyFrom = retrieveIndexInDataAndMessage(xValInPrevCheck, yValInPrevCheck, widthCheckerboardEachLevel, heightCheckerboardEachLevel, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir)); 

			//retrieve the previous message values bound to the appropriate texture (either from checkerboard 1 or 2 depending on the current checkerboard being processed...)
			prevMessValsU[movementXY] = tex1Dfetch(messageUTexCurrReadCheckerboard, indexCopyFrom);
			prevMessValsD[movementXY] = tex1Dfetch(messageDTexCurrReadCheckerboard, indexCopyFrom);
			prevMessValsL[movementXY] = tex1Dfetch(messageLTexCurrReadCheckerboard, indexCopyFrom);
			prevMessValsR[movementXY] = tex1Dfetch(messageRTexCurrReadCheckerboard, indexCopyFrom);
		}

		//declare the current movement "number" which starts at 0
		int currentMoveNum = 0;

		//assuming the number of movements in each hierarchical level is constant for now...
		
		//retrieve the starting index of the new starting movement in the x and y directions within the previous
		//movement range
		float startXMovePrevMessLev = (newParamMoveRange.x - currentMovementStartCurrentPix.x) /
			(currentBeliefPropParamsConstMem.currentMoveIncrementX);
		float startYMovePrevMessLev = (newParamMoveRange.y - currentMovementStartCurrentPix.y) /
			(currentBeliefPropParamsConstMem.currentMoveIncrementY);

		//declare the current x and y indices that represent the location in the "previous message" level to copy
		//into the current message level
		float currentXIndexPrevMessLev = startXMovePrevMessLev;
		float currentYIndexPrevMessLev = startYMovePrevMessLev;

		for (int newMoveYNum = 0; newMoveYNum < currentBeliefPropParamsConstMem.totalNumMovesYDir; newMoveYNum++)
		{
			//reset the current x within the previous message with each row
			currentXIndexPrevMessLev = startXMovePrevMessLev;

			for (int newMoveXNum = 0; newMoveXNum < currentBeliefPropParamsConstMem.totalNumMovesXDir; newMoveXNum++)
			{
				//retrieve the interpolated message values from the previous message values to copy into the message values for the "next" level
				float interpolatedPrevValU = retrieve2DArrayValFromIndex(currentXIndexPrevMessLev, currentYIndexPrevMessLev, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir, prevMessValsU);
				float interpolatedPrevValD = retrieve2DArrayValFromIndex(currentXIndexPrevMessLev, currentYIndexPrevMessLev, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir, prevMessValsD);
				float interpolatedPrevValL = retrieve2DArrayValFromIndex(currentXIndexPrevMessLev, currentYIndexPrevMessLev, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir, prevMessValsL);
				float interpolatedPrevValR = retrieve2DArrayValFromIndex(currentXIndexPrevMessLev, currentYIndexPrevMessLev, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir, prevMessValsR);

				int indexCopyTo = retrieveIndexInDataAndMessage(xValInPrevCheck, yValInPrevCheck, widthCheckerboardEachLevel, heightCheckerboardEachLevel, currentMoveNum, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));

				messagesDeviceCurrentCheckerboardCopyTo.messageUDevice[indexCopyTo] = interpolatedPrevValU;
				messagesDeviceCurrentCheckerboardCopyTo.messageDDevice[indexCopyTo] = interpolatedPrevValD;
				messagesDeviceCurrentCheckerboardCopyTo.messageLDevice[indexCopyTo] = interpolatedPrevValL;
				messagesDeviceCurrentCheckerboardCopyTo.messageRDevice[indexCopyTo] = interpolatedPrevValR;


				//increment the current "move number" which increases across then vertically
				currentMoveNum++;

				//use the defined proportion that the movement is adjusted between the current and next level to adjust the current
				//x and y index
				currentXIndexPrevMessLev += currentBeliefPropParamsConstMem.propChangeMoveNextLevel;
			}

			//use the defined proportion that the movement is adjusted between the current and next level to adjust the current
			//x and y index
			currentYIndexPrevMessLev += currentBeliefPropParamsConstMem.propChangeMoveNextLevel;
		}
	}
}

//use the retrieved output movement to initialize the x and y movement range at the next level
//for now, perform this in steps of "two", where the movement range is halved each time
__device__ currentStartMoveParamsPixelAtLevel retrieveMovementRangeAtNextLevel(currentStartMoveParamsPixelAtLevel inputStartPossMovePixelParam, float estXDirMotion, float estYDirMotion)
{
	//declare the current output pixel parameter
	currentStartMoveParamsPixelAtLevel outputStartPossMovePixelParam;

#if (RET_MOVE_RANGE_METHOD == USE_MID_VAL_RET_MOVE_RANGE)
	//allow for the motion to be extended by .5*currentParamsAllPixAtLevelConstMem.propChangeMoveNextLevel*(numPossMovesLevel-1)*incPossMovesLevel from the estimated motion in the
	//x and y directions
	outputStartPossMovePixelParam.x = estXDirMotion - 0.5f*(currentBeliefPropParamsConstMem.totalNumMovesXDir-1)*(currentBeliefPropParamsConstMem.propChangeMoveNextLevel*currentBeliefPropParamsConstMem.currentMoveIncrementX);
	outputStartPossMovePixelParam.y = estYDirMotion - 0.5f*(currentBeliefPropParamsConstMem.totalNumMovesYDir-1)*(currentBeliefPropParamsConstMem.propChangeMoveNextLevel*currentBeliefPropParamsConstMem.currentMoveIncrementY);

#elif (RET_MOVE_RANGE_METHOD == ROUND_MID_VAL_UP_RET_MOVE_RANGE)

	float numMovesBelowMidNextLevX = 0.5f * (currentBeliefPropParamsConstMem.totalNumMovesXDir-1);
	float numMovesBelowMidNextLevY = 0.5f * (currentBeliefPropParamsConstMem.totalNumMovesYDir-1);

	outputStartPossMovePixelParam.x = estXDirMotion - (floor(numMovesBelowMidNextLevX))*(currentBeliefPropParamsConstMem.propChangeMoveNextLevel*currentBeliefPropParamsConstMem.currentMoveIncrementX);
	outputStartPossMovePixelParam.y = estYDirMotion - (floor(numMovesBelowMidNextLevY))*(currentBeliefPropParamsConstMem.propChangeMoveNextLevel*currentBeliefPropParamsConstMem.currentMoveIncrementY);

#elif (RET_MOVE_RANGE_METHOD == ROUND_MID_VAL_DOWN_RET_MOVE_RANGE)

	float numMovesBelowMidNextLevX = 0.5f * (currentBeliefPropParamsConstMem.totalNumMovesXDir-1);
	float numMovesBelowMidNextLevY = 0.5f * (currentBeliefPropParamsConstMem.totalNumMovesYDir-1);

	outputStartPossMovePixelParam.x = estXDirMotion - (ceil(numMovesBelowMidNextLevX))*(currentBeliefPropParamsConstMem.propChangeMoveNextLevel*currentBeliefPropParamsConstMem.currentMoveIncrementX);
	outputStartPossMovePixelParam.y = estYDirMotion - (ceil(numMovesBelowMidNextLevY))*(currentBeliefPropParamsConstMem.propChangeMoveNextLevel*currentBeliefPropParamsConstMem.currentMoveIncrementY);

#endif //RET_MOVE_RANGE_METHOD

	//check if the starting x move or ending move is beyond the range of the input pixel...make adjustments if it is

	//if starting x or y move comes before the input starting pixel, set the starting pixel value to the starting movement
	//of the input pixel
	if (outputStartPossMovePixelParam.x < inputStartPossMovePixelParam.x)
	{
		#if (RANGE_BOUNDARIES_SETTING == DONT_ADJUST_RANGE_BOUND_FOR_EST_VAL)
				outputStartPossMovePixelParam.x = inputStartPossMovePixelParam.x;
		#elif (RANGE_BOUNDARIES_SETTING == ADJUST_RANGE_BOUND_FOR_EST_VAL)
			outputStartPossMovePixelParam.x = outputStartPossMovePixelParam.x + 
				(currentBeliefPropParamsConstMem.propChangeMoveNextLevel*currentBeliefPropParamsConstMem.currentMoveIncrementX) *
				(ceil((inputStartPossMovePixelParam.x - outputStartPossMovePixelParam.x) / 
				(currentBeliefPropParamsConstMem.propChangeMoveNextLevel*currentBeliefPropParamsConstMem.currentMoveIncrementX)));
		#endif 
	}
	if (outputStartPossMovePixelParam.y < inputStartPossMovePixelParam.y)
	{
		#if (RANGE_BOUNDARIES_SETTING == DONT_ADJUST_RANGE_BOUND_FOR_EST_VAL)
				outputStartPossMovePixelParam.y = inputStartPossMovePixelParam.y;
		#elif (RANGE_BOUNDARIES_SETTING == ADJUST_RANGE_BOUND_FOR_EST_VAL)
			outputStartPossMovePixelParam.y = outputStartPossMovePixelParam.y + 
				(currentBeliefPropParamsConstMem.propChangeMoveNextLevel*currentBeliefPropParamsConstMem.currentMoveIncrementY) *
				(ceil((inputStartPossMovePixelParam.y - outputStartPossMovePixelParam.y) / 
				(currentBeliefPropParamsConstMem.propChangeMoveNextLevel*currentBeliefPropParamsConstMem.currentMoveIncrementY)));
		#endif
	}

	//now retrieve the current "ending" possible x and y moves and also of the input params
	float currentEndXMove = outputStartPossMovePixelParam.x + (currentBeliefPropParamsConstMem.totalNumMovesXDir-1)*(currentBeliefPropParamsConstMem.propChangeMoveNextLevel*currentBeliefPropParamsConstMem.currentMoveIncrementX);
	float currentEndYMove = outputStartPossMovePixelParam.y + (currentBeliefPropParamsConstMem.totalNumMovesYDir-1)*(currentBeliefPropParamsConstMem.propChangeMoveNextLevel*currentBeliefPropParamsConstMem.currentMoveIncrementY);

	float inputEndXMove = inputStartPossMovePixelParam.x + (currentBeliefPropParamsConstMem.totalNumMovesXDir-1)*currentBeliefPropParamsConstMem.currentMoveIncrementX;
	float inputEndYMove = inputStartPossMovePixelParam.y + (currentBeliefPropParamsConstMem.totalNumMovesYDir-1)*currentBeliefPropParamsConstMem.currentMoveIncrementY;

	//if current ending x or y move is beyond the input ending x or y move, adjust current starting move so that the ending move matches up with the input ending move
	if (currentEndXMove > inputEndXMove)
	{
		#if (RANGE_BOUNDARIES_SETTING == DONT_ADJUST_RANGE_BOUND_FOR_EST_VAL)
				outputStartPossMovePixelParam.x = inputEndXMove - (currentBeliefPropParamsConstMem.totalNumMovesXDir-1)*(currentBeliefPropParamsConstMem.propChangeMoveNextLevel*currentBeliefPropParamsConstMem.currentMoveIncrementX);
		#elif (RANGE_BOUNDARIES_SETTING == ADJUST_RANGE_BOUND_FOR_EST_VAL)
			outputStartPossMovePixelParam.x = outputStartPossMovePixelParam.x - 
					(currentBeliefPropParamsConstMem.propChangeMoveNextLevel*currentBeliefPropParamsConstMem.currentMoveIncrementX) *
					(ceil((currentEndXMove - inputEndXMove) / 
					(currentBeliefPropParamsConstMem.propChangeMoveNextLevel*currentBeliefPropParamsConstMem.currentMoveIncrementX)));
		#endif 
	}
	if (currentEndYMove > inputEndYMove)
	{
		#if (RANGE_BOUNDARIES_SETTING == DONT_ADJUST_RANGE_BOUND_FOR_EST_VAL)
				outputStartPossMovePixelParam.y = inputEndYMove - (currentBeliefPropParamsConstMem.totalNumMovesYDir-1)*(currentBeliefPropParamsConstMem.propChangeMoveNextLevel*currentBeliefPropParamsConstMem.currentMoveIncrementY);
		#elif (RANGE_BOUNDARIES_SETTING == ADJUST_RANGE_BOUND_FOR_EST_VAL)
			outputStartPossMovePixelParam.y = outputStartPossMovePixelParam.y - 
					(currentBeliefPropParamsConstMem.propChangeMoveNextLevel*currentBeliefPropParamsConstMem.currentMoveIncrementY) *
					(ceil((currentEndYMove - inputEndYMove) / 
					(currentBeliefPropParamsConstMem.propChangeMoveNextLevel*currentBeliefPropParamsConstMem.currentMoveIncrementY)));
		#endif 
	}

	return outputStartPossMovePixelParam;
}


//use the floating-point 2-D indices and the current 2-D array to retrieve the current value
//using bilinear interpolation
__device__ float retrieve2DArrayValFromIndex(float xIndex, float yIndex, int width2DArray, int height2DArray, float current2DArray[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y])
{
	//retrieve the "floor" and "ceiling" of the x and y indices in order to retrieve the whole-number indices to use
	int xIndexFloor = (int)floor(xIndex);
	int yIndexFloor = (int)floor(yIndex);

	int xIndexCeil = (int)ceil(xIndex);
	int yIndexCeil = (int)ceil(yIndex);

	//retrieve the four points of the set of points used for interpolation in order to retrieve the desired value
	float floorXFloorYPoint = current2DArray[retrieveIndexCurrentPixel2DGrid(xIndexFloor, yIndexFloor, width2DArray, height2DArray)];
	float floorXCeilYPoint = current2DArray[retrieveIndexCurrentPixel2DGrid(xIndexFloor, yIndexCeil, width2DArray, height2DArray)];
	float ceilXFloorYPoint = current2DArray[retrieveIndexCurrentPixel2DGrid(xIndexCeil, yIndexFloor, width2DArray, height2DArray)];
	float ceilXCeilYPoint = current2DArray[retrieveIndexCurrentPixel2DGrid(xIndexCeil, yIndexCeil, width2DArray, height2DArray)];

#if (COPY_MESS_VAL_PASS_SCHEME == USE_INTERPOLATED_MESSAGE_VAL)

	//declare the average "bottom" and "top" values
	float avgYFloor;
	float avgYCeil;

	//set ceil and floor x values equal if they are the same
	if (xIndexFloor == xIndexCeil)
	{
		avgYFloor = floorXFloorYPoint;
		avgYCeil = floorXCeilYPoint;
	}
	else
	{
		//now retrieve the weighted values in the y floor and ceiling regions using the x index
		avgYFloor = (ceil(xIndex) - xIndex)*floorXFloorYPoint + (xIndex - floor(xIndex))*ceilXFloorYPoint;
		avgYCeil = (ceil(xIndex) - xIndex)*floorXCeilYPoint + (xIndex - floor(xIndex))*ceilXCeilYPoint;
	}

	//now retrieve the weighted average the two computed averages to retrieve the final interpolated value
	float averageOfYFloorCeilAverages;

	if (yIndexFloor == yIndexCeil)
	{
		averageOfYFloorCeilAverages = avgYFloor;
	}
	else
	{
		averageOfYFloorCeilAverages = (ceil(yIndex) - yIndex)*avgYFloor + (yIndex - floor(yIndex))*avgYCeil;
	}

	//now return the final interpolated value
	return averageOfYFloorCeilAverages;

#elif (COPY_MESS_VAL_PASS_SCHEME == USE_MIN_MESSAGE_VAL)

	return (min(floorXFloorYPoint, min(floorXCeilYPoint, min(ceilXFloorYPoint, ceilXCeilYPoint))));

#elif (COPY_MESS_VAL_PASS_SCHEME == USE_MAX_MESSAGE_VAL)

	return (max(floorXFloorYPoint, max(floorXCeilYPoint, max(ceilXFloorYPoint, ceilXCeilYPoint))));

#elif (COPY_MESS_VAL_PASS_SCHEME == RESET_MESSAGE_VAL_TO_DEFAULT)

	return DEFAULT_INITIAL_MESSAGE_VAL;

#endif //COPY_MESS_VAL_PASS_SCHEME
}


//retrieve the "best movement/disparity" from image 1 to image 2 for each pixel in parallel
//the message values and data values are all stored in textures
//use message values for current nodes
__global__ void retrieveOutputMovementCheckerboard(float* movementXBetweenImagesCurrentCheckerboardDevice, float* movementYBetweenImagesCurrentCheckerboardDevice, int widthLevel, int heightLevel)
{
	//retrieve the x and y values corresponding to the current pixel within the checkerboard
	int xVal = blockIdx.x * blockDim.x + threadIdx.x;
	int yVal = blockIdx.y * blockDim.y + threadIdx.y;

	int widthCheckerboard = widthLevel/2;
	int heightCheckerboard = heightLevel;

	if (withinImageBounds(xVal, yVal, widthCheckerboard, heightCheckerboard))
	{
		if ((xVal >= 0) && (xVal <= (widthCheckerboard- 1)) && (yVal >= 0) && (yVal <= (heightCheckerboard - 1)))
		{
			//retrieve the starting movement parameters for the current pixel
			currentStartMoveParamsPixelAtLevel currentStartMovePixelParams = tex1Dfetch(currentPixParamsTexCurrentCheckerboard, retrieveIndexCurrentPixel2DGrid(xVal, yVal,
					widthCheckerboard, heightCheckerboard));

			// keep track of "best" movement for current pixel
			int bestMovementXY = 0;
			float best_val = INF_BP;
			for (int movementXY = 0; movementXY < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementXY++)
			{
				float val = tex1Dfetch(messageUTexCurrReadCheckerboard, retrieveIndexInDataAndMessage(xVal, yVal, widthCheckerboard, heightCheckerboard, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir))) + 
					tex1Dfetch(messageDTexCurrReadCheckerboard, retrieveIndexInDataAndMessage(xVal, yVal, widthCheckerboard, heightCheckerboard, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir))) + 
					tex1Dfetch(messageLTexCurrReadCheckerboard, retrieveIndexInDataAndMessage(xVal, yVal, widthCheckerboard, heightCheckerboard, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir))) + 
					tex1Dfetch(messageRTexCurrReadCheckerboard, retrieveIndexInDataAndMessage(xVal, yVal, widthCheckerboard, heightCheckerboard, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir))) +
					tex1Dfetch(dataCostsCurrCheckerboard, retrieveIndexInDataAndMessage(xVal, yVal, widthCheckerboard, heightCheckerboard, movementXY, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir)));
			if (val < (best_val)) 
				{
					best_val = val;
					bestMovementXY = movementXY;
				}
			}

			//use the index value of the best movement along with the min movement in the x and y directions and the increment to retrieve the final movement
			movementXBetweenImagesCurrentCheckerboardDevice[yVal*widthCheckerboard + xVal] = currentStartMovePixelParams.x + ((float)(bestMovementXY % currentBeliefPropParamsConstMem.totalNumMovesXDir)) * currentBeliefPropParamsConstMem.currentMoveIncrementX; 
			movementYBetweenImagesCurrentCheckerboardDevice[yVal*widthCheckerboard + xVal] = currentStartMovePixelParams.y + ((float)(bestMovementXY / currentBeliefPropParamsConstMem.totalNumMovesXDir)) * currentBeliefPropParamsConstMem.currentMoveIncrementY;
		}
		else
		{
			movementXBetweenImagesCurrentCheckerboardDevice[yVal*widthCheckerboard + xVal] = 0.0f;
			movementYBetweenImagesCurrentCheckerboardDevice[yVal*widthCheckerboard + xVal] = 0.0f;
		}
	}
}




//device function to join together two float 2D array checkerboard portions for output as a single 2D array
__global__ void joinCheckerboardPortions(float* checkerboardPortion1Device, float* checkerboardPortion2Device, int widthLevel, int heightLevel, float* outputFloatArray)
{
	//retrieve the x and y values corresponding to the current thread
	int xVal = blockIdx.x*blockDim.x + threadIdx.x;
	int yVal = blockIdx.y*blockDim.y + threadIdx.y;

	//set the width and height of the checkerboard at the current level
	int widthCheckerboardLevel = widthLevel / 2;
	int heightCheckerboardLevel = heightLevel;

	//retrieve the x and y indices within each checkerboard
	int xValCheckerboard = xVal / 2;
	int yValCheckerboard = yVal;

	//if current pixel is within image bounds, then retrieve appropriate value within checkerboard and place in output 2D array
	if (withinImageBounds(xVal, yVal, widthLevel, heightLevel))
	{
		//if output float array is odd-sized, then values in last column don't correspond to either checkerboard and
		//are set to a default value
		if ((widthLevel%2 == 1) && (xVal == (widthLevel - 1)))
		{
			outputFloatArray[retrieveIndexCurrentPixel2DGrid(xVal, yVal, widthLevel, heightLevel)] = 
				DEFAULT_OUTPUT_FLOAT_ARRAY_VAL;
		}
		//retrieve the checkerboard portion
		
		//if true, then in part 1 of checkerboard
		if (((xVal + yVal)%2) == 0)
		{
			outputFloatArray[retrieveIndexCurrentPixel2DGrid(xVal, yVal, widthLevel, heightLevel)] = 
				checkerboardPortion1Device[retrieveIndexCurrentPixel2DGrid(xValCheckerboard, yValCheckerboard, widthCheckerboardLevel, heightCheckerboardLevel)];
		}
		else //in part 2 of checkerboard
		{
			outputFloatArray[retrieveIndexCurrentPixel2DGrid(xVal, yVal, widthLevel, heightLevel)] = 
				checkerboardPortion2Device[retrieveIndexCurrentPixel2DGrid(xValCheckerboard, yValCheckerboard, widthCheckerboardLevel, heightCheckerboardLevel)];
		}
	}
}


//device function to round each floating point value in an array to an whole number
__global__ void roundDeviceValsKernel(float* inputValsDevice, int widthVals, int heightVals, float* outputValsDevice)
{
	int xVal = blockIdx.x*blockDim.x + threadIdx.x;
	int yVal = blockIdx.y*blockDim.y + threadIdx.y;

	//if current pixel is within image bounds, then retrieve appropriate value within checkerboard and place in output 2D array
	if (withinImageBounds(xVal, yVal, widthVals, heightVals))
	{
		//add 0.5f and truncate if input value is positive; subtract 0.5f and truncate if input value is negative
		if (inputValsDevice[retrieveIndexCurrentPixel2DGrid(xVal, yVal, widthVals, heightVals)] > 0.0f)
		{
			outputValsDevice[retrieveIndexCurrentPixel2DGrid(xVal, yVal, widthVals, heightVals)] =
				(floor(inputValsDevice[retrieveIndexCurrentPixel2DGrid(xVal, yVal, widthVals, heightVals)] + 0.5f));
		}
		else
		{
			outputValsDevice[retrieveIndexCurrentPixel2DGrid(xVal, yVal, widthVals, heightVals)] =
				(floor(inputValsDevice[retrieveIndexCurrentPixel2DGrid(xVal, yVal, widthVals, heightVals)] + 0.5f));
		}
	}
}

//global function to retrieve the parameters at the 'higher' level of the pyramid...
//this is performed from the perspective of the `current' pixel, retrieving parameters from the `lower level'...
__global__ void retParamsHigherLevPyramid(currentStartMoveParamsPixelAtLevel* paramsCheckerboard1EachHierarchLevel, currentStartMoveParamsPixelAtLevel* paramsCheckerboard2EachHierarchLevel, int widthPrevLevCheckerboard, int heightPrevLevel, int widthCurrLevCheckerboard, int heightCurrLevel, size_t paramsOffsetPrevLev, size_t paramsOffsetCurrLevel, checkerboardPortionEnum checkerboardPart)
{
	int xVal = blockIdx.x*blockDim.x + threadIdx.x;
	int yVal = blockIdx.y*blockDim.y + threadIdx.y;

	
	//if current pixel is within image bounds in the 'current level', then retrieve appropriate parameters from the 'lower level' within checkerboard and place in output 2D array
	if (withinImageBounds(xVal, yVal, widthCurrLevCheckerboard, heightCurrLevel))
	{
		//set the pointers to the `current' checkerboard being updated...
		currentStartMoveParamsPixelAtLevel* pointerCurrCheckerboardUpdate;
	
		//retrieve the `checkerboard adjustment' for the current part...
		int checkerboardPartAdjustment;
		if (checkerboardPart == CHECKERBOARD_PART_1_ENUM)
		{
			pointerCurrCheckerboardUpdate = paramsCheckerboard1EachHierarchLevel;
			checkerboardPartAdjustment = (yVal%2);
		}
		else if (checkerboardPart == CHECKERBOARD_PART_2_ENUM)
		{
			pointerCurrCheckerboardUpdate = paramsCheckerboard2EachHierarchLevel;
			checkerboardPartAdjustment = ((yVal+1)%2);
		}

		//retrieve the values `from' for both checkerboard portions...
		int paramsIndexFrom = paramsOffsetPrevLev + (yVal*2)*widthPrevLevCheckerboard +  (xVal*2 + checkerboardPartAdjustment); 

		//initialize the array with the 4 corresponding values...
		currentStartMoveParamsPixelAtLevel paramsCorrLowerLevel[4];
		paramsCorrLowerLevel[0] = paramsCheckerboard1EachHierarchLevel[paramsIndexFrom];
		paramsCorrLowerLevel[1] = paramsCheckerboard2EachHierarchLevel[paramsIndexFrom];

		paramsIndexFrom = paramsOffsetPrevLev + (yVal*2 + 1)*widthPrevLevCheckerboard +  (xVal*2 + checkerboardPartAdjustment);

		paramsCorrLowerLevel[2] = paramsCheckerboard1EachHierarchLevel[paramsIndexFrom];
		paramsCorrLowerLevel[3] = paramsCheckerboard2EachHierarchLevel[paramsIndexFrom];

		//average the starting x and y movements from the `lower' level...
		float xStartMove = (paramsCorrLowerLevel[0].x + paramsCorrLowerLevel[1].x + paramsCorrLowerLevel[2].x + paramsCorrLowerLevel[3].x) / 4.0f;
		float yStartMove = (paramsCorrLowerLevel[0].y + paramsCorrLowerLevel[1].y + paramsCorrLowerLevel[2].y + paramsCorrLowerLevel[3].y) / 4.0f;

		//retrieve the parameter of the corresponding parameter at the 'lower' level
		//retrieve the index `to'
		int paramsIndexTo = paramsOffsetCurrLevel + (yVal)*widthCurrLevCheckerboard + (xVal); 

		//set the parameter of the movement at the next level...
		pointerCurrCheckerboardUpdate[paramsIndexTo].x = xStartMove;
		pointerCurrCheckerboardUpdate[paramsIndexTo].y = yStartMove;
	}
}


//global function to retrieve the data costs for the current level of the computation pyramid...
__global__ void retDataCostsCurrLevCompPyramid(float* dataCostsCurrentLevelCurrentCheckerboard, currentStartMoveParamsPixelAtLevel* paramsLevelCurrCheckerboard, int widthImageAtLevel, int heightImageAtLevel, int offsetIntoParams, checkerboardPortionEnum checkerboardPart)
{	
	int xValThread = blockIdx.x * blockDim.x + threadIdx.x;
	int yValThread = blockIdx.y * blockDim.y + threadIdx.y;

	//retrieve the width and height of the checkerboard by dividing the width
	//and setting the height of the checkerboard to the image height
	int widthCheckerboard = widthImageAtLevel / 2;
	int heightCheckerboard = heightImageAtLevel;

	//check to make sure that x and y are within the "checkerboard"
	if (withinImageBounds(xValThread, yValThread, widthCheckerboard, heightCheckerboard))
	{
		//used to adjust the pixel value based on the checkerboard portion
		int checkerboardPartAdjustment;

		//retrieve the index of the current pixel using the thread and current checkerboard portion
		if (checkerboardPart == CHECKERBOARD_PART_1_ENUM)
		{
			checkerboardPartAdjustment = (yValThread%2);
		}
		else if (checkerboardPart == CHECKERBOARD_PART_2_ENUM)
		{
			checkerboardPartAdjustment = ((yValThread+1)%2);
		}

		//declare and define the x and y indices of the current pixel within the "level image"
		int xIndexPixel = xValThread*2 + checkerboardPartAdjustment;
		int yIndexPixel = yValThread;


		//retrieve the width and the height of the full image via the BPSettings in constant memory
		int widthFullImage = currentBeliefPropParamsConstMem.widthImages;
		int heightFullImage = currentBeliefPropParamsConstMem.heightImages;

		//retrieve the sampling at the current level using the width of the full images and the width of the images at the current level
		//TODO: look into using "smaller" versions of the images (like mipmapping...)
		float samplingXDir = (float)widthFullImage / (float)widthImageAtLevel;
		float samplingYDir = (float)heightFullImage / (float)heightImageAtLevel;

		//retrieve the middle x-value and y-value in the image using the thread x and y values and also the sampling in each direction
		float midXValImage = ((float)xIndexPixel) * (samplingXDir) + samplingXDir / 2.0f;
		float midYValImage = ((float)yIndexPixel) * (samplingYDir) + samplingYDir / 2.0f;

		//use the sampling increment in the x and y directions for the extent of the data costs from the midpoint
		//subtract each by 1 since going from mid-point to mid-point
		float extentDataCostsSumAcross = floor(samplingXDir + 0.5f) - 1.0f;
		float extentDataCostsSumVertical = floor(samplingYDir + 0.5f) - 1.0f;

		int indexVal;

		if (withinImageBounds(midXValImage, midYValImage, widthFullImage, heightFullImage))
		{
			//retrieve the current min and max movements in the x and y directions using the current paramsCurrentPixelAtLevel
			//at the current pixel as well as the global currentParamsAllPixAtLevelConstMem which applies to all pixels at the
			//current level
			//retrieve the current paramsCurrentPixelAtLevel for the desired pixel

			currentStartMoveParamsPixelAtLevel currentStartMovePixelParams;

			//check the current "checkerboard" and retrieve the current parameters from the texture assumed to be bound to the
			//appropriate values
			currentStartMovePixelParams = paramsLevelCurrCheckerboard[offsetIntoParams + yValThread*widthCheckerboard + xValThread];
			
			//now use the given parameters to retrieve the min and max movements
			float maxMovementLeft;
			float maxMovementRight;
			float maxMovementUp;
			float maxMovementDown;

			if (currentBeliefPropParamsConstMem.directionPosMovement == POS_X_MOTION_RIGHT_POS_Y_MOTION_DOWN)
			{
				maxMovementLeft = -1.0f*(currentStartMovePixelParams.x);
				maxMovementRight = (currentStartMovePixelParams.x + currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.currentMoveIncrementX);
				maxMovementUp = -1.0f*(currentStartMovePixelParams.y);
				maxMovementDown = (currentStartMovePixelParams.y + currentBeliefPropParamsConstMem.totalNumMovesYDir * currentBeliefPropParamsConstMem.currentMoveIncrementY);
			}
			else if (currentBeliefPropParamsConstMem.directionPosMovement == POS_X_MOTION_LEFT_POS_Y_MOTION_UP)
			{
				maxMovementLeft = (currentStartMovePixelParams.x + currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.currentMoveIncrementX);
				maxMovementRight = -1.0f*(currentStartMovePixelParams.x);
				maxMovementUp = (currentStartMovePixelParams.y + currentBeliefPropParamsConstMem.totalNumMovesYDir * currentBeliefPropParamsConstMem.currentMoveIncrementY);
				maxMovementDown = -1.0f*(currentStartMovePixelParams.y);
			}
			else
			{
				maxMovementLeft = 0.0f;
				maxMovementRight = 0.0f;
				maxMovementUp = 0.0f;
				maxMovementDown = 0.0f;
			}

			//set the current index of the movement to 0
			int movementXYVal = 0;

			if ( (((midXValImage - extentDataCostsSumAcross/2.0f) - maxMovementLeft) >= 0) && (((midXValImage + extentDataCostsSumAcross/2.0f) + maxMovementRight) < (currentBeliefPropParamsConstMem.widthImages)) &&
				(((midYValImage - extentDataCostsSumVertical/2.0f) - maxMovementUp) >= 0) && (((midYValImage + extentDataCostsSumVertical/2.0f) + maxMovementDown) < (currentBeliefPropParamsConstMem.heightImages)))
			{
				//set current movement to the min movement in the x and y directions and then increase by current move increment in the loop
				float currentXMove = currentStartMovePixelParams.x;
				float currentYMove = currentStartMovePixelParams.y;


				//go through entire range of movements in the x and y directions
				for (int numMoveYInRange = 0; numMoveYInRange < (currentBeliefPropParamsConstMem.totalNumMovesYDir); numMoveYInRange++)
				{
					//reset the current x movement to the minimum x movement
					currentXMove = currentStartMovePixelParams.x;

					for (int numMoveXInRange = 0; numMoveXInRange < (currentBeliefPropParamsConstMem.totalNumMovesXDir); numMoveXInRange++)
					{
						
						float currentPixelImage1;
						float currentPixelImage2;

						//use the thread indices for the index value
						indexVal = retrieveIndexInDataAndMessage(xValThread, yValThread, widthCheckerboard, heightCheckerboard, movementXYVal, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));


						//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
						//save the data cost for the current pixel on the current "checkerboard"
						//initialize each data cost to 0 then add each value within the current range of data values to add up
						dataCostsCurrentLevelCurrentCheckerboard[indexVal] = 0.0f;


						//loop through all the pixels in the current range where the data costs are being summed
						for (float xPixLocation = (midXValImage - extentDataCostsSumAcross/2.0f); xPixLocation <= (midXValImage + extentDataCostsSumAcross/2.0f + SMALL_VALUE); xPixLocation += 1.0f)
						{
							for (float yPixLocation = (midYValImage - extentDataCostsSumVertical/2.0f); yPixLocation <= (midYValImage + extentDataCostsSumVertical/2.0f + SMALL_VALUE); yPixLocation += 1.0f)
							{
								float currMinDataCost = INF_BP;

								//declare the variable for the x and y moves to be tested
								float xMoveTest;
								float yMoveTest;

								//declare the variables which declare the "range" to check around the current move and initialize it to 0
								float xMoveRange = 0.0f;
								float yMoveRange = 0.0f;

								#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
									//declare the variables for the bounds of the change in the min and max moves in the x/y range
									float xMoveChangeBounds = 0.0f;
									float yMoveChangeBounds = 0.0f;
								#endif //CLAMP_EDGE_MOVES_DATA_COST_SETTING


								//if current move increment is beyond a certain point, then check various possible movements
								//perform sampling-invarient data costs beyond a certain move increment
								if ((currentBeliefPropParamsConstMem.currentMoveIncrementX > (currentBeliefPropParamsConstMem.motionIncBotLevX)) || (currentBeliefPropParamsConstMem.currentMoveIncrementY > (currentBeliefPropParamsConstMem.motionIncBotLevY)))
								{
									//go through the rand in increments of 0.5f for each pixel and take the minimum cost
									//if y range is greater than 1, then check values "around" for sampling-invarient data costs
									if (currentBeliefPropParamsConstMem.totalNumMovesXDir > 1)
									{
										//xMoveRange = currentBeliefPropParamsConstMem.currentMoveIncrement / 2.0f;
										float numXMoves = ceil((currentBeliefPropParamsConstMem.currentMoveIncrementX / 2.0f) / currentBeliefPropParamsConstMem.motionIncBotLevX);
										xMoveRange = numXMoves * currentBeliefPropParamsConstMem.motionIncBotLevX;

										#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
											xMoveChangeBounds = currentBeliefPropParamsConstMem.currentMoveIncrementX / 2.0f;
										#endif //CLAMP_EDGE_MOVES_DATA_COST_SETTING
									}

									if (currentBeliefPropParamsConstMem.totalNumMovesYDir > 1)
									{
										//yMoveRange = currentBeliefPropParamsConstMem.currentMoveIncrement / 2.0f;
										float numYMoves = ceil((currentBeliefPropParamsConstMem.currentMoveIncrementY / 2.0f) / currentBeliefPropParamsConstMem.motionIncBotLevY);
										yMoveRange = numYMoves * currentBeliefPropParamsConstMem.motionIncBotLevY;
										
										#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
											yMoveChangeBounds = currentBeliefPropParamsConstMem.currentMoveIncrementY / 2.0f;
										#endif //CLAMP_EDGE_MOVES_DATA_COST_SETTING
									}
								}

								//go through and retrieve the data costs of each move in the given range
								for (yMoveTest = (-1.0f * yMoveRange); yMoveTest <= (yMoveRange + SMALL_VALUE); yMoveTest += currentBeliefPropParamsConstMem.motionIncBotLevY)
								{
									for (xMoveTest = (-1.0f * xMoveRange); xMoveTest <= (xMoveRange + SMALL_VALUE); xMoveTest += currentBeliefPropParamsConstMem.motionIncBotLevX)
									{
										//declare the variables for the current x and y moves and initialize to the current move (clamped if that's the setting...)
										#if (CLAMP_EDGE_MOVES_DATA_COST_SETTING == DONT_CLAMP_EDGE_MOVES_DATA_COST)
											float currXMove = currentXMove + xMoveTest;
											float currYMove = currentYMove + yMoveTest;
										#elif (CLAMP_EDGE_MOVES_DATA_COST_SETTING == CLAMP_EDGE_MOVES_DATA_COST)
											float currXMove = currentXMove + max(-1.0f * xMoveChangeBounds, min(xMoveTest, xMoveChangeBounds));
											float currYMove = currentYMove + max(-1.0f * yMoveChangeBounds, min(yMoveTest, yMoveChangeBounds));
										#endif//CLAMP_EDGE_MOVES_DATA_COST_SETTING


										if (currentBeliefPropParamsConstMem.directionPosMovement == POS_X_MOTION_RIGHT_POS_Y_MOTION_DOWN)
										{
											//texture access is using normalized coordinate system
											currentPixelImage1 = tex2D(image1PixelsTexture, (((float)xPixLocation) / (float)widthFullImage), (((float)yPixLocation) / (float)heightFullImage));
											currentPixelImage2 = tex2D(image2PixelsTexture, (((float)(xPixLocation + currXMove))) / ((float)widthFullImage), (((float)(yPixLocation + currYMove))) / ((float)heightFullImage));
										}
										else if (currentBeliefPropParamsConstMem.directionPosMovement == POS_X_MOTION_LEFT_POS_Y_MOTION_UP)
										{
											currentPixelImage1 = tex2D(image1PixelsTexture, (((float)xPixLocation) / (float)widthFullImage), (((float)yPixLocation) / (float)heightFullImage));
											currentPixelImage2 = tex2D(image2PixelsTexture, (((float)(xPixLocation - currXMove))) / ((float)widthFullImage), ((((float)(yPixLocation - currYMove))) / ((float)heightFullImage)));
										}
										else
										{
											currentPixelImage1 = 0.0f;
											currentPixelImage2 = 0.0f;
										}

										currMinDataCost = min(currMinDataCost, currentBeliefPropParamsConstMem.dataCostWeight * min(abs(currentPixelImage1 - currentPixelImage2), currentBeliefPropParamsConstMem.dataCostCap));
									}
								}

								//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
								//save the data cost for the current pixel on the current "checkerboard"
								dataCostsCurrentLevelCurrentCheckerboard[indexVal] += currMinDataCost;
							}
						}

						movementXYVal++;

						//increment the x-movement by current movement increment
						currentXMove += currentBeliefPropParamsConstMem.currentMoveIncrementX;
					}

					//increment the y-movement by current movement increment
					currentYMove += currentBeliefPropParamsConstMem.currentMoveIncrementY;
				}

			}
			else
			{
				//go through entire range of movements in the x and y directions and set data cost to default value
				for (int movementXYVal = 0; movementXYVal < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementXYVal++)
				{
					indexVal = retrieveIndexInDataAndMessage(xValThread, yValThread, widthCheckerboard, heightCheckerboard, movementXYVal, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));

					//set each data cost to a default value if outside of range
					dataCostsCurrentLevelCurrentCheckerboard[indexVal] = currentBeliefPropParamsConstMem.dataCostWeight * DEFAULT_DATA_COST_VALUE;
				}
			}
		}
		else
		{
			//go through entire range of movements in the x and y directions and set data cost to default value
			for (int movementXYVal = 0; movementXYVal < (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir); movementXYVal++)
			{
				indexVal = retrieveIndexInDataAndMessage(xValThread, yValThread, widthCheckerboard, heightCheckerboard, movementXYVal, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));

				//set each data cost to a default value if outside of range
				dataCostsCurrentLevelCurrentCheckerboard[indexVal] = currentBeliefPropParamsConstMem.dataCostWeight * DEFAULT_DATA_COST_VALUE;
			}
		}
	}
}

//device function to compute the values in the next level of the computation pyramid
__device__ void compValsNextLevelCompPyramid(float prevMessValsU[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float prevMessValsD[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float prevMessValsL[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float prevMessValsR[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], currentStartMoveParamsPixelAtLevel currentMovementStartPrevLevPix, currentStartMoveParamsPixelAtLevel currentMovementStartNextLevPix,	float currMessValsU[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float currMessValsD[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float currMessValsL[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float currMessValsR[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y])
{
	//assuming the number of movements in each hierarchical level is constant for now...
		
	//retrieve the starting index of the new starting movement in the x and y directions within the previous
	//movement range
	float startXMovePrevMessLev = (currentMovementStartNextLevPix.x - currentMovementStartPrevLevPix.x) /
		(currentBeliefPropParamsConstMem.currentMoveIncrementX);
	float startYMovePrevMessLev = (currentMovementStartNextLevPix.y - currentMovementStartPrevLevPix.y) /
		(currentBeliefPropParamsConstMem.currentMoveIncrementY);

	//declare the current x and y indices that represent the location in the "previous message" level to copy
	//into the current message level
	float currentXIndexPrevMessLev = startXMovePrevMessLev;
	float currentYIndexPrevMessLev = startYMovePrevMessLev;

	//initialize the `current move num' to 0...
	int currentMoveNum = 0;

	for (int newMoveYNum = 0; newMoveYNum < currentBeliefPropParamsConstMem.totalNumMovesYDir; newMoveYNum++)
	{
		//reset the current x within the previous message with each row
		currentXIndexPrevMessLev = startXMovePrevMessLev;

		for (int newMoveXNum = 0; newMoveXNum < currentBeliefPropParamsConstMem.totalNumMovesXDir; newMoveXNum++)
		{
			//retrieve the interpolated message values from the previous message values to copy into the message values for the "next" level
			float interpolatedPrevValU = retrieve2DArrayValFromIndex(currentXIndexPrevMessLev, currentYIndexPrevMessLev, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir, prevMessValsU);
			float interpolatedPrevValD = retrieve2DArrayValFromIndex(currentXIndexPrevMessLev, currentYIndexPrevMessLev, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir, prevMessValsD);
			float interpolatedPrevValL = retrieve2DArrayValFromIndex(currentXIndexPrevMessLev, currentYIndexPrevMessLev, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir, prevMessValsL);
			float interpolatedPrevValR = retrieve2DArrayValFromIndex(currentXIndexPrevMessLev, currentYIndexPrevMessLev, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir, prevMessValsR);

		
			//place each of the values to `copy to' in the desired locations...
			currMessValsU[currentMoveNum] = prevMessValsU[currentMoveNum];//interpolatedPrevValU;
			currMessValsD[currentMoveNum] = prevMessValsD[currentMoveNum];//interpolatedPrevValD;
			currMessValsL[currentMoveNum] = prevMessValsL[currentMoveNum];//interpolatedPrevValL;
			currMessValsR[currentMoveNum] = prevMessValsR[currentMoveNum];//interpolatedPrevValR;


			//increment the current "move number" which increases across then vertically
			currentMoveNum++;

			//use the defined proportion that the movement is adjusted between the current and next level to adjust the current
			//x and y index
			currentXIndexPrevMessLev += currentBeliefPropParamsConstMem.propChangeMoveNextLevel;
		}

		//use the defined proportion that the movement is adjusted between the current and next level to adjust the current
		//x and y index
		currentYIndexPrevMessLev += currentBeliefPropParamsConstMem.propChangeMoveNextLevel;
	}
}


//device function to compute the values in the next level of the computation pyramid
__device__ void compValsSameLevNextInHierarchPyr(float prevMessValsU[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float prevMessValsD[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float prevMessValsL[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float prevMessValsR[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], currentStartMoveParamsPixelAtLevel currentMovementStartPrevLevPix, currentStartMoveParamsPixelAtLevel currentMovementStartNextLevPix,	float currMessValsU[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float currMessValsD[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float currMessValsL[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y], float currMessValsR[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y])
{
	//assuming the number of movements in each hierarchical level is constant for now...
		
	//retrieve the starting index of the new starting movement in the x and y directions within the previous
	//movement range
	float startXMovePrevMessLev = (currentMovementStartNextLevPix.x - currentMovementStartPrevLevPix.x) /
		(currentBeliefPropParamsConstMem.currentMoveIncrementX);
	float startYMovePrevMessLev = (currentMovementStartNextLevPix.y - currentMovementStartPrevLevPix.y) /
		(currentBeliefPropParamsConstMem.currentMoveIncrementY);

	//declare the current x and y indices that represent the location in the "previous message" level to copy
	//into the current message level
	float currentXIndexPrevMessLev = startXMovePrevMessLev;
	float currentYIndexPrevMessLev = startYMovePrevMessLev;

	//initialize the `current move num' to 0...
	int currentMoveNum = 0;

	for (int newMoveYNum = 0; newMoveYNum < currentBeliefPropParamsConstMem.totalNumMovesYDir; newMoveYNum++)
	{
		//reset the current x within the previous message with each row
		currentXIndexPrevMessLev = startXMovePrevMessLev;

		for (int newMoveXNum = 0; newMoveXNum < currentBeliefPropParamsConstMem.totalNumMovesXDir; newMoveXNum++)
		{
			//retrieve the interpolated message values from the previous message values to copy into the message values for the "next" level
			float interpolatedPrevValU = retrieve2DArrayValFromIndex(currentXIndexPrevMessLev, currentYIndexPrevMessLev, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir, prevMessValsU);
			float interpolatedPrevValD = retrieve2DArrayValFromIndex(currentXIndexPrevMessLev, currentYIndexPrevMessLev, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir, prevMessValsD);
			float interpolatedPrevValL = retrieve2DArrayValFromIndex(currentXIndexPrevMessLev, currentYIndexPrevMessLev, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir, prevMessValsL);
			float interpolatedPrevValR = retrieve2DArrayValFromIndex(currentXIndexPrevMessLev, currentYIndexPrevMessLev, currentBeliefPropParamsConstMem.totalNumMovesXDir, currentBeliefPropParamsConstMem.totalNumMovesYDir, prevMessValsR);

		
			//place each of the values to `copy to' in the desired locations...
			currMessValsU[currentMoveNum] = interpolatedPrevValU;
			currMessValsD[currentMoveNum] = interpolatedPrevValD;
			currMessValsL[currentMoveNum] = interpolatedPrevValL;
			currMessValsR[currentMoveNum] = interpolatedPrevValR;


			//increment the current "move number" which increases across then vertically
			currentMoveNum++;

			//use the defined proportion that the movement is adjusted between the current and next level to adjust the current
			//x and y index
			currentXIndexPrevMessLev += 1.0f; //currentBeliefPropParamsConstMem.propChangeMoveNextLevel;
		}

		//use the defined proportion that the movement is adjusted between the current and next level to adjust the current
		//x and y index
		currentYIndexPrevMessLev += 1.0f;// currentBeliefPropParamsConstMem.propChangeMoveNextLevel;
	}

}

//global function to copy the values `down' in the next level of the computation pyramid...
__global__ void copyMessValsDownCompPyramid(checkerboardMessagesDeviceStruct messagesDevicePrevCurrCheckerboard,
						checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboard1,
						checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboard2,
						int widthPrevLevel, int heightPrevLevel, 
						int widthNextLevel, int heightNextLevel,
						currentStartMoveParamsPixelAtLevel* paramsDeviceCheckerboard1,
						currentStartMoveParamsPixelAtLevel* paramsDeviceCheckerboard2, 
						size_t paramsOffsetPrevLev, size_t paramsOffsetCurrLevel, 
						checkerboardPortionEnum checkerboardPart)
{
	//retrieve the x and y indices of the thread
	//retrieve the parameters for the `previous' level
	int xVal = blockIdx.x * blockDim.x + threadIdx.x;
	int yVal = blockIdx.y * blockDim.y + threadIdx.y;

	//retrieve the width/height of the prev and next level checkerboards...
	int widthPrevLevelCheckerboard = widthPrevLevel / 2;
	int heightPrevLevelCheckerboard = heightPrevLevel;

	int widthNextLevelCheckerboard = widthNextLevel / 2;
	int heightNextLevelCheckerboard = heightNextLevel;

	
	 

	

	//going from perspective of `previous level', so checking that...
	if (withinImageBounds(xVal, yVal, widthPrevLevelCheckerboard, heightPrevLevelCheckerboard))
	{
		//initialize the variable for the parameters for the `current' level
		currentStartMoveParamsPixelAtLevel paramsCurrentLevel;

		//retrieve the movement parameters for the `previous' level...need to find 'correct' checkerboard...
		currentStartMoveParamsPixelAtLevel paramsPrevLevel;
		if (checkerboardPart == CHECKERBOARD_PART_1_ENUM)
		{
			paramsPrevLevel = paramsDeviceCheckerboard1[paramsOffsetPrevLev + yVal * widthPrevLevelCheckerboard + xVal];
		}
		else if (checkerboardPart == CHECKERBOARD_PART_2_ENUM)
		{
			paramsPrevLevel = paramsDeviceCheckerboard2[paramsOffsetPrevLev + yVal * widthPrevLevelCheckerboard + xVal];
		}

		//set the adjustment to make based on the `current checkerboard'...
		int checkerboardPartAdjustment;

		if (checkerboardPart == CHECKERBOARD_PART_1_ENUM)
		{
			checkerboardPartAdjustment = (yVal%2);
		}
		else if (checkerboardPart == CHECKERBOARD_PART_2_ENUM)
		{
			checkerboardPartAdjustment = ((yVal+1)%2);
		}

		//initialize the space for the 'previous message values'
		float prevMessValsU[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
		float prevMessValsD[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
		float prevMessValsL[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
		float prevMessValsR[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];

		//initialize the space for the 'current message values'...
		float currMessValsU[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
		float currMessValsD[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
		float currMessValsL[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];
		float currMessValsR[TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y];

		
		//retrieve the `previous' message values
		#pragma unroll 1
		for (int currentMovement = 0; currentMovement < TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y; currentMovement++)
		{
			int indexCopyFrom = retrieveIndexInDataAndMessage(xVal, yVal, widthPrevLevelCheckerboard, heightPrevLevelCheckerboard, currentMovement, TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y); 

			prevMessValsU[currentMovement] = messagesDevicePrevCurrCheckerboard.messageUDevice[indexCopyFrom];
			prevMessValsD[currentMovement] = messagesDevicePrevCurrCheckerboard.messageDDevice[indexCopyFrom];
			prevMessValsL[currentMovement] = messagesDevicePrevCurrCheckerboard.messageLDevice[indexCopyFrom];
			prevMessValsR[currentMovement] = messagesDevicePrevCurrCheckerboard.messageRDevice[indexCopyFrom];
		}

		


		//retrieve the parameters for the `current level'
		paramsCurrentLevel = paramsDeviceCheckerboard1[paramsOffsetCurrLevel + (yVal*2)*widthNextLevelCheckerboard + (xVal*2 + checkerboardPartAdjustment)];

		//retrieve the message values at the `current' level using the message values at the `previous level'
		/*compValsNextLevelCompPyramid*/ compValsSameLevNextInHierarchPyr(prevMessValsU, prevMessValsD, prevMessValsL, prevMessValsR, paramsPrevLevel, paramsCurrentLevel, currMessValsU, currMessValsD, currMessValsL, currMessValsR);

		for (int currMoveNum = 0; currMoveNum < TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y; currMoveNum++)
		{
			int indexCopyTo = retrieveIndexInDataAndMessage((xVal*2 + checkerboardPartAdjustment), (yVal*2), widthNextLevelCheckerboard, heightNextLevelCheckerboard, currMoveNum, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));

			messagesDeviceCurrentCheckerboard1.messageUDevice[indexCopyTo] = currMessValsU[currMoveNum];
			messagesDeviceCurrentCheckerboard1.messageDDevice[indexCopyTo] = currMessValsD[currMoveNum];
			messagesDeviceCurrentCheckerboard1.messageLDevice[indexCopyTo] = currMessValsL[currMoveNum];
			messagesDeviceCurrentCheckerboard1.messageRDevice[indexCopyTo] = currMessValsR[currMoveNum];

		}

		

		//retrieve the parameters for the `current level'
		paramsCurrentLevel = paramsDeviceCheckerboard2[paramsOffsetCurrLevel + (yVal*2)*widthNextLevelCheckerboard + (xVal*2 + checkerboardPartAdjustment)];
 
		//retrieve the message values for the `next level' of the computation hierarchy for four sets of message value (to copy `to')...
		/*compValsNextLevelCompPyramid*/ compValsSameLevNextInHierarchPyr(prevMessValsU, prevMessValsD, prevMessValsL, prevMessValsR, paramsPrevLevel, paramsCurrentLevel, currMessValsU, currMessValsD, currMessValsL, currMessValsR);
	
		for (int currMoveNum = 0; currMoveNum < TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y; currMoveNum++)
		{
			int indexCopyTo = retrieveIndexInDataAndMessage((xVal*2 + checkerboardPartAdjustment), (yVal*2), widthNextLevelCheckerboard, heightNextLevelCheckerboard, currMoveNum, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));

			messagesDeviceCurrentCheckerboard2.messageUDevice[indexCopyTo] = currMessValsU[currMoveNum];
			messagesDeviceCurrentCheckerboard2.messageDDevice[indexCopyTo] = currMessValsD[currMoveNum];
			messagesDeviceCurrentCheckerboard2.messageLDevice[indexCopyTo] = currMessValsL[currMoveNum];
			messagesDeviceCurrentCheckerboard2.messageRDevice[indexCopyTo] = currMessValsR[currMoveNum];
		}



		//retrieve the parameters for the `current level'
		paramsCurrentLevel = paramsDeviceCheckerboard1[paramsOffsetCurrLevel + (yVal*2 + 1)*widthNextLevelCheckerboard + (xVal*2 + checkerboardPartAdjustment)];

		//retrieve the message values for the `next level' of the computation hierarchy for four sets of message value (to copy `to')...
		/*compValsNextLevelCompPyramid*/ compValsSameLevNextInHierarchPyr(prevMessValsU, prevMessValsD, prevMessValsL, prevMessValsR, paramsPrevLevel, paramsCurrentLevel, currMessValsU, currMessValsD, currMessValsL, currMessValsR);

		for (int currMoveNum = 0; currMoveNum < TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y; currMoveNum++)
		{
			int indexCopyTo = retrieveIndexInDataAndMessage((xVal*2 + checkerboardPartAdjustment), (yVal*2 + 1), widthNextLevelCheckerboard, heightNextLevelCheckerboard, currMoveNum, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));

			messagesDeviceCurrentCheckerboard1.messageUDevice[indexCopyTo] = currMessValsU[currMoveNum];
			messagesDeviceCurrentCheckerboard1.messageDDevice[indexCopyTo] = currMessValsD[currMoveNum];
			messagesDeviceCurrentCheckerboard1.messageLDevice[indexCopyTo] = currMessValsL[currMoveNum];
			messagesDeviceCurrentCheckerboard1.messageRDevice[indexCopyTo] = currMessValsR[currMoveNum];

		}

		//retrieve the parameters for the `current level'
		paramsCurrentLevel = paramsDeviceCheckerboard2[paramsOffsetCurrLevel + (yVal*2 + 1)*widthNextLevelCheckerboard + (xVal*2 + checkerboardPartAdjustment)];

		//retrieve the message values for the `next level' of the computation hierarchy for four sets of message value (to copy `to')...
		/*compValsNextLevelCompPyramid*/ compValsSameLevNextInHierarchPyr(prevMessValsU, prevMessValsD, prevMessValsL, prevMessValsR, paramsPrevLevel, paramsCurrentLevel, currMessValsU, currMessValsD, currMessValsL, currMessValsR);

		for (int currMoveNum = 0; currMoveNum < TOTAL_POSS_MOVE_RANGE_X*TOTAL_POSS_MOVE_RANGE_Y; currMoveNum++)
		{
			int indexCopyTo = retrieveIndexInDataAndMessage((xVal*2 + checkerboardPartAdjustment), (yVal*2 + 1), widthNextLevelCheckerboard, heightNextLevelCheckerboard, currMoveNum, (currentBeliefPropParamsConstMem.totalNumMovesXDir * currentBeliefPropParamsConstMem.totalNumMovesYDir));

			messagesDeviceCurrentCheckerboard2.messageUDevice[indexCopyTo] = currMessValsU[currMoveNum];
			messagesDeviceCurrentCheckerboard2.messageDDevice[indexCopyTo] = currMessValsD[currMoveNum];
			messagesDeviceCurrentCheckerboard2.messageLDevice[indexCopyTo] = currMessValsL[currMoveNum];
			messagesDeviceCurrentCheckerboard2.messageRDevice[indexCopyTo] = currMessValsR[currMoveNum];
		}
	}
}



