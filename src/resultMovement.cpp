//resultMovement.cpp
//Scott Grauer-Gray
//June 29, 2009
//Defines the functions used in the resultMovement object

#include "resultMovement.h"

resultMovement::resultMovement(void)
{
}

//constructor to retrieve the movement given a disparity image
resultMovement::resultMovement(const char* filePathDispImage, float startMoveDispImage, int widthData, int heightData, float multiplierDispImage):
startMoveXDir(startMoveDispImage), startMoveYDir(DEFAULT_RESULT_MOVEMENT_VAL),
widthMovements(widthData), heightMovements(heightData),
locationMovementData(LOCATION_MOVEMENT_LOAD_FROM_DISP_IMAGE), dirPosMovement(MOVEMENT_DIR_GIVEN_DISP_IMAGE)
{
	//load the x movement from the file

	//first load the disparity image as unsigned int representing the pixel intensity values
	unsigned int* dispImg = new unsigned int[widthMovements * heightMovements];

	//declare and assign pointers to unsigned ints representing the width and height for loading the disparity image
	unsigned int* widthDispImage = new unsigned int;
	unsigned int* heightDispImage = new unsigned int;

	//set the width and height of the disparity imagewidthDispImage
	*widthDispImage = static_cast<unsigned int>(widthData);
	*heightDispImage = static_cast<unsigned int>(heightData);

	cutLoadPGMi( filePathDispImage, &dispImg, widthDispImage, heightDispImage);

	//delete the pointers to the width and height of the disparity image
	delete widthDispImage;
	delete heightDispImage;

	//declare and allocate the space for the x movement
	movementXDir = new float[widthMovements * heightMovements];

	//retrieve the x movement at each pixel by dividing the intensity value by the multiplier and subtracting the starting movement from it
	for (int pixVal = 0; pixVal < (widthMovements * heightMovements); pixVal++)
	{
		if (dispImg[pixVal] == 0)
		{
			movementXDir[pixVal] = THRESHOLD_MOVEMENT_UNKNOWN + 1;
		}
		else
		{
			movementXDir[pixVal] = -1.0 * ((static_cast<float>(dispImg[pixVal])) / multiplierDispImage - startMoveXDir);
		}
	}


	//delete the space allocated to the disparity image
	delete [] dispImg;

	//set the y-movement at each pixel to a default value
	movementYDir = new float[widthMovements * heightMovements];

	for (int pixVal = 0; pixVal < (widthMovements * heightMovements); pixVal++)
	{
		movementYDir[pixVal] = DEFAULT_RESULT_MOVEMENT_VAL;
	}
}

//constructor to retrieve the movement given the flo data
resultMovement::resultMovement(const char* floMovementData) : 
locationMovementData(LOCATION_MOVEMENT_LOAD_FROM_FLO_IMAGE), dirPosMovement(MOVEMENT_DIR_GIVEN_FLO_IMAGE)
{
	//go through the retrieved movement and retrieve the smallest movement in the x and y directions
	readMovementDataInFloFormat(floMovementData);

	//declare the temporary variables to store the minimum movement in the x and y directions
	float minMoveXDir = INF_VALUE_FLOAT;
	float minMoveYDir = INF_VALUE_FLOAT;

	for (int pixVal = 0; pixVal < (resultMovement::widthMovements * resultMovement::heightMovements); pixVal++)
	{
		if (movementXDir[pixVal] < minMoveXDir)
		{
			minMoveXDir = movementXDir[pixVal];
		}
		if (movementYDir[pixVal] < minMoveYDir)
		{
			minMoveYDir = movementYDir[pixVal];
		}
	}

	//set the desired starting movement in the x and y directions
	startMoveXDir = minMoveXDir;
	startMoveYDir = minMoveYDir;
}

resultMovement::~resultMovement(void)
{
	//delete the current movement data either on the host or device
	switch (resultMovement::locationMovementData)
	{
		case DATA_ON_HOST:
			delete [] movementXDir;
			delete [] movementYDir;
			break;
		case DATA_ON_DEVICE:
			freeArray(movementXDir);
			freeArray(movementYDir);
			break;
	}
}

//function to transfer movement data from host to device and vise versa
void resultMovement::transferMovementDataHostToDevice()
{
	//check if data already on device
	if (resultMovement::locationMovementData == DATA_ON_DEVICE)
	{
		printf("Data already on device; no need to transfer movement data\n");
	}
	else if (resultMovement::locationMovementData == DATA_ON_HOST)
	{
		//set pointers for the data on the host (which will eventually be deleted)
		float* xMoveHost = movementXDir;
		float* yMoveHost = movementYDir;

		//declare the pointers and allocate the space on the device for the movement data
		float* xMoveDevice;
		float* yMoveDevice;

		allocateArray(reinterpret_cast<void**>(&xMoveDevice), resultMovement::widthMovements * resultMovement::heightMovements * sizeof(float));
		allocateArray(reinterpret_cast<void**>(&yMoveDevice), resultMovement::widthMovements * resultMovement::heightMovements * sizeof(float));

		//transfer the data from the host to the device
		copyArrayToDevice(xMoveDevice, xMoveHost, resultMovement::widthMovements * resultMovement::heightMovements * sizeof(float));
		copyArrayToDevice(yMoveDevice, yMoveHost, resultMovement::widthMovements * resultMovement::heightMovements * sizeof(float));

		//free the memory allocated to the movement on the host
		delete [] xMoveHost;
		delete [] yMoveHost;

		//set the pointer to the data to the data on the device
		movementXDir = xMoveDevice;
		movementYDir = yMoveDevice;

		//set the location of the data to being on the device
		locationMovementData = DATA_ON_DEVICE;
	}
}

void resultMovement::transferMovementDataDeviceToHost()
{
	//check if data already on host
	if (resultMovement::locationMovementData == DATA_ON_HOST)
	{
		printf("Data already on host; no need to transfer movement data\n");
	}
	else if (resultMovement::locationMovementData == DATA_ON_DEVICE)
	{
		//set pointers for the data on the device (which will eventually be deleted)
		float* xMoveDevice = movementXDir;
		float* yMoveDevice = movementYDir;

		//declare the pointers and allocate the space on the host for the movement data
		float* xMoveHost = new float[resultMovement::widthMovements * resultMovement::heightMovements];
		float* yMoveHost = new float[resultMovement::widthMovements * resultMovement::heightMovements];

		//transfer the data from the host to the device
		copyArrayFromDevice(xMoveHost, xMoveDevice, resultMovement::widthMovements * resultMovement::heightMovements * sizeof(float));
		copyArrayFromDevice(yMoveHost, yMoveDevice, resultMovement::widthMovements * resultMovement::heightMovements * sizeof(float));

		//free the memory allocated to the movement on the device
		freeArray(xMoveDevice);
		freeArray(yMoveDevice);

		//set the pointer to the data to the data on the host
		movementXDir = xMoveHost;
		movementYDir = yMoveHost;

		//set the location of the data to being on the host
		locationMovementData = DATA_ON_HOST;
	}
}

//function to save the current movement data
void resultMovement::saveMovementData(const char *filePathSaveMovement, desiredMovement movementToSave, float movementScaleMultiplier)
{
	//declare and allocate the space for the movement data
	unsigned int* savedMovementData = new unsigned int[resultMovement::widthMovements * resultMovement::heightMovements];

	//declare the float pointer to the desired movement to save on the host
	float* desiredMovementToSaveHost;

	//declare the variable which stores the minimum movement in the desired direction
	float minMoveDesMovement;

	//check if current movement is on host or device; if on device, need to transfer to host
	if (resultMovement::locationMovementData == DATA_ON_DEVICE)
	{
		desiredMovementToSaveHost = new float[resultMovement::widthMovements * resultMovement::heightMovements];
	}

	switch (movementToSave)
	{
		case X_DIR_MOVEMENT:
			//depending on location of resulting data, either need to set pointer to movement in x direction or
			//transfer data from device to host
			switch (resultMovement::locationMovementData)
			{
				case DATA_ON_HOST:
					desiredMovementToSaveHost = resultMovement::movementXDir;
					break;
				case DATA_ON_DEVICE:
					copyArrayFromDevice(desiredMovementToSaveHost, (void*)resultMovement::movementXDir,
						resultMovement::widthMovements * resultMovement::heightMovements * sizeof(float));
					break;
			}
			minMoveDesMovement = resultMovement::startMoveXDir;
			break;
		case Y_DIR_MOVEMENT:
			//depending on location of resulting data, either need to set pointer to movement in x direction or
			//transfer data from device to host
			switch (resultMovement::locationMovementData)
			{
				case DATA_ON_HOST:
					desiredMovementToSaveHost = resultMovement::movementYDir;
					break;
				case DATA_ON_DEVICE:
					copyArrayFromDevice(desiredMovementToSaveHost, (void*)resultMovement::movementYDir,
						resultMovement::widthMovements * resultMovement::heightMovements * sizeof(float));
					break;
			}
			desiredMovementToSaveHost = resultMovement::movementYDir;
			minMoveDesMovement = resultMovement::startMoveYDir;
			break;
	}

	//go through each pixel and multiply it by the scale
	for (int pixelNum = 0; pixelNum < (resultMovement::widthMovements * resultMovement::heightMovements); pixelNum++)
	{
		savedMovementData[pixelNum] = (unsigned int)((((desiredMovementToSaveHost[pixelNum])) * ((float)movementScaleMultiplier * -1.0f)) + 0.5f);//(unsigned int)(((desiredMovementToSaveHost[pixelNum] - minMoveDesMovement) * movementScaleMultiplier) + 0.5f);
	}

	//save the computed desiredMovementToSave
	cutSavePGMi(filePathSaveMovement, savedMovementData, resultMovement::widthMovements, resultMovement::heightMovements);

	printf("OUT_IMAGE_SAVED\n");
	printf("FILE_SAVE_PATH: %s\n", filePathSaveMovement);

	//free the memory allocated to the savedMovementData
	delete [] savedMovementData;

	//if data transferred from device to host, then free the memory allocated to the desired movement on the host
	if (resultMovement::locationMovementData == DATA_ON_DEVICE)
	{
		delete [] desiredMovementToSaveHost;
	}
}

//function for retrieving the difference between two movements with a border which is ignored
//for now, data must be on host in order to perform this operation
float resultMovement::retrieveMovementDiff(resultMovement* movementToCompare, int borderWidth, moveDiffCalc moveDiff)
{
	//check if movement data on host; return error if not
	if (resultMovement::locationMovementData != DATA_ON_HOST)
	{
		printf("Error in retrieving movement difference, movement data must be on host\n");
		return ERROR_FLOAT_VAL;
	}

	//initialize the total difference between movements to 0
	float totalMoveDiff = 0.0f;

	//go through each pixel in the non-border
	for (int yCoord = borderWidth; yCoord < (heightMovements - borderWidth); yCoord++)
	{
		for (int xCoord = borderWidth; xCoord < (widthMovements - borderWidth); xCoord++)
		{
			//check if the movement is "unknown" at pixel
			if ((resultMovement::retrieveXMoveAtPix(xCoord, yCoord) < THRESHOLD_MOVEMENT_UNKNOWN) &&
				(resultMovement::retrieveYMoveAtPix(xCoord, yCoord) < THRESHOLD_MOVEMENT_UNKNOWN) &&
				(movementToCompare->retrieveXMoveAtPix(xCoord, yCoord) < THRESHOLD_MOVEMENT_UNKNOWN) && 
				(movementToCompare->retrieveYMoveAtPix(xCoord, yCoord) < THRESHOLD_MOVEMENT_UNKNOWN))
			{

				if (moveDiff == MANHATTAN_DIST_FOR_MOVE_DIFF)
				{
					totalMoveDiff += resultMovement::retrieveManhatDist(
						resultMovement::retrieveXMoveAtPix(xCoord, yCoord), 
						resultMovement::retrieveYMoveAtPix(xCoord, yCoord), 
						movementToCompare->retrieveXMoveAtPix(xCoord, yCoord), 
						movementToCompare->retrieveYMoveAtPix(xCoord, yCoord));
				}
				if (moveDiff == EUCLIDEAN_DIST_FOR_MOVE_DIFF)
				{
					totalMoveDiff += resultMovement::retrieveEuclidDist(
						resultMovement::retrieveXMoveAtPix(xCoord, yCoord), 
						resultMovement::retrieveYMoveAtPix(xCoord, yCoord), 
						movementToCompare->retrieveXMoveAtPix(xCoord, yCoord), 
						movementToCompare->retrieveYMoveAtPix(xCoord, yCoord));
				}
			}
		}
	}

	//return the computed movement difference
	return totalMoveDiff;
}

//function for retrieve the number of pixels which are "different" between two movements
//for now, data must be on host in order to perform this operation
int resultMovement::retrieveNumPixMoveDiff(resultMovement* movementToCompare, int borderWidth, moveDiffCalc moveDiff, float threshMoveDiff)
{
	//check if movement data on host; return error if not
	if (resultMovement::locationMovementData != DATA_ON_HOST)
	{
		printf("Error in retrieving number of pixel with movement difference, movement data must be on host\n");
		return ERROR_INT_VAL;
	}

	//initialize the total number of pixels different in movements to 0
	int numPixMoveDiff = 0;

	//go through each pixel in the non-border
	for (int yCoord = borderWidth; yCoord < (heightMovements - borderWidth); yCoord++)
	{
		for (int xCoord = borderWidth; xCoord < (widthMovements - borderWidth); xCoord++)
		{
			//check if the movement is "unknown" at pixel
			if ((resultMovement::retrieveXMoveAtPix(xCoord, yCoord) < THRESHOLD_MOVEMENT_UNKNOWN) &&
				(resultMovement::retrieveYMoveAtPix(xCoord, yCoord) < THRESHOLD_MOVEMENT_UNKNOWN) &&
				(movementToCompare->retrieveXMoveAtPix(xCoord, yCoord) < THRESHOLD_MOVEMENT_UNKNOWN) && 
				(movementToCompare->retrieveYMoveAtPix(xCoord, yCoord) < THRESHOLD_MOVEMENT_UNKNOWN))
			{

				//declare the float variable to store the floating-point difference in movement
				float movementDiff;

				//retrieve the difference in movement between the two movements
				if (moveDiff == MANHATTAN_DIST_FOR_MOVE_DIFF)
				{
					movementDiff = resultMovement::retrieveManhatDist(
						resultMovement::retrieveXMoveAtPix(xCoord, yCoord), 
						resultMovement::retrieveYMoveAtPix(xCoord, yCoord), 
						movementToCompare->retrieveXMoveAtPix(xCoord, yCoord), 
						movementToCompare->retrieveYMoveAtPix(xCoord, yCoord));
				}
				if (moveDiff == EUCLIDEAN_DIST_FOR_MOVE_DIFF)
				{
					movementDiff = resultMovement::retrieveEuclidDist(
						resultMovement::retrieveXMoveAtPix(xCoord, yCoord), 
						resultMovement::retrieveYMoveAtPix(xCoord, yCoord), 
						movementToCompare->retrieveXMoveAtPix(xCoord, yCoord), 
						movementToCompare->retrieveYMoveAtPix(xCoord, yCoord));
				}

				//check if the retrieved movement difference is beyond the given threshold
				//and increment the number of pixels with a difference if it is
				if (movementDiff > threshMoveDiff)
				{
					numPixMoveDiff++;
				}
			}
		}
	}

	//return the retrieve number of pixels with a difference in the movement beyond or equal to the given threshold
	return numPixMoveDiff;
}

//save the movement image 
void resultMovement::saveMovementImage(float rangeMovementPosAndNeg, const char* fileNameSaveVisualization)
{
	unsigned char* visualizationImageData = new unsigned char[NUM_CHANNELS_SAVE_IMAGE* resultMovement::widthMovements * resultMovement::heightMovements];

	//declare the pointers to the x and y movements on the host
	float* xMovementHost;
	float* yMovementHost;

	//depending on location of resulting data, either need to set pointer to movement in x direction or
	//transfer data from device to host
	switch (resultMovement::locationMovementData)
	{
		case DATA_ON_HOST:
			xMovementHost = resultMovement::movementXDir;
			yMovementHost = resultMovement::movementYDir;
			break;
		case DATA_ON_DEVICE:
			//set the space for the x and y movement on the host
			xMovementHost = new float[resultMovement::widthMovements * resultMovement::heightMovements];
			yMovementHost = new float[resultMovement::widthMovements * resultMovement::heightMovements];

			copyArrayFromDevice(xMovementHost, (void*)resultMovement::movementXDir,
				resultMovement::widthMovements * resultMovement::heightMovements * sizeof(float));
			copyArrayFromDevice(yMovementHost, (void*)resultMovement::movementYDir,
				resultMovement::widthMovements * resultMovement::heightMovements * sizeof(float));
			break;
	}

	int colorwheel[MAXCOLS][3];

	int ncolsUsedToColorCalc = 0;

	//generate the color wheel used 
	makecolorwheel(colorwheel, ncolsUsedToColorCalc);

	//go through every motion and determine the correct color (rgb channels) for visualization and the write it to the approprial place in visualizationImageData
	for (unsigned int numPixel = 0; numPixel < (resultMovement::widthMovements * resultMovement::heightMovements); numPixel++)
	{
		//if movement is beyond a certain threshold representing 'unknown' movement, then set color to default
		if ((movementXDir[numPixel] >= THRESHOLD_MOVEMENT_UNKNOWN) || (movementYDir[numPixel] >= THRESHOLD_MOVEMENT_UNKNOWN))
		{
			for (int pixChan = 0; pixChan < NUM_CHANNELS_SAVE_IMAGE; pixChan++)
			{
				visualizationImageData[NUM_CHANNELS_SAVE_IMAGE*numPixel + pixChan] = COLOR_UNKNOWN_MOTION;
			}
		}
		else
		{
			float magXDirMotion = xMovementHost[numPixel] / rangeMovementPosAndNeg;
			float magYDirMotion = yMovementHost[numPixel] / rangeMovementPosAndNeg;

			//compute the color 
			computeColor(magXDirMotion, magYDirMotion, &visualizationImageData[NUM_CHANNELS_SAVE_IMAGE*numPixel], colorwheel, ncolsUsedToColorCalc);
		}
	}

	//now save the resulting color PPM image with the visualization image data
	cutSavePPMub(fileNameSaveVisualization, visualizationImageData, resultMovement::widthMovements, resultMovement::heightMovements);

	//if data on device, then free memory allocated to data on host
	if (resultMovement::locationMovementData == DATA_ON_DEVICE)
	{
		delete [] xMovementHost;
		delete [] yMovementHost;
	}
}

//helper function used for saving the movement
void resultMovement::setcols(int colorwheel[MAXCOLS][3], int r, int g, int b, int k)
{
	colorwheel[k][0] = r;
	colorwheel[k][1] = g;
	colorwheel[k][2] = b;
}

//helper function used for saving the movement
void resultMovement::computeColor(float fx, float fy, unsigned char* pix, int colorwheel[MAXCOLS][3], int& ncolsUsedToColorCalc)
{
	float tempVal = fx;
	fx = fy;
	fy = tempVal;

	fx = fx * -1.0f;
	fy = fy * -1.0f;

	//reverse fx and fy...
	//fy = -1.0f*fy;

	//fy = fx;
	//fx = tempVal;
	
    if (ncolsUsedToColorCalc == 0)
	makecolorwheel(colorwheel, ncolsUsedToColorCalc);

    float rad = sqrt(fx * fx + fy * fy);
    float a = atan2(-fy, -fx) / M_PI;
    float fk = (a + 1.0) / 2.0 * (ncolsUsedToColorCalc-1);
    int k0 = (int)fk;
    int k1 = (k0 + 1) % ncolsUsedToColorCalc;
    float f = fk - k0;
    //f = 0; // uncomment to see original color wheel
    for (int b = 0; b < 3; b++) {
		float col0 = colorwheel[k0][b] / 255.0f;
		float col1 = colorwheel[k1][b] / 255.0f;
		float col = (1 - f) * col0 + f * col1;
		if (rad <= 1)
			col = 1 - rad * (1 - col); // increase saturation with radius
		else
			col *= .75f; // out of range
		pix[2 - b] = (int)(255.0 * col);
	}
}

//helper function used for saving the movement
void resultMovement::makecolorwheel(int colorwheel[MAXCOLS][3], int& ncolsUsedToColorCalc)
{
    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow 
    //  than between yellow and green)
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;
    ncolsUsedToColorCalc = RY + YG + GC + CB + BM + MR;

    int i;
    int k = 0;
    for (i = 0; i < RY; i++) setcols(colorwheel, 255,	   255*i/RY,	 0,	       k++);
    for (i = 0; i < YG; i++) setcols(colorwheel, 255-255*i/YG, 255,		 0,	       k++);
    for (i = 0; i < GC; i++) setcols(colorwheel, 0,		   255,		 255*i/GC,     k++);
    for (i = 0; i < CB; i++) setcols(colorwheel, 0,		   255-255*i/CB, 255,	       k++);
    for (i = 0; i < BM; i++) setcols(colorwheel, 255*i/BM,	   0,		 255,	       k++);
    for (i = 0; i < MR; i++) setcols(colorwheel, 255,	   0,		 255-255*i/MR, k++);
}

//write the movement data in flo format for evaluation using the Middlebury benchmark
void resultMovement::writeMovementDataInFloFormat(const char* filePathSave, bool multMovementByNegOne)
{
	FILE* fpMovementInFlo = fopen(filePathSave, "wb");

	//first write "PIEH" in ASCII
	fwrite("PIEH", sizeof(char), 4, fpMovementInFlo);

	//now write the width and then height as an integer
	fwrite(&(resultMovement::widthMovements), sizeof(int), 1, fpMovementInFlo);
	fwrite(&(resultMovement::heightMovements), sizeof(int), 1, fpMovementInFlo);

	//now go through every pixel in row order and write the x and y movements
	for (unsigned int numPixel = 0; numPixel < (resultMovement::widthMovements*resultMovement::heightMovements); numPixel++)
	{
		if (multMovementByNegOne == false)
		{
			fwrite(&movementXDir[numPixel], sizeof(float), 1, fpMovementInFlo);
			fwrite(&movementYDir[numPixel], sizeof(float), 1, fpMovementInFlo);
		}
		else
		{
			float xMoveNeg = movementXDir[numPixel] * -1.0f;
			float yMoveNeg = movementYDir[numPixel] * -1.0f;

			fwrite(&xMoveNeg, sizeof(float), 1, fpMovementInFlo);
			fwrite(&yMoveNeg, sizeof(float), 1, fpMovementInFlo);

		}
	}

	fclose(fpMovementInFlo);
}

//helper function to
//read the flow movement data in the flo format as defined in http://vision.middlebury.edu/flow/code/flow-code/README.txt
void resultMovement::readMovementDataInFloFormat(const char* filePathFloData)
{
	FILE* fpMovementInFlo = fopen(filePathFloData, "rb");

	const float sanityCheckValFlowData = 202021.25f;

	//retrieve the first four bytes and check if they equal 202021.25
	float floatVal;
	fread(&floatVal, sizeof(float), 1, fpMovementInFlo);

	if (floatVal != sanityCheckValFlowData)
	{
		printf("ERROR IN FLO FILE\n");
		return;
	}

	//now read the width and height of the images
	fread(&(resultMovement::widthMovements), sizeof(int), 1, fpMovementInFlo);
	fread(&(resultMovement::heightMovements), sizeof(int), 1, fpMovementInFlo);


	//allocate the space for the movement in the x and y directions
	movementXDir = new float[resultMovement::widthMovements*resultMovement::heightMovements];
	movementYDir = new float[resultMovement::widthMovements*resultMovement::heightMovements];

	//now read the interleaved x and y data in each row and column
	for (int currentPixelNum = 0; currentPixelNum < (resultMovement::widthMovements*resultMovement::heightMovements); currentPixelNum++)
	{
		fread(&movementXDir[currentPixelNum], sizeof(float), 1, fpMovementInFlo);
		fread(&movementYDir[currentPixelNum], sizeof(float), 1, fpMovementInFlo);
	}

	fclose(fpMovementInFlo);
}

//functions to retrieve the max and min x and y movements
float resultMovement::retrieveXMoveMax()
{
	//initialize the maximum movement in the x direction to negative infinity
	float currentXMoveMax = -1.0f*INF_VALUE_FLOAT;

	for (int currentPixelNum = 0; currentPixelNum < (resultMovement::widthMovements*resultMovement::heightMovements); currentPixelNum++)
	{
		if ((movementXDir[currentPixelNum] > currentXMoveMax) && (movementXDir[currentPixelNum] < THRESHOLD_MOVEMENT_UNKNOWN))
		{
			currentXMoveMax = movementXDir[currentPixelNum];
		}
	}

	return currentXMoveMax;
}

float resultMovement::retrieveYMoveMax()
{
	//initialize the maximum movement in the y direction to negative infinity
	float currentYMoveMax = -1.0f*INF_VALUE_FLOAT;

	for (int currentPixelNum = 0; currentPixelNum < (resultMovement::widthMovements*resultMovement::heightMovements); currentPixelNum++)
	{
		if ((movementYDir[currentPixelNum] > currentYMoveMax) && (movementYDir[currentPixelNum] < THRESHOLD_MOVEMENT_UNKNOWN))
		{
			currentYMoveMax = movementYDir[currentPixelNum];
		}
	}

	return currentYMoveMax;
}

float resultMovement::retrieveXMoveMin()
{
	//initialize the minimum movement in the y direction to infinity
	float currentXMoveMin = INF_VALUE_FLOAT;

	for (int currentPixelNum = 0; currentPixelNum < (resultMovement::widthMovements*resultMovement::heightMovements); currentPixelNum++)
	{
		if ((movementXDir[currentPixelNum] < currentXMoveMin) && (movementXDir[currentPixelNum] < THRESHOLD_MOVEMENT_UNKNOWN))
		{
			currentXMoveMin = movementXDir[currentPixelNum];
		}
	}

	return currentXMoveMin;
}

float resultMovement::retrieveYMoveMin()
{
	//initialize the minimum movement in the y direction to infinity
	float currentYMoveMin = INF_VALUE_FLOAT;

	for (int currentPixelNum = 0; currentPixelNum < (resultMovement::widthMovements*resultMovement::heightMovements); currentPixelNum++)
	{
		if ((movementYDir[currentPixelNum] < currentYMoveMin) && (movementYDir[currentPixelNum] < THRESHOLD_MOVEMENT_UNKNOWN))
		{
			currentYMoveMin = movementYDir[currentPixelNum];
		}
	}

	return currentYMoveMin;
}

//function to retrieve the number of values where the movement is not unknown
int resultMovement::numValsKnownMovement()
{
	//initialize the number of values where the movement is not unknown to 0
	int numValsMoveKnown = 0;

	for (int currentPixelNum = 0; currentPixelNum < (resultMovement::widthMovements*resultMovement::heightMovements); currentPixelNum++)
	{
		//check to make sure the movement in the x and y directions is "known"
		if ((movementXDir[currentPixelNum] < THRESHOLD_MOVEMENT_UNKNOWN) && (movementYDir[currentPixelNum] < THRESHOLD_MOVEMENT_UNKNOWN))
		{
			numValsMoveKnown++;
		}
	}

	return numValsMoveKnown;
}

