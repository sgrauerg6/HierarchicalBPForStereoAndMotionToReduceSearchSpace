//resultMovement.h
//Scott Grauer-Gray
//June 29, 2009
//Declares the class defining the resulting movement (likely computed via running BP)

#ifndef RESULT_MOVEMENT_H
#define RESULT_MOVEMENT_H

//needed for the general parameters and structs
#include "genParamsAndStructs.cuh"

//needed for general CUDA functions for allocating the transferring data to/from the device
#include "genCudaFuncts.cuh"

//needed for result movement parameters and constants
#include "resultMovementParamsAndStructs.cuh"

//needed for math functions
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159
#endif

const int NUM_CHANNELS_SAVE_IMAGE = 3;
const int MAXCOLS = 60;

class resultMovement
{
public:
	resultMovement(void);
	resultMovement(float* xMovePointer, float* yMovePointer, float startXMove, float startYMove, int widthData, int heightData,
		dataLocation movementDataLocation, posMoveDirection posMovementDir) :
		movementXDir(xMovePointer), movementYDir(yMovePointer), startMoveXDir(startXMove), startMoveYDir(startYMove), widthMovements(widthData),
		heightMovements(heightData), locationMovementData(movementDataLocation), dirPosMovement(posMovementDir) { };

	//constructor to retrieve the movement given a disparity image
	resultMovement(const char* filePathDispImage, float startMoveDispImage, int widthData, int heightData, float multiplierDispImage);

	//constructor to retrieve the movement given the flo data
	resultMovement(const char* floMovementData);

	~resultMovement(void);

	//save the x or y movement as a disparity map with a particular multiple
	void saveMovementData(const char* filePathSaveMovement, desiredMovement movementToSave, float movementScaleMultiplier);

	//setter function to set the values of the member variables
	void setMovementXDir(float* xDirMove) { movementXDir = xDirMove; };
	void setMovementYDir(float* yDirMove) { movementYDir = yDirMove; };
	
	void setStartMoveXDir(float xDirMoveStart) { startMoveXDir = xDirMoveStart; };
	void setStartMoveYDir(float yDirMoveStart) { startMoveYDir = yDirMoveStart; };

	void setWidthMovements(int widthMovementData) { widthMovements = widthMovementData; };
	void setHeightMovements(int heightMovementData) { heightMovements = heightMovementData; };

	void setLocationMovementData(dataLocation movementDataLocation) { locationMovementData = movementDataLocation; };
	void setPosMoveDirection(posMoveDirection currPosMoveDir) { dirPosMovement = currPosMoveDir; };

	//functions to retrieve the x and y movement for a given pixel
	float retrieveXMoveAtPix(int xCoordPix, int yCoordPix) { return movementXDir[retrievePixIndex(xCoordPix, yCoordPix, widthMovements, heightMovements)]; };
	float retrieveYMoveAtPix(int xCoordPix, int yCoordPix) { return movementYDir[retrievePixIndex(xCoordPix, yCoordPix, widthMovements, heightMovements)]; };

	//function to transfer movement data from host to device and vise versa
	void transferMovementDataHostToDevice();
	void transferMovementDataDeviceToHost();

	//function for retrieving the difference between two movements with a border which is ignored
	//for now, data must be on host in order to perform this operation
	float retrieveMovementDiff(resultMovement* movementToCompare, int borderWidth, moveDiffCalc moveDiff);

	//function for retrieve the number of pixels which are "different" between two movements
	//for now, data must be on host in order to perform this operation
	int retrieveNumPixMoveDiff(resultMovement* movementToCompare, int borderWidth, moveDiffCalc moveDiff, float threshMoveDiff = 0.001f);

	//save either the 'vertical' or 'horizontal' movement as a disparity map...
	//for now, simply saves the 'negative' horizontal movement...
	void saveMovementDispMap(float multiplierDispMap);

	//save the movement image 
	void saveMovementImage(float rangeMovementPosAndNeg, const char* fileNameSaveVisualization);

	//write the movement data in flo format for evaluation using the Middlebury benchmark
	void writeMovementDataInFloFormat(const char* filePathSave, bool multMovementByNegOne = false);

	//functions to retrieve the max and min x and y movements
	float retrieveXMoveMax();
	float retrieveYMoveMax();

	float retrieveXMoveMin();
	float retrieveYMoveMin();

	//function to retrieve the number of values where the movement is not unknown
	int numValsKnownMovement();

	//function to retrieve the movement in the x/y directions
	float* getXMovement() { return movementXDir; };
	float* getYMovement() { return movementYDir; };

private:
	//movement is stored as float values
	float* movementXDir;
	float* movementYDir;

	//define the starting movement in the x and y directions
	float startMoveXDir;
	float startMoveYDir;

	//define the width and height of the movement
	int widthMovements;
	int heightMovements;

	//define the location of the current movement data (device or host)
	dataLocation locationMovementData;

	//define the "direction" of the positive movement
	posMoveDirection dirPosMovement;

	//private helper functions to retrieve the pixel index in the movement 2-D array mapped to 1-D by column and then row
	int retrievePixIndex(int xCoord, int yCoord, int widthArray, int heightArray) { return (yCoord*widthArray + xCoord); };

	//private helper functions to retrieve the manhattan and euclidean distances
	float retrieveManhatDist(float xValPoint1, float yValPoint1, float xValPoint2, float yValPoint2) { return (abs(xValPoint1 - xValPoint2) + abs(yValPoint1 - yValPoint2)); };
	float retrieveEuclidDist(float xValPoint1, float yValPoint1, float xValPoint2, float yValPoint2) { return (sqrt((xValPoint1 - xValPoint2)*(xValPoint1 - xValPoint2) + (yValPoint1 - yValPoint2)*(yValPoint1 - yValPoint2))); };

	//helper function used for saving the movement
	void setcols(int colorwheel[MAXCOLS][3], int r, int g, int b, int k);

	//helper function used for saving the movement
	void computeColor(float fx, float fy, unsigned char* pix, int colorwheel[MAXCOLS][3], int& ncolsUsedToColorCalc);

	//helper function used for saving the movement
	void makecolorwheel(int colorwheel[MAXCOLS][3], int& ncolsUsedToColorCalc);

	//helper function to
	//read the flow movement data in the flo format as defined in http://vision.middlebury.edu/flow/code/flow-code/README.txt
	void readMovementDataInFloFormat(const char* filePathFloData);

};

#endif //RESULT_MOVEMENT_H
