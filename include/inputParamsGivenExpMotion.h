//inputParamsGivenExpMotion.h
//Scott Grauer-Gray
//November 29, 2010
//Header file for the input parameters

#ifndef INPUT_PARAMS_GIVEN_EXP_MOTION_H
#define INPUT_PARAMS_GIVEN_EXP_MOTION_H

//needed for 'atoi/atof'...
#include <cstdlib>

//needed for 'cout'
#include <iostream>

//using 'cout' and 'endl' for output printing...
using std::cout;
using std::endl;

//class representing the input parameters for cuda belief propagation
class inputParamsGivenExpMotion
{
    public:

        inputParamsGivenExpMotion();

	//constructor that takes in a string of strings with each parameter
	inputParamsGivenExpMotion(char* inParams[]);

	//"getter" function for each input parameter
	char* getImage1FilePath() { return image1FilePath; };
	char* getImage2FilePath() { return image2FilePath; };

	char* getMotionImageFileSave() { return motionImageFileSave; };
	char* getExpectedMotionFloFilePath() { return expectedMotionFloFilePath; };
	
	int getWidthInputImages() { return widthInputImages; };
	int getHeightInputImages() { return heightInputImages; };

	int getNumBeliefPropLevels() { return numBeliefPropLevels; };
	int getNumItersInBeliefPropLevel() { return numItersInBeliefPropLevel; };

	float getDataCostCap() { return dataCostCap; };
	float getSmoothnessCostCap() { return smoothnessCostCap; };
	float getDataCostWeight() { return dataCostWeight; };

	float getCurrMoveIncrX() { return currMoveIncrX; };
	float getCurrMoveIncrY() { return currMoveIncrY; };
	float getSamplingLevel() { return samplingLevel; };

	float getXMoveMin() { return xMoveMin; };
	float getXMoveMax() { return xMoveMax; };

	float getYMoveMin() { return yMoveMin; };
	float getYMoveMax() { return yMoveMax; };

	float getMotionDisplayRange() { return motionDisplayRange; };

	char* getGroundTruthMoveFlo() { return groundTruthMoveFlo; };
	char* getMotionFloFileSave() { return motionFloFileSave; };
	float getPropChangeMove() { return propChangeMove; };

	float getSmoothingSigma() { return smoothingSigma; };

	int getNumLevelsWithinLevPyrHierarch() { return numLevelsWithinLevPyrHierarch; };

	//functions to retrieve motion estimation stuff
	float getEstMovementCap() { return estMoveCap; } ;
	float getEstMovementWeight() { return estMoveWeight; } ; 

	//function to print out each of the input parameters
	void printInputParams();

private:
	char* image1FilePath;
	char* image2FilePath;

	char* motionImageFileSave;

	char* expectedMotionFloFilePath;
	
	int widthInputImages;
	int heightInputImages;

	int numBeliefPropLevels;
	int numItersInBeliefPropLevel;

	float dataCostCap;
	float smoothnessCostCap;
	float dataCostWeight;

	float currMoveIncrX;
	float currMoveIncrY;
	float samplingLevel;

	float xMoveMin;
	float xMoveMax;

	float yMoveMin;
	float yMoveMax;

	float motionDisplayRange;

	char* groundTruthMoveFlo;
	char* motionFloFileSave;
	float propChangeMove;

	float smoothingSigma;

	int numLevelsWithinLevPyrHierarch;

	//variables related to estimated movement...
	float estMoveCap;
	float estMoveWeight;
};

#endif //INPUT_PARAMS_GIVEN_EXP_MOTION_H
