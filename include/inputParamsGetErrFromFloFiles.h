//inputParamsGetErrFromFloFiles.h
//Scott Grauer-Gray
//December 1, 2010
//Header file for the input parameters for retrieving the error from flo files...

//needed for 'cout'
#include <iostream>

//using 'cout' and 'endl' for output printing...
using std::cout;
using std::endl;

//class representing the input parameters for cuda belief propagation
class inputParamsGetErrFromFloFiles
{
    public:

        inputParamsGetErrFromFloFiles();

	//constructor that takes in a string of strings with each parameter
	inputParamsGetErrFromFloFiles(char* inParams[]);

	//"getter" function for each input parameter
	char* getGroundTruthMovementFilePath() { return groundTruthMovementFilePath; };
	char* getOriginalMovementFilePath() { return origMovementFilePath; };
	char* getRoughMultiImagesFilePath() { return roughMultImagesMovementFilePath; };
	char* getAdjustedOriginalMovementFilePath() { return adjustedOriginalMovementFilePath; };
	char* getRefinedMovementFilePath() { return refinedMovementFilePath; };

	//function to print out each of the input parameters
	void printInputParams();

private:

	char* groundTruthMovementFilePath;
	char* origMovementFilePath;
	char* roughMultImagesMovementFilePath;
	char* adjustedOriginalMovementFilePath;
	char* refinedMovementFilePath;
};
