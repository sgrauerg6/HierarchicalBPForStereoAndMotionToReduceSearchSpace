//runBeliefPropWithEstMotion.h
//Scott Grauer-Gray
//September 27, 2010
//Header function for running belief propagation on multiple images

#ifndef RUN_BELIEF_PROP_MULT_IMAGES
#define RUN_BELIEF_PROP_MULT_IMAGES

//needed for the default parameters and structs
#include "beliefPropParamsAndStructs.cuh"

//needed for the input images
#include "bpImage.h"

//needed for resulting movement output
#include "resultMovement.h"

//needed for the host and device function in order to run belief propagation on the host and device using CUDA
#include "runBeliefPropHostFuncts.cuh"

//needed for general CUDA functions for allocating the transferring data to/from the device
#include "genCudaFuncts.cuh"

//needed for general math functions
#include <math.h>

#include <cutil.h>

//declarations for the class to run multiple 'belief propagation' images...
class runBeliefPropMultImages
{

public:

	//initialize running belief propagation with the desired parameters
	runBeliefPropMultImages(int numBpLevels, int numBpItersPerLevel, float bpDataCostCap, float bpSmoothnessCostCap, float bpDataCostWeight,
			float bpCurrentMoveIncX, float bpCurrentMoveIncY, float bpSamplingLevel, float bpXMoveMin, float bpXMoveMax, float bpYMoveMin, float bpYMoveMax,
			float bpPropChangeMoveNextLevel, discontCostType bpCurrDiscCostType, posMoveDirection bpCurrPosMoveDir, float bpMovementCostCap,
			float bpMovementCostWeight);

	~runBeliefPropMultImages(void);

	//defines the operator to call to run belief propagation
	//run belief propagation on each set of images and return the movement on the desired set of images...
	resultMovement* operator()(bpImage** bpImageSet, int numImages, int startImageDesire, int endImageDesire);

private:

	//define the number of hierarchical levels and iterations per level used in running belief propagation
	int numBeliefPropLevels;
	int numBeliefPropItersPerLevel;

	//define the current belief propagation parameters
	float dataCostCap;
	float smoothnessCostCap;
	float dataCostWeight;

	//define the current increment of possible movement in the x and y directions
	float currentMoveIncrementX;
	float currentMoveIncrementY;

	//define the level of sampling for the output movement
	//1.0 means each input pixel sample; less than 1.0 gives more samples, greater than 1.0 gives less samples
	float samplingLevel;

	//define the starting and ending x and y movements
	float xMoveMin;
	float xMoveMax;

	float yMoveMin;
	float yMoveMax;

	//defines the change in the movement increment between levels
	float propChangeMoveNextLevel;

	//define the parameters in terms of incorpating `expected movement'
	float movementCostCap;
	float movementCostWeight;

	//defines the current discontinuity cost type
	discontCostType currDiscCostType;

	//define the current positive movement direction
	posMoveDirection currPosMoveDir;

	//private helper function to convert the current runBeliefProp into a currBeliefPropParams struct
	currBeliefPropParams genCurrBeliefPropParamsStruct();

	//private helper function to plug in the current parameters into the resultMovement
	resultMovement* genResultMovementObject();
};




#endif //RUN_BELIEF_PROP_MULT_IMAGES

