//runBeliefProp.h
//Scott Grauer-Gray
//June 24, 2009
//Define the class used to run belief propagation using the overloaded () operator


#ifndef RUN_BELIEF_PROP_H
#define RUN_BELIEF_PROP_H

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


class runBeliefProp
{
public:

		//initialize running belief propagation with the desired parameters
	runBeliefProp(int numBpLevels, int numBpItersPerLevel, float bpDataCostCap, float bpSmoothnessCostCap, float bpDataCostWeight,
		float bpCurrentMoveIncX, float bpCurrentMoveIncY, float bpSamplingLevel, float bpXMoveMin, float bpXMoveMax, float bpYMoveMin, float bpYMoveMax,
		float bpPropChangeMoveNextLevel, discontCostType bpCurrDiscCostType, posMoveDirection bpCurrPosMoveDir, 
		int inputNumLevelsWithinLevPyrHierarch,
		currMethodProcessingHierarch inputCurrProcessingMethSetting = DEFAULT_PROCESSING_METHOD, 
		usePrevMovement inputUsePrevMovementSetting = DEFAULT_USE_PREV_MOVEMENT_METHOD);

	~runBeliefProp(void);

	//defines the operator to call to run belief propagation
	resultMovement* operator()(bpImage* bpImage1, bpImage* bpImage2);

	//defines the operator to call to run belief propagation with estimated movement...
	resultMovement* runBeliefPropWithEstMovement(bpImage* bpImage1, bpImage* bpImage2, float* estMotionX, float* estMotionY);

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

	//define the number of levels to run `within' each level in a pyramid hierarchy...
	int numLevelsWithinLevPyrHierarch;

	//defines the current discontinuity cost type
	discontCostType currDiscCostType;

	//define the current positive movement direction
	posMoveDirection currPosMoveDir;

	//private helper function to convert the current runBeliefProp into a currBeliefPropParams struct
	currBeliefPropParams genCurrBeliefPropParamsStruct();

	//private helper function to plug in the current parameters into the resultMovement
	resultMovement* genResultMovementObject();

	//define the current method of processing in terms of what type of hierarchy to use...
	currMethodProcessingHierarch currProcessingMethSetting;

	//define whether or not to use the `previous movement'
	usePrevMovement usePrevMovementSetting;
};

#endif //RUN_BELIEF_PROP_H
