# runMotionBeliefPropRefineFromFloResultsAndUseNoExpMoveFloResult.py
# Scott Grauer-Gray
# November 30, 2010
# Python script for running motion belief propagation taking into account the previously generated motion but also taking a result without using the expected motion

# needed for the constants related to running the implementation and taking the results into account...
import runMotionBeliefPropRefineFromFloResultsAndUseNoExpMoveFloResultConsts

# needed for the motion data
import motionData

# needed to clean/make cuda belief propagation
import cleanAndMakeRunBeliefPropMotion

# needed to execute cuda belief propagation
import exeBeliefPropMotion

# needed for the constants corresponding to running motion belief propagation
import runMotionBeliefPropConsts

# needed for the input parameters for multiple images...
import inputParamsMultImages

# needed for setting up CUDA belief propagation
import setupCudaBeliefProp

# needed for generating the header file...
import genHeaderFile

# needed for refining from the flo file...
import retrieveParamsRefineResults

# needed for constants related to refining from the flo file...
import retrieveParamsRefineResultsConsts

# needed for input parameters...
import runMotionBeliefPropGivenExpMotionInputs
import inputParamsGivenExpMotion

def runMotionBeliefPropTakePrevMotionAndNoExpMotionResultIntoAccount():

	# set up the environment variables in order to run CUDA belief propagation
	setupCudaBeliefProp.initToRunCudaBeliefProp()

	# retrieve the 'refining parameters' from the input flo file...
	refiningParams = retrieveParamsRefineResults.retParamsToRefineResults(runMotionBeliefPropGivenExpMotionInputs.FILE_PATH_EXPECTED_MOTION)

	# generate the input parameters to run CUDA belief propagation
	inParamCudaBeliefProp = inputParamsGivenExpMotion.InputParamsGivenExpMotion()

	# adjust the input parameters based on the 'flo' input...
	inParamCudaBeliefProp.adjustParticParams(refiningParams[retrieveParamsRefineResultsConsts.START_MOVE_INC_X_PARAM_KEY], refiningParams[retrieveParamsRefineResultsConsts.START_MOVE_INC_Y_PARAM_KEY], refiningParams[retrieveParamsRefineResultsConsts.MIN_X_MOVE_PARAM_KEY], refiningParams[retrieveParamsRefineResultsConsts.MAX_X_MOVE_PARAM_KEY], refiningParams[retrieveParamsRefineResultsConsts.MIN_Y_MOVE_PARAM_KEY], refiningParams[retrieveParamsRefineResultsConsts.MAX_Y_MOVE_PARAM_KEY])

	# adjust the output 'movement' image
	inParamCudaBeliefProp.adjustOutputMovementImage(runMotionBeliefPropRefineFromFloResultsAndUseNoExpMoveFloResultConsts.FILE_PATH_OUTPUT_IMAGE_NO_ACCOUNT_EXP_MOTION)		

	# adjust the input parameter to save the flo motion...
	inParamCudaBeliefProp.adjustOutputMotionFloFile(runMotionBeliefPropRefineFromFloResultsAndUseNoExpMoveFloResultConsts.PATH_FLO_FILE_EXP_MOTION_NOT_USED)	

	# adjust the parameter to not take expected motion into account...
	inParamCudaBeliefProp.adjustTakeExpMotionWeight(runMotionBeliefPropRefineFromFloResultsAndUseNoExpMoveFloResultConsts.WEIGHT_EXP_MOTION_WHEN_NOT_USED)	

	# adjust the smoothing sigma
	inParamCudaBeliefProp.adjustSmoothingSigmaInputImages(runMotionBeliefPropRefineFromFloResultsAndUseNoExpMoveFloResultConsts.SMOOTHING_SIGMA_WHEN_EXP_MOTION_NOT_USED)	

	# generate the header file for the current set of parameters defined in runMotionBeliefPropConsts.py
	genHeaderFile.genCurrHeaderFile()

	# clean and make cuda belief propagation
	cleanAndMakeRunBeliefPropMotion.cleanMotionBeliefProp(runMotionBeliefPropConsts.DIR_MAKEFILE)
	cleanAndMakeRunBeliefPropMotion.makeMotionBeliefProp(runMotionBeliefPropConsts.DIR_MAKEFILE)

	# run cuda belief propagation
	exeBeliefPropMotion.runBeliefPropMotion(inParamCudaBeliefProp)

	# retrieve the motion data of the 'rough results'
	motionDataRoughResults = motionData.MotionData()
	motionDataRoughResults.readMotionFromFloFile(runMotionBeliefPropGivenExpMotionInputs.FILE_PATH_EXPECTED_MOTION)

	# retrieve the motion data of the results that don't use the expected motion
	motionDontUseExpMotion = motionData.MotionData()
	motionDontUseExpMotion.readMotionFromFloFile(runMotionBeliefPropRefineFromFloResultsAndUseNoExpMoveFloResultConsts.PATH_FLO_FILE_EXP_MOTION_NOT_USED)

	# merge the two results
	motionDataRoughResults.mergeMotionData(motionDontUseExpMotion)
	
	# save the merged results
	motionDataRoughResults.saveMotionToFloFile(runMotionBeliefPropRefineFromFloResultsAndUseNoExpMoveFloResultConsts.PATH_FLO_FILE_MERGED_RESULT_ACCOUNT_BOTH_MOTIONS)

	# adjust the parameter to use the merged results 
	inParamCudaBeliefProp.adjustInputFloFile(runMotionBeliefPropRefineFromFloResultsAndUseNoExpMoveFloResultConsts.PATH_FLO_FILE_MERGED_RESULT_ACCOUNT_BOTH_MOTIONS)

	# adjust the output 'movement' image
	inParamCudaBeliefProp.adjustOutputMovementImage(runMotionBeliefPropGivenExpMotionInputs.FILE_PATH_SAVE_RESULTING_MOTION_IMAGE)	

	# adjust the parameter to take expected motion into account...
	inParamCudaBeliefProp.adjustTakeExpMotionWeight(runMotionBeliefPropGivenExpMotionInputs.EST_MOVEMENT_WEIGHT_BP_MOTION_TEST)	

	# adjust the parameter for the flo file of the output motion
	inParamCudaBeliefProp.adjustOutputMotionFloFile(runMotionBeliefPropGivenExpMotionInputs.FILE_PATH_SAVED_MOVEMENT_DATA_FLO_FORMAT)	

	# adjust the smoothing sigma
	inParamCudaBeliefProp.adjustSmoothingSigmaInputImages(runMotionBeliefPropGivenExpMotionInputs.SMOOTHING_SIGMA_BP_MOTION_TEST)	

	# now run the data using the merged results as the input 'flo' file...
	cleanAndMakeRunBeliefPropMotion.cleanMotionBeliefProp(runMotionBeliefPropConsts.DIR_MAKEFILE)
	cleanAndMakeRunBeliefPropMotion.makeMotionBeliefProp(runMotionBeliefPropConsts.DIR_MAKEFILE)

	# run cuda belief propagation
	exeBeliefPropMotion.runBeliefPropMotion(inParamCudaBeliefProp)
	

	

	
	
