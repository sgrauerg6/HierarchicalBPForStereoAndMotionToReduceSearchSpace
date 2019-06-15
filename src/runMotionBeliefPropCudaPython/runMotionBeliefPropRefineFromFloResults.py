# runMotionBeliefPropRefineFromFloResults.py
# Scott Grauer-Gray
# September 29, 2010
# Python script to run "motion" belief propagation using previous results from a 'flo' file...

# needed to clean/make cuda belief propagation
import cleanAndMakeRunBeliefPropMotion

# needed to execute cuda belief propagation
import exeBeliefPropMotion

# needed for the constants corresponding to running motion belief propagation
import runMotionBeliefPropConsts

# needed for the input parameters for motion retrieval given expected motion...
import inputParamsGivenExpMotion

# needed for setting up CUDA belief propagation
import setupCudaBeliefProp

# needed for generating the header file...
import genHeaderFile

# needed for refining the search from the expected motion...
import retrieveParamsRefineResults

# needed for constants related to refining the search from the expected motion...
import retrieveParamsRefineResultsConsts

# needed for the path of the 'flo' file...
import runMotionBeliefPropGivenExpMotionInputs

# needed for refining from the flo file...
import retrieveParamsRefineResults

# needed for constants related to refining from the flo file...
import retrieveParamsRefineResultsConsts

# function for running motion belief propagation 
def runMotionCudaBeliefPropRefineFromInputFloResults(inFloFile):

	# retrieve the 'refining parameters' from the input flo file...
	refiningParams = retrieveParamsRefineResults.retParamsToRefineResults(inFloFile)

	# set up the environment variables in order to run CUDA belief propagation
	setupCudaBeliefProp.initToRunCudaBeliefProp()

	# generate the input parameters to run CUDA belief propagation
	inParamCudaBeliefProp = inputParamsMultImages.InputParamsMultImages()

	# adjust the input parameters based on the 'flo' input...
	inParamCudaBeliefProp.adjustParticParams(refiningParams[retrieveParamsRefineResultsConsts.START_MOVE_INC_X_PARAM_KEY], refiningParams[retrieveParamsRefineResultsConsts.START_MOVE_INC_Y_PARAM_KEY], refiningParams[retrieveParamsRefineResultsConsts.MIN_X_MOVE_PARAM_KEY], refiningParams[retrieveParamsRefineResultsConsts.MAX_X_MOVE_PARAM_KEY], refiningParams[retrieveParamsRefineResultsConsts.MIN_Y_MOVE_PARAM_KEY], refiningParams[retrieveParamsRefineResultsConsts.MAX_Y_MOVE_PARAM_KEY])

	# generate the header file for the current set of parameters defined in runMotionBeliefPropConsts.py
	genHeaderFile.genCurrHeaderFile()

	# clean and make cuda belief propagation
	cleanAndMakeRunBeliefPropMotion.cleanMotionBeliefProp(runMotionBeliefPropConsts.DIR_MAKEFILE)
	cleanAndMakeRunBeliefPropMotion.makeMotionBeliefProp(runMotionBeliefPropConsts.DIR_MAKEFILE)

	# run cuda belief propagation
	exeBeliefPropMotion.runBeliefPropMotion(inParamCudaBeliefProp)
