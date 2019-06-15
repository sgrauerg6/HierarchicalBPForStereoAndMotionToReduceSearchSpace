# runRetrieveErrorFromFloFiles.py
# Scott Grauer-Gray
# December 1, 2010
# Python script to retrieve the error from flo files...

# needed to clean/make cuda belief propagation
import cleanAndMakeRunBeliefPropMotion

# needed to execute cuda belief propagation
import exeBeliefPropMotion

# needed for the constants corresponding to running motion belief propagation
import runMotionBeliefPropConsts

# needed for the input parameters for retrieving the data from flo files...
import inputParamsRetrieveDataFromFloFiles

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

# function for running motion belief propagation 
def runMotionCudaBeliefProp():

	# set up the environment variables in order to run CUDA belief propagation
	setupCudaBeliefProp.initToRunCudaBeliefProp()

	# generate the input parameters to run CUDA belief propagation
	inParamCudaBeliefProp = inputParamsRetrieveDataFromFloFiles.InputParamsRetrieveDataFromFloFiles()

	# clean and make cuda belief propagation
	cleanAndMakeRunBeliefPropMotion.cleanMotionBeliefProp(runMotionBeliefPropConsts.DIR_MAKEFILE)
	cleanAndMakeRunBeliefPropMotion.makeMotionBeliefProp(runMotionBeliefPropConsts.DIR_MAKEFILE)

	# run cuda belief propagation
	exeBeliefPropMotion.runBeliefPropMotion(inParamCudaBeliefProp)
