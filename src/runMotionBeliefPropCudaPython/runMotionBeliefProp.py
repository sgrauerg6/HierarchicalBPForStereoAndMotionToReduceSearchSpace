# runMotionBeliefProp.py
# Scott Grauer-Gray
# September 7, 2010
# Python script to run "motion" belief propagation

# needed to clean/make cuda belief propagation
import cleanAndMakeRunBeliefPropMotion

# needed to execute cuda belief propagation
import exeBeliefPropMotion

# needed for the constants corresponding to running motion belief propagation
import runMotionBeliefPropConsts

# needed for the input parameters
import inputParams

# needed for setting up CUDA belief propagation
import setupCudaBeliefProp

# needed for generating the header file...
import genHeaderFile

# function for running motion belief propagation 
def runMotionCudaBeliefProp():

	# set up the environment variables in order to run CUDA belief propagation
	setupCudaBeliefProp.initToRunCudaBeliefProp()

	# generate the input parameters to run CUDA belief propagation
	inParamCudaBeliefProp = inputParams.InputParams()

	# generate the header file for the current set of parameters defined in runMotionBeliefPropConsts.py
	genHeaderFile.genCurrHeaderFile()

	# clean and make cuda belief propagation
	cleanAndMakeRunBeliefPropMotion.cleanMotionBeliefProp(runMotionBeliefPropConsts.DIR_MAKEFILE)
	cleanAndMakeRunBeliefPropMotion.makeMotionBeliefProp(runMotionBeliefPropConsts.DIR_MAKEFILE)

	# run cuda belief propagation
	exeBeliefPropMotion.runBeliefPropMotion(inParamCudaBeliefProp)
