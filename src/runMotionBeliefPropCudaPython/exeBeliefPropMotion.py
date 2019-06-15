# exeBeliefPropMotion.py
# Scott Grauer-Gray
# August 30, 2010
# Python script with commands for executing belief propagation for motion

# needed for changing directories...
import os

# needed for actually running belief propagation motion
import subprocess

# needed for constants such as the directory of the executable...
import runMotionBeliefPropConsts

# needed since input parameters are used here...
import inputParams

# needed since the path of the output text is here...
import runMotionBeliefPropInputs

# function to actually "execute" belief propagation "motion"...
def runBeliefPropMotion(inputParamRunBeliefPropMotion, outTextFile = None):

	# go the the directory with belief propagation motion
	os.chdir(runMotionBeliefPropConsts.DIR_GENERATED_EXECUTABLE)

	# generate the string with the parameters for running belief propagation motion
	strBeliefPropMotionParams = inputParamRunBeliefPropMotion.genStringInputParams()

	print runMotionBeliefPropConsts.PATH_MOTION_BELIEF_PROP_EXECUTABLE + runMotionBeliefPropConsts.SPACE_CONST + strBeliefPropMotionParams

	# now execute belief propagation motion...
	if (outTextFile is None):

		p = subprocess.Popen(runMotionBeliefPropConsts.PATH_MOTION_BELIEF_PROP_EXECUTABLE + runMotionBeliefPropConsts.SPACE_CONST + strBeliefPropMotionParams, shell=True)

	else:
		p = subprocess.Popen(runMotionBeliefPropConsts.PATH_MOTION_BELIEF_PROP_EXECUTABLE + runMotionBeliefPropConsts.SPACE_CONST + strBeliefPropMotionParams + ' > ' + outTextFile, shell=True)

	# wait until the execution is complete...
	p.wait()


