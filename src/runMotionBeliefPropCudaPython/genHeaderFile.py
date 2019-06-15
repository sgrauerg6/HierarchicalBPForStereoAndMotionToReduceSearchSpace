# genHeaderFile.py
# Scott Grauer-Gray
# September 2, 2010
# Python script for the generation of the C/C++ header file

# needed for the general constants...
import runMotionBeliefPropConsts

# needed for the input for running motion belief propagation
import runMotionBeliefPropInputs

# needed for the constants related to the input parameters
import inputParamsConsts

# needed for os-related commands such as changing the directory
import os

# needed for the 'ceil' math operation
import math

# function for the generation of the desired header file
def genCurrHeaderFile():

	# retrieve the total number of moves in the x/y direction...
	numMovesXDir = int(math.ceil((runMotionBeliefPropInputs.X_MOVE_MAX_BP_MOTION_TEST - runMotionBeliefPropInputs.X_MOVE_MIN_BP_MOTION_TEST) / runMotionBeliefPropInputs.CURRENT_MOVE_INCREMENT_X_BP_MOTION_TEST)) + 1
	numMovesYDir = int(math.ceil((runMotionBeliefPropInputs.Y_MOVE_MAX_BP_MOTION_TEST - runMotionBeliefPropInputs.Y_MOVE_MIN_BP_MOTION_TEST) / runMotionBeliefPropInputs.CURRENT_MOVE_INCREMENT_Y_BP_MOTION_TEST)) + 1

	# go to the folder to place the header file...
	os.chdir(inputParamsConsts.FILE_PATH_HEADER_FILE)

	# generate a file pointer for the header file
	headerFilePointer = open(inputParamsConsts.FILE_NAME_HEADER_FILE, 'w')

	# write the 'top' of the header (the 'ifndef/define stuff)
	headerFilePointer.write(inputParamsConsts.START_HEADER_FILE_INPUT_PARAMS)

	# write the maximum movement in the x and y direction to the header file
	headerFilePointer.write(inputParamsConsts.DEFINE_DIR_HEADER_FILE)
	headerFilePointer.write(runMotionBeliefPropConsts.SPACE_CONST)
	headerFilePointer.write(inputParamsConsts.NAME_MAX_NUM_MOVES_X_DIR)
	headerFilePointer.write(runMotionBeliefPropConsts.SPACE_CONST)
	headerFilePointer.write(str(numMovesXDir))
	headerFilePointer.write(runMotionBeliefPropConsts.SPACE_CONST)
	headerFilePointer.write(runMotionBeliefPropConsts.NEW_LINE_CONST)	

	headerFilePointer.write(inputParamsConsts.DEFINE_DIR_HEADER_FILE)
	headerFilePointer.write(runMotionBeliefPropConsts.SPACE_CONST)
	headerFilePointer.write(inputParamsConsts.NAME_MAX_NUM_MOVES_Y_DIR)
	headerFilePointer.write(runMotionBeliefPropConsts.SPACE_CONST)
	headerFilePointer.write(str(numMovesYDir))
	headerFilePointer.write(runMotionBeliefPropConsts.SPACE_CONST)
	headerFilePointer.write(runMotionBeliefPropConsts.NEW_LINE_CONST)

	# write the 'end' of the header...
	headerFilePointer.write(inputParamsConsts.END_HEADER_FILE_INPUT_PARAMS)

	# close the file...
	headerFilePointer.close()

