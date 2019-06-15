# inputParamsRetrieveDataFromFloFiles.py
# Scott Grauer-Gray
# December 1, 2010
# Python script to generate the 'input parameters' file...

# needed for the constants...
import runRetrieveErrorFromFloFilesInputs
import runMotionBeliefPropConsts

# declaration for the 'input parameters' class...
class InputParamsRetrieveDataFromFloFiles:

	def __init__(self, inputGroundTruthMovementFilePath = runRetrieveErrorFromFloFilesInputs.GROUND_TRUTH_MOVEMENT_FILE_PATH, inputOrigMovementFilePath = runRetrieveErrorFromFloFilesInputs.ORIG_MOVEMENT_FILE_PATH, inputRoughMultImageMovementFilePath = runRetrieveErrorFromFloFilesInputs.ROUGH_MULT_IMAGE_MOVEMENT_FILE_PATH, inputAdjustOrigMovementFilePath = runRetrieveErrorFromFloFilesInputs.ADJUST_ORIG_MOVEMENT_FILE_PATH, inputRefinedMovementFilePath = runRetrieveErrorFromFloFilesInputs.REFINED_MOVEMENT_FILE_PATH):

		self.groundTruthMovementFilePath = inputGroundTruthMovementFilePath
		self.origMovementFilePath = inputOrigMovementFilePath
		self.roughMultImageMovementFilePath = inputRoughMultImageMovementFilePath
		self.adjustOrigMovementFilePath = inputAdjustOrigMovementFilePath
		self.refinedMovementFilePath = inputRefinedMovementFilePath		


	# function to generate the string for the input parameters
	def genStringInputParams(self):

		# initialize the string for the input parameters
		generatedInParamString = ''

		generatedInParamString += str(self.groundTruthMovementFilePath)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.origMovementFilePath)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.roughMultImageMovementFilePath)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.adjustOrigMovementFilePath)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.refinedMovementFilePath)

		return generatedInParamString

