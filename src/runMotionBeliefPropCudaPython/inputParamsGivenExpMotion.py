# inputParamsGivenExpMotion.py
# Scott Grauer-Gray
# November 29, 2010
# Python script to generate the 'input parameters' file given the expected motion (given via a flo file...)...

# needed for the constants...
import runMotionBeliefPropConsts
import runMotionBeliefPropGivenExpMotionInputs

# declaration for the 'input parameters' class...
class InputParamsGivenExpMotion:

	def __init__(self, inputFilePathImage1 = runMotionBeliefPropGivenExpMotionInputs.FILE_PATH_INPUT_IMAGE_1, inputFilePathImage2 = runMotionBeliefPropGivenExpMotionInputs.FILE_PATH_INPUT_IMAGE_2, inputFilePathSaveMotion = runMotionBeliefPropGivenExpMotionInputs.FILE_PATH_SAVE_RESULTING_MOTION_IMAGE, inputFloFileExpectedMotion = runMotionBeliefPropGivenExpMotionInputs.FILE_PATH_EXPECTED_MOTION, inputWidthInputImages = runMotionBeliefPropGivenExpMotionInputs.WIDTH_INPUT_IMAGE_BP_MOTION_TEST, inputHeightInputImages = runMotionBeliefPropGivenExpMotionInputs.HEIGHT_INPUT_IMAGE_BP_MOTION_TEST, 
			inputNumBeliefPropLevels = runMotionBeliefPropGivenExpMotionInputs.NUM_BELIEF_PROP_LEVELS_BP_MOTION_TEST, inputNumBeliefPropIters = runMotionBeliefPropGivenExpMotionInputs.NUM_BELIEF_PROP_ITERS_PER_LEVEL_BP_MOTION_TEST, inputDataCostCap = runMotionBeliefPropGivenExpMotionInputs.DATA_COST_CAP_BP_MOTION_TEST, inputSmoothnessCostCap = runMotionBeliefPropGivenExpMotionInputs.SMOOTHNESS_COST_CAP_BP_MOTION_TEST, inputDataCostWeight = runMotionBeliefPropGivenExpMotionInputs.DATA_COST_WEIGHT_BP_MOTION_TEST, 
			inputMoveIncrementX = runMotionBeliefPropGivenExpMotionInputs.CURRENT_MOVE_INCREMENT_X_BP_MOTION_TEST, inputMoveIncrementY = runMotionBeliefPropGivenExpMotionInputs.CURRENT_MOVE_INCREMENT_Y_BP_MOTION_TEST, inputSampLevel = runMotionBeliefPropGivenExpMotionInputs.SAMPLING_LEVEL_BP_MOTION_TEST, inputXMoveMin = runMotionBeliefPropGivenExpMotionInputs.X_MOVE_MIN_BP_MOTION_TEST, inputXMoveMax = runMotionBeliefPropGivenExpMotionInputs.X_MOVE_MAX_BP_MOTION_TEST, inputYMoveMin = runMotionBeliefPropGivenExpMotionInputs.Y_MOVE_MIN_BP_MOTION_TEST, inputYMoveMax = runMotionBeliefPropGivenExpMotionInputs.Y_MOVE_MAX_BP_MOTION_TEST, inputRangeDisp = runMotionBeliefPropGivenExpMotionInputs.RANGE_DISPLAY_MOTION_TEST,
			inputGroundTruthMovement = runMotionBeliefPropGivenExpMotionInputs.GROUND_TRUTH_MOVEMENT_FLO_FORMAT, inputFilePathSavedMovement = runMotionBeliefPropGivenExpMotionInputs.FILE_PATH_SAVED_MOVEMENT_DATA_FLO_FORMAT, inputPropChangeMove = runMotionBeliefPropGivenExpMotionInputs.PROP_CHANGE_MOVE_NEXT_LEVEL_BP_MOTION_TEST, inputSmoothingSigma = runMotionBeliefPropGivenExpMotionInputs.SMOOTHING_SIGMA_BP_MOTION_TEST,
inputEstMoveCap = runMotionBeliefPropGivenExpMotionInputs.EST_MOVEMENT_CAP_BP_MOTION_TEST, inputEstMoveWeight = runMotionBeliefPropGivenExpMotionInputs.EST_MOVEMENT_WEIGHT_BP_MOTION_TEST, inputNumPyrHierarchLevs = runMotionBeliefPropGivenExpMotionInputs.NUM_PYR_LEVELS_EACH_LEV_HIERARCHY_BP_MOTION_TEST):

		self.filePathImage1 = inputFilePathImage1
		self.filePathImage2 = inputFilePathImage2
		self.filePathSaveMotion = inputFilePathSaveMotion
		self.filePathExpectedMotion = inputFloFileExpectedMotion
		self.widthInputImages = inputWidthInputImages
		self.heightInputImages = inputHeightInputImages
		self.numBeliefPropLevels = inputNumBeliefPropLevels
		self.numBeliefPropIters = inputNumBeliefPropIters
		self.dataCostCap = inputDataCostCap
		self.smoothnessCostCap = inputSmoothnessCostCap
		self.dataCostWeight = inputDataCostWeight
		self.moveIncrementX = inputMoveIncrementX
		self.moveIncrementY = inputMoveIncrementY
		self.sampLevel = inputSampLevel
		self.xMoveMin = inputXMoveMin
		self.xMoveMax = inputXMoveMax
		self.yMoveMin = inputYMoveMin
		self.yMoveMax = inputYMoveMax
		self.rangeDisp = inputRangeDisp
		self.groundTruthMovement = inputGroundTruthMovement
		self.filePathSavedMovement = inputFilePathSavedMovement
		self.propChangeMove = inputPropChangeMove
		self.smoothingSigma = inputSmoothingSigma
		self.estMoveCap = inputEstMoveCap
		self.estMoveWeight = inputEstMoveWeight
		self.numPyrHierarchLevs = inputNumPyrHierarchLevs

	# function to adjust the particular parameters
	def adjustParticParams(self, inputMoveIncrementX, inputMoveIncrementY, inputXMoveMin, inputXMoveMax, inputYMoveMin, inputYMoveMax):

		self.moveIncrementX = inputMoveIncrementX
		self.moveIncrementY = inputMoveIncrementY
		self.xMoveMin = inputXMoveMin
		self.xMoveMax = inputXMoveMax
		self.yMoveMin = inputYMoveMin
		self.yMoveMax = inputYMoveMax

	# adjust the parameter to use as the expected motion 
	def adjustInputFloFile(self, inputFloFile):
		self.filePathExpectedMotion = inputFloFile

	# adjust the file path of the output 'movement' image
	def adjustOutputMovementImage(self, outputMovementImageFilePath):
		self.filePathSaveMotion = outputMovementImageFilePath

	# adjust the parameter for the weight of the expected motion...
	def adjustTakeExpMotionWeight(self, expMotionWeight):
		self.estMoveWeight = expMotionWeight	

	# adjust the parameter for the flo file of the output motion
	def adjustOutputMotionFloFile(self, outputMotionFloFile):
		self.filePathSavedMovement = outputMotionFloFile

	# adjust the smoothing sigma for the input images...
	def adjustSmoothingSigmaInputImages(self, inSmoothingSigma):
		self.smoothingSigma = inSmoothingSigma


	# function to retrieve the maximum number of moves in the x direction
	def retMaxNumMovesXDir(self):

		return (int(floor((self.xMoveMax - self.xMoveMin) / self.moveIncrementX)) + 1) 


	# function to retrieve the maximum number of moves in the y direction
	def retMaxNumMovesYDir(self):

		return (int(floor((self.yMoveMax - self.yMoveMin) / self.moveIncrementY)) + 1) 


	# function to generate the string for the input parameters
	def genStringInputParams(self):

		# initialize the string for the input parameters
		generatedInParamString = ''

		generatedInParamString += str(self.filePathImage1)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.filePathImage2)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.filePathSaveMotion)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.filePathExpectedMotion)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.widthInputImages)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.heightInputImages)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.numBeliefPropLevels)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.numBeliefPropIters)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.dataCostCap)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.smoothnessCostCap)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.dataCostWeight)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.moveIncrementX)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.moveIncrementY)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.sampLevel)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.xMoveMin)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.xMoveMax)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.yMoveMin)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.yMoveMax)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.rangeDisp)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.groundTruthMovement)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.filePathSavedMovement)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.propChangeMove)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.smoothingSigma)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.estMoveCap)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.estMoveWeight)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.numPyrHierarchLevs)

		return generatedInParamString
