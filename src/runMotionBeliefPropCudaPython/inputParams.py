# inputParams.py
# Scott Grauer-Gray
# August 30, 2010
# Python script to generate the 'input parameters' file...

# needed for the constants...
import runMotionBeliefPropConsts
import runMotionBeliefPropInputs

# declaration for the 'input parameters' class...
class InputParams:

	def __init__(self, inputFilePathImage1 = runMotionBeliefPropInputs.FILE_PATH_INPUT_IMAGE_1, inputFilePathImage2 = runMotionBeliefPropInputs.FILE_PATH_INPUT_IMAGE_2, inputFilePathSaveMotion = runMotionBeliefPropInputs.FILE_PATH_SAVE_RESULTING_MOTION_IMAGE, inputWidthInputImages = runMotionBeliefPropInputs.WIDTH_INPUT_IMAGE_BP_MOTION_TEST, inputHeightInputImages = runMotionBeliefPropInputs.HEIGHT_INPUT_IMAGE_BP_MOTION_TEST, 
			inputNumBeliefPropLevels = runMotionBeliefPropInputs.NUM_BELIEF_PROP_LEVELS_BP_MOTION_TEST, inputNumBeliefPropIters = runMotionBeliefPropInputs.NUM_BELIEF_PROP_ITERS_PER_LEVEL_BP_MOTION_TEST, inputDataCostCap = runMotionBeliefPropInputs.DATA_COST_CAP_BP_MOTION_TEST, inputSmoothnessCostCap = runMotionBeliefPropInputs.SMOOTHNESS_COST_CAP_BP_MOTION_TEST, inputDataCostWeight = runMotionBeliefPropInputs.DATA_COST_WEIGHT_BP_MOTION_TEST, 
			inputMoveIncrementX = runMotionBeliefPropInputs.CURRENT_MOVE_INCREMENT_X_BP_MOTION_TEST, inputMoveIncrementY = runMotionBeliefPropInputs.CURRENT_MOVE_INCREMENT_Y_BP_MOTION_TEST, inputSampLevel = runMotionBeliefPropInputs.SAMPLING_LEVEL_BP_MOTION_TEST, inputXMoveMin = runMotionBeliefPropInputs.X_MOVE_MIN_BP_MOTION_TEST, inputXMoveMax = runMotionBeliefPropInputs.X_MOVE_MAX_BP_MOTION_TEST, inputYMoveMin = runMotionBeliefPropInputs.Y_MOVE_MIN_BP_MOTION_TEST, inputYMoveMax = runMotionBeliefPropInputs.Y_MOVE_MAX_BP_MOTION_TEST, inputRangeDisp = runMotionBeliefPropInputs.RANGE_DISPLAY_MOTION_TEST,
			inputGroundTruthMovement = runMotionBeliefPropInputs.GROUND_TRUTH_MOVEMENT_FLO_FORMAT, inputFilePathSavedMovement = runMotionBeliefPropInputs.FILE_PATH_SAVED_MOVEMENT_DATA_FLO_FORMAT, inputPropChangeMove = runMotionBeliefPropInputs.PROP_CHANGE_MOVE_NEXT_LEVEL_BP_MOTION_TEST, inputSmoothingSigma = runMotionBeliefPropInputs.SMOOTHING_SIGMA_BP_MOTION_TEST, inputNumPyrHierarchLevs = runMotionBeliefPropInputs.NUM_PYR_LEVELS_EACH_LEV_HIERARCHY):

		self.filePathImage1 = inputFilePathImage1
		self.filePathImage2 = inputFilePathImage2
		self.filePathSaveMotion = inputFilePathSaveMotion
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
		self.numPyrHierarchLevs = inputNumPyrHierarchLevs

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
		generatedInParamString += str(self.numPyrHierarchLevs)


		return generatedInParamString

