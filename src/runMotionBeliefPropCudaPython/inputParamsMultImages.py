# inputParamsMultImages.py
# Scott Grauer-Gray
# September 28, 2010
# Python script to generate the 'input parameters' file...

# needed for the constants...
import runMotionBeliefPropConsts
import runMotionBeliefPropMultImages
import runMotionBeliefPropMultImagesInputs

# declaration for the 'input parameters' class...
class InputParamsMultImages:

	def __init__(self, inputFilePathBase = runMotionBeliefPropMultImagesInputs.FILE_PATH_INPUT_IMAGE_FILE_BASE, inputStartImageNum = runMotionBeliefPropMultImagesInputs.FILE_PATH_START_IMAGE_NUM, inputEndImageNum = runMotionBeliefPropMultImagesInputs.FILE_PATH_END_IMAGE_NUM, inputImageExtension = runMotionBeliefPropMultImagesInputs.FILE_PATH_END_EXTENSION, inputFilePathSaveMotion = runMotionBeliefPropMultImagesInputs.FILE_PATH_SAVE_RESULTING_MOTION_IMAGE, inputWidthInputImages = runMotionBeliefPropMultImagesInputs.WIDTH_INPUT_IMAGE_BP_MOTION_TEST, inputHeightInputImages = runMotionBeliefPropMultImagesInputs.HEIGHT_INPUT_IMAGE_BP_MOTION_TEST, 
			inputNumBeliefPropLevels = runMotionBeliefPropMultImagesInputs.NUM_BELIEF_PROP_LEVELS_BP_MOTION_TEST, inputNumBeliefPropIters = runMotionBeliefPropMultImagesInputs.NUM_BELIEF_PROP_ITERS_PER_LEVEL_BP_MOTION_TEST, inputDataCostCap = runMotionBeliefPropMultImagesInputs.DATA_COST_CAP_BP_MOTION_TEST, inputSmoothnessCostCap = runMotionBeliefPropMultImagesInputs.SMOOTHNESS_COST_CAP_BP_MOTION_TEST, inputDataCostWeight = runMotionBeliefPropMultImagesInputs.DATA_COST_WEIGHT_BP_MOTION_TEST, 
			inputMoveIncrementX = runMotionBeliefPropMultImagesInputs.CURRENT_MOVE_INCREMENT_X_BP_MOTION_TEST, inputMoveIncrementY = runMotionBeliefPropMultImagesInputs.CURRENT_MOVE_INCREMENT_Y_BP_MOTION_TEST, inputSampLevel = runMotionBeliefPropMultImagesInputs.SAMPLING_LEVEL_BP_MOTION_TEST, inputXMoveMin = runMotionBeliefPropMultImagesInputs.X_MOVE_MIN_BP_MOTION_TEST, inputXMoveMax = runMotionBeliefPropMultImagesInputs.X_MOVE_MAX_BP_MOTION_TEST, inputYMoveMin = runMotionBeliefPropMultImagesInputs.Y_MOVE_MIN_BP_MOTION_TEST, inputYMoveMax = runMotionBeliefPropMultImagesInputs.Y_MOVE_MAX_BP_MOTION_TEST, inputRangeDisp = runMotionBeliefPropMultImagesInputs.RANGE_DISPLAY_MOTION_TEST,
			inputGroundTruthMovement = runMotionBeliefPropMultImagesInputs.GROUND_TRUTH_MOVEMENT_FLO_FORMAT, inputFilePathSavedMovement = runMotionBeliefPropMultImagesInputs.FILE_PATH_SAVED_MOVEMENT_DATA_FLO_FORMAT, inputPropChangeMove = runMotionBeliefPropMultImagesInputs.PROP_CHANGE_MOVE_NEXT_LEVEL_BP_MOTION_TEST, inputSmoothingSigma = runMotionBeliefPropMultImagesInputs.SMOOTHING_SIGMA_BP_MOTION_TEST,
inputEstMoveCap = runMotionBeliefPropMultImagesInputs.EST_MOVEMENT_CAP_BP_MOTION_TEST, inputEstMoveWeight = runMotionBeliefPropMultImagesInputs.EST_MOVEMENT_WEIGHT_BP_MOTION_TEST, inputNumPyrHierarchLevs = runMotionBeliefPropMultImagesInputs.NUM_PYR_LEVELS_EACH_LEV_HIERARCHY_BP_MOTION_TEST):

		self.filePathBase = inputFilePathBase
		self.startImageNum = inputStartImageNum
		self.endImageNum = inputEndImageNum
		self.imageExtension = inputImageExtension
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

		generatedInParamString += str(self.filePathBase)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.startImageNum)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.endImageNum)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.imageExtension)
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
		generatedInParamString += str(self.estMoveCap)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.estMoveWeight)
		generatedInParamString += runMotionBeliefPropConsts.SPACE_CONST
		generatedInParamString += str(self.numPyrHierarchLevs)

		return generatedInParamString
