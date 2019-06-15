# retrieveParamsRefineResults.py
# Scott Grauer-Gray
# November 28, 2010
# Python script to retrieve the desired parameters to refine the results...

# needed to retrieve the extreme values from the flo file...
import retrieveExtremesFromFloFile

# needed for constants used for refining the results...
import retrieveParamsRefineResultsConsts

# needed for keys for getting the extreme values from the dictionary...
import retrieveExtremesFromFloFileConsts

# function that takes in the flo file of the 'rough' results and returns the parameters for the refined results
def retParamsToRefineResults(floFileRoughResults):

	# first retrieve the 'extreme' values
	dictExtremeVals = retrieveExtremesFromFloFile.getExtremesFromFloFile(floFileRoughResults)

	# retrieve the 'extreme values' to use
	minXValUse = max(min(0.0, dictExtremeVals[retrieveExtremesFromFloFileConsts.MIN_X_MOVE_KEY]) * (retrieveParamsRefineResultsConsts.MULTIPLIER_MIN_MAX_MOVE), retrieveParamsRefineResultsConsts.MIN_POSS_MOVE_X )
	maxXValUse = min(max(0.0, dictExtremeVals[retrieveExtremesFromFloFileConsts.MAX_X_MOVE_KEY]) * (retrieveParamsRefineResultsConsts.MULTIPLIER_MIN_MAX_MOVE), retrieveParamsRefineResultsConsts.MAX_POSS_MOVE_X )
	minYValUse = max(min(0.0, dictExtremeVals[retrieveExtremesFromFloFileConsts.MIN_Y_MOVE_KEY]) * (retrieveParamsRefineResultsConsts.MULTIPLIER_MIN_MAX_MOVE), retrieveParamsRefineResultsConsts.MIN_POSS_MOVE_Y )
	maxYValUse = min(max(0.0, dictExtremeVals[retrieveExtremesFromFloFileConsts.MAX_Y_MOVE_KEY]) * (retrieveParamsRefineResultsConsts.MULTIPLIER_MIN_MAX_MOVE), retrieveParamsRefineResultsConsts.MAX_POSS_MOVE_Y )

	# retrieve the increment of movement in the x and y directions
	incMoveXDir = (maxXValUse - minXValUse) / 4.0   
	incMoveYDir = (maxYValUse - minYValUse) / 4.0   

	# initialize, set the values in, and return a dictionary with the desired parameters...
	desiredParams = {}
	desiredParams[retrieveParamsRefineResultsConsts.MIN_X_MOVE_PARAM_KEY] = minXValUse
	desiredParams[retrieveParamsRefineResultsConsts.MIN_Y_MOVE_PARAM_KEY] = minYValUse
	desiredParams[retrieveParamsRefineResultsConsts.MAX_X_MOVE_PARAM_KEY] = maxXValUse
	desiredParams[retrieveParamsRefineResultsConsts.MAX_Y_MOVE_PARAM_KEY] = maxYValUse
	desiredParams[retrieveParamsRefineResultsConsts.START_MOVE_INC_X_PARAM_KEY] = incMoveXDir
	desiredParams[retrieveParamsRefineResultsConsts.START_MOVE_INC_Y_PARAM_KEY] = incMoveYDir

	print desiredParams

	# return the generated set of parameters
	return desiredParams
