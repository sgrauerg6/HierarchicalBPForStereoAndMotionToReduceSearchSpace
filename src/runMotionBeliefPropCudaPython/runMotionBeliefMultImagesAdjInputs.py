# runMotionBeliefPropMultImagesAdjInputs.py
# Scott Grauer-Gray
# October 11, 2010
# Python script for running motion belief propagation with adjustable inputs...

# needed for the inputs for the function
import runMotionBeliefPropMultImagesAdjInputsInVals

# needed for the makefile stuff
import runMotionBeliefPropConsts

# needed to extract output info
import extractResultAsDict

# needed for the input parameters...
import inputParamsMultImages

# needed for setting up belief prop...
import setupCudaBeliefProp

# needed to remove a file...
import os

# needed for executing program...
import exeBeliefPropMotion

# needed for cleaning/running program
import cleanAndMakeRunBeliefPropMotion

# needed for header file generation
import genHeaderFile

# needed for the names associated with the adjustable input values...
import runMotionBeliefPropMultImagesAdjInputsConsts

# needed to write the output results to a csv file...
import writeOutputDictToCsv

def runMotionBpMultImagesAdjInputs():

	# initialize a dictionary for all the results
	allResultsDict = {}

	# need a loop for each 'different' input

	# make the 'data cost' adjustable
	for currDataCost in runMotionBeliefPropMultImagesAdjInputsInVals.LIST_DATA_COST_CAPS:

		# make the 'data weight' adjustable
		for currDataWeight in runMotionBeliefPropMultImagesAdjInputsInVals.LIST_DATA_WEIGHTS:

			# make the 'smoothness cost' adjustable
			for currSmoothnessCost in runMotionBeliefPropMultImagesAdjInputsInVals.LIST_SMOOTHNESS_COST_CAPS:

				# make the 'estimated movement weight' adjustable
				for currEstMoveWeight in runMotionBeliefPropMultImagesAdjInputsInVals.LIST_EST_MOVE_WEIGHTS:

					# make the 'estimated movement cost' adjustable
					for currEstMoveCost in runMotionBeliefPropMultImagesAdjInputsInVals.LIST_EST_MOVE_COST_CAPS:

						# initialize the dictionary with the current set of results
						dictResultsSet = {runMotionBeliefPropMultImagesAdjInputsConsts.DATA_COST_CAP_NAME_CONST : currDataCost, runMotionBeliefPropMultImagesAdjInputsConsts.DATA_WEIGHT_NAME_CONST : currDataWeight, runMotionBeliefPropMultImagesAdjInputsConsts.SMOOTHNESS_COST_CAP_NAME_CONST : currSmoothnessCost, runMotionBeliefPropMultImagesAdjInputsConsts.EST_MOVEMENT_WEIGHT_NAME_CONST : currEstMoveWeight, runMotionBeliefPropMultImagesAdjInputsConsts.EST_MOVEMENT_COST_CAP_NAME_CONST : currEstMoveCost}


						# set up the environment variables in order to run CUDA belief propagation
						setupCudaBeliefProp.initToRunCudaBeliefProp()

						# generate the input parameters to run CUDA belief propagation
						inParamCudaBeliefProp = inputParamsMultImages.InputParamsMultImages(runMotionBeliefPropMultImagesAdjInputsInVals.FILE_PATH_INPUT_IMAGE_FILE_BASE, runMotionBeliefPropMultImagesAdjInputsInVals.FILE_PATH_START_IMAGE_NUM, runMotionBeliefPropMultImagesAdjInputsInVals.FILE_PATH_END_IMAGE_NUM, runMotionBeliefPropMultImagesAdjInputsInVals.FILE_PATH_END_EXTENSION, runMotionBeliefPropMultImagesAdjInputsInVals.FILE_PATH_SAVE_RESULTING_MOTION_IMAGE, runMotionBeliefPropMultImagesAdjInputsInVals.WIDTH_INPUT_IMAGE_BP_MOTION_TEST, runMotionBeliefPropMultImagesAdjInputsInVals.HEIGHT_INPUT_IMAGE_BP_MOTION_TEST, runMotionBeliefPropMultImagesAdjInputsInVals.NUM_BELIEF_PROP_LEVELS_BP_MOTION_TEST, runMotionBeliefPropMultImagesAdjInputsInVals.NUM_BELIEF_PROP_ITERS_PER_LEVEL_BP_MOTION_TEST, currDataCost, currSmoothnessCost, currDataWeight, runMotionBeliefPropMultImagesAdjInputsInVals.CURRENT_MOVE_INCREMENT_BP_MOTION_TEST, runMotionBeliefPropMultImagesAdjInputsInVals.SAMPLING_LEVEL_BP_MOTION_TEST, runMotionBeliefPropMultImagesAdjInputsInVals.X_MOVE_MIN_BP_MOTION_TEST, runMotionBeliefPropMultImagesAdjInputsInVals.X_MOVE_MAX_BP_MOTION_TEST, runMotionBeliefPropMultImagesAdjInputsInVals.Y_MOVE_MIN_BP_MOTION_TEST, runMotionBeliefPropMultImagesAdjInputsInVals.Y_MOVE_MAX_BP_MOTION_TEST, runMotionBeliefPropMultImagesAdjInputsInVals.RANGE_DISPLAY_MOTION_TEST, runMotionBeliefPropMultImagesAdjInputsInVals.GROUND_TRUTH_MOVEMENT_FLO_FORMAT, runMotionBeliefPropMultImagesAdjInputsInVals.FILE_PATH_SAVED_MOVEMENT_DATA_FLO_FORMAT, runMotionBeliefPropMultImagesAdjInputsInVals.PROP_CHANGE_MOVE_NEXT_LEVEL_BP_MOTION_TEST, runMotionBeliefPropMultImagesAdjInputsInVals.SMOOTHING_SIGMA_BP_MOTION_TEST, currEstMoveCost, currEstMoveWeight)

						# generate the header file for the current set of parameters defined in runMotionBeliefPropConsts.py
						genHeaderFile.genCurrHeaderFile()

						# clean and make cuda belief propagation
						cleanAndMakeRunBeliefPropMotion.cleanMotionBeliefProp(runMotionBeliefPropConsts.DIR_MAKEFILE)
						cleanAndMakeRunBeliefPropMotion.makeMotionBeliefProp(runMotionBeliefPropConsts.DIR_MAKEFILE)

						# run cuda belief propagation
						exeBeliefPropMotion.runBeliefPropMotion(inParamCudaBeliefProp, runMotionBeliefPropMultImagesAdjInputsInVals.FILE_SAVE_OUT_MOTION)
						
						# extract the results as a dictionary
						resultsDict = extractResultAsDict.retResultingDict(runMotionBeliefPropMultImagesAdjInputsInVals.FILE_SAVE_OUT_MOTION)

						# add the 'extracted results'
						dictResultsSet.update(resultsDict)

						print allResultsDict

						# go through the set of results and add to the overall results...
						for dictKey, dictVal in dictResultsSet.iteritems():

							if (dictKey in allResultsDict.keys()):

								allResultsDict[dictKey].append(dictVal)

							else:
			
								allResultsDict[dictKey] = [dictVal]

						# delete the file with the saved info
						os.remove(runMotionBeliefPropMultImagesAdjInputsInVals.FILE_SAVE_OUT_MOTION)

						print allResultsDict

	# print the output dictionary
	print allResultsDict

	# write the results to a csv file...
	writeOutputDictToCsv.writeOutputResultsCsvFile(allResultsDict, runMotionBeliefPropMultImagesAdjInputsInVals.FILE_OUT_MOTION_INFO)



# 'main' function to run motin belief propagation with adjustable inputs from the command line...
if __name__ == '__main__':

	# make the call to run cuda belief propagation
	runMotionBpMultImagesAdjInputs() 
	
