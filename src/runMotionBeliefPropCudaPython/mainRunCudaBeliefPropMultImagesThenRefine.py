# mainRunCudaBeliefPropRefineFromFloAndUseNoExpMotionResults.py
# Scott Grauer-Gray
# December 1, 2010
# Main function for first running belief propagation for motion to get rough results and then refine them...

# needed for running multiple images for "rough" results
import runMotionBeliefPropMultImages

# needed to "refine" results
import runMotionBeliefPropRefineFromFloResultsAndUseNoExpMoveFloResult

# function to run multiple images then refine
def runCudaBeliefPropRefineFromFloAndUseNoExpMotionResults():

	runMotionBeliefPropMultImages.runMotionCudaBeliefProp()
	runMotionBeliefPropRefineFromFloResultsAndUseNoExpMoveFloResult.runMotionBeliefPropTakePrevMotionAndNoExpMotionResultIntoAccount()

# run from the command line...
if __name__ == '__main__':

	# make the call to run cuda belief propagation
	runCudaBeliefPropRefineFromFloAndUseNoExpMotionResults() 
