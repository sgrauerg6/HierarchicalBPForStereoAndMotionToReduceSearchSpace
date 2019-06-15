# mainRunCudaBeliefPropRefineFromFloAndUseNoExpMotionResults.py
# Scott Grauer-Gray
# November 20, 2010
# Main python script to run CUDA belief propagation

# needed to actually 'run' cuda belief propagation with the flo file and refine...
import runMotionBeliefPropRefineFromFloResultsAndUseNoExpMoveFloResult


# driver function for running cuda belief propagation...
def runCudaBeliefPropMainRefineFromFloAndUseNoExpMotionResults():

	# make the call to run 'main' belief propagation
	runMotionBeliefPropRefineFromFloResultsAndUseNoExpMoveFloResult.runMotionBeliefPropTakePrevMotionAndNoExpMotionResultIntoAccount()



# 'main' function which is run from the command line...
if __name__ == '__main__':

	# make the call to run cuda belief propagation
	runCudaBeliefPropMainRefineFromFloAndUseNoExpMotionResults()
