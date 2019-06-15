# mainRunCudaBeliefPropRefineFromFlo.py
# Scott Grauer-Gray
# November 29, 2010
# Main python script to run CUDA belief propagation

# needed to actually 'run' cuda belief propagation with the flo file and refine...
import runMotionBeliefPropRefineFromFloResults

# needed for the flo file to run from...
import runMotionBeliefPropRefineFromFloResultsConsts


# driver function for running cuda belief propagation...
def runCudaBeliefPropMainRefineFromFlo():

	# make the call to run 'main' belief propagation
	runMotionBeliefPropRefineFromFloResults.runMotionCudaBeliefPropRefineFromInputFloResults(runMotionBeliefPropRefineFromFloResultsConsts.FLO_PATH_REFINE_FROM)



# 'main' function which is run from the command line...
if __name__ == '__main__':

	# make the call to run cuda belief propagation
	runCudaBeliefPropMainRefineFromFlo()
