# mainRunCudaBeliefPropFromExpMotion.py
# Scott Grauer-Gray
# November 29, 2010
# Main function to run belief propagation given a 'flo' file with the expected motion

# needed to actually 'run' cuda belief propagation with expected motion...
import runMotionBeliefPropGivenExpMotion


# driver function for running cuda belief propagation...
def runCudaBeliefPropMainGivenExpectedMotion():

	# make the call to run 'main' belief propagation
	runMotionBeliefPropGivenExpMotion.runMotionCudaBeliefProp()


# 'main' function which is run from the command line...
if __name__ == '__main__':

	# make the call to run cuda belief propagation
	runCudaBeliefPropMainGivenExpectedMotion()
