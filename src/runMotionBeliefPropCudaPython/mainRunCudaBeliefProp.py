# mainRunCudaBeliefProp.py
# Scott Grauer-Gray
# September 7, 2010
# Main python script to run CUDA belief propagation

# needed to actually 'run' cuda belief propagation
import runMotionBeliefProp


# driver function for running cuda belief propagation...
def runCudaBeliefPropMain():

	# make the call to run 'main' belief propagation
	runMotionBeliefProp.runMotionCudaBeliefProp()



# 'main' function which is run from the command line...
if __name__ == '__main__':

	# make the call to run cuda belief propagation
	runCudaBeliefPropMain()
