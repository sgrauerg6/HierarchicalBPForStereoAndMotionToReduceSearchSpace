# mainRunCudaBeliefPropMultImages.py
# Scott Grauer-Gray
# September 29, 2010
# Main python script to run CUDA belief propagation

# needed to actually 'run' cuda belief propagation with multiple input images...
import runMotionBeliefPropMultImages


# driver function for running cuda belief propagation...
def runCudaBeliefPropMainMultImages():

	# make the call to run 'main' belief propagation
	runMotionBeliefPropMultImages.runMotionCudaBeliefProp()



# 'main' function which is run from the command line...
if __name__ == '__main__':

	# make the call to run cuda belief propagation
	runCudaBeliefPropMainMultImages()
