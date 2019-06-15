# mainGetDataFromFloFiles.py
# Scott Grauer-Gray
# December 1, 2010
# Main function to get data from the flo files...

# needed to run the function to get the data...
import runRetrieveErrorFromFloFiles


# call the function to get the data...
def getDataFromFloFiles():

	runRetrieveErrorFromFloFiles.runMotionCudaBeliefProp()


# main so that it works from the command line
if __name__ == '__main__':

	getDataFromFloFiles()

