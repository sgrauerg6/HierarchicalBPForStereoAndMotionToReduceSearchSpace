# retrieveExtremesFromFloFile.py
# Scott Grauer-Gray
# November 28, 2010
# Python script to retrievew the extreme values in the x and y directions from the flo file...

# needed for the constants related to getting the extremes...
import retrieveExtremesFromFloFileConsts

# needed to deal with binary data...
import struct

# function that takes in the flo file path and returns a dictionary with the extreme values in the file...
def getExtremesFromFloFile(floFilePath):

	# open the flo file...
	floFilePointer = open(floFilePath, 'rb')

	# extract the first four bytes, which should be 'PIEH' in ASCII
	firstFourBytes = floFilePointer.read(4)

	print str(firstFourBytes)

	# extract the next four bytes, which is the width as an integer
	#widthFloBytes = (floFilePointer.read(4))
	widthFloBytes = struct.unpack('i', floFilePointer.read(4))
	print "width: " + str(widthFloBytes[0])


	# extract the next four bytes, which is the height as an integer
	heightFloBytes = struct.unpack('i', floFilePointer.read(4))
	print "height: " + str(heightFloBytes[0])

	# retrieve the number of bytes in the rest of the file
	numRemainingVals = widthFloBytes[0] * heightFloBytes[0]

	# retrieve the 'first' value in the x/y direction and make it the 'extreme'
	firstValXDir = (struct.unpack('f', floFilePointer.read(4)))[0]
	firstValYDir = (struct.unpack('f', floFilePointer.read(4)))[0]

	minXDir = firstValXDir
	minYDir = firstValYDir
	maxXDir = firstValXDir
	maxYDir = firstValYDir

	numRemainingVals = numRemainingVals - 1

	
	while (numRemainingVals > 0):
	
		valXDir = struct.unpack('f', floFilePointer.read(4))[0]

		if (valXDir < minXDir):
			minXDir = valXDir
		if (valXDir > maxXDir):
			maxXDir = valXDir

		valYDir = struct.unpack('f', floFilePointer.read(4))[0]
		
		if (valYDir < minYDir):
			minYDir = valYDir
		if (valYDir > maxYDir):
			maxYDir = valYDir

		numRemainingVals = numRemainingVals - 1



	# initialize the set the values in the dictionary with the extreme values
	extremeValsDict = {}
	extremeValsDict[retrieveExtremesFromFloFileConsts.MIN_X_MOVE_KEY] = minXDir
	extremeValsDict[retrieveExtremesFromFloFileConsts.MAX_X_MOVE_KEY] = maxXDir
	extremeValsDict[retrieveExtremesFromFloFileConsts.MIN_Y_MOVE_KEY] = minYDir
	extremeValsDict[retrieveExtremesFromFloFileConsts.MAX_Y_MOVE_KEY] = maxYDir

	print extremeValsDict

	# return the dictionary with the 'extreme vals'
	return extremeValsDict


# 'main' function which is run from the command line...
if __name__ == '__main__':

	# make the call to retrieve info from the flo file...
	getExtremesFromFloFile('/home/scott/cudaBeliefPropagation/eval-data-gray/Army/flow10SmoothCap34NoSmoothMultImsEstMoveCap34.flo')
