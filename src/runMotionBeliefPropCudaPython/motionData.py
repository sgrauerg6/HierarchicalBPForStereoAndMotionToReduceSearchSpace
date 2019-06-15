# motionData.py
# Scott Grauer-Gray
# November 30, 2010
# Python class for the 'motion data'

# needed for packing/unpacking
import struct

class MotionData:

	def __init__(self):

		# initialize the list with the x and y motions...
		self.xMotions = []
		self.yMotions = []
		self.widthData = -999
		self.heightData = -999


	def readMotionFromFloFile(self, filePathReadFloFile):

		# initialize the list with the x and y motions...
		self.xMotions = []
		self.yMotions = []

		# open the flo file...
		floFilePointer = open(filePathReadFloFile, 'rb')

		# extract the first four bytes, which should be 'PIEH' in ASCII
		firstFourBytes = floFilePointer.read(4)

		print str(firstFourBytes)

		# extract the next four bytes, which is the width as an integer
		#widthFloBytes = (floFilePointer.read(4))
		widthFloBytes = struct.unpack('i', floFilePointer.read(4))
		self.widthData = widthFloBytes[0]
		print "width: " + str(widthFloBytes[0])


		# extract the next four bytes, which is the height as an integer
		heightFloBytes = struct.unpack('i', floFilePointer.read(4))
		self.heightData = heightFloBytes[0]
		print "height: " + str(heightFloBytes[0])

		# retrieve the number of bytes in the rest of the file
		numRemainingVals = widthFloBytes[0] * heightFloBytes[0]

		# retrieve the 'first' value in the x/y direction and make it the 'extreme'
		firstValXDir = (struct.unpack('f', floFilePointer.read(4)))[0]
		firstValYDir = (struct.unpack('f', floFilePointer.read(4)))[0]

		self.xMotions.append(firstValXDir)
		self.yMotions.append(firstValYDir)

		numRemainingVals = numRemainingVals - 1
	
		while (numRemainingVals > 0):
	
			valXDir = struct.unpack('f', floFilePointer.read(4))[0]
			valYDir = struct.unpack('f', floFilePointer.read(4))[0]

			self.xMotions.append(valXDir)
			self.yMotions.append(valYDir)
		
			numRemainingVals = numRemainingVals - 1


	def saveMotionToFloFile(self, filePathSaveFloFile):

		# open the flo file for writing...
		floFilePointer = open(filePathSaveFloFile, 'wb')

		data = struct.pack('f', float(202021.25))
		floFilePointer.write(data)

		print "WIDTH_SAVE: " + str(self.widthData)
		print "HEIGHT_SAVE: " + str(self.heightData)

		data = struct.pack('i', int(self.widthData))
		floFilePointer.write(data)
		data = struct.pack('i', int(self.heightData))
		floFilePointer.write(data)

		for currMotionIndex in range(0, len(self.xMotions)):

			data = struct.pack('f', float(self.xMotions[currMotionIndex]))
			floFilePointer.write(data)
			data = struct.pack('f', float(self.yMotions[currMotionIndex]))
			floFilePointer.write(data)

		floFilePointer.close()


	def mergeMotionData(self, secondMotionData):

		for currMotionIndex in range(0, len(self.xMotions)):

			currXMotion = self.xMotions[currMotionIndex] 
			secondXMotion = secondMotionData.xMotions[currMotionIndex] 
			divVal = 0.0
			if (abs(currXMotion) < 2):
				divVal = 2.0
			else:
				divVal = currXMotion
			

			diffXMotion = abs(currXMotion - secondXMotion) / abs(divVal)

			if (diffXMotion < 0.25):
				self.xMotions[currMotionIndex] = float((currXMotion + secondXMotion) / 2.0)
			else:
				self.xMotions[currMotionIndex] = float(-999.0)
			

			currYMotion = self.yMotions[currMotionIndex] 
			secondYMotion = secondMotionData.yMotions[currMotionIndex]

			if (abs(currYMotion) < 2):
				divVal = 2.0
			else:
				divVal = currYMotion

			diffYMotion = abs(currYMotion - secondYMotion) / abs(divVal)

			if (diffYMotion < 0.25):
				self.yMotions[currMotionIndex] = float((currYMotion + secondYMotion) / 2.0)
			else:
				self.yMotions[currMotionIndex] = float(-999.0)

