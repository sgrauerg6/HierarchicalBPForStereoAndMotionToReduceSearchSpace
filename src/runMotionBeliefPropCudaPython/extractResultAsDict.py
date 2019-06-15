# extractResultAsDict.py
# Scott Grauer-Gray
# October 11, 2010
# Python script to extract the resulting values as a dictionary...

# needed for constants related to extracting the results...
import extractResultAsDictConsts

# function that takes in the input file which has the output and extracts the results as a dictionary...
def retResultingDict(fileOutputResults):
	
	# initialize a dictionary to store the results
	resultsDict = {}

	# open the output results file
	resultsPointer = open(fileOutputResults, 'r')

	# read the results line-by-line
	for currResultLine in resultsPointer:

		print currResultLine

		# split the line between the key and the value via the separator
		lineSplit = currResultLine.split(extractResultAsDictConsts.KEY_VAL_SEPARATOR)

		# add the result to the dictionary if it contains 2 elements
		numElementsResultLine = len(lineSplit)

		if (numElementsResultLine == 2):

			resultsDict[lineSplit[extractResultAsDictConsts.KEY_INDEX_IN_KEY_VAL_PAIR]] = lineSplit[extractResultAsDictConsts.VAL_INDEX_IN_KEY_VAL_PAIR].strip() # eliminate whitespace from the value...

	# close the results pointer
	resultsPointer.close()

	# return the resultant dictionary...
	return resultsDict


