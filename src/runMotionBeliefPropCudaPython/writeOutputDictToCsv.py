# writeOutputDictToCsv.py
# Scott Grauer-Gray
# October 11, 2010
# Python script for writing the output dictionary results to a csv file...

# needed for 'newline'...
import genCudaConsts

# delimiter used for writing the csv file (it's a comma...)
DELIMITER_CSV_FILE_OUTPUT_RESULTS = ','

# constant defining the starting index in a list
STARTING_INDEX_LIST_OUTPUT_RESULTS = 0

# output the current results as a csv file
# note that all values are converted to strings for output
def writeOutputResultsCsvFile(dictOutputResults, fileNameWriteOutputResults):

	# open the file to write the output results
	filePointerWriteOutputResults = open(fileNameWriteOutputResults, 'w')

	# create an empty 'list-of-lists' to store the values
	listOfListsStoreVals = []

	# go through the headings and write then to the output file separated with the delimiter...
	for currentKey, currentListVal in dictOutputResults.iteritems():

		# write the current heading to the file
		filePointerWriteOutputResults.write(str(currentKey))

		# now add the delimiter to separate the headings
		filePointerWriteOutputResults.write(DELIMITER_CSV_FILE_OUTPUT_RESULTS)

		# append the list of values corresponding to the key
		listOfListsStoreVals.append(currentListVal)

	# now go and write the values row-by-row
	for currentRowNum in range(STARTING_INDEX_LIST_OUTPUT_RESULTS, len(listOfListsStoreVals[0])):

		# now add a newline for the 'next line' which will have a fresh line of 'results'
		filePointerWriteOutputResults.write(genCudaConsts.NEWLINE_CHAR_CUDA_GEN_CONSTS)

		# now go through each element of the row of results and write it in the appropriate space in the csv file
		for currentKeyColNum in range(STARTING_INDEX_LIST_OUTPUT_RESULTS, len(listOfListsStoreVals)):

			# write the value mapped to the appropriate heading
			filePointerWriteOutputResults.write(str((listOfListsStoreVals[currentKeyColNum])[currentRowNum]))

			# now add the delimiter to separate the values
			filePointerWriteOutputResults.write(DELIMITER_CSV_FILE_OUTPUT_RESULTS)

	# close the file being written to...
	filePointerWriteOutputResults.close()
