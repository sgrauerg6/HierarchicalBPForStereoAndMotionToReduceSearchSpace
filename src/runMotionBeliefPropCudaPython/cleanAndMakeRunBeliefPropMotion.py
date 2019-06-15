# cleanAndMakeRunBeliefPropMotion.py
# Scott Grauer-Gray
# August 30, 2010
# Function for clean and making running belief propagation for motion

# needed for the constants to clean and make belief propagation for motion...
import runMotionBeliefPropConsts

# needed to change to the desired directory
import os

# needed to run the command in question
import subprocess

# function to 'clean' belief propagation motion
def cleanMotionBeliefProp(dirMakefile):

	# change the directory to the "desired one"
	os.chdir(dirMakefile)

	# run the "make clean" command
	process = subprocess.Popen(runMotionBeliefPropConsts.MAKE_CLEAN_COMMAND, shell=True)
	process.wait()


# function to 'make' belief propagation motion
def makeMotionBeliefProp(dirMakefile):

	# change the directory to the "desired one"
	os.chdir(dirMakefile)

	# run the "make" command
	process = subprocess.Popen(runMotionBeliefPropConsts.MAKE_COMMAND, shell=True)
	process.wait()
