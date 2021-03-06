/*
Copyright (C) 2010 Scott Grauer-Gray and Chandra Kambhammettu
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

Hierarchical belief propagation for stereo and motion estimation using CUDA

S. Grauer-Gray and C. Kambhammettu

This README describes an implementation of our hierarchical CUDA belief propagation 
for stereo and motion estimation as described in our paper "Hierarchical belief propagation 
to reduce search space using cuda for stereo and motion estimation" which was published at 
IEEE Workshop on Applications of Computer Vision (WACV) 2009 (available at
http://scottgg.net/hiarchBeliefProp.pdf) and containing extensions
for future work (that never ended up being published).

Please cite this work if using any of this code as part of a research paper or other work.

In addition, the code is distributed using the GNU General Public License, so any derivative 
work which is distributed must also contain this license.

Please email comments and bug reports to: sgrauerg@gmail.com.

Usage

Attached is code to run belief propagation for motion estimation on Linux...note that the C++/C/CUDA code is called via a python script that is used to run the program/adjust the parameters

NOTES: 
1.  FOLDER_DOWNLOADED_TO refers to the folder where this application was extracted to on the computer...
2.  The nVidia GPU Computing SDK available from nVidia's website at http://developer.nvidia.com/cuda-downloads must be installed since it contains the "cutil" files used in this program.


Steps for running application:

1.  Set the parameters in the following files:

Makefile:

CUDA_INSTALL_PATH = Path of cuda installation...likely /usr/local/cuda
NVIDIA_GPU_COMPUTING_SDK_DIR= directory of NVIDIA_GPU_Computing_SDK
BELIEF_PROP_ROOT = 'root' directory for application...likely FOLDER_DOWNLOADED_TO/cudaBeliefPropagation

common/common.mk (same as in Makefile...):

CUDA_INSTALL_PATH = Path of cuda installation...likely /usr/local/cuda
NVIDIA_GPU_COMPUTING_SDK_DIR= directory of NVIDIA_GPU_Computing_SDK
BELIEF_PROP_ROOT = 'root' directory for application...likely FOLDER_DOWNLOADED_TO/cudaBeliefPropagation

Also adjust the following in the "rule for compilation of .cu file into object file" to correspond to the compute capability of the target GPU:
-arch=compute_20 -code=sm_20

Examples:

If using GTX 480/580 / Tesla C2050/C2070/C2090 or another GPU with compute capability 2.0, keep rule as
-arch=compute_20 -code=sm_20

If using GTX 280/285 / Tesla C1060 or another GPU with compute capability 1.3, change rule to 
-arch=compute_13 -code=sm_13

Other current compute capabilities are 1.0, 1.1, 1.2, and 2.1, see CUDA programming guide for compute capability of target GPU


src/runMotionBeliefPropCudaPython/genCudaConsts.py:

CUDA_PATH_SYSTEM = path of CUDA on the system...likely /usr/local/cuda
OUTER_FOLDER_CUDA_BELIEF_PROP_APP_MOTION_EST = 'root' directory for application...likely FOLDER_DOWNLOADED_TO/cudaBeliefPropagation

src/runMotionBeliefPropCudaPython/runMotionBeliefPropInputs.py

Adjust input images/parameters as desired...see comments in file for descriptions...




2. Running the program:

After adjusting the parameters as desired, go into the folder 'src/runMotionBeliefPropCudaPython/' and run the following python command:

python mainRunCudaBeliefProp

Initial Motion Results:

By "default" the input image set and parameters are the Grove2 image set (which is part of this package) with the parameters used in our paper "Hierarchical belief propagation to reduce search space using cuda for stereo and motion estimation" noted above, which reproduces the initial motion results for the image set using the call "python mainRunCudaBeliefProp"



Motion Results Using expected motion:

1. Adjust cudaBeliefPropagation.cpp to call 'testRunBpMotionMultImages(argv);' (rather than 'testRunBpMotionWithCommLineArgs(argv);' as called initially...)

2. Use runMotionBeliefPropMultImagesInputs.py to adjust the inputs 

3. Run the command "python mainRunCudaBeliefPropMultImages.py"



Motion Results with refining range:

1. Adjust cudaBeliefPropagation.cpp to call testRunBpMotionGivenExpectedMotion(argv);

2. Use runMotionBeliefPropGivenExpMotionInputs.py to adjust the inputs 

3. Run the command "python mainRunCudaBeliefPropRefineFromFloAndUseNoExpMotionResults.py"



Motion Results with refining output:

1. Adjust cudaBeliefPropagation.cpp to call testRunBpMotionGivenExpectedMotion(argv);

2. Use runMotionBeliefPropGivenExpMotionInputs.py and runMotionBeliefPropRefineFromFloResultsAndUseNoExpMoveFloResultConsts.py to adjust the inputs (to reproduce the results in the paper, use the .flo result of the "Motion Results Using expected motion" section as the FILE_PATH_EXPECTED_MOTION parameter in runMotionBeliefPropGivenExpMotionInputs.py...note that these results involve a race condition so results may not be the same in every run.)

3. Run the command "python mainRunCudaBeliefPropRefineFromFlo.py"



Motion Results using a pyramid hierarchy within the cuboid hierarchy:

1. Adjust cudaBeliefPropagation.cpp to call 'testRunBpMotionWithCommLineArgs(argv);'...same as in initial results

2. In the beliefPropParamsAndStructs.cuh header file, adjust the following line:

#define DEFAULT_PROCESSING_METHOD constHierarch 

to

#define DEFAULT_PROCESSING_METHOD constHierarchPyrWithin 

3. Adjust the parameters in runMotionBeliefPropInputs.py as desired

4. Run the implementation using the following command 
(same as for initial motion):

python mainRunCudaBeliefProp
