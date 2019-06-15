//runSmoothImage.h
//Scott Grauer-Gray
//June 28, 2009
//Declares the class used to define the smooth image function

#ifndef RUN_SMOOTH_IMAGE_H
#define RUN_SMOOTH_IMAGE_H

//needed for the smooth image parameters
#include "smoothImageParamsAndStructs.cuh"

//needed for the CUDA host and device function to smooth the image using the device
#include "runSmoothImageHostFuncts.cuh"

//needed for general CUDA functions for allocating the transferring data to/from the device
#include "genCudaFuncts.cuh"

//needed for the input and output for the smooth image
#include "bpImage.h"

class runSmoothImage
{
public:
	runSmoothImage(void):
	sigmaSmoothImage(DEFAULT_SIGMA_SMOOTH_IMAGE),
	locationOutputSmoothedImage(DEFAULT_LOCATION_OUTPUT_SMOOTHED_IMAGE), 
	dataTypeOutputPixels(DEFAULT_DATA_TYPE_OUTPUT_PIXELS)
	{};

	runSmoothImage(float sigmaSmoothImageInput, dataLocation locationOutput, currentDataType dataTypeOutput)
		: sigmaSmoothImage(sigmaSmoothImageInput), locationOutputSmoothedImage(locationOutput), dataTypeOutputPixels(dataTypeOutput) {};

	virtual ~runSmoothImage(void);
	//overloaded () operator to perform the smoothing operation with the desired values
	bpImage* operator()(bpImage* currentSmoothImageInput);

private:
	//define the sigma value for image smoothing
	float sigmaSmoothImage;

	//define the location of the output data
	dataLocation locationOutputSmoothedImage;
	
	//define the type of the output pixels
	currentDataType dataTypeOutputPixels;
};

#endif //RUN_SMOOTH_IMAGE_H
