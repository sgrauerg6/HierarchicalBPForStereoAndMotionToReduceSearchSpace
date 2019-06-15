//runSmoothImageHostFuncts.cuh
//Scott Grauer-Gray
//June 28, 2009
//Declares the host functions used for "smoothing" and image with a given sigma value

#ifndef RUN_SMOOTH_IMAGE_HOST_FUNCTS_CUH
#define RUN_SMOOTH_IMAGE_HOST_FUNCTS_CUH

//needed for the smooth images parameters and structs
#include "smoothImageParamsAndStructs.cuh"


extern "C"
{

/* normalize mask so it integrates to one */
void normalizeFilter(float*& filter, int sizeFilter);

//this function creates a Gaussian filter given a sigma value
float* makeFilter(float sigma, int& sizeFilter);

//function to use the CUDA-image filter to apply a guassian filter to the a single images
//input images have each pixel stored as an unsigned char the device
//output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigmaVal
void smoothSingleImageAllDataInDevice(unsigned char* imageUCharInDevice, int widthImages, int heightImages, float sigmaVal, float* imageFloatSmoothedDevice);

}

#endif //RUN_SMOOTH_IMAGE_HOST_FUNCTS_CUH
