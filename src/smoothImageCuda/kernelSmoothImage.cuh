//kernelSmoothImage.cuh
//Scott Grauer-Gray
//June 28, 2009
//Declares the kernel functions used to smooth the image

#ifndef KERNEL_SMOOTH_IMAGE_CUH
#define KERNEL_SMOOTH_IMAGE_CUH

//needed for the general smooth image parameters and structs
#include "smoothImageParamsAndStructs.cuh"

//needed for CUDA utility functions
//#include <cutil.h>


//the filter and the size of the filter is held in constant memory as it is accessed by all threads at the same time,
//making it as fast as a register if in cache
__device__ __constant__ float imageFilterConst[MAX_SIZE_FILTER];
__device__ __constant__ int sizeFilterConst;
__device__ __constant__ int widthImageConstFilt;
__device__ __constant__ int heightImageConstFilt;

//declare the textures that the images to filter are potentially bound to
texture<unsigned char, 1, cudaReadModeElementType> imagePixelsUnsignedCharToFilterTexture;
texture<float, 1, cudaReadModeElementType> imagePixelsFloatToFilterTexture;

//checks if the current point is within the image bounds
__device__ bool withinImageBoundsSmoothIm(int xVal, int yVal, int width, int height);

//kernal to convert the unsigned char pixels to float pixels in an image when
//smoothing is not desired but the pixels need to be converted to floats
//the input image is stored as unsigned chars in the texture imagePixelsUnsignedCharToFilterTexture
//output filtered image stored in floatImagePixels
__global__ void convertUnsignedCharImageToFloat(float* floatImagePixels);

//kernal to apply a horizontal filter on each pixel of the image in parallel
//input image stored in texture imagePixelsFloatToFilterTexture
//output filtered image stored in filteredImagePixels
__global__ void filterFloatImageAcross(float* filteredImagePixels);

//kernal to apply a vertical filter on each pixel of the image in parallel
//input image stored in texture imagePixelsFloatToFilterTexture
//output filtered image stored in filteredImagePixels
__global__ void filterFloatImageVertical(float* filteredImagePixels);

//kernal to apply a horizontal filter on each pixel of the image in parallel
//the input image is stored as unsigned char in the texture imagePixelsUnsignedCharToFilterTexture
//the output filtered image is returned as an array of floats
__global__ void filterUnsignedIntImageAcross(float* filteredImagePixels);

//kernal to apply a vertical filter on each pixel of the image in parallel
//the input image is stored as unsigned ints in the texture imagePixelsUnsignedIntToFilterTexture
//the output filtered image is returned as an array of floats
__global__ void filterUnsignedCharImageVertical(float* filteredImagePixels);

#endif //KERNEL_SMOOTH_IMAGE_CUH
