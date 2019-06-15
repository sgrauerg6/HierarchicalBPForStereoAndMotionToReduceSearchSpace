//runSmoothImageHostFuncts.cu
//Scott Grauer-Gray
//June 28, 2009
//Defines the host functions used for smoothing the current image

#include "runSmoothImageHostFuncts.cuh"

//needed for the kernel functions used to smooth the image on the device
#include "kernelSmoothImage.cu"


extern "C"
{

/* normalize mask so it integrates to one */
void normalizeFilter(float*& filter, int sizeFilter) 
{
	float sum = 0;
	for (int i = 1; i < sizeFilter; i++) 
	{
		sum += fabs(filter[i]);
	}
	sum = 2*sum + fabs(filter[0]);
	for (int i = 0; i < sizeFilter; i++) 
	{
		filter[i] /= sum;
	}
}

//this function creates a Gaussian filter given a sigma value
float* makeFilter(float sigma, int& sizeFilter)
{
	sigma = max(sigma, 0.01f);
	sizeFilter = (int)ceil(sigma * WIDTH_SIGMA) + 1;
	float* mask = new float[sizeFilter];
	for (int i = 0; i < sizeFilter; i++) 
	{
		mask[i] = exp(-0.5f*((i/sigma) * (i/sigma)));
	}
	normalizeFilter(mask, sizeFilter);

	return mask;
}

//function to use the CUDA-image filter to apply a guassian filter to the a single images
//input images have each pixel stored as an unsigned char the device
//output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigmaVal
void smoothSingleImageAllDataInDevice(unsigned char* imageUCharInDevice, int widthImages, int heightImages, float sigmaVal, float* imageFloatSmoothedDevice)
{
	// setup execution parameters
	dim3 threads(BLOCK_SIZE_WIDTH_SMOOTH_IMAGE, BLOCK_SIZE_HEIGHT_SMOOTH_IMAGE);
	dim3 grid((unsigned int)(ceil((float)widthImages / (float)threads.x)), (unsigned int)(ceil((float)heightImages / (float)threads.y)));

	//copy the width and height of the images to the constant memory of the device
	cudaMemcpyToSymbol(widthImageConstFilt, &widthImages, sizeof(int));
	cudaMemcpyToSymbol(heightImageConstFilt, &heightImages, sizeof(int));

	//if sigmaVal < MIN_SIGMA_VAL_SMOOTH, then don't smooth image...just convert the input image
	//of unsigned ints to an output image of float values
	if (sigmaVal < MIN_SIGMA_VAL_SMOOTH_IMAGES)
	{
		//bind the first input image on the device to an unsigned int texture
		cudaBindTexture(0, imagePixelsUnsignedCharToFilterTexture, imageUCharInDevice, widthImages*heightImages*sizeof(unsigned char));

		//call kernal to convert input unsigned char pixels to output float pixels on the device
		convertUnsignedCharImageToFloat <<< grid, threads >>>(imageFloatSmoothedDevice);

		cudaThreadSynchronize();

		//unbind the texture the unsigned int textured that the input images were bound to
		cudaUnbindTexture( imagePixelsUnsignedCharToFilterTexture);
	}

	//otherwise apply a Guassian filter to the images
	else
	{

		//sizeFilter set in makeFilter based on sigmaVal
		int sizeFilter;
		float* filter = makeFilter(sigmaVal, sizeFilter);

		//copy the image filter and the size of the filter to constant memory on the GPU
		cudaMemcpyToSymbol(imageFilterConst, filter, sizeFilter*sizeof(float));
		cudaMemcpyToSymbol(sizeFilterConst, &sizeFilter, sizeof(int));

		float* intermediateImageDevice; 

		//allocate store for intermediate image where images is smoothed in one direction
		cudaMalloc((void**) &intermediateImageDevice, (widthImages*heightImages*sizeof(float)));

		//first smooth the image 1, so copy image 1 to GPU memory

		//bind the input image on the device to an unsigned int texture
		cudaBindTexture(0, imagePixelsUnsignedCharToFilterTexture, imageUCharInDevice, widthImages*heightImages*sizeof(unsigned char));

		//first filter the image horizontally, then vertically; the result is applying a 2D gaussian filter with the given sigma value to the image
		filterUnsignedIntImageAcross <<< grid, threads >>> (intermediateImageDevice);

		cudaThreadSynchronize();

		//bind the "float-valued" intermediate image on the device to a float texture
		cudaBindTexture(0, imagePixelsFloatToFilterTexture, intermediateImageDevice, widthImages*heightImages*sizeof(float));

		//now use the vertical filter to complete the smoothing of image 1 on the device
		filterFloatImageVertical <<< grid, threads >>> (imageFloatSmoothedDevice);

		cudaThreadSynchronize();

		//unbind the texture the unsigned int and float textures used for the smoothing
		cudaUnbindTexture( imagePixelsUnsignedCharToFilterTexture);
		cudaUnbindTexture( imagePixelsFloatToFilterTexture);

		//free the device memory used to store the images
		cudaFree(intermediateImageDevice);
	}
}
}
