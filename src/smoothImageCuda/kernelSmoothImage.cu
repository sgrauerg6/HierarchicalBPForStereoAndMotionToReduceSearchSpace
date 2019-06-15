//kernelSmoothImage.cu
//Scott Grauer-Gray
//June 28, 2009
//Defines the kernel functions used to smooth the image

#include "kernelSmoothImage.cuh"

//checks if the current point is within the image bounds
__device__ bool withinImageBoundsSmoothIm(int xVal, int yVal, int width, int height)
{
	return ((xVal >= 0) && (xVal < width) && (yVal >= 0) && (yVal < height));
}

//kernal to convert the unsigned char pixels to float pixels in an image when
//smoothing is not desired but the pixels need to be converted to floats
//the input image is stored as unsigned chars in the texture imagePixelsUnsignedCharToFilterTexture
//output filtered image stored in floatImagePixels
__global__ void convertUnsignedCharImageToFloat(float* floatImagePixels)
{
	//retrieve the indices of the current pixel
	int xVal = blockIdx.x*blockDim.x + threadIdx.x;
	int yVal = blockIdx.y*blockDim.y + threadIdx.y;


	//make sure that (xVal, yVal) is within image bounds
	if (withinImageBoundsSmoothIm(xVal, yVal, widthImageConstFilt, heightImageConstFilt))
	{
		//retrieve the float-value of the unsigned int pixel value at the current location
		float floatPixelVal = 1.0f*(unsigned int)tex1Dfetch(imagePixelsUnsignedCharToFilterTexture, yVal*widthImageConstFilt + xVal);

		floatImagePixels[yVal*widthImageConstFilt + xVal] = floatPixelVal;
	}
}

//kernal to apply a horizontal filter on each pixel of the image in parallel
//input image stored in texture imagePixelsFloatToFilterTexture
//output filtered image stored in filteredImagePixels
__global__ void filterFloatImageAcross(float* filteredImagePixels)
{
	//retrieve the indices of the current pixel
	int xVal = blockIdx.x*blockDim.x + threadIdx.x;
	int yVal = blockIdx.y*blockDim.y + threadIdx.y;

	//make sure that (xVal, yVal) is within image bounds
	if (withinImageBoundsSmoothIm(xVal, yVal, widthImageConstFilt, heightImageConstFilt))
	{
		float filteredPixelVal = imageFilterConst[0]*tex1Dfetch(imagePixelsFloatToFilterTexture, yVal*widthImageConstFilt + xVal) ;

		for (int i = 1; i < sizeFilterConst; i++) {
			filteredPixelVal += imageFilterConst[i] * (tex1Dfetch(imagePixelsFloatToFilterTexture, yVal*widthImageConstFilt + max(xVal-i, 0)) 
				+ tex1Dfetch(imagePixelsFloatToFilterTexture, yVal*widthImageConstFilt + min(xVal+i, widthImageConstFilt-1))); 
		}

		filteredImagePixels[yVal*widthImageConstFilt + xVal] = filteredPixelVal;
	}
}


//kernal to apply a vertical filter on each pixel of the image in parallel
//input image stored in texture imagePixelsFloatToFilterTexture
//output filtered image stored in filteredImagePixels
__global__ void filterFloatImageVertical(float* filteredImagePixels)
{
	//retrieve the indices of the current pixel
	int xVal = blockIdx.x*blockDim.x + threadIdx.x;
	int yVal = blockIdx.y*blockDim.y + threadIdx.y;

	//make sure that (xVal, yVal) is within image bounds
	if (withinImageBoundsSmoothIm(xVal, yVal, widthImageConstFilt, heightImageConstFilt))
	{
		float filteredPixelVal = imageFilterConst[0]*tex1Dfetch(imagePixelsFloatToFilterTexture, yVal*widthImageConstFilt + xVal);

		for (int i = 1; i < sizeFilterConst; i++) {
			filteredPixelVal += imageFilterConst[i] * (tex1Dfetch(imagePixelsFloatToFilterTexture, max(yVal-i, 0)*widthImageConstFilt + xVal) 
				+ tex1Dfetch(imagePixelsFloatToFilterTexture, min(yVal+i, heightImageConstFilt-1)*widthImageConstFilt + xVal)); 
		}

		filteredImagePixels[yVal*widthImageConstFilt + xVal] = filteredPixelVal;
	}
}

//kernal to apply a horizontal filter on each pixel of the image in parallel
//the input image is stored as unsigned char in the texture imagePixelsUnsignedCharToFilterTexture
//the output filtered image is returned as an array of floats
__global__ void filterUnsignedIntImageAcross(float* filteredImagePixels)
{
	//retrieve the indices of the current pixel
	int xVal = blockIdx.x*blockDim.x + threadIdx.x;
	int yVal = blockIdx.y*blockDim.y + threadIdx.y;

	//make sure that (xVal, yVal) is within image bounds
	if (withinImageBoundsSmoothIm(xVal, yVal, widthImageConstFilt, heightImageConstFilt))
	{

		float filteredPixelVal = imageFilterConst[0]*tex1Dfetch(imagePixelsUnsignedCharToFilterTexture, yVal*widthImageConstFilt + xVal) ;


		for (int i = 1; i < sizeFilterConst; i++) {
			filteredPixelVal += imageFilterConst[i] * (tex1Dfetch(imagePixelsUnsignedCharToFilterTexture, yVal*widthImageConstFilt + max(xVal-i, 0)) 
				+ tex1Dfetch(imagePixelsUnsignedCharToFilterTexture, yVal*widthImageConstFilt + min(xVal+i, widthImageConstFilt-1))); 
		}

		filteredImagePixels[yVal*widthImageConstFilt + xVal] = filteredPixelVal;
	}
}


//kernal to apply a vertical filter on each pixel of the image in parallel
//the input image is stored as unsigned ints in the texture imagePixelsUnsignedIntToFilterTexture
//the output filtered image is returned as an array of floats
__global__ void filterUnsignedCharImageVertical(float* filteredImagePixels)
{
	//retrieve the indices of the current pixel
	int xVal = blockIdx.x*blockDim.x + threadIdx.x;
	int yVal = blockIdx.y*blockDim.y + threadIdx.y;

	//make sure that (xVal, yVal) is within image bounds
	if (withinImageBoundsSmoothIm(xVal, yVal, widthImageConstFilt, heightImageConstFilt))
	{
		float filteredPixelVal = imageFilterConst[0]*tex1Dfetch(imagePixelsUnsignedCharToFilterTexture, yVal*widthImageConstFilt + xVal);

		for (int i = 1; i < sizeFilterConst; i++) {
			filteredPixelVal += imageFilterConst[i] * (tex1Dfetch(imagePixelsUnsignedCharToFilterTexture, max(yVal-i, 0)*widthImageConstFilt + xVal) 
				+ tex1Dfetch(imagePixelsUnsignedCharToFilterTexture, min(yVal+i, heightImageConstFilt-1)*widthImageConstFilt + xVal)); 
		}

		filteredImagePixels[yVal*widthImageConstFilt + xVal] = filteredPixelVal;
	}
}
