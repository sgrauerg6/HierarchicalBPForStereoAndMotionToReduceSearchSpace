//runSmoothImage.cpp
//Scott Grauer-Gray
//June 29, 2009
//Defines the functions for smoothing the image with a Guassian filter of a given sigma

#include "runSmoothImage.h"




//overloaded () operator to perform the smoothing operation with the desired values
bpImage* runSmoothImage::operator ()(bpImage* currentSmoothImageInput)
{
	//currently input image type must be unsigned char
	if (currentSmoothImageInput->getPixDataType() != UNSIGNED_CHAR_DATA)
	{
		printf("Error, input image must be of unsigned char type\n");
		return NULL;
	}

	//currently output image type must be float
	if (runSmoothImage::dataTypeOutputPixels != FLOAT_DATA)
	{
		printf("Error, output image must be of float type\n");
		return NULL;
	}

	//declare the pointer to the image pixels on the device for smoothing
	unsigned char* imageUCharInDevice;

	//check the location of input image and transfer to device if necessary
	switch (currentSmoothImageInput->getPixDataLocation())
	{
		case DATA_ON_HOST:

			//allocate the space on the device for the data and transfer
			allocateArray((void**)&imageUCharInDevice, currentSmoothImageInput->getImageWidth()*currentSmoothImageInput->getImageHeight()*sizeof(unsigned char));
			copyArrayToDevice(imageUCharInDevice, (currentSmoothImageInput->getImagePix()).uCharDataPointer, currentSmoothImageInput->getImageWidth()*currentSmoothImageInput->getImageHeight()*sizeof(unsigned char));
			break;

		case DATA_ON_DEVICE:

			//set the pointer to imageUCharInDevice to the input image
			imageUCharInDevice = (currentSmoothImageInput->getImagePix()).uCharDataPointer;
			break;

	}

	//declare and allocate the space for the output smoothed image on the device (it must be a float)
	float* smoothedImageDevice;
	allocateArray((void**)&smoothedImageDevice,
		currentSmoothImageInput->getImageWidth()*currentSmoothImageInput->getImageHeight()*sizeof(float));

	smoothSingleImageAllDataInDevice(imageUCharInDevice, currentSmoothImageInput->getImageWidth(),
		currentSmoothImageInput->getImageHeight(), sigmaSmoothImage, smoothedImageDevice);


	//declare the pointer to the output image on the host
	float* outputImHost;

	//declare the variables to store the smoothed image data location
	dataLocation outputDataLocation = locationOutputSmoothedImage;
	currentDataPointer outputPixPointer;
	currentDataType outputSmoothedImageDataType = FLOAT_DATA;


	//transfer the output smoothed image to the host if requested
	//assuming that output data must be of float type for now...
	switch (locationOutputSmoothedImage)
	{
		case DATA_ON_HOST:

			//transfer output image to host
			outputImHost = new float[currentSmoothImageInput->getImageWidth()*currentSmoothImageInput->getImageHeight()];
			copyArrayFromDevice(outputImHost, smoothedImageDevice,
				currentSmoothImageInput->getImageWidth()*currentSmoothImageInput->getImageHeight()*sizeof(float));
			
			//free the space on the device if the output is to be on the host
			freeArray(smoothedImageDevice);

			//point resulting output pix of the smoothed image to location on host
			outputPixPointer.floatDataPointer = outputImHost;

			break;
		case DATA_ON_DEVICE:

			//point the output pix pointer to the smoothedImageDevice with the smoothed image on the device as a float array
			outputPixPointer.floatDataPointer = smoothedImageDevice;

			break;
	}

	return (new bpImage(outputPixPointer.floatDataPointer, outputSmoothedImageDataType, outputDataLocation, 
		currentSmoothImageInput->getImageWidth(), currentSmoothImageInput->getImageHeight()));
}

runSmoothImage::~runSmoothImage(void)
{
}
