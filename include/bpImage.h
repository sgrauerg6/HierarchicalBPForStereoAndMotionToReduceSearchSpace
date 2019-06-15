//bpImage.h
//Scott Grauer-Gray
//June 27, 2009
//Declares the class used for an image used in running belief propagation

#ifndef BP_IMAGE_H
#define BP_IMAGE_H

//needed for the current belief propagation parameters and structs
#include "beliefPropParamsAndStructs.cuh"

//needed for output resulting movement
#include "resultMovement.h"

//needed for io stuff like printf
#include <stdio.h>

//needed for general CUDA functions for allocating the transferring data to/from the device
#include "genCudaFuncts.cuh"

//needed for general math functions
#include <math.h>



class bpImage
{
public:
	//defines the constructor with no input parameters...
	bpImage();

	//copy constructor for class
	bpImage(const bpImage& inBpImage);

	//overloaded equals for class
	bpImage& operator=(const bpImage& inBpImage);

	//defines the constructor that take in the file path and width/height of the image as a pgm
	bpImage(const char* filePathPgmIm, int widthImage, int heightImage);

	//defines the constructor that takes in the pointer to the current pixels, the type of pixels, the location of the pixels (host or device),
	//and the width and height
	bpImage(void* currentImagePix, currentDataType currTypeImagePixInput, dataLocation currentImageDataLocationInput, int widthImageInput, int heightImageInput);

	//virtual destructor function
	virtual ~bpImage(void);

	//function to transfer the image data from the host to the device in whatever type it currently is
	void tranferImageHostToDevice();

	//accessor functions to retrieve the image width/height
	int getImageWidth() { return widthImage; };
	int getImageHeight() { return heightImage; };

	//accessor function to retrieve the image pixels
	currentDataPointer getImagePix() { return bpImagePix; };

	//accessor functions to retrieve the type/location of image pixel data
	currentDataType getPixDataType() { return currTypeImagePix; };
	dataLocation getPixDataLocation() { return currentImageDataLocation; };

	//functions save the current image as a pgm
	void saveBpImageAsPgm(const char* filePathSaveImage);

	//'getters' for each of the attributes...
	currentDataPointer getBpImagePix()  const { return bpImagePix; } ;
	currentDataType getCurrTypeImagePix()  const { return currTypeImagePix; } ;
	dataLocation getCurrentImageDataLocation() const { return currentImageDataLocation; } ;
	int getWidthImage() const { return widthImage; } ;
	int getHeightImage() const { return heightImage; } ;


	void setBpImagePix(currentDataPointer inDataPointer) { bpImagePix = inDataPointer; } ;
	void setCurrTypeImagePix(currentDataType inDataType) { currTypeImagePix = inDataType; } ;
	void setCurrentImageDataLocation(dataLocation inDataLocation) { currentImageDataLocation = inDataLocation; };	
	void setWidthImage(int inWidthImage) { widthImage = inWidthImage; };
	void setHeightImage(int inHeightImage) { heightImage = inHeightImage; };

private:
	//pointer to the image pixels
	currentDataPointer bpImagePix;

	//defines the current type of image pixels
	currentDataType currTypeImagePix;

	//give the current location of the image data
	dataLocation currentImageDataLocation;

	//variables defining the width and height of the image
	int widthImage;
	int heightImage;
};

#endif //BP_IMAGE_H
