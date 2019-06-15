//bpImage.cpp
//Scott Grauer-Gray
//June 28, 2009
//Defines the functions used in the bpImage class

#include "bpImage.h"


//defines the constructor with no input parameters...
bpImage::bpImage()
{
}

//copy constructor for class
//just a shallow copy for now...
//TODO: deep copy...
bpImage::bpImage(const bpImage& inBpImage)
{
	setBpImagePix (bpImagePix) ;
	setCurrTypeImagePix ( currTypeImagePix ); 
	setCurrentImageDataLocation (currentImageDataLocation);	
	setWidthImage (widthImage);
	setHeightImage (heightImage);
}

//overloaded equals for class
//just a shallow copy for now...
//TODO: deep copy...
bpImage& bpImage::operator=(const bpImage& inBpImage)
{
	if (this != &inBpImage)
	{
		//now set the value for the out as equal to the inBpImage 
		setBpImagePix(inBpImage.getBpImagePix());
		setCurrTypeImagePix(inBpImage.getCurrTypeImagePix());
		setCurrentImageDataLocation(inBpImage.getCurrentImageDataLocation());	
		setWidthImage(inBpImage.getWidthImage());
		setHeightImage(inBpImage.getHeightImage());
	}

	//return the constructed output bp image...
	return *this;
}

//defines the constructor that take in the file path of the image as a pgm
bpImage::bpImage(const char* filePathPgmIm, int widthImage, int heightImage)
{
	//load the current image pixels from the file

	//loading image as unsigned char data
	bpImagePix.uCharDataPointer = new unsigned char[widthImage*heightImage];

	//current type of image is unsigned char
	currTypeImagePix = UNSIGNED_CHAR_DATA;

	//give the current location of the image data
	currentImageDataLocation = DATA_ON_HOST;

	unsigned int imageWidthUChar = (unsigned int)widthImage;
	unsigned int imageHeightUChar = (unsigned int)heightImage;

	//set the width and height of the image to the current input width and height arguments
	this->widthImage = widthImage;
	this->heightImage = heightImage;

	//use the CUDA utility function to load the image from the given file path as unsigned chars
	cutLoadPGMub( filePathPgmIm, &(bpImagePix.uCharDataPointer), &imageWidthUChar, &imageHeightUChar);
}

//defines the constructor that takes in the pointer to the current pixels, the type of pixels, the location of the pixels (host or device),
//and the width and height
bpImage::bpImage(void* currentImagePix, currentDataType currTypeImagePixInput, dataLocation currentImageDataLocationInput, int widthImageInput, int heightImageInput):
currTypeImagePix(currTypeImagePixInput), currentImageDataLocation(currentImageDataLocationInput), widthImage(widthImageInput), heightImage(heightImageInput)
{
	//check the current type of pixels set bpImagePix based on that
	switch (currTypeImagePix)
	{
	case FLOAT_DATA:
		bpImagePix.floatDataPointer = static_cast<float*>(currentImagePix);
		break;
	case INT_DATA:
		bpImagePix.intDataPointer = static_cast<int*>(currentImagePix);
		break;
	case UNSIGNED_INT_DATA:
		bpImagePix.uIntDataPointer = static_cast<unsigned int*>(currentImagePix);
		break;
	case CHAR_DATA:
		bpImagePix.charDataPointer = static_cast<char*>(currentImagePix);
		break;
	case UNSIGNED_CHAR_DATA:
		bpImagePix.uCharDataPointer = static_cast<unsigned char*>(currentImagePix);
		break;
	}
}


//function to transfer the image data from the host to the device in whatever type it currently is
void bpImage::tranferImageHostToDevice()
{
	//make sure image is on the host
	if (this->currentImageDataLocation != DATA_ON_HOST)
	{
		printf("Error in image transfer host to device, image data not on host\n");
		return;
	}

	//define a temporary currentDataPointer variable to store the input data on the device
	currentDataPointer dataPointDevice;

	//check the image type and transfer the image data host to device of the same image type
	//allocate the space for the transfer the data from the host to the device
	//delete the data on the host after copying to the device
	switch (this->currTypeImagePix)
	{
		case FLOAT_DATA:
			allocateArray((void**)&(dataPointDevice.floatDataPointer), this->widthImage * this->heightImage * sizeof(float));
			copyArrayToDevice(dataPointDevice.floatDataPointer, (this->bpImagePix).floatDataPointer, this->widthImage * this->heightImage * sizeof(float));
			delete [] (this->bpImagePix).floatDataPointer;
			break;
		case INT_DATA:
			allocateArray((void**)&(dataPointDevice.intDataPointer), this->widthImage * this->heightImage * sizeof(int));
			copyArrayToDevice(dataPointDevice.intDataPointer, (this->bpImagePix).intDataPointer, this->widthImage * this->heightImage * sizeof(int));
			delete [] (this->bpImagePix).intDataPointer;
			break;
		case UNSIGNED_INT_DATA:
			allocateArray((void**)&(dataPointDevice.uIntDataPointer), this->widthImage * this->heightImage * sizeof(unsigned int));
			copyArrayToDevice(dataPointDevice.uIntDataPointer, (this->bpImagePix).uIntDataPointer, this->widthImage * this->heightImage * sizeof(unsigned int));
			delete [] (this->bpImagePix).uIntDataPointer;
			break;
		case CHAR_DATA:
			allocateArray((void**)&(dataPointDevice.charDataPointer), this->widthImage * this->heightImage * sizeof(char));
			copyArrayToDevice(dataPointDevice.charDataPointer, (this->bpImagePix).charDataPointer, this->widthImage * this->heightImage * sizeof(char));
			delete [] (this->bpImagePix).charDataPointer;
			break;
		case UNSIGNED_CHAR_DATA:
			allocateArray((void**)&(dataPointDevice.uCharDataPointer), this->widthImage * this->heightImage * sizeof(unsigned char));
			copyArrayToDevice(dataPointDevice.uCharDataPointer, (this->bpImagePix).uCharDataPointer, this->widthImage * this->heightImage * sizeof(unsigned char));
			delete [] (this->bpImagePix).uCharDataPointer;
			break;
	}


	//adjust the currentImageDataLocation to reflect that the data is now on the device
	this->currentImageDataLocation = DATA_ON_DEVICE;
}


//functions save the current image as a pgm
void bpImage::saveBpImageAsPgm(const char* filePathSaveImage)
{
	unsigned char* pixSaveImage = new unsigned char[widthImage*heightImage];

	//holds the current image data on the host
	currentDataPointer currentDataOnHost;

	//need to convert image to unsigned char on host from whatever location and type it currently is
	switch(this->currentImageDataLocation)
	{
		//if data is currently on device, need to transfer to host
		case DATA_ON_DEVICE:
			switch (this->currTypeImagePix)
			{
				case FLOAT_DATA:
					currentDataOnHost.floatDataPointer = new float[widthImage*heightImage];
					copyArrayFromDevice(currentDataOnHost.floatDataPointer, bpImagePix.floatDataPointer, widthImage*heightImage*sizeof(float));
					break;
				case INT_DATA:
					currentDataOnHost.intDataPointer = new int[widthImage*heightImage];
					copyArrayFromDevice(currentDataOnHost.intDataPointer, bpImagePix.intDataPointer, widthImage*heightImage*sizeof(int));
					break;
				case UNSIGNED_INT_DATA:
					currentDataOnHost.uIntDataPointer = new unsigned int[widthImage*heightImage];
					copyArrayFromDevice(currentDataOnHost.uIntDataPointer, bpImagePix.uIntDataPointer, widthImage*heightImage*sizeof(unsigned int));
					break;
				case CHAR_DATA:
					currentDataOnHost.charDataPointer = new char[widthImage*heightImage];
					copyArrayFromDevice(currentDataOnHost.charDataPointer, bpImagePix.charDataPointer, widthImage*heightImage*sizeof(char));
					break;
				case UNSIGNED_CHAR_DATA:
					currentDataOnHost.uCharDataPointer = new unsigned char[widthImage*heightImage];
					copyArrayFromDevice(currentDataOnHost.uCharDataPointer, bpImagePix.uCharDataPointer, widthImage*heightImage*sizeof(unsigned char));
					break;
			}
			break;
		//if data on host, simply set current bpImage data to currentDataOnHost
		case DATA_ON_HOST:
			currentDataOnHost = this->bpImagePix;
	}

	//now convert each pixel to an unsigned char and store in pixSaveImage
	for (int pixNum = 0; pixNum < (widthImage*heightImage); pixNum++)
	{

		switch (this->currTypeImagePix)
		{
			case FLOAT_DATA:
				pixSaveImage[pixNum] = static_cast<unsigned char>(static_cast<unsigned int>
					(floor(currentDataOnHost.floatDataPointer[pixNum] + 0.5f)));
				break;
			case INT_DATA:
				pixSaveImage[pixNum] = static_cast<unsigned char>(static_cast<unsigned int>(currentDataOnHost.intDataPointer[pixNum]));
				break;
			case UNSIGNED_INT_DATA:
				pixSaveImage[pixNum] = static_cast<unsigned char>(currentDataOnHost.uIntDataPointer[pixNum]);
				break;
			case CHAR_DATA:
				pixSaveImage[pixNum] = static_cast<unsigned char>(currentDataOnHost.charDataPointer[pixNum]);
				break;
			case UNSIGNED_CHAR_DATA:
				pixSaveImage[pixNum] = currentDataOnHost.uCharDataPointer[pixNum];
				break;
		}
	}

	//use the CUDA utility function to save the image to the desired file path
	cutSavePGMub(filePathSaveImage, pixSaveImage, static_cast<unsigned int>(widthImage), static_cast<unsigned int>(heightImage));

	delete [] pixSaveImage;
}


bpImage::~bpImage(void)
{
	//delete the current image pixels of whatever type and wherever it is...
	switch (this->currentImageDataLocation)
	{
		case DATA_ON_DEVICE:
			switch (this->currTypeImagePix)
			{
				case FLOAT_DATA:
					freeArray(this->bpImagePix.floatDataPointer);
					break;
				case INT_DATA:
					freeArray(this->bpImagePix.intDataPointer);
					break;
				case UNSIGNED_INT_DATA:
					freeArray(this->bpImagePix.uIntDataPointer);
					break;
				case CHAR_DATA:
					freeArray(this->bpImagePix.charDataPointer);
					break;
				case UNSIGNED_CHAR_DATA:
					freeArray(this->bpImagePix.uCharDataPointer);
					break;
			}
			break;
		case DATA_ON_HOST:
			switch (this->currTypeImagePix)
			{
				case FLOAT_DATA:
					delete [] (this->bpImagePix.floatDataPointer);
					break;
				case INT_DATA:
					delete [] (this->bpImagePix.intDataPointer);
					break;
				case UNSIGNED_INT_DATA:
					delete [] (this->bpImagePix.uIntDataPointer);
					break;
				case CHAR_DATA:
					delete [] (this->bpImagePix.charDataPointer);
					break;
				case UNSIGNED_CHAR_DATA:
					delete [] (this->bpImagePix.uCharDataPointer);
					break;
			}
			break;
	}
}
