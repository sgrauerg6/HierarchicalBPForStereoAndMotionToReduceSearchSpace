//runBeliefPropHostFuncts.cu
//Scott Grauer-Gray
//June 25, 2009
//Defines the host functions used for running belief propagation

#include "runBeliefPropHostFuncts.cuh"

#include "kernelRunBeliefProp.cu"


extern "C"
{

//set the current BP settings in the host in constant memory on the device
void setCurrBeliefPropParamsInConstMem(currBeliefPropParams& currentBeliefPropParams)
{
	cudaMemcpyToSymbol(currentBeliefPropParamsConstMem, &currentBeliefPropParams, sizeof(currBeliefPropParams));
}

//run the kernel function to round the set of values on the device
void roundDeviceVals(float* inputDeviceVals, float* outputDeviceVals, int widthVals, int heightVals)
{
	//set the execution parameters for running BP at the current level
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid((unsigned int)ceil((float)widthVals / (float)threads.x), (unsigned int)ceil((float)heightVals / (float)threads.y));

	//run the kernel function to round the values in the device
	roundDeviceValsKernel <<< grid, threads >>> (inputDeviceVals, widthVals, heightVals, outputDeviceVals);

	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

//run the given number of iterations of BP at the current level using the given message values in global device memory...no textures...
void runBPAtCurrentLevelNoTextures(
		checkerboardMessagesDeviceStruct messageValsCheckerboard1, checkerboardMessagesDeviceStruct messageValsCheckerboard2,
		float* dataCostDeviceCheckerboard1, float* dataCostDeviceCheckerboard2,
		currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1, currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
		int numIterationsAtLevel, int widthLevelActualIntegerSize, int heightLevelActualIntegerSize, 
		size_t numBytesDataAndMessageSetInCheckerboardAtLevel, size_t paramsOffsetLevel)
{
	//retrieve the width and height of the checkerboard
	int widthOfCheckerboard = widthLevelActualIntegerSize / 2;
	int heightOfCheckerboard = heightLevelActualIntegerSize;

	//set the execution parameters for running BP at the current level
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid((unsigned int)ceil((float)widthOfCheckerboard / (float)threads.x), (unsigned int)ceil((float)heightOfCheckerboard / (float)threads.y));

	printf("NumLevelIterations: %d\n", numIterationsAtLevel);

	//at each level, run BP for numIterations, alternating between updating the messages between the two "checkerboards"
	for (int iterationNum = 0; iterationNum < numIterationsAtLevel; iterationNum++)
	{
		if ((iterationNum % 2) == 0)
		{
		
			//run belief propagation updating the message values in 'part 2' of the checkerboard
			runBPIterationUsingCheckerboardUpdatesNoTextures <<< grid, threads >>> (messageValsCheckerboard2, messageValsCheckerboard1, widthLevelActualIntegerSize, heightLevelActualIntegerSize, CHECKERBOARD_PART_2_ENUM, &(paramsCurrentLevelDeviceCheckerboard2[paramsOffsetLevel]), &(paramsCurrentLevelDeviceCheckerboard1[paramsOffsetLevel]), dataCostDeviceCheckerboard2);
			
			CUDA_SAFE_CALL(cudaThreadSynchronize());


		}
		else
		{
	
			//run belief propagation updating the message values in 'part 1' of the checkerboard
			runBPIterationUsingCheckerboardUpdatesNoTextures <<< grid, threads >>> (messageValsCheckerboard1, messageValsCheckerboard2, widthLevelActualIntegerSize, heightLevelActualIntegerSize, CHECKERBOARD_PART_1_ENUM, &(paramsCurrentLevelDeviceCheckerboard1[paramsOffsetLevel]), &(paramsCurrentLevelDeviceCheckerboard2[paramsOffsetLevel]), dataCostDeviceCheckerboard1);
			
			CUDA_SAFE_CALL(cudaThreadSynchronize());


		}
	}

}

//run the given number of iterations of BP at the current level using the given message values in global device memory
void runBPAtCurrentLevel(
		checkerboardMessagesDeviceStruct messageValsCheckerboard1, checkerboardMessagesDeviceStruct messageValsCheckerboard2,
		float* dataCostDeviceCheckerboard1, float* dataCostDeviceCheckerboard2,
		currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1, currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
		int numIterationsAtLevel, int widthLevelActualIntegerSize, int heightLevelActualIntegerSize, 
		size_t numBytesDataAndMessageSetInCheckerboardAtLevel, size_t paramsOffsetLevel)
{
	//retrieve the width and height of the checkerboard
	int widthOfCheckerboard = widthLevelActualIntegerSize / 2;
	int heightOfCheckerboard = heightLevelActualIntegerSize;

	//set the execution parameters for running BP at the current level
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid((unsigned int)ceil((float)widthOfCheckerboard / (float)threads.x), (unsigned int)ceil((float)heightOfCheckerboard / (float)threads.y));

	//declare the retrieve the number total number of pixels in each checkerboard at the current level
	int numPixCheckerboardAtLevel = widthOfCheckerboard * heightOfCheckerboard;

	//needed for the `offset' for the texture
	size_t dataTexOffset;

	//needed for the `offset' for the parameters texture...
	size_t paramsTexOffset;

	//at each level, run BP for numIterations, alternating between updating the messages between the two "checkerboards"
	for (int iterationNum = 0; iterationNum < numIterationsAtLevel; iterationNum++)
	{
		if ((iterationNum % 2) == 0)
		{
			//if iteration number is even, update part 2 of checkerboard so neighbors are from part 1
			cudaBindTexture(0, messageUTexCurrReadCheckerboard, messageValsCheckerboard1.messageUDevice,
				numBytesDataAndMessageSetInCheckerboardAtLevel);
			cudaBindTexture(0, messageDTexCurrReadCheckerboard, messageValsCheckerboard1.messageDDevice,
				numBytesDataAndMessageSetInCheckerboardAtLevel);
			cudaBindTexture(0, messageLTexCurrReadCheckerboard, messageValsCheckerboard1.messageLDevice,
				numBytesDataAndMessageSetInCheckerboardAtLevel);
			cudaBindTexture(0, messageRTexCurrReadCheckerboard, messageValsCheckerboard1.messageRDevice,
				numBytesDataAndMessageSetInCheckerboardAtLevel);

			//bind each checkerboard set of movement parameters to the appropriate texture
			//assume that sizeof(paramsCurrentPixelAtLevel) is same as size of float2 since that seems necessary for the texture...
			//updating checkerboard part 2 so bind current params texture to part 2 of checkerboard and "neighbor" params to part 1
			cudaBindTexture(&paramsTexOffset, currentPixParamsTexCurrentCheckerboard, &(paramsCurrentLevelDeviceCheckerboard2[paramsOffsetLevel]), numPixCheckerboardAtLevel*sizeof(currentStartMoveParamsPixelAtLevel));
			cudaBindTexture(&paramsTexOffset, currentPixParamsTexNeighCheckerboard, &(paramsCurrentLevelDeviceCheckerboard1[paramsOffsetLevel]), numPixCheckerboardAtLevel*sizeof(currentStartMoveParamsPixelAtLevel));


			//updating checkerboard part 2 so bind current data texture to part 2 of checkerboard
			//cudaBindTexture(0, dataCostsCurrCheckerboard, dataCostDeviceCheckerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel);
			cudaBindTexture(&dataTexOffset, dataCostsCurrCheckerboard, dataCostDeviceCheckerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel);

			runBPIterationUsingCheckerboardUpdates <<< grid, threads >>> (messageValsCheckerboard2, widthLevelActualIntegerSize, heightLevelActualIntegerSize, CHECKERBOARD_PART_2_ENUM, dataTexOffset, paramsTexOffset);
			
			CUDA_SAFE_CALL(cudaThreadSynchronize());

			//now unbind textures bound to movement parameters, messages, and data costs
			cudaUnbindTexture(currentPixParamsTexCurrentCheckerboard);
			cudaUnbindTexture(currentPixParamsTexNeighCheckerboard);

			cudaUnbindTexture(messageUTexCurrReadCheckerboard);
			cudaUnbindTexture(messageDTexCurrReadCheckerboard);
			cudaUnbindTexture(messageLTexCurrReadCheckerboard);
			cudaUnbindTexture(messageRTexCurrReadCheckerboard);

			cudaUnbindTexture(dataCostsCurrCheckerboard);
		}
		else
		{
			//if iteration number is even, update part 1 of checkerboard so neighbors are from part 2
			cudaBindTexture(0, messageUTexCurrReadCheckerboard, messageValsCheckerboard2.messageUDevice,
				numBytesDataAndMessageSetInCheckerboardAtLevel);
			cudaBindTexture(0, messageDTexCurrReadCheckerboard, messageValsCheckerboard2.messageDDevice,
				numBytesDataAndMessageSetInCheckerboardAtLevel);
			cudaBindTexture(0, messageLTexCurrReadCheckerboard, messageValsCheckerboard2.messageLDevice,
				numBytesDataAndMessageSetInCheckerboardAtLevel);
			cudaBindTexture(0, messageRTexCurrReadCheckerboard, messageValsCheckerboard2.messageRDevice,
				numBytesDataAndMessageSetInCheckerboardAtLevel);

			//bind each checkerboard set of movement parameters to the appropriate texture
			//assume that sizeof(paramsCurrentPixelAtLevel) is same as size of float2 since that seems necessary for the texture...
			//updating checkerboard part 1 so bind current params texture to part 1 of checkerboard and "neighbor" params to part 2
			cudaBindTexture(&paramsTexOffset, currentPixParamsTexCurrentCheckerboard, paramsCurrentLevelDeviceCheckerboard1, numPixCheckerboardAtLevel*sizeof(currentStartMoveParamsPixelAtLevel));
			cudaBindTexture(&paramsTexOffset, currentPixParamsTexNeighCheckerboard, paramsCurrentLevelDeviceCheckerboard2, numPixCheckerboardAtLevel*sizeof(currentStartMoveParamsPixelAtLevel));


			//updating checkerboard part 1 so bind current data texture to part 1 of checkerboard
			//cudaBindTexture(0, dataCostsCurrCheckerboard, dataCostDeviceCheckerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel);
			cudaBindTexture(&dataTexOffset, dataCostsCurrCheckerboard, dataCostDeviceCheckerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel);

			runBPIterationUsingCheckerboardUpdates <<< grid, threads >>> (messageValsCheckerboard1, widthLevelActualIntegerSize, heightLevelActualIntegerSize, CHECKERBOARD_PART_1_ENUM, dataTexOffset, paramsTexOffset);

			CUDA_SAFE_CALL(cudaThreadSynchronize());

			//now unbind textures bound to movement parameters, messages, and data costs
			cudaUnbindTexture(currentPixParamsTexCurrentCheckerboard);
			cudaUnbindTexture(currentPixParamsTexNeighCheckerboard);

			cudaUnbindTexture(messageUTexCurrReadCheckerboard);
			cudaUnbindTexture(messageDTexCurrReadCheckerboard);
			cudaUnbindTexture(messageLTexCurrReadCheckerboard);
			cudaUnbindTexture(messageRTexCurrReadCheckerboard);

			cudaUnbindTexture(dataCostsCurrCheckerboard);
		}
	}

	//now unbind textures bound to movement parameters, messages, and data costs
	cudaUnbindTexture(currentPixParamsTexCurrentCheckerboard);
	cudaUnbindTexture(currentPixParamsTexNeighCheckerboard);

	cudaUnbindTexture(messageUTexCurrReadCheckerboard);
	cudaUnbindTexture(messageDTexCurrReadCheckerboard);
	cudaUnbindTexture(messageLTexCurrReadCheckerboard);
	cudaUnbindTexture(messageRTexCurrReadCheckerboard);

	cudaUnbindTexture(dataCostsCurrCheckerboard);
}

//function to adjust the parameters based on the estimated movement...


//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
//in the next level down
//need two different "sets" of message values to avoid read-write conflicts
//this step requires adjustments from the estimated value at one level of the hierarchy to the next level down depending on the
//estimated value at each pixel
void copyMessageValuesToNextLevelDown(
	checkerboardMessagesDeviceStruct messageDeviceCheckerboard1CopyFrom, 
	checkerboardMessagesDeviceStruct messageDeviceCheckerboard2CopyFrom, 
	currentStartMoveParamsPixelAtLevel* paramsPrevLevelDeviceCheckerboard1, currentStartMoveParamsPixelAtLevel* paramsPrevLevelDeviceCheckerboard2,
	currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboard1, currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboard2,
	float* estimatedMovementXCheckerboard1Device, float* estimatedMovementYCheckerboard1Device,
	float* estimatedMovementXCheckerboard2Device, float* estimatedMovementYCheckerboard2Device,
	checkerboardMessagesDeviceStruct messageDeviceCheckerboard1CopyTo,
	checkerboardMessagesDeviceStruct messageDeviceCheckerboard2CopyTo,
	int widthLevelActualIntegerSize, int heightLevelActualIntegerSize,
	int widthNextLevelActualIntegerSize, int heightNextLevelActualIntegerSize,
	size_t numBytesDataAndMessageSetInCheckerboardAtLevel)
{
	//retrieve the width and height of the checkerboard
	int widthOfCheckerboard = widthLevelActualIntegerSize / 2;
	int heightOfCheckerboard = heightLevelActualIntegerSize;

	//set the execution parameters for copying the message values to the next level
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid((unsigned int)ceil((float)widthOfCheckerboard / (float)threads.x), (unsigned int)ceil((float)heightOfCheckerboard / (float)threads.y));

	//declare and set the number of pixels in the checkerboard at the current level
	int numPixInCheckerboardCurrentLevel = widthOfCheckerboard * heightOfCheckerboard;

	//bind the linear memory storing to computed message values to copy from to a texture for the "first" checkerboard
	cudaBindTexture(0, messageUTexCurrReadCheckerboard, messageDeviceCheckerboard1CopyFrom.messageUDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageDTexCurrReadCheckerboard, messageDeviceCheckerboard1CopyFrom.messageDDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageLTexCurrReadCheckerboard, messageDeviceCheckerboard1CopyFrom.messageLDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageRTexCurrReadCheckerboard, messageDeviceCheckerboard1CopyFrom.messageRDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);

	//bind the previous level parameters to the appropriate texture in the device
	cudaBindTexture(0, currentPixParamsTexCurrentCheckerboard, paramsPrevLevelDeviceCheckerboard1, numPixInCheckerboardCurrentLevel*sizeof(currentStartMoveParamsPixelAtLevel));

	//bind the current motion estimates to the appropriate texture in the device
	cudaBindTexture(0, estimatedXMovementTexDeviceCurrentCheckerboard, estimatedMovementXCheckerboard1Device, numPixInCheckerboardCurrentLevel*sizeof(float));
	cudaBindTexture(0, estimatedYMovementTexDeviceCurrentCheckerboard, estimatedMovementYCheckerboard1Device, numPixInCheckerboardCurrentLevel*sizeof(float));


	//run the kernel to copy the message values from the previous level to the next level for part 2 of the checkerboard
	//and adjust the current movement parameters based on the current estimated movement
	copyPrevLevelToNextLevelBPCheckerboard <<< grid, threads >>> 
															(messageDeviceCheckerboard1CopyTo, messageDeviceCheckerboard2CopyTo,
															widthLevelActualIntegerSize, heightLevelActualIntegerSize,
															widthNextLevelActualIntegerSize, heightNextLevelActualIntegerSize,
															paramsNextLevelDeviceCheckerboard1, paramsNextLevelDeviceCheckerboard2,
															CHECKERBOARD_PART_1_ENUM);

	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	//unbind the texture that was bound to the current movement parameters
	cudaUnbindTexture(currentPixParamsTexCurrentCheckerboard);

	//unbind the texture that was bound to the estimated motions for checkerboard 1
	cudaUnbindTexture(estimatedXMovementTexDeviceCurrentCheckerboard);
	cudaUnbindTexture(estimatedYMovementTexDeviceCurrentCheckerboard);

	//unbind the texture that was bound to the previous message values
	cudaUnbindTexture(messageUTexCurrReadCheckerboard);
	cudaUnbindTexture(messageDTexCurrReadCheckerboard);
	cudaUnbindTexture(messageLTexCurrReadCheckerboard);
	cudaUnbindTexture(messageRTexCurrReadCheckerboard);

	//bind the current movement parameters for checkerboard 2 since that is now being updated
	cudaBindTexture(0, currentPixParamsTexCurrentCheckerboard, paramsPrevLevelDeviceCheckerboard2, numPixInCheckerboardCurrentLevel*sizeof(currentStartMoveParamsPixelAtLevel));

	//bind the estimated motions for checkerboard 2 since that is now being updated
	cudaBindTexture(0, estimatedXMovementTexDeviceCurrentCheckerboard, estimatedMovementXCheckerboard2Device, numPixInCheckerboardCurrentLevel*sizeof(float));
	cudaBindTexture(0, estimatedYMovementTexDeviceCurrentCheckerboard, estimatedMovementYCheckerboard2Device, numPixInCheckerboardCurrentLevel*sizeof(float));

	//bind the linear memory storing to computed message values to copy from to a texture for the "second" checkerboard
	cudaBindTexture(0, messageUTexCurrReadCheckerboard, messageDeviceCheckerboard2CopyFrom.messageUDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageDTexCurrReadCheckerboard, messageDeviceCheckerboard2CopyFrom.messageDDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageLTexCurrReadCheckerboard, messageDeviceCheckerboard2CopyFrom.messageLDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageRTexCurrReadCheckerboard, messageDeviceCheckerboard2CopyFrom.messageRDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);


	//run the kernel to copy the message values from the previous level to the next level for part 2 of the checkerboard
	//and adjust the current movement parameters based on the current estimated movement
	copyPrevLevelToNextLevelBPCheckerboard <<< grid, threads >>> 
															(messageDeviceCheckerboard1CopyTo, messageDeviceCheckerboard2CopyTo,
															widthLevelActualIntegerSize, heightLevelActualIntegerSize, 
															widthNextLevelActualIntegerSize, heightNextLevelActualIntegerSize,
															paramsNextLevelDeviceCheckerboard1, paramsNextLevelDeviceCheckerboard2,
															CHECKERBOARD_PART_2_ENUM);

	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	//unbind the texture that was bound to the current movement parameters
	cudaUnbindTexture(currentPixParamsTexCurrentCheckerboard);

	//unbind the texture that was bound to the previous message values
	cudaUnbindTexture(messageUTexCurrReadCheckerboard);
	cudaUnbindTexture(messageDTexCurrReadCheckerboard);
	cudaUnbindTexture(messageLTexCurrReadCheckerboard);
	cudaUnbindTexture(messageRTexCurrReadCheckerboard);

	//unbind the texture that was bound to the estimated motions for checkerboard 2
	cudaUnbindTexture(estimatedXMovementTexDeviceCurrentCheckerboard);
	cudaUnbindTexture(estimatedYMovementTexDeviceCurrentCheckerboard);
}


//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
//pyramid; here the size of each level is constant
void copyMessageValuesToNextLevelDownLevelSizeConst(
	checkerboardMessagesDeviceStruct messageDeviceCheckerboard1CopyFrom, 
	checkerboardMessagesDeviceStruct messageDeviceCheckerboard2CopyFrom, 
	currentStartMoveParamsPixelAtLevel* paramsPrevLevelDeviceCheckerboard1, currentStartMoveParamsPixelAtLevel* paramsPrevLevelDeviceCheckerboard2,
	currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboard1, currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboard2,
	float* estimatedMovementXCheckerboard1Device, float* estimatedMovementYCheckerboard1Device,
	float* estimatedMovementXCheckerboard2Device, float* estimatedMovementYCheckerboard2Device,
	checkerboardMessagesDeviceStruct messageDeviceCheckerboard1CopyTo,
	checkerboardMessagesDeviceStruct messageDeviceCheckerboard2CopyTo,
	int widthLevelActualIntegerSize, int heightLevelActualIntegerSize,
	size_t numBytesDataAndMessageSetInCheckerboardAtLevel)
{
	//retrieve the width and height of the checkerboard
	int widthOfCheckerboard = widthLevelActualIntegerSize / 2;
	int heightOfCheckerboard = heightLevelActualIntegerSize;

	//set the execution parameters for copying the message values to the next level
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid((unsigned int)ceil((float)widthOfCheckerboard / (float)threads.x), (unsigned int)ceil((float)heightOfCheckerboard / (float)threads.y));

	//declare and set the number of pixels in the checkerboard at the current level
	int numPixInCheckerboardCurrentLevel = widthOfCheckerboard * heightOfCheckerboard;

	//bind the linear memory storing to computed message values to copy from to a texture for the "first" checkerboard
	cudaBindTexture(0, messageUTexCurrReadCheckerboard, messageDeviceCheckerboard1CopyFrom.messageUDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageDTexCurrReadCheckerboard, messageDeviceCheckerboard1CopyFrom.messageDDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageLTexCurrReadCheckerboard, messageDeviceCheckerboard1CopyFrom.messageLDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageRTexCurrReadCheckerboard, messageDeviceCheckerboard1CopyFrom.messageRDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);

	//bind the previous level parameters to the appropriate texture in the device
	cudaBindTexture(0, currentPixParamsTexCurrentCheckerboard, paramsPrevLevelDeviceCheckerboard1, numPixInCheckerboardCurrentLevel*sizeof(currentStartMoveParamsPixelAtLevel));

	//bind the current motion estimates to the appropriate texture in the device
	cudaBindTexture(0, estimatedXMovementTexDeviceCurrentCheckerboard, estimatedMovementXCheckerboard1Device, numPixInCheckerboardCurrentLevel*sizeof(float));
	cudaBindTexture(0, estimatedYMovementTexDeviceCurrentCheckerboard, estimatedMovementYCheckerboard1Device, numPixInCheckerboardCurrentLevel*sizeof(float));


	//run the kernel to copy the message values from the previous level to the next level for part 1 of the checkerboard
	//and adjust the current movement parameters based on the current estimated movement
	copyPrevLevelToNextLevelBPCheckerboardLevelSizeConst <<< grid, threads >>> 
															(messageDeviceCheckerboard1CopyTo,
															widthLevelActualIntegerSize, heightLevelActualIntegerSize,
															paramsNextLevelDeviceCheckerboard1);

	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	//unbind the texture that was bound to the current movement parameters
	cudaUnbindTexture(currentPixParamsTexCurrentCheckerboard);

	//unbind the texture that was bound to the estimated motions for checkerboard 1
	cudaUnbindTexture(estimatedXMovementTexDeviceCurrentCheckerboard);
	cudaUnbindTexture(estimatedYMovementTexDeviceCurrentCheckerboard);

	//unbind the texture that was bound to the previous message values
	cudaUnbindTexture(messageUTexCurrReadCheckerboard);
	cudaUnbindTexture(messageDTexCurrReadCheckerboard);
	cudaUnbindTexture(messageLTexCurrReadCheckerboard);
	cudaUnbindTexture(messageRTexCurrReadCheckerboard);

	//bind the current movement parameters for checkerboard 2 since that is now being updated
	cudaBindTexture(0, currentPixParamsTexCurrentCheckerboard, paramsPrevLevelDeviceCheckerboard2, numPixInCheckerboardCurrentLevel*sizeof(currentStartMoveParamsPixelAtLevel));

	//bind the estimated motions for checkerboard 2 since that is now being updated
	cudaBindTexture(0, estimatedXMovementTexDeviceCurrentCheckerboard, estimatedMovementXCheckerboard2Device, numPixInCheckerboardCurrentLevel*sizeof(float));
	cudaBindTexture(0, estimatedYMovementTexDeviceCurrentCheckerboard, estimatedMovementYCheckerboard2Device, numPixInCheckerboardCurrentLevel*sizeof(float));

	//bind the linear memory storing to computed message values to copy from to a texture for the "second" checkerboard
	cudaBindTexture(0, messageUTexCurrReadCheckerboard, messageDeviceCheckerboard2CopyFrom.messageUDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageDTexCurrReadCheckerboard, messageDeviceCheckerboard2CopyFrom.messageDDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageLTexCurrReadCheckerboard, messageDeviceCheckerboard2CopyFrom.messageLDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageRTexCurrReadCheckerboard, messageDeviceCheckerboard2CopyFrom.messageRDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);


	//run the kernel to copy the message values from the previous level to the next level for part 2 of the checkerboard
	//and adjust the current movement parameters based on the current estimated movement
	copyPrevLevelToNextLevelBPCheckerboardLevelSizeConst <<< grid, threads >>> 
															(messageDeviceCheckerboard2CopyTo,
															widthLevelActualIntegerSize, heightLevelActualIntegerSize, 
															paramsNextLevelDeviceCheckerboard2);

	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	//unbind the texture that was bound to the current movement parameters
	cudaUnbindTexture(currentPixParamsTexCurrentCheckerboard);

	//unbind the texture that was bound to the previous message values
	cudaUnbindTexture(messageUTexCurrReadCheckerboard);
	cudaUnbindTexture(messageDTexCurrReadCheckerboard);
	cudaUnbindTexture(messageLTexCurrReadCheckerboard);
	cudaUnbindTexture(messageRTexCurrReadCheckerboard);

	//unbind the texture that was bound to the estimated motions for checkerboard 2
	cudaUnbindTexture(estimatedXMovementTexDeviceCurrentCheckerboard);
	cudaUnbindTexture(estimatedYMovementTexDeviceCurrentCheckerboard);
}

//initialize the combined data and estimated movement costs at each pixel...assuming method is sampling invarient...
void initializeDataAndEstMoveCostsCurrentLevel(float* image1PixelsDevice, float* image2PixelsDevice,
														float* dataCostDeviceCheckerboard1, float* dataCostDeviceCheckerboard2,
														currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1,
														currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
														currBeliefPropParams& currentBeliefPropParams,
														int widthCurrentLevel, int heightCurrentLevel, float* estMoveXDevice, float* estMoveYDevice)
{
	//set the width and height of checkerboard, which is half the width and equal to the height of the current level
	int widthOfCheckerboard = widthCurrentLevel / 2;
	int heightOfCheckerboard = heightCurrentLevel;

	//declare and set the number of pixel in a checkerboard at the current level
	int numPixInCheckerboardAtLevel = widthOfCheckerboard * heightOfCheckerboard;

	//allocate array and copy image data
	//data is in the single-float value format
	cudaChannelFormatDesc channelDescImages = cudaCreateChannelDesc<float>();

	//store the two image pixels in the GPU in a CUDA array
	cudaArray* cu_arrayImage1BP;
	cudaArray* cu_arrayImage2BP;

	//allocate and then copy the image pixel data for the two images on the GPU
	CUDA_SAFE_CALL( cudaMallocArray( &cu_arrayImage1BP, &channelDescImages, currentBeliefPropParams.widthImages, currentBeliefPropParams.heightImages )); 
	CUDA_SAFE_CALL( cudaMallocArray( &cu_arrayImage2BP, &channelDescImages, currentBeliefPropParams.widthImages, currentBeliefPropParams.heightImages )); 

	CUDA_SAFE_CALL( cudaMemcpyToArray( cu_arrayImage1BP, 0, 0, image1PixelsDevice, currentBeliefPropParams.widthImages*currentBeliefPropParams.heightImages*sizeof(float), cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL( cudaMemcpyToArray( cu_arrayImage2BP, 0, 0, image2PixelsDevice, currentBeliefPropParams.widthImages*currentBeliefPropParams.heightImages*sizeof(float), cudaMemcpyDeviceToDevice));

	// set texture parameters for the CUDA arrays to hold the input images
	image1PixelsTexture.addressMode[0] = cudaAddressModeClamp;
	image1PixelsTexture.addressMode[1] = cudaAddressModeClamp;
	image1PixelsTexture.filterMode = cudaFilterModeLinear; //use linear interpolation on reading points
	image1PixelsTexture.normalized = true; // access with NORMALIZED texture coordinates

	image2PixelsTexture.addressMode[0] = cudaAddressModeClamp;
	image2PixelsTexture.addressMode[1] = cudaAddressModeClamp;
	image2PixelsTexture.filterMode = cudaFilterModeLinear; //use linear interpolation on reading points
	image2PixelsTexture.normalized = true; // access with NORMALIZED texture coordinates

	//Bind the CUDA Arrays holding the input image pixel arrays to the appropriate texture
	CUDA_SAFE_CALL( cudaBindTextureToArray( image1PixelsTexture, cu_arrayImage1BP, channelDescImages));
	CUDA_SAFE_CALL( cudaBindTextureToArray( image2PixelsTexture, cu_arrayImage2BP, channelDescImages));

	//bind the current movement parameters for checkerboard 1 since that is now being updated
	cudaBindTexture(0, currentPixParamsTexCurrentCheckerboard, paramsCurrentLevelDeviceCheckerboard1, numPixInCheckerboardAtLevel*sizeof(currentStartMoveParamsPixelAtLevel));

	//set the current execution parameters
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid((unsigned int)ceil((float)widthOfCheckerboard / (float)threads.x), (unsigned int)ceil((float)heightOfCheckerboard / (float)threads.y));


	//set the data costs of the first part of the checkerboard

	#if (DATA_COST_METHOD == ADD_COSTS_METHOD)
	//initialDataCostsAtCurrentLevelAddCostsCheckerboard <<< grid, threads >>> (dataCostDeviceCheckerboard1,
																	widthCurrentLevel, heightCurrentLevel,
																	CHECKERBOARD_PART_1_ENUM);

	#elif (DATA_COST_METHOD == SAMP_INVARIENT_SAMPLING)

	initialDataAndEstMoveCostsAtCurrentLevelAddCostsCheckerboardSampInvarient <<< grid, threads >>> (dataCostDeviceCheckerboard1,
																	widthCurrentLevel, heightCurrentLevel,
																	CHECKERBOARD_PART_1_ENUM,
														estMoveXDevice, estMoveYDevice);

	#endif //DATA_COST_METHOD

	CUDA_SAFE_CALL(cudaThreadSynchronize());

	//unbind the texture attached to the current movement parameters
	cudaUnbindTexture(currentPixParamsTexCurrentCheckerboard);

	//bind the current movement parameters for checkerboard 2 since that is now being updated
	cudaBindTexture(0, currentPixParamsTexCurrentCheckerboard, paramsCurrentLevelDeviceCheckerboard2, numPixInCheckerboardAtLevel*sizeof(currentStartMoveParamsPixelAtLevel));


	//set the data costs of the second part of the checkerboard
	#if (DATA_COST_METHOD == ADD_COSTS_METHOD)
	//initialDataCostsAtCurrentLevelAddCostsCheckerboard <<< grid, threads >>> (dataCostDeviceCheckerboard2,
																	widthCurrentLevel, heightCurrentLevel,
																	CHECKERBOARD_PART_2_ENUM);

	#elif (DATA_COST_METHOD == SAMP_INVARIENT_SAMPLING)
	initialDataAndEstMoveCostsAtCurrentLevelAddCostsCheckerboardSampInvarient <<< grid, threads >>> (dataCostDeviceCheckerboard2,
																	widthCurrentLevel, heightCurrentLevel,
																	CHECKERBOARD_PART_2_ENUM,
														estMoveXDevice, estMoveYDevice);
	#endif //DATA_COST_METHOD

	CUDA_SAFE_CALL(cudaThreadSynchronize());

	//unbind the texture attached to the current movement parameters
	cudaUnbindTexture(currentPixParamsTexCurrentCheckerboard);

	//unbind the texture attached to the image pixel values
	cudaUnbindTexture( image1PixelsTexture);
	cudaUnbindTexture( image2PixelsTexture);

	//image data no longer needed after data costs are computed
	CUDA_SAFE_CALL(cudaFreeArray(cu_arrayImage1BP));
	CUDA_SAFE_CALL(cudaFreeArray(cu_arrayImage2BP));
}


//initialize the data cost at each pixel at the current level
//assume that motion parameters for current level are set
void initializeDataCostsCurrentLevel(float* image1PixelsDevice, float* image2PixelsDevice,
														float* dataCostDeviceCheckerboard1, float* dataCostDeviceCheckerboard2,
														currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1,
														currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
														currBeliefPropParams& currentBeliefPropParams,
														int widthCurrentLevel, int heightCurrentLevel)
{
	//set the width and height of checkerboard, which is half the width and equal to the height of the current level
	int widthOfCheckerboard = widthCurrentLevel / 2;
	int heightOfCheckerboard = heightCurrentLevel;

	//declare and set the number of pixel in a checkerboard at the current level
	int numPixInCheckerboardAtLevel = widthOfCheckerboard * heightOfCheckerboard;

	//allocate array and copy image data
	//data is in the single-float value format
	cudaChannelFormatDesc channelDescImages = cudaCreateChannelDesc<float>();

	//store the two image pixels in the GPU in a CUDA array
	cudaArray* cu_arrayImage1BP;
	cudaArray* cu_arrayImage2BP;

	//allocate and then copy the image pixel data for the two images on the GPU
	CUDA_SAFE_CALL( cudaMallocArray( &cu_arrayImage1BP, &channelDescImages, currentBeliefPropParams.widthImages, currentBeliefPropParams.heightImages )); 
	CUDA_SAFE_CALL( cudaMallocArray( &cu_arrayImage2BP, &channelDescImages, currentBeliefPropParams.widthImages, currentBeliefPropParams.heightImages )); 

	CUDA_SAFE_CALL( cudaMemcpyToArray( cu_arrayImage1BP, 0, 0, image1PixelsDevice, currentBeliefPropParams.widthImages*currentBeliefPropParams.heightImages*sizeof(float), cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL( cudaMemcpyToArray( cu_arrayImage2BP, 0, 0, image2PixelsDevice, currentBeliefPropParams.widthImages*currentBeliefPropParams.heightImages*sizeof(float), cudaMemcpyDeviceToDevice));

	// set texture parameters for the CUDA arrays to hold the input images
	image1PixelsTexture.addressMode[0] = cudaAddressModeClamp;
	image1PixelsTexture.addressMode[1] = cudaAddressModeClamp;
	image1PixelsTexture.filterMode = cudaFilterModeLinear; //use linear interpolation on reading points
	image1PixelsTexture.normalized = true; // access with NORMALIZED texture coordinates

	image2PixelsTexture.addressMode[0] = cudaAddressModeClamp;
	image2PixelsTexture.addressMode[1] = cudaAddressModeClamp;
	image2PixelsTexture.filterMode = cudaFilterModeLinear; //use linear interpolation on reading points
	image2PixelsTexture.normalized = true; // access with NORMALIZED texture coordinates

	//Bind the CUDA Arrays holding the input image pixel arrays to the appropriate texture
	CUDA_SAFE_CALL( cudaBindTextureToArray( image1PixelsTexture, cu_arrayImage1BP, channelDescImages));
	CUDA_SAFE_CALL( cudaBindTextureToArray( image2PixelsTexture, cu_arrayImage2BP, channelDescImages));

	//bind the current movement parameters for checkerboard 1 since that is now being updated
	cudaBindTexture(0, currentPixParamsTexCurrentCheckerboard, paramsCurrentLevelDeviceCheckerboard1, numPixInCheckerboardAtLevel*sizeof(currentStartMoveParamsPixelAtLevel));

	//set the current execution parameters
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid((unsigned int)ceil((float)widthOfCheckerboard / (float)threads.x), (unsigned int)ceil((float)heightOfCheckerboard / (float)threads.y));


	//set the data costs of the first part of the checkerboard

	#if (DATA_COST_METHOD == ADD_COSTS_METHOD)
	initialDataCostsAtCurrentLevelAddCostsCheckerboard <<< grid, threads >>> (dataCostDeviceCheckerboard1,
																	widthCurrentLevel, heightCurrentLevel,
																	CHECKERBOARD_PART_1_ENUM);

	#elif (DATA_COST_METHOD == SAMP_INVARIENT_SAMPLING)

	initialDataCostsAtCurrentLevelAddCostsCheckerboardSampInvarient <<< grid, threads >>> (dataCostDeviceCheckerboard1,
																	widthCurrentLevel, heightCurrentLevel,
																	CHECKERBOARD_PART_1_ENUM);

	#endif //DATA_COST_METHOD

	CUDA_SAFE_CALL(cudaThreadSynchronize());

	//unbind the texture attached to the current movement parameters
	cudaUnbindTexture(currentPixParamsTexCurrentCheckerboard);

	//bind the current movement parameters for checkerboard 2 since that is now being updated
	cudaBindTexture(0, currentPixParamsTexCurrentCheckerboard, paramsCurrentLevelDeviceCheckerboard2, numPixInCheckerboardAtLevel*sizeof(currentStartMoveParamsPixelAtLevel));


	//set the data costs of the second part of the checkerboard
	#if (DATA_COST_METHOD == ADD_COSTS_METHOD)
	initialDataCostsAtCurrentLevelAddCostsCheckerboard <<< grid, threads >>> (dataCostDeviceCheckerboard2,
																	widthCurrentLevel, heightCurrentLevel,
																	CHECKERBOARD_PART_2_ENUM);

	#elif (DATA_COST_METHOD == SAMP_INVARIENT_SAMPLING)
	initialDataCostsAtCurrentLevelAddCostsCheckerboardSampInvarient <<< grid, threads >>> (dataCostDeviceCheckerboard2,
																	widthCurrentLevel, heightCurrentLevel,
																	CHECKERBOARD_PART_2_ENUM);
	#endif //DATA_COST_METHOD

	CUDA_SAFE_CALL(cudaThreadSynchronize());

	//unbind the texture attached to the current movement parameters
	cudaUnbindTexture(currentPixParamsTexCurrentCheckerboard);

	//unbind the texture attached to the image pixel values
	cudaUnbindTexture( image1PixelsTexture);
	cudaUnbindTexture( image2PixelsTexture);

	//image data no longer needed after data costs are computed
	CUDA_SAFE_CALL(cudaFreeArray(cu_arrayImage1BP));
	CUDA_SAFE_CALL(cudaFreeArray(cu_arrayImage2BP));
}


//initialize the data cost at each pixel at the current level
//assume that motion parameters for current level are set
//no textures are used for the parameters here...
void initializeDataCostsCurrentLevelNoTexParams(float* image1PixelsDevice, float* image2PixelsDevice,
														float* dataCostDeviceCheckerboard1, float* dataCostDeviceCheckerboard2,
														currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1,
														currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
														currBeliefPropParams& currentBeliefPropParams,
														int widthCurrentLevel, int heightCurrentLevel, int paramsOffset)
{
	//set the width and height of checkerboard, which is half the width and equal to the height of the current level
	int widthOfCheckerboard = widthCurrentLevel / 2;
	int heightOfCheckerboard = heightCurrentLevel;

	//allocate array and copy image data
	//data is in the single-float value format
	cudaChannelFormatDesc channelDescImages = cudaCreateChannelDesc<float>();

	//store the image pixels in the GPU in a CUDA array
	cudaArray* cu_arrayImage1BP;
	cudaArray* cu_arrayImage2BP;

	//allocate and then copy the image pixel data for the two images on the GPU
	CUDA_SAFE_CALL( cudaMallocArray( &cu_arrayImage1BP, &channelDescImages, currentBeliefPropParams.widthImages, currentBeliefPropParams.heightImages )); 
	CUDA_SAFE_CALL( cudaMallocArray( &cu_arrayImage2BP, &channelDescImages, currentBeliefPropParams.widthImages, currentBeliefPropParams.heightImages )); 

	CUDA_SAFE_CALL( cudaMemcpyToArray( cu_arrayImage1BP, 0, 0, image1PixelsDevice, currentBeliefPropParams.widthImages*currentBeliefPropParams.heightImages*sizeof(float), cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL( cudaMemcpyToArray( cu_arrayImage2BP, 0, 0, image2PixelsDevice, currentBeliefPropParams.widthImages*currentBeliefPropParams.heightImages*sizeof(float), cudaMemcpyDeviceToDevice));

	// set texture parameters for the CUDA arrays to hold the input images
	image1PixelsTexture.addressMode[0] = cudaAddressModeClamp;
	image1PixelsTexture.addressMode[1] = cudaAddressModeClamp;
	image1PixelsTexture.filterMode = cudaFilterModeLinear; //use linear interpolation on reading points
	image1PixelsTexture.normalized = true; // access with NORMALIZED texture coordinates

	image2PixelsTexture.addressMode[0] = cudaAddressModeClamp;
	image2PixelsTexture.addressMode[1] = cudaAddressModeClamp;
	image2PixelsTexture.filterMode = cudaFilterModeLinear; //use linear interpolation on reading points
	image2PixelsTexture.normalized = true; // access with NORMALIZED texture coordinates

	//Bind the CUDA Arrays holding the input image pixel arrays to the appropriate texture
	CUDA_SAFE_CALL( cudaBindTextureToArray( image1PixelsTexture, cu_arrayImage1BP, channelDescImages));
	CUDA_SAFE_CALL( cudaBindTextureToArray( image2PixelsTexture, cu_arrayImage2BP, channelDescImages));

	
	//set the current execution parameters
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid((unsigned int)ceil((float)widthOfCheckerboard / (float)threads.x), (unsigned int)ceil((float)heightOfCheckerboard / (float)threads.y));

	#if (DATA_COST_METHOD == SAMP_INVARIENT_SAMPLING)


	//need to use 'sampling invarient' method...
	printf("ParamsOffset: %d\n", paramsOffset);
	printf("widthCurrLevel: %d\n", widthCurrentLevel);
	printf("heightCurrLevel: %d\n", heightCurrentLevel);

	//set the data costs of the first part of the checkerboard				
					
	initialDataCostsAtCurrentLevelAddCostsCheckerboardSampInvarientNoTextures <<< grid, threads >>> (dataCostDeviceCheckerboard1,
											widthCurrentLevel, heightCurrentLevel,
											CHECKERBOARD_PART_1_ENUM, &(paramsCurrentLevelDeviceCheckerboard1[paramsOffset]));


	CUDA_SAFE_CALL(cudaThreadSynchronize());

	
	//set the data costs of the second part of the checkerboard
	
	initialDataCostsAtCurrentLevelAddCostsCheckerboardSampInvarientNoTextures <<< grid, threads >>> (dataCostDeviceCheckerboard2,
											widthCurrentLevel, heightCurrentLevel,
											CHECKERBOARD_PART_2_ENUM, &(paramsCurrentLevelDeviceCheckerboard2[paramsOffset]));
	#endif //DATA_COST_METHOD

	CUDA_SAFE_CALL(cudaThreadSynchronize());


	//image data no longer needed after data costs are computed
	CUDA_SAFE_CALL(cudaFreeArray(cu_arrayImage1BP));
	CUDA_SAFE_CALL(cudaFreeArray(cu_arrayImage2BP));
}





//initialize the estimated movement costs and add it to the data costs in the overall computation...
void initEstMovementInDataCosts(float* estMovesXDir, float* estMovesYDir, 
				float* dataCostDeviceCheckerboard1, float* dataCostDeviceCheckerboard2,
				currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1,
				currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
				currBeliefPropParams& currentBeliefPropParams,
				int widthOfCheckerboard, int heightOfCheckerboard,
				int widthCurrentLevel, int heightCurrentLevel,
				int numBytesDataAndMessagesCurrLevel)
{
	//declare and set the number of pixel in a checkerboard at the current level
	int numPixInCheckerboardAtLevel = widthOfCheckerboard * heightOfCheckerboard;

	//allocate space for the estimated movement costs on the device
	float* estMoveCostsCheckerboard1Device;
	float* estMoveCostsCheckerboard2Device;
	cudaMalloc((void**)&estMoveCostsCheckerboard1Device, numBytesDataAndMessagesCurrLevel);
	cudaMalloc((void**)&estMoveCostsCheckerboard2Device, numBytesDataAndMessagesCurrLevel);

	//set the execution parameters for setting the parameters for the estimated costs...
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid((unsigned int)ceil((float)widthOfCheckerboard / (float)threads.x), (unsigned int)ceil((float)heightOfCheckerboard / (float)threads.y));

	//bind the current movement parameters for checkerboard 1 since that is now being updated
	cudaBindTexture(0, currentPixParamsTexCurrentCheckerboard, paramsCurrentLevelDeviceCheckerboard1, numPixInCheckerboardAtLevel*sizeof(currentStartMoveParamsPixelAtLevel));
	
	//call the thread to get the `deviation' from the estimated movement costs in `part 1' of the checkerboard
	getDevFromEstMoveCosts <<< grid, threads >>> (estMovesXDir, estMovesYDir,
				estMoveCostsCheckerboard1Device, widthCurrentLevel, heightCurrentLevel, CHECKERBOARD_PART_1_ENUM);

	//synchronize between the kernel calls...
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	//unbind the texture attached to the current movement parameters
	cudaUnbindTexture(currentPixParamsTexCurrentCheckerboard);

	//bind the current movement parameters for checkerboard 2 since that is now being updated
	cudaBindTexture(0, currentPixParamsTexCurrentCheckerboard, paramsCurrentLevelDeviceCheckerboard2, numPixInCheckerboardAtLevel*sizeof(currentStartMoveParamsPixelAtLevel));

	//call the thread to get the `deviation' from the estimated movement costs in `part 2' of the checkerboard
	getDevFromEstMoveCosts <<< grid, threads >>> (estMovesXDir, estMovesYDir,
				estMoveCostsCheckerboard2Device, widthCurrentLevel, heightCurrentLevel, CHECKERBOARD_PART_2_ENUM);

	//synchronize after the kernel call...
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	//now add the estimated movement costs to the data costs for each checkerboard portion...
	addInputData <<< grid, threads >>> (dataCostDeviceCheckerboard1, estMoveCostsCheckerboard1Device, widthCurrentLevel, heightCurrentLevel);

	//synchronize between the kernel calls...
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	addInputData <<< grid, threads >>> (dataCostDeviceCheckerboard2, estMoveCostsCheckerboard2Device, widthCurrentLevel, heightCurrentLevel);

	//synchronize after the kernel call...
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	//free the space allocated to the estimated move costs since it's been integrated with the data costs...
	cudaFree(estMoveCostsCheckerboard1Device);
	cudaFree(estMoveCostsCheckerboard2Device);
}


//initialize the estimated movement costs and add it to the data costs in the overall computation...without using textures...
void initEstMovementInDataCostsNoTextures(float* estMovesXDir, float* estMovesYDir, 
				float* dataCostDeviceCheckerboard1, float* dataCostDeviceCheckerboard2,
				currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1,
				currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
				currBeliefPropParams& currentBeliefPropParams,
				int widthOfCheckerboard, int heightOfCheckerboard,
				int widthCurrentLevel, int heightCurrentLevel,
				int numBytesDataAndMessagesCurrLevel,
				int paramOffset)
{
	//allocate space for the estimated movement costs on the device
	float* estMoveCostsCheckerboard1Device;
	float* estMoveCostsCheckerboard2Device;
	cudaMalloc((void**)&estMoveCostsCheckerboard1Device, numBytesDataAndMessagesCurrLevel);
	cudaMalloc((void**)&estMoveCostsCheckerboard2Device, numBytesDataAndMessagesCurrLevel);

	//set the execution parameters for setting the parameters for the estimated costs...
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid((unsigned int)ceil((float)widthOfCheckerboard / (float)threads.x), (unsigned int)ceil((float)heightOfCheckerboard / (float)threads.y));

	//call the thread to get the `deviation' from the estimated movement costs in `part 1' of the checkerboard
	getDevFromEstMoveCostsNoTextures <<< grid, threads >>> (estMovesXDir, estMovesYDir,
				estMoveCostsCheckerboard1Device, widthCurrentLevel, heightCurrentLevel, CHECKERBOARD_PART_1_ENUM, &(paramsCurrentLevelDeviceCheckerboard1[paramOffset]));

	//synchronize between the kernel calls...
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	//call the thread to get the `deviation' from the estimated movement costs in `part 2' of the checkerboard
	getDevFromEstMoveCostsNoTextures <<< grid, threads >>> (estMovesXDir, estMovesYDir,
				estMoveCostsCheckerboard2Device, widthCurrentLevel, heightCurrentLevel, CHECKERBOARD_PART_2_ENUM, &(paramsCurrentLevelDeviceCheckerboard2[paramOffset]));

	//synchronize after the kernel call...
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	//now add the estimated movement costs to the data costs for each checkerboard portion...
	addInputData <<< grid, threads >>> (dataCostDeviceCheckerboard1, estMoveCostsCheckerboard1Device, widthCurrentLevel, heightCurrentLevel);

	//synchronize between the kernel calls...
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	addInputData <<< grid, threads >>> (dataCostDeviceCheckerboard2, estMoveCostsCheckerboard2Device, widthCurrentLevel, heightCurrentLevel);

	//synchronize after the kernel call...
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	//free the space allocated to the estimated move costs since it's been integrated with the data costs...
	cudaFree(estMoveCostsCheckerboard1Device);
	cudaFree(estMoveCostsCheckerboard2Device);
}

//function to take the resultant movement and generate the estimated movement for the 'next image set'
void genEstMoveNextImSet(float* resultantXMovePrevIm, float* resultantYMovePrevIm, float* estMovementDataX, float* estMovementDataY, int widthImageAtLevel, int heightImageAtLevel)
{
	//set the execution parameters for setting the parameters for the estimated costs...
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid((unsigned int)ceil((float)widthImageAtLevel / (float)threads.x), (unsigned int)ceil((float)heightImageAtLevel / (float)threads.y));

	//call the CUDA kernel to set 'no movement' by default
	setNoMoveDataAtPixels <<< grid, threads >>> (estMovementDataX, estMovementDataY, widthImageAtLevel, heightImageAtLevel);

	//synchronize after the kernel call...
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	//now use the output movement to estimate the movement in the next `image set'
	setEstMoveAtPixels <<< grid, threads >>> (resultantXMovePrevIm, resultantYMovePrevIm, estMovementDataX,
				estMovementDataY, widthImageAtLevel, heightImageAtLevel);

	//synchronize after the kernel call...
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}


//initialize the parameters defining the starting movement at the current level on the device
void initializeStartMovementParams(currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1, currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
												int widthLevel, int heightLevel, currentStartMoveParamsPixelAtLevel startMoveParams)
{
	//retrieve the width and height of the checkerboard at the current level
	int widthOfCheckerboard = widthLevel / 2;
	int heightOfCheckerboard = heightLevel;

	//set the execution parameters for setting the parameters at the current level
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid((unsigned int)ceil((float)widthOfCheckerboard / (float)threads.x), (unsigned int)ceil((float)heightOfCheckerboard / (float)threads.y));

	//run the kernel to set the parameters at the current level
	setParamsToGivenParamKernel <<< grid, threads >>> (startMoveParams, paramsCurrentLevelDeviceCheckerboard1,
														paramsCurrentLevelDeviceCheckerboard2, widthLevel, heightLevel);

	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}


//initialize the parameters defining the starting movement at the current level on the device using the hierarchical implementation...
void initializeStartMovementParamsHierarchImp(currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1, currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
						currentStartMoveParamsPixelAtLevel startMoveParams, paramOffsetsSizes* hierarchParams, int numHierarchLevels)
{
	//go through each level of the hierarchy...
	for (int currLevelNum = 0; currLevelNum < numHierarchLevels; currLevelNum++) 
	{
		//retrieve the width and height of the checkerboard at the current level
		int widthOfCheckerboard = hierarchParams[currLevelNum].levelWidthCheckerboard;
		int heightOfCheckerboard = hierarchParams[currLevelNum].levelHeightCheckerboard;

		//set the execution parameters for setting the parameters at the current level
		dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
		dim3 grid((unsigned int)ceil((float)widthOfCheckerboard / (float)threads.x), (unsigned int)ceil((float)heightOfCheckerboard / (float)threads.y));

		//run the kernel to set the parameters at the current level
		//input is the full width of each level rather than the `checkerboard'
		setParamsToGivenParamKernel <<< grid, threads >>> (startMoveParams, &(paramsCurrentLevelDeviceCheckerboard1[hierarchParams[currLevelNum].offset]),
									&(paramsCurrentLevelDeviceCheckerboard2[hierarchParams[currLevelNum].offset]), (hierarchParams[currLevelNum].levelWidthCheckerboard)*2, hierarchParams[currLevelNum].levelHeightCheckerboard);

		CUDA_SAFE_CALL( cudaThreadSynchronize() );
	}
}

//initialize the message values with no previous message values...all message values are set to DEFAULT_INITIAL_MESSAGE_VAL
void initializeMessageVals(checkerboardMessagesDeviceStruct messageDeviceCheckerboard1,
									checkerboardMessagesDeviceStruct messageDeviceCheckerboard2,
									int widthCurrentLevel, int heightCurrentLevel)
{
	//set the width and height of checkerboard, which is half the width and equal to the height of the current level
	int widthOfCheckerboard = widthCurrentLevel / 2;
	int heightOfCheckerboard = heightCurrentLevel;

	//set the execution parameters for the kernel used to initialize the message values
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid((unsigned int)ceil((float)widthOfCheckerboard / (float)threads.x), (unsigned int)ceil((float)heightOfCheckerboard / (float)threads.y));

	initMessageValsToDefaultCurrCheckerboard <<< grid, threads >>> (messageDeviceCheckerboard1, messageDeviceCheckerboard2,
																	widthCurrentLevel, heightCurrentLevel);

	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

//run the kernel function to retrieve the "best" estimate for each pixel at the current level on the device
//this is used to initialize each level of the hierarchy as to what the motion range is "allowed" at the next level
void retrieveBestMotionEstLevel(float* movementXBetweenImagesCheckerboard1Device, float* movementYBetweenImagesCheckerboard1Device,
								float* movementXBetweenImagesCheckerboard2Device, float* movementYBetweenImagesCheckerboard2Device,
								checkerboardMessagesDeviceStruct messageDeviceCheckerboard1,
								checkerboardMessagesDeviceStruct messageDeviceCheckerboard2,
								float* dataCostsDeviceCheckerboard1, float* dataCostsDeviceCheckerboard2,
								currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1,
								currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
								size_t numBytesDataAndMessageSetInCheckerboardAtLevel, int widthLevel, int heightLevel)
{
	//retrieve the width and height of the checkerboard at the current level
	int widthOfCheckerboard = widthLevel / 2;
	int heightOfCheckerboard = heightLevel;

	//set the execution parameters for setting the parameters at the current level
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid((unsigned int)ceil((float)widthOfCheckerboard / (float)threads.x), (unsigned int)ceil((float)heightOfCheckerboard / (float)threads.y));

	//bind the linear memory storing the computed message values for the current checkerboard to the appropriate texture
	cudaBindTexture(0, messageUTexCurrReadCheckerboard, messageDeviceCheckerboard1.messageUDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageDTexCurrReadCheckerboard, messageDeviceCheckerboard1.messageDDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageLTexCurrReadCheckerboard, messageDeviceCheckerboard1.messageLDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageRTexCurrReadCheckerboard, messageDeviceCheckerboard1.messageRDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);

	//bind the linear memory storing the data costs for the current checkerboard to the appropriate texture
	cudaBindTexture(0, dataCostsCurrCheckerboard, dataCostsDeviceCheckerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel);

	//bind the linear memory storing the current movement parameters for each pixel to the appropriate texture
	cudaBindTexture(0, currentPixParamsTexCurrentCheckerboard, paramsCurrentLevelDeviceCheckerboard1, widthOfCheckerboard*heightOfCheckerboard*sizeof(currentStartMoveParamsPixelAtLevel));

	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	//retrieve the estimated motion of the "first" checkerboard
	retrieveOutputMovementCheckerboard <<< grid, threads >>> (movementXBetweenImagesCheckerboard1Device, movementYBetweenImagesCheckerboard1Device, widthLevel, heightLevel);

	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	//unbind the texture bound to the message values, data costs, and current pixel parameters
	cudaUnbindTexture(messageUTexCurrReadCheckerboard);
	cudaUnbindTexture(messageDTexCurrReadCheckerboard);
	cudaUnbindTexture(messageLTexCurrReadCheckerboard);
	cudaUnbindTexture(messageRTexCurrReadCheckerboard);

	cudaUnbindTexture(dataCostsCurrCheckerboard);

	cudaUnbindTexture(currentPixParamsTexCurrentCheckerboard);


	//bind the linear memory storing the computed message values for the current checkerboard to the appropriate texture
	cudaBindTexture(0, messageUTexCurrReadCheckerboard, messageDeviceCheckerboard2.messageUDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageDTexCurrReadCheckerboard, messageDeviceCheckerboard2.messageDDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageLTexCurrReadCheckerboard, messageDeviceCheckerboard2.messageLDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageRTexCurrReadCheckerboard, messageDeviceCheckerboard2.messageRDevice, numBytesDataAndMessageSetInCheckerboardAtLevel);

	//bind the linear memory storing the data costs for the current checkerboard to the appropriate texture
	cudaBindTexture(0, dataCostsCurrCheckerboard, dataCostsDeviceCheckerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel);

	//bind the linear memory storing the current movement parameters for each pixel to the appropriate texture
	cudaBindTexture(0, currentPixParamsTexCurrentCheckerboard, paramsCurrentLevelDeviceCheckerboard2, widthOfCheckerboard*heightOfCheckerboard*sizeof(currentStartMoveParamsPixelAtLevel));

	//now retrieve the estimated motion of the "second" checkerboard
	retrieveOutputMovementCheckerboard <<< grid, threads >>> (movementXBetweenImagesCheckerboard2Device, movementYBetweenImagesCheckerboard2Device, widthLevel, heightLevel);

	cudaThreadSynchronize();

	//unbind the texture bound to the message values, data costs, and current pixel parameters
	cudaUnbindTexture(messageUTexCurrReadCheckerboard);
	cudaUnbindTexture(messageDTexCurrReadCheckerboard);
	cudaUnbindTexture(messageLTexCurrReadCheckerboard);
	cudaUnbindTexture(messageRTexCurrReadCheckerboard);

	cudaUnbindTexture(dataCostsCurrCheckerboard);

	cudaUnbindTexture(currentPixParamsTexCurrentCheckerboard);
}

//host function to set the initial movement parameters at the "top" level for each individual pixel and also the parameters for "all" pixels
//at the top level
void initializeParamsAndMovementOnDevice(currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1,
												currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
												int widthLevel, int heightLevel, 
												currBeliefPropParams& currentBeliefPropParams)
{
	//set the initial bp parameters using the currentBeliefPropParams
	currentStartMoveParamsPixelAtLevel initStartPossMoveParams;
	initStartPossMoveParams.x = currentBeliefPropParams.startPossMoveX;
	initStartPossMoveParams.y = currentBeliefPropParams.startPossMoveY;

	//copy the algorithm parameters for all pixels to constant memory
	setCurrBeliefPropParamsInConstMem(currentBeliefPropParams);

	//set the movement parameters for each pixel on the device
	initializeStartMovementParams(paramsCurrentLevelDeviceCheckerboard1, paramsCurrentLevelDeviceCheckerboard2,
												widthLevel, heightLevel, initStartPossMoveParams);
}

//host function to set the initial movement parameters at the "top" level for each individual pixel in each level of the computation hierarchy and also the parameters for "all" pixels
//at the top level
void initializeParamsAndMovementOnDeviceHierarchImp(currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard1, currentStartMoveParamsPixelAtLevel* paramsCurrentLevelDeviceCheckerboard2,
							paramOffsetsSizes* hierarchParams, int numHierarchLevels, currBeliefPropParams& currentBeliefPropParams)
{
	//set the initial bp parameters using the currentBeliefPropParams
	currentStartMoveParamsPixelAtLevel initStartPossMoveParams;
	initStartPossMoveParams.x = currentBeliefPropParams.startPossMoveX;
	initStartPossMoveParams.y = currentBeliefPropParams.startPossMoveY;

	//copy the algorithm parameters for all pixels to constant memory
	setCurrBeliefPropParamsInConstMem(currentBeliefPropParams);

	//set the movement parameters for each pixel on the device
	initializeStartMovementParamsHierarchImp(paramsCurrentLevelDeviceCheckerboard1, paramsCurrentLevelDeviceCheckerboard2,
							initStartPossMoveParams, hierarchParams, numHierarchLevels);
}

//adjust the parameters for the currBeliefPropParams and update in constant memory
void adjustMovementAllPixParams(currBeliefPropParams& currentBeliefPropParams, int widthCurrentLevel, int heightCurrentLevel)
{
	//set the parameters defining all pixels at the current level using the startHierarchBpAdjustRangeParams
	
	//adjust the increment of possible movements using the current possible movements and the proportion
	//the movement changes between levels
	currentBeliefPropParams.currentMoveIncrementX = currentBeliefPropParams.currentMoveIncrementX * currentBeliefPropParams.propChangeMoveNextLevel;
	currentBeliefPropParams.currentMoveIncrementY = currentBeliefPropParams.currentMoveIncrementY * currentBeliefPropParams.propChangeMoveNextLevel;

	//set the width and height of the pixels in the current level to the value in the current parameters
	currentBeliefPropParams.widthCurrentLevel = widthCurrentLevel;
	currentBeliefPropParams.heightCurrentLevel = heightCurrentLevel;

	//decrement the current level number
	currentBeliefPropParams.currentLevelNum = currentBeliefPropParams.currentLevelNum - 1;

	//copy the updated parameters to constant memory on the device
	setCurrBeliefPropParamsInConstMem(currentBeliefPropParams);
}

//host function that takes the x and y movements in each checkerboard and combines them
void combineXYCheckerboardMovements(float* movementXBetweenImagesCheckerboard1Device, float* movementYBetweenImagesCheckerboard1Device,
									float* movementXBetweenImagesCheckerboard2Device, float* movementYBetweenImagesCheckerboard2Device,
									float* movementXFromImage1To2Device, float* movementYFromImage1To2Device, int widthLevel, int heightLevel)
{
	//set the execution parameters for combining the movements for each checkerboard
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid((unsigned int)ceil((float)widthLevel / (float)threads.x), (unsigned int)ceil((float)heightLevel / (float)threads.y));

	//combine the x and y checkerboard movements for output in a single 2D array
	joinCheckerboardPortions <<< grid, threads >>> (movementXBetweenImagesCheckerboard1Device, movementXBetweenImagesCheckerboard2Device, widthLevel, heightLevel, movementXFromImage1To2Device);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	joinCheckerboardPortions <<< grid, threads >>> (movementYBetweenImagesCheckerboard1Device, movementYBetweenImagesCheckerboard2Device, widthLevel, heightLevel, movementYFromImage1To2Device);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}


//run the belief propagation algorithm on CUDA for motion estimation where the x and the y movements are computed where the two portions of the "checkerboard" are split
//the BP implementation is run such that the range is adjusted in each level in the hierarchy
void runBeliefPropMotionEstimationCUDA(float* image1PixelsDevice, float* image2PixelsDevice, float* movementXFromImage1To2Device, float* movementYFromImage1To2Device, currBeliefPropParams currentBeliefPropParams)
{
	//declare and start the timer
	unsigned int timer;
	cutCreateTimer(&timer);
	cutResetTimer(timer);

	cutStartTimer(timer);

	//define structure containing the current device for each checkerboard
	bpCurrentValsCheckerboard currentBpDeviceValsCheckerboard1;
	bpCurrentValsCheckerboard currentBpDeviceValsCheckerboard2;

	//set the number of possible movements (assume this is the same at every level...)
	int totalNumPossMovements = (currentBeliefPropParams.totalNumMovesXDir*currentBeliefPropParams.totalNumMovesYDir);

	//declare the float for the width and height of the current level
	float widthLevel = currentBeliefPropParams.widthImages;
	float heightLevel = currentBeliefPropParams.heightImages;

	//initialize the movement increment at the current level
	float currentLevMoveIncX = currentBeliefPropParams.currentMoveIncrementX;
	float currentLevMoveIncY = currentBeliefPropParams.currentMoveIncrementY;

	//retrieve the width and height at the top level (it will be a float that is then truncated into an integer as needed)
	//also retrieve the increment in the movement at the "last" level
	for (int numLevel = 0; numLevel < (currentBeliefPropParams.numBpLevels - 1); numLevel++)
	{ 
		//divide the width and height of the level by 2 each time
		widthLevel /= 2.0f;
		heightLevel /= 2.0f;

		//update the movement increment
		currentLevMoveIncX *= currentBeliefPropParams.propChangeMoveNextLevel;		
		currentLevMoveIncY *= currentBeliefPropParams.propChangeMoveNextLevel;


	}




	//set the movement increment at the "bottom" level
	currentBeliefPropParams.motionIncBotLevX = currentLevMoveIncX;
	currentBeliefPropParams.motionIncBotLevY = currentLevMoveIncY;

	int widthCheckerboardLevel = (int)floor(widthLevel) / 2;
	int heightCheckerboardLevel = (int)floor(heightLevel);
	
	//allocate the space for the movement parameters on the device
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.paramsCurrentLevel), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(currentStartMoveParamsPixelAtLevel));
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.paramsCurrentLevel), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(currentStartMoveParamsPixelAtLevel));

	//initialize the start movement parameters at the top level
	initializeParamsAndMovementOnDevice(currentBpDeviceValsCheckerboard1.paramsCurrentLevel, currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
										(int)floor(widthLevel), (int)floor(heightLevel), currentBeliefPropParams);

	//retrieve the number of bytes needed for the data and messages in the checkerboard at the current level
	size_t numBytesDataAndMessagesCurrLevel = widthCheckerboardLevel * heightCheckerboardLevel * totalNumPossMovements * sizeof(float);


	//allocate the space of the messages on the device
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageUDevice), numBytesDataAndMessagesCurrLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageDDevice), numBytesDataAndMessagesCurrLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageLDevice), numBytesDataAndMessagesCurrLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageRDevice), numBytesDataAndMessagesCurrLevel);

	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageUDevice), numBytesDataAndMessagesCurrLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageDDevice), numBytesDataAndMessagesCurrLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageLDevice), numBytesDataAndMessagesCurrLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageRDevice), numBytesDataAndMessagesCurrLevel);


	//initialize the message values at the top level
	initializeMessageVals(currentBpDeviceValsCheckerboard1.checkerboardMessVals, currentBpDeviceValsCheckerboard2.checkerboardMessVals,
						(int)floor(widthLevel), (int)floor(heightLevel));


	//run BP at each level
	for (int numLevel = (currentBeliefPropParams.numBpLevels - 1); numLevel >= 0; numLevel--)
	{
		//allocate the space for the data costs at the current level
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.dataCostsVals), numBytesDataAndMessagesCurrLevel);
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.dataCostsVals), numBytesDataAndMessagesCurrLevel);

		//set the data costs at the current level
		initializeDataCostsCurrentLevel(image1PixelsDevice, image2PixelsDevice,
										currentBpDeviceValsCheckerboard1.dataCostsVals, currentBpDeviceValsCheckerboard2.dataCostsVals,
										currentBpDeviceValsCheckerboard1.paramsCurrentLevel, currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
										currentBeliefPropParams, (int)floor(widthLevel), (int)floor(heightLevel));


		//don't run BP beyond the lowest level
		//if (numLevel == algSettings.numLevels - 1)
		{
			//run BP at the current level
			runBPAtCurrentLevel(currentBpDeviceValsCheckerboard1.checkerboardMessVals, currentBpDeviceValsCheckerboard2.checkerboardMessVals, 
								currentBpDeviceValsCheckerboard1.dataCostsVals, currentBpDeviceValsCheckerboard2.dataCostsVals,
								currentBpDeviceValsCheckerboard1.paramsCurrentLevel, currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
								currentBeliefPropParams.numBpIterations, (int)floor(widthLevel), (int)floor(heightLevel), 
								numBytesDataAndMessagesCurrLevel);
		}


		//allocate the space for the estimated movement at the current level
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.estimatedXMovement), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(float));
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.estimatedYMovement), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(float));

		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.estimatedXMovement), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(float));
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.estimatedYMovement), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(float));

		retrieveBestMotionEstLevel(currentBpDeviceValsCheckerboard1.estimatedXMovement, currentBpDeviceValsCheckerboard1.estimatedYMovement,
									currentBpDeviceValsCheckerboard2.estimatedXMovement, currentBpDeviceValsCheckerboard2.estimatedYMovement,
									currentBpDeviceValsCheckerboard1.checkerboardMessVals,
									currentBpDeviceValsCheckerboard2.checkerboardMessVals,
									currentBpDeviceValsCheckerboard1.dataCostsVals,
									currentBpDeviceValsCheckerboard2.dataCostsVals,
									currentBpDeviceValsCheckerboard1.paramsCurrentLevel,
									currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
									numBytesDataAndMessagesCurrLevel, (int)floor(widthLevel), (int)floor(heightLevel));

		//free the device memory allocated to the data costs
		cudaFree(currentBpDeviceValsCheckerboard1.dataCostsVals);
		cudaFree(currentBpDeviceValsCheckerboard2.dataCostsVals);

		//copy the current message values to the next level and update the parameters if not in bottom level
		if (numLevel > 0)
		{
			//declare and allocate the space to copy the message values to

			//retrieve the width and height of the next level to use
			float widthNextLevel = widthLevel * 2.0f;
			float heightNextLevel = heightLevel * 2.0f;

			//retrieve the width and height of the checkerboard in the next level
			int widthCheckerboardNextLevel = ((int)floor(widthNextLevel)) / 2;
			int heightCheckerboardNextLevel = (int)floor(heightNextLevel);

			size_t numBytesDataAndMessagesNextLevel = widthCheckerboardNextLevel * heightCheckerboardNextLevel * totalNumPossMovements * sizeof(float);

			checkerboardMessagesDeviceStruct messageDeviceCheckerboard1CopyTo;
			checkerboardMessagesDeviceStruct messageDeviceCheckerboard2CopyTo;

			cudaMalloc((void**)&(messageDeviceCheckerboard1CopyTo.messageUDevice), numBytesDataAndMessagesNextLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard1CopyTo.messageDDevice), numBytesDataAndMessagesNextLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard1CopyTo.messageLDevice), numBytesDataAndMessagesNextLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard1CopyTo.messageRDevice), numBytesDataAndMessagesNextLevel);

			cudaMalloc((void**)&(messageDeviceCheckerboard2CopyTo.messageUDevice), numBytesDataAndMessagesNextLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard2CopyTo.messageDDevice), numBytesDataAndMessagesNextLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard2CopyTo.messageLDevice), numBytesDataAndMessagesNextLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard2CopyTo.messageRDevice), numBytesDataAndMessagesNextLevel);

			//declare the movement parameters for the "next" level and allocate space for them
			currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboard1;
			currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboard2;

			cudaMalloc((void**)&paramsNextLevelDeviceCheckerboard1, widthCheckerboardNextLevel * heightCheckerboardNextLevel * sizeof(currentStartMoveParamsPixelAtLevel));
			cudaMalloc((void**)&paramsNextLevelDeviceCheckerboard2, widthCheckerboardNextLevel * heightCheckerboardNextLevel * sizeof(currentStartMoveParamsPixelAtLevel));


			copyMessageValuesToNextLevelDown(
											currentBpDeviceValsCheckerboard1.checkerboardMessVals, 
											currentBpDeviceValsCheckerboard2.checkerboardMessVals, 
											currentBpDeviceValsCheckerboard1.paramsCurrentLevel,
											currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
											paramsNextLevelDeviceCheckerboard1,
											paramsNextLevelDeviceCheckerboard2,
											currentBpDeviceValsCheckerboard1.estimatedXMovement, currentBpDeviceValsCheckerboard1.estimatedYMovement,
											currentBpDeviceValsCheckerboard2.estimatedXMovement, currentBpDeviceValsCheckerboard2.estimatedYMovement,
											messageDeviceCheckerboard1CopyTo,
											messageDeviceCheckerboard2CopyTo,
											(int)floor(widthLevel), (int)floor(heightLevel),
											(int)floor(widthNextLevel), (int)floor(heightNextLevel),
											numBytesDataAndMessagesCurrLevel);

			cudaThreadSynchronize();

			//free the device memory allocated to the previous message values now that they've been copied
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageUDevice);
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageDDevice);
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageLDevice);
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageRDevice);

			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageUDevice);
			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageDDevice);
			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageLDevice);
			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageRDevice);

			//free the device memory allocated to the parameters at the previous level
			cudaFree(currentBpDeviceValsCheckerboard1.paramsCurrentLevel);
			cudaFree(currentBpDeviceValsCheckerboard2.paramsCurrentLevel);


			//set the checkerboard device messages to the messages that were copied to
			currentBpDeviceValsCheckerboard1.checkerboardMessVals = messageDeviceCheckerboard1CopyTo;
			currentBpDeviceValsCheckerboard2.checkerboardMessVals = messageDeviceCheckerboard2CopyTo;

			//set the "current level" parameter to the set "next" level parameters
			currentBpDeviceValsCheckerboard1.paramsCurrentLevel = paramsNextLevelDeviceCheckerboard1;
			currentBpDeviceValsCheckerboard2.paramsCurrentLevel = paramsNextLevelDeviceCheckerboard2;


			//free the device memory allocated to the calculated movement at the previous level
			cudaFree(currentBpDeviceValsCheckerboard1.estimatedXMovement);
			cudaFree(currentBpDeviceValsCheckerboard1.estimatedYMovement);
			cudaFree(currentBpDeviceValsCheckerboard2.estimatedXMovement);
			cudaFree(currentBpDeviceValsCheckerboard2.estimatedYMovement);

			//set the width, height, and the number of bytes in the data and messages in the current level to the "next" level
			widthLevel = widthNextLevel;
			heightLevel = heightNextLevel;

			widthCheckerboardLevel = widthCheckerboardNextLevel;
			heightCheckerboardLevel = heightCheckerboardNextLevel;

			numBytesDataAndMessagesCurrLevel = numBytesDataAndMessagesNextLevel;

			//adjust the movement for all pixels at the "next" level
			adjustMovementAllPixParams(currentBeliefPropParams, (int)floor(widthLevel), (int)floor(heightLevel));
		}
	}

	//free device memory used for storing the message values and movement parameters
	cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageUDevice);
	cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageDDevice);
	cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageLDevice);
	cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageRDevice);

	cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageUDevice);
	cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageDDevice);
	cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageLDevice);
	cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageRDevice);

	cudaFree(currentBpDeviceValsCheckerboard1.paramsCurrentLevel);
	cudaFree(currentBpDeviceValsCheckerboard2.paramsCurrentLevel);

	//if desired, round the resulting movement...
	#if (ROUND_RESULTING_MOVE_VALS_SETTING == ROUND_RESULTING_MOVE_VALS)
		//allocate space for the final unrounded x and y movement
		float* unroundedXMoveDevice;
		float* unroundedYMoveDevice;

		cudaMalloc((void**)&unroundedXMoveDevice, widthLevel*heightLevel*sizeof(float));
		cudaMalloc((void**)&unroundedYMoveDevice, widthLevel*heightLevel*sizeof(float));

		//combine the x and y checkerboard movement into a single 2D array for output
		combineXYCheckerboardMovements(currentBpDeviceValsCheckerboard1.estimatedXMovement, currentBpDeviceValsCheckerboard1.estimatedYMovement,
										currentBpDeviceValsCheckerboard2.estimatedXMovement, currentBpDeviceValsCheckerboard2.estimatedYMovement,
										unroundedXMoveDevice, unroundedYMoveDevice, (int)floor(widthLevel), (int)floor(heightLevel));

		//run the function to round the movement on the device
		roundDeviceVals(unroundedXMoveDevice, movementXFromImage1To2Device, widthLevel, heightLevel);
		roundDeviceVals(unroundedYMoveDevice, movementYFromImage1To2Device, widthLevel, heightLevel);

		//now free the unrounded movement and set it to the rounded movement
		cudaFree(unroundedXMoveDevice);
		cudaFree(unroundedYMoveDevice);


	//if not rounding, then simply retrieve final movement
	#elif (ROUND_RESULTING_MOVE_VALS_SETTING == DONT_ROUND_RESULTING_MOVE_VALS)

		//combine the x and y checkerboard movement into a single 2D array for output
		combineXYCheckerboardMovements(currentBpDeviceValsCheckerboard1.estimatedXMovement, currentBpDeviceValsCheckerboard1.estimatedYMovement,
										currentBpDeviceValsCheckerboard2.estimatedXMovement, currentBpDeviceValsCheckerboard2.estimatedYMovement,
										movementXFromImage1To2Device, movementYFromImage1To2Device, (int)floor(widthLevel), (int)floor(heightLevel));

	#endif //ROUND_RESULTING_MOVE_VALS_SETTING

	//free the movement stored in checkerboards now that it has been combined
	cudaFree(currentBpDeviceValsCheckerboard1.estimatedXMovement);
	cudaFree(currentBpDeviceValsCheckerboard1.estimatedYMovement);

	cudaFree(currentBpDeviceValsCheckerboard2.estimatedXMovement);
	cudaFree(currentBpDeviceValsCheckerboard2.estimatedYMovement);

	//stop the timer and print the total running time to run belief propagation
	cutStopTimer(timer);
	printf("Total belief propagation running time: %f \n", cutGetTimerValue(timer));
}


//run the belief propagation algorithm on CUDA for motion estimation where the x and the y movements are computed where the two portions of the "checkerboard" are split
//the BP implementation is run such that the range is adjusted in each level in the hierarchy
//the width and height of each level are the same
void runBeliefPropMotionEstimationCUDAUseConstLevelSize(float* image1PixelsDevice, float* image2PixelsDevice, float* movementXFromImage1To2Device, float* movementYFromImage1To2Device, currBeliefPropParams currentBeliefPropParams)
{
	//declare and start the timer
	unsigned int timer;
	cutCreateTimer(&timer);
	cutResetTimer(timer);

	cutStartTimer(timer);

	//define structure containing the current device for each checkerboard
	bpCurrentValsCheckerboard currentBpDeviceValsCheckerboard1;
	bpCurrentValsCheckerboard currentBpDeviceValsCheckerboard2;

	//set the number of possible movements (assume this is the same at every level...)
	int totalNumPossMovements = (currentBeliefPropParams.totalNumMovesXDir*currentBeliefPropParams.totalNumMovesYDir);

	//declare the float for the width and height of the current level
	float widthLevel = currentBeliefPropParams.widthImages;
	float heightLevel = currentBeliefPropParams.heightImages;

	int widthCheckerboardLevel = (int)floor(widthLevel) / 2;
	int heightCheckerboardLevel = (int)floor(heightLevel);

	//initialize the movement increment at the current level
	float currentLevMoveIncX = currentBeliefPropParams.currentMoveIncrementX;
	float currentLevMoveIncY = currentBeliefPropParams.currentMoveIncrementY;

	//retrieve the increment in the movement at the "last" level
	for (int numLevel = 0; numLevel < (currentBeliefPropParams.numBpLevels - 1); numLevel++)
	{ 
		//update the movement increment
		currentLevMoveIncX *= currentBeliefPropParams.propChangeMoveNextLevel;
		currentLevMoveIncY *= currentBeliefPropParams.propChangeMoveNextLevel;
	}

	//set the movement increment at the "bottom" level
	currentBeliefPropParams.motionIncBotLevX = currentLevMoveIncX;
	currentBeliefPropParams.motionIncBotLevY = currentLevMoveIncY;
	
	//allocate the space for the movement parameters on the device
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.paramsCurrentLevel), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(currentStartMoveParamsPixelAtLevel));
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.paramsCurrentLevel), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(currentStartMoveParamsPixelAtLevel));

	//initialize the start movement parameters at the top level
	initializeParamsAndMovementOnDevice(currentBpDeviceValsCheckerboard1.paramsCurrentLevel, currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
										(int)floor(widthLevel), (int)floor(heightLevel), currentBeliefPropParams);

	//retrieve the number of bytes needed for the data and messages in the checkerboard in each level
	size_t numBytesDataAndMessagesEachLevel = widthCheckerboardLevel * heightCheckerboardLevel * totalNumPossMovements * sizeof(float);


	//allocate the space of the messages on the device
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageUDevice), numBytesDataAndMessagesEachLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageDDevice), numBytesDataAndMessagesEachLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageLDevice), numBytesDataAndMessagesEachLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageRDevice), numBytesDataAndMessagesEachLevel);

	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageUDevice), numBytesDataAndMessagesEachLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageDDevice), numBytesDataAndMessagesEachLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageLDevice), numBytesDataAndMessagesEachLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageRDevice), numBytesDataAndMessagesEachLevel);

	//initialize the message values at the top level
	initializeMessageVals(currentBpDeviceValsCheckerboard1.checkerboardMessVals, currentBpDeviceValsCheckerboard2.checkerboardMessVals,
						(int)floor(widthLevel), (int)floor(heightLevel));


	//run BP at each level
	for (int numLevel = (currentBeliefPropParams.numBpLevels - 1); numLevel >= 0; numLevel--)
	{
		//allocate the space for the data costs at the current level
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.dataCostsVals), numBytesDataAndMessagesEachLevel);
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.dataCostsVals), numBytesDataAndMessagesEachLevel);

		//set the data costs at the current level
		initializeDataCostsCurrentLevel(image1PixelsDevice, image2PixelsDevice,
										currentBpDeviceValsCheckerboard1.dataCostsVals, currentBpDeviceValsCheckerboard2.dataCostsVals,
										currentBpDeviceValsCheckerboard1.paramsCurrentLevel, currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
										currentBeliefPropParams, (int)floor(widthLevel), (int)floor(heightLevel));


		//don't run BP beyond the lowest level
		//if (numLevel == algSettings.numLevels - 1)
		{
			//run BP at the current level
			runBPAtCurrentLevel(currentBpDeviceValsCheckerboard1.checkerboardMessVals, currentBpDeviceValsCheckerboard2.checkerboardMessVals, 
								currentBpDeviceValsCheckerboard1.dataCostsVals, currentBpDeviceValsCheckerboard2.dataCostsVals,
								currentBpDeviceValsCheckerboard1.paramsCurrentLevel, currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
								currentBeliefPropParams.numBpIterations, (int)floor(widthLevel), (int)floor(heightLevel), 
								numBytesDataAndMessagesEachLevel);
		}


		//allocate the space for the estimated movement at the current level
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.estimatedXMovement), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(float));
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.estimatedYMovement), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(float));

		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.estimatedXMovement), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(float));
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.estimatedYMovement), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(float));

		retrieveBestMotionEstLevel(currentBpDeviceValsCheckerboard1.estimatedXMovement, currentBpDeviceValsCheckerboard1.estimatedYMovement,
									currentBpDeviceValsCheckerboard2.estimatedXMovement, currentBpDeviceValsCheckerboard2.estimatedYMovement,
									currentBpDeviceValsCheckerboard1.checkerboardMessVals,
									currentBpDeviceValsCheckerboard2.checkerboardMessVals,
									currentBpDeviceValsCheckerboard1.dataCostsVals,
									currentBpDeviceValsCheckerboard2.dataCostsVals,
									currentBpDeviceValsCheckerboard1.paramsCurrentLevel,
									currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
									numBytesDataAndMessagesEachLevel, (int)floor(widthLevel), (int)floor(heightLevel));

		//free the device memory allocated to the data costs
		cudaFree(currentBpDeviceValsCheckerboard1.dataCostsVals);
		cudaFree(currentBpDeviceValsCheckerboard2.dataCostsVals);

		//copy the current message values to the next level and update the parameters if not in bottom level
		if (numLevel > 0)
		{

			checkerboardMessagesDeviceStruct messageDeviceCheckerboard1CopyTo;
			checkerboardMessagesDeviceStruct messageDeviceCheckerboard2CopyTo;

			cudaMalloc((void**)&(messageDeviceCheckerboard1CopyTo.messageUDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard1CopyTo.messageDDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard1CopyTo.messageLDevice), numBytesDataAndMessagesEachLevel);

			cudaMalloc((void**)&(messageDeviceCheckerboard1CopyTo.messageRDevice), numBytesDataAndMessagesEachLevel);

			cudaMalloc((void**)&(messageDeviceCheckerboard2CopyTo.messageUDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard2CopyTo.messageDDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard2CopyTo.messageLDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard2CopyTo.messageRDevice), numBytesDataAndMessagesEachLevel);

			//declare the movement parameters for the "next" level and allocate space for them
			currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboard1;
			currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboard2;

			cudaMalloc((void**)&paramsNextLevelDeviceCheckerboard1, widthCheckerboardLevel * heightCheckerboardLevel * sizeof(currentStartMoveParamsPixelAtLevel));
			cudaMalloc((void**)&paramsNextLevelDeviceCheckerboard2, widthCheckerboardLevel * heightCheckerboardLevel * sizeof(currentStartMoveParamsPixelAtLevel));


			copyMessageValuesToNextLevelDownLevelSizeConst(
											currentBpDeviceValsCheckerboard1.checkerboardMessVals, 
											currentBpDeviceValsCheckerboard2.checkerboardMessVals, 
											currentBpDeviceValsCheckerboard1.paramsCurrentLevel,
											currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
											paramsNextLevelDeviceCheckerboard1,
											paramsNextLevelDeviceCheckerboard2,
											currentBpDeviceValsCheckerboard1.estimatedXMovement, currentBpDeviceValsCheckerboard1.estimatedYMovement,
											currentBpDeviceValsCheckerboard2.estimatedXMovement, currentBpDeviceValsCheckerboard2.estimatedYMovement,
											messageDeviceCheckerboard1CopyTo,
											messageDeviceCheckerboard2CopyTo,
											(int)floor(widthLevel), (int)floor(heightLevel),
											numBytesDataAndMessagesEachLevel);

			cudaThreadSynchronize();

			//free the device memory allocated to the previous message values now that they've been copied
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageUDevice);
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageDDevice);
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageLDevice);
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageRDevice);

			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageUDevice);
			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageDDevice);
			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageLDevice);
			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageRDevice);

			//free the device memory allocated to the parameters at the previous level
			cudaFree(currentBpDeviceValsCheckerboard1.paramsCurrentLevel);
			cudaFree(currentBpDeviceValsCheckerboard2.paramsCurrentLevel);


			//set the checkerboard device messages to the messages that were copied to
			currentBpDeviceValsCheckerboard1.checkerboardMessVals = messageDeviceCheckerboard1CopyTo;
			currentBpDeviceValsCheckerboard2.checkerboardMessVals = messageDeviceCheckerboard2CopyTo;

			//set the "current level" parameter to the set "next" level parameters
			currentBpDeviceValsCheckerboard1.paramsCurrentLevel = paramsNextLevelDeviceCheckerboard1;
			currentBpDeviceValsCheckerboard2.paramsCurrentLevel = paramsNextLevelDeviceCheckerboard2;


			//free the device memory allocated to the calculated movement at the previous level
			cudaFree(currentBpDeviceValsCheckerboard1.estimatedXMovement);
			cudaFree(currentBpDeviceValsCheckerboard1.estimatedYMovement);
			cudaFree(currentBpDeviceValsCheckerboard2.estimatedXMovement);
			cudaFree(currentBpDeviceValsCheckerboard2.estimatedYMovement);

			//adjust the movement for all pixels at the "next" level
			adjustMovementAllPixParams(currentBeliefPropParams, (int)floor(widthLevel), (int)floor(heightLevel));
		}
	}

	//free device memory used for storing the message values and movement parameters
	cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageUDevice);
	cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageDDevice);
	cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageLDevice);
	cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageRDevice);

	cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageUDevice);
	cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageDDevice);
	cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageLDevice);
	cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageRDevice);

	cudaFree(currentBpDeviceValsCheckerboard1.paramsCurrentLevel);
	cudaFree(currentBpDeviceValsCheckerboard2.paramsCurrentLevel);

	//if desired, round the resulting movement...
	#if (ROUND_RESULTING_MOVE_VALS_SETTING == ROUND_RESULTING_MOVE_VALS)
		//allocate space for the final unrounded x and y movement
		float* unroundedXMoveDevice;
		float* unroundedYMoveDevice;

		cudaMalloc((void**)&unroundedXMoveDevice, widthLevel*heightLevel*sizeof(float));
		cudaMalloc((void**)&unroundedYMoveDevice, widthLevel*heightLevel*sizeof(float));

		//combine the x and y checkerboard movement into a single 2D array for output
		combineXYCheckerboardMovements(currentBpDeviceValsCheckerboard1.estimatedXMovement, currentBpDeviceValsCheckerboard1.estimatedYMovement,
										currentBpDeviceValsCheckerboard2.estimatedXMovement, currentBpDeviceValsCheckerboard2.estimatedYMovement,
										unroundedXMoveDevice, unroundedYMoveDevice, (int)floor(widthLevel), (int)floor(heightLevel));

		//run the function to round the movement on the device
		roundDeviceVals(unroundedXMoveDevice, movementXFromImage1To2Device, widthLevel, heightLevel);
		roundDeviceVals(unroundedYMoveDevice, movementYFromImage1To2Device, widthLevel, heightLevel);

		//now free the unrounded movement and set it to the rounded movement
		cudaFree(unroundedXMoveDevice);
		cudaFree(unroundedYMoveDevice);


	//if not rounding, then simply retrieve final movement
	#elif (ROUND_RESULTING_MOVE_VALS_SETTING == DONT_ROUND_RESULTING_MOVE_VALS)

		//combine the x and y checkerboard movement into a single 2D array for output
		combineXYCheckerboardMovements(currentBpDeviceValsCheckerboard1.estimatedXMovement, currentBpDeviceValsCheckerboard1.estimatedYMovement,
										currentBpDeviceValsCheckerboard2.estimatedXMovement, currentBpDeviceValsCheckerboard2.estimatedYMovement,
										movementXFromImage1To2Device, movementYFromImage1To2Device, (int)floor(widthLevel), (int)floor(heightLevel));

	#endif //ROUND_RESULTING_MOVE_VALS_SETTING

	//free the movement stored in checkerboards now that it has been combined
	cudaFree(currentBpDeviceValsCheckerboard1.estimatedXMovement);
	cudaFree(currentBpDeviceValsCheckerboard1.estimatedYMovement);

	cudaFree(currentBpDeviceValsCheckerboard2.estimatedXMovement);
	cudaFree(currentBpDeviceValsCheckerboard2.estimatedYMovement);

	//stop the timer and print the total running time to run belief propagation
	cutStopTimer(timer);
	printf("Total belief propagation running time: %f \n", cutGetTimerValue(timer));
}


//run the belief propagation algorithm on CUDA for motion estimation where the x and the y movements are computed where the two portions of the "checkerboard" are split
//the BP implementation is run such that the range is adjusted in each level in the hierarchy
//the width and height of each level are the same
//prevAndCurrMovementXFromImage1To2Device and prevAndCurrMovementYFromImage1To2Device represents both the `input movement' from the previous iteration and the `current movement' in the output of the current iteration...
void runBeliefPropMotionEstimationCUDAUseConstLevelSizeUseEstMovement(float* image1PixelsDevice, float* image2PixelsDevice, float* prevAndCurrMovementXFromImage1To2Device, float* prevAndCurrMovementYFromImage1To2Device, currBeliefPropParams currentBeliefPropParams)
{
	//declare and start the timer
	unsigned int timer;
	cutCreateTimer(&timer);
	cutResetTimer(timer);

	cutStartTimer(timer);

	//initialize and `malloc' the space for the expected movements on the device...
	float* expectedMovementXDirDevice;
	float* expectedMovementYDirDevice;

	//`malloc' the space for the expected movements on the device...
	cudaMalloc((void**)&expectedMovementXDirDevice, (int)floor(currentBeliefPropParams.widthImages) * (int)floor(currentBeliefPropParams.heightImages) * sizeof(float));
	cudaMalloc((void**)&expectedMovementYDirDevice, (int)floor(currentBeliefPropParams.widthImages) * (int)floor(currentBeliefPropParams.heightImages) * sizeof(float));

	//generate the `expected movement' in the x and y directions using the movement in the previous images
	genEstMoveNextImSet(prevAndCurrMovementXFromImage1To2Device, prevAndCurrMovementYFromImage1To2Device,
			expectedMovementXDirDevice, expectedMovementYDirDevice, 
			(int)floor(currentBeliefPropParams.widthImages), (int)floor(currentBeliefPropParams.heightImages));

	//define structure containing the current device for each checkerboard
	bpCurrentValsCheckerboard currentBpDeviceValsCheckerboard1;
	bpCurrentValsCheckerboard currentBpDeviceValsCheckerboard2;

	//set the number of possible movements (assume this is the same at every level...)
	int totalNumPossMovements = (currentBeliefPropParams.totalNumMovesXDir*currentBeliefPropParams.totalNumMovesYDir);

	//declare the float for the width and height of the current level
	float widthLevel = currentBeliefPropParams.widthImages;
	float heightLevel = currentBeliefPropParams.heightImages;

	int widthCheckerboardLevel = (int)floor(widthLevel) / 2;
	int heightCheckerboardLevel = (int)floor(heightLevel);

	//initialize the movement increment at the current level
	float currentLevMoveIncX = currentBeliefPropParams.currentMoveIncrementX;
	float currentLevMoveIncY = currentBeliefPropParams.currentMoveIncrementY;

	//retrieve the increment in the movement at the "last" level
	for (int numLevel = 0; numLevel < (currentBeliefPropParams.numBpLevels - 1); numLevel++)
	{ 
		//update the movement increment
		currentLevMoveIncX *= currentBeliefPropParams.propChangeMoveNextLevel;
		currentLevMoveIncY *= currentBeliefPropParams.propChangeMoveNextLevel;
	}

	//set the movement increment at the "bottom" level
	currentBeliefPropParams.motionIncBotLevX = currentLevMoveIncX;
	currentBeliefPropParams.motionIncBotLevY = currentLevMoveIncY;
	
	//allocate the space for the movement parameters on the device
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.paramsCurrentLevel), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(currentStartMoveParamsPixelAtLevel));
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.paramsCurrentLevel), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(currentStartMoveParamsPixelAtLevel));

	//initialize the start movement parameters at the top level
	initializeParamsAndMovementOnDevice(currentBpDeviceValsCheckerboard1.paramsCurrentLevel, currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
										(int)floor(widthLevel), (int)floor(heightLevel), currentBeliefPropParams);

	//retrieve the number of bytes needed for the data and messages in the checkerboard in each level
	size_t numBytesDataAndMessagesEachLevel = widthCheckerboardLevel * heightCheckerboardLevel * totalNumPossMovements * sizeof(float);


	//allocate the space of the messages on the device
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageUDevice), numBytesDataAndMessagesEachLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageDDevice), numBytesDataAndMessagesEachLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageLDevice), numBytesDataAndMessagesEachLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageRDevice), numBytesDataAndMessagesEachLevel);

	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageUDevice), numBytesDataAndMessagesEachLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageDDevice), numBytesDataAndMessagesEachLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageLDevice), numBytesDataAndMessagesEachLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageRDevice), numBytesDataAndMessagesEachLevel);

	//initialize the message values at the top level
	initializeMessageVals(currentBpDeviceValsCheckerboard1.checkerboardMessVals, currentBpDeviceValsCheckerboard2.checkerboardMessVals,
						(int)floor(widthLevel), (int)floor(heightLevel));


	//run BP at each level
	for (int numLevel = (currentBeliefPropParams.numBpLevels - 1); numLevel >= 0; numLevel--)
	{
		//allocate the space for the data costs at the current level
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.dataCostsVals), numBytesDataAndMessagesEachLevel);
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.dataCostsVals), numBytesDataAndMessagesEachLevel);

		

		
		//if estimated moves is not 'null', then generate the movement costs and incorporate it as part of the implementation
		if ((prevAndCurrMovementXFromImage1To2Device != NULL) && (prevAndCurrMovementYFromImage1To2Device != NULL))
		{
			//printf("initDataEstMove\n");

			//initialize the estimated movement costs and add it to the data costs in the overall computation...
			/*initEstMovementInDataCosts(expectedMovementXDirDevice, expectedMovementYDirDevice, 
					currentBpDeviceValsCheckerboard1.dataCostsVals, currentBpDeviceValsCheckerboard2.dataCostsVals,
					currentBpDeviceValsCheckerboard1.paramsCurrentLevel, 
					currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
					currentBeliefPropParams,
					(int)floor(widthLevel/2), (int)floor(heightLevel),
					(int)floor(widthLevel), (int)floor(heightLevel),
					numBytesDataAndMessagesEachLevel);*/

			initializeDataAndEstMoveCostsCurrentLevel(image1PixelsDevice, image2PixelsDevice,
										currentBpDeviceValsCheckerboard1.dataCostsVals, currentBpDeviceValsCheckerboard2.dataCostsVals,
										currentBpDeviceValsCheckerboard1.paramsCurrentLevel, currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
										currentBeliefPropParams, (int)floor(widthLevel), (int)floor(heightLevel), expectedMovementXDirDevice, expectedMovementYDirDevice);
		}
		else
		{
			//set the data costs at the current level
			initializeDataCostsCurrentLevel(image1PixelsDevice, image2PixelsDevice,
										currentBpDeviceValsCheckerboard1.dataCostsVals, currentBpDeviceValsCheckerboard2.dataCostsVals,
										currentBpDeviceValsCheckerboard1.paramsCurrentLevel, currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
										currentBeliefPropParams, (int)floor(widthLevel), (int)floor(heightLevel));
		}


		//don't run BP beyond the lowest level
		//if (numLevel == algSettings.numLevels - 1)
		{
			//run BP at the current level
			runBPAtCurrentLevel(currentBpDeviceValsCheckerboard1.checkerboardMessVals, currentBpDeviceValsCheckerboard2.checkerboardMessVals, 
								currentBpDeviceValsCheckerboard1.dataCostsVals, currentBpDeviceValsCheckerboard2.dataCostsVals,
								currentBpDeviceValsCheckerboard1.paramsCurrentLevel, currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
								currentBeliefPropParams.numBpIterations, (int)floor(widthLevel), (int)floor(heightLevel), 
								numBytesDataAndMessagesEachLevel);
		}


		//allocate the space for the estimated movement at the current level
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.estimatedXMovement), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(float));
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.estimatedYMovement), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(float));

		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.estimatedXMovement), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(float));
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.estimatedYMovement), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(float));

		retrieveBestMotionEstLevel(currentBpDeviceValsCheckerboard1.estimatedXMovement, currentBpDeviceValsCheckerboard1.estimatedYMovement,
									currentBpDeviceValsCheckerboard2.estimatedXMovement, currentBpDeviceValsCheckerboard2.estimatedYMovement,
									currentBpDeviceValsCheckerboard1.checkerboardMessVals,
									currentBpDeviceValsCheckerboard2.checkerboardMessVals,
									currentBpDeviceValsCheckerboard1.dataCostsVals,
									currentBpDeviceValsCheckerboard2.dataCostsVals,
									currentBpDeviceValsCheckerboard1.paramsCurrentLevel,
									currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
									numBytesDataAndMessagesEachLevel, (int)floor(widthLevel), (int)floor(heightLevel));

		//free the device memory allocated to the data costs
		cudaFree(currentBpDeviceValsCheckerboard1.dataCostsVals);
		cudaFree(currentBpDeviceValsCheckerboard2.dataCostsVals);

		//copy the current message values to the next level and update the parameters if not in bottom level
		if (numLevel > 0)
		{

			checkerboardMessagesDeviceStruct messageDeviceCheckerboard1CopyTo;
			checkerboardMessagesDeviceStruct messageDeviceCheckerboard2CopyTo;

			cudaMalloc((void**)&(messageDeviceCheckerboard1CopyTo.messageUDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard1CopyTo.messageDDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard1CopyTo.messageLDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard1CopyTo.messageRDevice), numBytesDataAndMessagesEachLevel);

			cudaMalloc((void**)&(messageDeviceCheckerboard2CopyTo.messageUDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard2CopyTo.messageDDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard2CopyTo.messageLDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard2CopyTo.messageRDevice), numBytesDataAndMessagesEachLevel);

			//declare the movement parameters for the "next" level and allocate space for them
			currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboard1;
			currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboard2;

			cudaMalloc((void**)&paramsNextLevelDeviceCheckerboard1, widthCheckerboardLevel * heightCheckerboardLevel * sizeof(currentStartMoveParamsPixelAtLevel));
			cudaMalloc((void**)&paramsNextLevelDeviceCheckerboard2, widthCheckerboardLevel * heightCheckerboardLevel * sizeof(currentStartMoveParamsPixelAtLevel));


			copyMessageValuesToNextLevelDownLevelSizeConst(
											currentBpDeviceValsCheckerboard1.checkerboardMessVals, 
											currentBpDeviceValsCheckerboard2.checkerboardMessVals, 

											currentBpDeviceValsCheckerboard1.paramsCurrentLevel,
											currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
											paramsNextLevelDeviceCheckerboard1,
											paramsNextLevelDeviceCheckerboard2,
											currentBpDeviceValsCheckerboard1.estimatedXMovement, currentBpDeviceValsCheckerboard1.estimatedYMovement,
											currentBpDeviceValsCheckerboard2.estimatedXMovement, currentBpDeviceValsCheckerboard2.estimatedYMovement,
											messageDeviceCheckerboard1CopyTo,
											messageDeviceCheckerboard2CopyTo,
											(int)floor(widthLevel), (int)floor(heightLevel),
											numBytesDataAndMessagesEachLevel);

			cudaThreadSynchronize();

			//free the device memory allocated to the previous message values now that they've been copied
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageUDevice);
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageDDevice);
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageLDevice);
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageRDevice);

			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageUDevice);
			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageDDevice);
			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageLDevice);
			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageRDevice);

			//free the device memory allocated to the parameters at the previous level
			cudaFree(currentBpDeviceValsCheckerboard1.paramsCurrentLevel);
			cudaFree(currentBpDeviceValsCheckerboard2.paramsCurrentLevel);


			//set the checkerboard device messages to the messages that were copied to
			currentBpDeviceValsCheckerboard1.checkerboardMessVals = messageDeviceCheckerboard1CopyTo;
			currentBpDeviceValsCheckerboard2.checkerboardMessVals = messageDeviceCheckerboard2CopyTo;

			//set the "current level" parameter to the set "next" level parameters
			currentBpDeviceValsCheckerboard1.paramsCurrentLevel = paramsNextLevelDeviceCheckerboard1;
			currentBpDeviceValsCheckerboard2.paramsCurrentLevel = paramsNextLevelDeviceCheckerboard2;


			//free the device memory allocated to the calculated movement at the previous level
			cudaFree(currentBpDeviceValsCheckerboard1.estimatedXMovement);
			cudaFree(currentBpDeviceValsCheckerboard1.estimatedYMovement);
			cudaFree(currentBpDeviceValsCheckerboard2.estimatedXMovement);
			cudaFree(currentBpDeviceValsCheckerboard2.estimatedYMovement);

			//adjust the movement for all pixels at the "next" level
			adjustMovementAllPixParams(currentBeliefPropParams, (int)floor(widthLevel), (int)floor(heightLevel));
		}
	}

	//free the 'estimated movement' on the device...
	cudaFree(expectedMovementXDirDevice);
	cudaFree(expectedMovementYDirDevice);

	//free device memory used for storing the message values and movement parameters
	cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageUDevice);
	cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageDDevice);
	cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageLDevice);
	cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageRDevice);

	cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageUDevice);
	cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageDDevice);
	cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageLDevice);
	cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageRDevice);

	cudaFree(currentBpDeviceValsCheckerboard1.paramsCurrentLevel);
	cudaFree(currentBpDeviceValsCheckerboard2.paramsCurrentLevel);

	//if desired, round the resulting movement...
	#if (ROUND_RESULTING_MOVE_VALS_SETTING == ROUND_RESULTING_MOVE_VALS)
		//allocate space for the final unrounded x and y movement
		float* unroundedXMoveDevice;
		float* unroundedYMoveDevice;

		cudaMalloc((void**)&unroundedXMoveDevice, widthLevel*heightLevel*sizeof(float));
		cudaMalloc((void**)&unroundedYMoveDevice, widthLevel*heightLevel*sizeof(float));

		//combine the x and y checkerboard movement into a single 2D array for output
		combineXYCheckerboardMovements(currentBpDeviceValsCheckerboard1.estimatedXMovement, currentBpDeviceValsCheckerboard1.estimatedYMovement,
										currentBpDeviceValsCheckerboard2.estimatedXMovement, currentBpDeviceValsCheckerboard2.estimatedYMovement,
										unroundedXMoveDevice, unroundedYMoveDevice, (int)floor(widthLevel), (int)floor(heightLevel));

		//run the function to round the movement on the device
		roundDeviceVals(unroundedXMoveDevice, movementXFromImage1To2Device, widthLevel, heightLevel);
		roundDeviceVals(unroundedYMoveDevice, movementYFromImage1To2Device, widthLevel, heightLevel);

		//now free the unrounded movement and set it to the rounded movement
		cudaFree(unroundedXMoveDevice);
		cudaFree(unroundedYMoveDevice);


	//if not rounding, then simply retrieve final movement
	#elif (ROUND_RESULTING_MOVE_VALS_SETTING == DONT_ROUND_RESULTING_MOVE_VALS)

		//combine the x and y checkerboard movement into a single 2D array for output
		combineXYCheckerboardMovements(currentBpDeviceValsCheckerboard1.estimatedXMovement, currentBpDeviceValsCheckerboard1.estimatedYMovement,
										currentBpDeviceValsCheckerboard2.estimatedXMovement, currentBpDeviceValsCheckerboard2.estimatedYMovement,
										prevAndCurrMovementXFromImage1To2Device, prevAndCurrMovementYFromImage1To2Device, (int)floor(widthLevel), (int)floor(heightLevel));

	#endif //ROUND_RESULTING_MOVE_VALS_SETTING

	


	//free the movement stored in checkerboards now that it has been combined
	cudaFree(currentBpDeviceValsCheckerboard1.estimatedXMovement);
	cudaFree(currentBpDeviceValsCheckerboard1.estimatedYMovement);

	cudaFree(currentBpDeviceValsCheckerboard2.estimatedXMovement);
	cudaFree(currentBpDeviceValsCheckerboard2.estimatedYMovement);
	

	//stop the timer and print the total running time to run belief propagation
	cutStopTimer(timer);
	printf("Total belief propagation running time: %f \n", cutGetTimerValue(timer));
}


//run the belief propagation algorithm on CUDA for motion estimation where the x and the y movements are computed where the two portions of the "checkerboard" are split
//the BP implementation is run such that the range is adjusted in each level in the hierarchy
//the width and height of each level are the same
//the estimated movement between the two images is given as an input parameter...
void runBeliefPropMotionEstimationCUDAUseConstLevelSizeGivenEstMovement(float* image1PixelsDevice, float* image2PixelsDevice, float* currMovementXFromImage1To2Device, float* currMovementYFromImage1To2Device, currBeliefPropParams currentBeliefPropParams, float* expectedMovementXDirHost, float* expectedMovementYDirHost)
{
	//declare and start the timer
	unsigned int timer;
	cutCreateTimer(&timer);
	cutResetTimer(timer);

	cutStartTimer(timer);

	//initialize and `malloc' the space for the expected movements on the device...
	float* expectedMovementXDirDevice;
	float* expectedMovementYDirDevice;

	//`malloc' the space for the expected movements on the device...
	cudaMalloc((void**)&expectedMovementXDirDevice, (int)floor(currentBeliefPropParams.widthImages) * (int)floor(currentBeliefPropParams.heightImages) * sizeof(float));
	cudaMalloc((void**)&expectedMovementYDirDevice, (int)floor(currentBeliefPropParams.widthImages) * (int)floor(currentBeliefPropParams.heightImages) * sizeof(float));

	//transfer the estimated movement from the host to the device...
	cudaMemcpy(expectedMovementXDirDevice, expectedMovementXDirHost, (int)floor(currentBeliefPropParams.widthImages) * (int)floor(currentBeliefPropParams.heightImages) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(expectedMovementYDirDevice, expectedMovementYDirHost, (int)floor(currentBeliefPropParams.widthImages) * (int)floor(currentBeliefPropParams.heightImages) * sizeof(float), cudaMemcpyHostToDevice);


	//define structure containing the current device for each checkerboard
	bpCurrentValsCheckerboard currentBpDeviceValsCheckerboard1;
	bpCurrentValsCheckerboard currentBpDeviceValsCheckerboard2;

	//set the number of possible movements (assume this is the same at every level...)
	int totalNumPossMovements = (currentBeliefPropParams.totalNumMovesXDir*currentBeliefPropParams.totalNumMovesYDir);

	//declare the float for the width and height of the current level
	float widthLevel = currentBeliefPropParams.widthImages;
	float heightLevel = currentBeliefPropParams.heightImages;

	int widthCheckerboardLevel = (int)floor(widthLevel) / 2;
	int heightCheckerboardLevel = (int)floor(heightLevel);

	//initialize the movement increment at the current level
	float currentLevMoveIncX = currentBeliefPropParams.currentMoveIncrementX;
	float currentLevMoveIncY = currentBeliefPropParams.currentMoveIncrementY;

	//retrieve the increment in the movement at the "last" level
	for (int numLevel = 0; numLevel < (currentBeliefPropParams.numBpLevels - 1); numLevel++)
	{ 
		//update the movement increment
		currentLevMoveIncX *= currentBeliefPropParams.propChangeMoveNextLevel;
		currentLevMoveIncY *= currentBeliefPropParams.propChangeMoveNextLevel;
	}

	//set the movement increment at the "bottom" level
	currentBeliefPropParams.motionIncBotLevX = currentLevMoveIncX;
	currentBeliefPropParams.motionIncBotLevY = currentLevMoveIncY;
	
	//allocate the space for the movement parameters on the device
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.paramsCurrentLevel), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(currentStartMoveParamsPixelAtLevel));
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.paramsCurrentLevel), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(currentStartMoveParamsPixelAtLevel));

	//initialize the start movement parameters at the top level
	initializeParamsAndMovementOnDevice(currentBpDeviceValsCheckerboard1.paramsCurrentLevel, currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
										(int)floor(widthLevel), (int)floor(heightLevel), currentBeliefPropParams);

	//retrieve the number of bytes needed for the data and messages in the checkerboard in each level
	size_t numBytesDataAndMessagesEachLevel = widthCheckerboardLevel * heightCheckerboardLevel * totalNumPossMovements * sizeof(float);


	//allocate the space of the messages on the device
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageUDevice), numBytesDataAndMessagesEachLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageDDevice), numBytesDataAndMessagesEachLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageLDevice), numBytesDataAndMessagesEachLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageRDevice), numBytesDataAndMessagesEachLevel);

	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageUDevice), numBytesDataAndMessagesEachLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageDDevice), numBytesDataAndMessagesEachLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageLDevice), numBytesDataAndMessagesEachLevel);
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageRDevice), numBytesDataAndMessagesEachLevel);

	//initialize the message values at the top level
	initializeMessageVals(currentBpDeviceValsCheckerboard1.checkerboardMessVals, currentBpDeviceValsCheckerboard2.checkerboardMessVals,
						(int)floor(widthLevel), (int)floor(heightLevel));


	//run BP at each level
	for (int numLevel = (currentBeliefPropParams.numBpLevels - 1); numLevel >= 0; numLevel--)
	{
		//allocate the space for the data costs at the current level
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.dataCostsVals), numBytesDataAndMessagesEachLevel);
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.dataCostsVals), numBytesDataAndMessagesEachLevel);

		//if estimated moves is not 'null', then generate the movement costs and incorporate it as part of the implementation
		//if ((prevAndCurrMovementXFromImage1To2Device != NULL) && (prevAndCurrMovementYFromImage1To2Device != NULL))
		{

			//initialize the estimated movement costs and add it to the data costs in the overall computation...
			/*initEstMovementInDataCosts(expectedMovementXDirDevice, expectedMovementYDirDevice, 
					currentBpDeviceValsCheckerboard1.dataCostsVals, currentBpDeviceValsCheckerboard2.dataCostsVals,
					currentBpDeviceValsCheckerboard1.paramsCurrentLevel, 
					currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
					currentBeliefPropParams,
					(int)floor(widthLevel/2), (int)floor(heightLevel),
					(int)floor(widthLevel), (int)floor(heightLevel),
					numBytesDataAndMessagesEachLevel);*/

			initializeDataAndEstMoveCostsCurrentLevel(image1PixelsDevice, image2PixelsDevice,
										currentBpDeviceValsCheckerboard1.dataCostsVals, currentBpDeviceValsCheckerboard2.dataCostsVals,
										currentBpDeviceValsCheckerboard1.paramsCurrentLevel, currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
										currentBeliefPropParams, (int)floor(widthLevel), (int)floor(heightLevel), expectedMovementXDirDevice, expectedMovementYDirDevice);
		}
		/*else
		{
			//set the data costs at the current level
			initializeDataCostsCurrentLevel(image1PixelsDevice, image2PixelsDevice,
										currentBpDeviceValsCheckerboard1.dataCostsVals, currentBpDeviceValsCheckerboard2.dataCostsVals,
										currentBpDeviceValsCheckerboard1.paramsCurrentLevel, currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
										currentBeliefPropParams, (int)floor(widthLevel), (int)floor(heightLevel));
		}*/



		//don't run BP beyond the lowest level
		//if (numLevel == algSettings.numLevels - 1)
		{
			//run BP at the current level
			runBPAtCurrentLevel(currentBpDeviceValsCheckerboard1.checkerboardMessVals, currentBpDeviceValsCheckerboard2.checkerboardMessVals, 
								currentBpDeviceValsCheckerboard1.dataCostsVals, currentBpDeviceValsCheckerboard2.dataCostsVals,
								currentBpDeviceValsCheckerboard1.paramsCurrentLevel, currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
								currentBeliefPropParams.numBpIterations, (int)floor(widthLevel), (int)floor(heightLevel), 
								numBytesDataAndMessagesEachLevel);
		}


		//allocate the space for the estimated movement at the current level
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.estimatedXMovement), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(float));
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.estimatedYMovement), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(float));

		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.estimatedXMovement), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(float));
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.estimatedYMovement), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(float));

		retrieveBestMotionEstLevel(currentBpDeviceValsCheckerboard1.estimatedXMovement, currentBpDeviceValsCheckerboard1.estimatedYMovement,
									currentBpDeviceValsCheckerboard2.estimatedXMovement, currentBpDeviceValsCheckerboard2.estimatedYMovement,
									currentBpDeviceValsCheckerboard1.checkerboardMessVals,
									currentBpDeviceValsCheckerboard2.checkerboardMessVals,
									currentBpDeviceValsCheckerboard1.dataCostsVals,
									currentBpDeviceValsCheckerboard2.dataCostsVals,
									currentBpDeviceValsCheckerboard1.paramsCurrentLevel,
									currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
									numBytesDataAndMessagesEachLevel, (int)floor(widthLevel), (int)floor(heightLevel));

		//free the device memory allocated to the data costs
		cudaFree(currentBpDeviceValsCheckerboard1.dataCostsVals);
		cudaFree(currentBpDeviceValsCheckerboard2.dataCostsVals);

		//copy the current message values to the next level and update the parameters if not in bottom level
		if (numLevel > 0)
		{

			checkerboardMessagesDeviceStruct messageDeviceCheckerboard1CopyTo;
			checkerboardMessagesDeviceStruct messageDeviceCheckerboard2CopyTo;

			cudaMalloc((void**)&(messageDeviceCheckerboard1CopyTo.messageUDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard1CopyTo.messageDDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard1CopyTo.messageLDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard1CopyTo.messageRDevice), numBytesDataAndMessagesEachLevel);

			cudaMalloc((void**)&(messageDeviceCheckerboard2CopyTo.messageUDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard2CopyTo.messageDDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard2CopyTo.messageLDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard2CopyTo.messageRDevice), numBytesDataAndMessagesEachLevel);

			//declare the movement parameters for the "next" level and allocate space for them
			currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboard1;
			currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboard2;

			cudaMalloc((void**)&paramsNextLevelDeviceCheckerboard1, widthCheckerboardLevel * heightCheckerboardLevel * sizeof(currentStartMoveParamsPixelAtLevel));
			cudaMalloc((void**)&paramsNextLevelDeviceCheckerboard2, widthCheckerboardLevel * heightCheckerboardLevel * sizeof(currentStartMoveParamsPixelAtLevel));


			copyMessageValuesToNextLevelDownLevelSizeConst(
											currentBpDeviceValsCheckerboard1.checkerboardMessVals, 
											currentBpDeviceValsCheckerboard2.checkerboardMessVals, 

											currentBpDeviceValsCheckerboard1.paramsCurrentLevel,
											currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
											paramsNextLevelDeviceCheckerboard1,
											paramsNextLevelDeviceCheckerboard2,
											currentBpDeviceValsCheckerboard1.estimatedXMovement, currentBpDeviceValsCheckerboard1.estimatedYMovement,
											currentBpDeviceValsCheckerboard2.estimatedXMovement, currentBpDeviceValsCheckerboard2.estimatedYMovement,
											messageDeviceCheckerboard1CopyTo,
											messageDeviceCheckerboard2CopyTo,
											(int)floor(widthLevel), (int)floor(heightLevel),
											numBytesDataAndMessagesEachLevel);

			cudaThreadSynchronize();

			//free the device memory allocated to the previous message values now that they've been copied
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageUDevice);
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageDDevice);
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageLDevice);
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageRDevice);

			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageUDevice);
			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageDDevice);
			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageLDevice);
			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageRDevice);

			//free the device memory allocated to the parameters at the previous level
			cudaFree(currentBpDeviceValsCheckerboard1.paramsCurrentLevel);
			cudaFree(currentBpDeviceValsCheckerboard2.paramsCurrentLevel);


			//set the checkerboard device messages to the messages that were copied to
			currentBpDeviceValsCheckerboard1.checkerboardMessVals = messageDeviceCheckerboard1CopyTo;
			currentBpDeviceValsCheckerboard2.checkerboardMessVals = messageDeviceCheckerboard2CopyTo;

			//set the "current level" parameter to the set "next" level parameters
			currentBpDeviceValsCheckerboard1.paramsCurrentLevel = paramsNextLevelDeviceCheckerboard1;
			currentBpDeviceValsCheckerboard2.paramsCurrentLevel = paramsNextLevelDeviceCheckerboard2;


			//free the device memory allocated to the calculated movement at the previous level
			cudaFree(currentBpDeviceValsCheckerboard1.estimatedXMovement);
			cudaFree(currentBpDeviceValsCheckerboard1.estimatedYMovement);
			cudaFree(currentBpDeviceValsCheckerboard2.estimatedXMovement);
			cudaFree(currentBpDeviceValsCheckerboard2.estimatedYMovement);

			//adjust the movement for all pixels at the "next" level
			adjustMovementAllPixParams(currentBeliefPropParams, (int)floor(widthLevel), (int)floor(heightLevel));
		}
	}

	//free the 'estimated movement' on the device...
	cudaFree(expectedMovementXDirDevice);
	cudaFree(expectedMovementYDirDevice);

	//free device memory used for storing the message values and movement parameters
	cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageUDevice);
	cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageDDevice);
	cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageLDevice);
	cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageRDevice);

	cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageUDevice);
	cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageDDevice);
	cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageLDevice);
	cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageRDevice);

	cudaFree(currentBpDeviceValsCheckerboard1.paramsCurrentLevel);
	cudaFree(currentBpDeviceValsCheckerboard2.paramsCurrentLevel);

	//if desired, round the resulting movement...
	#if (ROUND_RESULTING_MOVE_VALS_SETTING == ROUND_RESULTING_MOVE_VALS)
		//allocate space for the final unrounded x and y movement
		float* unroundedXMoveDevice;
		float* unroundedYMoveDevice;

		cudaMalloc((void**)&unroundedXMoveDevice, widthLevel*heightLevel*sizeof(float));
		cudaMalloc((void**)&unroundedYMoveDevice, widthLevel*heightLevel*sizeof(float));

		//combine the x and y checkerboard movement into a single 2D array for output
		combineXYCheckerboardMovements(currentBpDeviceValsCheckerboard1.estimatedXMovement, currentBpDeviceValsCheckerboard1.estimatedYMovement,
										currentBpDeviceValsCheckerboard2.estimatedXMovement, currentBpDeviceValsCheckerboard2.estimatedYMovement,
										unroundedXMoveDevice, unroundedYMoveDevice, (int)floor(widthLevel), (int)floor(heightLevel));

		//run the function to round the movement on the device
		roundDeviceVals(unroundedXMoveDevice, movementXFromImage1To2Device, widthLevel, heightLevel);
		roundDeviceVals(unroundedYMoveDevice, movementYFromImage1To2Device, widthLevel, heightLevel);

		//now free the unrounded movement and set it to the rounded movement
		cudaFree(unroundedXMoveDevice);
		cudaFree(unroundedYMoveDevice);


	//if not rounding, then simply retrieve final movement
	#elif (ROUND_RESULTING_MOVE_VALS_SETTING == DONT_ROUND_RESULTING_MOVE_VALS)

		//combine the x and y checkerboard movement into a single 2D array for output
		combineXYCheckerboardMovements(currentBpDeviceValsCheckerboard1.estimatedXMovement, currentBpDeviceValsCheckerboard1.estimatedYMovement,
										currentBpDeviceValsCheckerboard2.estimatedXMovement, currentBpDeviceValsCheckerboard2.estimatedYMovement,
										currMovementXFromImage1To2Device, currMovementYFromImage1To2Device, (int)floor(widthLevel), (int)floor(heightLevel));

	#endif //ROUND_RESULTING_MOVE_VALS_SETTING

	


	//free the movement stored in checkerboards now that it has been combined
	cudaFree(currentBpDeviceValsCheckerboard1.estimatedXMovement);
	cudaFree(currentBpDeviceValsCheckerboard1.estimatedYMovement);

	cudaFree(currentBpDeviceValsCheckerboard2.estimatedXMovement);
	cudaFree(currentBpDeviceValsCheckerboard2.estimatedYMovement);
	

	//stop the timer and print the total running time to run belief propagation
	cutStopTimer(timer);
	printf("Total belief propagation running time: %f \n", cutGetTimerValue(timer));
}





//function to retrieve the total number of 'values' needed for the parameter hierarchy given the starting `checkerboard' width and height and the number of levels
int numValsParamHierarchy(int imageCheckerboardWidth, int imageCheckerboardHeight, int numLevelsInHierarch)
{
	//set the width of the checkerboard width/height to the values at the lowest levels...
	int currLevelImageCheckerboardWidth = imageCheckerboardWidth;
	int currLevelImageCheckerboardHeight = imageCheckerboardHeight;

	//initialize the total `checkerboard' values to the number of values in the `current checkerboard'
	int totNumParamVals = currLevelImageCheckerboardWidth * currLevelImageCheckerboardHeight;

	//now go through each level and set the offset and width/height...
	for (int currLevel = 1; currLevel < numLevelsInHierarch; currLevel++)
	{
		//adjust the values for the current level...
		currLevelImageCheckerboardWidth /= 2;
		currLevelImageCheckerboardHeight /= 2;
		
		//add the number of parameter values needed for the current level...
		totNumParamVals += currLevelImageCheckerboardWidth*currLevelImageCheckerboardHeight;
	}

	//return the generated parameter offsets and sizes...
	return totNumParamVals;
}

//function to retrieve the `offsets' and `size' at each level of the `parameters hierarchy'
paramOffsetsSizes* getOffsetsParamsHierarch(int imageCheckerboardWidth, int imageCheckerboardHeight, int numLevelsInHierarch)
{
	//allocate the space to store each of the parameters
	paramOffsetsSizes* currParamOffsetsSizes = (paramOffsetsSizes*)malloc(numLevelsInHierarch*sizeof(paramOffsetsSizes));

	//initialize the current image width and height and offset for the bottom `level'
	int currLevelImageCheckerboardWidth = imageCheckerboardWidth;
	int currLevelImageCheckerboardHeight = imageCheckerboardHeight;
	int currLevelParamOffset = 0;

	//set the stuff at the 'bottom level'
	currParamOffsetsSizes[0].offset = currLevelParamOffset;
	currParamOffsetsSizes[0].levelWidthCheckerboard = imageCheckerboardWidth;
	currParamOffsetsSizes[0].levelHeightCheckerboard = imageCheckerboardHeight;

	//now go through each level and set the offset and width/height...
	for (int currLevel = 1; currLevel < numLevelsInHierarch; currLevel++)
	{
		//adjust the values for the current level...
		currLevelParamOffset += currLevelImageCheckerboardWidth*currLevelImageCheckerboardHeight;
		currLevelImageCheckerboardWidth /= 2;
		currLevelImageCheckerboardHeight /= 2;
		
		//set the parameters at the current level...
		currParamOffsetsSizes[currLevel].offset = currLevelParamOffset;
		currParamOffsetsSizes[currLevel].levelWidthCheckerboard = currLevelImageCheckerboardWidth;
		currParamOffsetsSizes[currLevel].levelHeightCheckerboard = currLevelImageCheckerboardHeight;
	}

	//return the generated parameter offsets and sizes...
	return currParamOffsetsSizes;
}



//function to retrieve the message values at the `next' level given the values at the previous level...
void getMessValsNextLevel(checkerboardMessagesDeviceStruct messagesDevicePrevCurrCheckerboard1,
				checkerboardMessagesDeviceStruct messagesDevicePrevCurrCheckerboard2,
				checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboard1,
				checkerboardMessagesDeviceStruct messagesDeviceCurrentCheckerboard2,
				int widthPrevLevel, int heightPrevLevel, int widthNextLevel, int heightNextLevel,
				currentStartMoveParamsPixelAtLevel* paramsDeviceCheckerboard1,
				currentStartMoveParamsPixelAtLevel* paramsDeviceCheckerboard2, 
				size_t paramsOffsetPrevLev, size_t paramsOffsetCurrLevel)
{

	//from the perspective of the `previous level'...so use the width/height of the previous level when setting execution parameters...
	int widthOfCheckerboard = widthPrevLevel / 2;
	int heightOfCheckerboard = heightPrevLevel;

	//set the execution parameters for setting the parameters at the current level
	dim3 threadBlockDims(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 gridDims((unsigned int)ceil((float)widthOfCheckerboard / (float)threadBlockDims.x), (unsigned int)ceil((float)(heightOfCheckerboard) / (float)threadBlockDims.y));


	copyMessValsDownCompPyramid <<< gridDims, threadBlockDims >>> (messagesDevicePrevCurrCheckerboard1,
						messagesDeviceCurrentCheckerboard1,
						messagesDeviceCurrentCheckerboard2,
						widthPrevLevel, heightPrevLevel, widthNextLevel, heightNextLevel,
						paramsDeviceCheckerboard1,
						paramsDeviceCheckerboard2,
						paramsOffsetPrevLev, paramsOffsetCurrLevel, CHECKERBOARD_PART_1_ENUM);

	cudaThreadSynchronize();

	copyMessValsDownCompPyramid <<< gridDims, threadBlockDims >>> (messagesDevicePrevCurrCheckerboard2,
						messagesDeviceCurrentCheckerboard1,
						messagesDeviceCurrentCheckerboard2,
						widthPrevLevel, heightPrevLevel, widthNextLevel, heightNextLevel,
						paramsDeviceCheckerboard1,
						paramsDeviceCheckerboard2,
						paramsOffsetPrevLev, paramsOffsetCurrLevel, CHECKERBOARD_PART_2_ENUM);

	cudaThreadSynchronize();
}


//function to retrieve the parameters at each level in the hierarchy
//assuming that parameters at bottom level of hierarchy have been set...
void getParamsInHierarchy(paramOffsetsSizes* currParamInfo, currentStartMoveParamsPixelAtLevel* paramsEachHierarchyLevelDeviceCheckerboard1,
			currentStartMoveParamsPixelAtLevel* paramsEachHierarchyLevelDeviceCheckerboard2, int numLevelsInHierarch)
{
	//define the parameters for the grid/thread block...
	dim3 gridDims;
	dim3 threadBlockDims(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);

	 //now go through each level and set the parameters...
	for (int currLevel = 1; currLevel < numLevelsInHierarch; currLevel++)
	{
		//adjust the grid size depending on the level...
		gridDims.x = ceil(((float)currParamInfo[currLevel].levelWidthCheckerboard / (float)threadBlockDims.x));
		gridDims.y = ceil(((float)currParamInfo[currLevel].levelHeightCheckerboard  / (float)threadBlockDims.y));

		//call the kernel to set the parameters at the current level for each `checkerboard' portion...
		retParamsHigherLevPyramid <<< gridDims, threadBlockDims  >>> (paramsEachHierarchyLevelDeviceCheckerboard1, paramsEachHierarchyLevelDeviceCheckerboard2, currParamInfo[currLevel-1].levelWidthCheckerboard, currParamInfo[currLevel-1].levelHeightCheckerboard, currParamInfo[currLevel].levelWidthCheckerboard, currParamInfo[currLevel].levelHeightCheckerboard, currParamInfo[currLevel-1].offset, currParamInfo[currLevel].offset, CHECKERBOARD_PART_1_ENUM);
		
		cudaThreadSynchronize();

		retParamsHigherLevPyramid <<< gridDims, threadBlockDims  >>> (paramsEachHierarchyLevelDeviceCheckerboard1, paramsEachHierarchyLevelDeviceCheckerboard2, currParamInfo[currLevel-1].levelWidthCheckerboard, currParamInfo[currLevel-1].levelHeightCheckerboard, currParamInfo[currLevel].levelWidthCheckerboard, currParamInfo[currLevel].levelHeightCheckerboard, currParamInfo[currLevel-1].offset, currParamInfo[currLevel].offset, CHECKERBOARD_PART_2_ENUM);

		cudaThreadSynchronize();
	}
}

//function for retrieving the starting parameters for the 'next level' at every pixel in the bottom level for each `checkerboard'...
void retStartParamsNextLevel(currentStartMoveParamsPixelAtLevel* startParamsCheckerboard1, currentStartMoveParamsPixelAtLevel* startParamsCheckerboard2, float* estMovementXCheckerboard1,
				float* estMovementYCheckerboard1, float* estMovementXCheckerboard2, float* estMovementYCheckerboard2, int widthCheckerboard, int heightCheckerboard)
{
	//set the execution parameters for setting the parameters at the current level
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid((unsigned int)ceil((float)widthCheckerboard / (float)threads.x), (unsigned int)ceil((float)heightCheckerboard / (float)threads.y));

	//retrieve the `next parameter set' for both `checkerboard portions'

	getNextParamSet <<< grid, threads >>> (startParamsCheckerboard1, estMovementXCheckerboard1, estMovementYCheckerboard1, widthCheckerboard, heightCheckerboard);

	cudaThreadSynchronize();

	getNextParamSet <<< grid, threads >>> (startParamsCheckerboard2, estMovementXCheckerboard2, estMovementYCheckerboard2, widthCheckerboard, heightCheckerboard); 

	cudaThreadSynchronize();

}


//run the belief propagation algorithm on CUDA for motion estimation where the x and the y movements are computed where the two portions of the "checkerboard" are split
//the BP implementation is run such that the range is adjusted in each level in the hierarchy
//the width and height of each level are the same
//prevAndCurrMovementXFromImage1To2Device and prevAndCurrMovementYFromImage1To2Device represents both the `input movement' from the previous iteration and the `current movement' in the output of the current iteration...
//run using a hierarchy within each level to reduce the number of iterations...
void runBeliefPropMotionEstimationCUDAUseConstLevelSizeHierarchInLevUseEstMovement(float* image1PixelsDevice, float* image2PixelsDevice, float* prevAndCurrMovementXFromImage1To2Device, float* prevAndCurrMovementYFromImage1To2Device, currBeliefPropParams currentBeliefPropParams)
{
	//declare and start the timer
	unsigned int timer;
	cutCreateTimer(&timer);
	cutResetTimer(timer);

	cutStartTimer(timer);

	//define structure containing the current device for each checkerboard
	bpCurrentValsCheckerboard currentBpDeviceValsCheckerboard1;
	bpCurrentValsCheckerboard currentBpDeviceValsCheckerboard2;

	//generate the `hierarchical parameters'
	paramOffsetsSizes* hierarchParams = getOffsetsParamsHierarch((currentBeliefPropParams.widthImages / 2), (currentBeliefPropParams.heightImages), currentBeliefPropParams.numPyrHierarchLevels);

	//retrieve the number of `parameter values'
	int numParamVals = numValsParamHierarchy((currentBeliefPropParams.widthImages / 2), (currentBeliefPropParams.heightImages), currentBeliefPropParams.numPyrHierarchLevels);

	//allocate the space for the movement parameters on the device
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.paramsCurrentLevel), numParamVals * sizeof(currentStartMoveParamsPixelAtLevel));
	cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.paramsCurrentLevel), numParamVals * sizeof(currentStartMoveParamsPixelAtLevel));

	//initialize and `malloc' the space for the expected movements on the device...
	float* expectedMovementXDirDevice;
	float* expectedMovementYDirDevice;

	//`malloc' the space for the expected movements on the device if previous movement is defined...
	//if setting is to use previous movements, then generate the movement costs and incorporate it as part of the implementation
	if (currentBeliefPropParams.usePrevMovementSetting == yesUsePrevMovement)
	{
		printf("inUsePrevMovement\n");

		cudaMalloc((void**)&expectedMovementXDirDevice, (int)floor(currentBeliefPropParams.widthImages) * (int)floor(currentBeliefPropParams.heightImages) * sizeof(float));
		cudaMalloc((void**)&expectedMovementYDirDevice, (int)floor(currentBeliefPropParams.widthImages) * (int)floor(currentBeliefPropParams.heightImages) * sizeof(float));
	
		//generate the `expected movement' in the x and y directions using the movement in the previous images
		genEstMoveNextImSet(prevAndCurrMovementXFromImage1To2Device, prevAndCurrMovementYFromImage1To2Device,
				expectedMovementXDirDevice, expectedMovementYDirDevice, 
				(int)floor(currentBeliefPropParams.widthImages), (int)floor(currentBeliefPropParams.heightImages));
	}

	//set the number of possible movements (assume this is the same at every level...)
	int totalNumPossMovements = (currentBeliefPropParams.totalNumMovesXDir*currentBeliefPropParams.totalNumMovesYDir);

	//declare the float for the width and height of the current level
	float widthLevel = currentBeliefPropParams.widthImages;
	float heightLevel = currentBeliefPropParams.heightImages;

	int widthCheckerboardLevel = (int)floor(widthLevel) / 2;
	int heightCheckerboardLevel = (int)floor(heightLevel);

	printf("OrigWidthCheckerboardLevel: %d\n", widthCheckerboardLevel);
	printf("OrigHeightCheckerboardLevel: %d\n", heightCheckerboardLevel);

	//initialize the movement increment at the current level
	float currentLevMoveIncX = currentBeliefPropParams.currentMoveIncrementX;
	float currentLevMoveIncY = currentBeliefPropParams.currentMoveIncrementY;

	//retrieve the increment in the movement at the "last" level
	for (int numLevel = 0; numLevel < (currentBeliefPropParams.numBpLevels - 1); numLevel++)
	{ 
		//update the movement increment
		currentLevMoveIncX *= currentBeliefPropParams.propChangeMoveNextLevel;
		currentLevMoveIncY *= currentBeliefPropParams.propChangeMoveNextLevel;
	}

	//set the movement increment at the "bottom" level
	currentBeliefPropParams.motionIncBotLevX = currentLevMoveIncX;
	currentBeliefPropParams.motionIncBotLevY = currentLevMoveIncY;	
	
	//initialize the start movement parameters and place the 'other' parameters in constant memory 
	initializeParamsAndMovementOnDeviceHierarchImp(currentBpDeviceValsCheckerboard1.paramsCurrentLevel, currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
							hierarchParams, currentBeliefPropParams.numPyrHierarchLevels, currentBeliefPropParams);


	//run BP at each level
	for (int numLevel = (currentBeliefPropParams.numBpLevels - 1); numLevel >= 0; numLevel--)
	{
		//set the parameters at the current 'bottom' level...
		printf("numLevel: %d\n", numLevel);

		//retrieve the parameters at each level...
		//retrieve the parameters for each level of the hierarchy
		getParamsInHierarchy(hierarchParams, currentBpDeviceValsCheckerboard1.paramsCurrentLevel,
			currentBpDeviceValsCheckerboard2.paramsCurrentLevel, currentBeliefPropParams.numPyrHierarchLevels);

		//retrieve the width/height @ the 'top' level for allocation of the data cost/message value stuff...
		int widthCheckerboardTopLevel = hierarchParams[currentBeliefPropParams.numPyrHierarchLevels - 1].levelWidthCheckerboard; 
		int heightCheckerboardTopLevel = hierarchParams[currentBeliefPropParams.numPyrHierarchLevels - 1].levelHeightCheckerboard;

		//retrieve the number of bytes needed for the data and messages in the checkerboard at the top level...
		size_t numBytesDataAndMessagesTopLevel = widthCheckerboardTopLevel * heightCheckerboardTopLevel * totalNumPossMovements * sizeof(float);
	
		//retrieve the total number of bytes to allocate for data/message values at the  current level...

		//allocate the space of the messages on the device on the `top level'
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageUDevice), numBytesDataAndMessagesTopLevel);
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageDDevice), numBytesDataAndMessagesTopLevel);
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageLDevice), numBytesDataAndMessagesTopLevel);
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageRDevice), numBytesDataAndMessagesTopLevel);

		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageUDevice), numBytesDataAndMessagesTopLevel);
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageDDevice), numBytesDataAndMessagesTopLevel);
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageLDevice), numBytesDataAndMessagesTopLevel);
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageRDevice), numBytesDataAndMessagesTopLevel);

		//set the width/height of the level...note that width is double that of the `checkerboard'...height is the same...
		widthLevel = widthCheckerboardTopLevel*2; 
		heightLevel = heightCheckerboardTopLevel; 

		//initialize the message values at the top level
		initializeMessageVals(currentBpDeviceValsCheckerboard1.checkerboardMessVals, currentBpDeviceValsCheckerboard2.checkerboardMessVals,
						(int)floor(widthLevel), (int)floor(heightLevel));

		//declare the integer with the number of bytes for data/messages...
		size_t numBytesDataAndMessagesCurrLevel;

		//allocate a second set of  checkerboards for `next level'
		//define structure containing the current device for each checkerboard
		bpCurrentValsCheckerboard currentBpDeviceValsCheckerboard1_Set2;
		bpCurrentValsCheckerboard currentBpDeviceValsCheckerboard2_Set2;

		//set the data costs at each level from the bottom level "up"
		for (int levelNumHierarchComp = (currentBeliefPropParams.numPyrHierarchLevels - 1); levelNumHierarchComp >= 0; levelNumHierarchComp--)
		{
			printf("numHierachLevel: %d\n", levelNumHierarchComp);

			//retrieve and width/height of the current level and also of the checkerboard used for computation...
			widthLevel = hierarchParams[levelNumHierarchComp].levelWidthCheckerboard * 2;
			heightLevel = hierarchParams[levelNumHierarchComp].levelHeightCheckerboard;

			int widthCompCheckerboard = hierarchParams[levelNumHierarchComp].levelWidthCheckerboard;
			int heightCompCheckerboard = hierarchParams[levelNumHierarchComp].levelHeightCheckerboard;

			//retrieve the number of bytes in the data/message set for the checkerboard at the current level
			numBytesDataAndMessagesCurrLevel = widthCompCheckerboard * heightCompCheckerboard * totalNumPossMovements * sizeof(float);

			//allocate the space for the data costs (which may also incorporate the estimated movement costs...)
			CUDA_SAFE_CALL((cudaMalloc((void**) &(currentBpDeviceValsCheckerboard1.dataCostsVals), numBytesDataAndMessagesCurrLevel*sizeof(float)))); 
			CUDA_SAFE_CALL((cudaMalloc((void**) &(currentBpDeviceValsCheckerboard2.dataCostsVals), numBytesDataAndMessagesCurrLevel*sizeof(float))));

			
			

			//set the data costs at the current level
			initializeDataCostsCurrentLevelNoTexParams(image1PixelsDevice, image2PixelsDevice,
							currentBpDeviceValsCheckerboard1.dataCostsVals, currentBpDeviceValsCheckerboard2.dataCostsVals,
							currentBpDeviceValsCheckerboard1.paramsCurrentLevel, currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
							currentBeliefPropParams, (int)floor(widthLevel), (int)floor(heightLevel), hierarchParams[levelNumHierarchComp].offset);

			

		
			//if setting is to use previous movement, then generate the movement costs and incorporate it as part of the implementation
			if (currentBeliefPropParams.usePrevMovementSetting == yesUsePrevMovement)
			{

				//initialize the estimated movement costs at the top level and add it to the data costs in the overall computation...
				initEstMovementInDataCostsNoTextures(expectedMovementXDirDevice, expectedMovementYDirDevice, 
						currentBpDeviceValsCheckerboard1.dataCostsVals, currentBpDeviceValsCheckerboard2.dataCostsVals,
						currentBpDeviceValsCheckerboard1.paramsCurrentLevel, 
						currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
						currentBeliefPropParams,
						widthCompCheckerboard, heightCompCheckerboard,
						(int)floor(widthLevel), (int)floor(heightLevel),
						numBytesDataAndMessagesCurrLevel,
						hierarchParams[levelNumHierarchComp].offset);

			}



			runBPAtCurrentLevelNoTextures(
					currentBpDeviceValsCheckerboard1.checkerboardMessVals, currentBpDeviceValsCheckerboard2.checkerboardMessVals, 
					currentBpDeviceValsCheckerboard1.dataCostsVals, currentBpDeviceValsCheckerboard2.dataCostsVals,
					currentBpDeviceValsCheckerboard1.paramsCurrentLevel, currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
					currentBeliefPropParams.numBpIterations, (int)floor(widthLevel), (int)floor(heightLevel), 
					numBytesDataAndMessagesCurrLevel, hierarchParams[levelNumHierarchComp].offset);




			//if not at the bottom level, allocate the message values for the next value and transfer...also free the data cost stuff since recomputed in each level...
			if (levelNumHierarchComp > 0)
			{
				//free the data cost stuff...
				cudaFree(currentBpDeviceValsCheckerboard1.dataCostsVals);
				cudaFree(currentBpDeviceValsCheckerboard2.dataCostsVals);

				//retrieve the number of bytes at the next level
				int numBytesDataAndMessagesNextLevel = hierarchParams[levelNumHierarchComp-1].levelWidthCheckerboard * hierarchParams[levelNumHierarchComp-1].levelHeightCheckerboard * totalNumPossMovements * sizeof(float);

				

				//allocate a third set of checkerboard for `temporary storage'

				//allocate the space of the messages on the device on the `top level'
				cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1_Set2.checkerboardMessVals.messageUDevice), numBytesDataAndMessagesNextLevel);
				cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1_Set2.checkerboardMessVals.messageDDevice), numBytesDataAndMessagesNextLevel);
				cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1_Set2.checkerboardMessVals.messageLDevice), numBytesDataAndMessagesNextLevel);
				cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1_Set2.checkerboardMessVals.messageRDevice), numBytesDataAndMessagesNextLevel);

				cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2_Set2.checkerboardMessVals.messageUDevice), numBytesDataAndMessagesNextLevel);
				cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2_Set2.checkerboardMessVals.messageDDevice), numBytesDataAndMessagesNextLevel);
				cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2_Set2.checkerboardMessVals.messageLDevice), numBytesDataAndMessagesNextLevel);
				cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2_Set2.checkerboardMessVals.messageRDevice), numBytesDataAndMessagesNextLevel);

				//retrieve the message values at the `next level' given the values at the current level...
				getMessValsNextLevel(currentBpDeviceValsCheckerboard1.checkerboardMessVals,
							currentBpDeviceValsCheckerboard2.checkerboardMessVals,
							currentBpDeviceValsCheckerboard1_Set2.checkerboardMessVals,
							currentBpDeviceValsCheckerboard2_Set2.checkerboardMessVals,
							hierarchParams[levelNumHierarchComp].levelWidthCheckerboard * 2, 
							hierarchParams[levelNumHierarchComp].levelHeightCheckerboard, hierarchParams[levelNumHierarchComp-1].levelWidthCheckerboard * 2, 
							hierarchParams[levelNumHierarchComp-1].levelHeightCheckerboard,
							currentBpDeviceValsCheckerboard1.paramsCurrentLevel,
							currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
							hierarchParams[levelNumHierarchComp].offset, hierarchParams[levelNumHierarchComp-1].offset);

				//now free the spaces allocated to the 'current level'
				cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageUDevice);
				cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageDDevice);
				cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageLDevice);
				cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageRDevice);

				cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageUDevice);
				cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageDDevice);
				cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageLDevice);
				cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageRDevice);

				//now set the values at the 'next level' to be the 'current level'...
				currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageUDevice = currentBpDeviceValsCheckerboard1_Set2.checkerboardMessVals.messageUDevice;
				currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageDDevice = currentBpDeviceValsCheckerboard1_Set2.checkerboardMessVals.messageDDevice;
				currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageLDevice = currentBpDeviceValsCheckerboard1_Set2.checkerboardMessVals.messageLDevice;
				currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageRDevice = currentBpDeviceValsCheckerboard1_Set2.checkerboardMessVals.messageRDevice;

				currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageUDevice = currentBpDeviceValsCheckerboard2_Set2.checkerboardMessVals.messageUDevice;
				currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageDDevice = currentBpDeviceValsCheckerboard2_Set2.checkerboardMessVals.messageDDevice;
				currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageLDevice = currentBpDeviceValsCheckerboard2_Set2.checkerboardMessVals.messageLDevice;
				currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageRDevice = currentBpDeviceValsCheckerboard2_Set2.checkerboardMessVals.messageRDevice;
			}
		

		}
		
		printf("WidthFinalLevel: %f\n", widthLevel);
		printf("HeightFinalLevel: %f\n", heightLevel);

		printf("WidthCheckerboardFinalLevel: %d\n", widthCheckerboardLevel);
		printf("HeightCheckerboardFinalLevel: %d\n", heightCheckerboardLevel);

		//allocate the space for the estimated movement at the current level
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.estimatedXMovement), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(float));
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard1.estimatedYMovement), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(float));

		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.estimatedXMovement), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(float));
		cudaMalloc((void**)&(currentBpDeviceValsCheckerboard2.estimatedYMovement), widthCheckerboardLevel * heightCheckerboardLevel * sizeof(float));

		retrieveBestMotionEstLevel(currentBpDeviceValsCheckerboard1.estimatedXMovement, currentBpDeviceValsCheckerboard1.estimatedYMovement,
									currentBpDeviceValsCheckerboard2.estimatedXMovement, currentBpDeviceValsCheckerboard2.estimatedYMovement,
									currentBpDeviceValsCheckerboard1.checkerboardMessVals,
									currentBpDeviceValsCheckerboard2.checkerboardMessVals,
									currentBpDeviceValsCheckerboard1.dataCostsVals,
									currentBpDeviceValsCheckerboard2.dataCostsVals,
									currentBpDeviceValsCheckerboard1.paramsCurrentLevel,
									currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
									numBytesDataAndMessagesCurrLevel, (int)floor(widthLevel), (int)floor(heightLevel));

		//free the device memory allocated to the data costs
		cudaFree(currentBpDeviceValsCheckerboard1.dataCostsVals);
		cudaFree(currentBpDeviceValsCheckerboard2.dataCostsVals);

		//copy the current message values to the next level and update the parameters if not in bottom level
		//FOR NOW, NOT DOING THIS; SIMPLY RESETTING MESSAGE VALUES EACH TIME...
		//do need to 'free' the space allocated for the message values on the device...
		if (numLevel > 0)
		{

			//free the space allocated to the message values on the device...
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageUDevice);
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageDDevice);
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageLDevice);
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageRDevice);

			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageUDevice);
			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageDDevice);
			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageLDevice);
			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageRDevice);

			
			//retrieve the starting parameters in the 'next level'
			retStartParamsNextLevel(currentBpDeviceValsCheckerboard1.paramsCurrentLevel, currentBpDeviceValsCheckerboard2.paramsCurrentLevel, currentBpDeviceValsCheckerboard1.estimatedXMovement,
				currentBpDeviceValsCheckerboard1.estimatedYMovement, currentBpDeviceValsCheckerboard2.estimatedXMovement, currentBpDeviceValsCheckerboard2.estimatedYMovement, 
				widthCheckerboardLevel, heightCheckerboardLevel);

			//adjust the movement for all pixels at the "next" level
			adjustMovementAllPixParams(currentBeliefPropParams, (int)floor(widthLevel), (int)floor(heightLevel));


			/*checkerboardMessagesDeviceStruct messageDeviceCheckerboard1CopyTo;
			checkerboardMessagesDeviceStruct messageDeviceCheckerboard2CopyTo;

			cudaMalloc((void**)&(messageDeviceCheckerboard1CopyTo.messageUDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard1CopyTo.messageDDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard1CopyTo.messageLDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard1CopyTo.messageRDevice), numBytesDataAndMessagesEachLevel);

			cudaMalloc((void**)&(messageDeviceCheckerboard2CopyTo.messageUDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard2CopyTo.messageDDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard2CopyTo.messageLDevice), numBytesDataAndMessagesEachLevel);
			cudaMalloc((void**)&(messageDeviceCheckerboard2CopyTo.messageRDevice), numBytesDataAndMessagesEachLevel);

			//declare the movement parameters for the "next" level and allocate space for them
			currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboard1;
			currentStartMoveParamsPixelAtLevel* paramsNextLevelDeviceCheckerboard2;

			cudaMalloc((void**)&paramsNextLevelDeviceCheckerboard1, widthCheckerboardLevel * heightCheckerboardLevel * sizeof(currentStartMoveParamsPixelAtLevel));
			cudaMalloc((void**)&paramsNextLevelDeviceCheckerboard2, widthCheckerboardLevel * heightCheckerboardLevel * sizeof(currentStartMoveParamsPixelAtLevel));


			copyMessageValuesToNextLevelDownLevelSizeConst(
											currentBpDeviceValsCheckerboard1.checkerboardMessVals, 
											currentBpDeviceValsCheckerboard2.checkerboardMessVals, 
											currentBpDeviceValsCheckerboard1.paramsCurrentLevel,
											currentBpDeviceValsCheckerboard2.paramsCurrentLevel,
											paramsNextLevelDeviceCheckerboard1,
											paramsNextLevelDeviceCheckerboard2,
											currentBpDeviceValsCheckerboard1.estimatedXMovement, currentBpDeviceValsCheckerboard1.estimatedYMovement,
											currentBpDeviceValsCheckerboard2.estimatedXMovement, currentBpDeviceValsCheckerboard2.estimatedYMovement,
											messageDeviceCheckerboard1CopyTo,
											messageDeviceCheckerboard2CopyTo,
											(int)floor(widthLevel), (int)floor(heightLevel),
											numBytesDataAndMessagesEachLevel);

			cudaThreadSynchronize();

			//free the device memory allocated to the previous message values now that they've been copied
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageUDevice);
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageDDevice);
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageLDevice);
			cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageRDevice);

			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageUDevice);
			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageDDevice);
			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageLDevice);
			cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageRDevice);

			//free the device memory allocated to the parameters at the previous level
			cudaFree(currentBpDeviceValsCheckerboard1.paramsCurrentLevel);
			cudaFree(currentBpDeviceValsCheckerboard2.paramsCurrentLevel);


			//set the checkerboard device messages to the messages that were copied to
			currentBpDeviceValsCheckerboard1.checkerboardMessVals = messageDeviceCheckerboard1CopyTo;
			currentBpDeviceValsCheckerboard2.checkerboardMessVals = messageDeviceCheckerboard2CopyTo;

			//set the "current level" parameter to the set "next" level parameters
			currentBpDeviceValsCheckerboard1.paramsCurrentLevel = paramsNextLevelDeviceCheckerboard1;
			currentBpDeviceValsCheckerboard2.paramsCurrentLevel = paramsNextLevelDeviceCheckerboard2;


			//free the device memory allocated to the calculated movement at the previous level
			cudaFree(currentBpDeviceValsCheckerboard1.estimatedXMovement);
			cudaFree(currentBpDeviceValsCheckerboard1.estimatedYMovement);
			cudaFree(currentBpDeviceValsCheckerboard2.estimatedXMovement);
			cudaFree(currentBpDeviceValsCheckerboard2.estimatedYMovement);*/

			
		}
	}

	//free the 'estimated movement' on the device if using previous movement...
	if (currentBeliefPropParams.usePrevMovementSetting == yesUsePrevMovement)
	{
		cudaFree(expectedMovementXDirDevice);
		cudaFree(expectedMovementYDirDevice);
	}

	//free device memory used for storing the message values and movement parameters
	cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageUDevice);
	cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageDDevice);
	cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageLDevice);
	cudaFree(currentBpDeviceValsCheckerboard1.checkerboardMessVals.messageRDevice);

	cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageUDevice);
	cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageDDevice);
	cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageLDevice);
	cudaFree(currentBpDeviceValsCheckerboard2.checkerboardMessVals.messageRDevice);

	cudaFree(currentBpDeviceValsCheckerboard1.paramsCurrentLevel);
	cudaFree(currentBpDeviceValsCheckerboard2.paramsCurrentLevel);


	//retrieve final movement (assume not rounding...)

	//combine the x and y checkerboard movement into a single 2D array for output
	combineXYCheckerboardMovements(currentBpDeviceValsCheckerboard1.estimatedXMovement, currentBpDeviceValsCheckerboard1.estimatedYMovement,
					currentBpDeviceValsCheckerboard2.estimatedXMovement, currentBpDeviceValsCheckerboard2.estimatedYMovement,
					prevAndCurrMovementXFromImage1To2Device, prevAndCurrMovementYFromImage1To2Device, (int)floor(widthLevel), (int)floor(heightLevel));



	//free the movement stored in checkerboards now that it has been combined
	cudaFree(currentBpDeviceValsCheckerboard1.estimatedXMovement);
	cudaFree(currentBpDeviceValsCheckerboard1.estimatedYMovement);

	cudaFree(currentBpDeviceValsCheckerboard2.estimatedXMovement);
	cudaFree(currentBpDeviceValsCheckerboard2.estimatedYMovement);
	

	//stop the timer and print the total running time to run belief propagation
	cutStopTimer(timer);
	printf("Total belief propagation running time: %f \n", cutGetTimerValue(timer));
}


}
