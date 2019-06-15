//genCudaFuncts.cu
//Scott Grauer-Gray
//July 2, 2009
//Defines the general CUDA functions

#include "genCudaFuncts.cuh"

//functions are C, so need extern
extern "C"
{

	//function to initialize CUDA
	void cudaInit(int argc, char **argv)
	{
		CUT_DEVICE_INIT(argc, argv);
	}

	//function to exit CUDA
	void cudaExit(int argc, char **argv)
	{
		CUT_EXIT(argc, argv);
	}


	//function to allocate an array of data on the device
	void allocateArray(void **devPtr, int size)
	{
		cudaMalloc(devPtr, size);
	}


	//function to free an array of data on the device
	void freeArray(void *devPtr)
	{
		cudaFree(devPtr);
	}

	//function to synchronize the threads
	void threadSync()
	{
		cudaThreadSynchronize();
	}

	//function to copy an array of data from the device to the host
	void copyArrayFromDevice(void* host, const void* device, int size)
	{
		cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
	}

	//function to copy data from the host to the device
	void copyArrayToDevice(void* device, const void* host, int size)
	{
		cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
	}

	//function to copy data within the current device
	void copyArrayWithinDevice(void* deviceTo, const void* deviceFrom, int size)
	{
		cudaMemcpy(deviceTo, deviceFrom, size, cudaMemcpyDeviceToDevice);
	}
}
