//genCudaFuncts.cuh
//Scott Grauer-Gray
//July 2, 2009
//Declares the general CUDA functions

#ifndef GEN_CUDA_FUNCTS_CUH
#define GEN_CUDA_FUNCTS_CUH

//needed for general CUDA utility functions
#include <cutil.h>

//needed for general I/O operations
#include <stdio.h>

//functions are C, so need extern
extern "C"
{

	//function to initialize CUDA
	void cudaInit(int argc, char **argv);
	
	//function to exit CUDA
	void cudaExit(int argc, char **argv);

	//function to allocate an array of data on the device
	void allocateArray(void **devPtr, int size);

	//function to free an array of data on the device
	void freeArray(void *devPtr);

	//function to synchronize the threads
	void threadSync();

	//function to copy an array of data from the device to the host
	void copyArrayFromDevice(void* host, const void* device, int size);

	//function to copy data from the host to the device
	void copyArrayToDevice(void* device, const void* host, int size);

	//function to copy data within the current device
	void copyArrayWithinDevice(void* deviceTo, const void* deviceFrom, int size);
}

#endif //GEN_CUDA_FUNCTS_CUH
