//smoothImageParamsAndStructs.cuh
//Scott Grauer-Gray
//June 28, 2009
//Declares the parameters and structures used in smoothing the images

#ifndef SMOOTH_IMAGE_PARAMS_AND_STRUCTS_CUH
#define SMOOTH_IMAGE_PARAMS_AND_STRUCTS_CUH

//needed for the general parameters and structs
#include "genParamsAndStructs.cuh"

//define the width and height of the thread blocks used to smooth the image
#define BLOCK_SIZE_WIDTH_SMOOTH_IMAGE 16
#define BLOCK_SIZE_HEIGHT_SMOOTH_IMAGE 16

//define the sigma value to determine whether or not to actually smooth the image
#define MIN_SIGMA_VAL_SMOOTH_IMAGES 0.1f

//define the width of the sigma
#define WIDTH_SIGMA 4.0f

//define the maximum filter size
//TODO: make sure that filter can't go beyond max size
#define MAX_SIZE_FILTER 25

//#define the default runSmoothImage params
#define DEFAULT_SIGMA_SMOOTH_IMAGE 1.0f
#define DEFAULT_LOCATION_OUTPUT_SMOOTHED_IMAGE DATA_ON_DEVICE
#define DEFAULT_DATA_TYPE_OUTPUT_PIXELS FLOAT_DATA

#endif //SMOOTH_IMAGE_PARAMS_AND_STRUCTS_CUH
