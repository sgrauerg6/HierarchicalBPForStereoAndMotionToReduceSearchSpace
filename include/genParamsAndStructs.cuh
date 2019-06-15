//genParamsAndStructs.cuh
//Scott Grauer-Gray
//June 29, 2009
//Declares the general parameters and structs

#ifndef GEN_PARAMS_AND_STRUCTS_CUH
#define GEN_PARAMS_AND_STRUCTS_CUH

//define a constant small value
#define SMALL_VALUE 0.01f

//define an "infinite" integer and float value
#define INF_VALUE_INT 1000000000000
#define INF_VALUE_FLOAT 1000000000000.0f

//define "negative infinity" integer and float values
#define NEG_INF_VALUE_INT -1000000000000
#define NEG_INF_VALUE_FLOAT -1000000000000.0f

//enum giving the location of the data (host/device)
typedef enum
{
	DATA_ON_HOST,
	DATA_ON_DEVICE
} dataLocation;

//union giving pointers to data of various types
typedef union
{
	float* floatDataPointer;
	int* intDataPointer;
	unsigned int* uIntDataPointer;
	char* charDataPointer;
	unsigned char* uCharDataPointer;
} currentDataPointer;

//enum giving the various forms of data
typedef enum
{
	FLOAT_DATA,
	INT_DATA,
	UNSIGNED_INT_DATA,
	CHAR_DATA,
	UNSIGNED_CHAR_DATA
} currentDataType;

//define the possible movements and the default motion direction
typedef enum
{
	POS_X_MOTION_RIGHT_POS_Y_MOTION_DOWN,
	POS_X_MOTION_LEFT_POS_Y_MOTION_UP
} posMoveDirection;

//define the current movement direction
typedef enum
{
	X_DIR_MOVEMENT,
	Y_DIR_MOVEMENT
} desiredMovement;


#endif //GEN_PARAMS_AND_STRUCTS_CUH
