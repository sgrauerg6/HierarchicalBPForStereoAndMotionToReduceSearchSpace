/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

/* Template project which demonstrates the basics on how to setup a project 
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil_inline.h>

//files for running the implementation
#include "testRunBpMotionCommLineArgs.h"
#include "testRunBpMotionMultImages.h"
#include "testRunBpMotionGivenExpMotion.h"


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{

	printf("Num args: %d\n", argc);

	if (argc == 24)
	{
		testRunBpMotionWithCommLineArgs(argv);
	}
	else if (argc == 28)
	{
		testRunBpMotionMultImages(argv);
	}
	else
	{    
		testRunBpMotionGivenExpectedMotion(argv);
	}

	return 1;
	//cutilExit(argc, argv);
}
