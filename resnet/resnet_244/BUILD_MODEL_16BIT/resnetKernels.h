#ifndef __RESNETKERNEL_H__
#define __RESNETKERNEL_H__

#include "AutoTilerLibTypes.h"
#include "CNN_BasicKernels.h"
#include "resnet.h"
#define _resnet_L1_Memory_SIZE 48576
#define _resnet_L2_Memory_SIZE 46080
extern char *resnet_L1_Memory; /* Size given for generation: 48736 bytes, used: 48576 bytes */
extern char *resnet_L2_Memory; /* Size used for generation: 46080 bytes */
extern void S4_Conv2d_8x1x7x7_MaxPool_3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out);
extern void S7_Conv2d_8x8x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out);
extern void S10_Conv2d_8x8x3x3(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out);
extern void S11_MatAdd_8x60x80(
		short int * __restrict__ In1,
		short int * __restrict__ In2,
		short int * __restrict__ Out);
extern void S14_Conv2d_8x8x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out);
extern void S17_Conv2d_8x8x3x3(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out);
extern void S18_MatAdd_8x60x80(
		short int * __restrict__ In1,
		short int * __restrict__ In2,
		short int * __restrict__ Out);
extern void S21_Conv2d_16x8x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out);
extern void S24_Conv2d_16x8x1x1(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out);
extern void S27_Conv2d_16x16x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out);
extern void S28_MatAdd_16x30x40(
		short int * __restrict__ In1,
		short int * __restrict__ In2,
		short int * __restrict__ Out);
extern void S31_Conv2d_16x16x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out);
extern void S34_Conv2d_16x16x3x3(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out);
extern void S35_MatAdd_16x30x40(
		short int * __restrict__ In1,
		short int * __restrict__ In2,
		short int * __restrict__ Out);
extern void S38_Conv2d_32x16x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out);
extern void S41_Conv2d_32x16x1x1(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out);
extern void S44_Conv2d_32x32x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out);
extern void S45_MatAdd_32x15x20(
		short int * __restrict__ In1,
		short int * __restrict__ In2,
		short int * __restrict__ Out);
extern void S48_Conv2d_32x32x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out);
extern void S51_Conv2d_32x32x3x3(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out);
extern void S52_MatAdd_32x15x20(
		short int * __restrict__ In1,
		short int * __restrict__ In2,
		short int * __restrict__ Out);
extern void S55_Conv2d_64x32x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out);
extern void S58_Conv2d_64x32x1x1(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out);
extern void S61_Conv2d_64x64x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out);
extern void S62_MatAdd_64x8x10(
		short int * __restrict__ In1,
		short int * __restrict__ In2,
		short int * __restrict__ Out);
extern void S65_Conv2d_64x64x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out);
extern void S68_Conv2d_64x64x3x3(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out);
extern void S69_MatAdd_64x8x10(
		short int * __restrict__ In1,
		short int * __restrict__ In2,
		short int * __restrict__ Out);
extern void S70_AveragePool_2x2(
		short int * __restrict__ In,
		short int * __restrict__ Out);
extern void S73_Linear_7x64x4x5(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out);
extern void S74_SoftMax(
		short int * __restrict__ In,
		short int * __restrict__ Out);
extern int resnetCNN_Construct();
extern int resnetCNN_Destruct();
extern int resnetCNN(
		signed short * __restrict__ Input_1,
		signed short * __restrict__ Output_1);
#endif
