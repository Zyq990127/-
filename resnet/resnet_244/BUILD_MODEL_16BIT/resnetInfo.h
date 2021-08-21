#ifndef RESNET_GRAPHINFO_H
#define RESNET_GRAPHINFO_H
// Quantized scales can be used round_norm(val * QSCALE, QNORM) giving the real value in Q8

// S0_Op_input_1
#define S0_Op_input_1_Q	15
// S1_Op_DEPTHWISE_CONV_2D_0_1_r_hwc_chw
#define S1_Op_DEPTHWISE_CONV_2D_0_1_r_hwc_chw_Q	15
// S2_Op_conv1kernel
#define S2_Op_conv1kernel_Q	11
// S3_Op_conv1Conv2D_bias
#define S3_Op_conv1Conv2D_bias_Q	12
// S4_Conv2d_8x1x7x7_MaxPool_3x3_Relu
#define S4_Conv2d_8x1x7x7_MaxPool_3x3_Relu_Q	11
// S5_Op_res2bbranch2akernel
#define S5_Op_res2bbranch2akernel_Q	15
// S6_Op_res2bbranch2aConv2D_bias
#define S6_Op_res2bbranch2aConv2D_bias_Q	14
// S7_Conv2d_8x8x3x3_Relu
#define S7_Conv2d_8x8x3x3_Relu_Q	11
// S8_Op_res2bbranch2bkernel
#define S8_Op_res2bbranch2bkernel_Q	14
// S9_Op_res2bbranch2bConv2D_bias
#define S9_Op_res2bbranch2bConv2D_bias_Q	13
// S10_Conv2d_8x8x3x3
#define S10_Conv2d_8x8x3x3_Q	10
// S11_MatAdd_8x60x80
#define S11_MatAdd_8x60x80_Q	10
// S12_Op_res2cbranch2akernel
#define S12_Op_res2cbranch2akernel_Q	15
// S13_Op_res2cbranch2aConv2D_bias
#define S13_Op_res2cbranch2aConv2D_bias_Q	14
// S14_Conv2d_8x8x3x3_Relu
#define S14_Conv2d_8x8x3x3_Relu_Q	10
// S15_Op_res2cbranch2bkernel
#define S15_Op_res2cbranch2bkernel_Q	15
// S16_Op_res2cbranch2bConv2D_bias
#define S16_Op_res2cbranch2bConv2D_bias_Q	14
// S17_Conv2d_8x8x3x3
#define S17_Conv2d_8x8x3x3_Q	10
// S18_MatAdd_8x60x80
#define S18_MatAdd_8x60x80_Q	9
// S19_Op_res3a_branch2akernel
#define S19_Op_res3a_branch2akernel_Q	15
// S20_Op_res3a_branch2aConv2D_bias
#define S20_Op_res3a_branch2aConv2D_bias_Q	15
// S21_Conv2d_16x8x3x3_Relu
#define S21_Conv2d_16x8x3x3_Relu_Q	11
// S22_Op_res3a_branch1kernel
#define S22_Op_res3a_branch1kernel_Q	15
// S23_Op_res3a_branch1Conv2D_bias
#define S23_Op_res3a_branch1Conv2D_bias_Q	15
// S24_Conv2d_16x8x1x1
#define S24_Conv2d_16x8x1x1_Q	10
// S25_Op_res3a_branch2bkernel
#define S25_Op_res3a_branch2bkernel_Q	15
// S26_Op_res3a_branch2bConv2D_bias
#define S26_Op_res3a_branch2bConv2D_bias_Q	13
// S27_Conv2d_16x16x3x3_Relu
#define S27_Conv2d_16x16x3x3_Relu_Q	11
// S28_MatAdd_16x30x40
#define S28_MatAdd_16x30x40_Q	9
// S29_Op_res3bbranch2akernel
#define S29_Op_res3bbranch2akernel_Q	15
// S30_Op_res3bbranch2aConv2D_bias
#define S30_Op_res3bbranch2aConv2D_bias_Q	14
// S31_Conv2d_16x16x3x3_Relu
#define S31_Conv2d_16x16x3x3_Relu_Q	11
// S32_Op_res3bbranch2bkernel
#define S32_Op_res3bbranch2bkernel_Q	15
// S33_Op_res3bbranch2bConv2D_bias
#define S33_Op_res3bbranch2bConv2D_bias_Q	13
// S34_Conv2d_16x16x3x3
#define S34_Conv2d_16x16x3x3_Q	10
// S35_MatAdd_16x30x40
#define S35_MatAdd_16x30x40_Q	9
// S36_Op_res4a_branch2akernel
#define S36_Op_res4a_branch2akernel_Q	15
// S37_Op_res4a_branch2aConv2D_bias
#define S37_Op_res4a_branch2aConv2D_bias_Q	15
// S38_Conv2d_32x16x3x3_Relu
#define S38_Conv2d_32x16x3x3_Relu_Q	11
// S39_Op_res4a_branch1kernel
#define S39_Op_res4a_branch1kernel_Q	15
// S40_Op_res4a_branch1Conv2D_bias
#define S40_Op_res4a_branch1Conv2D_bias_Q	15
// S41_Conv2d_32x16x1x1
#define S41_Conv2d_32x16x1x1_Q	10
// S42_Op_res4a_branch2bkernel
#define S42_Op_res4a_branch2bkernel_Q	15
// S43_Op_res4a_branch2bConv2D_bias
#define S43_Op_res4a_branch2bConv2D_bias_Q	13
// S44_Conv2d_32x32x3x3_Relu
#define S44_Conv2d_32x32x3x3_Relu_Q	11
// S45_MatAdd_32x15x20
#define S45_MatAdd_32x15x20_Q	9
// S46_Op_res4bbranch2akernel
#define S46_Op_res4bbranch2akernel_Q	15
// S47_Op_res4bbranch2aConv2D_bias
#define S47_Op_res4bbranch2aConv2D_bias_Q	15
// S48_Conv2d_32x32x3x3_Relu
#define S48_Conv2d_32x32x3x3_Relu_Q	11
// S49_Op_res4bbranch2bkernel
#define S49_Op_res4bbranch2bkernel_Q	15
// S50_Op_res4bbranch2bConv2D_bias
#define S50_Op_res4bbranch2bConv2D_bias_Q	13
// S51_Conv2d_32x32x3x3
#define S51_Conv2d_32x32x3x3_Q	10
// S52_MatAdd_32x15x20
#define S52_MatAdd_32x15x20_Q	9
// S53_Op_res5a_branch2akernel
#define S53_Op_res5a_branch2akernel_Q	15
// S54_Op_res5a_branch2aConv2D_bias
#define S54_Op_res5a_branch2aConv2D_bias_Q	15
// S55_Conv2d_64x32x3x3_Relu
#define S55_Conv2d_64x32x3x3_Relu_Q	11
// S56_Op_res5a_branch1kernel
#define S56_Op_res5a_branch1kernel_Q	15
// S57_Op_res5a_branch1Conv2D_bias
#define S57_Op_res5a_branch1Conv2D_bias_Q	14
// S58_Conv2d_64x32x1x1
#define S58_Conv2d_64x32x1x1_Q	10
// S59_Op_res5a_branch2bkernel
#define S59_Op_res5a_branch2bkernel_Q	15
// S60_Op_res5a_branch2bConv2D_bias
#define S60_Op_res5a_branch2bConv2D_bias_Q	13
// S61_Conv2d_64x64x3x3_Relu
#define S61_Conv2d_64x64x3x3_Relu_Q	11
// S62_MatAdd_64x8x10
#define S62_MatAdd_64x8x10_Q	10
// S63_Op_res5bbranch2akernel
#define S63_Op_res5bbranch2akernel_Q	15
// S64_Op_res5bbranch2aConv2D_bias
#define S64_Op_res5bbranch2aConv2D_bias_Q	15
// S65_Conv2d_64x64x3x3_Relu
#define S65_Conv2d_64x64x3x3_Relu_Q	11
// S66_Op_res5bbranch2bkernel
#define S66_Op_res5bbranch2bkernel_Q	15
// S67_Op_res5bbranch2bConv2D_bias
#define S67_Op_res5bbranch2bConv2D_bias_Q	13
// S68_Conv2d_64x64x3x3
#define S68_Conv2d_64x64x3x3_Q	10
// S69_MatAdd_64x8x10
#define S69_MatAdd_64x8x10_Q	9
// S70_AveragePool_2x2
#define S70_AveragePool_2x2_Q	9
// S71_Op_full_connection7kerneltranspos
#define S71_Op_full_connection7kerneltranspos_Q	15
// S72_Op_full_connection7MatMul_bias
#define S72_Op_full_connection7MatMul_bias_Q	15
// S73_Linear_7x64x4x5
#define S73_Linear_7x64x4x5_Q	9
// S74_SoftMax
#define S74_SoftMax_Q	15
// S75_Op_output_1
#define S75_Op_output_1_Q	15
#endif //RESNET_GRAPHINFO_H