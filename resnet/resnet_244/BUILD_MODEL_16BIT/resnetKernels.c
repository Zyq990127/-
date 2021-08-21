#include "resnetKernels.h"
L1_CL_MEM AT_L1_POINTER resnet_L1_Memory;
L2_MEM AT_L2_POINTER resnet_L2_Memory;
AT_HYPERRAM_POINTER resnet_L3_Memory;
extern AT_HYPERRAM_T HyperRam;
static AT_HYPERFLASH_FS_T HyperFlash;
void S4_Conv2d_8x1x7x7_MaxPool_3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out)

{
	/* Shared L1: 33072 bytes, L2 buffer: 17616 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;
	KerReLUPool_fp_T S_KerArg3, *KerArg3 = &S_KerArg3;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	unsigned int _N_In;
	unsigned int _SN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 8, Tiled: 1][Tile0 Dim: 60][D0 Dim: Init: 1, Tiled: 1]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 60 logical tiles, 60 physical tiles
			Total Size: 76800 [D1, [0 x 76800, 76800]][Tile0, 60:[80x1, 58:80x1, 80x1], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 76800, 76800]][Tile0, 60:[80x1, 58:80x1, 80x1], 2]
		Tile0: [0, 1280, 160], Tile1: [160, 1280, 160], Tile2; [320, 1280, 160]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 784 [D1, [0 x 784, 784]][D0, [0 x 98, 98]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 784, 784]][D0, [0 x 98, 98]]
		Tile0: [0, 784, 784], Tile1: [0, 784, 784], Tile2; [0, 784, 784]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 60 logical tiles, 60 physical tiles
			Total Size: 158112 [D0, [0 x 158112, 158112]][Tile0, 60:[324x8, 58:324x11, 324x11], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 60:[324x8], 2][D0, [0 x 158112, 158112]]
		Tile0: [0, 5184, 5184], Tile1: [648, 7128, 7128], Tile2; [3240, 7128, 7128]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 60 logical tiles, 1 physical tiles
			Total Size: 623392 [D1, [0 x 623392, 623392]][Tile0, 60:[161x3, 58:161x3, 161x3], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 623392, 623392]][Tile0, 60:[161x3, 58:161x3, 161x3], 4]
		Tile0: [0, 15456, 1932], Tile1: [0, 15456, 1932], Tile2; [0, 15456, 1932]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (resnet_L1_Memory+17616);
	KerArg0->W = (unsigned short int) (161);
	KerArg0->H = (unsigned short int) (3);
	KerArg0->OutFeatures = (unsigned short int) (8);
	KerArg0->Bias = (short int * __restrict__) (resnet_L1_Memory+14256);
	KerArg0->NormBias = (signed char) (14);
	KerArg1->W = (unsigned short int) (324);
	KerArg1->UsedW = (unsigned short int) (324);
	KerArg1->InFeatures = (unsigned short int) (1);
	KerArg1->OutFeatures = (unsigned short int) (8);
	KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+14272);
	KerArg1->Out = (int * __restrict__) (resnet_L1_Memory+17616);
	KerArg1->Norm = (unsigned char) (15);
	KerArg1->TotalInFeatures = (short int) (1);
	KerArg1->Orientation = (unsigned char) (1);
	KerArg1->S = (unsigned char) (2);
	KerArg2->In = (int * __restrict__) (resnet_L1_Memory+17616);
	KerArg2->W = (unsigned short int) (161);
	KerArg2->H = (unsigned short int) (3);
	KerArg2->Out = (short int * __restrict__) (resnet_L1_Memory+17616);
	KerArg2->Norm = (unsigned char) (15);
	KerArg2->InFeatures = (unsigned short int) (8);
	KerArg2->LB = (int) (0);
	KerArg2->UB = (int) (32767);
	KerArg3->In = (short int * __restrict__) (resnet_L1_Memory+17616);
	KerArg3->W = (unsigned short int) (161);
	KerArg3->UsedW = (unsigned short int) (161);
	KerArg3->H = (unsigned short int) (3);
	KerArg3->UsedH = (unsigned short int) (3);
	KerArg3->OutFeatures = (unsigned short int) (8);
	KerArg3->Pad = (v4s) 0;
	KerArg3->M = (unsigned char) (3);
	KerArg3->S = (unsigned char) (2);
	KerArg3->Orientation = (unsigned char) (1);
	KerArg3->Oper = (unsigned char) (1);
	KerArg3->LB = (int) (0);
	KerArg3->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=1280; _LC_Out=160;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+14256), 16, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+14272), 784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 5184, 0, &DmaR_Evt3);
	_N_In=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<60; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==59), T0Ind_NextLast = ((T0Ind+1)==59);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1, D0Ind_NextLast = 1;
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(T0Ind_Last)) {
					_N_In = _N_In + (2592-(1944*(T0Ind==0))); _SN_In = (1*((T0Ind_NextLast)?7128:7128)); 
				} else if (!(1)) {
					_N_In = _N_In + (-150984); _SN_In = (1*(5184)); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+7128*((D0Ind_Total+1)%2)),
							_SN_In, 0, &DmaR_Evt3);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (resnet_L1_Memory+0+7128*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (11-3*(T0Ind==0)-0*(T0Ind_Last));
				KerArg1->UsedH = (unsigned short int) (11-3*(T0Ind==0)-0*(T0Ind_Last));
				KerArg1->Pad = (v4s) ((v4s){3,0,3*(T0Ind==0),0*(T0Ind_Last)});
				AT_FORK(gap_ncore(), (void *) KerParConv7x7StrideS_DP_fp, (void *) KerArg1);
				__CALL(KerParConv7x7StrideS_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
				D0Ind_Total++;
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerDP_IO_fp, (void *) KerArg2);
			__CALL(KerDP_IO_fp, KerArg2);
			KerArg3->Out = (short int * __restrict__) (resnet_L1_Memory+15056+1280*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerParPoolNxNStrideS_fp, (void *) KerArg3);
			__CALL(KerParPoolNxNStrideS_fp, KerArg3);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+15056+1280*((T0Ind_Total)%2)),
					_SC_Out, 9600, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (160); _LC_Out = (160); _SC_Out = (8*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S7_Conv2d_8x8x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out)

{
	/* Shared L1: 48528 bytes, L2 buffer: 28048 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 8, Tiled: 1][Tile0 Dim: 8][D0 Dim: Init: 8, Tiled: 4]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 32 logical tiles, 32 physical tiles
			Total Size: 76800 [D0, [3 x 19200, 19200]][Tile0, 8:[80x9, 6:80x10, 80x5], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 8:[80x9, 2:80x10, 80x5], 2][D0, [3 x 19200, 19200]]
		Tile0: [0, 2880, 1440], Tile1: [19200, 2880, 1440], Tile2; [38400, 2880, 1440]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 1152 [D1, [0 x 1152, 1152]][D0, [3 x 36, 36]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 1152, 1152]][D0, [3 x 36, 36]]
		Tile0: [0, 1152, 1152], Tile1: [0, 1152, 1152], Tile2; [0, 1152, 1152]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 8 logical tiles, 8 physical tiles
			Total Size: 76800 [D1, [0 x 76800, 76800]][Tile0, 8:[80x8, 6:80x8, 80x4], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 76800, 76800]][Tile0, 8:[80x8, 6:80x8, 80x4], 2]
		Tile0: [0, 10240, 1280], Tile1: [1280, 10240, 1280], Tile2; [2560, 10240, 1280]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 8 logical tiles, 1 physical tiles
			Total Size: 153600 [D1, [0 x 153600, 153600]][Tile0, 8:[80x8, 6:80x8, 80x4], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 153600, 153600]][Tile0, 8:[80x8, 6:80x8, 80x4], 4]
		Tile0: [0, 20480, 2560], Tile1: [0, 20480, 2560], Tile2; [0, 20480, 2560]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (resnet_L1_Memory+28048);
	KerArg0->W = (unsigned short int) (80);
	KerArg0->OutFeatures = (unsigned short int) (8);
	KerArg0->Bias = (short int * __restrict__) (resnet_L1_Memory+6400);
	KerArg0->NormBias = (signed char) (12);
	KerArg1->W = (unsigned short int) (80);
	KerArg1->UsedW = (unsigned short int) (80);
	KerArg1->InFeatures = (unsigned short int) (2);
	KerArg1->OutFeatures = (unsigned short int) (8);
	KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+6416);
	KerArg1->Out = (int * __restrict__) (resnet_L1_Memory+28048);
	KerArg1->Norm = (unsigned char) (15);
	KerArg1->TotalInFeatures = (short int) (8);
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (resnet_L1_Memory+28048);
	KerArg2->W = (unsigned short int) (80);
	KerArg2->Norm = (unsigned char) (15);
	KerArg2->InFeatures = (unsigned short int) (8);
	KerArg2->LB = (int) (0);
	KerArg2->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 2880, 9600, 1440, 0, &DmaR_Evt1);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+6400), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+6416), 1152, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Filter */
	_C_Out=0; _SC_Out=10240; _LC_Out=1280;
	_SP_Out=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<8; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==7), T0Ind_NextLast = ((T0Ind+1)==7);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->H = (unsigned short int) (T0Ind_Last?4:8);
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			for (D0Ind=0; D0Ind<4; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==3), D0Ind_NextLast = ((D0Ind+1)==3);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (19200); _LN_In = ((T0Ind_Last)?800:(1600-160*(T0Ind==0))); _SN_In = (2*_LN_In); 
				} else if (!(T0Ind_Last)) {
					_N_In = _N_In + (1280-(160*(T0Ind==0)))+(-57600); _LN_In = ((T0Ind_NextLast)?800:1600); _SN_In = (2*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-8800)+(-57600); _LN_In = (1440); _SN_In = (2*_LN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+3200*((D0Ind_Total+1)%2)),
							_SN_In, 9600, _LN_In, 0, &DmaR_Evt1);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (resnet_L1_Memory+0+3200*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (((T0Ind_Last)?5:10)-1*(T0Ind==0));
				KerArg1->UsedH = (unsigned short int) (((T0Ind_Last)?5:10)-1*(T0Ind==0));
				KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+6416+((D0Ind)*36));
				KerArg1->Pad = (v4s) ((v4s){1,1,1*(T0Ind==0),1*(T0Ind_Last)});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_DP_fp, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->H = (unsigned short int) (T0Ind_Last?4:8);
			KerArg2->Out = (short int * __restrict__) (resnet_L1_Memory+7568+10240*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerDP_fp, (void *) KerArg2);
			__CALL(KerDP_fp, KerArg2);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+7568+10240*((T0Ind_Total)%2)),
					_SC_Out, 9600, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (1280); _LC_Out = ((T0Ind_NextLast)?640:1280); _SC_Out = (8*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S10_Conv2d_8x8x3x3(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out)

{
	/* Shared L1: 48528 bytes, L2 buffer: 28048 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	AT_HYPERRAM_CL_EVENT UchanHR1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	unsigned int _P_Out, _C_Out;
	unsigned int _SPP_Out, _SP_Out, _SC_Out;
	unsigned int _LPP_Out, _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 8, Tiled: 1][Tile0 Dim: 8][D0 Dim: Init: 8, Tiled: 4]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 32 logical tiles, 32 physical tiles
			Total Size: 76800 [D0, [3 x 19200, 19200]][Tile0, 8:[80x9, 6:80x10, 80x5], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 8:[80x9, 2:80x10, 80x5], 2][D0, [3 x 19200, 19200]]
		Tile0: [0, 2880, 1440], Tile1: [19200, 2880, 1440], Tile2; [38400, 2880, 1440]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 1152 [D1, [0 x 1152, 1152]][D0, [3 x 36, 36]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 1152, 1152]][D0, [3 x 36, 36]]
		Tile0: [0, 1152, 1152], Tile1: [0, 1152, 1152], Tile2; [0, 1152, 1152]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -2, Max Pipe Depth: 0
		KerArgItSpace: 8 logical tiles, 8 physical tiles
			Total Size: 76800 [D1, [0 x 76800, 76800]][Tile0, 8:[80x8, 6:80x8, 80x4], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 76800, 76800]][Tile0, 8:[80x8, 6:80x8, 80x4], 2]
		Tile0: [0, 10240, 1280], Tile1: [1280, 10240, 1280], Tile2; [2560, 10240, 1280]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 8 logical tiles, 1 physical tiles
			Total Size: 153600 [D1, [0 x 153600, 153600]][Tile0, 8:[80x8, 6:80x8, 80x4], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 153600, 153600]][Tile0, 8:[80x8, 6:80x8, 80x4], 4]
		Tile0: [0, 20480, 2560], Tile1: [0, 20480, 2560], Tile2; [0, 20480, 2560]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (resnet_L1_Memory+28048);
	KerArg0->W = (unsigned short int) (80);
	KerArg0->OutFeatures = (unsigned short int) (8);
	KerArg0->Bias = (short int * __restrict__) (resnet_L1_Memory+6400);
	KerArg0->NormBias = (signed char) (12);
	KerArg1->W = (unsigned short int) (80);
	KerArg1->UsedW = (unsigned short int) (80);
	KerArg1->InFeatures = (unsigned short int) (2);
	KerArg1->OutFeatures = (unsigned short int) (8);
	KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+6416);
	KerArg1->Out = (int * __restrict__) (resnet_L1_Memory+28048);
	KerArg1->Norm = (unsigned char) (15);
	KerArg1->TotalInFeatures = (short int) (8);
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (resnet_L1_Memory+28048);
	KerArg2->W = (unsigned short int) (80);
	KerArg2->Norm = (unsigned char) (15);
	KerArg2->InFeatures = (unsigned short int) (8);
	KerArg2->LB = (int) (-32768);
	KerArg2->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 2880, 9600, 1440, 0, &DmaR_Evt1);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+6400), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+6416), 1152, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Filter */
	_C_Out=0; _SC_Out=10240; _LC_Out=1280;
	_SPP_Out=0; _SP_Out=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<8; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==7), T0Ind_NextLast = ((T0Ind+1)==7);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->H = (unsigned short int) (T0Ind_Last?4:8);
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			for (D0Ind=0; D0Ind<4; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==3), D0Ind_NextLast = ((D0Ind+1)==3);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (19200); _LN_In = ((T0Ind_Last)?800:(1600-160*(T0Ind==0))); _SN_In = (2*_LN_In); 
				} else if (!(T0Ind_Last)) {
					_N_In = _N_In + (1280-(160*(T0Ind==0)))+(-57600); _LN_In = ((T0Ind_NextLast)?800:1600); _SN_In = (2*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-8800)+(-57600); _LN_In = (1440); _SN_In = (2*_LN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+3200*((D0Ind_Total+1)%2)),
							_SN_In, 9600, _LN_In, 0, &DmaR_Evt1);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (resnet_L1_Memory+0+3200*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (((T0Ind_Last)?5:10)-1*(T0Ind==0));
				KerArg1->UsedH = (unsigned short int) (((T0Ind_Last)?5:10)-1*(T0Ind==0));
				KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+6416+((D0Ind)*36));
				KerArg1->Pad = (v4s) ((v4s){1,1,1*(T0Ind==0),1*(T0Ind_Last)});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_DP_fp, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->H = (unsigned short int) (T0Ind_Last?4:8);
			KerArg2->Out = (short int * __restrict__) (resnet_L1_Memory+7568+10240*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerDP_fp, (void *) KerArg2);
			__CALL(KerDP_fp, KerArg2);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			if (_SPP_Out) AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR1); /* Wait previous uDMA write Out */
			if (_SP_Out) AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) Out+_P_Out), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory+178352+10240*((T0Ind_Total+-1)%2)),
						_SP_Out, 9600, _LP_Out, 1, &UchanHR1);
			AT_L2_COPY(0, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L2_Memory+178352+10240*((T0Ind_Total)%2)), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+7568+10240*((T0Ind_Total)%2)),
					_SC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SPP_Out = _SP_Out;_LPP_Out = _LP_Out;
			_P_Out = _C_Out;_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (1280); _LC_Out = ((T0Ind_NextLast)?640:1280); _SC_Out = (8*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	if (_SPP_Out) AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR1); /* Wait previous uDMA write Out */
	AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) Out+_P_Out), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory+178352+10240*((T0Ind_Total+-1)%2)), _SP_Out, 9600, _LP_Out, 1, &UchanHR1);
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR1); /* Wait current uDMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S11_MatAdd_8x60x80(
		short int * __restrict__ In1,
		short int * __restrict__ In2,
		short int * __restrict__ Out)

{
	/* Shared L1: 46080 bytes, L2 buffer: 46080 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaW_Evt1;
	AT_HYPERRAM_CL_EVENT UchanHR1;
	KerMat3_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last, D0Ind_NextLast, D0Ind_NextNextLast;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast, T0Ind_NextNextLast;
	/* User kernel arguments related variables */
	unsigned int _NN_In1;
	unsigned int _SN_In1, _SNN_In1;
	unsigned int _LN_In1, _LNN_In1;
	unsigned int _N_In2;
	unsigned int _SN_In2;
	unsigned int _LN_In2;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 8, Tiled: 1][Tile0 Dim: 10]
	Ker Arg: In1, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 2
		KerArgItSpace: 10 logical tiles, 10 physical tiles
			Total Size: 76800 [D0, [0 x 76800, 76800]][Tile0, 10:[60x8, 8:60x8, 60x8], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 76800, 76800]][Tile0, 10:[60x8, 8:60x8, 60x8], 2]
		Tile0: [0, 7680, 960], Tile1: [960, 7680, 960], Tile2; [1920, 7680, 960]
	Ker Arg: In2, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 10 logical tiles, 10 physical tiles
			Total Size: 76800 [D0, [0 x 76800, 76800]][Tile0, 10:[60x8, 8:60x8, 60x8], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 76800, 76800]][Tile0, 10:[60x8, 8:60x8, 60x8], 2]
		Tile0: [0, 7680, 960], Tile1: [960, 7680, 960], Tile2; [1920, 7680, 960]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 10 logical tiles, 10 physical tiles
			Total Size: 76800 [D0, [0 x 76800, 76800]][Tile0, 10:[60x8, 8:60x8, 60x8], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 76800, 76800]][Tile0, 10:[60x8, 8:60x8, 60x8], 2]
		Tile0: [0, 7680, 960], Tile1: [960, 7680, 960], Tile2; [1920, 7680, 960]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->W = (unsigned short int) (60);
	KerArg0->H = (unsigned short int) (8);
	KerArg0->N = (unsigned short int) (8);
	KerArg0->LB = (int) (-32768);
	KerArg0->UB = (int) (32767);
	KerArg0->In1_Q = (unsigned char) (10);
	KerArg0->In2_Q = (unsigned char) (11);
	KerArg0->Out_Q = (unsigned char) (10);
	/*================================= Read Tiles Prolog ===============================*/
	AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) In1+0), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory+178352+0), 7680, 9600, 960, 0, &UchanHR1);
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR1); /* Wait previous uDMA read In1 */
	AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) In1+960), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory+178352+7680), 7680, 9600, 960, 0, &UchanHR1);
	AT_L2_COPY(0, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L2_Memory+178352+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 7680, 0, &DmaR_Evt1);
	_NN_In1=960; _SN_In1=7680;
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+15360+0), 7680, 9600, 960, 0, &DmaR_Evt2);
	_N_In2=0;
	_C_Out=0; _SC_Out=7680; _LC_Out=960;
	_SP_Out=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1, D0Ind_NextLast = 1, D0Ind_NextNextLast = 1;
		for (T0Ind=0; T0Ind<10; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==9), T0Ind_NextLast = ((T0Ind+1)==9), T0Ind_NextNextLast = ((T0Ind+2)==9);
			/*================================= Prepare Tiles ===================================*/
			_SNN_In1 = 0;
			if (!(T0Ind_Last)) {
				if (!(T0Ind_NextLast)) {
					_NN_In1 = _NN_In1 + (960); _LNN_In1 = (960); _SNN_In1 = (8*_LNN_In1); 
				}
			}
			_SN_In2 = 0;
			if (!(T0Ind_Last)) {
				_N_In2 = _N_In2 + (960); _LN_In2 = (960); _SN_In2 = (8*_LN_In2); 
			}
			/*============================= End Prepare Tiles ===================================*/
			/*================================= Read Tiles ======================================*/
			AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR1); /* Wait previous uDMA read In1 */
			if (_SNN_In1) {
				AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) In1+_NN_In1), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory+178352+7680*((T0Ind_Total)%2)),
						_SNN_In1, 9600, _LNN_In1, 0, &UchanHR1);
			}
			AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
			if (_SN_In1) {
				AT_L2_COPY(0, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L2_Memory+178352+7680*((T0Ind_Total+1)%2)), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+7680*((T0Ind_Total+1)%2)),
						_SN_In1, 0, &DmaR_Evt1);
			}
			AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
			if (_SN_In2) {
				AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In2+_N_In2), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+15360+7680*((T0Ind_Total+1)%2)),
						_SN_In2, 9600, _LN_In2, 0, &DmaR_Evt2);
			}
			/*============================= End Read Tiles ======================================*/
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->In1 = (short int *__restrict__) (resnet_L1_Memory+0+7680*((T0Ind_Total)%2));
			KerArg0->In2 = (short int *__restrict__) (resnet_L1_Memory+15360+7680*((T0Ind_Total)%2));
			KerArg0->Out = (short int *__restrict__) (resnet_L1_Memory+30720+7680*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerParMatAddDynAdjust_fp, (void *) KerArg0);
			__CALL(KerParMatAddDynAdjust_fp, KerArg0);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+30720+7680*((T0Ind_Total)%2)),
					_SC_Out, 9600, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SN_In1 = _SNN_In1;_LN_In1 = _LNN_In1;
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (960); _LC_Out = (960); _SC_Out = (8*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S14_Conv2d_8x8x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out)

{
	/* Shared L1: 48528 bytes, L2 buffer: 28048 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 8, Tiled: 1][Tile0 Dim: 8][D0 Dim: Init: 8, Tiled: 4]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 32 logical tiles, 32 physical tiles
			Total Size: 76800 [D0, [3 x 19200, 19200]][Tile0, 8:[80x9, 6:80x10, 80x5], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 8:[80x9, 2:80x10, 80x5], 2][D0, [3 x 19200, 19200]]
		Tile0: [0, 2880, 1440], Tile1: [19200, 2880, 1440], Tile2; [38400, 2880, 1440]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 1152 [D1, [0 x 1152, 1152]][D0, [3 x 36, 36]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 1152, 1152]][D0, [3 x 36, 36]]
		Tile0: [0, 1152, 1152], Tile1: [0, 1152, 1152], Tile2; [0, 1152, 1152]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 8 logical tiles, 8 physical tiles
			Total Size: 76800 [D1, [0 x 76800, 76800]][Tile0, 8:[80x8, 6:80x8, 80x4], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 76800, 76800]][Tile0, 8:[80x8, 6:80x8, 80x4], 2]
		Tile0: [0, 10240, 1280], Tile1: [1280, 10240, 1280], Tile2; [2560, 10240, 1280]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 8 logical tiles, 1 physical tiles
			Total Size: 153600 [D1, [0 x 153600, 153600]][Tile0, 8:[80x8, 6:80x8, 80x4], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 153600, 153600]][Tile0, 8:[80x8, 6:80x8, 80x4], 4]
		Tile0: [0, 20480, 2560], Tile1: [0, 20480, 2560], Tile2; [0, 20480, 2560]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (resnet_L1_Memory+28048);
	KerArg0->W = (unsigned short int) (80);
	KerArg0->OutFeatures = (unsigned short int) (8);
	KerArg0->Bias = (short int * __restrict__) (resnet_L1_Memory+6400);
	KerArg0->NormBias = (signed char) (11);
	KerArg1->W = (unsigned short int) (80);
	KerArg1->UsedW = (unsigned short int) (80);
	KerArg1->InFeatures = (unsigned short int) (2);
	KerArg1->OutFeatures = (unsigned short int) (8);
	KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+6416);
	KerArg1->Out = (int * __restrict__) (resnet_L1_Memory+28048);
	KerArg1->Norm = (unsigned char) (15);
	KerArg1->TotalInFeatures = (short int) (8);
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (resnet_L1_Memory+28048);
	KerArg2->W = (unsigned short int) (80);
	KerArg2->Norm = (unsigned char) (15);
	KerArg2->InFeatures = (unsigned short int) (8);
	KerArg2->LB = (int) (0);
	KerArg2->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 2880, 9600, 1440, 0, &DmaR_Evt1);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+6400), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+6416), 1152, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Filter */
	_C_Out=0; _SC_Out=10240; _LC_Out=1280;
	_SP_Out=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<8; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==7), T0Ind_NextLast = ((T0Ind+1)==7);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->H = (unsigned short int) (T0Ind_Last?4:8);
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			for (D0Ind=0; D0Ind<4; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==3), D0Ind_NextLast = ((D0Ind+1)==3);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (19200); _LN_In = ((T0Ind_Last)?800:(1600-160*(T0Ind==0))); _SN_In = (2*_LN_In); 
				} else if (!(T0Ind_Last)) {
					_N_In = _N_In + (1280-(160*(T0Ind==0)))+(-57600); _LN_In = ((T0Ind_NextLast)?800:1600); _SN_In = (2*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-8800)+(-57600); _LN_In = (1440); _SN_In = (2*_LN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+3200*((D0Ind_Total+1)%2)),
							_SN_In, 9600, _LN_In, 0, &DmaR_Evt1);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (resnet_L1_Memory+0+3200*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (((T0Ind_Last)?5:10)-1*(T0Ind==0));
				KerArg1->UsedH = (unsigned short int) (((T0Ind_Last)?5:10)-1*(T0Ind==0));
				KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+6416+((D0Ind)*36));
				KerArg1->Pad = (v4s) ((v4s){1,1,1*(T0Ind==0),1*(T0Ind_Last)});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_DP_fp, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->H = (unsigned short int) (T0Ind_Last?4:8);
			KerArg2->Out = (short int * __restrict__) (resnet_L1_Memory+7568+10240*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerDP_fp, (void *) KerArg2);
			__CALL(KerDP_fp, KerArg2);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+7568+10240*((T0Ind_Total)%2)),
					_SC_Out, 9600, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (1280); _LC_Out = ((T0Ind_NextLast)?640:1280); _SC_Out = (8*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S17_Conv2d_8x8x3x3(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out)

{
	/* Shared L1: 48528 bytes, L2 buffer: 28048 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	AT_HYPERRAM_CL_EVENT UchanHR1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	unsigned int _P_Out, _C_Out;
	unsigned int _SPP_Out, _SP_Out, _SC_Out;
	unsigned int _LPP_Out, _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 8, Tiled: 1][Tile0 Dim: 8][D0 Dim: Init: 8, Tiled: 4]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 32 logical tiles, 32 physical tiles
			Total Size: 76800 [D0, [3 x 19200, 19200]][Tile0, 8:[80x9, 6:80x10, 80x5], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 8:[80x9, 2:80x10, 80x5], 2][D0, [3 x 19200, 19200]]
		Tile0: [0, 2880, 1440], Tile1: [19200, 2880, 1440], Tile2; [38400, 2880, 1440]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 1152 [D1, [0 x 1152, 1152]][D0, [3 x 36, 36]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 1152, 1152]][D0, [3 x 36, 36]]
		Tile0: [0, 1152, 1152], Tile1: [0, 1152, 1152], Tile2; [0, 1152, 1152]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -2, Max Pipe Depth: 0
		KerArgItSpace: 8 logical tiles, 8 physical tiles
			Total Size: 76800 [D1, [0 x 76800, 76800]][Tile0, 8:[80x8, 6:80x8, 80x4], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 76800, 76800]][Tile0, 8:[80x8, 6:80x8, 80x4], 2]
		Tile0: [0, 10240, 1280], Tile1: [1280, 10240, 1280], Tile2; [2560, 10240, 1280]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 8 logical tiles, 1 physical tiles
			Total Size: 153600 [D1, [0 x 153600, 153600]][Tile0, 8:[80x8, 6:80x8, 80x4], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 153600, 153600]][Tile0, 8:[80x8, 6:80x8, 80x4], 4]
		Tile0: [0, 20480, 2560], Tile1: [0, 20480, 2560], Tile2; [0, 20480, 2560]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (resnet_L1_Memory+28048);
	KerArg0->W = (unsigned short int) (80);
	KerArg0->OutFeatures = (unsigned short int) (8);
	KerArg0->Bias = (short int * __restrict__) (resnet_L1_Memory+6400);
	KerArg0->NormBias = (signed char) (11);
	KerArg1->W = (unsigned short int) (80);
	KerArg1->UsedW = (unsigned short int) (80);
	KerArg1->InFeatures = (unsigned short int) (2);
	KerArg1->OutFeatures = (unsigned short int) (8);
	KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+6416);
	KerArg1->Out = (int * __restrict__) (resnet_L1_Memory+28048);
	KerArg1->Norm = (unsigned char) (15);
	KerArg1->TotalInFeatures = (short int) (8);
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (resnet_L1_Memory+28048);
	KerArg2->W = (unsigned short int) (80);
	KerArg2->Norm = (unsigned char) (15);
	KerArg2->InFeatures = (unsigned short int) (8);
	KerArg2->LB = (int) (-32768);
	KerArg2->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 2880, 9600, 1440, 0, &DmaR_Evt1);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+6400), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+6416), 1152, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Filter */
	_C_Out=0; _SC_Out=10240; _LC_Out=1280;
	_SPP_Out=0; _SP_Out=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<8; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==7), T0Ind_NextLast = ((T0Ind+1)==7);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->H = (unsigned short int) (T0Ind_Last?4:8);
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			for (D0Ind=0; D0Ind<4; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==3), D0Ind_NextLast = ((D0Ind+1)==3);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (19200); _LN_In = ((T0Ind_Last)?800:(1600-160*(T0Ind==0))); _SN_In = (2*_LN_In); 
				} else if (!(T0Ind_Last)) {
					_N_In = _N_In + (1280-(160*(T0Ind==0)))+(-57600); _LN_In = ((T0Ind_NextLast)?800:1600); _SN_In = (2*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-8800)+(-57600); _LN_In = (1440); _SN_In = (2*_LN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+3200*((D0Ind_Total+1)%2)),
							_SN_In, 9600, _LN_In, 0, &DmaR_Evt1);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (resnet_L1_Memory+0+3200*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (((T0Ind_Last)?5:10)-1*(T0Ind==0));
				KerArg1->UsedH = (unsigned short int) (((T0Ind_Last)?5:10)-1*(T0Ind==0));
				KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+6416+((D0Ind)*36));
				KerArg1->Pad = (v4s) ((v4s){1,1,1*(T0Ind==0),1*(T0Ind_Last)});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_DP_fp, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->H = (unsigned short int) (T0Ind_Last?4:8);
			KerArg2->Out = (short int * __restrict__) (resnet_L1_Memory+7568+10240*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerDP_fp, (void *) KerArg2);
			__CALL(KerDP_fp, KerArg2);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			if (_SPP_Out) AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR1); /* Wait previous uDMA write Out */
			if (_SP_Out) AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) Out+_P_Out), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory+178352+10240*((T0Ind_Total+-1)%2)),
						_SP_Out, 9600, _LP_Out, 1, &UchanHR1);
			AT_L2_COPY(0, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L2_Memory+178352+10240*((T0Ind_Total)%2)), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+7568+10240*((T0Ind_Total)%2)),
					_SC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SPP_Out = _SP_Out;_LPP_Out = _LP_Out;
			_P_Out = _C_Out;_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (1280); _LC_Out = ((T0Ind_NextLast)?640:1280); _SC_Out = (8*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	if (_SPP_Out) AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR1); /* Wait previous uDMA write Out */
	AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) Out+_P_Out), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory+178352+10240*((T0Ind_Total+-1)%2)), _SP_Out, 9600, _LP_Out, 1, &UchanHR1);
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR1); /* Wait current uDMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S18_MatAdd_8x60x80(
		short int * __restrict__ In1,
		short int * __restrict__ In2,
		short int * __restrict__ Out)

{
	/* Shared L1: 46080 bytes, L2 buffer: 46080 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaW_Evt1;
	AT_HYPERRAM_CL_EVENT UchanHR1;
	KerMat3_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last, D0Ind_NextLast, D0Ind_NextNextLast;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast, T0Ind_NextNextLast;
	/* User kernel arguments related variables */
	unsigned int _NN_In1;
	unsigned int _SN_In1, _SNN_In1;
	unsigned int _LN_In1, _LNN_In1;
	unsigned int _N_In2;
	unsigned int _SN_In2;
	unsigned int _LN_In2;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 8, Tiled: 1][Tile0 Dim: 10]
	Ker Arg: In1, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 2
		KerArgItSpace: 10 logical tiles, 10 physical tiles
			Total Size: 76800 [D0, [0 x 76800, 76800]][Tile0, 10:[60x8, 8:60x8, 60x8], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 76800, 76800]][Tile0, 10:[60x8, 8:60x8, 60x8], 2]
		Tile0: [0, 7680, 960], Tile1: [960, 7680, 960], Tile2; [1920, 7680, 960]
	Ker Arg: In2, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 10 logical tiles, 10 physical tiles
			Total Size: 76800 [D0, [0 x 76800, 76800]][Tile0, 10:[60x8, 8:60x8, 60x8], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 76800, 76800]][Tile0, 10:[60x8, 8:60x8, 60x8], 2]
		Tile0: [0, 7680, 960], Tile1: [960, 7680, 960], Tile2; [1920, 7680, 960]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 10 logical tiles, 10 physical tiles
			Total Size: 76800 [D0, [0 x 76800, 76800]][Tile0, 10:[60x8, 8:60x8, 60x8], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 76800, 76800]][Tile0, 10:[60x8, 8:60x8, 60x8], 2]
		Tile0: [0, 7680, 960], Tile1: [960, 7680, 960], Tile2; [1920, 7680, 960]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->W = (unsigned short int) (60);
	KerArg0->H = (unsigned short int) (8);
	KerArg0->N = (unsigned short int) (8);
	KerArg0->LB = (int) (-32768);
	KerArg0->UB = (int) (32767);
	KerArg0->In1_Q = (unsigned char) (10);
	KerArg0->In2_Q = (unsigned char) (10);
	KerArg0->Out_Q = (unsigned char) (9);
	/*================================= Read Tiles Prolog ===============================*/
	AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) In1+0), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory+178352+0), 7680, 9600, 960, 0, &UchanHR1);
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR1); /* Wait previous uDMA read In1 */
	AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) In1+960), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory+178352+7680), 7680, 9600, 960, 0, &UchanHR1);
	AT_L2_COPY(0, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L2_Memory+178352+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 7680, 0, &DmaR_Evt1);
	_NN_In1=960; _SN_In1=7680;
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+15360+0), 7680, 9600, 960, 0, &DmaR_Evt2);
	_N_In2=0;
	_C_Out=0; _SC_Out=7680; _LC_Out=960;
	_SP_Out=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1, D0Ind_NextLast = 1, D0Ind_NextNextLast = 1;
		for (T0Ind=0; T0Ind<10; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==9), T0Ind_NextLast = ((T0Ind+1)==9), T0Ind_NextNextLast = ((T0Ind+2)==9);
			/*================================= Prepare Tiles ===================================*/
			_SNN_In1 = 0;
			if (!(T0Ind_Last)) {
				if (!(T0Ind_NextLast)) {
					_NN_In1 = _NN_In1 + (960); _LNN_In1 = (960); _SNN_In1 = (8*_LNN_In1); 
				}
			}
			_SN_In2 = 0;
			if (!(T0Ind_Last)) {
				_N_In2 = _N_In2 + (960); _LN_In2 = (960); _SN_In2 = (8*_LN_In2); 
			}
			/*============================= End Prepare Tiles ===================================*/
			/*================================= Read Tiles ======================================*/
			AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR1); /* Wait previous uDMA read In1 */
			if (_SNN_In1) {
				AT_HYPERRAM_CL_COPY2D(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) In1+_NN_In1), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory+178352+7680*((T0Ind_Total)%2)),
						_SNN_In1, 9600, _LNN_In1, 0, &UchanHR1);
			}
			AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
			if (_SN_In1) {
				AT_L2_COPY(0, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L2_Memory+178352+7680*((T0Ind_Total+1)%2)), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+7680*((T0Ind_Total+1)%2)),
						_SN_In1, 0, &DmaR_Evt1);
			}
			AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
			if (_SN_In2) {
				AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In2+_N_In2), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+15360+7680*((T0Ind_Total+1)%2)),
						_SN_In2, 9600, _LN_In2, 0, &DmaR_Evt2);
			}
			/*============================= End Read Tiles ======================================*/
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->In1 = (short int *__restrict__) (resnet_L1_Memory+0+7680*((T0Ind_Total)%2));
			KerArg0->In2 = (short int *__restrict__) (resnet_L1_Memory+15360+7680*((T0Ind_Total)%2));
			KerArg0->Out = (short int *__restrict__) (resnet_L1_Memory+30720+7680*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerParMatAddDynAdjust_fp, (void *) KerArg0);
			__CALL(KerParMatAddDynAdjust_fp, KerArg0);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+30720+7680*((T0Ind_Total)%2)),
					_SC_Out, 9600, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SN_In1 = _SNN_In1;_LN_In1 = _LNN_In1;
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (960); _LC_Out = (960); _SC_Out = (8*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S21_Conv2d_16x8x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out)

{
	/* Shared L1: 47776 bytes, L2 buffer: 29856 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 5][D0 Dim: Init: 8, Tiled: 4]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 5 logical tiles, 5 physical tiles
			Total Size: 38400 [D1, [0 x 38400, 38400]][Tile0, 5:[40x7, 3:40x7, 40x2], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 38400, 38400]][Tile0, 5:[40x7, 3:40x7, 40x2], 2]
		Tile0: [0, 8960, 560], Tile1: [560, 8960, 560], Tile2; [1120, 8960, 560]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [D1, [0 x 32, 32]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 32, 32]]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 2304 [D1, [0 x 2304, 2304]][D0, [3 x 576, 576]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 2304, 2304]][D0, [3 x 576, 576]]
		Tile0: [0, 2304, 2304], Tile1: [0, 2304, 2304], Tile2; [0, 2304, 2304]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 20 logical tiles, 20 physical tiles
			Total Size: 76800 [D0, [3 x 19200, 19200]][Tile0, 5:[80x15, 3:80x15, 80x4], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 5:[80x15, 2:80x15, 80x4], 2][D0, [3 x 19200, 19200]]
		Tile0: [0, 4800, 2400], Tile1: [19200, 4800, 2400], Tile2; [38400, 4800, 2400]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 5 logical tiles, 1 physical tiles
			Total Size: 76800 [D1, [0 x 76800, 76800]][Tile0, 5:[40x7, 3:40x7, 40x2], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 76800, 76800]][Tile0, 5:[40x7, 3:40x7, 40x2], 4]
		Tile0: [0, 17920, 1120], Tile1: [0, 17920, 1120], Tile2; [0, 17920, 1120]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (resnet_L1_Memory+29856);
	KerArg0->W = (unsigned short int) (40);
	KerArg0->OutFeatures = (unsigned short int) (16);
	KerArg0->Bias = (short int * __restrict__) (resnet_L1_Memory+9600);
	KerArg0->NormBias = (signed char) (9);
	KerArg1->W = (unsigned short int) (80);
	KerArg1->UsedW = (unsigned short int) (80);
	KerArg1->InFeatures = (unsigned short int) (2);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+9632);
	KerArg1->Out = (int * __restrict__) (resnet_L1_Memory+29856);
	KerArg1->Norm = (unsigned char) (13);
	KerArg1->TotalInFeatures = (short int) (8);
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (resnet_L1_Memory+29856);
	KerArg2->W = (unsigned short int) (40);
	KerArg2->Norm = (unsigned char) (13);
	KerArg2->InFeatures = (unsigned short int) (16);
	KerArg2->LB = (int) (0);
	KerArg2->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=8960; _LC_Out=560;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+9600), 32, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+9632), 2304, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Filter */
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 4800, 9600, 2400, 0, &DmaR_Evt3);
	_N_In=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<5; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==4), T0Ind_NextLast = ((T0Ind+1)==4);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->H = (unsigned short int) (T0Ind_Last?2:7);
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			for (D0Ind=0; D0Ind<4; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==3), D0Ind_NextLast = ((D0Ind+1)==3);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (19200); _LN_In = ((T0Ind_Last)?640:(2400-0*(T0Ind==0))); _SN_In = (2*_LN_In); 
				} else if (!(T0Ind_Last)) {
					_N_In = _N_In + (2240)+(-57600); _LN_In = ((T0Ind_NextLast)?640:2400); _SN_In = (2*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-8960)+(-57600); _LN_In = (2400); _SN_In = (2*_LN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+4800*((D0Ind_Total+1)%2)),
							_SN_In, 9600, _LN_In, 0, &DmaR_Evt3);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (resnet_L1_Memory+0+4800*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (((T0Ind_Last)?4:15)-0*(T0Ind==0));
				KerArg1->UsedH = (unsigned short int) (((T0Ind_Last)?4:15)-0*(T0Ind==0));
				KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+9632+((D0Ind)*36));
				KerArg1->Pad = (v4s) ((v4s){0,1,0*(T0Ind==0),1*(T0Ind_Last)});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride2_DP_fp, (void *) KerArg1);
				__CALL(KerParConv3x3Stride2_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->H = (unsigned short int) (T0Ind_Last?2:7);
			KerArg2->Out = (short int * __restrict__) (resnet_L1_Memory+11936+8960*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerDP_fp, (void *) KerArg2);
			__CALL(KerDP_fp, KerArg2);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+11936+8960*((T0Ind_Total)%2)),
					_SC_Out, 2400, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (560); _LC_Out = ((T0Ind_NextLast)?160:560); _SC_Out = (16*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S24_Conv2d_16x8x1x1(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out)

{
	/* Shared L1: 38688 bytes, L2 buffer: 28448 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 8][D0 Dim: Init: 8, Tiled: 1]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 8 logical tiles, 8 physical tiles
			Total Size: 38400 [D1, [0 x 38400, 38400]][Tile0, 8:[40x4, 6:40x4, 40x2], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 38400, 38400]][Tile0, 8:[40x4, 6:40x4, 40x2], 2]
		Tile0: [0, 5120, 320], Tile1: [320, 5120, 320], Tile2; [640, 5120, 320]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [D1, [0 x 32, 32]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 32, 32]]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [D1, [0 x 256, 256]][D0, [0 x 256, 256]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 256, 256]][D0, [0 x 256, 256]]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 8 logical tiles, 8 physical tiles
			Total Size: 76800 [D0, [0 x 76800, 76800]][Tile0, 8:[80x7, 6:80x7, 80x3], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 8:[80x7], 2][D0, [0 x 76800, 76800]]
		Tile0: [0, 8960, 1120], Tile1: [1280, 8960, 1120], Tile2; [2560, 8960, 1120]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 8 logical tiles, 1 physical tiles
			Total Size: 76800 [D1, [0 x 76800, 76800]][Tile0, 8:[40x4, 6:40x4, 40x2], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 76800, 76800]][Tile0, 8:[40x4, 6:40x4, 40x2], 4]
		Tile0: [0, 10240, 640], Tile1: [0, 10240, 640], Tile2; [0, 10240, 640]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (resnet_L1_Memory+28448);
	KerArg0->W = (unsigned short int) (40);
	KerArg0->OutFeatures = (unsigned short int) (16);
	KerArg0->Bias = (short int * __restrict__) (resnet_L1_Memory+17920);
	KerArg0->NormBias = (signed char) (9);
	KerArg1->W = (unsigned short int) (80);
	KerArg1->UsedW = (unsigned short int) (79);
	KerArg1->InFeatures = (unsigned short int) (8);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+17952);
	KerArg1->Out = (int * __restrict__) (resnet_L1_Memory+28448);
	KerArg1->Norm = (unsigned char) (14);
	KerArg1->TotalInFeatures = (short int) (8);
	KerArg1->Pad = (v4s) 0;
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (resnet_L1_Memory+28448);
	KerArg2->W = (unsigned short int) (40);
	KerArg2->Norm = (unsigned char) (14);
	KerArg2->InFeatures = (unsigned short int) (16);
	KerArg2->LB = (int) (-32768);
	KerArg2->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=5120; _LC_Out=320;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+17920), 32, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+17952), 256, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Filter */
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 8960, 9600, 1120, 0, &DmaR_Evt3);
	_N_In=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<8; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==7), T0Ind_NextLast = ((T0Ind+1)==7);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->H = (unsigned short int) (T0Ind_Last?2:4);
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1, D0Ind_NextLast = 1;
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(T0Ind_Last)) {
					_N_In = _N_In + (1280); _LN_In = ((T0Ind_NextLast)?480:1120); _SN_In = (8*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-8960); _LN_In = (1120); _SN_In = (8*_LN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+8960*((D0Ind_Total+1)%2)),
							_SN_In, 9600, _LN_In, 0, &DmaR_Evt3);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (resnet_L1_Memory+0+8960*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (T0Ind_Last?3:7);
				KerArg1->UsedH = (unsigned short int) (T0Ind_Last?3:7);
				AT_FORK(gap_ncore(), (void *) KerParConv1x1Stride2_DP_fp, (void *) KerArg1);
				__CALL(KerParConv1x1Stride2_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
				D0Ind_Total++;
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->H = (unsigned short int) (T0Ind_Last?2:4);
			KerArg2->Out = (short int * __restrict__) (resnet_L1_Memory+18208+5120*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerDP_fp, (void *) KerArg2);
			__CALL(KerDP_fp, KerArg2);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+18208+5120*((T0Ind_Total)%2)),
					_SC_Out, 2400, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (320); _LC_Out = ((T0Ind_NextLast)?160:320); _SC_Out = (16*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S27_Conv2d_16x16x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out)

{
	/* Shared L1: 48160 bytes, L2 buffer: 32800 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 5][D0 Dim: Init: 16, Tiled: 2]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 10 logical tiles, 10 physical tiles
			Total Size: 38400 [D0, [1 x 24000, 14400]][Tile0, 5:[40x7, 3:40x8, 40x7], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 5:[40x7, 40x7], 2][D0, [1 x 24000, 14400]]
		Tile0: [0, 5600, 560], Tile1: [24000, 3360, 560], Tile2; [400, 6400, 640]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [D1, [0 x 32, 32]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 32, 32]]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 4608 [D1, [0 x 4608, 4608]][D0, [1 x 2880, 1728]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4608, 4608]][D0, [1 x 2880, 1728]]
		Tile0: [0, 4608, 4608], Tile1: [0, 4608, 4608], Tile2; [0, 4608, 4608]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 5 logical tiles, 5 physical tiles
			Total Size: 38400 [D1, [0 x 38400, 38400]][Tile0, 5:[40x6, 3:40x6, 40x6], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 38400, 38400]][Tile0, 5:[40x6, 3:40x6, 40x6], 2]
		Tile0: [0, 7680, 480], Tile1: [480, 7680, 480], Tile2; [960, 7680, 480]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 5 logical tiles, 1 physical tiles
			Total Size: 76800 [D1, [0 x 76800, 76800]][Tile0, 5:[40x6, 3:40x6, 40x6], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 76800, 76800]][Tile0, 5:[40x6, 3:40x6, 40x6], 4]
		Tile0: [0, 15360, 960], Tile1: [0, 15360, 960], Tile2; [0, 15360, 960]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (resnet_L1_Memory+32800);
	KerArg0->W = (unsigned short int) (40);
	KerArg0->H = (unsigned short int) (6);
	KerArg0->OutFeatures = (unsigned short int) (16);
	KerArg0->Bias = (short int * __restrict__) (resnet_L1_Memory+12800);
	KerArg0->NormBias = (signed char) (13);
	KerArg1->W = (unsigned short int) (40);
	KerArg1->UsedW = (unsigned short int) (40);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+12832);
	KerArg1->Out = (int * __restrict__) (resnet_L1_Memory+32800);
	KerArg1->Norm = (unsigned char) (15);
	KerArg1->TotalInFeatures = (short int) (16);
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (resnet_L1_Memory+32800);
	KerArg2->W = (unsigned short int) (40);
	KerArg2->H = (unsigned short int) (6);
	KerArg2->Norm = (unsigned char) (15);
	KerArg2->InFeatures = (unsigned short int) (16);
	KerArg2->LB = (int) (0);
	KerArg2->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 5600, 2400, 560, 0, &DmaR_Evt1);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+12800), 32, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+12832), 4608, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Filter */
	_C_Out=0; _SC_Out=7680; _LC_Out=480;
	_SP_Out=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<5; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==4), T0Ind_NextLast = ((T0Ind+1)==4);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			for (D0Ind=0; D0Ind<2; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==1), D0Ind_NextLast = ((D0Ind+1)==1);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (24000); _LN_In = ((T0Ind_Last)?560:(640-80*(T0Ind==0))); _SN_In = (((1)?6:10)*_LN_In); 
				} else if (!(T0Ind_Last)) {
					_N_In = _N_In + (480-(80*(T0Ind==0)))+(-24000); _LN_In = ((T0Ind_NextLast)?560:640); _SN_In = (10*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-1840)+(-24000); _LN_In = (560); _SN_In = (10*_LN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+6400*((D0Ind_Total+1)%2)),
							_SN_In, 2400, _LN_In, 0, &DmaR_Evt1);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (resnet_L1_Memory+0+6400*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (8-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->UsedH = (unsigned short int) (8-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->InFeatures = (unsigned short int) (D0Ind_Last?6:10);
				KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+12832+((D0Ind)*180));
				KerArg1->Pad = (v4s) ((v4s){1,1,1*(T0Ind==0),1*(T0Ind_Last)});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_DP_fp, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->Out = (short int * __restrict__) (resnet_L1_Memory+17440+7680*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerDP_fp, (void *) KerArg2);
			__CALL(KerDP_fp, KerArg2);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+17440+7680*((T0Ind_Total)%2)),
					_SC_Out, 2400, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (480); _LC_Out = (480); _SC_Out = (16*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S28_MatAdd_16x30x40(
		short int * __restrict__ In1,
		short int * __restrict__ In2,
		short int * __restrict__ Out)

{
	/* Shared L1: 46080 bytes, L2 buffer: 46080 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last, D0Ind_NextLast;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In1;
	unsigned int _SN_In1;
	unsigned int _LN_In1;
	unsigned int _N_In2;
	unsigned int _SN_In2;
	unsigned int _LN_In2;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 5]
	Ker Arg: In1, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 5 logical tiles, 5 physical tiles
			Total Size: 38400 [D0, [0 x 38400, 38400]][Tile0, 5:[30x8, 3:30x8, 30x8], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 38400, 38400]][Tile0, 5:[30x8, 3:30x8, 30x8], 2]
		Tile0: [0, 7680, 480], Tile1: [480, 7680, 480], Tile2; [960, 7680, 480]
	Ker Arg: In2, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 5 logical tiles, 5 physical tiles
			Total Size: 38400 [D0, [0 x 38400, 38400]][Tile0, 5:[30x8, 3:30x8, 30x8], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 38400, 38400]][Tile0, 5:[30x8, 3:30x8, 30x8], 2]
		Tile0: [0, 7680, 480], Tile1: [480, 7680, 480], Tile2; [960, 7680, 480]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 5 logical tiles, 5 physical tiles
			Total Size: 38400 [D0, [0 x 38400, 38400]][Tile0, 5:[30x8, 3:30x8, 30x8], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 38400, 38400]][Tile0, 5:[30x8, 3:30x8, 30x8], 2]
		Tile0: [0, 7680, 480], Tile1: [480, 7680, 480], Tile2; [960, 7680, 480]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->W = (unsigned short int) (30);
	KerArg0->H = (unsigned short int) (8);
	KerArg0->N = (unsigned short int) (16);
	KerArg0->LB = (int) (-32768);
	KerArg0->UB = (int) (32767);
	KerArg0->In1_Q = (unsigned char) (11);
	KerArg0->In2_Q = (unsigned char) (10);
	KerArg0->Out_Q = (unsigned char) (9);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 7680, 2400, 480, 0, &DmaR_Evt1);
	_N_In1=0;
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+15360+0), 7680, 2400, 480, 0, &DmaR_Evt2);
	_N_In2=0;
	_C_Out=0; _SC_Out=7680; _LC_Out=480;
	_SP_Out=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1, D0Ind_NextLast = 1;
		for (T0Ind=0; T0Ind<5; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==4), T0Ind_NextLast = ((T0Ind+1)==4);
			/*================================= Prepare Tiles ===================================*/
			_SN_In1 = 0;
			if (!(T0Ind_Last)) {
				_N_In1 = _N_In1 + (480); _LN_In1 = (480); _SN_In1 = (16*_LN_In1); 
			}
			_SN_In2 = 0;
			if (!(T0Ind_Last)) {
				_N_In2 = _N_In2 + (480); _LN_In2 = (480); _SN_In2 = (16*_LN_In2); 
			}
			/*============================= End Prepare Tiles ===================================*/
			/*================================= Read Tiles ======================================*/
			AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
			if (_SN_In1) {
				AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In1+_N_In1), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+7680*((T0Ind_Total+1)%2)),
						_SN_In1, 2400, _LN_In1, 0, &DmaR_Evt1);
			}
			AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
			if (_SN_In2) {
				AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In2+_N_In2), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+15360+7680*((T0Ind_Total+1)%2)),
						_SN_In2, 2400, _LN_In2, 0, &DmaR_Evt2);
			}
			/*============================= End Read Tiles ======================================*/
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->In1 = (short int *__restrict__) (resnet_L1_Memory+0+7680*((T0Ind_Total)%2));
			KerArg0->In2 = (short int *__restrict__) (resnet_L1_Memory+15360+7680*((T0Ind_Total)%2));
			KerArg0->Out = (short int *__restrict__) (resnet_L1_Memory+30720+7680*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerParMatAddDynAdjust_fp, (void *) KerArg0);
			__CALL(KerParMatAddDynAdjust_fp, KerArg0);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+30720+7680*((T0Ind_Total)%2)),
					_SC_Out, 2400, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (480); _LC_Out = (480); _SC_Out = (16*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S31_Conv2d_16x16x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out)

{
	/* Shared L1: 48160 bytes, L2 buffer: 32800 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 5][D0 Dim: Init: 16, Tiled: 2]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 10 logical tiles, 10 physical tiles
			Total Size: 38400 [D0, [1 x 24000, 14400]][Tile0, 5:[40x7, 3:40x8, 40x7], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 5:[40x7, 40x7], 2][D0, [1 x 24000, 14400]]
		Tile0: [0, 5600, 560], Tile1: [24000, 3360, 560], Tile2; [400, 6400, 640]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [D1, [0 x 32, 32]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 32, 32]]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 4608 [D1, [0 x 4608, 4608]][D0, [1 x 2880, 1728]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4608, 4608]][D0, [1 x 2880, 1728]]
		Tile0: [0, 4608, 4608], Tile1: [0, 4608, 4608], Tile2; [0, 4608, 4608]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 5 logical tiles, 5 physical tiles
			Total Size: 38400 [D1, [0 x 38400, 38400]][Tile0, 5:[40x6, 3:40x6, 40x6], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 38400, 38400]][Tile0, 5:[40x6, 3:40x6, 40x6], 2]
		Tile0: [0, 7680, 480], Tile1: [480, 7680, 480], Tile2; [960, 7680, 480]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 5 logical tiles, 1 physical tiles
			Total Size: 76800 [D1, [0 x 76800, 76800]][Tile0, 5:[40x6, 3:40x6, 40x6], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 76800, 76800]][Tile0, 5:[40x6, 3:40x6, 40x6], 4]
		Tile0: [0, 15360, 960], Tile1: [0, 15360, 960], Tile2; [0, 15360, 960]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (resnet_L1_Memory+32800);
	KerArg0->W = (unsigned short int) (40);
	KerArg0->H = (unsigned short int) (6);
	KerArg0->OutFeatures = (unsigned short int) (16);
	KerArg0->Bias = (short int * __restrict__) (resnet_L1_Memory+12800);
	KerArg0->NormBias = (signed char) (10);
	KerArg1->W = (unsigned short int) (40);
	KerArg1->UsedW = (unsigned short int) (40);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+12832);
	KerArg1->Out = (int * __restrict__) (resnet_L1_Memory+32800);
	KerArg1->Norm = (unsigned char) (13);
	KerArg1->TotalInFeatures = (short int) (16);
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (resnet_L1_Memory+32800);
	KerArg2->W = (unsigned short int) (40);
	KerArg2->H = (unsigned short int) (6);
	KerArg2->Norm = (unsigned char) (13);
	KerArg2->InFeatures = (unsigned short int) (16);
	KerArg2->LB = (int) (0);
	KerArg2->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 5600, 2400, 560, 0, &DmaR_Evt1);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+12800), 32, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+12832), 4608, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Filter */
	_C_Out=0; _SC_Out=7680; _LC_Out=480;
	_SP_Out=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<5; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==4), T0Ind_NextLast = ((T0Ind+1)==4);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			for (D0Ind=0; D0Ind<2; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==1), D0Ind_NextLast = ((D0Ind+1)==1);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (24000); _LN_In = ((T0Ind_Last)?560:(640-80*(T0Ind==0))); _SN_In = (((1)?6:10)*_LN_In); 
				} else if (!(T0Ind_Last)) {
					_N_In = _N_In + (480-(80*(T0Ind==0)))+(-24000); _LN_In = ((T0Ind_NextLast)?560:640); _SN_In = (10*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-1840)+(-24000); _LN_In = (560); _SN_In = (10*_LN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+6400*((D0Ind_Total+1)%2)),
							_SN_In, 2400, _LN_In, 0, &DmaR_Evt1);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (resnet_L1_Memory+0+6400*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (8-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->UsedH = (unsigned short int) (8-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->InFeatures = (unsigned short int) (D0Ind_Last?6:10);
				KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+12832+((D0Ind)*180));
				KerArg1->Pad = (v4s) ((v4s){1,1,1*(T0Ind==0),1*(T0Ind_Last)});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_DP_fp, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->Out = (short int * __restrict__) (resnet_L1_Memory+17440+7680*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerDP_fp, (void *) KerArg2);
			__CALL(KerDP_fp, KerArg2);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+17440+7680*((T0Ind_Total)%2)),
					_SC_Out, 2400, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (480); _LC_Out = (480); _SC_Out = (16*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S34_Conv2d_16x16x3x3(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out)

{
	/* Shared L1: 48160 bytes, L2 buffer: 32800 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 5][D0 Dim: Init: 16, Tiled: 2]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 10 logical tiles, 10 physical tiles
			Total Size: 38400 [D0, [1 x 24000, 14400]][Tile0, 5:[40x7, 3:40x8, 40x7], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 5:[40x7, 40x7], 2][D0, [1 x 24000, 14400]]
		Tile0: [0, 5600, 560], Tile1: [24000, 3360, 560], Tile2; [400, 6400, 640]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [D1, [0 x 32, 32]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 32, 32]]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 4608 [D1, [0 x 4608, 4608]][D0, [1 x 2880, 1728]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4608, 4608]][D0, [1 x 2880, 1728]]
		Tile0: [0, 4608, 4608], Tile1: [0, 4608, 4608], Tile2; [0, 4608, 4608]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 5 logical tiles, 5 physical tiles
			Total Size: 38400 [D1, [0 x 38400, 38400]][Tile0, 5:[40x6, 3:40x6, 40x6], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 38400, 38400]][Tile0, 5:[40x6, 3:40x6, 40x6], 2]
		Tile0: [0, 7680, 480], Tile1: [480, 7680, 480], Tile2; [960, 7680, 480]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 5 logical tiles, 1 physical tiles
			Total Size: 76800 [D1, [0 x 76800, 76800]][Tile0, 5:[40x6, 3:40x6, 40x6], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 76800, 76800]][Tile0, 5:[40x6, 3:40x6, 40x6], 4]
		Tile0: [0, 15360, 960], Tile1: [0, 15360, 960], Tile2; [0, 15360, 960]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (resnet_L1_Memory+32800);
	KerArg0->W = (unsigned short int) (40);
	KerArg0->H = (unsigned short int) (6);
	KerArg0->OutFeatures = (unsigned short int) (16);
	KerArg0->Bias = (short int * __restrict__) (resnet_L1_Memory+12800);
	KerArg0->NormBias = (signed char) (13);
	KerArg1->W = (unsigned short int) (40);
	KerArg1->UsedW = (unsigned short int) (40);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+12832);
	KerArg1->Out = (int * __restrict__) (resnet_L1_Memory+32800);
	KerArg1->Norm = (unsigned char) (16);
	KerArg1->TotalInFeatures = (short int) (16);
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (resnet_L1_Memory+32800);
	KerArg2->W = (unsigned short int) (40);
	KerArg2->H = (unsigned short int) (6);
	KerArg2->Norm = (unsigned char) (16);
	KerArg2->InFeatures = (unsigned short int) (16);
	KerArg2->LB = (int) (-32768);
	KerArg2->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 5600, 2400, 560, 0, &DmaR_Evt1);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+12800), 32, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+12832), 4608, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Filter */
	_C_Out=0; _SC_Out=7680; _LC_Out=480;
	_SP_Out=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<5; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==4), T0Ind_NextLast = ((T0Ind+1)==4);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			for (D0Ind=0; D0Ind<2; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==1), D0Ind_NextLast = ((D0Ind+1)==1);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (24000); _LN_In = ((T0Ind_Last)?560:(640-80*(T0Ind==0))); _SN_In = (((1)?6:10)*_LN_In); 
				} else if (!(T0Ind_Last)) {
					_N_In = _N_In + (480-(80*(T0Ind==0)))+(-24000); _LN_In = ((T0Ind_NextLast)?560:640); _SN_In = (10*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-1840)+(-24000); _LN_In = (560); _SN_In = (10*_LN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+6400*((D0Ind_Total+1)%2)),
							_SN_In, 2400, _LN_In, 0, &DmaR_Evt1);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (resnet_L1_Memory+0+6400*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (8-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->UsedH = (unsigned short int) (8-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->InFeatures = (unsigned short int) (D0Ind_Last?6:10);
				KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+12832+((D0Ind)*180));
				KerArg1->Pad = (v4s) ((v4s){1,1,1*(T0Ind==0),1*(T0Ind_Last)});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_DP_fp, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->Out = (short int * __restrict__) (resnet_L1_Memory+17440+7680*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerDP_fp, (void *) KerArg2);
			__CALL(KerDP_fp, KerArg2);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+17440+7680*((T0Ind_Total)%2)),
					_SC_Out, 2400, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (480); _LC_Out = (480); _SC_Out = (16*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S35_MatAdd_16x30x40(
		short int * __restrict__ In1,
		short int * __restrict__ In2,
		short int * __restrict__ Out)

{
	/* Shared L1: 46080 bytes, L2 buffer: 46080 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last, D0Ind_NextLast;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In1;
	unsigned int _SN_In1;
	unsigned int _LN_In1;
	unsigned int _N_In2;
	unsigned int _SN_In2;
	unsigned int _LN_In2;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 5]
	Ker Arg: In1, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 5 logical tiles, 5 physical tiles
			Total Size: 38400 [D0, [0 x 38400, 38400]][Tile0, 5:[30x8, 3:30x8, 30x8], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 38400, 38400]][Tile0, 5:[30x8, 3:30x8, 30x8], 2]
		Tile0: [0, 7680, 480], Tile1: [480, 7680, 480], Tile2; [960, 7680, 480]
	Ker Arg: In2, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 5 logical tiles, 5 physical tiles
			Total Size: 38400 [D0, [0 x 38400, 38400]][Tile0, 5:[30x8, 3:30x8, 30x8], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 38400, 38400]][Tile0, 5:[30x8, 3:30x8, 30x8], 2]
		Tile0: [0, 7680, 480], Tile1: [480, 7680, 480], Tile2; [960, 7680, 480]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 5 logical tiles, 5 physical tiles
			Total Size: 38400 [D0, [0 x 38400, 38400]][Tile0, 5:[30x8, 3:30x8, 30x8], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 38400, 38400]][Tile0, 5:[30x8, 3:30x8, 30x8], 2]
		Tile0: [0, 7680, 480], Tile1: [480, 7680, 480], Tile2; [960, 7680, 480]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->W = (unsigned short int) (30);
	KerArg0->H = (unsigned short int) (8);
	KerArg0->N = (unsigned short int) (16);
	KerArg0->LB = (int) (-32768);
	KerArg0->UB = (int) (32767);
	KerArg0->In1_Q = (unsigned char) (10);
	KerArg0->In2_Q = (unsigned char) (9);
	KerArg0->Out_Q = (unsigned char) (9);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 7680, 2400, 480, 0, &DmaR_Evt1);
	_N_In1=0;
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+15360+0), 7680, 2400, 480, 0, &DmaR_Evt2);
	_N_In2=0;
	_C_Out=0; _SC_Out=7680; _LC_Out=480;
	_SP_Out=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1, D0Ind_NextLast = 1;
		for (T0Ind=0; T0Ind<5; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==4), T0Ind_NextLast = ((T0Ind+1)==4);
			/*================================= Prepare Tiles ===================================*/
			_SN_In1 = 0;
			if (!(T0Ind_Last)) {
				_N_In1 = _N_In1 + (480); _LN_In1 = (480); _SN_In1 = (16*_LN_In1); 
			}
			_SN_In2 = 0;
			if (!(T0Ind_Last)) {
				_N_In2 = _N_In2 + (480); _LN_In2 = (480); _SN_In2 = (16*_LN_In2); 
			}
			/*============================= End Prepare Tiles ===================================*/
			/*================================= Read Tiles ======================================*/
			AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
			if (_SN_In1) {
				AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In1+_N_In1), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+7680*((T0Ind_Total+1)%2)),
						_SN_In1, 2400, _LN_In1, 0, &DmaR_Evt1);
			}
			AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
			if (_SN_In2) {
				AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In2+_N_In2), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+15360+7680*((T0Ind_Total+1)%2)),
						_SN_In2, 2400, _LN_In2, 0, &DmaR_Evt2);
			}
			/*============================= End Read Tiles ======================================*/
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->In1 = (short int *__restrict__) (resnet_L1_Memory+0+7680*((T0Ind_Total)%2));
			KerArg0->In2 = (short int *__restrict__) (resnet_L1_Memory+15360+7680*((T0Ind_Total)%2));
			KerArg0->Out = (short int *__restrict__) (resnet_L1_Memory+30720+7680*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerParMatAddDynAdjust_fp, (void *) KerArg0);
			__CALL(KerParMatAddDynAdjust_fp, KerArg0);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+30720+7680*((T0Ind_Total)%2)),
					_SC_Out, 2400, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (480); _LC_Out = (480); _SC_Out = (16*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S38_Conv2d_32x16x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out)

{
	/* Shared L1: 45440 bytes, L2 buffer: 32640 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 32, Tiled: 1][Tile0 Dim: 3][D0 Dim: Init: 16, Tiled: 3]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 3 physical tiles
			Total Size: 19200 [D1, [0 x 19200, 19200]][Tile0, 3:[20x5, 1:20x5, 20x5], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19200, 19200]][Tile0, 3:[20x5, 1:20x5, 20x5], 2]
		Tile0: [0, 6400, 200], Tile1: [200, 6400, 200], Tile2; [400, 6400, 200]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 1 physical tiles
			Total Size: 9216 [D1, [0 x 9216, 9216]][D0, [2 x 3456, 2304]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 9216, 9216]][D0, [2 x 3456, 2304]]
		Tile0: [0, 9216, 9216], Tile1: [0, 9216, 9216], Tile2; [0, 9216, 9216]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 9 logical tiles, 9 physical tiles
			Total Size: 38400 [D0, [2 x 14400, 9600]][Tile0, 3:[40x11, 1:40x11, 40x10], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 3:[40x11, 1:40x11, 40x10], 2][D0, [2 x 14400, 9600]]
		Tile0: [0, 5280, 880], Tile1: [14400, 5280, 880], Tile2; [28800, 3520, 880]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 1 physical tiles
			Total Size: 38400 [D1, [0 x 38400, 38400]][Tile0, 3:[20x5, 1:20x5, 20x5], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 38400, 38400]][Tile0, 3:[20x5, 1:20x5, 20x5], 4]
		Tile0: [0, 12800, 400], Tile1: [0, 12800, 400], Tile2; [0, 12800, 400]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (resnet_L1_Memory+32640);
	KerArg0->W = (unsigned short int) (20);
	KerArg0->H = (unsigned short int) (5);
	KerArg0->OutFeatures = (unsigned short int) (32);
	KerArg0->Bias = (short int * __restrict__) (resnet_L1_Memory+10560);
	KerArg0->NormBias = (signed char) (9);
	KerArg1->W = (unsigned short int) (40);
	KerArg1->UsedW = (unsigned short int) (40);
	KerArg1->OutFeatures = (unsigned short int) (32);
	KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+10624);
	KerArg1->Out = (int * __restrict__) (resnet_L1_Memory+32640);
	KerArg1->Norm = (unsigned char) (13);
	KerArg1->TotalInFeatures = (short int) (16);
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (resnet_L1_Memory+32640);
	KerArg2->W = (unsigned short int) (20);
	KerArg2->H = (unsigned short int) (5);
	KerArg2->Norm = (unsigned char) (13);
	KerArg2->InFeatures = (unsigned short int) (32);
	KerArg2->LB = (int) (0);
	KerArg2->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=6400; _LC_Out=200;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+10560), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+10624), 9216, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Filter */
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 5280, 2400, 880, 0, &DmaR_Evt3);
	_N_In=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<3; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==2), T0Ind_NextLast = ((T0Ind+1)==2);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			for (D0Ind=0; D0Ind<3; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==2), D0Ind_NextLast = ((D0Ind+1)==2);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (14400); _LN_In = ((T0Ind_Last)?800:(880-0*(T0Ind==0))); _SN_In = (((D0Ind_NextLast)?4:6)*_LN_In); 
				} else if (!(T0Ind_Last)) {
					_N_In = _N_In + (800)+(-28800); _LN_In = ((T0Ind_NextLast)?800:880); _SN_In = (6*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-1600)+(-28800); _LN_In = (880); _SN_In = (6*_LN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+5280*((D0Ind_Total+1)%2)),
							_SN_In, 2400, _LN_In, 0, &DmaR_Evt3);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (resnet_L1_Memory+0+5280*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (11-0*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->UsedH = (unsigned short int) (11-0*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->InFeatures = (unsigned short int) (D0Ind_Last?4:6);
				KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+10624+((D0Ind)*108));
				KerArg1->Pad = (v4s) ((v4s){0,1,0*(T0Ind==0),1*(T0Ind_Last)});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride2_DP_fp, (void *) KerArg1);
				__CALL(KerParConv3x3Stride2_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->Out = (short int * __restrict__) (resnet_L1_Memory+19840+6400*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerDP_fp, (void *) KerArg2);
			__CALL(KerDP_fp, KerArg2);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+19840+6400*((T0Ind_Total)%2)),
					_SC_Out, 600, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (200); _LC_Out = (200); _SC_Out = (32*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S41_Conv2d_32x16x1x1(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out)

{
	/* Shared L1: 39488 bytes, L2 buffer: 29248 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 32, Tiled: 1][Tile0 Dim: 4][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 4 physical tiles
			Total Size: 19200 [D1, [0 x 19200, 19200]][Tile0, 4:[20x4, 2:20x4, 20x3], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19200, 19200]][Tile0, 4:[20x4, 2:20x4, 20x3], 2]
		Tile0: [0, 5120, 160], Tile1: [160, 5120, 160], Tile2; [320, 5120, 160]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1024 [D1, [0 x 1024, 1024]][D0, [0 x 1024, 1024]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 1024, 1024]][D0, [0 x 1024, 1024]]
		Tile0: [0, 1024, 1024], Tile1: [0, 1024, 1024], Tile2; [0, 1024, 1024]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 4 logical tiles, 4 physical tiles
			Total Size: 38400 [D0, [0 x 38400, 38400]][Tile0, 4:[40x7, 2:40x7, 40x5], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 4:[40x7], 2][D0, [0 x 38400, 38400]]
		Tile0: [0, 8960, 560], Tile1: [640, 8960, 560], Tile2; [1280, 8960, 560]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 38400 [D1, [0 x 38400, 38400]][Tile0, 4:[20x4, 2:20x4, 20x3], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 38400, 38400]][Tile0, 4:[20x4, 2:20x4, 20x3], 4]
		Tile0: [0, 10240, 320], Tile1: [0, 10240, 320], Tile2; [0, 10240, 320]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (resnet_L1_Memory+29248);
	KerArg0->W = (unsigned short int) (20);
	KerArg0->OutFeatures = (unsigned short int) (32);
	KerArg0->Bias = (short int * __restrict__) (resnet_L1_Memory+17920);
	KerArg0->NormBias = (signed char) (9);
	KerArg1->W = (unsigned short int) (40);
	KerArg1->UsedW = (unsigned short int) (39);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (32);
	KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+17984);
	KerArg1->Out = (int * __restrict__) (resnet_L1_Memory+29248);
	KerArg1->Norm = (unsigned char) (14);
	KerArg1->TotalInFeatures = (short int) (16);
	KerArg1->Pad = (v4s) 0;
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (resnet_L1_Memory+29248);
	KerArg2->W = (unsigned short int) (20);
	KerArg2->Norm = (unsigned char) (14);
	KerArg2->InFeatures = (unsigned short int) (32);
	KerArg2->LB = (int) (-32768);
	KerArg2->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=5120; _LC_Out=160;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+17920), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+17984), 1024, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Filter */
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 8960, 2400, 560, 0, &DmaR_Evt3);
	_N_In=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<4; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==3), T0Ind_NextLast = ((T0Ind+1)==3);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->H = (unsigned short int) (T0Ind_Last?3:4);
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1, D0Ind_NextLast = 1;
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(T0Ind_Last)) {
					_N_In = _N_In + (640); _LN_In = ((T0Ind_NextLast)?400:560); _SN_In = (16*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-1920); _LN_In = (560); _SN_In = (16*_LN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+8960*((D0Ind_Total+1)%2)),
							_SN_In, 2400, _LN_In, 0, &DmaR_Evt3);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (resnet_L1_Memory+0+8960*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (T0Ind_Last?5:7);
				KerArg1->UsedH = (unsigned short int) (T0Ind_Last?5:7);
				AT_FORK(gap_ncore(), (void *) KerParConv1x1Stride2_DP_fp, (void *) KerArg1);
				__CALL(KerParConv1x1Stride2_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
				D0Ind_Total++;
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->H = (unsigned short int) (T0Ind_Last?3:4);
			KerArg2->Out = (short int * __restrict__) (resnet_L1_Memory+19008+5120*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerDP_fp, (void *) KerArg2);
			__CALL(KerDP_fp, KerArg2);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+19008+5120*((T0Ind_Total)%2)),
					_SC_Out, 600, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (160); _LC_Out = ((T0Ind_NextLast)?120:160); _SC_Out = (32*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S44_Conv2d_32x32x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out)

{
	/* Shared L1: 48576 bytes, L2 buffer: 35776 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 32, Tiled: 1][Tile0 Dim: 3][D0 Dim: Init: 32, Tiled: 4]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 12 logical tiles, 12 physical tiles
			Total Size: 19200 [D0, [3 x 4800, 4800]][Tile0, 3:[20x6, 1:20x7, 20x6], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 3:[20x6, 2:20x7, 20x6], 2][D0, [3 x 4800, 4800]]
		Tile0: [0, 1920, 240], Tile1: [4800, 1920, 240], Tile2; [9600, 1920, 240]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 18432 [D1, [0 x 18432, 18432]][D0, [3 x 4608, 4608]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 18432, 18432]][D0, [3 x 4608, 4608]]
		Tile0: [0, 18432, 18432], Tile1: [0, 18432, 18432], Tile2; [0, 18432, 18432]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 3 physical tiles
			Total Size: 19200 [D1, [0 x 19200, 19200]][Tile0, 3:[20x5, 1:20x5, 20x5], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19200, 19200]][Tile0, 3:[20x5, 1:20x5, 20x5], 2]
		Tile0: [0, 6400, 200], Tile1: [200, 6400, 200], Tile2; [400, 6400, 200]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 1 physical tiles
			Total Size: 38400 [D1, [0 x 38400, 38400]][Tile0, 3:[20x5, 1:20x5, 20x5], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 38400, 38400]][Tile0, 3:[20x5, 1:20x5, 20x5], 4]
		Tile0: [0, 12800, 400], Tile1: [0, 12800, 400], Tile2; [0, 12800, 400]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (resnet_L1_Memory+35776);
	KerArg0->W = (unsigned short int) (20);
	KerArg0->H = (unsigned short int) (5);
	KerArg0->OutFeatures = (unsigned short int) (32);
	KerArg0->Bias = (short int * __restrict__) (resnet_L1_Memory+4480);
	KerArg0->NormBias = (signed char) (13);
	KerArg1->W = (unsigned short int) (20);
	KerArg1->UsedW = (unsigned short int) (20);
	KerArg1->InFeatures = (unsigned short int) (8);
	KerArg1->OutFeatures = (unsigned short int) (32);
	KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+4544);
	KerArg1->Out = (int * __restrict__) (resnet_L1_Memory+35776);
	KerArg1->Norm = (unsigned char) (15);
	KerArg1->TotalInFeatures = (short int) (32);
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (resnet_L1_Memory+35776);
	KerArg2->W = (unsigned short int) (20);
	KerArg2->H = (unsigned short int) (5);
	KerArg2->Norm = (unsigned char) (15);
	KerArg2->InFeatures = (unsigned short int) (32);
	KerArg2->LB = (int) (0);
	KerArg2->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 1920, 600, 240, 0, &DmaR_Evt1);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+4480), 64, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+4544), 18432, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Filter */
	_C_Out=0; _SC_Out=6400; _LC_Out=200;
	_SP_Out=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<3; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==2), T0Ind_NextLast = ((T0Ind+1)==2);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			for (D0Ind=0; D0Ind<4; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==3), D0Ind_NextLast = ((D0Ind+1)==3);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (4800); _LN_In = ((T0Ind_Last)?240:(280-40*(T0Ind==0))); _SN_In = (8*_LN_In); 
				} else if (!(T0Ind_Last)) {
					_N_In = _N_In + (200-(40*(T0Ind==0)))+(-14400); _LN_In = ((T0Ind_NextLast)?240:280); _SN_In = (8*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-360)+(-14400); _LN_In = (240); _SN_In = (8*_LN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+2240*((D0Ind_Total+1)%2)),
							_SN_In, 600, _LN_In, 0, &DmaR_Evt1);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (resnet_L1_Memory+0+2240*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (7-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->UsedH = (unsigned short int) (7-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+4544+((D0Ind)*144));
				KerArg1->Pad = (v4s) ((v4s){1,1,1*(T0Ind==0),1*(T0Ind_Last)});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_DP_fp, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->Out = (short int * __restrict__) (resnet_L1_Memory+22976+6400*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerDP_fp, (void *) KerArg2);
			__CALL(KerDP_fp, KerArg2);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+22976+6400*((T0Ind_Total)%2)),
					_SC_Out, 600, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (200); _LC_Out = (200); _SC_Out = (32*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S45_MatAdd_32x15x20(
		short int * __restrict__ In1,
		short int * __restrict__ In2,
		short int * __restrict__ Out)

{
	/* Shared L1: 46080 bytes, L2 buffer: 46080 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last, D0Ind_NextLast;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In1;
	unsigned int _SN_In1;
	unsigned int _LN_In1;
	unsigned int _N_In2;
	unsigned int _SN_In2;
	unsigned int _LN_In2;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 32, Tiled: 1][Tile0 Dim: 3]
	Ker Arg: In1, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 3 logical tiles, 3 physical tiles
			Total Size: 19200 [D0, [0 x 19200, 19200]][Tile0, 3:[15x8, 1:15x8, 15x4], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 19200, 19200]][Tile0, 3:[15x8, 1:15x8, 15x4], 2]
		Tile0: [0, 7680, 240], Tile1: [240, 7680, 240], Tile2; [480, 3840, 120]
	Ker Arg: In2, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 3 logical tiles, 3 physical tiles
			Total Size: 19200 [D0, [0 x 19200, 19200]][Tile0, 3:[15x8, 1:15x8, 15x4], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 19200, 19200]][Tile0, 3:[15x8, 1:15x8, 15x4], 2]
		Tile0: [0, 7680, 240], Tile1: [240, 7680, 240], Tile2; [480, 3840, 120]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 3 physical tiles
			Total Size: 19200 [D0, [0 x 19200, 19200]][Tile0, 3:[15x8, 1:15x8, 15x4], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 19200, 19200]][Tile0, 3:[15x8, 1:15x8, 15x4], 2]
		Tile0: [0, 7680, 240], Tile1: [240, 7680, 240], Tile2; [480, 3840, 120]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->W = (unsigned short int) (15);
	KerArg0->N = (unsigned short int) (32);
	KerArg0->LB = (int) (-32768);
	KerArg0->UB = (int) (32767);
	KerArg0->In1_Q = (unsigned char) (11);
	KerArg0->In2_Q = (unsigned char) (10);
	KerArg0->Out_Q = (unsigned char) (9);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 7680, 600, 240, 0, &DmaR_Evt1);
	_N_In1=0;
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+15360+0), 7680, 600, 240, 0, &DmaR_Evt2);
	_N_In2=0;
	_C_Out=0; _SC_Out=7680; _LC_Out=240;
	_SP_Out=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1, D0Ind_NextLast = 1;
		for (T0Ind=0; T0Ind<3; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==2), T0Ind_NextLast = ((T0Ind+1)==2);
			/*================================= Prepare Tiles ===================================*/
			_SN_In1 = 0;
			if (!(T0Ind_Last)) {
				_N_In1 = _N_In1 + (240); _LN_In1 = ((T0Ind_NextLast)?120:240); _SN_In1 = (32*_LN_In1); 
			}
			_SN_In2 = 0;
			if (!(T0Ind_Last)) {
				_N_In2 = _N_In2 + (240); _LN_In2 = ((T0Ind_NextLast)?120:240); _SN_In2 = (32*_LN_In2); 
			}
			/*============================= End Prepare Tiles ===================================*/
			/*================================= Read Tiles ======================================*/
			AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
			if (_SN_In1) {
				AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In1+_N_In1), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+7680*((T0Ind_Total+1)%2)),
						_SN_In1, 600, _LN_In1, 0, &DmaR_Evt1);
			}
			AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
			if (_SN_In2) {
				AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In2+_N_In2), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+15360+7680*((T0Ind_Total+1)%2)),
						_SN_In2, 600, _LN_In2, 0, &DmaR_Evt2);
			}
			/*============================= End Read Tiles ======================================*/
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->In1 = (short int *__restrict__) (resnet_L1_Memory+0+7680*((T0Ind_Total)%2));
			KerArg0->In2 = (short int *__restrict__) (resnet_L1_Memory+15360+7680*((T0Ind_Total)%2));
			KerArg0->Out = (short int *__restrict__) (resnet_L1_Memory+30720+7680*((T0Ind_Total)%2));
			KerArg0->H = (unsigned short int) (T0Ind_Last?4:8);
			AT_FORK(gap_ncore(), (void *) KerParMatAddDynAdjust_fp, (void *) KerArg0);
			__CALL(KerParMatAddDynAdjust_fp, KerArg0);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+30720+7680*((T0Ind_Total)%2)),
					_SC_Out, 600, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (240); _LC_Out = ((T0Ind_NextLast)?120:240); _SC_Out = (32*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S48_Conv2d_32x32x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out)

{
	/* Shared L1: 48576 bytes, L2 buffer: 35776 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 32, Tiled: 1][Tile0 Dim: 3][D0 Dim: Init: 32, Tiled: 4]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 12 logical tiles, 12 physical tiles
			Total Size: 19200 [D0, [3 x 4800, 4800]][Tile0, 3:[20x6, 1:20x7, 20x6], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 3:[20x6, 2:20x7, 20x6], 2][D0, [3 x 4800, 4800]]
		Tile0: [0, 1920, 240], Tile1: [4800, 1920, 240], Tile2; [9600, 1920, 240]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 18432 [D1, [0 x 18432, 18432]][D0, [3 x 4608, 4608]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 18432, 18432]][D0, [3 x 4608, 4608]]
		Tile0: [0, 18432, 18432], Tile1: [0, 18432, 18432], Tile2; [0, 18432, 18432]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 3 physical tiles
			Total Size: 19200 [D1, [0 x 19200, 19200]][Tile0, 3:[20x5, 1:20x5, 20x5], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19200, 19200]][Tile0, 3:[20x5, 1:20x5, 20x5], 2]
		Tile0: [0, 6400, 200], Tile1: [200, 6400, 200], Tile2; [400, 6400, 200]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 1 physical tiles
			Total Size: 38400 [D1, [0 x 38400, 38400]][Tile0, 3:[20x5, 1:20x5, 20x5], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 38400, 38400]][Tile0, 3:[20x5, 1:20x5, 20x5], 4]
		Tile0: [0, 12800, 400], Tile1: [0, 12800, 400], Tile2; [0, 12800, 400]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (resnet_L1_Memory+35776);
	KerArg0->W = (unsigned short int) (20);
	KerArg0->H = (unsigned short int) (5);
	KerArg0->OutFeatures = (unsigned short int) (32);
	KerArg0->Bias = (short int * __restrict__) (resnet_L1_Memory+4480);
	KerArg0->NormBias = (signed char) (9);
	KerArg1->W = (unsigned short int) (20);
	KerArg1->UsedW = (unsigned short int) (20);
	KerArg1->InFeatures = (unsigned short int) (8);
	KerArg1->OutFeatures = (unsigned short int) (32);
	KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+4544);
	KerArg1->Out = (int * __restrict__) (resnet_L1_Memory+35776);
	KerArg1->Norm = (unsigned char) (13);
	KerArg1->TotalInFeatures = (short int) (32);
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (resnet_L1_Memory+35776);
	KerArg2->W = (unsigned short int) (20);
	KerArg2->H = (unsigned short int) (5);
	KerArg2->Norm = (unsigned char) (13);
	KerArg2->InFeatures = (unsigned short int) (32);
	KerArg2->LB = (int) (0);
	KerArg2->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 1920, 600, 240, 0, &DmaR_Evt1);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+4480), 64, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+4544), 18432, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Filter */
	_C_Out=0; _SC_Out=6400; _LC_Out=200;
	_SP_Out=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<3; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==2), T0Ind_NextLast = ((T0Ind+1)==2);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			for (D0Ind=0; D0Ind<4; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==3), D0Ind_NextLast = ((D0Ind+1)==3);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (4800); _LN_In = ((T0Ind_Last)?240:(280-40*(T0Ind==0))); _SN_In = (8*_LN_In); 
				} else if (!(T0Ind_Last)) {
					_N_In = _N_In + (200-(40*(T0Ind==0)))+(-14400); _LN_In = ((T0Ind_NextLast)?240:280); _SN_In = (8*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-360)+(-14400); _LN_In = (240); _SN_In = (8*_LN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+2240*((D0Ind_Total+1)%2)),
							_SN_In, 600, _LN_In, 0, &DmaR_Evt1);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (resnet_L1_Memory+0+2240*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (7-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->UsedH = (unsigned short int) (7-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+4544+((D0Ind)*144));
				KerArg1->Pad = (v4s) ((v4s){1,1,1*(T0Ind==0),1*(T0Ind_Last)});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_DP_fp, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->Out = (short int * __restrict__) (resnet_L1_Memory+22976+6400*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerDP_fp, (void *) KerArg2);
			__CALL(KerDP_fp, KerArg2);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+22976+6400*((T0Ind_Total)%2)),
					_SC_Out, 600, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (200); _LC_Out = (200); _SC_Out = (32*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S51_Conv2d_32x32x3x3(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out)

{
	/* Shared L1: 48576 bytes, L2 buffer: 35776 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 32, Tiled: 1][Tile0 Dim: 3][D0 Dim: Init: 32, Tiled: 4]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 12 logical tiles, 12 physical tiles
			Total Size: 19200 [D0, [3 x 4800, 4800]][Tile0, 3:[20x6, 1:20x7, 20x6], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 3:[20x6, 2:20x7, 20x6], 2][D0, [3 x 4800, 4800]]
		Tile0: [0, 1920, 240], Tile1: [4800, 1920, 240], Tile2; [9600, 1920, 240]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 18432 [D1, [0 x 18432, 18432]][D0, [3 x 4608, 4608]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 18432, 18432]][D0, [3 x 4608, 4608]]
		Tile0: [0, 18432, 18432], Tile1: [0, 18432, 18432], Tile2; [0, 18432, 18432]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 3 physical tiles
			Total Size: 19200 [D1, [0 x 19200, 19200]][Tile0, 3:[20x5, 1:20x5, 20x5], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19200, 19200]][Tile0, 3:[20x5, 1:20x5, 20x5], 2]
		Tile0: [0, 6400, 200], Tile1: [200, 6400, 200], Tile2; [400, 6400, 200]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 1 physical tiles
			Total Size: 38400 [D1, [0 x 38400, 38400]][Tile0, 3:[20x5, 1:20x5, 20x5], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 38400, 38400]][Tile0, 3:[20x5, 1:20x5, 20x5], 4]
		Tile0: [0, 12800, 400], Tile1: [0, 12800, 400], Tile2; [0, 12800, 400]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (resnet_L1_Memory+35776);
	KerArg0->W = (unsigned short int) (20);
	KerArg0->H = (unsigned short int) (5);
	KerArg0->OutFeatures = (unsigned short int) (32);
	KerArg0->Bias = (short int * __restrict__) (resnet_L1_Memory+4480);
	KerArg0->NormBias = (signed char) (13);
	KerArg1->W = (unsigned short int) (20);
	KerArg1->UsedW = (unsigned short int) (20);
	KerArg1->InFeatures = (unsigned short int) (8);
	KerArg1->OutFeatures = (unsigned short int) (32);
	KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+4544);
	KerArg1->Out = (int * __restrict__) (resnet_L1_Memory+35776);
	KerArg1->Norm = (unsigned char) (16);
	KerArg1->TotalInFeatures = (short int) (32);
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (resnet_L1_Memory+35776);
	KerArg2->W = (unsigned short int) (20);
	KerArg2->H = (unsigned short int) (5);
	KerArg2->Norm = (unsigned char) (16);
	KerArg2->InFeatures = (unsigned short int) (32);
	KerArg2->LB = (int) (-32768);
	KerArg2->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 1920, 600, 240, 0, &DmaR_Evt1);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+4480), 64, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+4544), 18432, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Filter */
	_C_Out=0; _SC_Out=6400; _LC_Out=200;
	_SP_Out=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<3; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==2), T0Ind_NextLast = ((T0Ind+1)==2);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			for (D0Ind=0; D0Ind<4; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==3), D0Ind_NextLast = ((D0Ind+1)==3);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (4800); _LN_In = ((T0Ind_Last)?240:(280-40*(T0Ind==0))); _SN_In = (8*_LN_In); 
				} else if (!(T0Ind_Last)) {
					_N_In = _N_In + (200-(40*(T0Ind==0)))+(-14400); _LN_In = ((T0Ind_NextLast)?240:280); _SN_In = (8*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-360)+(-14400); _LN_In = (240); _SN_In = (8*_LN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+2240*((D0Ind_Total+1)%2)),
							_SN_In, 600, _LN_In, 0, &DmaR_Evt1);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (resnet_L1_Memory+0+2240*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (7-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->UsedH = (unsigned short int) (7-1*(T0Ind==0)-1*(T0Ind_Last));
				KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+4544+((D0Ind)*144));
				KerArg1->Pad = (v4s) ((v4s){1,1,1*(T0Ind==0),1*(T0Ind_Last)});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_DP_fp, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->Out = (short int * __restrict__) (resnet_L1_Memory+22976+6400*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerDP_fp, (void *) KerArg2);
			__CALL(KerDP_fp, KerArg2);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+22976+6400*((T0Ind_Total)%2)),
					_SC_Out, 600, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (200); _LC_Out = (200); _SC_Out = (32*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S52_MatAdd_32x15x20(
		short int * __restrict__ In1,
		short int * __restrict__ In2,
		short int * __restrict__ Out)

{
	/* Shared L1: 46080 bytes, L2 buffer: 46080 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last, D0Ind_NextLast;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In1;
	unsigned int _SN_In1;
	unsigned int _LN_In1;
	unsigned int _N_In2;
	unsigned int _SN_In2;
	unsigned int _LN_In2;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 32, Tiled: 1][Tile0 Dim: 3]
	Ker Arg: In1, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 3 logical tiles, 3 physical tiles
			Total Size: 19200 [D0, [0 x 19200, 19200]][Tile0, 3:[15x8, 1:15x8, 15x4], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 19200, 19200]][Tile0, 3:[15x8, 1:15x8, 15x4], 2]
		Tile0: [0, 7680, 240], Tile1: [240, 7680, 240], Tile2; [480, 3840, 120]
	Ker Arg: In2, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 3 logical tiles, 3 physical tiles
			Total Size: 19200 [D0, [0 x 19200, 19200]][Tile0, 3:[15x8, 1:15x8, 15x4], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 19200, 19200]][Tile0, 3:[15x8, 1:15x8, 15x4], 2]
		Tile0: [0, 7680, 240], Tile1: [240, 7680, 240], Tile2; [480, 3840, 120]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 3 physical tiles
			Total Size: 19200 [D0, [0 x 19200, 19200]][Tile0, 3:[15x8, 1:15x8, 15x4], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 19200, 19200]][Tile0, 3:[15x8, 1:15x8, 15x4], 2]
		Tile0: [0, 7680, 240], Tile1: [240, 7680, 240], Tile2; [480, 3840, 120]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->W = (unsigned short int) (15);
	KerArg0->N = (unsigned short int) (32);
	KerArg0->LB = (int) (-32768);
	KerArg0->UB = (int) (32767);
	KerArg0->In1_Q = (unsigned char) (10);
	KerArg0->In2_Q = (unsigned char) (9);
	KerArg0->Out_Q = (unsigned char) (9);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 7680, 600, 240, 0, &DmaR_Evt1);
	_N_In1=0;
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+15360+0), 7680, 600, 240, 0, &DmaR_Evt2);
	_N_In2=0;
	_C_Out=0; _SC_Out=7680; _LC_Out=240;
	_SP_Out=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1, D0Ind_NextLast = 1;
		for (T0Ind=0; T0Ind<3; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==2), T0Ind_NextLast = ((T0Ind+1)==2);
			/*================================= Prepare Tiles ===================================*/
			_SN_In1 = 0;
			if (!(T0Ind_Last)) {
				_N_In1 = _N_In1 + (240); _LN_In1 = ((T0Ind_NextLast)?120:240); _SN_In1 = (32*_LN_In1); 
			}
			_SN_In2 = 0;
			if (!(T0Ind_Last)) {
				_N_In2 = _N_In2 + (240); _LN_In2 = ((T0Ind_NextLast)?120:240); _SN_In2 = (32*_LN_In2); 
			}
			/*============================= End Prepare Tiles ===================================*/
			/*================================= Read Tiles ======================================*/
			AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
			if (_SN_In1) {
				AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In1+_N_In1), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+7680*((T0Ind_Total+1)%2)),
						_SN_In1, 600, _LN_In1, 0, &DmaR_Evt1);
			}
			AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
			if (_SN_In2) {
				AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In2+_N_In2), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+15360+7680*((T0Ind_Total+1)%2)),
						_SN_In2, 600, _LN_In2, 0, &DmaR_Evt2);
			}
			/*============================= End Read Tiles ======================================*/
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->In1 = (short int *__restrict__) (resnet_L1_Memory+0+7680*((T0Ind_Total)%2));
			KerArg0->In2 = (short int *__restrict__) (resnet_L1_Memory+15360+7680*((T0Ind_Total)%2));
			KerArg0->Out = (short int *__restrict__) (resnet_L1_Memory+30720+7680*((T0Ind_Total)%2));
			KerArg0->H = (unsigned short int) (T0Ind_Last?4:8);
			AT_FORK(gap_ncore(), (void *) KerParMatAddDynAdjust_fp, (void *) KerArg0);
			__CALL(KerParMatAddDynAdjust_fp, KerArg0);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+30720+7680*((T0Ind_Total)%2)),
					_SC_Out, 600, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (240); _LC_Out = ((T0Ind_NextLast)?120:240); _SC_Out = (32*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S55_Conv2d_64x32x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out)

{
	/* Shared L1: 37856 bytes, L2 buffer: 17376 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last, D1Ind_NextLast;
	int T0Ind, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_Filter;
	unsigned int _SN_Filter;
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 64, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 32, Tiled: 16]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 20480 [D1, [0 x 20480, 20480]][Tile0, 1:[10x8], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 20480, 20480]][Tile0, 1:[10x8], 4]
		Tile0: [0, 20480, 320], Tile1: [0, 20480, 320], Tile2; [0, 20480, 320]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [D1, [0 x 128, 128]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 128, 128]]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: Filter, Tiled Space: D0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 16 logical tiles, 16 physical tiles
			Total Size: 36864 [D1, [0 x 36864, 36864]][D0, [15 x 2304, 2304]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 36864, 36864]][D0, [15 x 2304, 2304]]
		Tile0: [0, 2304, 2304], Tile1: [2304, 2304, 2304], Tile2; [4608, 2304, 2304]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 10240 [D1, [0 x 10240, 10240]][Tile0, 1:[10x8], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 10240, 10240]][Tile0, 1:[10x8], 2]
		Tile0: [0, 10240, 10240], Tile1: [0, 10240, 10240], Tile2; [0, 10240, 10240]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 16 logical tiles, 16 physical tiles
			Total Size: 19200 [D0, [15 x 1200, 1200]][Tile0, 1:[20x15], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[20x16, 14:20x17, 20x16], 2][D0, [15 x 1200, 1200]]
		Tile0: [0, 1200, 600], Tile1: [1200, 1200, 600], Tile2; [2400, 1200, 600]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (resnet_L1_Memory+17376);
	KerArg0->W = (unsigned short int) (10);
	KerArg0->H = (unsigned short int) (8);
	KerArg0->OutFeatures = (unsigned short int) (64);
	KerArg0->Bias = (short int * __restrict__) (resnet_L1_Memory+2400);
	KerArg0->NormBias = (signed char) (9);
	KerArg1->W = (unsigned short int) (20);
	KerArg1->UsedW = (unsigned short int) (20);
	KerArg1->InFeatures = (unsigned short int) (2);
	KerArg1->OutFeatures = (unsigned short int) (64);
	KerArg1->Out = (int * __restrict__) (resnet_L1_Memory+17376);
	KerArg1->Norm = (unsigned char) (13);
	KerArg1->TotalInFeatures = (short int) (2);
	KerArg1->Pad = (v4s) ((v4s){0,1,1,1});
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (resnet_L1_Memory+17376);
	KerArg2->W = (unsigned short int) (10);
	KerArg2->H = (unsigned short int) (8);
	KerArg2->Out = (short int * __restrict__) (resnet_L1_Memory+7136);
	KerArg2->Norm = (unsigned char) (13);
	KerArg2->InFeatures = (unsigned short int) (64);
	KerArg2->LB = (int) (0);
	KerArg2->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+2400), 128, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+2528+0), 2304, 0, &DmaR_Evt2);
	_N_Filter=0;
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 1200, 600, 600, 0, &DmaR_Evt3);
	_N_In=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1, D1Ind_NextLast = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1, T0Ind_NextLast = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			for (D0Ind=0; D0Ind<16; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==15), D0Ind_NextLast = ((D0Ind+1)==15);
				/*================================= Prepare Tiles ===================================*/
				_SN_Filter = 0;
				if (!(D0Ind_Last)) {
					_N_Filter = _N_Filter + ((2304)); _SN_Filter = (((1)?(2304):(2304))); 
				} else if (!((1))) {
					_N_Filter = _N_Filter + ((-34560)); _SN_Filter = (((1)?(2304):(2304))); 
				}
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (1200); _LN_In = (600); _SN_In = (2*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-18000); _LN_In = (600); _SN_In = (2*_LN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Filter */
				if (_SN_Filter) {
					AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+_N_Filter), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+2528+2304*((D0Ind_Total+1)%2)),
							_SN_Filter, 0, &DmaR_Evt2);
				}
				AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+1200*((D0Ind_Total+1)%2)),
							_SN_In, 600, _LN_In, 0, &DmaR_Evt3);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (resnet_L1_Memory+0+1200*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (17-1*(1)-1*(1));
				KerArg1->UsedH = (unsigned short int) (17-1*(1)-1*(1));
				KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+2528+2304*((D0Ind_Total)%2));
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride2_DP_fp, (void *) KerArg1);
				__CALL(KerParConv3x3Stride2_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerDP_fp, (void *) KerArg2);
			__CALL(KerDP_fp, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+7136), 10240, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S58_Conv2d_64x32x1x1(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out)

{
	/* Shared L1: 42624 bytes, L2 buffer: 32384 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 64, Tiled: 1][Tile0 Dim: 2][D0 Dim: Init: 32, Tiled: 1]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 2 physical tiles
			Total Size: 10240 [D1, [0 x 10240, 10240]][Tile0, 2:[10x4, 10x4], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 10240, 10240]][Tile0, 2:[10x4, 10x4], 2]
		Tile0: [0, 5120, 80], Tile1: [80, 5120, 80], Tile2; [0, 5120, 80]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [D1, [0 x 128, 128]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 128, 128]]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4096 [D1, [0 x 4096, 4096]][D0, [0 x 4096, 4096]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4096, 4096]][D0, [0 x 4096, 4096]]
		Tile0: [0, 4096, 4096], Tile1: [0, 4096, 4096], Tile2; [0, 4096, 4096]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 2 logical tiles, 2 physical tiles
			Total Size: 19200 [D0, [0 x 19200, 19200]][Tile0, 2:[20x7, 20x7], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 2:[20x7], 2][D0, [0 x 19200, 19200]]
		Tile0: [0, 8960, 280], Tile1: [320, 8960, 280], Tile2; [0, 8960, 280]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 20480 [D1, [0 x 20480, 20480]][Tile0, 2:[10x4, 10x4], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 20480, 20480]][Tile0, 2:[10x4, 10x4], 4]
		Tile0: [0, 10240, 160], Tile1: [0, 10240, 160], Tile2; [0, 10240, 160]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (resnet_L1_Memory+32384);
	KerArg0->W = (unsigned short int) (10);
	KerArg0->H = (unsigned short int) (4);
	KerArg0->OutFeatures = (unsigned short int) (64);
	KerArg0->Bias = (short int * __restrict__) (resnet_L1_Memory+17920);
	KerArg0->NormBias = (signed char) (10);
	KerArg1->W = (unsigned short int) (20);
	KerArg1->UsedW = (unsigned short int) (19);
	KerArg1->H = (unsigned short int) (7);
	KerArg1->UsedH = (unsigned short int) (7);
	KerArg1->InFeatures = (unsigned short int) (32);
	KerArg1->OutFeatures = (unsigned short int) (64);
	KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+18048);
	KerArg1->Out = (int * __restrict__) (resnet_L1_Memory+32384);
	KerArg1->Norm = (unsigned char) (14);
	KerArg1->TotalInFeatures = (short int) (32);
	KerArg1->Pad = (v4s) 0;
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (resnet_L1_Memory+32384);
	KerArg2->W = (unsigned short int) (10);
	KerArg2->H = (unsigned short int) (4);
	KerArg2->Norm = (unsigned char) (14);
	KerArg2->InFeatures = (unsigned short int) (64);
	KerArg2->LB = (int) (-32768);
	KerArg2->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=5120; _LC_Out=80;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+17920), 128, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+18048), 4096, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Filter */
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 8960, 600, 280, 0, &DmaR_Evt3);
	_N_In=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<2; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==1), T0Ind_NextLast = ((T0Ind+1)==1);
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1, D0Ind_NextLast = 1;
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(T0Ind_Last)) {
					_N_In = _N_In + (320); _LN_In = (280); _SN_In = (32*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-320); _LN_In = (280); _SN_In = (32*_LN_In); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+8960*((D0Ind_Total+1)%2)),
							_SN_In, 600, _LN_In, 0, &DmaR_Evt3);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (resnet_L1_Memory+0+8960*((D0Ind_Total)%2));
				AT_FORK(gap_ncore(), (void *) KerParConv1x1Stride2_DP_fp, (void *) KerArg1);
				__CALL(KerParConv1x1Stride2_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
				D0Ind_Total++;
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->Out = (short int * __restrict__) (resnet_L1_Memory+22144+5120*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerDP_fp, (void *) KerArg2);
			__CALL(KerDP_fp, KerArg2);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+22144+5120*((T0Ind_Total)%2)),
					_SC_Out, 160, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (80); _LC_Out = (80); _SC_Out = (64*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S61_Conv2d_64x64x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out)

{
	/* Shared L1: 36096 bytes, L2 buffer: 15616 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last, D1Ind_NextLast;
	int T0Ind, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	unsigned int _N_Filter;
	unsigned int _SN_Filter;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 64, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 64, Tiled: 32]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 32 logical tiles, 32 physical tiles
			Total Size: 10240 [D0, [31 x 320, 320]][Tile0, 1:[10x8], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[10x9, 30:10x10, 10x9], 2][D0, [31 x 320, 320]]
		Tile0: [0, 320, 160], Tile1: [320, 320, 160], Tile2; [640, 320, 160]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [D1, [0 x 128, 128]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 128, 128]]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: Filter, Tiled Space: D0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 32 logical tiles, 32 physical tiles
			Total Size: 73728 [D1, [0 x 73728, 73728]][D0, [31 x 2304, 2304]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 73728, 73728]][D0, [31 x 2304, 2304]]
		Tile0: [0, 2304, 2304], Tile1: [2304, 2304, 2304], Tile2; [4608, 2304, 2304]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 10240 [D1, [0 x 10240, 10240]][Tile0, 1:[10x8], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 10240, 10240]][Tile0, 1:[10x8], 2]
		Tile0: [0, 10240, 10240], Tile1: [0, 10240, 10240], Tile2; [0, 10240, 10240]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 20480 [D1, [0 x 20480, 20480]][Tile0, 1:[10x8], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 20480, 20480]][Tile0, 1:[10x8], 4]
		Tile0: [0, 20480, 320], Tile1: [0, 20480, 320], Tile2; [0, 20480, 320]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (resnet_L1_Memory+15616);
	KerArg0->W = (unsigned short int) (10);
	KerArg0->H = (unsigned short int) (8);
	KerArg0->OutFeatures = (unsigned short int) (64);
	KerArg0->Bias = (short int * __restrict__) (resnet_L1_Memory+640);
	KerArg0->NormBias = (signed char) (13);
	KerArg1->W = (unsigned short int) (10);
	KerArg1->UsedW = (unsigned short int) (10);
	KerArg1->InFeatures = (unsigned short int) (2);
	KerArg1->OutFeatures = (unsigned short int) (64);
	KerArg1->Out = (int * __restrict__) (resnet_L1_Memory+15616);
	KerArg1->Norm = (unsigned char) (15);
	KerArg1->TotalInFeatures = (short int) (2);
	KerArg1->Pad = (v4s) ((v4s){1,1,1,1});
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (resnet_L1_Memory+15616);
	KerArg2->W = (unsigned short int) (10);
	KerArg2->H = (unsigned short int) (8);
	KerArg2->Out = (short int * __restrict__) (resnet_L1_Memory+5376);
	KerArg2->Norm = (unsigned char) (15);
	KerArg2->InFeatures = (unsigned short int) (64);
	KerArg2->LB = (int) (0);
	KerArg2->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 320, 160, 160, 0, &DmaR_Evt1);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+640), 128, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+768+0), 2304, 0, &DmaR_Evt3);
	_N_Filter=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1, D1Ind_NextLast = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1, T0Ind_NextLast = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			for (D0Ind=0; D0Ind<32; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==31), D0Ind_NextLast = ((D0Ind+1)==31);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (320); _LN_In = (160); _SN_In = (2*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-9920); _LN_In = (160); _SN_In = (2*_LN_In); 
				}
				_SN_Filter = 0;
				if (!(D0Ind_Last)) {
					_N_Filter = _N_Filter + ((2304)); _SN_Filter = (((1)?(2304):(2304))); 
				} else if (!((1))) {
					_N_Filter = _N_Filter + ((-71424)); _SN_Filter = (((1)?(2304):(2304))); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+320*((D0Ind_Total+1)%2)),
							_SN_In, 160, _LN_In, 0, &DmaR_Evt1);
				}
				AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Filter */
				if (_SN_Filter) {
					AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+_N_Filter), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+768+2304*((D0Ind_Total+1)%2)),
							_SN_Filter, 0, &DmaR_Evt3);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (resnet_L1_Memory+0+320*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (10-1*(1)-1*(1));
				KerArg1->UsedH = (unsigned short int) (10-1*(1)-1*(1));
				KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+768+2304*((D0Ind_Total)%2));
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_DP_fp, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerDP_fp, (void *) KerArg2);
			__CALL(KerDP_fp, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+5376), 10240, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S62_MatAdd_64x8x10(
		short int * __restrict__ In1,
		short int * __restrict__ In2,
		short int * __restrict__ Out)

{
	/* Shared L1: 30720 bytes, L2 buffer: 30720 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 64, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 10240 [D0, [0 x 10240, 10240]][Tile0, 1:[8x10], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 10240, 10240]][Tile0, 1:[8x10], 2]
		Tile0: [0, 10240, 10240], Tile1: [0, 10240, 10240], Tile2; [0, 10240, 10240]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 10240 [D0, [0 x 10240, 10240]][Tile0, 1:[8x10], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 10240, 10240]][Tile0, 1:[8x10], 2]
		Tile0: [0, 10240, 10240], Tile1: [0, 10240, 10240], Tile2; [0, 10240, 10240]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 10240 [D0, [0 x 10240, 10240]][Tile0, 1:[8x10], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 10240, 10240]][Tile0, 1:[8x10], 2]
		Tile0: [0, 10240, 10240], Tile1: [0, 10240, 10240], Tile2; [0, 10240, 10240]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (short int *__restrict__) (resnet_L1_Memory+0);
	KerArg0->In2 = (short int *__restrict__) (resnet_L1_Memory+10240);
	KerArg0->Out = (short int *__restrict__) (resnet_L1_Memory+20480);
	KerArg0->W = (unsigned short int) (8);
	KerArg0->H = (unsigned short int) (10);
	KerArg0->N = (unsigned short int) (64);
	KerArg0->LB = (int) (-32768);
	KerArg0->UB = (int) (32767);
	KerArg0->In1_Q = (unsigned char) (11);
	KerArg0->In2_Q = (unsigned char) (10);
	KerArg0->Out_Q = (unsigned char) (10);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0), 10240, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+10240), 10240, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAddDynAdjust_fp, (void *) KerArg0);
			__CALL(KerParMatAddDynAdjust_fp, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+20480), 10240, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S65_Conv2d_64x64x3x3_Relu(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out)

{
	/* Shared L1: 36096 bytes, L2 buffer: 15616 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last, D1Ind_NextLast;
	int T0Ind, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	unsigned int _N_Filter;
	unsigned int _SN_Filter;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 64, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 64, Tiled: 32]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 32 logical tiles, 32 physical tiles
			Total Size: 10240 [D0, [31 x 320, 320]][Tile0, 1:[10x8], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[10x9, 30:10x10, 10x9], 2][D0, [31 x 320, 320]]
		Tile0: [0, 320, 160], Tile1: [320, 320, 160], Tile2; [640, 320, 160]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [D1, [0 x 128, 128]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 128, 128]]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: Filter, Tiled Space: D0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 32 logical tiles, 32 physical tiles
			Total Size: 73728 [D1, [0 x 73728, 73728]][D0, [31 x 2304, 2304]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 73728, 73728]][D0, [31 x 2304, 2304]]
		Tile0: [0, 2304, 2304], Tile1: [2304, 2304, 2304], Tile2; [4608, 2304, 2304]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 10240 [D1, [0 x 10240, 10240]][Tile0, 1:[10x8], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 10240, 10240]][Tile0, 1:[10x8], 2]
		Tile0: [0, 10240, 10240], Tile1: [0, 10240, 10240], Tile2; [0, 10240, 10240]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 20480 [D1, [0 x 20480, 20480]][Tile0, 1:[10x8], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 20480, 20480]][Tile0, 1:[10x8], 4]
		Tile0: [0, 20480, 320], Tile1: [0, 20480, 320], Tile2; [0, 20480, 320]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (resnet_L1_Memory+15616);
	KerArg0->W = (unsigned short int) (10);
	KerArg0->H = (unsigned short int) (8);
	KerArg0->OutFeatures = (unsigned short int) (64);
	KerArg0->Bias = (short int * __restrict__) (resnet_L1_Memory+640);
	KerArg0->NormBias = (signed char) (10);
	KerArg1->W = (unsigned short int) (10);
	KerArg1->UsedW = (unsigned short int) (10);
	KerArg1->InFeatures = (unsigned short int) (2);
	KerArg1->OutFeatures = (unsigned short int) (64);
	KerArg1->Out = (int * __restrict__) (resnet_L1_Memory+15616);
	KerArg1->Norm = (unsigned char) (14);
	KerArg1->TotalInFeatures = (short int) (2);
	KerArg1->Pad = (v4s) ((v4s){1,1,1,1});
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (resnet_L1_Memory+15616);
	KerArg2->W = (unsigned short int) (10);
	KerArg2->H = (unsigned short int) (8);
	KerArg2->Out = (short int * __restrict__) (resnet_L1_Memory+5376);
	KerArg2->Norm = (unsigned char) (14);
	KerArg2->InFeatures = (unsigned short int) (64);
	KerArg2->LB = (int) (0);
	KerArg2->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 320, 160, 160, 0, &DmaR_Evt1);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+640), 128, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+768+0), 2304, 0, &DmaR_Evt3);
	_N_Filter=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1, D1Ind_NextLast = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1, T0Ind_NextLast = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			for (D0Ind=0; D0Ind<32; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==31), D0Ind_NextLast = ((D0Ind+1)==31);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (320); _LN_In = (160); _SN_In = (2*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-9920); _LN_In = (160); _SN_In = (2*_LN_In); 
				}
				_SN_Filter = 0;
				if (!(D0Ind_Last)) {
					_N_Filter = _N_Filter + ((2304)); _SN_Filter = (((1)?(2304):(2304))); 
				} else if (!((1))) {
					_N_Filter = _N_Filter + ((-71424)); _SN_Filter = (((1)?(2304):(2304))); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+320*((D0Ind_Total+1)%2)),
							_SN_In, 160, _LN_In, 0, &DmaR_Evt1);
				}
				AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Filter */
				if (_SN_Filter) {
					AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+_N_Filter), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+768+2304*((D0Ind_Total+1)%2)),
							_SN_Filter, 0, &DmaR_Evt3);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (resnet_L1_Memory+0+320*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (10-1*(1)-1*(1));
				KerArg1->UsedH = (unsigned short int) (10-1*(1)-1*(1));
				KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+768+2304*((D0Ind_Total)%2));
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_DP_fp, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerDP_fp, (void *) KerArg2);
			__CALL(KerDP_fp, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+5376), 10240, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S68_Conv2d_64x64x3x3(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out)

{
	/* Shared L1: 36096 bytes, L2 buffer: 15616 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_fpd_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_DP_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerDP_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last, D1Ind_NextLast;
	int T0Ind, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	unsigned int _N_Filter;
	unsigned int _SN_Filter;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 64, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 64, Tiled: 32]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 32 logical tiles, 32 physical tiles
			Total Size: 10240 [D0, [31 x 320, 320]][Tile0, 1:[10x8], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[10x9, 30:10x10, 10x9], 2][D0, [31 x 320, 320]]
		Tile0: [0, 320, 160], Tile1: [320, 320, 160], Tile2; [640, 320, 160]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [D1, [0 x 128, 128]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 128, 128]]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: Filter, Tiled Space: D0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 32 logical tiles, 32 physical tiles
			Total Size: 73728 [D1, [0 x 73728, 73728]][D0, [31 x 2304, 2304]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 73728, 73728]][D0, [31 x 2304, 2304]]
		Tile0: [0, 2304, 2304], Tile1: [2304, 2304, 2304], Tile2; [4608, 2304, 2304]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 10240 [D1, [0 x 10240, 10240]][Tile0, 1:[10x8], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 10240, 10240]][Tile0, 1:[10x8], 2]
		Tile0: [0, 10240, 10240], Tile1: [0, 10240, 10240], Tile2; [0, 10240, 10240]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 20480 [D1, [0 x 20480, 20480]][Tile0, 1:[10x8], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 20480, 20480]][Tile0, 1:[10x8], 4]
		Tile0: [0, 20480, 320], Tile1: [0, 20480, 320], Tile2; [0, 20480, 320]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (resnet_L1_Memory+15616);
	KerArg0->W = (unsigned short int) (10);
	KerArg0->H = (unsigned short int) (8);
	KerArg0->OutFeatures = (unsigned short int) (64);
	KerArg0->Bias = (short int * __restrict__) (resnet_L1_Memory+640);
	KerArg0->NormBias = (signed char) (13);
	KerArg1->W = (unsigned short int) (10);
	KerArg1->UsedW = (unsigned short int) (10);
	KerArg1->InFeatures = (unsigned short int) (2);
	KerArg1->OutFeatures = (unsigned short int) (64);
	KerArg1->Out = (int * __restrict__) (resnet_L1_Memory+15616);
	KerArg1->Norm = (unsigned char) (16);
	KerArg1->TotalInFeatures = (short int) (2);
	KerArg1->Pad = (v4s) ((v4s){1,1,1,1});
	KerArg1->Orientation = (unsigned char) (1);
	KerArg2->In = (int * __restrict__) (resnet_L1_Memory+15616);
	KerArg2->W = (unsigned short int) (10);
	KerArg2->H = (unsigned short int) (8);
	KerArg2->Out = (short int * __restrict__) (resnet_L1_Memory+5376);
	KerArg2->Norm = (unsigned char) (16);
	KerArg2->InFeatures = (unsigned short int) (64);
	KerArg2->LB = (int) (-32768);
	KerArg2->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+0), 320, 160, 160, 0, &DmaR_Evt1);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+640), 128, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+768+0), 2304, 0, &DmaR_Evt3);
	_N_Filter=0;
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1, D1Ind_NextLast = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1, T0Ind_NextLast = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParSetBias_DP_fp, (void *) KerArg0);
			__CALL(KerParSetBias_DP_fp, KerArg0);
			for (D0Ind=0; D0Ind<32; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==31), D0Ind_NextLast = ((D0Ind+1)==31);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (320); _LN_In = (160); _SN_In = (2*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-9920); _LN_In = (160); _SN_In = (2*_LN_In); 
				}
				_SN_Filter = 0;
				if (!(D0Ind_Last)) {
					_N_Filter = _N_Filter + ((2304)); _SN_Filter = (((1)?(2304):(2304))); 
				} else if (!((1))) {
					_N_Filter = _N_Filter + ((-71424)); _SN_Filter = (((1)?(2304):(2304))); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0+320*((D0Ind_Total+1)%2)),
							_SN_In, 160, _LN_In, 0, &DmaR_Evt1);
				}
				AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Filter */
				if (_SN_Filter) {
					AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+_N_Filter), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+768+2304*((D0Ind_Total+1)%2)),
							_SN_Filter, 0, &DmaR_Evt3);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (short int * __restrict__) (resnet_L1_Memory+0+320*((D0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (10-1*(1)-1*(1));
				KerArg1->UsedH = (unsigned short int) (10-1*(1)-1*(1));
				KerArg1->Filter = (short int * __restrict__) (resnet_L1_Memory+768+2304*((D0Ind_Total)%2));
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_DP_fp, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_DP_fp, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerDP_fp, (void *) KerArg2);
			__CALL(KerDP_fp, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+5376), 10240, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S69_MatAdd_64x8x10(
		short int * __restrict__ In1,
		short int * __restrict__ In2,
		short int * __restrict__ Out)

{
	/* Shared L1: 30720 bytes, L2 buffer: 30720 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 64, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 10240 [D0, [0 x 10240, 10240]][Tile0, 1:[8x10], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 10240, 10240]][Tile0, 1:[8x10], 2]
		Tile0: [0, 10240, 10240], Tile1: [0, 10240, 10240], Tile2; [0, 10240, 10240]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 10240 [D0, [0 x 10240, 10240]][Tile0, 1:[8x10], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 10240, 10240]][Tile0, 1:[8x10], 2]
		Tile0: [0, 10240, 10240], Tile1: [0, 10240, 10240], Tile2; [0, 10240, 10240]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 10240 [D0, [0 x 10240, 10240]][Tile0, 1:[8x10], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 10240, 10240]][Tile0, 1:[8x10], 2]
		Tile0: [0, 10240, 10240], Tile1: [0, 10240, 10240], Tile2; [0, 10240, 10240]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (short int *__restrict__) (resnet_L1_Memory+0);
	KerArg0->In2 = (short int *__restrict__) (resnet_L1_Memory+10240);
	KerArg0->Out = (short int *__restrict__) (resnet_L1_Memory+20480);
	KerArg0->W = (unsigned short int) (8);
	KerArg0->H = (unsigned short int) (10);
	KerArg0->N = (unsigned short int) (64);
	KerArg0->LB = (int) (-32768);
	KerArg0->UB = (int) (32767);
	KerArg0->In1_Q = (unsigned char) (10);
	KerArg0->In2_Q = (unsigned char) (10);
	KerArg0->Out_Q = (unsigned char) (9);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0), 10240, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+10240), 10240, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAddDynAdjust_fp, (void *) KerArg0);
			__CALL(KerParMatAddDynAdjust_fp, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+20480), 10240, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S70_AveragePool_2x2(
		short int * __restrict__ In,
		short int * __restrict__ Out)

{
	/* Shared L1: 12800 bytes, L2 buffer: 12800 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerReLUPool_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 64, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 10240 [D0, [0 x 10240, 10240]][Tile0, 1:[10x8], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 10240, 10240]][Tile0, 1:[10x8], 2]
		Tile0: [0, 10240, 10240], Tile1: [0, 10240, 10240], Tile2; [0, 10240, 10240]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2560 [D0, [0 x 2560, 2560]][Tile0, 1:[5x4], 2]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 2560, 2560]][Tile0, 1:[5x4], 2]
		Tile0: [0, 2560, 2560], Tile1: [0, 2560, 2560], Tile2; [0, 2560, 2560]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (short int * __restrict__) (resnet_L1_Memory+0);
	KerArg0->W = (unsigned short int) (10);
	KerArg0->UsedW = (unsigned short int) (10);
	KerArg0->H = (unsigned short int) (8);
	KerArg0->OutFeatures = (unsigned short int) (64);
	KerArg0->Out = (short int * __restrict__) (resnet_L1_Memory+10240);
	KerArg0->Pad = (v4s) 0;
	KerArg0->Orientation = (unsigned char) (1);
	KerArg0->Oper = (unsigned char) (2);
	KerArg0->LB = (int) (-32768);
	KerArg0->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0), 10240, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->UsedH = (unsigned short int) (8);
			AT_FORK(gap_ncore(), (void *) KerParPool2x2Stride2_fp, (void *) KerArg0);
			__CALL(KerParPool2x2Stride2_fp, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+10240), 2560, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S73_Linear_7x64x4x5(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Bias,
		short int * __restrict__ Out)

{
	/* Shared L1: 20512 bytes, L2 buffer: 20512 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	AT_HYPERRAM_CL_EVENT UchanHR1;
	KerLinearLayerReLU_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 7, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2560 [Tile0, 1:[1x1], 2560]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 2560]
		Tile0: [0, 2560, 2560], Tile1: [0, 2560, 2560], Tile2; [0, 2560, 2560]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 17920 [D0, [0 x 17920, 17920]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 17920, 17920]]
		Tile0: [0, 17920, 17920], Tile1: [0, 17920, 17920], Tile2; [0, 17920, 17920]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 14 [D0, [0 x 14, 14]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 14, 14]]
		Tile0: [0, 14, 14], Tile1: [0, 14, 14], Tile2; [0, 14, 14]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 14 [D0, [0 x 14, 14]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 14, 14]]
		Tile0: [0, 14, 14], Tile1: [0, 14, 14], Tile2; [0, 14, 14]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (short int * __restrict__) (resnet_L1_Memory+0);
	KerArg0->InSize = (unsigned short int) (1280);
	KerArg0->TotalInSize = (unsigned short int) (1280);
	KerArg0->OutSize = (unsigned short int) (7);
	KerArg0->Filter = (short int * __restrict__) (resnet_L1_Memory+2560);
	KerArg0->Bias = (short int * __restrict__) (resnet_L1_Memory+20480);
	KerArg0->Out = (short int * __restrict__) (resnet_L1_Memory+20496);
	KerArg0->Norm = (unsigned char) (15);
	KerArg0->NormBias = (signed char) (9);
	KerArg0->LB = (int) (-32768);
	KerArg0->UB = (int) (32767);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0), 2560, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) Filter+0), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory+27328+0), 17920, 0, &UchanHR1);
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR1); /* Wait previous uDMA read Filter */
	AT_L2_COPY(0, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L2_Memory+27328+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+2560), 17920, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+20480), 14, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Bias */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParLinearLayerReLU_fp, (void *) KerArg0);
			__CALL(KerParLinearLayerReLU_fp, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+20496), 14, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S74_SoftMax(
		short int * __restrict__ In,
		short int * __restrict__ Out)

{
	/* Shared L1: 32 bytes, L2 buffer: 32 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerSoftMax_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 14 [Tile0, 1:[1x7], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x7], 2]
		Tile0: [0, 14, 14], Tile1: [0, 14, 14], Tile2; [0, 14, 14]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 14 [Tile0, 1:[1x7], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x7], 2]
		Tile0: [0, 14, 14], Tile1: [0, 14, 14], Tile2; [0, 14, 14]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (short int *__restrict__) (resnet_L1_Memory+0);
	KerArg0->N = (unsigned short int) (7);
	KerArg0->Norm = (unsigned short int) (9);
	KerArg0->Out = (short int *__restrict__) (resnet_L1_Memory+16);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+0), 14, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) KerParSoftMax_fp, (void *) KerArg0);
		__CALL(KerParSoftMax_fp, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) resnet_L1_Memory+16), 14, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
int resnetCNN_Construct()

{
	AT_HYPERFLASH_FS_FC_EVENT UchanHF1;
	AT_HYPERRAM_FC_EVENT UchanHR2;
	AT_HYPERFLASH_FS_CONF_T HyperFlashConf;
	int Error;
	AT_HYPERFLASH_FS_CONF_INIT(&HyperFlashConf, AT_MEM_L3_HFLASH, 0);
	AT_HYPERFLASH_FS_OPEN(&HyperFlash, &HyperFlashConf, "resnet_L3_Flash_Const.dat", &Error);
	if (Error) return 1;
	resnet_L3_Memory = (AT_HYPERRAM_POINTER) AT_HYPERRAM_ALLOC(&HyperRam, 444590);
	if (resnet_L3_Memory == 0) return 2;
	resnet_L2_Memory = (AT_L2_POINTER) AT_L2_ALLOC(0, 200000);
	if (resnet_L2_Memory == 0) return 3;
	resnet_L1_Memory = (AT_L1_POINTER) AT_L1_ALLOC(0, 48576);
	if (resnet_L1_Memory == 0) return 4;
	/* Moving Res2bbranch2akernel, size 1152 from HyperFlash at 360704 to (size 1152) HyperRam at 360704..361855 */
	{
		int Size = 1152, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 360704+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 360704+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res2bbranch2bkernel, size 1152 from HyperFlash at 361856 to (size 1152) HyperRam at 361856..363007 */
	{
		int Size = 1152, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 361856+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 361856+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res2bbranch2bconv2d_bias, size 16 from HyperFlash at 368528 to (size 16) HyperRam at 367728..367743 */
	{
		int Size = 16, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 368528+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367728+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res2cbranch2akernel, size 1152 from HyperFlash at 363008 to (size 1152) HyperRam at 363008..364159 */
	{
		int Size = 1152, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 363008+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 363008+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res2cbranch2aconv2d_bias, size 16 from HyperFlash at 368544 to (size 16) HyperRam at 367744..367759 */
	{
		int Size = 16, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 368544+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367744+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res2cbranch2bkernel, size 1152 from HyperFlash at 364160 to (size 1152) HyperRam at 364160..365311 */
	{
		int Size = 1152, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 364160+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 364160+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res2cbranch2bconv2d_bias, size 16 from HyperFlash at 368560 to (size 16) HyperRam at 367760..367775 */
	{
		int Size = 16, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 368560+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367760+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res3a_branch2akernel, size 2304 from HyperFlash at 358400 to (size 2304) HyperRam at 358400..360703 */
	{
		int Size = 2304, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 358400+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 358400+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res3a_branch2aconv2d_bias, size 32 from HyperFlash at 368336 to (size 32) HyperRam at 367552..367583 */
	{
		int Size = 32, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 368336+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367552+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res3a_branch1conv2d_bias, size 32 from HyperFlash at 368368 to (size 32) HyperRam at 367584..367615 */
	{
		int Size = 32, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 368368+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367584+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res3a_branch2bconv2d_bias, size 32 from HyperFlash at 368400 to (size 32) HyperRam at 367616..367647 */
	{
		int Size = 32, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 368400+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367616+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res3bbranch2akernel, size 4608 from HyperFlash at 345088 to (size 4608) HyperRam at 345088..349695 */
	{
		int Size = 4608, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 345088+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 345088+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res3bbranch2aconv2d_bias, size 32 from HyperFlash at 368432 to (size 32) HyperRam at 367648..367679 */
	{
		int Size = 32, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 368432+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367648+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res3bbranch2bkernel, size 4608 from HyperFlash at 349696 to (size 4608) HyperRam at 349696..354303 */
	{
		int Size = 4608, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 349696+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 349696+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res3bbranch2bconv2d_bias, size 32 from HyperFlash at 368464 to (size 32) HyperRam at 367680..367711 */
	{
		int Size = 32, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 368464+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367680+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res4a_branch2akernel, size 9216 from HyperFlash at 331264 to (size 9216) HyperRam at 331264..340479 */
	{
		int Size = 9216, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 331264+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 331264+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res4a_branch2aconv2d_bias, size 64 from HyperFlash at 368016 to (size 64) HyperRam at 367232..367295 */
	{
		int Size = 64, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 368016+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367232+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res4a_branch1kernel, size 1024 from HyperFlash at 365312 to (size 1024) HyperRam at 365312..366335 */
	{
		int Size = 1024, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 365312+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 365312+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res4a_branch1conv2d_bias, size 64 from HyperFlash at 368080 to (size 64) HyperRam at 367296..367359 */
	{
		int Size = 64, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 368080+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367296+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res4a_branch2bconv2d_bias, size 64 from HyperFlash at 368144 to (size 64) HyperRam at 367360..367423 */
	{
		int Size = 64, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 368144+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367360+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res4bbranch2akernel, size 18432 from HyperFlash at 276480 to (size 18432) HyperRam at 276480..294911 */
	{
		int Size = 18432, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 276480+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 276480+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res4bbranch2aconv2d_bias, size 64 from HyperFlash at 368208 to (size 64) HyperRam at 367424..367487 */
	{
		int Size = 64, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 368208+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367424+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res4bbranch2bkernel, size 18432 from HyperFlash at 294912 to (size 18432) HyperRam at 294912..313343 */
	{
		int Size = 18432, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 294912+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 294912+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res4bbranch2bconv2d_bias, size 64 from HyperFlash at 368272 to (size 64) HyperRam at 367488..367551 */
	{
		int Size = 64, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 368272+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367488+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res5a_branch2akernel, size 36864 from HyperFlash at 221184 to (size 36864) HyperRam at 221184..258047 */
	{
		int Size = 36864, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 221184+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 221184+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res5a_branch1kernel, size 4096 from HyperFlash at 354304 to (size 4096) HyperRam at 354304..358399 */
	{
		int Size = 4096, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 354304+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 354304+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res5a_branch2bkernel, size 73728 from HyperFlash at 0 to (size 73728) HyperRam at 0..73727 */
	{
		int Size = 73728, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 0+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 0+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res5bbranch2akernel, size 73728 from HyperFlash at 73728 to (size 73728) HyperRam at 73728..147455 */
	{
		int Size = 73728, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 73728+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 73728+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Res5bbranch2bkernel, size 73728 from HyperFlash at 147456 to (size 73728) HyperRam at 147456..221183 */
	{
		int Size = 73728, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 147456+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 147456+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Full_connection7kerneltranspos, size 17920 from HyperFlash at 313344 to (size 17920) HyperRam at 313344..331263 */
	{
		int Size = 17920, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 313344+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 313344+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Full_connection7matmul_bias, size 14 from HyperFlash at 368576 to (size 14) HyperRam at 367776..367789 */
	{
		int Size = 14, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 368576+Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 0, &UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367776+Base), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 0), Chunk, 1, &UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, &UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Conv1kernel, size 784 from HyperFlash at 366336 to (size 784) L2 at 0..783 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 366336), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 0), 784, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving Conv1conv2d_bias, size 16 from HyperFlash at 368496 to (size 16) L2 at 784..799 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 368496), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 784), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving Res2bbranch2aconv2d_bias, size 16 from HyperFlash at 368512 to (size 16) L2 at 24736..24751 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 368512), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 24736), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving Res3a_branch1kernel, size 256 from HyperFlash at 367120 to (size 256) L2 at 23840..24095 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 367120), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 23840), 256, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving Res3a_branch2bkernel, size 4608 from HyperFlash at 340480 to (size 4608) L2 at 19232..23839 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 340480), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 19232), 4608, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving Res4a_branch2bkernel, size 18432 from HyperFlash at 258048 to (size 18432) L2 at 800..19231 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 258048), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 800), 18432, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving Res5a_branch2aconv2d_bias, size 128 from HyperFlash at 367376 to (size 128) L2 at 24096..24223 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 367376), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 24096), 128, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving Res5a_branch1conv2d_bias, size 128 from HyperFlash at 367504 to (size 128) L2 at 24224..24351 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 367504), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 24224), 128, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving Res5a_branch2bconv2d_bias, size 128 from HyperFlash at 367632 to (size 128) L2 at 24352..24479 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 367632), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 24352), 128, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving Res5bbranch2aconv2d_bias, size 128 from HyperFlash at 367760 to (size 128) L2 at 24480..24607 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 367760), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 24480), 128, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving Res5bbranch2bconv2d_bias, size 128 from HyperFlash at 367888 to (size 128) L2 at 24608..24735 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) resnet_L3_Flash + 367888), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) resnet_L2_Memory + 24608), 128, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	return 0;
}
int resnetCNN_Destruct()

{
	AT_HYPERRAM_FREE(&HyperRam, resnet_L3_Memory, 444590);
	AT_L2_FREE(0, resnet_L2_Memory, 200000);
	AT_L1_FREE(0, resnet_L1_Memory, 48576);
	AT_HYPERFLASH_FS_CLOSE(&HyperFlash);
	return 0;
}
int resnetCNN(
		signed short * __restrict__ Input_1,
		signed short * __restrict__ Output_1)

{
	AT_HYPERRAM_CL_EVENT UchanHR0;
	AT_HYPERRAM_CL_EVENT UchanHR1;
	AT_HYPERRAM_CL_EVENT UchanHR2;
	AT_HYPERRAM_CL_EVENT UchanHR3;
	AT_HYPERRAM_CL_EVENT UchanHR4;
	/* Moving Res2bbranch2akernel, size 1152 from HyperRam at 360704 to (size 1152) L2 at 178352 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 360704), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 178352), 1152, 0, &UchanHR0);
	/* Moving Res2bbranch2bkernel, size 1152 from HyperRam at 361856 to (size 1152) L2 at 198832 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 361856), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 198832), 1152, 0, &UchanHR1);
	/* Moving Res2bbranch2bconv2d_bias, size 16 from HyperRam at 367728 to (size 16) L2 at 199984 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367728), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 199984), 16, 0, &UchanHR2);
	S4_Conv2d_8x1x7x7_MaxPool_3x3_Relu(
		((signed short * __restrict__) Input_1), /* In */
		((signed short * __restrict__) (resnet_L2_Memory+0)), /* Filter */
		((signed short * __restrict__) (resnet_L2_Memory+784)), /* Bias */
		((signed short * __restrict__) (resnet_L2_Memory+24752)) /* Out */
	);
	/* Waiting completion of transfer of Res2bbranch2akernel using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR0);
	S7_Conv2d_8x8x3x3_Relu(
		((signed short * __restrict__) (resnet_L2_Memory+24752)), /* In */
		((signed short * __restrict__) (resnet_L2_Memory+178352)), /* Filter */
		((signed short * __restrict__) (resnet_L2_Memory+24736)), /* Bias */
		((signed short * __restrict__) (resnet_L2_Memory+101552)) /* Out */
	);
	/* Waiting completion of transfer of Res2bbranch2bkernel using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR1);
	/* Waiting completion of transfer of Res2bbranch2bconv2d_bias using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR2);
	S10_Conv2d_8x8x3x3(
		((signed short * __restrict__) (resnet_L2_Memory+101552)), /* In */
		((signed short * __restrict__) (resnet_L2_Memory+198832)), /* Filter */
		((signed short * __restrict__) (resnet_L2_Memory+199984)), /* Bias */
		((signed short * __restrict__) (resnet_L3_Memory+367792)) /* Out */
	);
	/* Moving Res2cbranch2bkernel, size 1152 from HyperRam at 364160 to (size 1152) L2 at 198832 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 364160), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 198832), 1152, 0, &UchanHR0);
	/* Moving Res2cbranch2bconv2d_bias, size 16 from HyperRam at 367760 to (size 16) L2 at 199984 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367760), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 199984), 16, 0, &UchanHR1);
	S11_MatAdd_8x60x80(
		((signed short * __restrict__) (resnet_L3_Memory+367792)), /* In1 */
		((signed short * __restrict__) (resnet_L2_Memory+24752)), /* In2 */
		((signed short * __restrict__) (resnet_L2_Memory+101552)) /* Out */
	);
	/* Moving Res2cbranch2akernel, size 1152 from HyperRam at 363008 to (size 1152) L2 at 178352 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 363008), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 178352), 1152, 0, &UchanHR2);
	/* Moving Res2cbranch2aconv2d_bias, size 16 from HyperRam at 367744 to (size 16) L2 at 179504 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367744), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 179504), 16, 0, &UchanHR3);
	/* Waiting completion of transfer of Res2cbranch2akernel using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR2);
	/* Waiting completion of transfer of Res2cbranch2aconv2d_bias using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR3);
	S14_Conv2d_8x8x3x3_Relu(
		((signed short * __restrict__) (resnet_L2_Memory+101552)), /* In */
		((signed short * __restrict__) (resnet_L2_Memory+178352)), /* Filter */
		((signed short * __restrict__) (resnet_L2_Memory+179504)), /* Bias */
		((signed short * __restrict__) (resnet_L2_Memory+24752)) /* Out */
	);
	/* Waiting completion of transfer of Res2cbranch2bkernel using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR0);
	/* Waiting completion of transfer of Res2cbranch2bconv2d_bias using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR1);
	S17_Conv2d_8x8x3x3(
		((signed short * __restrict__) (resnet_L2_Memory+24752)), /* In */
		((signed short * __restrict__) (resnet_L2_Memory+198832)), /* Filter */
		((signed short * __restrict__) (resnet_L2_Memory+199984)), /* Bias */
		((signed short * __restrict__) (resnet_L3_Memory+367792)) /* Out */
	);
	S18_MatAdd_8x60x80(
		((signed short * __restrict__) (resnet_L3_Memory+367792)), /* In1 */
		((signed short * __restrict__) (resnet_L2_Memory+101552)), /* In2 */
		((signed short * __restrict__) (resnet_L2_Memory+24752)) /* Out */
	);
	/* Moving Res3a_branch2akernel, size 2304 from HyperRam at 358400 to (size 2304) L2 at 139952 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 358400), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 139952), 2304, 0, &UchanHR0);
	/* Moving Res3a_branch2aconv2d_bias, size 32 from HyperRam at 367552 to (size 32) L2 at 142256 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367552), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 142256), 32, 0, &UchanHR1);
	/* Moving Res3a_branch1conv2d_bias, size 32 from HyperRam at 367584 to (size 32) L2 at 178608 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367584), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 178608), 32, 0, &UchanHR2);
	/* Moving Res3a_branch2bconv2d_bias, size 32 from HyperRam at 367616 to (size 32) L2 at 182960 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367616), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 182960), 32, 0, &UchanHR3);
	/* Waiting completion of transfer of Res3a_branch2akernel using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR0);
	/* Waiting completion of transfer of Res3a_branch2aconv2d_bias using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR1);
	S21_Conv2d_16x8x3x3_Relu(
		((signed short * __restrict__) (resnet_L2_Memory+24752)), /* In */
		((signed short * __restrict__) (resnet_L2_Memory+139952)), /* Filter */
		((signed short * __restrict__) (resnet_L2_Memory+142256)), /* Bias */
		((signed short * __restrict__) (resnet_L2_Memory+101552)) /* Out */
	);
	/* Waiting completion of transfer of Res3a_branch2bconv2d_bias using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR3);
	S27_Conv2d_16x16x3x3_Relu(
		((signed short * __restrict__) (resnet_L2_Memory+101552)), /* In */
		((signed short * __restrict__) (resnet_L2_Memory+19232)), /* Filter */
		((signed short * __restrict__) (resnet_L2_Memory+182960)), /* Bias */
		((signed short * __restrict__) (resnet_L2_Memory+139952)) /* Out */
	);
	/* Waiting completion of transfer of Res3a_branch1conv2d_bias using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR2);
	S24_Conv2d_16x8x1x1(
		((signed short * __restrict__) (resnet_L2_Memory+24752)), /* In */
		((signed short * __restrict__) (resnet_L2_Memory+23840)), /* Filter */
		((signed short * __restrict__) (resnet_L2_Memory+178608)), /* Bias */
		((signed short * __restrict__) (resnet_L2_Memory+101552)) /* Out */
	);
	S28_MatAdd_16x30x40(
		((signed short * __restrict__) (resnet_L2_Memory+139952)), /* In1 */
		((signed short * __restrict__) (resnet_L2_Memory+101552)), /* In2 */
		((signed short * __restrict__) (resnet_L2_Memory+24752)) /* Out */
	);
	/* Moving Res3bbranch2akernel, size 4608 from HyperRam at 345088 to (size 4608) L2 at 101552 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 345088), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 101552), 4608, 0, &UchanHR0);
	/* Moving Res3bbranch2aconv2d_bias, size 32 from HyperRam at 367648 to (size 32) L2 at 106160 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367648), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 106160), 32, 0, &UchanHR1);
	/* Moving Res3bbranch2bkernel, size 4608 from HyperRam at 349696 to (size 4608) L2 at 139952 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 349696), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 139952), 4608, 0, &UchanHR2);
	/* Moving Res3bbranch2bconv2d_bias, size 32 from HyperRam at 367680 to (size 32) L2 at 144560 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367680), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 144560), 32, 0, &UchanHR3);
	/* Moving Res4a_branch2aconv2d_bias, size 64 from HyperRam at 367232 to (size 64) L2 at 149168 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367232), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 149168), 64, 0, &UchanHR4);
	/* Waiting completion of transfer of Res3bbranch2akernel using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR0);
	/* Waiting completion of transfer of Res3bbranch2aconv2d_bias using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR1);
	S31_Conv2d_16x16x3x3_Relu(
		((signed short * __restrict__) (resnet_L2_Memory+24752)), /* In */
		((signed short * __restrict__) (resnet_L2_Memory+101552)), /* Filter */
		((signed short * __restrict__) (resnet_L2_Memory+106160)), /* Bias */
		((signed short * __restrict__) (resnet_L2_Memory+63152)) /* Out */
	);
	/* Waiting completion of transfer of Res3bbranch2bkernel using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR2);
	/* Waiting completion of transfer of Res3bbranch2bconv2d_bias using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR3);
	S34_Conv2d_16x16x3x3(
		((signed short * __restrict__) (resnet_L2_Memory+63152)), /* In */
		((signed short * __restrict__) (resnet_L2_Memory+139952)), /* Filter */
		((signed short * __restrict__) (resnet_L2_Memory+144560)), /* Bias */
		((signed short * __restrict__) (resnet_L2_Memory+101552)) /* Out */
	);
	/* Moving Res4a_branch2akernel, size 9216 from HyperRam at 331264 to (size 9216) L2 at 139952 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 331264), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 139952), 9216, 0, &UchanHR0);
	S35_MatAdd_16x30x40(
		((signed short * __restrict__) (resnet_L2_Memory+101552)), /* In1 */
		((signed short * __restrict__) (resnet_L2_Memory+24752)), /* In2 */
		((signed short * __restrict__) (resnet_L2_Memory+63152)) /* Out */
	);
	/* Moving Res4a_branch1kernel, size 1024 from HyperRam at 365312 to (size 1024) L2 at 120752 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 365312), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 120752), 1024, 0, &UchanHR1);
	/* Moving Res4a_branch1conv2d_bias, size 64 from HyperRam at 367296 to (size 64) L2 at 121776 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367296), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 121776), 64, 0, &UchanHR2);
	/* Moving Res4a_branch2bconv2d_bias, size 64 from HyperRam at 367360 to (size 64) L2 at 62384 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367360), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 62384), 64, 0, &UchanHR3);
	/* Waiting completion of transfer of Res4a_branch2akernel using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR0);
	/* Waiting completion of transfer of Res4a_branch2aconv2d_bias using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR4);
	S38_Conv2d_32x16x3x3_Relu(
		((signed short * __restrict__) (resnet_L2_Memory+63152)), /* In */
		((signed short * __restrict__) (resnet_L2_Memory+139952)), /* Filter */
		((signed short * __restrict__) (resnet_L2_Memory+149168)), /* Bias */
		((signed short * __restrict__) (resnet_L2_Memory+24752)) /* Out */
	);
	/* Waiting completion of transfer of Res4a_branch2bconv2d_bias using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR3);
	S44_Conv2d_32x32x3x3_Relu(
		((signed short * __restrict__) (resnet_L2_Memory+24752)), /* In */
		((signed short * __restrict__) (resnet_L2_Memory+800)), /* Filter */
		((signed short * __restrict__) (resnet_L2_Memory+62384)), /* Bias */
		((signed short * __restrict__) (resnet_L2_Memory+101552)) /* Out */
	);
	/* Moving Res5a_branch2akernel, size 36864 from HyperRam at 221184 to (size 36864) L2 at 147120 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 221184), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 147120), 36864, 0, &UchanHR0);
	/* Waiting completion of transfer of Res4a_branch1kernel using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR1);
	/* Waiting completion of transfer of Res4a_branch1conv2d_bias using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR2);
	S41_Conv2d_32x16x1x1(
		((signed short * __restrict__) (resnet_L2_Memory+63152)), /* In */
		((signed short * __restrict__) (resnet_L2_Memory+120752)), /* Filter */
		((signed short * __restrict__) (resnet_L2_Memory+121776)), /* Bias */
		((signed short * __restrict__) (resnet_L2_Memory+43952)) /* Out */
	);
	/* Moving Res4bbranch2akernel, size 18432 from HyperRam at 276480 to (size 18432) L2 at 63152 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 276480), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 63152), 18432, 0, &UchanHR1);
	/* Moving Res4bbranch2aconv2d_bias, size 64 from HyperRam at 367424 to (size 64) L2 at 81584 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367424), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 81584), 64, 0, &UchanHR2);
	/* Moving Res4bbranch2bkernel, size 18432 from HyperRam at 294912 to (size 18432) L2 at 82352 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 294912), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 82352), 18432, 0, &UchanHR3);
	/* Moving Res4bbranch2bconv2d_bias, size 64 from HyperRam at 367488 to (size 64) L2 at 100784 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367488), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 100784), 64, 0, &UchanHR4);
	S45_MatAdd_32x15x20(
		((signed short * __restrict__) (resnet_L2_Memory+101552)), /* In1 */
		((signed short * __restrict__) (resnet_L2_Memory+43952)), /* In2 */
		((signed short * __restrict__) (resnet_L2_Memory+24752)) /* Out */
	);
	/* Waiting completion of transfer of Res4bbranch2akernel using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR1);
	/* Waiting completion of transfer of Res4bbranch2aconv2d_bias using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR2);
	S48_Conv2d_32x32x3x3_Relu(
		((signed short * __restrict__) (resnet_L2_Memory+24752)), /* In */
		((signed short * __restrict__) (resnet_L2_Memory+63152)), /* Filter */
		((signed short * __restrict__) (resnet_L2_Memory+81584)), /* Bias */
		((signed short * __restrict__) (resnet_L2_Memory+43952)) /* Out */
	);
	/* Waiting completion of transfer of Res4bbranch2bkernel using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR3);
	/* Waiting completion of transfer of Res4bbranch2bconv2d_bias using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR4);
	S51_Conv2d_32x32x3x3(
		((signed short * __restrict__) (resnet_L2_Memory+43952)), /* In */
		((signed short * __restrict__) (resnet_L2_Memory+82352)), /* Filter */
		((signed short * __restrict__) (resnet_L2_Memory+100784)), /* Bias */
		((signed short * __restrict__) (resnet_L2_Memory+63152)) /* Out */
	);
	S52_MatAdd_32x15x20(
		((signed short * __restrict__) (resnet_L2_Memory+63152)), /* In1 */
		((signed short * __restrict__) (resnet_L2_Memory+24752)), /* In2 */
		((signed short * __restrict__) (resnet_L2_Memory+43952)) /* Out */
	);
	/* Moving Res5a_branch1kernel, size 4096 from HyperRam at 354304 to (size 4096) L2 at 34992 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 354304), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 34992), 4096, 0, &UchanHR1);
	/* Moving Res5a_branch2bkernel, size 73728 from HyperRam at 0 to (size 73728) L2 at 73392 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 0), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 73392), 73728, 0, &UchanHR2);
	/* Waiting completion of transfer of Res5a_branch2akernel using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR0);
	S55_Conv2d_64x32x3x3_Relu(
		((signed short * __restrict__) (resnet_L2_Memory+43952)), /* In */
		((signed short * __restrict__) (resnet_L2_Memory+147120)), /* Filter */
		((signed short * __restrict__) (resnet_L2_Memory+24096)), /* Bias */
		((signed short * __restrict__) (resnet_L2_Memory+24752)) /* Out */
	);
	/* Waiting completion of transfer of Res5a_branch2bkernel using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR2);
	S61_Conv2d_64x64x3x3_Relu(
		((signed short * __restrict__) (resnet_L2_Memory+24752)), /* In */
		((signed short * __restrict__) (resnet_L2_Memory+73392)), /* Filter */
		((signed short * __restrict__) (resnet_L2_Memory+24352)), /* Bias */
		((signed short * __restrict__) (resnet_L2_Memory+63152)) /* Out */
	);
	/* Moving Res5bbranch2akernel, size 73728 from HyperRam at 73728 to (size 73728) L2 at 73392 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 73728), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 73392), 73728, 0, &UchanHR0);
	/* Waiting completion of transfer of Res5a_branch1kernel using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR1);
	S58_Conv2d_64x32x1x1(
		((signed short * __restrict__) (resnet_L2_Memory+43952)), /* In */
		((signed short * __restrict__) (resnet_L2_Memory+34992)), /* Filter */
		((signed short * __restrict__) (resnet_L2_Memory+24224)), /* Bias */
		((signed short * __restrict__) (resnet_L2_Memory+24752)) /* Out */
	);
	S62_MatAdd_64x8x10(
		((signed short * __restrict__) (resnet_L2_Memory+63152)), /* In1 */
		((signed short * __restrict__) (resnet_L2_Memory+24752)), /* In2 */
		((signed short * __restrict__) (resnet_L2_Memory+34992)) /* Out */
	);
	/* Waiting completion of transfer of Res5bbranch2akernel using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR0);
	S65_Conv2d_64x64x3x3_Relu(
		((signed short * __restrict__) (resnet_L2_Memory+34992)), /* In */
		((signed short * __restrict__) (resnet_L2_Memory+73392)), /* Filter */
		((signed short * __restrict__) (resnet_L2_Memory+24480)), /* Bias */
		((signed short * __restrict__) (resnet_L2_Memory+24752)) /* Out */
	);
	/* Moving Res5bbranch2bkernel, size 73728 from HyperRam at 147456 to (size 73728) L2 at 65712 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 147456), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 65712), 73728, 0, &UchanHR0);
	/* Waiting completion of transfer of Res5bbranch2bkernel using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR0);
	S68_Conv2d_64x64x3x3(
		((signed short * __restrict__) (resnet_L2_Memory+24752)), /* In */
		((signed short * __restrict__) (resnet_L2_Memory+65712)), /* Filter */
		((signed short * __restrict__) (resnet_L2_Memory+24608)), /* Bias */
		((signed short * __restrict__) (resnet_L2_Memory+55472)) /* Out */
	);
	S69_MatAdd_64x8x10(
		((signed short * __restrict__) (resnet_L2_Memory+55472)), /* In1 */
		((signed short * __restrict__) (resnet_L2_Memory+34992)), /* In2 */
		((signed short * __restrict__) (resnet_L2_Memory+45232)) /* Out */
	);
	/* Moving Full_connection7matmul_bias, size 14 from HyperRam at 367776 to (size 14) L2 at 55472 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) resnet_L3_Memory + 367776), ((AT_HYPERRAM_INT_ADDR_TYPE) resnet_L2_Memory + 55472), 14, 0, &UchanHR0);
	S70_AveragePool_2x2(
		((signed short * __restrict__) (resnet_L2_Memory+45232)), /* In */
		((signed short * __restrict__) (resnet_L2_Memory+24752)) /* Out */
	);
	/* Waiting completion of transfer of Full_connection7matmul_bias using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, &UchanHR0);
	S73_Linear_7x64x4x5(
		((signed short * __restrict__) (resnet_L2_Memory+24752)), /* In */
		((signed short * __restrict__) (resnet_L3_Memory+313344)), /* Filter */
		((signed short * __restrict__) (resnet_L2_Memory+55472)), /* Bias */
		((signed short * __restrict__) (resnet_L2_Memory+27312)) /* Out */
	);
	S74_SoftMax(
		((signed short * __restrict__) (resnet_L2_Memory+27312)), /* In */
		((signed short * __restrict__) Output_1) /* Out */
	);
	return 0;
}
