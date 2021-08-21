
#########################################
#0-4 put inside file model_rules.mk and include bottom in Makefile
#0. prepare MODEL_BUILD
$(MODEL_BUILD):
	mkdir $(MODEL_BUILD)

# Converts the H5 file to TFLITE format
$(MODEL_TFLITE): $(MODEL_BUILD)
	cp $(MODEL_TFLITE) $(MODEL_BUILD)

#1. use nntool to generate body_detection.json
#MODEL_STATE = body_detection.json(comes from nntool save state)
$(MODEL_STATE): $(MODEL_TFLITE)  # MODEL_TFLITE =BUILD_MODEL_16BIT/body_detection.tflite
	nntool -s $(NNTOOL_SCRIPT) $(MODEL_BUILD)/$(MODEL_NAME)

#2. use nntool to generate body_detectionModel.c, tensors and body_detectionInfo.h 
# Runs NNTOOL with its state file to generate the autotiler model code
$(MODEL_BUILD)/$(MODEL_SRC): $(MODEL_STATE) $(MODEL_TFLITE) #MODEL_BUILD_16BIT/body_detectionModel.c: MODEL_BUILD_16BIT/body_detection.json(comes from nntool save state)  model/body_detection.tflite
	nntool -g $(MODEL_STATE) -M $(MODEL_BUILD) -m $(MODEL_SRC) -T $(TENSORS_DIR) -H $(MODEL_HEADER) $(MODEL_GENFLAGS_EXTRA)


#3. compile ./GenTile
# Build the code generator from the model code
$(MODEL_GEN_EXE): $(MODEL_BUILD)/$(MODEL_SRC)   #./GenTile: MODEL_BUILD_16BIT/body_detectionModel.c
	gcc -g -o $(MODEL_GEN_EXE) -I. -I$(TILER_INC) -I$(TILER_EMU_INC) $(CNN_GEN_INCLUDE) $(CNN_LIB_INCLUDE) $(MODEL_BUILD)/$(MODEL_SRC) $(CNN_GEN) $(TILER_LIB) ${RESIZE_GEN}


#4. ./GenTile will generate body_detectionKernels.c/h
# Run the code generator to generate GAP graph and kernel code
##MODEL_GEN_C = MODEL_BUILD_16BIT/body_detectionKernels.c
#Here ./GenTile generate body_detectionKernels.c/h  automatically
$(MODEL_GEN_C): $(MODEL_GEN_EXE)  #MODEL_GEN_EXE= ./GenTile  inside the  MODEL_BUILD_16BIT/body_detectionModel.c, there is a main() entry to GenerateTilingCode();
	$(MODEL_GEN_EXE) -o $(MODEL_BUILD) -c $(MODEL_BUILD) $(MODEL_GEN_EXTRA_FLAGS)

###############################################
