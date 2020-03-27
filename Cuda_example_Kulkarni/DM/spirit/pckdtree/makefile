#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
# * Purdue University. All Rights Reserved. See Copyright.txt
#*************************************************************************************************/
all : KD

include ../common/Makefile.common

BIN_DEPS += PCKDTreeVisitor.cpp Test.cpp

KD: $(BIN_DEPS)
	$(CPP) $(CFLAGS) $(INCLUDES) -o $@ $^
clean:
	-rm KD
	

