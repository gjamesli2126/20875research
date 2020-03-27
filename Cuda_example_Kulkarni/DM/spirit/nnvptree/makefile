#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
# * Purdue University. All Rights Reserved. See Copyright.txt
#*************************************************************************************************/
all : VP

include ../common/Makefile.common

BIN_DEPS += VPTreeVisitor.cpp Test.cpp

VP: $(BIN_DEPS)
	$(CPP) $(CFLAGS) $(INCLUDES) -o $@ $^
clean:
	-rm VP
	

