#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
# * Purdue University. All Rights Reserved. See Copyright.txt
#*************************************************************************************************/
all : BH

include ../common/Makefile.common

BIN_DEPS += OctreeVisitor.cpp Test.cpp

BH: $(BIN_DEPS)
	$(CPP) $(CFLAGS) $(INCLUDES) -o $@ $^
clean:
	-rm BH
	

