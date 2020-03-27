#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
# * Purdue University. All Rights Reserved. See Copyright.txt
#************************************************************************************************/
#flags=-DPRETTY_PRINT -G -DCPU_KD -DGPU_KD -DGPU -lm -arch sm_35
flags=-DPRETTY_PRINT -G -DCPU_KD -lm -arch sm_35
all: kdkmeans

kdkmeans : kdkmeans.cu kd.cu kd.h common.h
	nvcc -o kdkmeans kdkmeans.cu $(flags)

# Can remove -G flag for GPU. But GPU_KD code is not working without
# -G. So -G needs to be there for GPU_KD. (-G specifies -O0).

clean : 
	-rm -rf *.o *~ kdkmeans
