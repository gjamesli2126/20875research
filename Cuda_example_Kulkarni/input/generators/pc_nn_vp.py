#! /usr/bin/env python2.6

import os, sys, random

DIMENSION=7
N=10000000
if len(sys.argv) >= 3:
    N = int(sys.argv[1])
    DIMENSION = int(sys.argv[2])

for i in range(N):
#    vals=[(1-(random.random()*2))*100 for x in range(DIMENSION)]
    vals=[random.random() for x in range(DIMENSION)]
    for j in range(DIMENSION):
        sys.stdout.write("%f " % vals[j])
	
    sys.stdout.write("\n")

sys.exit(0)
