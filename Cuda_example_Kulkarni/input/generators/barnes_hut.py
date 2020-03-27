#! /usr/bin/env python2.6

import os, sys, random

N=1000000
if len(sys.argv) >= 2:
    N = int(sys.argv[1])

if (N > 1000000):
    mass = 1.0/1000000
else:
    mass=1.0/N

print N
print 5    #number of time steps
print "0.025"  
print "0.05" 
print "0.5" #tol / angle-of-opening

for i in range(N):
    vals=[random.random() for x in range(6)]
    vals.insert(0, mass)
    
    sys.stdout.write("%f %f %f %f %f %f %f\n" % tuple(vals))

sys.exit(0)
