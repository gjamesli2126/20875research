#! /usr/bin/env python
#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#*************************************************************************************************

import os
import sys
import pickle
import math
import numpy
import mean_confidence_interval
from mean_confidence_interval import *
#import pretty_table

if len(sys.argv) < 2:
    print "usage: summarize.py table_file1 [table_file2] [table_file3] ..."
    print "   table_file is a file name of the table built by running collect.py"
    print
    sys.exit(1)

for table_file in sys.argv[1:]:

    table={}
    summary={}

    try:
        f=open(table_file, "r")
        table=pickle.load(f)
        f.close()
    except Exception as e:
        print "Could not load table from %s: %s" % (table_file,e)
        sys.exit(1)

        # summarize all of the data to:
        # count sum min max avg stddev
	
    name_length=20
    value_length=10

#    for col in table:
#        row = table[col]
#        mean = sum([float(x) for x in row])/len(row)
#	print "%s\n" % (col),
#	for x in row:
#	    print "\t%s\n" % (str(x)),
#	print ("\tmean: %15.3f\n") % (mean)
#
#    print

    for col in table:
        row = table[col]
        print "%s\n" % (col),
#   for x in row:
#       print "\t%10s" % (str(x))
#        mean = numpy.average(row);
        print ("mean: %10.3f") % (numpy.average(row)),
        print ("\tmedian: %10.3f") % (numpy.median(row)),
        print ("\tstd: %10.3f") % (numpy.std(row)),
        print ("\tconfidence field: %10.3f +/- %0.3f") % (mean_confidence_interval(row))
        print 


sys.exit(0)
