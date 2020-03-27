#! /usr/bin/env python
# Collects output into a table of results
# collect.py table_file
#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#*************************************************************************************************

import os
import sys
import pickle

if len(sys.argv) != 2:
    print "usage: collect.py table_file"
    print "  Results should be redurected into this script for processing"
    print "  data will be merged with the existing table_file if it exits."
    print
    sys.exit(1)

table_file=sys.argv[1]
table={}

if os.path.exists(table_file):
    print "Loading data from %s..." % (table_file,)
    
    try:
        f = open(table_file, "r")
        table=pickle.load(f)
        f.close()

    except Exception as e:
        print "Could not read/parse %s: %s" % (table_file,e)
        sys.exit(1)

    else:
        print table
else:
    print "Results wll be collected to %s..." % (table_file,)

for line in sys.stdin:
    line = line.strip()
    if line.startswith("@"):
        data=line.split()
        name=data[1]
        for i in range(2, len(data)):
            try:
                value=float(data[i])
            except Exception as e:
                pass # just ignore non numeric stuff
            else:
                # add the item
                if name in table:
                    table[name].append(value)
                else:
                    table[name] = [value]
        
    sys.stdout.write("%s\n" % (line,))

# write the table back out to memory
try:
    f=open(table_file, "w")
    pickle.dump(table, f)
    f.close()
except Exception as e:
    print "Could not save table. %s" % (e,)
    sys.exit(1)

print "Collected results into %s" % (table_file)
print

sys.exit(0)
