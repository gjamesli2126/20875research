#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
# * Purdue University. All Rights Reserved. See Copyright.txt
#*************************************************************************************************/
import subprocess
import sys
import os
import time
import re

#For each application in apps, a matching test case is run from args. 
#Each test case is represented as a tuple: <app, exename, args, number of nodes traversed>. The last value in the tuple is used for validating.
#For each test case, number of processes (numts) and BLOCK_SIZE (blocks) can be varied. SUBTREE_HEIGHT needs to be specified along with the args field of the test case.

installdir = "/home/min/a/hegden/Research/spirit_sourcecode"
apps = ["pckdtree", "nnkdtree", "nnvptree", "bhoctree"]  
test1 = 'pckdtree', 'KD', '/home/min/a/hegden/ECE573/PointCorrInputs/mnist.txt 1000000 0.03 SUBTREE_HEIGHT=10', 3188374984 
test2 = 'nnkdtree', 'KD', '/home/min/a/hegden/ECE573/PointCorrInputs/mnist.txt 1000000 SUBTREE_HEIGHT=10', 3696977726
test3 = 'nnvptree', 'VP', '/home/min/a/hegden/ECE573/PointCorrInputs/mnist.txt 1000000 SUBTREE_HEIGHT=10', 1409164839
test4 = 'bhoctree', 'BH', '/home/min/a/hegden/Research/BarnesHut/inputs/BarnesHutC.in SUBTREE_HEIGHT=8', 2708787795
#test4 = 'bhoctree', 'BH', '/home/min/a/hegden/Research/BarnesHut/LoadBalance/Trunk/Baseline/testinputs/test.in SUBTREE_HEIGHT=7', 12618093
args = test1, test2, test3, test4

runs = 1
numts = [16]
blocks = [4096]

mail_text = ""
def exec_cmd(cmd):
  global mail_text
  #print cmd
  mail_text = mail_text + cmd + "\n"
  ret = subprocess.check_output(cmd, shell=True)
  print ret
  result = re.search('(.*) traversed (.*)',ret)
  if result:
    numtraversed=result.group().split(' ')
    return numtraversed[3]
  else:
    return 0

def get_cmd(execname, block, arg, numt = 1, runs = 8):
  return "mpirun --hostfile /home/min/a/hegden/Research/mpd_wabash.hosts -np {3} ./{0} {2} BLOCK_SIZE={1}".format(execname, block, arg, numt, runs)

for app in apps:
    os.chdir(installdir + "/" + app)
    print os.getcwd()
    exec_cmd("make clean")
    if app == "bhoctree":
      exec_cmd("make AGGR=1 DIMENSION=3")
    else:
      exec_cmd("make AGGR=1")
    for arg in args:
      if arg[0] == app:
        for block in blocks:
	    tmparg = arg[2]+" MAX_POINTS_PER_CELL=1"
            for numt in numts:
  	      if numt == 256:
	        tmparg = tmparg + " BATCH_SIZE=800000"
              for run in range(0, runs):
                cmd = get_cmd(arg[1], block, tmparg, numt, run)
	        print cmd
                result = exec_cmd(cmd)
		if int(result) != arg[3]:
		  print "Test Failed {0}, {1}\n".format(result,arg[3])
		else:
		  print "Test Passed\n"	
	        time.sleep(10)
					
