#! /usr/bin/env python
#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#*************************************************************************************************
import subprocess
import sys
import os
import time

distributed = 0
papi = 0
installdir = "/home/min/a/hegden/Research/delme"
app = "treelogy/SHM/barneshut"
versions = ["octree/block", "kdtree/block"]
execs = ["barneshut"]

#versions.append(Version("block", ""))

runs = 5
numts = [128,64,32,16,8,4,2,1]
blocks = [64]

mail_text = ""
args = [
	"/home/min/a/hegden/Research/BarnesHut/inputs/BarnesHutC.in 1000000",
	"/home/min/a/hegden/Research/BarnesHut/inputs/BHRan10M.in 10000000",
	]
def exec_cmd(cmd):
  global mail_text
  print cmd
  mail_text = mail_text + cmd + "\n"
  ret = subprocess.call(cmd, shell=True)
  if ret != 0:
    print cmd + " failed\n"
    mail_text = mail_text + "failed\n"
    #send_email()
    sys.exit(0)
 
def get_cmd(execname, block, arg, numt = 1, runs = 8):
  if distributed == 1:
    return "mpirun --hostfile /home/min/a/hegden/Research/mpd_wabash.hosts -np {3} ./{0} {2} BLOCK_SIZE={1} SUBTREE_HEIGHT=100 BATCH_SIZE=1000000".format(
        execname, block, arg, numt, runs)
  else:
    return "./{0} -t {3} --block {1} {2}".format(execname, block, arg, numt, runs)

for execname in execs:
  for version in versions:
    os.chdir(installdir + "/" + app + "/" + version)
    exec_cmd("make clean")
    exec_cmd("make")
    print os.getcwd()
    for arg in args:
      for block in blocks:
        for numt in numts:
          for run in range(0, runs):
            cmd = get_cmd(execname, block, arg, numt, run)
            exec_cmd(cmd)
	    time.sleep(10)
					
