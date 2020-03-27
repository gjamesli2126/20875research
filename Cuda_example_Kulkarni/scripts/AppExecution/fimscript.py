#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#*************************************************************************************************
import subprocess
import sys
import os
import time

papi = 0
distributed = 0
installdir = "/home/min/a/hegden/Research/delme"
app = "treelogy"
versions = ["SHM/freqmine"]
execs = ["fpgrowth"]

#versions.append(Version("block", ""))

runs = 1
numts = [1]
blocks = [1]
blockintakelimit = [1]
depths = [1]
spad = [0]

mail_text = ""
args = ["TRANSACTION_DB=/home/min/a/hegden/Research/FPGrowth/datasets/BMS2.txt",
	"TRANSACTION_DB=/home/min/a/hegden/Research/FPGrowth/datasets/syn5.txt",
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
 
def get_cmd(execname, block, arg, depth, numt = 1, runs = 8):
  if distributed == 1:
    return "mpirun --hostfile /home/min/a/hegden/Research/mpd_wabash.hosts -np {4} ./{0} {2} BLOCK_SIZE={1} SPLICE_DEPTH={3} MIN_SUPPORT=40 BATCH_SIZE=5000000".format(execname, block, arg, depth, numt, runs)
  else:
    return "./{0} {2} BLOCK_SIZE={1} SPLICE_DEPTH={3} MIN_SUPPORT=40 NUM_THREADS={4}".format(
    execname, block, arg, depth, numt, runs)


for version in versions:
  if version == "DSM/freqmine":
    distributed=1
  os.chdir(installdir + "/" + app + "/" + version)
  exec_cmd("make clean")
  exec_cmd("make")
  print os.getcwd()
  for arg in args:
    for block in blocks:
      for depth in depths:
        for numt in numts:
          for execname in execs:
            for run in range(0, runs):
              cmd = get_cmd(execname, block, arg, depth, numt, run)
              exec_cmd(cmd)
	      time.sleep(10)
		
