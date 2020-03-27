#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#*************************************************************************************************
import subprocess
import sys
import os
import time

distributed = 0
installdir = "/home/min/a/hegden/Research/delme"
app = "treelogy/SHM"
versions = ["knearestneighbor/balltree/block"]
execs = ["knn"]

#versions.append(Version("block", ""))

runs = 1
numts = [128,64,32,16,8,4,2,1]
blocks = [4096]

mail_text = ""
args = [
	"1 500000 500000 /home/min/a/hegden/ECE573/PointCorrInputs/mnist.txt",
	"1 5000000 5000000 /home/min/a/hegden/ECE573/PointCorrInputs/VP10M_7D.txt"
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
    return "mpirun --hostfile /home/min/a/hegden/Research/mpd_wabash.hosts -np {3} ./{0} {2}".format(execname, block, arg, numt, runs)
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
					
