#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#*************************************************************************************************
import subprocess
import sys
import os
import time

distributed = 1
installdir = "/home/min/a/hegden/Research/delme"
app = "treelogy/DM"
versions = ["fastmultipole"]
execs = ["FMM"]

runs = 10
numts = [1,2,4,8,16,32,64,128]
blocks = [1]
blockintakelimit = [1]
spad = [0]

mail_text = ""
args = ["/home/min/a/hegden/ECE573/PointCorrInputs/input2.txt 1000000"]
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
  if(distributed == 1):
    return "mpirun --hostfile /home/min/a/hegden/Research/mpd_wabash.hosts -np {3} ./{0} {2} SUBTREE_HEIGHT=100 BATCH_SIZE=131072".format(execname, block, arg, numt, runs)
  else:
    return "./{0} {2} {3}".format(execname, block, arg, numt, runs)

for version in versions:
  os.chdir(installdir + "/" + app + "/" + version)
  print os.getcwd()
  for arg in args:
      for block in blocks:
        for numt in numts:
    	  for execname in execs:
            for run in range(0, runs):
              cmd = get_cmd(execname, block, arg, numt, run)
              exec_cmd(cmd)
	      time.sleep(10)
		
