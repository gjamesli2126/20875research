#! /usr/bin/env python
#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#*************************************************************************************************
import subprocess
import sys
import os
import time
import shutil

# System Path
Result_Path		= "/home/min/a/liu1274/TTBS/results/"
Temp_Path		= "/home/min/a/liu1274/TTBS/results/tmp/"
Time_Path		= "/home/min/a/liu1274/TTBS/results/time/"
Input_Path		= "/home/min/a/liu1274/TTBS/input/bh/"
Script_Path 	= "/home/min/a/liu1274/TTBS/scripts/DataCollection/"
GPU_Code_Path	= "/home/min/a/liu1274/TTBS/GPU/"

Collect = Script_Path + "collect.py"
Show = Script_Path + "show_results.py"

# Atrribute
Application = ["barneshut"]
Tree = ["octree", "kdtree"]
Lockstep = ["hybrid"]

# Make Config
Num_Of_Warps_Per_Block = 6
#Num_Of_Warps_Per_Block = [8, 16, 32]
#Num_Of_Blocks_Per_SM = [1, 2, 4, 8, 16]
Num_Of_Blocks= [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
SPLICE_DEPTH = [3, 9]
Make_Config = " TRACK_TRAVERSALS=1 SPLICE_DEPTH="

# Runtime Config
Exec_Name = "./bh "
Runtime_Flag = ""
Inputs = "BarnesHutPlummer1M.in"
Num_Points = ""
Run_Times = 15

# Clean up temp and report folder
if (os.path.isdir(Temp_Path)):
	shutil.rmtree(Temp_Path)
os.mkdir(Temp_Path)
print "Temp_Path is ready!"

#if (os.path.isdir(Time_Path)):
#	shutil.rmtree(Time_Path)
#os.mkdir(Time_Path)
#print "Time_Path is ready!"

Full_Path = ""
for app in Application:
	# We need this variable to choose different SPLICE_DEPTH for different trees
	tree_id = 0
	for tree in Tree:
		depth = SPLICE_DEPTH[tree_id]
		for lock in Lockstep:
			Full_Path = GPU_Code_Path + app + "/" + tree + "/" + lock + "/"
			os.chdir(Full_Path)
			print Full_Path
			for nblocks in Num_Of_Blocks:
				# Generate related intemediate file full name
				stats_file = Temp_Path + app + "_" + tree + "_" + lock + "_depth_" + str(depth) + "_nwarps_" + str(Num_Of_Warps_Per_Block) + "_nblocks_" + str(nblocks) + ".stats"
				time_file = Time_Path + app + "_" + tree + "_" + lock + "_depth_" + str(depth) + "_nwarps_" + str(Num_Of_Warps_Per_Block) + "_nblocks_" + str(nblocks) + ".time"
				out_file = Temp_Path + app + "_" + tree + "_" + lock + "_depth_" + str(depth) + "_nwarps_" + str(Num_Of_Warps_Per_Block) + "_nblocks_" + str(nblocks) + ".out"

				# Generate make command & Make
				make_command = " make clean; make" + Make_Config + str(depth)
				make_command = make_command + " NUM_OF_WARPS_PER_BLOCK=" + str(Num_Of_Warps_Per_Block) + " NUM_OF_BLOCKS=" + str(nblocks)
				try:
					output = subprocess.call(make_command, shell = True)
					print output
					print "Succeed in making at " + Full_Path
				except:
					print os.getcwd()
					print "Failure in making at " + Full_Path
					break

				# Generate exec command & Execute
				for run in range(0, Run_Times):
					print "Runing %dth iteration", run
					exec_command = Exec_Name + Runtime_Flag + Input_Path + Inputs + Num_Points
					record_command = " 2>&1 | " + Collect + " " + stats_file + " 2>&1 >> " + out_file
					exec_command = exec_command + record_command
					try:
						output = subprocess.call(exec_command, shell = True)
						print output
						print "Succeed in executing " + exec_command
					except:
						print "Failure in executing " + exec_command
						break

				# Generate analyze command & Analyze data
				analyze_command = Show + " " + stats_file + " 2>&1 > " + time_file
				try:
					output = subprocess.call(analyze_command, shell = True)
					print output
					print "Succeed in executing " + analyze_command
				except:
					print "Failure in executing " + analyze_command
					break

		tree_id = tree_id + 1









