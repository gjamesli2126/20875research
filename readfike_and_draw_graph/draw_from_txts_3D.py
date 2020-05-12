import os
import glob
from pprint import pprint as pp
from sys import exit
import matplotlib.pyplot as plt
import re
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
dir=os.path.join("D:\OneDrive - purdue.edu\school PU_senior\\20875 research\\20875research\\two_bintree_approximate___CUDA")
read_folder="try query points & tree_num & 1 block & N Threads"
def readfile(full_path):
    print(f"under{full_path}\n")
    lst_of_file=[]
    for each_file in glob.glob(f"{full_path}*.txt"):
        with open(each_file) as f:
             lst_of_file.append((f.readlines(),each_file))
    return lst_of_file
def plot_line(x_lsts, y_lsts,z_lsts, name, color=0, name_x="x", name_y="y",name_z="z", num=3, labels=None):
    # if labels is None:
    #     labels = ["predict", "truth"]
    fig=plt.figure()
    ax=plt.axes(projection="3d")
    print("start plotting")
    ax.set_xscale('symlog',basex=2)
    ax.set_yscale('symlog',basex=2)
    # ax.plot(x_lst,y_lst,z_lst,color=f"C{color}",linestyle='solid',linewidth=3)
    for ind,(x_lst,y_lst,z_lst) in enumerate(zip(x_lsts,y_lsts,z_lsts)):
        ax.plot3D(x_lst, y_lst,z_lst,c=f"C{color+ind}",marker='.',linestyle='solid',linewidth=1)
    print("plot Fin")
    ax.set_xlabel(name_x)
    ax.set_ylabel(name_y)
    ax.set_zlabel(name_z)



    plt.title(name)
    # plt.tight_layout()
    # plt.savefig(f"{name}.png")
    plt.savefig(read_folder)
    plt.show()
    plt.clf()
def draw_statistic(lst_of_file):
    num_trees_s = []
    num_queries_s = []
    time_takenS_s = []
    for txt_list,path in lst_of_file:
        num_trees = []
        num_queries = []
        time_takenS = []
        # print(path)
        # file_name=re.search(r"(?<=\\)[\w]+(?=\.txt)",path.split(read_folder)[-1])[0]
        # print(file_name)
        for each_line in txt_list[1:]:
            num_tree,num_query,time_taken=each_line.strip().split("|")
            num_trees.append(float(num_tree))
            num_queries.append(float(num_query))
            time_takenS.append(float(time_taken))
        num_queries_s.append(num_queries)
        num_trees_s.append(num_trees)
        time_takenS_s.append(time_takenS)
    # start drawing
    plot_line(num_trees_s, num_queries_s, time_takenS_s, name_x="num_trees", name_y="num_queries", name_z="time_taken",
              name=read_folder)
    # for num_trees,num_queries,time_takenS in zip(num_trees_s,num_queries_s,time_takenS_s):
    # print("check dimension:",end=" ")
    # if len(num_trees)==len(num_queries)==len(time_takenS):
    #     print("pass")
    #     plot_line(num_trees, num_queries, time_takenS,name_x="num_trees", name_y="num_queries",name_z="time_taken",name=file_name)
    # else:
    #     print("Diff dim")


if __name__ == "__main__":
    lst_of_file=readfile(os.path.join(dir,read_folder+"\\"))
    draw_statistic(lst_of_file)

