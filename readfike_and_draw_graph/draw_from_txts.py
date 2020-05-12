import os
import glob
from pprint import pprint as pp
from sys import exit
from matplotlib import pyplot as plt
import re
import matplotlib
# matplotlib.rcParams.title
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
dir=os.path.join("D:\OneDrive - purdue.edu\school PU_senior\\20875 research\\20875research\\two_bintree_approximate___CUDA")
# read_folder="try query points & tree_num & 1 block & N Threads"
# read_folder="vary block num thread num with overlap"
# read_folder="vary block num thread num"
# read_folder="acceptable_sample_block_thread_test2"
# read_folder="reverse tree order"
read_folder="use this nonreversed"
# read_folder="use this reversed"
def readfile(full_path):
    print(f"under{full_path}\n")
    lst_of_file=[]
    for each_file in glob.glob(f"{full_path}*.txt"):
        with open(each_file) as f:
             lst_of_file.append((f.readlines(),each_file))
    return lst_of_file
def plot_line(x_lsts, y_lsts, name_S, color=0, name_x="x", name_y="y",labels=None):
    print("start plotting")
    # ax.plot(x_lst,y_lst,z_lst,color=f"C{color}",linestyle='solid',linewidth=3)
    # exit(len(x_lsts))

    fig = plt.figure(dpi=800)

    for ind,(x_lst,y_lst,name) in enumerate(zip(x_lsts,y_lsts,name_S)):
        plt.xscale('log', basex=2)
        plt.scatter(x_lst, y_lst,c=f"C{color+ind}",marker='.',linestyle='solid',linewidth=1)

        # print(x_lst,y_lst)
    # plt.plot(x_lsts, y_lsts,c="C0", marker='.', linestyle='solid', linewidth=0.4)
        # print(".", end="\n\n\n\n")

        plt.xlabel(name_x)
        plt.ylabel(name_y)
        plt.title(f"{name}{labels}",fontsize=8)
        # plt.tight_layout()
        plt.savefig(f"{name}{labels}.png")

        # plt.savefig(read_folder)
        plt.show()
        plt.clf()

    print("plot Fin")
def plot_overlap(x_lsts, y_lsts, name_S, color=0, name_x="x", name_y="y",labels=None):
    print("start plotting")
    # ax.plot(x_lst,y_lst,z_lst,color=f"C{color}",linestyle='solid',linewidth=3)
    # exit(len(x_lsts))

    fig = plt.figure(dpi=800)
    plt.xscale('log', basex=2)
    for ind,(x_lst,y_lst,name) in enumerate(zip(x_lsts,y_lsts,name_S)):
        name=re.search(r"(block_num)([\d]+)(,threads_per_block)([\d]+)",name)
        name=f"B{name.group(2)} T{name.group(4)}"

        plt.scatter(x_lst, y_lst,c=f"C{color+ind}",marker=',',linestyle='solid',linewidth=0.4,label=name)

        # print(x_lst,y_lst)
    # plt.plot(x_lsts, y_lsts,c="C0", marker='.', linestyle='solid', linewidth=0.4)
        # print(".", end="\n\n\n\n")
    plt.legend(loc='upper left',fontsize=5)
    plt.xlabel(name_x)
    plt.ylabel(name_y)

    plt.title(f"overlap{labels}",fontsize=8)
    plt.savefig(f"overlap{labels}.png")
    plt.show()
    plt.clf()
    print("plot Fin")
def plot_who(num_queries_S, time_takenS_S,num_trees_S,mode,file_name_S):
    if mode=="num_queries,time_takens":
        # if len(set(num_trees))==1:#means only one tree num
            # exit(len(num_queries))
        plot_line(num_queries_S, time_takenS_S, labels="", name_x="num_queries", name_y="time_taken(s)",
                      name_S=file_name_S)
        # else:
        #     print("num_queries,time_takens must have the same tree number.")
        #     exit(5)
    # elif mode=="num_queries,num_trees":
    #     plot_line(num_queries, num_trees, labels=time_takenS, name_x="num_queries", name_y="num_trees",
    #               name=read_folder)
    elif mode=="time_taken,num_trees":
        pass
    elif mode=="overlap":
        plot_overlap(num_queries_S, time_takenS_S, labels="", name_x="num_queries", name_y="time_taken(s)",
                      name_S=file_name_S)
    else:
        print("commend not found")
        exit(4)
def draw_statistic(lst_of_file):
    num_trees_s = []
    num_queries_s = []
    time_takenS_s = []
    file_name_S=[]
    for txt_list,path in lst_of_file:
        num_trees = []
        num_queries = []
        time_takenS = []
        # exit(len(txt_list))
        time_taken_prev=0
        print(path)
        file_name=re.search(r"(?<=\\)[\w|\,|\-|\s]+(?=\.txt)",path.split(read_folder)[-1])[0]
        print(file_name)

        # continue
        for each_line in txt_list:#org: [1:]
            num_tree,num_query,time_taken=each_line.strip().split("|")
            #if detected over ram phonemenum-> skip
            if time_taken_prev==time_taken:
                # continue
                break
            num_trees.append(float(num_tree))
            num_queries.append(float(num_query))
            time_takenS.append(float(time_taken))
            time_taken_prev=time_taken

        num_queries_s.append(num_queries)
        num_trees_s.append(num_trees)
        time_takenS_s.append(time_takenS)
        file_name_S.append(file_name)
    # start drawing
    #     exit(len(num_queries))

    # plot_who(num_queries_s, time_takenS_s,num_trees_s,"num_queries,time_takens",file_name_S)
    plot_who(num_queries_s, time_takenS_s, num_trees_s, "overlap", file_name_S)




    # for num_trees,num_queries,time_takenS in zip(num_trees_s,num_queries_s,time_takenS_s): print("check
    # dimension:",end=" ") if len(num_trees)==len(num_queries)==len(time_takenS): print("pass") plot_line(num_trees,
    # num_queries, time_takenS,name_x="num_trees", name_y="num_queries",name_z="time_taken",name=file_name) else:
    # print("Diff dim")


if __name__ == "__main__":
    lst_of_file=readfile(os.path.join(dir,read_folder+"\\"))
    print("get paths Fin")
    draw_statistic(lst_of_file)
    # clean_out_of_memory_data(lst_of_file)
