from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from pprint import pprint as pp
import re
from matplotlib.pyplot import figure

from sys import exit
def plot_2D(X,Y,Z):
    if (len(X)==len(Y)==len(Z))==False:
        print(len(X),len(Y),len(Z))
        exit(2)
    plt.figure(figsize=(120, 16),dpi=128)
    plt.plot(X,Z)
    plt.title("block_number----speed")
    plt.xlabel('block_number')
    plt.ylabel('speed')
    plt.savefig("2D_blocknumber_speed.png")
    plt.close()
    plt.clf()

    plt.figure(figsize=(120, 16),dpi=128)
    plt.plot(Y, Z)
    plt.title("blocksize----speed")
    plt.xlabel('blocksize')
    plt.ylabel('speed')
    plt.savefig("2D_blocksize_speed.png")
    plt.close()
    plt.clf()




def plot_3D(X,Y,Z):
    if (len(X)==len(Y)==len(Z))==False:
        print(len(X),len(Y),len(Z))
        exit(3)
    # X=np.array([X])
    # Y= np.array([Y])
    # Z= np.array([Z])
    print(len(X),len(Y),len(Z))
    #plot

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c='r', marker='.')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.savefig("3D.png")
    plt.clf()

def read_file_parse(fname):
    with open(fname,"r") as f:
        all=f.read()

    found=re.findall(r"Block number:([\d.\s]+)Block Size:([\d\s.]+)cpu:([\d\s.]+)gpu:([\d\s.]+)speed :([\d\s.]+)r/ms",all)
    # parse:
    Block_number=[]#0
    Block_size=[]#1
    speed=[]#group4
    for each in found:
        Block_number.append(float(each[0].strip()))
        Block_size.append(float(each[1].strip()))
        speed.append(float(each[4].strip()))

    return Block_number,Block_size,speed


if __name__=="__main__":

    X,Y,Z=read_file_parse("try_thread_block.txt")
    plot_3D(X,Y,Z)
    plot_2D(X,Y,Z)