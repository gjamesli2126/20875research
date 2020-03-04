# Python program to do inorder traversal without recursion and
# without stack Morris inOrder Traversal
from time import sleep
# A binary tree node
class Node:

    # Constructor to create a new node
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def Kdtraverse_approx(root,target):
    loopchk=0
    current=root
    while(current.left is not None or current.right is not None):# is parent
        print("data",current.data,"__",loopchk);loopchk+=1
        if(current.data>target and current.left is not None):#got left
            current=current.left
        elif(current.data<=target and current.right is not None):
            current = current.right
        else:
            if(current.left is not None):
                current=current.left
            else:
                current = current.right

    return current.data


# Iterative function for inorder tree traversal
def MorrisTraversal(root):
    # Set current to root of binary tree
    loopchk=0
    current = root

    while (current is not None):# is parent
        # print("__",loopchk,current.left,current.data,current.right)
        print("__", loopchk)
        loopchk+=1
        # if current.left is None:
        #     print("left None ",current.data,end="")
        # if current.right is None:
        #     print("right None",current.data,end="")
        # print()

        if current.left is None:# meet child but bug in python mode it should be current.left & current.right==None
            #IF left no path -> Go dig right path
            current = current.right
        else:
            #If current got LEFT child -> pre=left child
            # Find the inorder predecessor of current

            pre = current.left
            while (pre.right is not None and pre.right != current):# value different
                #IF child's right E child-> pre=child's right
                pre = pre.right# break if current.right==None
                print(pre.right,"pre.right should be None")

            # Make current as right child of its inorder predecessor
            if (pre.right is None):

                pre.right = current
                current = current.left

            # Revert the changes made in if part to restore the
            # original tree i.e., fix the right child of predecessor
            else:
                pre.right = None
                # print(current.data)
                current = current.right

            # Driver program to test the above function


""" 
Constructed binary tree is 
		 2.5 
		/   \ 
      1.5    3.5 
	  / \   /   \ 
	1	 2 3     4
	    /         \    
	   1.7          5
	                \ 
	                 6
"""
root=Node(2.5)
root.left=Node(1.5)
root.left.left=Node(1)
root.left.right=Node(2)
root.left.right.left=Node(13)
root.right=Node(3.5)
root.right.left=Node(3)
root.right.right=Node(4)
root.right.right.right=Node(5)
root.right.right.right.right=Node(6)


print(Kdtraverse_approx(root,3.4))
# MorrisTraversal(root)

# This code is contributed by Naveen Aili
