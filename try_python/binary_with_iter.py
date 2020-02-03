class Node:
    def __init__(self):
        self.key = 0
        self.left = None
        self.right = None


# Utility function to create new Node
def newNode(key: int) -> Node:
    temp = Node()
    temp.key = key
    temp.left = None
    temp.right = None
    return temp


# Utility function to perform
# inorder traversal of the tree
def inorder(root: Node):
    if root is not None:
        inorder(root.left)
        print(root.key, end=" ")
        inorder(root.right)

    # Function to construct a Binary Tree from parent array


def createTree(parent: list, n: int) -> Node:
    # A map to keep track of all the nodes created.
    # Key: node value; Value: Pointer to that Node
    m = dict()
    root = Node()

    # Iterate for all elements of the parent array.
    for i in range(n):

        # Node i does not exist in the map
        if i not in m:
            # Create a new node for the current index
            temp = newNode(i)

            # Entry of the node in the map with
            # key as i and value as temp
            m[i] = temp

            # If parent is -1
        # Current node i is the root
        # So mark it as the root of the tree
        if parent[i] == -1:
            root = m[i]

            # Current node is not root and parent
        # of that node is not created yet
        elif parent[i] not in m:

            # Create the parent
            temp = newNode(parent[i])

            # Assign the node as the
            # left child of the parent
            temp.left = m[i]

            # Entry of parent in map
            m[parent[i]] = temp

            # Current node is not root and parent
        # of that node is already created
        else:

            # Left child of the parent doesn't exist
            if m[parent[i]].left is None:
                m[parent[i]].left = m[i]

                # Right child of the parent doesn't exist
            else:
                m[parent[i]].right = m[i]
    return root


# Driver Code
if __name__ == "__main__":
    parent = [-1, 0, 0, 1, 1, 3, 5]
    n = len(parent)
    root = createTree(parent, n)
    print("Inorder Traversal of constructed tree")
    inorder(root)
