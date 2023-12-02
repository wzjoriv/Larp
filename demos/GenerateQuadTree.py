# -*- coding: utf-8 -*-

"""
Authors: pruth, Josue N Rivera
"""
    
class Node:

    """ childn_idx = {
        "tl": 0 # Top Left
    } 
    neigh_idx = {
     "tr": 0,
     "top": 1
    }
    """

    def __init__(self, val, isLeaf, tL, tR, bL, bR):
        self.val = val
        self.isLeaf = isLeaf
        self.tL = tL
        self.tR = tR
        self.bL = bL
        self.bR = bR
        # add center location for routing purposes
        # self.depth = depth
        # self.parent = parent
        # position wrt parent
        # point to parent
        self.neighs = [None] * 8 # Order: R,tR,T,tL,L,bL,B,bR

                
class QuadTree:
    def construct(self, grid):

        def dfs(n, r, c): # n = size of grid; r = row number of topleft of grid; c = column of topLeft of grid;
            allSame = True
            # TODO: Detect values within range (Comment for Pruthvi)
            for i in range(n):
                for j in range(n):
                    if grid[r][c] != grid[r+i][c+j]:
                        allSame = False
                        break
            if allSame:
                return Node(grid[r][c], True, (r,c), (r,c+n-1), (c,r+n-1), (c+n-1,r+n-1)) # coordinates given as (x,y)
            n = n // 2
            tL = dfs(n, r, c)
            tR = dfs(n, r, c+n)
            bL = dfs(n, r+n, c)
            bR = dfs(n, r+n, c+n)

            return Node(0, False, tL, tR, bL, bR)
        return dfs(len(grid), 0, 0)
        
    

def displayQuadTree(quadTree):
    def getOutputBFS(node): # displays the quadtree in a breadth-first manner
        if not node.isLeaf:
            print(("Leaf" if node.tL.isLeaf else "Node", node.tL.val if node.tL.isLeaf else "None"))
            print(("Leaf" if node.tR.isLeaf else "Node", node.tR.val if node.tR.isLeaf else "None"))
            print(("Leaf" if node.bL.isLeaf else "Node:", node.bL.val if node.bL.isLeaf else "None"))
            print(("Leaf" if node.bR.isLeaf else "Node", node.bR.val if node.bR.isLeaf else "None"))
            getOutputBFS(node.tL)
            getOutputBFS(node.tR)
            getOutputBFS(node.bL)
            getOutputBFS(node.bR)
        else:
            return()
        
    print("\nQuadTree Breadth-first Display: (Node/Leaf, Value)")
    print(("Leaf" if quadTree.isLeaf else "Node", quadTree.val if quadTree.isLeaf else "None"))
    getOutputBFS(quadTree)
    print("")
    return()


def generateQuadTree(grid):
    solution = QuadTree()
    quadTree = solution.construct(grid)
    return quadTree

def populate_neighs(root):
    if root.isLeaf: return
    # Populate neigbors for tR child
    neighs = [None]*8
    neighs[4] = root.tL
    neighs[5] = root.bL
    neighs[6] = root.bR
    neighs[0] = (root.neighs[0] if (not root.neighs[0] or root.neighs[0].isLeaf) else root.neighs[0].tL)
    neighs[1] = (root.neighs[1] if (not root.neighs[1] or root.neighs[1].isLeaf) else root.neighs[1].bL)
    neighs[2] = (root.neighs[2] if (not root.neighs[2] or root.neighs[2].isLeaf) else root.neighs[2].bR)
    neighs[3] = (root.neighs[2] if (not root.neighs[2] or root.neighs[2].isLeaf) else root.neighs[2].bL)
    neighs[7] = (root.neighs[0] if (not root.neighs[0] or root.neighs[0].isLeaf) else root.neighs[0].bL)
    root.tR.neighs = neighs
    # Populate neighbors for tL child
    neighs = [None]*8
    neighs[0] = root.tR
    neighs[6] = root.bL
    neighs[7] = root.bR
    neighs[1] = (root.neighs[2] if (not root.neighs[2] or root.neighs[2].isLeaf) else root.neighs[2].bR)
    neighs[2] = (root.neighs[2] if (not root.neighs[2] or root.neighs[2].isLeaf) else root.neighs[2].bL)
    neighs[3] = (root.neighs[3] if (not root.neighs[3] or root.neighs[3].isLeaf) else root.neighs[3].bR)
    neighs[4] = (root.neighs[4] if (not root.neighs[4] or root.neighs[4].isLeaf) else root.neighs[4].tR)
    neighs[5] = (root.neighs[4] if (not root.neighs[4] or root.neighs[4].isLeaf) else root.neighs[4].bR)
    root.tL.neighs = neighs
    # Populate neighbors for bL child
    neighs = [None]*8
    neighs[0] = root.bR
    neighs[1] = root.tR
    neighs[2] = root.tL
    neighs[3] = (root.neighs[4] if (not root.neighs[4] or root.neighs[4].isLeaf) else root.neighs[4].tR)
    neighs[4] = (root.neighs[4] if (not root.neighs[4] or root.neighs[4].isLeaf) else root.neighs[4].bR)
    neighs[5] = (root.neighs[5] if (not root.neighs[5] or root.neighs[5].isLeaf) else root.neighs[5].tR)
    neighs[6] = (root.neighs[6] if (not root.neighs[6] or root.neighs[6].isLeaf) else root.neighs[6].tL)
    neighs[7] = (root.neighs[6] if (not root.neighs[6] or root.neighs[6].isLeaf) else root.neighs[6].tR)
    root.bL.neighs = neighs
    # Populate neighbors for bR child
    neighs = [None]*8
    neighs[2] = root.tR
    neighs[3] = root.tL
    neighs[4] = root.bL
    neighs[0] = (root.neighs[0] if (not root.neighs[0] or root.neighs[0].isLeaf) else root.neighs[0].bL)
    neighs[1] = (root.neighs[0] if (not root.neighs[0] or root.neighs[0].isLeaf) else root.neighs[0].tL)
    neighs[5] = (root.neighs[6] if (not root.neighs[6] or root.neighs[6].isLeaf) else root.neighs[6].tL)
    neighs[6] = (root.neighs[6] if (not root.neighs[6] or root.neighs[6].isLeaf) else root.neighs[6].tR)
    neighs[7] = (root.neighs[7] if (not root.neighs[7] or root.neighs[7].isLeaf) else root.neighs[7].tL)
    root.bR.neighs = neighs
    # Recursive call to children
    populate_neighs(root.tL)
    populate_neighs(root.tR)
    populate_neighs(root.bL)
    populate_neighs(root.bR)
    """
    - Currently, this is nearly complete. We just need to add post-processing to simplify
      all leaves that hold other nodes as their neighbors rather than other leaves. 
    - We should first make a list of all the leaves.
    - Then, going through all these leaves, we can check all their neighrbos and if those 
      neighors are not leaves, we must process then accordingly to find the leaves on the correct side.
    """

def findLeaves(node):
    leaves = []
    def dfs(node):
        if node.isLeaf:
            leaves.append(node)
        else:
            for child in [node.tL, node.tR, node.bL, node.bR]:
                dfs(child)
    dfs(node)
    return leaves

def makeAdjacency(leaves):
    # applies the extend() function to all leaf nodes to make their adjacency lists
    for leaf in leaves:
        neighs = set() # set for neighbors
        for i in range(8):
            dfs_neighs_leaf_extend(neighs, leaf.neighs[i], i)
        leaf.neighs = neighs
    return

def dfs_neighs_leaf_extend(neighs, node, pos):
    # This function expands the neighbor information for a node to include all its neighbors that are leaves
    if node and node.isLeaf: 
        neighs.add(node)
        return

    directions = [
        (node.tL, node.bL),  # right
        (node.bL,),          # topRight
        (node.bL, node.bR),  # top
        (node.bR,),          # topLeft
        (node.bR, node.tR),  # left
        (node.tR,),          # bottomLeft
        (node.tL, node.tR),  # bottom
        (node.tL,)           # bottomRight
    ]

    for next_node in directions[pos]:
        dfs_neighs_leaf_extend(neighs, next_node, pos)

if __name__ == "__main__":
    grid2 =[[1,1,1,1,0,0,0,0],
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1],
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,0,0,0,0]]
    root = generateQuadTree(grid2)
    populate_neighs(root)
    leaves = findLeaves(root)
    makeAdjacency(leaves)
    """x = root.tR.tR.neighs
    for z in x:
        print(z.tL)
    """
    print("Hello, World!")


"""
1:)
- generate the quadtree
- add the topright row/col and bottomLeft row/col
2:)
- seperate leaf nodes and find a way to connect them in a new graph based on physical adjacency
3:)
- determine quad stopping threshold based on potential field
"""


