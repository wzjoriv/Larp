# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 01:19:49 2023

@author: pruth
"""
def generateQuadTree(grid):
    
    class Node:
        def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
            self.val = val
            self.isLeaf = isLeaf
            self.topLeft = topLeft
            self.topRight = topRight
            self.bottomLeft = bottomLeft
            self.bottomRight = bottomRight
            #self.depth = depth
            #self.parent = parent
            # position wrt parent
            # point to parent
                   
    class QuadTree:
        def construct(self, grid):
            def dfs(n, r, c): # n = size of grid; r = row number of topleft of grid; c = column of topLeft of grid;
                allSame = True
                for i in range(n):
                    for j in range(n):
                        if grid[r][c] != grid[r+i][c+j]:
                            allSame = False
                            break
                if allSame:
                    return Node(grid[r][c], True, (r,c), (r,c+n-1), (r+n-1,c), (r+n-1,c+n-1))
                n = n // 2
                topLeft = dfs(n, r, c)
                topRight = dfs(n, r, c+n)
                bottomLeft = dfs(n, r+n, c)
                bottomRight = dfs(n, r+n, c+n)
                return Node(0, False, topLeft, topRight, bottomLeft, bottomRight)
            return dfs(len(grid), 0, 0)
        
    solution = QuadTree()
    quadTree = solution.construct(grid)
    return(quadTree)

def displayQuadTree(quadTree):
    def getOutputBFS(node): # displays the quadtree in a breadth-first manner
        if not node.isLeaf:
            print(("Leaf" if node.topLeft.isLeaf else "Node", node.topLeft.val if node.topLeft.isLeaf else "None"))
            print(("Leaf" if node.topRight.isLeaf else "Node", node.topRight.val if node.topRight.isLeaf else "None"))
            print(("Leaf" if node.bottomLeft.isLeaf else "Node:", node.bottomLeft.val if node.bottomLeft.isLeaf else "None"))
            print(("Leaf" if node.bottomRight.isLeaf else "Node", node.bottomRight.val if node.bottomRight.isLeaf else "None"))
            getOutputBFS(node.topLeft)
            getOutputBFS(node.topRight)
            getOutputBFS(node.bottomLeft)
            getOutputBFS(node.bottomRight)
        else:
            return()
        
    print("\nQuadTree Breadth-first Display: (Node/Leaf, Value)")
    print(("Leaf" if quadTree.isLeaf else "Node", quadTree.val if quadTree.isLeaf else "None"))
    getOutputBFS(quadTree)
    print("")
    return()




"""
1:)
- generate the quadtree
- add the topright row/col and bottomLeft row/col
2:)
- seperate leaf nodes and find a way to connect them in a new graph based on physical adjacency
3:)
- determine quad stopping threshold based on potential field
"""





# idea: add depth level to node information