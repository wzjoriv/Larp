# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22, 2023

@author: pruthvi
"""
from GenerateQuadTree import generateQuadTree
from GenerateQuadTree import displayQuadTree
def findLeaves(node):
    leaves = []
    def dfs(node):
        if node.isLeaf:
            leaves.append(node)
        else:
            for child in [node.topLeft, node.topRight, node.bottomLeft, node.bottomRight]:
                dfs(child)
    dfs(node)
    return leaves

def findAdjacency(leaves):
    ajList = {}
    for x in leaves:
        neighbors = []
        xtrx, xtry = x.topRight
        xtlx = x.topLeft[0]
        xtly = x.topLeft[1]
        xblx = x.bottomLeft[0]
        xbly = x.bottomLeft[1]
        xbrx = x.bottomRight[0]
        xbry = x.bottomRight[1]
        for y in leaves:
            ytrx = y.topRight[0]
            ytry = y.topRight[1]
            ytlx = y.topLeft[0]
            ytly = y.topLeft[1]
            yblx = y.bottomLeft[0]
            ybly = y.bottomLeft[1]
            ybrx = y.bottomRight[0]
            ybry = y.bottomRight[1]
            if (x.topLeft != y.topLeft) and (
                ((ytlx == xtrx+1) and (((ytly >= xbry-1) and (ytly <= xtry+1)) or ((ybly >= xbry-1) and (ybly <= xtry+1)))) or #if node is to right
                ((ybly == xtly+1) and (((ybrx >= xtlx-1) and (ybrx <= xtrx+1)) or ((yblx >= xtlx-1) and (yblx <= xtrx+1)))) or # if node is above
                ((ybrx == xtlx-1) and (((ybry >= xbly-1) and (ybry <= xtly+1)) or ((ytry >= xbly-1) and (ytry <= xtly+1)))) or # if node is to left
                ((ytry == xbly-1) and (((ytrx >= xblx-1) and (ytrx <= xbrx+1)) or ((ytlx >= xblx-1) and (ytlx <= xbrx+1)))) # if node is below
            ):
                neighbors.append(y.topLeft)
        ajList[x.topLeft] = neighbors
    return(ajList)
                    

grid1 = [[0,1],
        [1,0]]
grid2 = [[1,1,1,1,0,0,0,0],
        [1,1,1,1,0,0,0,0],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,0,0,0,0],
        [1,1,1,1,0,0,0,0],
        [1,1,1,1,0,0,0,0],
        [1,1,1,1,0,0,0,0]]

quadTree = generateQuadTree(grid2)
displayQuadTree(quadTree)
leaves = findLeaves(quadTree)
#ajList = findAdjacency(leaves)
#print(ajList)

""" 
Notes:
- make dictionary ({key}: {value}) holding information for all the leafs of the quadTree
- have the {key} be the center position of the leaf. 
  - get this position by taking average of topLeft and bottomRight positions in leaf properties.
- have the {value} be an object:
  - object.value: value of the leaf
  - object.neighbors: list of tuples for the neighbors of this leaf

"""