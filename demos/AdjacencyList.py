# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22, 2023

@author: pruthvi
"""
from GenerateQuadTree import generateQuadTree
from GenerateQuadTree import displayQuadTree

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

""" 
Notes:
- make dictionary ({key}: {value}) holding information for all the leafs of the quadTree
- have the {key} be the center position of the leaf. 
  - get this position by taking average of topLeft and bottomRight positions in leaf properties.
- have the {value} be an object:
  - object.value: value of the leaf
  - object.neighbors: list of tuples for the neighbors of this leaf

"""