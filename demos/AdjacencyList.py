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