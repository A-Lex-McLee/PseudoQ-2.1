#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 20:55:23 2025

@author: alexanderpfaff
"""

import numpy as np
import random
from collections import defaultdict
from random import shuffle 
from itertools import permutations, combinations, product
from typing import  Tuple, Set, Optional
from math import sqrt

## careful!
from PsQ_Grid import *
from PsQ_GridCollection import GridCollection as GC




# TRIPLET  API --  working version; the code will be updated and documented bit by bit  



def _latinSquare_templates(triplet, full_perm=False):
    """ aux-function """
    perms = list(permutations(triplet))
    combis = list(combinations(perms, len(triplet))) 
    relevant = [c for c in combis if c[0] == tuple(triplet)]
    valid_squares = []
    for lat_square in relevant:
        np_square = np.array(lat_square) 
                
        # Check each column has exactly the right distinct values
        if all(len(np.unique(np_square[:, c])) == len(triplet) 
               for c in range(np_square.shape[1])): 
                        
            for perm in permutations(np_square): 
                if full_perm:
                    valid_squares.append(np.array(perm))
                    
                elif tuple(perm[0]) == tuple(triplet):
                    valid_squares.append(np.array(perm))
                                   
    return valid_squares


def _merge_boxcols(*ordering, 
                   idx1: bool = False, 
                   idx2: bool = False, 
                   idx3: bool = False, 
                   full_perm: bool = False,
                   abc: bool = False):
    """ aux-function """
    triplet_templates = [ 
                            [
                                ('A','B','C'),
                                ('D','E','F'),
                                ('G','H','I')
                                ],
                            [
                                ('A','C','B'),
                                ('D','F','E'),
                                ('G','I','H')
                                ]
                        ] if abc else [ 
                            [
                                (1, 2, 3),
                                (4, 5, 6),
                                (7, 8, 9)
                                ],
                            [
                                (1, 3, 2),
                                (4, 6, 5),
                                (7, 9, 8)
                                ]
                        ]
    
    boxcolumns = []
    
    #top_squares = _latinSquare_templates(triplet_templates[idx1][ordering[0]])    
    top_squares = _latinSquare_templates(triplet_templates[idx1][ordering[0]], full_perm=full_perm)     #test
    mid_squares = _latinSquare_templates(triplet_templates[idx2][ordering[1]], full_perm=True)
    bott_squares = _latinSquare_templates(triplet_templates[idx3][ordering[2]], full_perm=True)
        
    for sqA, sqB, sqC in product(top_squares, mid_squares, bott_squares):
        # Stack the three squares as an array of shape (3,3,3)
        stacked = np.stack([sqA, sqB, sqC], axis=1)  # shape (3,3,3)

        # Reshape to interleave: row1(A,B,C), row2(A,B,C), row3(A,B,C)
        boxcol = stacked.reshape(9, 3)

        boxcolumns.append(boxcol)
    return boxcolumns





def generate_3tripletCollection(idx_arr: np.ndarray = np.zeros((3,3), dtype='int8'),
                                full_perm: bool = False,
                                abc: bool = False) -> np.ndarray: 
    """
    Creates a core set of grids with n = 3 triplets. 

    Parameters
    ----------
    idx_arr : np.ndarray, optional; the default is np.zeros((3,3)). 
        The Compatibility Index; encodes the distribution of (Latin square) 
        compatibility classes per triplets and boxcolumns. 
    full_perm : bool, optional
        Performs the full permutation; unneccesary!
    abc : bool, optional; default is False.
        If True, output is returned as abc-grids.

    Returns
    -------
    TYPE: np.ndarray
        Grid collection with n = 3 and parametrized compatibility index.

    """
    
    boxcol1 = _merge_boxcols(0, 1, 2, 
                             idx1=idx_arr[0,0],
                             idx2=idx_arr[1,0],
                             idx3=idx_arr[2,0],   
                             full_perm=full_perm,
                             abc=abc)
        
    boxcol2 = _merge_boxcols(1, 2, 0, 
                             idx1=idx_arr[1,1],
                             idx2=idx_arr[2,1],
                             idx3=idx_arr[0,1],                             
                             full_perm=full_perm,
                             abc=abc)
    
    
    boxcol3 = _merge_boxcols(2, 0, 1, 
                             idx1=idx_arr[2,2],
                             idx2=idx_arr[0,2],
                             idx3=idx_arr[1,2],                             
                             full_perm=full_perm,
                             abc=abc)

    grids = []
    
    for bcT, bcM, bcB in product(boxcol1, boxcol2, boxcol3):
        stacked = np.stack([bcT, bcM, bcB], axis=1)  # shape (3,3, 3)
        boxcol = stacked.reshape(9, 9)
        
        grids.append(boxcol)
        
    return np.array(grids)





def get_uniques(arr):
    arr = np.ascontiguousarray(arr)
    flat = arr.reshape(arr.shape[0], -1)

    dtype = np.dtype((np.void, flat.dtype.itemsize * flat.shape[1]))
    view = flat.view(dtype).ravel()

    _, idx = np.unique(view, return_index=True)
    return arr[idx]


 



def permute_rows(G, rows=(0, 1, 2)):                    # 2239488  /  2239488
    """
    G: array of shape (n, 9, 9)
    rows: tuple of row indices to permute (must be length 3)
    """
    perms = np.array(list(permutations(rows)))  # shape (6, 3)

    out = []
    for p in perms:
        H = G.copy()
        H[:, rows, :] = H[:, p, :]
        out.append(H)

    return np.concatenate(out, axis=0)




def permute_boxrows(G):                          #  2239488  /  373248 ??
    """
    G: array of shape (n, 9, 9)
    rows: tuple of row indices to permute (must be length 3)
    """ 
    bc1 = range(0, 3)
    bc2 = range(3, 6)
    bc3 = range(6, 9)
    perms = np.array(list(permutations([bc1, bc2, bc3])))  # shape (6, 3)

    out = []
    for p in perms:
        p_idx = np.concatenate(p)
        
        H = G.copy()
        H[:, :, :] = H[:, p_idx, :]
        out.append(H)

    return np.concatenate(out, axis=0)








def permute_cols(G, cols=(0, 1, 2)):                 #  2239488  /  746496
    """
    G: array of shape (n, 9, 9)
    cols: tuple of column indices to permute (must be length 3)
    """
    perms = np.array(list(permutations(cols)))  # shape (6, 3)

    out = []
    for p in perms:
        H = G.copy()
        H[:, :, cols] = H[:, :, p]
        out.append(H)

    return np.concatenate(out, axis=0)



def permute_boxcols(G):                          #  2239488  /  746496
    """
    G: array of shape (n, 9, 9)
    rows: tuple of row indices to permute (must be length 3)
    """ 
    bc1 = range(0, 3)
    bc2 = range(3, 6)
    bc3 = range(6, 9)
    perms = np.array(list(permutations([bc1, bc2, bc3])))  # shape (6, 3)

    out = []
    for p in perms:
        p_idx = np.concatenate(p)
        
        H = G.copy()
        H[:, :, :] = H[:, :, p_idx]
        out.append(H)

    return np.concatenate(out, axis=0)









def generate_triplets(n: int) -> np.array: 
    triplets = np.array(list(combinations(range(1, 10), 3)))   
    n_idx = np.random.choice(triplets.shape[0], n, 
                             replace=False)   
    return triplets[n_idx]




def get_uniqueTriplets(grid: np.ndarray, basenumber=3) -> Tuple[Tuple[int]]:
    grd = grid.copy().reshape(-1,basenumber)
    trip_set: Set[int] = set()
    for g in grd:
        g = tuple(np.sort(g))
        trip_set.add(g)
    return sorted(tuple(trip_set))


def get_triplets(grid: np.ndarray, basenumber=3) -> Tuple[Tuple[int]]:
    grd = grid.copy().reshape(-1,basenumber)
    tripList = []
    for g in grd:
        g = tuple(np.sort(g))
        tripList.append(g)
    return tuple(tripList)


def _get_triplets_encoder(grid):
    triplets = get_triplets(grid)
#    abc = 'abcdefghijklmnopqrstuvwxyzMTQY'
    abc = ['T' +  (len(str(i)) % 2) * '0' + str(i) 
           for i in range(1,28) ]
    sortedEncoder = dict()
    seen = []
    idx=0
    for triplet in triplets:
        trpl = tuple(triplet)
        if not trpl in seen: 
            sortedEncoder[trpl] = abc[idx]
            idx+=1
            seen.append(trpl)
    return triplets, sortedEncoder
    
    

def make_tripGrid(grid, shape=(-1,3)):
	triplets, encoder = _get_triplets_encoder(grid) 
	tripletArray = []
	for triplet in triplets:
		tripletArray.append(encoder[triplet])
	return np.array(tripletArray).reshape(shape)
        
    


def make_tripletGraph_dict(grid):
    tripDict = _get_triplets_encoder(grid)[1]
    out = dict()
    for v, k in tripDict.items():
        out[k] = dict()
        out[k]["values"] = v 
        for v2, k2 in tripDict.items():
            if k != k2: 
                out[k][k2] = dict()
                overlap = set(v).intersection((set(v2))) 
                out[k][k2]["num_intersect"] = len(overlap)
                out[k][k2]["intersection"] = overlap        
    return out


def get_digits_inTriplets(grid):
    tripDict = _get_triplets_encoder(grid)[1]
    digitDict = dict()
    for digit in range(1,10): 
        digitDict[digit] = set(label 
                               for digits, label in tripDict.items()
                               if digit in digits)
    return digitDict


def make_tripletAdjacencyMatrix(grid):
    # Step 1: get unique triplets + labels
    triplets, encoder = _get_triplets_encoder(grid)  # triplet → label
    labels = list(encoder.values())                 # e.g. ['A','B','C',...]
    tripSet = list(encoder.keys())                  # list of triplets
    
    n = len(tripSet)
    M = np.zeros((n, n), dtype=int)

    # Step 2: fill matrix with intersection sizes
    for i in range(n):
        for j in range(n):
            if i != j:
                overlap = set(tripSet[i]).intersection(set(tripSet[j]))
                M[i, j] = len(overlap)
            # optional: put 3 on the diagonal since every triplet has size 3
            # M[i, i] = 3
    
    return M, labels, tripSet









def has_feasibleDigitCount(triplets):

    n = len(triplets)

    # initial mandatory usage
    counts = [1] * n

    # digit counts
    digit_counts = [0] * 10

    for T in triplets:
        for d in T:
            digit_counts[d] += 1

    remaining = 27 - n

    solutions = []

    def search(start, remaining):

        # prune if any digit exceeds 9
        if any(c > 9 for c in digit_counts[1:]):
            return

        # finished
        if remaining == 0:

            if all(c == 9 for c in digit_counts[1:]):

                solution = {
                    triplets[i]: counts[i]
                    for i in range(n)
                }

                solutions.append(solution)

            return

        # recursive extension
        for i in range(start, n):

            counts[i] += 1

            for d in triplets[i]:
                digit_counts[d] += 1

            search(i, remaining - 1)

            # undo
            counts[i] -= 1

            for d in triplets[i]:
                digit_counts[d] -= 1

    search(0, remaining)

    return solutions












































def to_tup(arr: np.ndarray) -> tuple:
    return tuple(tuple(t) for t in arr)







