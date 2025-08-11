#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 19:22:21 2025

@author: alexanderpfaff
"""

import numpy as np
from typing import Set, Optional, TypeVar, Union, Sequence, Iterable, Tuple #  List, ,  Dict, Collection, , 
from tqdm import tqdm
from sklearn.model_selection import train_test_split



V = TypeVar('V', str, int) 

# Allows string input for convenience (e.g. "192837465") — parsed via _str_toSeq
SequenceLike = Union[Sequence[V], np.ndarray]




def _str_toSeq(grid_str: str) -> np.ndarray:
    """
    Converts a grid string to a sequence

    Parameters
    ----------
    grid_str : str
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    types: Set[type] = set(
        str if c.isalpha()
        else int if c.isdigit()
        else None
        for c in grid_str
    )
    
    if len(types) != 1 or None in types:
        raise ValueError("The sequence contains invalid or mixed datatypes!")
    
    datatype = types.pop()        
    grid_str = grid_str.upper()
    return np.array([datatype(c) 
                     for c in grid_str], 
                    dtype=datatype)

        

def _type_checker(test_array: np.ndarray) -> None:

    # Check for valid dtype
    dtype = test_array.dtype
    
    # if dtype == object:
    #     raise ValueError("GridCollection does not accept dtype=object")

    is_valid_char_dtype = np.issubdtype(dtype, np.str_) and dtype.itemsize == np.dtype('U1').itemsize
    is_valid_int_dtype  = np.issubdtype(dtype, np.integer)
    
    if not (is_valid_char_dtype or is_valid_int_dtype):
        raise ValueError(
            f"Invalid dtype for GridCollection: {dtype}. "
            "Expected integer type or Unicode string array with one-character elements (dtype='<U1')."
        )




def _infer_basenumber(grid_size: int) -> int:
    base = round(grid_size ** 0.25)
    if base ** 4 != grid_size:
        raise ValueError(f"Grid size {grid_size} is not a valid 4th power. Cannot infer base number.")
    return base




def _is_validgrid(gridlike: SequenceLike, 
                  basenumber: Optional[int] = None
                  ) -> bool:
    """
    Checks whether the current grid is completely filled
    and valid according to Sudoku rules: each digit from 1 to DIMENSION 
    occurs exactly once in every row, column, and box.

    Returns
    -------
    bool
        True if the grid is a valid complete Sudoku solution, else False.
    """
    if isinstance(gridlike, str):
        gridlike = _str_toSeq(gridlike)
    gridlike = np.asarray(gridlike).flatten()    

    if basenumber is None:
        basenumber = _infer_basenumber(len(gridlike))
    dimension = basenumber**2
    
    if gridlike.dtype == np.dtype('U1'):
        expected = set(chr(i) 
                       for i in range(65, 65 + dimension))
    elif np.issubdtype(gridlike.dtype, np.integer):
        expected = set(range(1, dimension + 1))
    else:
        raise ValueError(f"Invalid dtype: {gridlike.dtype}")

    gridlike = gridlike.reshape(dimension, dimension)
    # Check rows and columns
    for i in range(dimension):
        if set(gridlike[i, :]) != expected:
            return False
        if set(gridlike[:, i]) != expected:
            return False

    # Check boxes
    for r in range(0, dimension, basenumber):
        for c in range(0, dimension, basenumber):
            box = gridlike[r:r + basenumber, c:c + basenumber].flatten()
            if set(box) != expected:
                return False
    return True




def _fill_from_final_series(candidate_gc: np.ndarray, 
                            last_gc: np.ndarray, 
                            how_many: int) -> np.ndarray:
    """
    Final deduplication and padding of guest grids.

    Parameters:
    - candidate_gc: All collected grids (may include duplicates)
    - last_gc: Last generated batch (known disjoint from abc_gc)
    - how_many: Desired total number of unique grids

    Returns:
    - final_gc: Array of shape (how_many, 81), unique and non-overlapping with abc_gc
    """
    # Step 1: Deduplicate the current candidate grids
    flat_grids = [tuple(grid) for grid in tqdm(candidate_gc,
                                               desc="Checking for duplicates")]
    unique_set = dict.fromkeys(flat_grids)  # preserves order

    # Step 2: Compute how many more are needed
    missing = how_many - len(unique_set)
    if missing <= 0:
        final_gc = list(unique_set.keys())[:how_many]
        return np.array(final_gc, dtype=np.int8)

    # Step 3: Try to fill from last_gc
    flat_last = [tuple(grid) for grid in tqdm(last_gc, desc="Generating missing grids")]
    fillers = [grid for grid in flat_last if grid not in unique_set]

    # Add missing grids (up to requirement)
    for grid in tqdm(fillers[:missing], desc="Adding missing grids"):
        unique_set[grid] = None

    final_gc = list(unique_set.keys())[:how_many]

    if len(final_gc) < how_many:
        raise ValueError(f"Could only generate {len(final_gc)} unique grids out of requested {how_many}.")

    return np.array(final_gc, dtype=np.int8)





def _arrays_from_stringCollection(str_coll: Iterable[str]) -> np.ndarray:
    """
    aux-method
    Converts an iterable of grid strings to a NumPy array of shape (n, gridsize).
    Validates gridsize with _infer_basenumber.
    """
    str_list = list(str_coll)
    if not str_list:
        return np.empty((0, 0), dtype=np.uint8)  # handle empty input gracefully

    gridsize = len(str_list[0])
    base_number = _infer_basenumber(gridsize)  # assume this raises if invalid

    # Validate all strings have the same length
    if any(len(s) != gridsize for s in str_list):
        raise ValueError("Inconsistent grid string lengths in collection.")

    n = len(str_list)
    flat_array = np.fromiter(
        (int(ch) for s in str_list for ch in s),
        dtype=np.uint8,
        count=n * gridsize
    )
    return flat_array.reshape((n, gridsize))










def _split_dataset(
                   X: np.ndarray,
                   y: np.ndarray,
                   stratify_by: Optional[np.ndarray] = None,
                   train_ratio: float = 0.8,
                   seed: Optional[int] = None
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
    """
    General utility method to split datasets into training and testing sets,
    optionally stratifying by a given label or metadata array.
    
    Parameters
    ----------
    X : np.ndarray
        Dataset samples, e.g., Sudoku grids or feature arrays.
        
    y : np.ndarray
        Corresponding labels or target values.
        
    stratify_by : np.ndarray or None, optional (default=None)
        Array used for stratified sampling; must align with X and y in length.
        If None, stratification is not applied.
        
    train_ratio : float, optional (default=0.8)
        Fraction of data to use for training; remainder used for testing.
        
    seed : int or None, optional (default=None)
        Random seed for reproducibility of splits.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        `(X_train, X_test, y_train, y_test)` splits of the data.
    """                
    if seed is not None:
        np.random.seed(seed)
    
    perm_idx = np.random.permutation(len(X))
    X_ = X[perm_idx]
    y_ = y[perm_idx]
    z_ = stratify_by[perm_idx] if stratify_by is not None else None

    return train_test_split(
        X_, y_,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=z_
    )




















