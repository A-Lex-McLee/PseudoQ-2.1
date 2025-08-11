#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PseudoQ_Grid  

The Syntax of Sudoku Grids, Part A: Local Infrastructure 
 
Created in Spring 2020; revised winter 2022/23, spring 2025

@author: AlexPfaff

"""

from __future__ import annotations
import numpy as np
from itertools import permutations, product  # combinations, 
from typing import Optional, TypeVar, List, Tuple,  Dict, Sequence, Union, Iterator # Collection, Set, 
from random import shuffle, seed as rnd_seed   #choice, sample
from math import factorial # sqrt
from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Callable
from tqdm import tqdm
import pandas as pd
# import sqlite3

import warnings


from utilFunX import _str_toSeq, _type_checker, _infer_basenumber, _is_validgrid
from PsQ_GridCollection import GridCollection

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

V = TypeVar('V', str, int) 

# Allows string input for convenience (e.g. "192837465") — parsed via _str_toSeq
SequenceLike = Union[Sequence[V], np.ndarray]


@dataclass
class Cell: 
    """
    Minimal unit of a grid structure; contains the actual value and 
    the grid coordinates: 
        run -- running number 
        row -- row number 
        col -- column number 
        box -- box number 
        box_row -- wrapper structure comprising rows of box size 
        box_col -- wrapper structure comprising columns of box size 
        
    """
    run: int 
    row: int
    col: int 
    box_row: int 
    box_col: int 
    box: int 
    val: V = 0
    
    
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

# Functional parameters  ==> to be passed in as arguments to certain
#                             methods of the Grid class
#                             (e.g. gridOut, )
#                        ==> they return the value of the eponymous attribute
#                             of some cell

row: int     = lambda c : c.row
col: int     = lambda c : c.col
box: int     = lambda c : c.box
box_row: int = lambda c : c.box_row
box_col: int = lambda c : c.box_col



# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

class Grid:
    """ 
        Provides a (baseNumber x baseNumber) X (baseNumber x baseNumber) Sudoku Grid,
        default setting: baseNumber = 3 ==> classical 9 X 9 Sdoku grid (recommended!);
        The constructor generates a base grid (baseGrid), which is effectively
        a coordinate system of Cell objects with the coordinates:
           -- run: running number (1 to baseNumber**4) from top-left to bottom-right,
           -- col: column number (1 to baseNumber**2) from left to right,
           -- row: row number (1 to baseNumber**2) top-down,
           -- box: box number (1 to baseNumber**2) from top-left to bottom-right,
           -- val: the actual value in a valid Sudoku grid; default setting: 0
 
           -- box_col: number of column with box-width (1 to baseNumber), left to right,
           -- box_row: number of row with box-width (1 to baseNumber), top-down.
           
           An illustration; the following grid: 
               
                *************************************
                * 2 | 6 | 4 * 3 | 8 | 9 * 5 | 1 | 7 *
                *-----------*-----------*-----------*
                * 5 | 1 | 7 * 6 | 4 | 2 * 9 | 8 | 3 *
                *-----------*-----------*-----------*
                * 3 | 8 | 9 * 7 | 5 | 1 * 4 | 6 | 2 *
                *************************************
                * 4 | 2 | 6 * 5 | 1 | 7 * 3 | 9 | 8 *
                *-----------*-----------*-----------*
                * 9 | 3 | 8 * 2 | 6 | 4 * 1 | 7 | 5 *
                *-----------*-----------*-----------*
                * 1 | 7 | 5 * 8 | 9 | 3 * 6 | 2 | 4 *
                *************************************
                * 6 | 4 | 2 * 1 | 3 | 8 * 7 | 5 | 9 *
                *-----------*-----------*-----------*
                * 7 | 5 | 3 * 9 | 2 | 6 * 8 | 4 | 1 *
                *-----------*-----------*-----------*
                * 8 | 9 | 1 * 4 | 7 | 5 * 2 | 3 | 6 *
                *************************************
                
               has a.o. the following coordinates (matrix notation): 
                   
                row 1:      [2, 6, 4, 3, 8, 9, 5, 1, 7]  
                
                col 2:      [6, 1, 8, 2, 3, 7, 4, 5, 9]T
                
                box 3:      [5, 1, 7,
                             9, 8, 3,
                             4, 6, 2]
                
                box_row 3: [[6, 4, 2, 1, 3, 8, 7, 5, 9]
                            [7, 5, 3, 9, 2, 6, 8, 4, 1]
                            [8, 9, 1, 4, 7, 5, 2, 3, 6]]
                
                col_row 2: [[3, 6, 7, 5, 2, 8, 1, 9, 4]T
                            [8, 4, 5, 1, 6, 9, 3, 2, 7]T
                            [9, 2, 1, 7, 4, 3, 8, 6, 5]T]
                    
                         
           The constructor initializes a grid with coordinates calculated from {baseNumer}
           with values set to zero. 
           The class provides a method to initialize the values via insertion 
           where the inserted object must be a sequential container object (tuple, list, string)
           containing legal value types, viz. int or str. 
           
           Further functionalities are included, such as methods for
             -- visualizing (= printing out) the grid,
             -- generating random grids, 
             -- geometric manipulation of the grid,
             -- permuting the grid coordinates,
                  notably, for generating the full permutation series of a given grid 
                  resulting in a collection of {baseNumber}!^8 * 2 grid permutations
                  (for baseNumber=3 <=> classical Sodoku grid, this means 3.359.232 grids!)
             -- alphabetizing the grid <==> de-/encoding the grid alphabetically.              
    """
    
    """ Default setting classical 9 X 9 Sudoku at the class level -> class methods"""
    _BASE_NUMBER: int = 3
    _DIMENSION: int = _BASE_NUMBER**2
    _SIZE: int = _DIMENSION**2

    def __init__(self, baseNumber: int = 3): 
        self.__BASE_NUMBER: int = baseNumber
        self.__DIMENSION: int = baseNumber**2
        self.__SIZE: int = baseNumber**4
        """ Algorithm to calculate the grid coordinates. """
        self._baseGrid: Tuple[Cell] = tuple(
            Cell(run=(self.DIMENSION)*out + self.BASE_NUMBER*mid + inn + 1,
                  row=out + 1,
                  col=self.BASE_NUMBER*mid + inn + 1,
                  box_row=out//self.BASE_NUMBER + 1,
                  box_col=mid + 1,
                  box=(out//self.BASE_NUMBER)*self.BASE_NUMBER + mid + 1) 
             for out in range(self.DIMENSION)
             for mid in range(self.BASE_NUMBER)
             for inn in range(self.BASE_NUMBER) 
            )
        self._arrayGrid = np.zeros(shape=(self.DIMENSION, self.DIMENSION), dtype=np.int8)
        

    def __str__(self) -> str:
        """Returns a human-readable summary of the current grid state."""
        return (
            f"Grid[\n"
            f"  base number     : {self.BASE_NUMBER}  \n"
            f"   -> dimension   : {self.DIMENSION}\t\t ( = {self.BASE_NUMBER} x {self.BASE_NUMBER} ) \n"
            f"   -> size        : {self.SIZE}\t\t ( = {self.DIMENSION} X {self.DIMENSION} ) \n"
            f"  is initialized  : {self.is_initialized()}\n"
            f"  is alphabetized : {self.is_alphabetized()}\n"
            f"  is valid        : {self.is_validGrid(self._arrayGrid)}\n"
            f"]"
        )

    def __repr__(self) -> str:
        return "<class 'Grid'>"

    def __eq__(self, other: Grid) -> bool:
        """Checks strict equality based on grid array contents."""
        if not isinstance(other, Grid):
            return NotImplemented
        return np.array_equal(self._arrayGrid, other._arrayGrid)

    def __hash__(self) -> int:
        return hash(tuple(self._arrayGrid.flatten()))
    
    def __getitem__(self, idx: int) -> V:
        return self.grid_toArray((self.SIZE,))[idx-1]
    
    def __iter__(self) -> Iterator:
        return iter(self.grid_toArray((self.SIZE))) 


    @property 
    def BASE_NUMBER(self):
        return self.__BASE_NUMBER

    @property 
    def DIMENSION(self):
        return self.__DIMENSION

    @property 
    def SIZE(self):
        return self.__SIZE 
    
    @property 
    def baseGrid(self):
        return self._baseGrid[:]


    def canonical_abc_key(self) -> Tuple[str]: 
        """
        Returns the abc-grid representation of this grid. The abc grid can be 
        conceived of as the key (or prototype) of a class of distributionally 
        identical grids, viz. those that are generated by relabelling / value
        substitution. 
        """
        return tuple(self.to_abc_Grid(insert=False).flatten())


    def is_abc_identical(self, other: Union[Grid, SequenceLike]) -> bool:
        """
        Checks for deep identity = distributional identity by testing whether
        both can be translated to the same abc-grid.

        Parameters
        ----------
        other : Grid or SequencLike [= str, list, tuple / np.ndarray]
            Grid object to be compared to this Grid.

        Returns
        -------
        bool
            This Grid is distributionally identical to other Grid.

        """
        if isinstance(other, Sequence) or isinstance(other, np.ndarray): 
            if isinstance(other, str): 
                other = _str_toSeq(other) 
            other = np.array(other)
            _type_checker(other)
            other = Grid.from_grid(other)
                
        if isinstance(other, Grid):
            return self.canonical_abc_key() == other.canonical_abc_key()

        else:
            raise ValueError(f"Grid object can only be compared to other Grid or grid sequence; submitted: {type(other)}")


# * * * * * * * * * * * * * *  INVENTORY  * * * * * * * * * * * * * * * * * * * 

    """ 0.    INSERT a GRID """

    def insert(self, seq: SequenceLike[V], automatic_check: bool = True) -> None:
        """
        inserts a sequence of values (integers or characters) into the internal 
        grid structure of this Grid.  
    
        Parameters
        ----------
        seq : SequenceLike[V]
            Iterable (e.g. list, tuple, string, or NumPy array of appropriate shape) 
            containing grid values. 
            Must be ordered and contain exactly `self.SIZE` elements.
        automatic_check : bool, default=True
            If True (default), the method checks whether the grid is valid before insertion.
            Set to False to skip validity checks — use with caution.
    
        Raises
        ------
        ValueError
            - If the sequence is not of length `SIZE` / sequence length is not a power of 4.
            - If the sequence has unsupported types (not int or single-character strings).
            - If `automatic_check` is True and the grid is not a valid Sudoku configuration.
    
        Notes
        -----
        - This method flattens the sequence (if needed) and reshapes it into a 2D grid.
        - Supported input types include: list, tuple, str, np.ndarray.
        - Setting `automatic_check=False` disables grid validity checking. This allows arbitrary
          data insertion, but may lead to inconsistent or unsolvable states.
    
        Examples
        --------
        >>> grid.insert([1,2,3,...,81])  # Must be exactly 81 elements
        >>> grid.insert("123...81")      # String of 81 characters
        >>> grid.insert(arr, automatic_check=False)  # Skip validity checking
            
        """
        if isinstance(seq, str):
            seq = _str_toSeq(seq)

        arr = np.asarray(seq).flatten()

        _type_checker(arr) 
        
        if automatic_check:
            if not _is_validgrid(seq):
                raise ValueError("Not a valid Sudoku Grid!")
    
        if arr.shape[0] != self.SIZE:
            raise ValueError(f"The sequence submitted does not contain the required number of cells: {self.SIZE}")

        self._quick_insert(arr)

    def _quick_insert(self, arr: np.ndarray) -> None: 
        self._arrayGrid = np.array(arr).reshape(self.DIMENSION, self.DIMENSION) 
        for i in range(self.SIZE):
            self._baseGrid[i].val = arr[i]
            


    def is_initialized(self) -> bool:
        """Return True if grid contains no zeros."""
        return not (0 in self._arrayGrid)
        

    def setZero(self) -> None:
        """
        Empties the grid by setting all values to zero
        """
#        self.insert(np.zeros(shape=(self.SIZE,)))
        self._quick_insert(np.zeros(shape=(self.SIZE,)))
        


    @classmethod
    def from_grid(cls, grid: SequenceLike) -> Grid:
        """
        Class method to instantiate a Grid object from a given sequence/array.

        Parameters
        ----------
        grid : SequenceLike
            Sequence of values instantiating a valid Sudoku grid.

        Returns
        -------
        Grid
            An instance of this class with `grid as current internal state.

        Raises
        ------
        ValueError
            If the length of `grid` is not a power of 4 (classical 81 = 9 x 9), 
            `_infer_basenumber()` will raise this exception.
    
        Notes
        -----
        Internally, this method flattens the input if it is a NumPy array, 
        infers the Sudoku base number from its length, and inserts the grid values  
        into a new `Grid` object.
        """
        if isinstance(grid, np.ndarray):
            grid = grid.flatten()
        size = len(grid)
        basenumber = _infer_basenumber(size) 
        g = Grid(baseNumber=basenumber)
        g.insert(grid)
        return g



    """ A.    VISUAL """

    def showFrame(self) -> None:
        """prints the current grid (= baseGrid) as a sequence of coordinates """
        for cell in self._baseGrid:
            outStr: str = f'run: {cell.run:2};   '  \
                + f'row: {cell.row};   '            \
                + f'col: {cell.col};   '            \
                + f'box: {cell.box};   '            \
                + f'box_col: {cell.box_col};   '    \
                + f'box_row: {cell.box_row};   '    \
                + f'value: {cell.val} '    
            print(outStr)

    def showGrid(self) -> None:      
        """prints out the current grid as Sudoku grid """
        grdLen: int = (self.DIMENSION * 2 + self.BASE_NUMBER*2 + 1)
        print()
        print("-" * grdLen)
        for i in range(self.DIMENSION):
            print("|", end=" ")
            for u in range(self.DIMENSION):
                print(self._baseGrid[self.DIMENSION*i+u].val, end=" ")
                if (self.DIMENSION*i+u) % self.BASE_NUMBER == self.BASE_NUMBER - 1: #2:
                    print("|", end=" ")
            print()
            if (i+1) % self.BASE_NUMBER == 0:
                print("-" * grdLen)




    """ B.    VALUES / Value TUPLES / ARRAYS (by COORDINATE) """
    
    def getRun(self, r: int, c: int) -> int:
        """ calculates the running number from row number r and column number c 
            @requires: 1 <= r,c <= 9
        """
        return (r-1) * self.DIMENSION + c
        
    def getCell_fromRun(self, r: int) -> Cell:
        """ returns the cell with running number r """ 
        if r < 1 or r > self.BASE_NUMBER**4:
            raise ValueError(f"Invalid running number (choose 1 - {self.BASE_NUMBER**4})")
        return deepcopy(self._baseGrid[r-1])
        
    
    def getRelativeBox_c(self, center: int) -> np.ndarray:
        """
        Returns a BASE_NUMBER x BASE_NUMBER box centered at the given 'center' cell,
        using modular wraparound (overflow) indexing on the grid.
    
        Parameters
        ----------
        center : int
            1-based running index (1 to SIZE) of the center cell.
    
        Returns
        -------
        np.ndarray
            A (BASE_NUMBER x BASE_NUMBER) array centered on the cell.
        """
        if not (1 <= center <= self.SIZE):
            raise ValueError(f"Center must be in range 1 to {self.SIZE}")
    
        center -= 1  # convert to 0-based
        row = center // self.DIMENSION
        col = center % self.DIMENSION
    
        half_box = self.BASE_NUMBER // 2
    
        # Generate centered row/col indices with wraparound
        row_indices = (row - half_box + np.arange(self.BASE_NUMBER)) % self.DIMENSION
        col_indices = (col - half_box + np.arange(self.BASE_NUMBER)) % self.DIMENSION
    
        return self._arrayGrid[np.ix_(row_indices, col_indices)]


    
    def getRelativeBox_tl(self, topLeft: int) -> np.ndarray:
        """
        Returns a BASE_NUMBER x BASE_NUMBER box starting at the 'topLeft' cell
        using modular wrapping (overflow) indexing on the grid.
        
        Parameters
        ----------
        topLeft : int
            1-based running index (1 to SIZE) of the top left cell.
        
        Returns
        -------
        np.ndarray
            A (BASE_NUMBER x BASE_NUMBER) array from the grid, wrapped.
        """
    
        if not (1 <= topLeft <= self.SIZE):
            raise ValueError(f"Center must be in range 1 to {self.SIZE}")
    
        # Convert to 0-based row, col
        topLeft -= 1
        row = topLeft // self.DIMENSION
        col = topLeft % self.DIMENSION
    
        # Get starting row/col indices (top-left corner of box)
        row_indices = (row + np.arange(self.BASE_NUMBER)) % self.DIMENSION
        col_indices = (col + np.arange(self.BASE_NUMBER)) % self.DIMENSION
    
        # Extract box using np.ix_ for 2D indexing
        return self._arrayGrid[np.ix_(row_indices, col_indices)]
    

    def gridRow(self, r: int) -> Tuple[int]:
        """ returns the values in row r as array """ 
        if r < 1 or r > self.DIMENSION:
            raise ValueError(f"Invalid row number (choose 1 - {self.DIMENSION})")
        return self._arrayGrid[r-1, :].copy() 
    

    def gridCol(self, c: int) -> Tuple[int]:
        """ returns the values in column c as array """ 
        if c < 1 or c > self.DIMENSION:
            raise ValueError(f"Invalid column number (choose 1 - {self.DIMENSION})")
        return self._arrayGrid[:, c-1].copy()
            

    def gridBox(self, b: int) -> np.ndarray:
        """ 
        Returns the values in box b as (flattened) array; 
        in order to re-box-ify: .gridBox(b).reshape(basenumber, basenumber)
        """ 
        if b < 1 or b > self.DIMENSION:
            raise ValueError(f"Invalid box number (choose 1 - {self.DIMENSION})")
        row_start: int = ((b - 1) // self.BASE_NUMBER) * self.BASE_NUMBER
        col_start: int = ((b - 1) % self.BASE_NUMBER) * self.BASE_NUMBER
        box_out: np.ndarray = self._arrayGrid[row_start:row_start+self.BASE_NUMBER, 
                                              col_start:col_start+self.BASE_NUMBER]
        return box_out.flatten()



    def getBoxCol(self, bc: int) -> np.ndarray:
        """ returns the values in box column bc as (self.DIMENSION, self.BASE_NUMBER) array """ 
        if bc < 1 or bc > self.BASE_NUMBER:
            raise ValueError(f"Invalid box column number (choose 1 - {self.BASE_NUMBER})")
        col_start: int = ((bc - 1) % self.BASE_NUMBER) * self.BASE_NUMBER
        print(col_start)
        return self._arrayGrid[:, col_start:col_start+self.BASE_NUMBER].copy()
        

    def getBoxRow(self, br: int) -> np.ndarray:
        """ returns the values in box row br as (self.BASE_NUMBER, self.DIMENSION) array """ 
        if br < 1 or br > self.BASE_NUMBER:
            raise ValueError(f"Invalid box row number (choose 1 - {self.BASE_NUMBER})")
        row_start: int = ((br - 1) % self.BASE_NUMBER) * self.BASE_NUMBER
        return self._arrayGrid[row_start:row_start+self.BASE_NUMBER, :].copy()
        
        
    def gridOut(self, coord: Callable[Cell, int] = row) -> Tuple[int]:
        """ 
        returns the <entire> current grid as a tuple; 
        legal arguments: 
            -- row (default): sorted row-wise (= by running number)
                                      ==> de facto IS the grid,
            -- col:           sorted column-wise (= top-down),
            -- box:           sorted box-wise (box-internally: row-wise)
            -- box_row:       sorted box-row-wise (box-internally: row-wise)
            -- box_col:       sorted box-column-wise (box-internally: col-wise)
        
        >> NB: if row is a valid grid,
        >>      -->  col produces a valid grid 
        >>      -->  the others do not (normally!?) produce a valid grid
        """
        out: list[int] = []
        for i in range(1, self.DIMENSION + 1):
            for cell in self._baseGrid:
                if coord(cell) == i:
                    out.append(cell.val)   
        return tuple(out)


    def grid_toArray(self, shape=(81,)) -> np.ndarray:
        """ 
        returns the  current grid as a numpy array; 

        Parameter
        ----------
        shape : Tuple[int]
            must be reshapeable in accordance with self.SIZE;
            i.e. the product of (int_1, int_2 ... int_n) must equal self.SIZE.

        """
        return self._arrayGrid.copy().reshape(shape)


    """ C.    RANDOM GRID GENERATION 
          ==> with subsequent insertion 
                (i.e. the current grid will be overwritten; 
                 for grid generation without insertion, 
                 see external function  @generate_validGrid )
    """
    
    
    def generate_rndGrid(self, 
                         seed: int = None, 
                         max_attempts: int = 10,
                         history: bool = False) -> Optional[Tuple[Tuple[int]]]:
        """
        Generate a random valid Sudoku grid and insert it into this object.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility. Default is None.
        max_attempts : int, optional
            Number of attempts to generate a valid grid if the initial attempt fails.
            Default is 10.
        history : bool
            If True, method returns the history of grid generation (positions).
    
        Returns
        -------
        None
        """
        grid, _history = generate_validGrid(base_number=self.BASE_NUMBER, 
                                            seed = seed, 
                                            max_attempts = max_attempts,
                                            history=history)
        assert self.is_validGrid(grid)
        self.insert(grid)
        if history:
            return _history
            



    """ D.    CHECK the GRID   """

    def gridCheckZero(self) -> bool:
        """
        Checks whether the current grid is still valid:
        - No digit appears more than once in any row, column, or box.
        - Zeros (empty cells) are ignored.
        Returns
        -------
        bool
            True if no violations are found; False otherwise.
        """
        # Check rows
        for i in range(self.DIMENSION):
            row = self._arrayGrid[i, :]
            nonzero = row[row > 0]
            if len(nonzero) != len(np.unique(nonzero)):
                return False
    
        # Check columns
        for j in range(self.DIMENSION):
            col = self._arrayGrid[:, j]
            nonzero = col[col > 0]
            if len(nonzero) != len(np.unique(nonzero)):
                return False
    
        # Check boxes
        for box_row in range(0, self.DIMENSION, self.BASE_NUMBER):
            for box_col in range(0, self.DIMENSION, self.BASE_NUMBER):
                box = self._arrayGrid[box_row:box_row + self.BASE_NUMBER,
                                box_col:box_col + self.BASE_NUMBER].flatten()
                nonzero = box[box > 0]
                if len(nonzero) != len(np.unique(nonzero)):
                    return False
        return True


    
    def gridCheckZero_Pos(self) -> tuple[bool, tuple[int, int] | None]:
        """
        Checks whether the current grid is valid:
        - No digit appears more than once in any row, column, or box.
        - Zeros (empty cells) are ignored.
    
        Returns
        -------
        (True, None) if valid.
        (False, (i, j)) where (i, j) indicates the row/column of the first violation:
            - For rows and columns: (row_idx, col_idx)
            - For boxes: (row_idx, col_idx) of *actual conflict*, not just top-left box
        """
        # Check rows
        for i in range(self.DIMENSION):
            row = self._arrayGrid[i, :]
            nonzero, counts = np.unique(row[row > 0], return_counts=True)
            if np.any(counts > 1):
                dup_val = nonzero[np.argmax(counts > 1)]
                j = np.where(row == dup_val)[0][0]  # First conflict column
                return False, (i, j)
    
        # Check columns
        for j in range(self.DIMENSION):
            col = self._arrayGrid[:, j]
            nonzero, counts = np.unique(col[col > 0], return_counts=True)
            if np.any(counts > 1):
                dup_val = nonzero[np.argmax(counts > 1)]
                i = np.where(col == dup_val)[0][0]  # First conflict row
                return False, (i, j)
    
        # Check boxes
        for box_row in range(0, self.DIMENSION, self.BASE_NUMBER):
            for box_col in range(0, self.DIMENSION, self.BASE_NUMBER):
                box = self._arrayGrid[box_row:box_row + self.BASE_NUMBER,
                                      box_col:box_col + self.BASE_NUMBER]
                flat = box.flatten()
                nonzero, counts = np.unique(flat[flat > 0], return_counts=True)
                if np.any(counts > 1):
                    dup_val = nonzero[np.argmax(counts > 1)]
                    # Get coordinates of first duplicated value within the box
                    rel_pos = np.argwhere(box == dup_val)[0]
                    i = box_row + rel_pos[0]
                    j = box_col + rel_pos[1]
                    return False, (i, j)
                
        return True, None

        
    






    def is_valid(self) -> bool:        
        """
        Checks whether the current grid is completely filled
        and valid according to Sudoku rules: each digit from 1 to `DIMENSION` 
        occurs exactly once in every row, column, and box.
    
        Returns
        -------
        bool
            True if the grid is a valid complete Sudoku solution, else False.
        """
        return _is_validgrid(gridlike=self._arrayGrid, basenumber=self.BASE_NUMBER)
    

    @classmethod
    def is_validGrid(cls, grid: SequenceLike[int]) -> bool:
        """
        Checks whether grid is completely filled and valid according to Sudoku rules: 
        each digit from 1 to DIMENSION occurs exactly once in every row, column, and box.
    
        Returns
        -------
        bool
            True if the grid is a valid complete Sudoku solution, else False.
        """ 
        return _is_validgrid(gridlike=grid)


       
    

    """ E.    SYMMETRY / GEOMETRY"""

    def rotate(self, k=-1) -> None:
        """ 
        rotate the current grid by 90 degrees `k` times counter-clockwise.
        Parameters
        ----------
        k : int, optional
            Number of times to rotate by 90° counter-clockwise.
            - k = -1: 90° clockwise
            - k = 1: 90° counter-clockwise
            - k = 2: 180°
            Defaults to -1 (i.e., 90° clockwise).
    
        Notes
        -----
        This method uses `np.rot90`; see NumPy documentation for advanced usage.
        The original grid is overwritten with the rotated version.                
        """
        rotation: np.ndarray = np.rot90(self._arrayGrid, k=k)        
        self.insert(rotation)


    def diaflect(self) -> None:
        """ 
            reflects the grid diagonally along top-left <=> bottom-right axis;
            (= transpose in matrix terms).
        """
        self.insert(self._arrayGrid.T)
        


    def seqOverflow(self, coord: int = row) -> Tuple[int]:
        """
            Sequential overflow;
            moves running number += 1, with final+1 to initial position
        """
        movedSeq: list[int] = list(self.gridOut(coord))
        movedSeq.insert(0, movedSeq.pop(self.SIZE-1))
        return tuple(movedSeq)
    
    
    
    
    def _getIndices(self) -> np.ndarray:
        """
        aux-method @

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        running = np.arange(1, self.SIZE + 1).reshape(self.DIMENSION, self.DIMENSION)
        half = self.BASE_NUMBER // 2
        return running[half:self.DIMENSION-half, half:self.DIMENSION-half].flatten()
        
    
    def scanGrid(self, overflow: bool = False) -> pd.DataFrame:
        """
        Scans the current grid box-wise (i.e. window size = BASE_NUMBER × BASE_NUMBER) 
        and compares the sum of the current box to the expected 'kleiner Gauss' sum.  
        You can choose whether to allow edge-overflow.
        
        Parameters
        ----------
        overflow : bool, default False
            If True, scans all 81 grid positions (with possible edge overflow).
            If False, scans only safe inner positions (without overflow).
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing columns:
            - 'running_number': position in grid (1-81)
            - 'sum': the sum of values in the box
            - 'deficit': difference between expected and actual box sum
            - 'box_window': the extracted BASE_NUMBER×BASE_NUMBER array
            - 'flat_window': flattened version of 'box_window'
        """
        legalBoxSum: int = (self.DIMENSION * (self.DIMENSION +1)) // 2
        scannedGrid: List[Tuple[int, np.ndarray, int]] = []
        running_idx = np.arange(1, self.SIZE + 1) if overflow else self._getIndices()
        
        for _run in running_idx:
            _box = self.getRelativeBox_c(_run) 
            _currentBoxSum: int = _box.sum()  
            out: Dict[int, np.ndarray, int] = {"running_number" :_run, 
                                               "sum" : _currentBoxSum, 
                                               "deficit" : legalBoxSum-_currentBoxSum,
                                               "box_window" : _box, 
                                               "flat_window" : _box.flatten() }
            scannedGrid.append(out)
        return pd.DataFrame(scannedGrid)






    """ F.    PERMUTE (parts of) the GRID """
    
    

    def _permuteGridAxis(self, grid: np.ndarray, axis: int) -> List[np.ndarray]:
        """
        aux-method
        Generalized row/column permutation preserving Sudoku constraints,
        valid for BASE_NUMBER >= 2.
        """
        b = self.BASE_NUMBER
        block_indices = [list(range(i * b, (i + 1) * b)) for i in range(b)]
    
        outer_perms = list(permutations(range(b)))         # Box orderings
        inner_perms = list(permutations(range(b)))         # Within each box
    
        # For each inner permutation combo (b of them), construct index plan
        all_inner_combos = list(product(inner_perms, repeat=b))
    
        all_grids = []
    
        for outer in outer_perms:
            box_order = [block_indices[i] for i in outer]  # permuted box groups
    
            for inner_combo in all_inner_combos:
                full_order = []
                for group, perm in zip(box_order, inner_combo):
                    full_order.extend([group[i] for i in perm])
                if axis == 0:
                    permuted = grid[full_order, :]
                else:
                    permuted = grid[:, full_order]
                all_grids.append(permuted.copy())
    
        return all_grids
        
    
    
    def permuteRows(self, grid: np.ndarray = None) -> List[np.ndarray]:
        if grid is None:
            grid = self.grid_toArray(shape=(self.DIMENSION, self.DIMENSION))
        return self._permuteGridAxis(grid=grid, axis=0)
    
    def permuteCols(self, grid: np.ndarray = None) -> List[np.ndarray]:
        if grid is None:
            grid = self.grid_toArray(shape=(self.DIMENSION, self.DIMENSION))
        return self._permuteGridAxis(grid=grid, axis=1)
    
    
    
    
    def permuteGrids(self, 
                     toCollection: bool = True, 
                     toDB: bool = False, 
                     db_name: str = None
                     ) -> GridCollection: 
        """
        Generates all column X row permutations for this grid (= .grid_toArray()).
        For classical 9 x 9 Sudoku grids, this means 1296 X 1296 = 1679616 permutations,
        generally: base_number!^(base_number+1) X base_number!^(base_number+1).
        
        Parameters:
        - toCollection: bool
        - toDB: bool
        - db_name: str
        
        Returns:
        - optionally returns a GridCollection object or np.ndarray
        """
        from PsQ_GridCollection import GridCollection

        grid = self.grid_toArray((self.DIMENSION, self.DIMENSION))
        all_permutations = [] 
        perms_per_axis: int = factorial(self.BASE_NUMBER)**(self.BASE_NUMBER+1)
        
        # start with column permutations
        column_permutations: list[np.ndarray] = self.permuteCols(grid)
        for colPerm in tqdm(column_permutations, desc=f"Generating {perms_per_axis} X {perms_per_axis} Grid Permutations"): 
            all_permutations += self.permuteRows(colPerm)
            
        out_grids: np.ndarray = np.array(all_permutations).reshape(-1, self.SIZE)
        
        if toDB:
            pass
        
        if toCollection:
            return GridCollection(out_grids)
        else:
            return out_grids



    def permuteGrids_T(self, toCollection: bool = True, 
                              toDB: bool = False, db_name: str = None) -> GridCollection: 
        """
        Produces all column X row permutations and adds reflection. For classical 9 x 9
        Sudoku: 1296 x 1296 X 2 = 3359232 grids in total
        Output can be retrieved as GridCollection object (default), an np.ndarray,
        or it can be stored in a sql database (under construction!).
        
        Parameters:
        - toCollection: bool
        - toDB: bool
        - db_name: str
        
        Returns:
        - optionally returns a GridCollection object or np.ndarray
        """
        from PsQ_GridCollection import GridCollection

        row_col_perms = self.permuteGrids(False, False) 
        row_col_perms = row_col_perms.reshape(-1, self.DIMENSION, self.DIMENSION)
        
        all_permutations: list[np.ndarray] = []

        for perm in tqdm(row_col_perms, desc="Diaflecting grid collection"):            
            all_permutations.append(perm)
            all_permutations.append(perm.copy().T)
        
        out_grids: np.ndarray = np.asarray(all_permutations).reshape(-1, self.SIZE)
        
        if toDB:
            GridCollection.save_toSQL(out_grids, db_name)
        
        if toCollection:
            return GridCollection(out_grids)
        else:
            return out_grids


    def grid_toString(self) -> str:
        """
        Returns a string representation of the current grid.
    
        The grid is first flattened (row-major order) and then converted into 
        a single concatenated string of length `SIZE` (typically 81). This works 
        for both integer-based and alphabetic (abc) grids.
    
        Returns
        -------
        str
            A string of length equal to the number of grid cells, representing 
            the grid contents in row-wise order.
        """
        grid: np.ndarray = self.grid_toArray((81,)) 
        gridString: str = "".join(map(str, grid))
        return gridString



    def _substituteValues(self, 
                         recoder: List[V],
                         insert: bool = True
                         ) -> Optional[np.ndarray]:        
        """
        Internal aux method to perform value substitution across the grid via top-row mapping.
    
        This function maps each grid value using a recoder list, where the top row 
        defines the domain and the recoder defines the image. The transformation is 
        applied element-wise using a vectorized map.
    
        Parameters
        ----------
        recoder : List[str] | List[int]
            The sequence of values used for substitution, one per symbol in the top row.
        insert : bool, default=True
            Whether to modify the internal grid in-place. If False, a new grid is returned.
    
        Returns
        -------
        Optional[np.ndarray]
            The recoded grid, if `insert=False`; otherwise, returns None.
    
        Notes
        -----
        This method assumes:
        - The grid is valid and fully populated (no zero entries).
        - The recoder has already been validated for type and uniqueness.
        - The top row of the grid contains all the unique symbols to be mapped.    
        """
        grid: np.ndarray = self.grid_toArray((self.DIMENSION, self.DIMENSION))
        top_row = grid[0].tolist()   
        
        # Create mapping from char/digit → digit (char!) <==> the actual recoder
        mapping: Dict[V, int] = dict(zip(top_row, recoder))

        vectorized_map = np.vectorize(mapping.get)
        newGrid = vectorized_map(grid)

        if insert:
            self.insert(newGrid)
        else:
            return newGrid



    def recode(self, 
               recoder: SequenceLike, 
               insert: bool = True,
               to_alpha: bool = False
               ) -> Optional[np.ndarray]:
        """
        Recode the grid using a custom symbol mapping derived from the top row.
    
        This general-purpose recoding replaces grid values based on a user-supplied 
        sequence. The top row defines the mapping domain, and the `recoder` provides 
        the image. The substitution is applied element-wise across the entire grid.
    
        Parameters
        ----------
        recoder : SequenceLike
            A sequence (list, string, or array) of new symbols (str or int) that must be
            the same length as the grid dimension and contain only unique values.
            If passed as a string (e.g. "192837465"), it is interpreted via `_str_toSeq`.
    
        insert : bool, default=True
            Whether to perform the transformation in-place. If False, the transformed 
            grid is returned.
    
        to_alpha : bool, default=False
            Required if recoding to arbitrary alphabetic characters. Prevents accidental 
            conversion to an unintended symbol grid.
    
        Returns
        -------
        Optional[np.ndarray]
            The transformed grid if `insert=False`; otherwise, returns None.
    
        Raises
        ------
        ValueError
            If the grid is invalid or the recoder fails validation (e.g., duplicates,
            wrong length, unsupported type).
    
        Notes
        -----
        This function generalizes `to_abcGrid`. It allows arbitrary recoding provided:
        - The top row defines the substitution domain
        - The recoder is one-to-one and type-consistent
        - Alphabetic recoding is explicitly opted in via `to_alpha=True`
        """        
        if not self.is_valid():
            raise ValueError("Invalid grids cannot be translated!")        
        
        if isinstance(recoder, str):
            recoder = _str_toSeq(recoder)
        recoder = list(recoder)   
        size_test = (len(recoder) == self.DIMENSION == len(set(recoder)))  
        recode_arr: np.ndarray = np.array(recoder)
        _type_checker(recode_arr)
        
        if not size_test:
            raise ValueError(f"Proper recoder requires {self.DIMENSION} distinct values of the same type!")
        
        is_str: bool = np.issubdtype(recode_arr.dtype, np.str_) or np.issubdtype(recode_arr.dtype, np.unicode_)
        if is_str and not to_alpha:
            raise ValueError("(General) Alphabetic recoding requires to_alpha=True")
        
        self._substituteValues(recoder=recoder, insert=insert)



    def to_abc_Grid(self, insert: bool = True) -> Optional[np.ndarray]:
        """
        Convert a valid Sudoku grid to an alphabetized abc-grid, issuing a warning if already alphabetized.
    
        This recoding uses a rigid mapping: values from the top-row of the original
        grid are mapped to the first {DIMENSION} characters in the alphabet -- in 
        alphabetical order. The transformation is one-to-one and intended to produce  
        a valid abc-grid. 
        Differently from a general alphabetized grid, a proper 'abc-grid' has 
        a fixed recoder, and thus a fixed top row (see below).  
    
        Parameters
        ----------
        insert : bool, default=True
            Whether to perform the transformation in-place. If False, the transformed 
            grid is returned.
    
        Returns
        -------
        Optional[np.ndarray]
            The transformed grid if `insert=False`; otherwise, returns None.
            
        Warnings
        --------
        UserWarning
            If the grid is already an abc-grid, a warning is issued and the method returns without changes.
            
        Raises
        ------
        ValueError
            If the grid is not valid, uninitialized (contains 0), or already an abc-grid.
    
        Notes
        -----
        Re-coding: top row of a given grid is mapped to a sequence of equal size:
            a_1, a_2, ... a_n => b_1, b_2, ... b_n; uniqueness must be ensured. 
            This mapping is the key for general grid substitution where every instance
            of a_1 is replace with b_1 etc.
            
        A proper abc-grid is a special recoding, it must:
            - have as a top row: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'] (classical 9 x 9)
            - be a valid Sudoku grid (row/col/box constraints).
            
        There are 6.7*10^21 / 9! distinct abc-grids (for size 9 x 9).
        """
        if not self.is_valid():
            raise ValueError("Invalid grids cannot be translated!")
        if 0 in self._arrayGrid:
            raise ValueError("Grid not initialized!")            
        if self.is_correctly_alphabetized() and insert:
            warnings.warn("ABC Grid already in shipshape! — skipping transformation.", UserWarning)
            return None
            
        recoder = [chr(c) for c in range(65, 65 + self.DIMENSION)]
        return self._substituteValues(recoder=recoder, insert=insert)

    

    def is_alphabetized(self) -> bool:
        """Return True if grid contains strings rather than integers."""
        grid = self.grid_toArray()
        return np.issubdtype(grid.dtype, np.str_) or np.issubdtype(grid.dtype, np.unicode_)


    def is_correctly_alphabetized(self) -> bool:
        """
        Check whether the grid is a proper abc-grid, which hinges on three conditions:
            (i)   the grid is alphabetized in the first place,
            (ii)  the top row corresponds to the canonical encoding: ['A', 'B', 'C', ... (up to DIMENSION)] 
                    in alphabetic order.
            (iii) the grid is valid according to Sudoku rules

        NB: return value False is ambiguous; conditions (i) and (iii) can be checked separately
            via .is_alphabetized() and .is_validGrid().

        Returns
        -------
        bool
            True if this grid is a proper abc-grid (alphabetized, correctly encoded, and valid).

        """
        if self.is_alphabetized():
            expected_key: np.ndarray = np.array(
                [chr(i) 
                 for i in range(65, 65+self.DIMENSION)], 
                dtype='U1'
                )
            top_row_matches_key = np.all(self.gridRow(1) == expected_key)

            return top_row_matches_key and self.is_valid()
        return False
        













# ****************************   special for gui   ************************************** #
# *****************************    keep for now   *************************************** #

    def _permuteCoord(self, pos: int = 1) -> Tuple[Tuple[int]]:
        """
        _aux_method@permutePos
        
        produces permutations of coordinates within the given positional 
        dimension; possible values for pos: 1 - self.BASE_NUMBER
        """
        coordinateSpan: List[int] = list(range(1,self.BASE_NUMBER + 1))
        lowestIndex: int = (pos-1) * self.BASE_NUMBER
        coordinateRange: list[int] = [lowestIndex + i
                           for i in coordinateSpan]
        return tuple(permutations(coordinateRange))

    def _permuteDiff(self, pos: int = 1):
        """
        _aux_method@permutePos
        
        calculates the differences between old and new positions, and produces
        permutations of the respective differences
        """
        out: List[Tuple[int]] = []
        per_mutations: Tuple[Tuple[int]] = self._permuteCoord(pos=pos)
        base: Tuple[int] = per_mutations[0]
        for perm in per_mutations:
            out.append(tuple(perm[i] - base[i]
                             for i in range(len(perm))))
        return out

    def permutePos(self, pos: int = 1, grdCoord: Callable[Cell, int] = col) -> Tuple[Tuple[int]]:
        """
        creates the self.BASE_NUMBER! permutations of grdCoord in the curent grid
        within the positional dimension pos


        Parameters
        ----------
        pos : int, optional
                position of the higher unit of the {grdCoord} coordinate, i.e.
                    * for row & col ==> box_row & box_col, respectively;
                        thus legal values are 1, 2 .. BASE_NUMBER        
                    * for box_row & box_col ==> grid.DIMENSION;
                        thus the only legal value is 1        
        grdCoord : Callable[Cell, int], optional
                    Functional parameter to specify the coordinate to be permuted:
                    row, col (= default), box_row, box_col. 

        Returns :  
        -------
        TYPE:  Tuple[Tuple[int]]
            contains the grid permutations for {grd} as tuples.
        """
        out = []
        per_mutations = self._permuteCoord(pos=pos)
        permute_Diffs = self._permuteDiff(pos=pos)
        base = per_mutations[0]
        for perm in permute_Diffs:
            tempGrid = list(self.gridOut()) 
            for cells in self.baseGrid:
                for i in range(self.BASE_NUMBER):
                    if grdCoord(cells) == base[i]:
                        if grdCoord == col:
                            diff = perm[i] * self.BASE_NUMBER**0
                        if grdCoord == box_col:
                            diff = perm[i] * self.BASE_NUMBER**1
                        if grdCoord == row:
                            diff = perm[i] * self.BASE_NUMBER**2
                        if grdCoord == box_row:
                            diff = perm[i] * self.BASE_NUMBER**3
                        ix = cells.run + diff
                        tempGrid[ix-1] = cells.val
            out.append(tuple(tempGrid))
        return tuple(out)

# ******************************************************************************

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


class GridGenerator:
    def __init__(self, base_number: int = 3, history: bool = False) -> None:
        self.BASE_NUMBER: int = base_number
        self.DIMENSION: int = base_number**2
        self.grid: np.ndarray = np.zeros((self.DIMENSION, self.DIMENSION), dtype=int)
        self._history: bool = history
        self._history_list: List[Tuple[int]] = [] if history else None # Type

    def generate(self) -> Tuple[np.ndarray, Optional[List[Tuple[int]]]]:
        self.grid[:] = 0
        if self._fill_grid(0, 0):
            return self.grid.copy(), self._history_list
        else:
            raise RuntimeError("Unable to generate a valid grid")

    def _fill_grid(self, row: int, col: int) -> bool:
        if row == self.DIMENSION:
            return True  # Entire grid is filled

        next_row, next_col = (row, col + 1) if col < self.DIMENSION - 1 else (row + 1, 0)

        digits = list(range(1, self.DIMENSION + 1))
        shuffle(digits)  # Randomize attempts for more grid diversity

        for num in digits:
            if self._is_valid(num, row, col):
                self.grid[row, col] = num
                if self._history:
                    self._history_list.append((row, col))

                if self._fill_grid(next_row, next_col):
                    return True
                self.grid[row, col] = 0  # Backtrack

        return False  # No valid number could be placed

    def _is_valid(self, num: int, row: int, col: int) -> bool:
        # Row & column check
        if num in self.grid[row, :]:
            return False
        if num in self.grid[:, col]:
            return False

        # Box check
        box_row = (row // self.BASE_NUMBER) * self.BASE_NUMBER
        box_col = (col // self.BASE_NUMBER) * self.BASE_NUMBER
        if num in self.grid[box_row:box_row + self.BASE_NUMBER, box_col:box_col + self.BASE_NUMBER]:
            return False

        return True


def generate_validGrid(base_number = 3, 
                       seed: int = 42, 
                       history: bool = False, 
                       max_attempts: int = 10) -> np.ndarray:
    generator = GridGenerator(base_number=base_number, history=history)
    for _ in range(max_attempts):
        try:
            if seed is not None:
                rnd_seed(seed)
            return generator.generate()
        except RuntimeError:
            seed = seed * 2 + (seed // 3) 
            continue
    raise RuntimeError(f"Failed to generate a valid grid after {max_attempts} attempts")
  


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


#     Test grids for illustration and well ... testing; you're welcome ;)  


grd1 = (7, 2, 4, 9, 1, 3, 5, 6, 8,
        5, 1, 9, 6, 8, 7, 3, 4, 2,
        3, 8, 6, 2, 5, 4, 1, 9, 7,
        2, 3, 1, 4, 7, 9, 6, 8, 5,
        4, 6, 7, 5, 3, 8, 2, 1, 9,
        8, 9, 5, 1, 6, 2, 7, 3, 4,
        1, 7, 8, 3, 4, 5, 9, 2, 6,
        9, 4, 3, 7, 2, 6, 8, 5, 1,
        6, 5, 2, 8, 9, 1, 4, 7, 3)

grd2 = (2, 7, 4, 9, 1, 3, 5, 6, 8,
        1, 5, 9, 6, 8, 7, 3, 4, 2,
        8, 3, 6, 2, 5, 4, 1, 9, 7,
        3, 2, 1, 4, 7, 9, 6, 8, 5,
        6, 4, 7, 5, 3, 8, 2, 1, 9,
        9, 8, 5, 1, 6, 2, 7, 3, 4,
        7, 1, 8, 3, 4, 5, 9, 2, 6,
        4, 9, 3, 7, 2, 6, 8, 5, 1,
        5, 6, 2, 8, 9, 1, 4, 7, 3)

grd3 = (3, 7, 8, 4, 2, 6, 5, 9, 1,
        2, 5, 4, 1, 8, 9, 6, 3, 7,
        1, 9, 6, 7, 3, 5, 4, 2, 8,
        5, 6, 9, 3, 7, 8, 2, 1, 4,
        7, 3, 2, 5, 1, 4, 8, 6, 9,
        4, 8, 1, 9, 6, 2, 7, 5, 3,
        9, 2, 5, 8, 4, 3, 1, 7, 6,
        8, 1, 3, 6, 5, 7, 9, 4, 2,
        6, 4, 7, 2, 9, 1, 3, 8, 5)

grd4 = (2, 6, 4, 3, 8, 9, 5, 1, 7,
        5, 1, 7, 6, 4, 2, 9, 8, 3,
        3, 8, 9, 7, 5, 1, 4, 6, 2,
        4, 2, 6, 5, 1, 7, 3, 9, 8,
        9, 3, 8, 2, 6, 4, 1, 7, 5,
        1, 7, 5, 8, 9, 3, 6, 2, 4,
        6, 4, 2, 1, 3, 8, 7, 5, 9,
        7, 5, 3, 9, 2, 6, 8, 4, 1,
        8, 9, 1, 4, 7, 5, 2, 3, 6)

grd5 =  (3, 1, 7, 2, 4, 6, 5, 9, 8, 
         5, 8, 6, 7, 3, 9, 2, 1, 4, 
         4, 9, 2, 1, 8, 5, 6, 7, 3, 
         9, 3, 4, 5, 2, 8, 1, 6, 7, 
         7, 2, 8, 6, 9, 1, 4, 3, 5, 
         6, 5, 1, 4, 7, 3, 9, 8, 2, 
         8, 6, 5, 3, 1, 4, 7, 2, 9, 
         1, 7, 3, 9, 5, 2, 8, 4, 6, 
         2, 4, 9, 8, 6, 7, 3, 5, 1) 






    
intro = ("\n\n\t\t====================================================\n"
         "\t\t*                                                  * \n"
         "\t\t*                  PSEUDO_Q  v2.1                  * \n"
         "\t\t*                                                  * \n"
         "\t\t*            @author: AlexPfaff (2020/21)          * \n"
         "\t\t*                                                  * \n"
         "\t\t*            The syntax of Sudoku Grids            * \n"
         "\t\t*                                                  * \n"
         "\t\t*              part 1a: Infrastructure             * \n"
         "\t\t*                   Cells, Grids,                  * \n"
         "\t\t*                 and permutations                 * \n"
         "\t\t*                  (and much more)                 * \n"
         "\t\t*                                                  * \n"
         "\t\t==================================================== ")



if __name__ == '__main__': 
    print(intro)



















##########################################


# 9**81 = 
    # 196627050475552913618075908526912116283103450944214766927315415537966391196809 (len 78)

# faculty(81) / (faculty(9)**9) = 
    # 53130688706387570345024083447116905297676780137518287775972350551392256        (len 71)



