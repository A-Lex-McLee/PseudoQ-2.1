#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PseudoQ_GridCollection 

The Syntax of Sudoku Grids, Part B: Larger Infrastructure  
 
Created in January 2023, revised 2024-25

@author: AlexPfaff

"""

from __future__ import annotations
from itertools import permutations 
from typing import Union, Tuple, TypeVar, Sequence, Optional, Dict, Iterator, Iterable # List , Set, Collection
from math import factorial 
from collections import defaultdict
import numpy as np
from numpy import dtype as np_dtype
from tqdm import tqdm
import sqlite3

from utilFunX import (_str_toSeq, _type_checker, _infer_basenumber, _fill_from_final_series, 
                      _arrays_from_stringCollection, _split_dataset)


V = TypeVar('V', str, int) 
SequenceLike = Union[Sequence[V], np.ndarray]




class GridCollection:
    """
    A container for managing and manipulating collections of Sudoku grids for ML and DL pipelines.

    This class stores an internal grid collection (`self.collection`) as a flat (n, N^4) NumPy array,
    where N is the Sudoku base (e.g., 3 for 9x9 grids). It provides powerful tools for:
    
    - **Permutation-based series generation**, including:
        - Vertical series: structure-preserving row/column/box permutations (e.g., 3!^8 = 1,679,616 variants for standard 9x9 Sudoku)
        - Horizontal series: label permutations or digit recodings (9! = 362,880 possible recodings)
        - Deflection: transposition of currently active series (swap rows ↔ columns)
        - Additional augmentation mechanisms to be defined

    - **Export-ready garbage series**: invalid grids for supervised training (e.g., off-by-X violations, wrong cardinality, etc.)

    - **Multiple internal series**:
        - `activeSeries`: current working permutation series
        - `garbageCollection`: curated invalid or adversarial grids
        - `abcCollection`: alphabetic version of the grids for human readability
        - `one-hot`: status flag for downstream model compatibility

    - **Factory methods**:
        - `.from_scratch()`: initialize from generated or hand-defined arrays
        - `.from_sql()`: import a grid collection from SQL backends

    - **Persistence**: designed for seamless storage/export/import to and from SQL databases.

    This class enables high-throughput experimentation with data augmentation, error injection, and classification pipelines,
    using Sudoku as a structured domain with rich internal symmetries. 

    This class exposes selected methods of the internal NumPy array (.collection) for 
    convenience. When available, these methods match the behavior of their NumPy counterparts.
    
    """
            
        
    def __init__(self, collection: np.ndarray) -> None:
        """
        Initialize a GridCollection instance with a NumPy-based grid collection array.

        Parameters:
        ----------
        collection : np.ndarray
            A NumPy array of shape (n, N^2, N^2) or flat shape (n, N^4), where N is the Sudoku base number (e.g., 3 for 9x9 grids).
            The array must be rectangular and convertible to (n, N^2, N^2).

        Raises:
        ------
        ValueError
            If the provided array cannot be reshaped into a valid Sudoku grid collection.
        """
        _type_checker(collection)
        _total: int = collection.shape[0]
        _gridsize: int = round(collection.size / _total)        
        _base_number = _infer_basenumber(_gridsize)        
        try:
            collection.reshape(_total, _base_number**2, _base_number**2)
            if np.issubdtype(collection.dtype, np.integer):
                _dtype = np.uint8
            else:
                _dtype = '<U1'
        except ValueError:
            raise ValueError(f"Cannot reshape input of size {collection.size} into ({_total}, {_base_number**2}, {_base_number**2})")

        self.__BASE_NUMBER:         int         = _base_number
        self.__DIMENSION:           int         = _base_number**2
        self.__SIZE_grid:           int         = _base_number**4
        
        self.__collection:          np.ndarray  = collection.astype(_dtype).reshape(_total, 
                                                                                    self.__SIZE_grid)  
        
        self.clear_all()
            

    def __repr__(self) -> str:
        return("<class 'GridCollection'>")

    def __str__(self) -> str:
        """Returns a detailed summary of the current grid collection state."""
        max_key_len = max(len(k) for k in self.getGarbageStatus)

        garbage_info = "\n".join(f"\t\t\t\t\t {k:<{max_key_len}} : {v}" for k, v in self.getGarbageStatus.items())
        return (
            f"GridCollection[ \n" 
            f"                 internal shape: {self.shape}  \n " 
            f"                active series: {self.__prefix + self.activationStatus} \n"
            f"\t                size: {self.SIZE_activeSeries}  \n" 
            f"                 labeled series: {self.classes} {self.__distributor}  \n" 
            f"                 labels: {self.SIZE_labels != 0}  \n" 
            f"\t               size: {self.SIZE_labels}  \n" 
            f"                 garbage: {bool(self.SIZE_garbageCollection)} \n" 
            f"\t               total size: {self.SIZE_garbageCollection} \n"
            f"{garbage_info} \n"
            f"                 one_hot encoding: {self.oneHotStatus}  \n"
            f"                 internal abc collection: {self.abcStatus}  \n"
            f"               ] "
            )


    def __iter__(self) -> Iterator[np.ndarray]: 
        """
        Yields one grid at a time from this grid collection.
        
        Convenience implementation, allows:
            for grid in gc: ... instead of: for grid in gc.collection:
        """
        return iter(self.collection)
    
    def __getitem__(self, index) -> np.ndarray:
        """
        Returns the `index`-th grid from the underlying collection.
        
        Convenience implementation, allows:
            gc[12345]  instead of: gc.collection[12345]
        """
        return self.__collection[index]
    

    def __eq__(self, other: object) -> bool:
        """
        Compares two GridCollection objects for equality, ignoring internal ordering.
        
        Returns True if both collections contain the same set of grids, regardless of order.
        Each grid is flattened to a 1D array, and the sets of grids are compared
        after lexicographic sorting.
        
        Parameters
        ----------
        other : object
            Another GridCollection or ndarray to compare against.
        
        Returns
        -------
        bool
        """     
        try:
            A_flat = self.collection.reshape(-1, self.SIZE_grid) 
            if isinstance(other, GridCollection):                             
                B_flat = other.collection.reshape(-1, self.SIZE_grid)
            elif isinstance(other, np.ndarray):                
                B_flat = other.reshape(-1, self.SIZE_grid)
            else:
                raise TypeError()
            if A_flat.shape != B_flat.shape:
                return False  # early exit if shape mismatch
    
            # Lexicographically sort rows
            A_sorted = A_flat[np.lexsort(A_flat.T[::-1])]
            B_sorted = B_flat[np.lexsort(B_flat.T[::-1])]
    
            return np.array_equal(A_sorted, B_sorted)
    
        except (AttributeError, TypeError, ValueError):
            return False
    



##########################################################################################
##                                                                                      ##
##         B A S I C  F U N C T I O N A L I T I E S  &  P R O P E R T I E S             ##     
##                                                                                      ##
##########################################################################################

    def getInfo(self) -> None:
        """
        Get information about the current state of this GridCollection:
            activation status, garbage status, labels, classes, counts.
        """
        print(self) 
        
    @property 
    def dtype(self) -> np_dtype:
        """ Returns the dtype of the underlying collection.  """
        return self.collection.dtype
        
    
    @property 
    def BASE_NUMBER(self) -> int:
        """Returns the base size N of the grid (e.g. 3 for 9x9 Sudoku)."""
        return self.__BASE_NUMBER
    
    @property 
    def DIMENSION(self) -> int:
        """Returns the grid dimension N^2 (e.g. 9 for standard Sudoku)."""
        return self.__DIMENSION

    @property 
    def SIZE_grid(self) -> int:
        """Returns the flattened size of a single grid: N^4 (e.g. 81 for 9x9 Sudoku)."""
        return self.__SIZE_grid


    @property 
    def SIZE_collection(self) -> int:
        """Returns the number of grids in the collection."""
        return self.__collection.shape[0]
    
    @property 
    def SIZE_activeSeries(self) -> int:
        """Returns the number of grids in the currently active series."""
        return self.__activeSeries.shape[0]
    
    @property 
    def SIZE_garbageCollection(self) -> int:
        """Returns the number of grids in the garbage (invalid grid) collection."""
        return self.__garbageCollection.shape[0] 
    
    @property 
    def SIZE_labels(self) -> int:
        """
        Returns the number labels in the label collection; if non-zero, it should
        be equal to SIZE_activeSeries. """
        return self.__labels.shape[0] 
        
    
    @property 
    def classes(self) -> int:
        """Returns the number of unique labels in the multi-class series."""
        return self.__classes 

    @property 
    def labels(self) -> np.ndarray:
        """
        Returns a copy of the current label array associated with active grids --
        will be non-zero if 'multiclass' or 'puzzle' series is activated.
        """
        return self.__labels.copy() 

    @property 
    def activationStatus(self) -> str:
        """Returns the activation status of the grid series (e.g. 'not_activated', 'vertical', etc.)."""
        return self.__activationStatus


    @property
    def shape(self):
        """
        Returns the shape of the underlying internal grid collection array. 
        NB: collection can be reshaped via self.reshape()
        """
        return self.__collection.shape

    @property
    def collection(self):
        """Returns a copy of the internal grid collection array (n, SIZE_grid)."""
        return self.__collection.copy()

    @property 
    def activeSeries(self):
        """Returns a copy of the current active grid series (permuted/augmented grids)."""
        return self.__activeSeries.copy()

    @property 
    def garbageCollection(self):
        """Returns a copy of the current garbage (invalid) grid series."""
        return self.__garbageCollection.copy()
    
    @property
    def getGarbageStatus(self) -> Dict[str, int]:
        """Returns a summary dictionary describing the types and counts of invalid grids currently stored."""
        return self.__garbageStatus

    @property
    def abcStatus(self) -> bool:
        """Returns True  if this object currently hosts a collection of alphabetized
        (= abc) grids based on the internal grid collection. """
        return self.__abcStatus
    
    @property
    def prefix(self) -> str:
        """Returns the prefix tag used to modify the active grid series."""
        return self.__prefix
    
    @property 
    def oneHotStatus(self) -> bool:
        """Returns True if the current active grids are in one-hot format; otherwise False."""
        return self.__oneHotStatus

    @property 
    def abcCollection(self) -> np.ndarray:
        """Returns the current alphabetic ('A'-'I') representation of the grid collection."""
        return self.__abcSeries.copy()


    def clear_all(self) -> None:
        """ 
        Clears a mutable attributes by setting:
                - collectionst to np.empty()
                - string attribute to "not_activated" / ""
                - boolean attributes to False
                - value attributes to 0
        """
        self.clear_activeSeries()
        self.clear_garbage()
        self.clear_muliClassSeries()
        self.clear_abcSeries()
        self.clear_oneHot()
        

    def clear_activeSeries(self): 
        """ Clears the activeSeries collection. """
        self.__activeSeries: np.ndarray  = np.empty((0, self.SIZE_grid), dtype=np.uint8) 
        self.__activationStatus: str = "not_activated"
        self.__prefix: str = ""

    def clear_garbage(self):
        """ Clears the garbage collection. """
        self.__garbageCollection: np.ndarray  = np.empty((0, self.SIZE_grid), dtype=np.uint8) 
        self.__garbageStatus = defaultdict(int, {
            "guest_grids"               : 0,
            "false_fromCurrent_seq"     : 0,
            "false_fromCurrent_switch"  : 0,
            "false_cardinality"         : 0,
            "false_off_by_X"            : 0,
            "false_arbitrary"           : 0
        })

    def clear_muliClassSeries(self):
        """ Clears the muliClassSeries collection. """
        self.__classes: int = 0
        self.clear_labels()
        self.clear_activeSeries() 
        
    def clear_labels(self) -> None:
        """ Clears the labels collection. """
        self.__labels: np.ndarray = np.empty((0,), dtype=np.uint8)    
        self.__distributor: str = ""
        

    def clear_abcSeries(self): 
        """ Clears the abcSeries collection. """
        self.__abcSeries: np.ndarray  = np.empty((0, self.SIZE_grid), dtype='<U1') 
        self.__abcStatus: str = False

    def clear_oneHot(self) -> None:
        """ Sets oneHot status to False. """
        self.__oneHotStatus: bool = False



    def reshape(self, *shape: int, inplace: bool = False) -> Union[None, np.ndarray]: 
        """
        Reshape the internal grid collection.
    
        This method mirrors `np.ndarray.reshape()`, allowing access to reshaped versions
        of the internal collection. If `inplace=True`, the internal state is updated; 
        otherwise, a reshaped copy is returned.
    
        Parameters
        ----------
        *shape : int
            The new shape to give to the collection. Must be compatible with the total number of elements.
        inplace : bool, default False
            If True, modifies the internal array in-place. If False, returns a reshaped copy.
    
        Returns
        -------
        np.ndarray or None
            The reshaped array if `inplace=False`; otherwise, None.
    
        Notes
        -----
        This is a convenience wrapper around the internal NumPy array's `reshape()` method.
    
        Examples
        --------
        >>> gc = GridCollection(...)
        >>> reshaped = gc.reshape(1000, 81)  # returns new array
        >>> gc.reshape(1000, 81, inplace=True)  # modifies internal state
        """
        if inplace:
            self.__collection = self.__collection.reshape(*shape)
        else:
            return self.__collection.reshape(*shape).copy()
            
        



##########################################################################################
##                                                                                      ##
##            V A L I D I T Y   &   C O N T A I N M E N T   C H E C K S                 ##     
##                                                                                      ##
##########################################################################################


    def is_valid(self) -> bool:
        """ Returns True if all grids in the internal collection are valid Sudoku grids. """
        valid, _, _ = self._is_valid_grid_collection(self.collection)
        return valid 

    @classmethod
    def is_valid_CLS(cls, grids: np.ndarray) -> bool:
        """ Returns True if all grids in the array argument are valid Sudoku grids. """
        valid, _, _ = cls._is_valid_grid_collection(grids)
        return valid 

    
    @staticmethod
    def _is_valid_grid_collection(grids: np.ndarray):
        """ @aux_method:
        Vectorized validator for a collection of 9x9 Sudoku grids.
    
        Parameters
        ----------
        grids : np.ndarray of shape (n, 9, 9)
            Collection of Sudoku grids to validate.
    
        Returns
        -------
        is_all_valid : bool
            True if all grids are valid.
        validity_mask : np.ndarray of shape (n,)
            Boolean mask indicating which grids are valid.
        invalid_indices : np.ndarray
            Indices of invalid grids.
        """
        basenumber: int = _infer_basenumber(grids[0].flatten().shape[0])
        dimension = basenumber**2
            
        try:
            grids = grids.reshape(-1, dimension, dimension)
        except Exception:
            raise ValueError
    
        n, h, w = grids.shape
        assert h == dimension and w == dimension, f"Grids must be {dimension} x {dimension}"
    
        # STEP 1: Determine expected values
        if grids.dtype == np.dtype('U1'):
            expected = np.array([chr(i) 
                                 for i in range(65, dimension + 65)])
        elif np.issubdtype(grids.dtype, np.integer):
            expected = np.arange(1, dimension + 1) 
        else:
            raise ValueError(f"Invalid dtype: {grids.dtype}")    
    
        expected_sorted = np.sort(expected)
    
        # STEP 2: Define a helper function to check 9x9 grids row-wise
        def is_unit_valid(units: np.ndarray):
            """Check if each unit (row/col/box) contains all values in `expected`"""
            return np.all(np.sort(units, axis=-1) == expected_sorted, axis=-1)
    
        # --- Row validity ---
        rows_valid = is_unit_valid(grids)  # shape (n, 9)
    
        # --- Column validity ---
        cols_valid = is_unit_valid(np.transpose(grids, (0, 2, 1)))  # shape (n, 9)
    
        # --- Box validity --- insanity!
        # Split 9x9 into 9 boxes of 3x3 using reshape and transpose
        boxes = grids.reshape(n, 
                              basenumber, 
                              basenumber, 
                              basenumber, 
                              basenumber
                              ).transpose(0, 1, 3, 2, 4).reshape(n, 
                                                                 dimension, 
                                                                 dimension)
        boxes_valid = is_unit_valid(boxes.reshape(n, dimension, dimension)) #.reshape(n, 9, 9))  # shape (n, 9)
    
        # Combine all validity checks per grid
        grid_valid_mask = (
            np.all(rows_valid, axis=1) &
            np.all(cols_valid, axis=1) &
            np.all(boxes_valid, axis=1)
        )
    
        # Results
        all_valid = np.all(grid_valid_mask)
        invalid_indices = np.flatnonzero(~grid_valid_mask)
    
        return all_valid, grid_valid_mask, invalid_indices
    


    def invalid_grids_idx(self) -> np.ndarray: 
        """ Returns the indices of invalid Sudoku grids contained in the internal collection. """
        grids = self.collection
        _, _, idx = self._is_valid_grid_collection(grids)
        return idx 

    @classmethod
    def invalid_grids_idx_CLS(cls, grids: np.ndarray) -> bool:
        """ Returns the indices of invalid Sudoku grids contained in array argument. """
        _, _, idx = cls._is_valid_grid_collection(grids)
        return idx 




    
    def is_abc_contained(self, grid: SequenceLike) -> bool:
        """
        Probes for deep containment:
        Checks if the given grid is contained in this collection by deep abc-identity.
    
        Parameters
        ----------
        grid : SequenceLike
            The target grid to find.
    
        Returns
        -------
        bool 
            True if the abc_key of grid is found in the abc_collection.
        """
        if self.abcStatus:
            contained, _ = GridCollection.is_abc_contained_CLS(grid=grid,
                                                               gc=self.abcCollection)
        else:
            contained, _ = GridCollection.is_abc_contained_CLS(grid=grid,
                                                               gc=self.to_abc_collection())
        return contained


    
    def abc_location(self, grid: SequenceLike) -> int:
        """
        Checks if the `grid` argument is abc_contained in this collection, and returns
        the index position in the internal collection if found, -1 otherwise.
    
        Parameters
        ----------
        grid : SequenceLike
            The target grid to find.
    
        Returns
        -------
        int
            Indicates the position in the internal collection, or -1 if not found.
        """
        if self.abcStatus:
            _, idx = GridCollection.is_abc_contained_CLS(grid=grid,
                                                               gc=self.abcCollection)
        else:
            _, idx = GridCollection.is_abc_contained_CLS(grid=grid,
                                                               gc=self.to_abc_collection())
        return idx


    @classmethod 
    def is_abc_contained_CLS(cls, 
                             grid: SequenceLike, 
                             gc: np.ndarray
                             ) -> Tuple[bool, int]:
        """
        Probes for deep containment:
        Checks whether a given abc-grid is contained in a collection, by comparing
        canonical abc-keys.
    
        Parameters
        ----------
        grid : array-like of size dim**2 (where dim=9 for classical Sudoku)
            The target grid (int or abc-style) to search for.
        gc : np.ndarray;  must be reshapable to (n, dim, dim)
            The collection of grids to search in.
    
        Returns
        -------
        (contained: bool, index: int)
            Whether the grid is found, and its index (or -1 if not found).
        """
        from PsQ_Grid import Grid
        g = Grid.from_grid(grid)
        dim = g.DIMENSION 
        
        try:
            gc.reshape(-1, dim, dim)  # probe reshaping only
        except ValueError:
            raise ValueError(f"Shape incompatibility: collection cannot be reshaped to (n, {dim}, {dim})")

        if np.issubdtype(gc.dtype, np.integer): 
            recoder = np.array([chr(c) for c in range(65, 65 + dim)])
            abc_gc = cls.recode_collection_CLS(recoder, gc).reshape(-1, dim**2)
        else:
            abc_gc = gc.reshape(-1, dim**2)
    
        abc_key = g.canonical_abc_key()        
        abc_gen = (tuple(row) for row in abc_gc)
        for ix, grd in enumerate(tqdm(abc_gen, desc="Checking for containment")):
            if grd == abc_key:
                contained = True
                idx = ix
                break
        else:
            idx = -1
            contained = False

        return contained, idx

        




        

##########################################################################################
##                                                                                      ##
##                A D V A N C E D   F U N C T I O N A L I T I E S                       ##     
##                                                                                      ##
##########################################################################################
    
    
    @staticmethod
    def gridCollection_setDifference(gc_1: np.ndarray, gc_2: np.ndarray) -> np.ndarray:
        """ Returns the set difference gc_1 - gc_2 for two grid collections. """
        keys_1 = GridCollection._get_flat_keys(gc_1)
        keys_2 = GridCollection._get_flat_keys(gc_2)
        keys_2_set = set(keys_2)
        keep_mask = np.array([k not in keys_2_set for k in keys_1])
        return gc_1[keep_mask]
    
    
    @staticmethod
    def gridCollection_setIntersection(gc_1: np.ndarray, gc_2: np.ndarray) -> np.ndarray:
        """ Returns the set intersection gc_1 ∩ gc_2 for two grid collections. """
        keys_1 = GridCollection._get_flat_keys(gc_1)
        keys_2 = GridCollection._get_flat_keys(gc_2)
        keys_2_set = set(keys_2)
        keep_mask = np.array([k in keys_2_set for k in keys_1])
        return gc_1[keep_mask]
    
        
    @staticmethod
    def gridCollection_setUnion(gc_1: np.ndarray, gc_2: np.ndarray) -> np.ndarray:
        """ Returns the set union gc_1 ∪ gc_2 for two grid collections. """
        keys_1 = GridCollection._get_flat_keys(gc_1)
        keys_2 = GridCollection._get_flat_keys(gc_2)
    
        combined = np.concatenate([gc_1, gc_2], axis=0)
        combined_keys = keys_1 + keys_2
    
        seen = {}
        for idx, k in enumerate(combined_keys):
            if k not in seen:
                seen[k] = idx
    
        unique_indices = list(seen.values())
        return combined[unique_indices]
    
        
        
    @staticmethod
    def _get_flat_keys(gc: np.ndarray):
        """
        @aux_method
        Flattens a grid collection and produces a hashable key per grid.
    
        Parameters
        ----------
        gc : np.ndarray of shape (n, 9, 9) or (n, 81)
            Grid collection, either character-based or integer-based.
    
        Returns
        -------
        flat_grids : np.ndarray of shape (n, 81)
            Flattened version of the grid collection.
        keys : list of str or bytes
            Unique, hashable keys for each grid.
        """
        flat_grids = gc.reshape((gc.shape[0], -1))
        if gc.dtype.kind in {'U', 'S'}:
            return [''.join(row) for row in flat_grids]
        else:
            return [row.tobytes() for row in flat_grids]
    


    






##########################################################################################
##                                                                                      ##
##        F I L L I N G  the  A C T I V E  C O L L E C T I O N  -->  .activate()        ##     
##                                                                                      ##
##########################################################################################

# -- 'horizontal' permutation == label permutation == DIMENSION! x recoding of abc-grid:

    def activate_horizontalSeries(self, idx: int=0) -> None:
        """
        Generates all digit relabelings (i.e., horizontal label permutations) of a base grid.
    
        For classical Sudoku, this corresponds to generating all 9! = 362880 permutations
        of the digits [1, ..., 9], and applying each to the given base grid `g`. Each 
        permutation defines a unique recoding of the digit labels while preserving the
        grid's structural validity.
    
        The resulting collection of relabeled grids is stored in `.activeSeries`, and the
        activation status is set to `"horizontal"`.
    
        Parameters
        ----------
        idx : int, optional
            Index of the base grid `g` in the internal `.collection` to apply permutations to.
            Must satisfy 0 ≤ idx < SIZE_collection.
    
        Raises
        ------
        ValueError
            If `idx` is outside the valid range.
    
        Returns
        -------
        None
            The modified series is stored in `.activeSeries`.
        """
        if idx < 0 or idx >= self.SIZE_collection:
            raise ValueError(f"Index value idx must be 0 <= idx < {self.SIZE_collection}")
        
        grid = self.collection[idx]  

        perms = np.array(list(permutations(range(1, self.DIMENSION + 1))), dtype=np.uint8)  # = (DIMENSION!,)
        
        # Create an array for all recoded grids
        recoded_grids = np.zeros((len(perms), self.SIZE_grid), dtype=np.uint8)
        
        # Create a lookup index: grid maps values 1–9 to positions 0–8
        grid_index = grid - 1  # values 1–9 → 0–8
    
        # Use broadcasting: For each permutation, index into the perm array with grid_index
        recoded_grids = perms[:, grid_index]  # shape: (362880, 81)
    
        self.__activeSeries = recoded_grids.reshape(-1, self.SIZE_grid)
        self.__activationStatus = "horizontal"



# -- 'vertical' permutation == grid permutations == swapping rows/columns; 
#    by default: == internal collection
    
    def activate_verticalSeries(self, 
                                double: Optional[bool] = None,
                                explicit_validation: bool = False,
                                new_series_fromIndex: Optional[int] = None
                                ) -> None:
        """
        Activates a vertical series — i.e., a collection of grid structure permutations — 
        from the current internal collection or by generating a new one.
    
        Vertical permutations include legal rearrangements of rows, columns, or blocks that
        preserve Sudoku validity (1679616 possible for classical Sudoku, or 3359232 if 
        diaflection is included). Many internal methods assume that `.collection` is already
        a vertical series.
    
        Depending on the parameters, this method:
        - Loads the current `.collection` as the active series (default),
        - Validates `.collection` as a vertical series (`explicit_validation=True`),
        - Generates a new vertical series from a given grid index (`new_series_fromIndex`),
          optionally using double permutations (`double=True`).
    
        The resulting series is stored in `.activeSeries`, and the activation status 
        is updated accordingly.
    
        Parameters
        ----------
        new_series_fromIndex : Optional[int], default=None
            If provided, generates a new vertical series from the grid at this index in `.collection`.
    
        double : Optional[bool], default=None
            If `True`, and `new_series_fromIndex` is given, uses the double vertical group 
            (includes diaflections) to generate the series.
    
        explicit_validation : bool, default=False
            If `True`, the method validates that `.collection` is a proper vertical series
            by reconstructing and comparing expected permutations.
    
        Returns
        -------
        None
            The modified series is stored in `.activeSeries`.
    
        Notes
        -----
        - If `.collection` is not known to be a valid vertical series (e.g., after loading 
          from an unverified source), set `explicit_validation=True`.
        - If a fresh vertical series is needed, use `new_series_fromIndex` instead.
        """
        if new_series_fromIndex is not None:
            self._activate_from_index(new_series_fromIndex, double=double)
            return
        
        if explicit_validation:
            self._validate_vertical_series()
    
        self.__activeSeries = self.collection.reshape(-1, self.SIZE_grid)
        self.__activationStatus = "vertical"
    
    
    def _activate_from_index(self, idx: int, double: bool = False) -> None:             
        """
        Internal helper to generate a vertical permutation series from a base grid at index `idx`.
    
        This constructs a new `GridCollection` from the specified grid using either single or 
        double vertical permutation logic (depending on `double`), and activates it by storing
        the result in `.activeSeries`.
    
        Parameters
        ----------
        idx : int
            Index in the internal `.collection` to use as the base grid.
    
        double : bool, default=False
            If `True`, includes diaflection-based permutations (double vertical group).
    
        Raises
        ------
        ValueError
            If `idx` is not within the range [0, SIZE_collection).
    
        Returns
        -------
        None
        """
        if not (0 <= idx < self.SIZE_collection):
            raise ValueError(f"Index {idx} out of bounds (0 ≤ idx < {self.SIZE_collection})")
        grid = self.collection[idx]
        new_gc = GridCollection.from_scratch(grd=grid, double=double)
        self.__activeSeries = new_gc.collection.reshape(-1, self.SIZE_grid)
        self.__activationStatus = "vertical_new"
    

    def _validate_vertical_series(self) -> None: 
        """
        Internal helper to verify whether the current internal collection conforms 
        to the structure of a vertical permutation series.
    
        A vertical series consists of all legal permutations of a base grid under
        allowed row/column (and optionally diaflection) operations. This check
        first assesses whether the size is plausible, then performs a full
        content-level equivalence check.
    
        Raises
        ------
        ValueError
            If the collection does not resemble or match a valid vertical series.
        """        
        if not self._looks_like_vertical_series():
            raise ValueError("Collection does not resemble a vertical series. "
                             "Use `activate_newVerticalSeries(idx)` instead.")
        if not self._is_vertical_series(double=self._guess_double()):
            raise ValueError("Collection fails vertical series validation.")

    def _looks_like_vertical_series(self) -> bool:
        """
        Internal helper to check whether the size of the internal collection matches 
        that of a plausible vertical permutation series.
    
        Returns
        -------
        bool
            True if the collection size matches the known cardinalities of 
            vertical series (with or without diaflection), False otherwise.
        """
        return self.SIZE_collection in {
            factorial(self.BASE_NUMBER) ** (2 * (self.BASE_NUMBER + 1)),
            2 * factorial(self.BASE_NUMBER) ** (2 * (self.BASE_NUMBER + 1))
        }
    
    def _guess_double(self) -> bool:
        """
        Internal helper: guesses whether the current vertical series includes diaflection 
        permutations (i.e. is 'double') based on its size.
    
        Returns
        -------
        bool
            True if the collection size implies a double vertical series.
        """        
        
        return self.SIZE_collection == 2 * factorial(self.BASE_NUMBER) ** (2 * (self.BASE_NUMBER + 1))

        
            
    def _is_vertical_series(self, double: bool = False) -> bool:
        """
        Compares the internal collection against a freshly generated vertical 
        permutation series to verify full structural and content equivalence.
    
        Parameters
        ----------
        double : bool, optional
            If True, includes diaflection permutations in the comparison.
    
        Returns
        -------
        bool
            True if the internal collection exactly matches the expected vertical 
            permutation set (modulo ordering), False otherwise.

        Notes
        -------
        The base grid is taken from this internal collection; if this collection really
        is a vertical series then a series generated from any of its elements must 
        be identical (i.e. contain the same permutations) -- modulo ordering.
        """        
        from PsQ_Grid import Grid
        grid = self.collection[0] 
        g = Grid.from_grid(grid)  
        if double:
            perms = g.permuteGrids_T(toCollection=False)
        else:
            perms = g.permuteGrids(toCollection=False)
        self.reshape(-1, self.SIZE_grid, inplace=True)
        perms = perms.reshape(-1, self.SIZE_grid)

        A_sorted = self.collection[np.lexsort(self.collection.T[::-1])]
        B_sorted = perms[np.lexsort(perms.T[::-1])]
        return np.array_equal(A_sorted, B_sorted)



# -- 'THIS' == internal collection, if origin / 'vertical'-status uncertain:
         
    def activate_thisSeries(self, name: str = "THIS") -> None:
        """
        Treats the current internal collection as the active working series,
        regardless of its origin or structure.
    
        This method is used when geometric or structural assumptions about 
        the collection are uncertain but it is still desirable to set it 
        as active for downstream operations.
    
        Parameters
        ----------
        name : str, optional
            Label for this activation status. Default is "THIS".
        """        
        self.__activeSeries = self.collection.reshape(-1, self.SIZE_grid) 
        self.__activationStatus = name
        


# -- 'random' collection; valid grids, but not necessarily related by transformation:

    def activate_randomSeries(self, 
                              how_many: int = 100000, 
                              series: int = 3, 
                              seed: int = 47,
                              checkContainment: bool = False
                              ) -> None:             
        """
        Activates a randomly generated collection of valid Sudoku grids as the
        new `.activeSeries`.
    
        The grids are generated in batches (<==> 'series') for improved diversity: 
        for higher values for 'series', fewer samples from more series are included. 
        Optionally, generated grids can be filtered to ensure they 
        do not overlap with the current internal collection.
    
        Parameters
        ----------
        how_many : int, optional
            Total number of grids to generate. Default is 100000.
        series : int, optional
            Number of base series to generate. A higher number increases variety 
            but may increase runtime. Default is 3.
        seed : int, optional
            Initial random seed used for reproducibility. Default is 47.
        checkContainment : bool, optional
            If True, ensures none of the generated grids are already present 
            in the internal `.collection`. This adds overhead but may be useful 
            for controlled comparisons. Default is False.
        """
        if checkContainment:
            controlCollection = self.collection
        else:
            controlCollection = None
        
        self.__activeSeries = GridCollection.generate_randomGrids(how_many = how_many, 
                                                                  series = series, 
                                                                  seed = seed,
                                                                  basenumber = self.BASE_NUMBER,
                                                                  controlCollection = controlCollection) 
        self.__activationStatus = "random"


    @staticmethod
    def generate_randomGrids(how_many: int = 100000, 
                             series: int = 1, 
                             seed: int = 42,
                             basenumber: int = 3,
                             controlCollection: Optional[np.ndarray] = None
                             ) -> np.ndarray:          
        """
        Generates a specified number of valid Sudoku grids through randomized 
        canonical permutations.
    
        Optionally avoids overlap with a reference collection by checking 
        canonical keys.
    
        Parameters
        ----------
        how_many : int, optional
            Total number of grids to generate. Default is 100000.
        series : int, optional
            Number of permutation batches to generate. Default is 1.
        seed : int, optional
            Initial seed for random number generation. Default is 42.
        basenumber : int, optional
            Sudoku base number (e.g. 3 for 9x9). Default is 3.
        controlCollection : np.ndarray, optional
            A reference collection used to avoid generating duplicate grids.
            If provided, new grids are checked against its canonical keys.
    
        Returns
        -------
        np.ndarray
            A (how_many, 81)-shaped array of flattened integer grids.
        """        
        from PsQ_Grid import Grid
        if controlCollection is not None:
            g = Grid.from_grid(controlCollection[0])
            if np.issubdtype(controlCollection.dtype, np.integer):
                recoder = np.array([chr(i) for i in range(65, 65 + g.DIMENSION)])
                grids: np.ndarray = controlCollection.reshape(-1, g.DIMENSION, g.DIMENSION)
                grids = GridCollection._batch_topRow_recode(grids, recoder)
                    
            checkGrids = set(tuple(g.flatten()) for g in controlCollection)
        else:
            g = Grid(basenumber)
            
        guest_gc = np.empty((0, g.SIZE), dtype=np.uint8)
        per_series: int = how_many // series
        
        for grids_nr in range(1, series+1):
            
            print(f"Generating guest grids, series {grids_nr}: ") 
            
            while guest_gc.shape[0] < grids_nr * per_series:
                g.generate_rndGrid(seed=seed)
                
                if controlCollection:
                    flat_abc: Tuple[str] = g.canonical_abc_key() 
                    if flat_abc in checkGrids:
                        continue
                    
                gc = g.permuteGrids(toCollection=False)
                idx = np.random.choice(gc.shape[0], per_series)
                guest_gc = np.concatenate([guest_gc, gc[idx]])
                    
                seed += 17 
                print(f"Current size: {guest_gc.shape[0]}")

        guest_gc = _fill_from_final_series(guest_gc, gc, how_many)
        return guest_gc.reshape(-1, g.SIZE)



# -- 'multiclass' <-- multiple horizontal series

    def activate_multiClassSeries(self, 
                                  class_idx: SequenceLike = None, 
                                  classes: int = 3, 
                                  make_garbage: bool = True,
                                  garbage_size: Optional[int] = None
                                  ) -> None:             
        """
        Activates a labeled multiclass dataset composed of several horizontal 
        series and, optionally, a garbage class.
    
        This method combines grids from multiple distinct horizontal series (each 
        generated from a different canonical base) along with a negative class 
        formed from the garbage collection.
    
        Parameters
        ----------
        class_idx : sequence of int, optional
            Indices into the internal `.collection` to use as base grids for 
            each class. If None, random base indices are chosen. Default is None.
        classes : int, optional
            Number of labeled horizontal series (i.e., valid Sudoku classes) 
            to include. Default is 3.
        make_garbage : bool, optional
            Whether to include a garbage class labeled as 0. Default is True.
        garbage_size : int, optional
            How many garbage grids to include. If None, defaults to the full 
            factorial size of a single series. Raises an error if the 
            garbage collection is too small.
    
        Raises
        ------
        ValueError
            If `garbage_size` is requested but insufficient garbage is available. 
            
        Notes
        ------ 
        - Each real class is horizontal series; motivation: size 362880 per series 
            is doable for multiple series, and horizontal permutation (= substitution) 
            can be learned extremely well (in a binary classification task, the accuracy
            is usually at 100% from epoch 2 onward). But it should be kept in mind
            that those series are closed classes, there are no new data. Thus no class 
            beyond the (training &) test set can be properly classified; for this 
            reason, it is useful to include a (mixed) garbage class, in which case 
            the expectation for a good model is to predict 0 for basically anything else. 
        
        - If make_garbage = True, the garbage classe gets the label 0, which seems 
            straightforward. All real classes get their label in ascending order 0/1,2,3...
            in the order in which they were generated. The labels merely express distinctness,
            but have no inherent meaning.
            
        """        
        offset: int = int(make_garbage)
        self.clear_activeSeries() 
        
        grid_list, label_list = [], []

        if make_garbage:        
            if garbage_size is None:
                garbage_size = factorial(self.DIMENSION)
            if self.SIZE_garbageCollection < garbage_size: 
                raise ValueError(f"Not enough garbage: {self.SIZE_garbageCollection} -- required: {garbage_size}")
            _garb_idx = np.random.choice(self.SIZE_garbageCollection,
                                        garbage_size,
                                        replace=False)
            _garb_coll = self.garbageCollection[_garb_idx] 
            
            grid_list.append(_garb_coll)
            label_list.append(np.full((garbage_size,), 0, dtype=np.uint8))
                    
        if class_idx is None:
            class_idx = np.random.choice(self.SIZE_collection, classes, replace=False)

        for idx in range(offset, classes + offset): 
            
            self.activate_horizontalSeries(class_idx[idx-offset])
            labels = np.full((self.SIZE_activeSeries), idx, dtype=np.uint8)
            
            grid_list.append(self.activeSeries)
            label_list.append(labels)
                                                                 
        self.__activeSeries = np.concatenate(grid_list, axis=0)
        self.__labels = np.concatenate(label_list, axis=0)            
            
        self.__activationStatus = "multiclass"
        self.__distributor = "classes"
        self.__classes = classes + offset
                
        assert self.__labels.shape[0] == self.__activeSeries.shape[0]




###   -------------------   Modifications from existing series   -------------------   ###

# -- 'diaflect' == dia (-gonally re-) flect == matrix transposition

    def activate_diaflectedSeries(self,  
                                  idx: int = 42, 
                                  how_many: int = 362880) -> None:         
        """
        Activates the transposed (diaflected) version of the currently active series.
    
        Performs a matrix transposition (diagonal reflection) of all grids currently
        in `.activeSeries` and reassigns the result to `.activeSeries`.
    
        If no series is currently active, defaults to activating the current internal 
        collection using `activate_thisSeries()`.
    
        Parameters
        ----------
        idx : int, optional. Default is 42.
            Reserved for future use (e.g., targeted selection or logging).
        how_many : int, optional. Default is 362880.
            Reserved for future use (e.g., limiting transformation size).
        """        
        if self.activationStatus == "not_activated":
            self.activate_thisSeries()
                   
        self.__activeSeries =  GridCollection.diaflect_Collection(self.activeSeries)
        self.__prefix = "diaflect_from_" + self.__prefix


    @staticmethod
    def diaflect_Collection(gridCollection: np.ndarray) -> np.ndarray:
        """
        Applies matrix transposition to a collection of Sudoku grids.
    
        Each 2D grid in the collection is transposed across its main diagonal, 
        resulting in a 'diaflected' version. Works on flat (n, SIZE) or reshaped 
        (n, N, N) arrays.
    
        Parameters
        ----------
        gridCollection : np.ndarray
            Input array of flattened or reshaped grids.
    
        Returns
        -------
        np.ndarray
            Transposed (diaflected) grid collection in flattened format.
    
        Raises
        ------
        ValueError
            If the input shape or inferred base number is invalid.
        """
        try:
            _type_checker(gridCollection)
            _total: int = gridCollection.shape[0]
            _gridsize: int = round(gridCollection.size / _total)        
            _base_number = _infer_basenumber(_gridsize)        
            reshaped = gridCollection.reshape(_total, 
                                              _base_number**2, 
                                              _base_number**2
                                              ).astype(gridCollection.dtype)
        except ValueError as ve:
            raise ValueError(f"Invalid size/shape {gridCollection.shape}: {ve}")
            
        gc_transposed = reshaped.transpose(0, 2, 1).copy()
        return gc_transposed.reshape(-1, _gridsize)



# -- 'abc' == alphabetized grid collection, rigid decoder

    def activate_abcSeries(self,
                           to_active: bool = True,
                           to_storage: bool = False,
                           from_active: bool = True,
                           warning_active: bool = True)  -> None:         
        """
        Activates or stores an alphabetized version of a grid collection.
    
        Alphabetization replaces the grid's digit vocabulary with the first 
        `DIMENSION` uppercase letters (A–I for standard 9×9). Can update both the 
        `.activeSeries` and the `.abcSeries` storage, depending on arguments.
    
        Parameters
        ----------
        to_active : bool, optional. Default is True.
            Whether to activate the abc-transformed collection as `.activeSeries`.
    
        to_storage : bool, optional. Default is False.
            Whether to store the abc-transformed collection in `.abcSeries`.
    
        from_active : bool, optional. Default is True.
            If True, uses `.activeSeries` as source for transformation.
            Otherwise, uses the internal collection.
    
        warning_active : bool, optional. Default is True.
            Emits a warning if `.activeSeries` is inactive or inconsistent.
    
        Raises
        ------
        ValueError
            If one-hot encoding is active or neither `to_active` nor `to_storage` is set. 
    
        Notes
        ------
        Enabling `to_storage=True` stores the abc-transformed collection in `.abcSeries`. 
        This may consume additional memory, but is beneficial if multiple identity or 
        containment checks in abc-space are expected. The performance gain from avoiding 
        repeated recomputation may outweigh the memory cost.
        """
        if self.oneHotStatus:
            raise ValueError("Onhe-hot encoded series cannot be alphabetized")
        if not (to_active or to_storage):
            raise ValueError("At least one of to_series or to_storage must be True.")
        if to_storage:
            self.__abcSeries = self.to_abc_collection().reshape(-1, self.SIZE_grid)
            self.__abcStatus = True
        if to_active:            
            if from_active:
                recoder = np.array([chr(i) for i in range(65, 65 + self.DIMENSION)])
                self.__activeSeries = self.recode_collection(recoder=recoder,
                                                             from_active=from_active,
                                                             warning_active=warning_active
                                                             ).reshape(-1, self.SIZE_grid)
            else:    
                self.__activeSeries = self.to_abc_collection()
            
            self.__prefix = "abc_from_" + self.__prefix
                        
    
# -- 'recode' == swap labels according to given recoder 

    def activate_recodedSeries(self,
                               recoder: SequenceLike,
                               from_active: bool = False,
                               warning_active: bool = True)  -> None:         
        """
        Activates a version of the collection with values relabeled via a custom mapping.
    
        Applies the provided `recoder` to either the active series or internal collection. 
        The transformation maps the grid's existing symbols (digits or characters) to 
        new ones according to `recoder`, which must match the vocabulary size.
    
        Parameters
        ----------
        recoder : SequenceLike
            Mapping from original vocabulary to new labels (e.g., 1→7, 2→9, ...).
    
        from_active : bool, optional. Default is False.
            If True, uses `.activeSeries` as source. Otherwise, uses internal collection.
    
        warning_active : bool, optional. Default is True.
            Emits a warning if `.activeSeries` is missing when `from_active=True`.
    
        Raises
        ------
        ValueError
            If recoder length mismatches dimension or input is malformed.
        """        
        grids = self.recode_collection(recoder=recoder,
                                       from_active=from_active,
                                       warning_active=warning_active)
        self.__activeSeries = grids.reshape(-1, self.SIZE_grid)
        self.__prefix = "recoded_from_" + self.__prefix
    


# -- 'puzzle':  grids with blanks (= 0s) 

    def activate_puzzleSeries(self,
                              k: int,
                              extra_high_vals: bool = True,
                              include_parentSet: bool = True,
                              run: int = 3
                              ) -> None: 
        """
        Activates a puzzle (blanked) version of the currently active grid series.
    
        This method replaces the active series with a new collection of Sudoku puzzles,
        each created by blanking out `k` or fewer positions (set to 0). Optionally,
        additional examples with more blanks are generated for training diversity.
    
        Parameters
        ----------
        k : int
            Maximum number of blanks per grid (minimum is 1).
        extra_high_vals : bool, default=True
            If True, generate additional samples with more blanks (up to `k`),
            using a higher range of blank counts to encourage difficulty variance.
        include_parentSet : bool, default=True
            If True, include the original (unmasked) grids in the dataset as labels.
            If False, only the masked versions are retained.
        run : int, default=3
            Multiplier for dataset augmentation; each base grid is used `run` times
            to generate `run` distinct masked variants.
    
        Notes
        -----
        After execution:
        - `self.activeSeries` contains the blanked (puzzle) grids.
        - `self.labels` holds the corresponding solution grids.
        - `self.k_blanks` records the number of blanks in each grid.
        - The prefix and class-distributor values are updated accordingly.
        """        
        if self.activationStatus == "not_activated":
            raise ValueError("No series activated")         
        self.clear_labels()
        
        if include_parentSet: 
            self.__labels = self.activeSeries.reshape(-1, self.SIZE_grid)  
            self.__maskedGridCollection = self.activeSeries.reshape(-1, self.SIZE_grid) 
            self.__k_blanks = [0 for x in range(self.SIZE_activeSeries)]
        else: 
            self.__labels = np.empty((0, self.SIZE_grid), dtype=self.activeSeries.dtype) 
            self.__maskedGridCollection = np.empty((0, self.SIZE_grid), dtype=self.activeSeries.dtype) 
            self.__k_blanks = []
        
        self._mask_normal(k=k,
                          run=run) 
        if extra_high_vals:
            self._mask_highValues(k=k, 
                                  run=run)
        self.__activeSeries = self.__maskedGridCollection.copy()
        del self.__maskedGridCollection 
        self.__k_blanks = np.asarray(self.__k_blanks)
        
        self.__prefix  = f"k_{k}_max_puzzle_from_" + self.__prefix
        self.__distributor = "k_blanks"
        self.__classes = k 
        
        assert len(self.__k_blanks) == self.SIZE_activeSeries == self.SIZE_labels
      

    def _mask_normal(self, 
                     k: int, 
                     run: int = 3
                     ) -> None:             
        """
        Internal method: generates `run` masked variants per grid with 1 to `k` blanks.
    
        Applies random masking to the active series by removing a small, varying number
        of digits per grid. The blanked positions are filled with 0s. Used as baseline
        augmentation for puzzle generation.
    
        Parameters
        ----------
        k : int
            Maximum number of blanks per grid.
        run : int, default=3
            Number of masked variants to generate per original grid.
        """
        
        
        size: int       = self.SIZE_activeSeries
        n_samples: int  =  size * run
        output = np.empty((n_samples, self.SIZE_grid), dtype=self.activeSeries.dtype)
        labels = np.empty((n_samples, self.SIZE_grid), dtype=self.activeSeries.dtype)
        
        for iteration in tqdm(range(n_samples), desc=f"Generating grids with 1-{k} blanks"): 
            idx = iteration % size 
            k_blanks = iteration % k + 1 
            mask = np.random.choice(range(self.SIZE_grid), k_blanks, replace=False) 
            grid = self.__activeSeries[idx].copy()
            grid[mask] = 0 
            output[iteration] = grid
            labels[iteration] = self.__activeSeries[idx].copy()
            self.__k_blanks.append(k_blanks)


        self.__maskedGridCollection = np.concatenate([self.__maskedGridCollection, output])
        self.__labels = np.concatenate([self.__labels, labels])


    def _mask_highValues(self, 
                         k: int, 
                         run: int = 3
                         ) -> None: 
        """
        Internal method: generates masked variants with more challenging blank ranges.
    
        Similar to `_mask_normal`, but draws `k_blanks` from the upper range of [int(k*0.6)+2, k],
        increasing puzzle difficulty. Used as an optional step to diversify the training set.
    
        Parameters
        ----------
        k : int
            Maximum number of blanks per grid.
        run : int, default=3
            Number of masked variants to generate per original grid.
        """        
        size: int       = self.SIZE_activeSeries
        n_samples: int  =  size * run

        output = np.empty((n_samples, self.SIZE_grid), dtype=self.activeSeries.dtype)
        labels = np.empty((n_samples, self.SIZE_grid), dtype=self.activeSeries.dtype)

        for iteration in tqdm(range(n_samples), desc=f"Generating grids with {int(k *(3/5) + 2)}-{k} blanks"): 
            idx = iteration % size 
            low = int(k *(3/5) + 2)
            diff = k - low + 1
            k_blanks = iteration % low + diff   
            mask = np.random.choice(range(self.SIZE_grid), k_blanks, replace=False) 
            grid = self.__activeSeries[idx].copy()
            grid[mask] = 0 
            output[iteration] = grid
            labels[iteration] = self.__activeSeries[idx].copy()
            self.__k_blanks.append(k_blanks)

        self.__maskedGridCollection = np.concatenate([self.__maskedGridCollection, output])
        self.__labels = np.concatenate([self.__labels, labels])



    def generate_blankedGrids(self, how_many: int = 100000, k_blanks: int = 42) -> np.ndarray:             
        """
        Generates a standalone collection of blanked grids with exactly `k_blanks` blanks per grid.
    
        Parameters
        ----------
        how_many : int, optional
            Total number of puzzles to generate. Default is 100,000.
        k_blanks : int, optional 
            Number of blank entries (set to 0) per grid. Default is 42.
    
        Returns
        -------
        output : np.ndarray
            A (how_many, 81)-shaped array of puzzle grids, each with `k_blanks` blanks
            and `81 - k_blanks` known digits.
        """
        output = np.empty((how_many, self.SIZE_grid), dtype=self.activeSeries.dtype)
        
        for iteration in tqdm(range(how_many), 
                              desc=f"Generating grids with {k_blanks} blanks"): 
            idx = iteration % self.SIZE_activeSeries 
            mask = np.random.choice(range(self.SIZE_grid), k_blanks, replace=False) 
            grid = self.__activeSeries[idx].copy()
            grid[mask] = 0 
            output[iteration] = grid
            
        return output


    


# -- 'one_hot' encoding of current series

    def activate_oneHotSeries(self, encode_garbage: bool = True) -> None:
        
                
        
        if self.activationStatus == "not_activated":
            raise ValueError("No series activated") 

        if encode_garbage and self.activationStatus != "multiclass":
            if self.SIZE_garbageCollection == 0:
                raise ValueError("Garbage collection is empty")
            self.__garbageCollection = GridCollection.to_oneHot(self.garbageCollection)
            # self.__garbageStatus["this_is_one_hot_garbage"] = self.SIZE_garbageCollection
            
        self.__activeSeries = GridCollection.to_oneHot(self.activeSeries)
        # self.__prefix = "one_hot_from_" + self.__prefix
        self.__oneHotStatus = True
        



        
    @staticmethod
    def to_oneHot(grid_coll: np.ndarray = None) -> np.ndarray:
        """
        Convert an ndarray of int values to one-hot encoded form.
        
        Parameters:
            grid_coll (np.ndarray): Input array of shape (n, SIZE_grid) 
                                    containing ints from 1 to DIMENSION.
    
        Returns:
            np.ndarray: One-hot encoded array of shape (n , SIZE_grid, DIMENSION)
        """
        try:
            _type_checker(grid_coll)
            _total: int = grid_coll.shape[0]
            _gridsize: int = round(grid_coll.size / _total)        
            _base_number = _infer_basenumber(_gridsize)        
            dimension = _base_number**2
            grid_coll = grid_coll.reshape(-1, dimension, dimension)
        
            if np.issubdtype(grid_coll.dtype, np.character):
                lut = np.zeros(256, dtype=np.uint8)
                lut[ord('A') : ord('A') + dimension] = np.arange(dimension)
                ascii_codes = grid_coll.astype('S1').view(np.uint8)
                grid_int = lut[ascii_codes]
            else:
                grid_int = grid_coll - 1  # zero-based
    
            return np.eye(dimension, dtype=np.uint8)[grid_int]

        except Exception as e:
            raise ValueError(f"Invalid size/shape or dtype for one-hot encoding: {e}")

        
        


    


##########################################################################################
##                                                                                      ##
##        F I L L I N G  the  G A R B A G E  C O L L E C T I O N  -->  .make()          ##     
##                                                                                      ##
##########################################################################################

# -- adding guest grids:

    def make_guestGrids(self, 
                        how_many: int = 100000, 
                        series: int = 1, 
                        seed: int = 42,
                        toGarbage: bool = True
                        ): 
        """
        Generates valid Sudoku grids that, however, do not belong to this permutation family
        (or at least, are not included in the internal collection). This is useful for 
        classification tasks that do not simply contrast valid vs. invalid, but probe
        for geometric properties. 
        A new gridcollection is generated, from which a random sample of grids is taken. 
        Via the parameter `series` the contribution of this procedure can be distributed
        among several new gridcollections: a higher value ensures more diversity, but
        potentially increases runtime. 

        Parameters
        ----------
        how_many : int, optional
            Number of grids to be generated. The default is 100000.
        series : int, optional
            DESCRIPTION. The default is 1.
        seed : int, optional
            Random seed. The default is 42.
        toGarbage : bool, optional
            If True (default), the resulting grids are added to the grabage collection.

        """
        from PsQ_Grid import Grid
        abc_gc = self.to_abc_collection()
        abc_set = set(tuple(g.flatten()) for g in abc_gc)
        guest_gc = np.empty((0, self.SIZE_grid), dtype=np.uint8)
        g = Grid()
        per_series: int = how_many // series
        
        for grids_nr in range(1, series+1):
            
            print(f"Generating guest grids, series {grids_nr}: ") 
            
            while guest_gc.shape[0] < grids_nr * per_series:
                g.generate_rndGrid(seed=seed)
                flat_abc: Tuple[str] = g.canonical_abc_key() 
            
                if not flat_abc in abc_set:
                    gc = g.permuteGrids(toCollection=False)
                    idx = np.random.choice(gc.shape[0], per_series)
                    guest_gc = np.concatenate([guest_gc, gc[idx]])
                    
                seed += 17 
                print(f"Current size: {guest_gc.shape[0]}")

        if series > 1:
            guest_gc = _fill_from_final_series(guest_gc, gc, how_many)

        if toGarbage:
            self.__garbageCollection = np.concatenate([self.__garbageCollection,
                                                       guest_gc.reshape(-1, self.SIZE_grid)])
            np.random.shuffle(self.__garbageCollection)
            self.__garbageStatus["guest_grids"] += how_many
        else:
            return guest_gc.reshape(-1, self.SIZE_grid)


# -- switch two cells:
            
    def makeFalseGrids_fromCurrent_switch(self, 
                                   active: bool = True,
                                   seed: int = None,
                                   clear_garbage: bool = True
                                   ) -> None:
        """
        Creates a collection of invalid grids on the basis of the current active series:
            for every grid in the active series, a random cell is picked and its value
            swapped with the value of a neighbour cell in the same row. This will always 
            lead to a violation of the column constraint (and sometimes of box constraint).
        Resulting size equals size of active series. 
        
        NB: by default, the garbage collection is cleared before the new grids are added. 
             This setting can be disabled by clear_garbage = False.  

        Parameters
        ----------
        active : bool, optional
            If True (default), false grids are created from the active series; in this
            case, the active series cannot be null. If set to False, the internal 
            collection will serve as a basis for false grids instead.
        clear_garbage : bool, optional
            If True (default), the garbage collection is cleared first.             
        seed : int, optional.  
            Random seed. The default is None.
        
        """
        if active:
            if self.__activationStatus == "not_activated":
                raise ValueError("No series activated!")

            grids = GridCollection.switch_toFalse(grids=self.activeSeries,
                                                  seed=seed
                                                  )
        else: 
            grids = GridCollection.switch_toFalse(grids=self.collection,
                                                  seed=seed
                                                  )
        if clear_garbage:
            self.clear_garbage() 
        grids = grids.reshape(-1, self.SIZE_grid) 
        self.__garbageCollection = np.concatenate([self.__garbageCollection, grids])                
        self.__garbageStatus["false_fromCurrent_switch"] = grids.shape[0]


    @staticmethod
    def switch_toFalse(grids: np.ndarray,
                                 seed: int = None,
                                 ) -> np.ndarray:
        
        if seed is not None:
            np.random.seed(seed)
        
        _type_checker(grids)
        try:
            _total: int = grids.shape[0]
            _gridsize: int = round(grids.size / _total)            
            _base_number: int = _infer_basenumber(_gridsize)
            dim: int = _base_number**2
            grids = grids.reshape(-1, dim, dim) 
        except ValueError:
            raise ValueError(f"Given grid array {grids.shape} cannot be reshaped to {(-1, dim, dim)} ")
        
        size = grids.shape[0]
        row_idx = np.random.choice(dim, size, replace=True)
        col_idx = np.random.choice(dim - 1, size, replace=True)
        grid_idx = np.arange(size)

        temp = grids[grid_idx, row_idx, col_idx].copy() 
        grids[grid_idx, row_idx, col_idx] = grids[grid_idx, row_idx, col_idx + 1]
        grids[grid_idx, row_idx, col_idx + 1] = temp
        
        return np.array(grids).reshape(-1, _gridsize)
            


# -- sequential overflow:
            
    def makeFalseGrids_fromCurrent_seq(self, 
                                   active: bool = True,
                                   shift: int = 1,
                                   clear_garbage: bool = True
                                   ) -> None: 
        """
        Generate invalid grids by sequentially shifting cell values within each grid.
    
        This method takes each grid from the current active series (or the full internal collection if `active=False`)
        and performs a right circular shift of its cells by `shift` positions. Cells moved off the end are wrapped 
        around to the start, which breaks Sudoku constraints except when the shift is a multiple of BASENUMBER^3 
        (corresponding to boxrow permutations).
    
        The resulting invalid grids are stored in the internal garbage collection.
    
        Parameters
        ----------
        active : bool, optional
            If True (default), operate on the active series of grids.
            If False, operate on the full internal collection.
        shift : int, optional
            Number of cell positions to shift to the right (default is 1).
            Must not be a multiple of BASENUMBER^3; otherwise, valid grids may result.
        clear_garbage : bool, optional
            If True (default), clear the garbage collection before adding new false grids.
    
        Raises
        ------
        ValueError
            If `shift` is a multiple of BASENUMBER^3, indicating the operation would produce valid grids.
    
        Notes
        -----
        - The size of the generated false grids collection matches that of the input grids. 
        - Be aware that the garbage collection is cleared by default.
        """  
        if active:
            if self.__activationStatus == "not_activated":
                raise ValueError("No series activated!")
            grids = GridCollection.sequentialOverflow_toFalse(self.activeSeries, 
                                                              shift=shift)
        else: 
            grids = GridCollection.sequentialOverflow_toFalse(self.collection,
                                                              shift=shift)

        if clear_garbage:
            self.clear_garbage()
        grids = grids.reshape(-1, self.SIZE_grid) 
        self.__garbageCollection = np.concatenate([self.__garbageCollection, grids])        
        self.__garbageStatus["false_fromCurrent_seq"] = grids.shape[0]


    @staticmethod 
    def sequentialOverflow_toFalse(grids: np.ndarray, shift: int = 1) -> np.ndarray: 
        try:
            _total: int = grids.shape[0]
            _gridsize: int = round(grids.size / _total)            
            _base_number: int = _infer_basenumber(_gridsize)
            grids = grids.reshape(-1, _gridsize) 
            if shift % _base_number**3 == 0: 
                raise ValueError(f"A shift by {shift} cells leads to a valid grid!")
        except ValueError as ve:
            raise ValueError(f"{ve}") 
            
        return np.roll(grids, shift=shift, axis=1) 



# -- shuffle grids, cardinality respected:

    def makeFalseGrids_cardinality(self, 
                              how_many: int = 100000, 
                              toGarbage: bool = True,
                              seed: Optional[int] = None,
                              check_explicitly: bool = False
                              ) -> Optional[np.ndarray]:             
        """
        Create grids that respect Sudoku digit cardinality but are invalid in other ways.
    
        Each generated grid contains every digit exactly DIMENSION times, satisfying the cardinality constraint.
        However, the grids will generally violate Sudoku rules (e.g., row, column, or box uniqueness).
    
        Due to the astronomical improbability of randomly generating valid Sudoku grids,
        these are virtually guaranteed to be invalid. If you want to be certain, set
        `check_explicitly=True` to filter valid grids out.
    
        Parameters
        ----------
        how_many : int, optional
            Number of false grids to generate (default 100,000).
        toGarbage : bool, optional
            If True (default), append the generated grids to the internal garbage collection.
            Otherwise, return them directly.
        seed : int, optional
            Optional random seed for reproducibility (default None).
        check_explicitly : bool, optional
            If True, explicitly verify and retain only invalid grids using class validation.
    
        Returns
        -------
        Optional[np.ndarray]
            If `toGarbage` is False, returns the generated grids as a numpy array
            of shape (how_many, SIZE_grid). Otherwise, returns None.
        """
        if seed is not None:
            np.random.seed(seed)
            
        flatrow: np.ndarray = np.arange(1, self.DIMENSION + 1)
        grid: np.ndarray = np.tile(flatrow, (1, self.DIMENSION))
        grdColl: np.ndarray = np.tile(grid, (how_many, 1))
        
        for grd in grdColl:
            np.random.shuffle(grd) 
        
        if check_explicitly: 
            false_idx = GridCollection.invalid_grids_idx_CLS(grdColl)
            grdColl = grdColl[false_idx]

        if toGarbage:
            self.__garbageCollection = np.concatenate([self.__garbageCollection,
                                                       grdColl.reshape(-1, self.SIZE_grid)])
            np.random.shuffle(self.__garbageCollection)
            self.__garbageStatus["false_cardinality"] += how_many
        else:
            return grdColl.reshape(-1, self.SIZE_grid)


    
# -- off by X (NB: applies pairwise): 

    def makeFalseGrids_offBy_X(self, 
                               how_many: int = 100000, 
                               X: int = 1,
                               how_far: int = 1,
                               toGarbage: bool = True
                               ) -> Optional[np.ndarray]:         
        """
        Generate grids that violate cardinality by altering exactly X values.
    
        Starting from grids that satisfy cardinality, this method selects `X` cell positions 
        within each grid and increments their digit values by `how_far` modulo DIMENSION,
        thus breaking cardinality constraints pairwise.
    
        Parameters
        ----------
        how_many : int, optional
            Number of false grids to generate (default 100,000).
        X : int, optional
            Number of cell positions (columns) to modify (default 1).
            Must satisfy 1 <= X < SIZE_grid.
        how_far : int, optional
            Offset by which to increment digits at selected positions (default 1).
            Will be wrapped modulo DIMENSION.
        toGarbage : bool, optional
            If True (default), append generated grids to the internal garbage collection.
            Otherwise, return them directly.
    
        Raises
        ------
        ValueError
            If `X` is not in the valid range [1, SIZE_grid - 1].
    
        Returns
        -------
        Optional[np.ndarray]
            If `toGarbage` is False, returns the generated grids as a numpy array
            of shape (how_many, SIZE_grid). Otherwise, returns None.

        """
        if not 1 <= X < self.SIZE_grid:
            raise ValueError(f"X must be in [1, {self.SIZE_grid - 1}] for cardinality violation")
        #assert 1 <= X < self.SIZE_grid, f"X must be in [1, {self.SIZE_grid - 1}] for cardinality violation"

        card_grids = self.makeFalseGrids_cardinality(how_many=how_many, toGarbage=False) 
        idx: np.ndarray = np.random.choice(self.SIZE_grid, X, replace=False)
        how_far = ((how_far - 1) % (self.DIMENSION - 1)) + 1  # make sure how_far != 0 and how_far != 9
            
        card_grids[:, idx] = ((card_grids[:, idx] - 1 + how_far) % self.DIMENSION) + 1    
    
        if toGarbage:
            self.__garbageCollection = np.concatenate([self.__garbageCollection,
                                                       card_grids.reshape(-1, self.SIZE_grid)])
            np.random.shuffle(self.__garbageCollection)
            self.__garbageStatus["false_off_by_X"] += how_many
        else:
            return card_grids.reshape(-1, self.SIZE_grid)


# -- arbitrary int-grids: 

    def makeFalseGrids_arbitrary(self,
                                 how_many: int = 100000, 
                                 toGarbage: bool = True,
                                 seed: Optional[int] = None,
                                 check_explicitly: bool = False
                                 ) -> Optional[np.ndarray]: 
        """
        Generate completely random integer grids, very unlikely to be valid Sudoku solutions.
    
        The grids consist of random digits sampled uniformly from 1 to DIMENSION, inclusive.
        Due to the negligible probability of producing valid Sudoku grids at random,
        these grids can be assumed invalid unless explicitly checked.
    
        Parameters
        ----------
        how_many : int, optional
            Number of random grids to generate (default 100,000).
        toGarbage : bool, optional
            If True (default), append the generated grids to the internal garbage collection.
            Otherwise, return them directly.
        seed : int, optional
            Optional random seed for reproducibility (default None).
        check_explicitly : bool, optional
            If True, filter the generated grids to keep only invalid ones using class validation.
    
        Returns
        -------
        Optional[np.ndarray]
            If `toGarbage` is False, returns the generated grids as a numpy array
            of shape (how_many, SIZE_grid). Otherwise, returns None.
        """
        if seed is not None:
            np.random.seed(seed)

        grdColl: np.ndarray = np.random.randint(low=1, 
                                                high=self.DIMENSION+1, 
                                                size=(how_many, self.SIZE_grid))
        
        if check_explicitly: 
            false_idx = GridCollection.invalid_grids_idx_CLS(grdColl)
            grdColl = grdColl[false_idx]
        
        if toGarbage:
            self.__garbageCollection = np.concatenate([self.__garbageCollection,
                                                       grdColl])
            np.random.shuffle(self.__garbageCollection)
            self.__garbageStatus["false_arbitrary"] += how_many
        else:
            return grdColl.reshape(-1, self.SIZE_grid)





##########################################################################################
##                                                                                      ##
##            A B C - G R I D S  and  generalized  R E C O D I N G                      ##     
##                                                                                      ##
##########################################################################################
        
        
    def to_abc_collection(self) -> np.ndarray:
        """
        Convert the current grid collection into its canonical 'abc grid' form.
    
        The 'abc grid' encoding fixes the top row as the ordered sequence 
        ['A', 'B', 'C', ..., ] for the grid dimension. This recoding effectively 
        normalizes all digit label permutations by choosing the top row as the 
        defining horizontal permutation series — a canonical representative 
        that captures the grid's deep identity modulo digit relabelings.
    
        Returns
        -------
        np.ndarray
            Array of grids recoded so that their top row is the standard abc sequence,
            thereby collapsing the digit-permutation symmetry class into a single form.
        """
        # The canonical recoder maps digits to letters A, B, C,... (based on dimension)
        recoder = np.array([chr(i) for i in range(65, 65 + self.DIMENSION)])

        # Reshape flat collection into (n_grids, dim, dim) for batch processing
        grids: np.ndarray = self.collection.reshape(-1, self.DIMENSION, self.DIMENSION)
        
        # Perform batch recoding based on each grid's top row
        return self._batch_topRow_recode(grids, recoder)    
    
    
    
    
        

        
    def recode_collection(self, 
                          recoder: SequenceLike,
                          from_active: bool = False,
                          warning_active: bool = True)  -> np.ndarray: 
        """
        Recode all grids in the collection according to a specified mapping of digit labels.
    
        This method generalizes the concept of abc grids by allowing arbitrary
        recoders (i.e., permutations or mappings of the digit/alphabet labels). 
        
        The recoding is performed per grid based on the top row values which
        act as the permutation key that aligns the digit distribution to the 
        target recoder.
    
        Parameters
        ----------
        recoder : SequenceLike
            A sequence representing the target label set to which grid digits are mapped.
            Must contain distinct values equal to the dimension.
        from_active : bool, optional
            If True, recode the currently active series of grids instead of the full collection.
            Raises warnings or errors for certain activation states.
        warning_active : bool, optional
            Enable or disable warnings when recoding from active grids.
    
        Raises
        ------
        RuntimeError
            If attempting to recode from a horizontally permuted active series (meaningless).
        AttributeError
            If no active series has been set when requested.
    
        Returns
        -------
        np.ndarray
            Recoded grid collection with digits relabeled according to the target recoder.
        """        
        if from_active:
            if warning_active and self.activationStatus in {"horizontal", "multiclass"}:
                msg = ("This is a really stupid idea! \n"
                       "\tA collection based on horizontal permutation reduces to a collection "
                       "of identical grids. \n"
                       "\tIf you insist, set warning_active = False") 
                raise RuntimeError(msg)
            if self.activationStatus == "not_activated":
                raise AttributeError("No series has been activated!") 
            grids = self.activeSeries.reshape(-1, self.DIMENSION, self.DIMENSION)
            
        else:
            grids = self.collection.reshape(-1, self.DIMENSION, self.DIMENSION) 
            
        if isinstance(recoder, str):
            recoder = _str_toSeq(recoder)
        recoder = np.asarray(recoder)
        _type_checker(recoder)
        
        size_test = (len(recoder) == self.DIMENSION == len(set(recoder)))
        if not size_test:
            raise ValueError(f"Proper recoder requires {self.DIMENSION} distinct values!")
                
        return self._batch_topRow_recode(grids, recoder)

        
    @classmethod
    def recode_collection_CLS(cls, recoder: SequenceLike, grids: np.ndarray = None)  -> np.ndarray: 
        """
        Class method variant of recode_collection for arbitrary input grids.
    
        This facilitates static usage outside of instances, supporting flexible
        recoding of any passed collection of grids according to the top-row key.
    
        Parameters
        ----------
        recoder : SequenceLike
            Target recoder sequence of distinct digit/label values.
        grids : np.ndarray, optional
            Array of Sudoku grids to recode; must be reshaped to (n, dim, dim).
    
        Raises
        ------
        ValueError
            - If grids cannot be reshaped to proper dimensions.
            - If recoder is invalid.
        TypeError
            - If input grids are malformed.
    
        Returns
        -------
        np.ndarray
            Recoded grid collection.
        """          
        try:
            basenumber: int = _infer_basenumber(grids.shape[0])
            dimension = basenumber**2
            grids = grids.reshape(-1, dimension, dimension)
        except Exception as e:
            raise e

        if isinstance(recoder, str):
            recoder = _str_toSeq(recoder)
        recoder = np.asarray(recoder)
        _type_checker(recoder)
        
        size_test = (len(recoder) == dimension == len(set(recoder)))
        if not size_test:
            raise ValueError(f"Proper recoder requires {dimension} distinct values!")
                
        return cls._batch_topRow_recode(grids, recoder)



    
    @staticmethod
    def _batch_topRow_recode(grids: np.ndarray, recoder: SequenceLike) -> np.ndarray: 
        
        """
        Vectorized batch recoding of Sudoku grids using top-row digit-to-label mappings.
    
        Core idea:
        Each grid’s top row acts as a *label permutation key* that encodes the horizontal
        digit permutation series. By constructing a lookup table (LUT) from the top row
        to the target recoder, the entire grid is recoded accordingly.
    
        This method operationalizes the core principle behind abc grids: **the top row
        uniquely defines the digit relabeling to normalize all grids into a canonical form.**
    
        Parameters
        ----------
        grids : np.ndarray
            Array of shape (n_grids, dim, dim) representing Sudoku grids.
            Can contain integers or characters.
        recoder : Sequence[int] or Sequence[str]
            Target sequence of labels/digits to map to (length = dim).
    
        Returns
        -------
        np.ndarray
            Recoded grids with digits replaced by the corresponding recoder values,
            shape (n_grids, dim, dim).
    
        Raises
        ------
        ValueError
            If any top row contains duplicates or recoding fails.
        TypeError
            If input grid format is invalid or recoding cannot be performed.
        """        
        try:
            grids = np.asarray(grids)
            _total: int = grids.shape[0]
            _gridsize: int = round(grids.size / _total)
            _base_number = _infer_basenumber(_gridsize)
            dim = _base_number**2
            grids = grids.reshape(-1, dim, dim)
        except ValueError as ve:
            raise ve
        except Exception as e:
            raise TypeError(f"Invalid input for grids: expected array-like of digits or chars. Details: {e}")
    
        # Get top rows: shape (_total, dim)
        top_rows = grids[:, 0, :]
    
        # Determine LUT size (safe upper bound)
        if top_rows.dtype.kind in {'U', 'S'}:
            max_index = dim  # 'A' maps to 0, so 'I' (or max char) maps to dim-1
        else:
            max_index = np.max(grids) + 1
        lut_size = max(128, max_index + 1)  # Ensure large enough LUT even for chars
    
        # Init LUT array
        luts = np.empty((_total, lut_size), dtype=np.asarray(recoder).dtype)
    
        for i in tqdm(range(_total), desc="Recoding grids with top-row LUT"):
            top_row = top_rows[i]
    
            # Convert to integer indices
            if top_row.dtype.kind in {'U', 'S'}:
                try:
                    idx = np.fromiter((ord(ch.upper()) - 65 for ch in top_row), dtype=int)
                except Exception as e:
                    raise ValueError(f"Invalid character in top row of grid {i}: {top_row}") from e
            else:
                idx = top_row.astype(int)
    
            # Safety: check for duplicates
            if len(set(idx)) != len(idx):
                raise ValueError(f"Top row of grid {i} contains duplicates: {top_row}")
    
            # Construct LUT
            lut = np.full(lut_size, -1, dtype=np.asarray(recoder).dtype)
            lut[idx] = recoder
    
            # Safety check
            if not np.all(lut[idx] == recoder):
                raise ValueError(f"LUT mismatch for grid {i}: mapping failed or overwritten.")
    
            luts[i] = lut
    
        # Flatten grids and apply LUT
        flat_grids = grids.reshape(_total, -1)
        recoded = np.empty_like(flat_grids, dtype=np.asarray(recoder).dtype)
    
        for i in range(_total):
            if flat_grids[i].dtype.kind in {'U', 'S'}:
                grid_idx = np.fromiter((ord(ch.upper()) - 65 for ch in flat_grids[i]), dtype=int)
            else:
                grid_idx = flat_grids[i]
            recoded[i] = luts[i][grid_idx]
    
        return recoded.reshape(_total, dim, dim)




##########################################################################################
##                                                                                      ##
##                 S T A T I C   C O N S T R U C T O R   M E T H O D S                  ##     
##                                                                                      ##
##########################################################################################

    @classmethod
    def from_scratch(cls, grd: SequenceLike = None, 
                     double: bool = False, 
                     seed: Optional[int] = None
                     ) -> GridCollection: 
        """
        Class-level factory method to construct a new GridCollection instance from a base grid.
    
        Parameters
        ----------
        grd : SequenceLike, optional
            An initial Sudoku grid representation (e.g., a 9x9 array or 81-length sequence).
            If provided, the grid will be used as a base to generate permutations.
            If None, a new random valid grid is generated instead.
        double : bool, default False. 
            If False (default), generates the vertical permutation series (= box/row permutations)
            from a valid initial grid (for classical 9x9 Sudoku -> 1679616 permutations),
            and instantiates a new GridCollection object with that set of permutation 
            as its internal collection.        
            If True, the transposed (= diaflected) version of each grid is added 
            (resulting size: 2 x 1679616).  
        seed : Optional[int], default None
            Random seed to ensure reproducible grid generation when no initial grid is provided.
    
        Returns
        -------
        GridCollection
            A collection of Sudoku grids derived from the base grid by applying permutation methods.
    
        Notes
        -----
        This method supports the creation of grid collections that can later be recoded into
        'abc grids' or other custom codings. By generating a base set of permutations, the collection
        maintains the structural properties of Sudoku but allows flexible transformations.
    
        The 'abc grid' concept — where the top row is always recoded to the letters 'A' through 'I' (for
        a 9x9 grid) — relies on having a consistent base grid collection, which this method helps establish.
    
        This factory method abstracts away the details of grid initialization and permutation,
        allowing higher-level workflows to focus on grid transformation and analysis.
        """                
        from PsQ_Grid import Grid
        if grd is not None:
            grid = Grid.from_grid(grd)
        else:
            grid = Grid()
            grid.generate_rndGrid(seed=seed)
        if double:
            return grid.permuteGrids_T(toCollection=True)
        else: 
            return grid.permuteGrids(toCollection=True)
        


    @classmethod
    def from_sql(cls, db_name: str, how_many: Optional[int] = None) -> GridCollection:
        """
        Constructs a GridCollection object from a SQLite database.
    
        Parameters
        ----------
        db_name : str
            Path to the SQLite database file.
        how_many : int, optional
            Number of random grids to fetch from the database. 
            If None, all available grids are retrieved.
    
        Returns
        -------
        GridCollection
            A new GridCollection instance with loaded grids.
        """
        if how_many is None:
            grids: np.ndarray = GridCollection.fetch_gridCollection_fromSQL(db_name)
        else:
            grids: np.ndarray = GridCollection.fetch_grids_fromSQL(db_name, how_many)
        return cls(grids)


    # TODO 
    @classmethod
    def from_file(cls, filepath: str):
        raise NotImplementedError("TODO!")






##########################################################################################
##                                                                                      ##
##        exporting / importing  D A T A S E T S  to / from  S Q L  DATABASES           ##     
##                                                                                      ##
##########################################################################################

    @staticmethod    
    def collection_toStrings(grids: np.ndarray) -> np.ndarray:
        """
        Converts a collection of 9x9 grids into an array of string representations.
    
        Parameters
        ----------
        grids : np.ndarray
            A numpy array of shape (n, 9, 9) or (n, 81) containing int or str/char entries.
    
        Returns
        -------
        np.ndarray
            An array of strings of shape (n,), where each string has length 81.

        Notes
        -------
        - Must not be used for basenumber > 3 because double-digit integers (10, 11 ...)
            cause len(grid_str) > grid size.
            
        """
        try:
            _type_checker(grids)
            _total: int = grids.shape[0]
            _gridsize: int = round(grids.size / _total)        
            _base_number = _infer_basenumber(_gridsize)
            grids = grids.reshape(-1, _gridsize)
            if _base_number > 3:
                raise ValueError(f"Basenumber {_base_number} is too large for char-based grid strings!")
        except ValueError as ve:
            raise ValueError(f"Invalid size/shape: {ve}")
    
        return np.fromiter((''.join(map(str, row)) 
                            for row in grids), dtype=f'<U{_gridsize}')



    def internal_toString(self) -> np.ndarray: 
        grids = self.collection.reshape(-1, self.SIZE_grid)
        return GridCollection.collection_toStrings(grids) 


    def active_toString(self) -> np.ndarray: 
        if self.activationStatus == "not_activated":
            raise ValueError("No series activated!")
        grids = self.activeSeries.reshape(-1, self.SIZE_grid)
        return GridCollection.collection_toStrings(grids) 

    def garbage_toString(self) -> np.ndarray:
        if self.SIZE_garbageCollection == 0:
            raise ValueError("No garbage in collection")
        grids = self.garbageCollection.reshape(-1, self.SIZE_grid)
        return GridCollection.collection_toStrings(grids) 


    def abc_toString(self) -> np.ndarray:
        grids = self.to_abc_collection().reshape(-1, self.SIZE_grid)
        return GridCollection.collection_toStrings(grids) 



    @staticmethod
    def gridStringCollection_toSQL(gridCollection: Iterable[str], db_name: str):
        """
        Inserts a collection of grid strings into a SQLite database.
        Ensures uniqueness of each grid via SQL-level constraint.
    
        Parameters
        ----------
        gridCollection : Iterable[str]
            Collection of grid strings (each of length 81).
        db_name : str
            Path to SQLite database.
        """      
        db_name += ".sqlite3"
        # Connect to SQLite database (it will create the database if it doesn't exist)
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
    
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS s_grids (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                value TEXT UNIQUE
            )
        ''')
    
        cursor.executemany('''
            INSERT OR IGNORE INTO s_grids (value) VALUES (?)
        ''', ((item,) for item in tqdm(gridCollection, desc="Processing"))) 
    
        conn.commit()
        conn.close()


    @staticmethod
    def save_toSQL(gridStringCollection: np.ndarray, db_name: str) -> None:             
        """
        Static helper method to persist a collection of Sudoku grids, encoded as strings,
        into an SQL database.
    
        Parameters
        ----------
        gridStringCollection : np.ndarray
            An array of strings representing Sudoku grids, typically generated by conversion
            methods such as `collection_toStrings` or `to_abc_collection`.
        db_name : str
            The filename or connection string for the target SQL database.
    
        Notes
        -----
        This method delegates actual SQL insertion to `gridStringCollection_toSQL`,
        which handles connection management, table creation, and batch inserts.
    
        The use of string representations (rather than raw numeric arrays) helps standardize
        storage format and facilitates easier retrieval and querying.
        """        
        GridCollection.gridStringCollection_toSQL(gridStringCollection, db_name)


    def internal_toSQL(self, db_name: str):  
        """
        Export the 'internal' grid collection (the current core dataset) to an SQL database.
    
        Parameters
        ----------
        db_name : str
            Target SQL database filename or connection string.
    
        This method fetches the internal grid collection, converts it to string format,
        and then saves it using the static `save_toSQL` method.
    
        """        
        grids: np.ndarray = self.internal_toString()
        GridCollection.save_toSQL(grids, db_name)
        

    def active_toSQL(self, db_name: str):  
        """
        Export the 'active' grid collection subset — representing currently 'activated'
        grids under some transformation or permutation series — to an SQL database.
    
        Parameters
        ----------
        db_name : str
            Target SQL database filename or connection string.
    
        This method converts the active grid subset to strings and exports them via
        `save_toSQL`.
    
        It allows easy checkpointing or exporting of focused grid collections,
        e.g., those subject to current analysis or model training.
        """        
        grids: np.ndarray = self.active_toString()
        GridCollection.save_toSQL(grids, db_name)
        
        
    def garbage_toSQL(self, db_name: str):
            
        """
        Export the 'garbage' grid collection to an SQL database. The garbage collection
        may contain a range of invalid grids, but also potentially, valid grids 
        belonging to a different permutation family. 
    
        Parameters
        ----------
        db_name : str
            Target SQL database filename or connection string.
    
        This method converts the garbage collection to strings and saves them.
    
        Exporting garbage collections may be useful for the creation of a repository
        of false grids that can be re-used.
        """        
        
        grids: np.ndarray = self.garbage_toString()
        GridCollection.save_toSQL(grids, db_name)


    @staticmethod
    def fetch_gridCollection_fromSQL(db_name: str) -> np.ndarray:
        """
        Fetches the entire collection from the SQLite DB and returns it as (n, gridsize) array.
        """
        with sqlite3.connect(db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM s_grids")
            rows = cursor.fetchall()

        str_coll = (row[0] for row in rows)
        return _arrays_from_stringCollection(str_coll)


    @staticmethod
    def fetch_grids_fromSQL(db_name: str, how_many: int) -> np.ndarray:
        """
        Fetches exactly how_many random grids from the DB, requires DB has >= how_many entries.
        Returns (how_many, gridsize) array.
        """
        with sqlite3.connect(db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM s_grids")
            total = cursor.fetchone()[0]

            if total < how_many:
                raise ValueError(f"Requested {how_many} grids but DB contains only {total}.")

            cursor.execute(f"""
                SELECT value FROM s_grids
                ORDER BY RANDOM()
                LIMIT {how_many}
            """)
            rows = cursor.fetchall()

        str_coll = (row[0] for row in rows)
        return _arrays_from_stringCollection(str_coll)


    def active_from_sql(self, 
                        db_name: str, 
                        how_many: Optional[int] = None,
                        p_status: str = "from_sql"
                        ) -> None:
        """
        Initializes active series from database.
    
        Parameters
        ----------
        db_name : str
            Path to the SQLite database file.
        how_many : int, optional
            Number of random grids to fetch from the database. 
            If None, all available grids are retrieved.
        p_status : str; default is "from_sql"
            Customized naming for the new active series. 
        """
        if how_many is None:
            grids: np.ndarray = GridCollection.fetch_gridCollection_fromSQL(db_name)
        else:
            grids: np.ndarray = GridCollection.fetch_grids_fromSQL(db_name, how_many)
        
        self.clear_activeSeries()
        self.activationStatus = p_status
        self.__activeSeries = grids



    def garbage_from_sql(self, 
                        db_name: str, 
                        how_many: Optional[int] = None,
                        g_status: str = "from_sql",
                        clear_garbage: bool = False
                        ) -> None:
        """
        Adds garbages series from database to -- or optionally overrides -- 
        the current garbage collection.
    
        Parameters
        ----------
        db_name : str
            Path to the SQLite database file.
        how_many : int, optional
            Number of random grids to fetch from the database. 
            If None, all available grids are retrieved.
        g_status : str; default is "from_sql"
            Customized naming for the garbage contribution. 
        """
        if how_many is None:
            grids: np.ndarray = GridCollection.fetch_gridCollection_fromSQL(db_name)
        else:
            grids: np.ndarray = GridCollection.fetch_grids_fromSQL(db_name, how_many)
        
        if clear_garbage:
            self.clear_garbage()
        self.__garbageCollection = grids
        self.__garbageStatus[g_status] += grids.shape[0]



    @staticmethod
    def containsGrid_SQL_DB(db_name: str, grid_str: str) -> bool:
        """
        Checks whether a given grid is already in the DB.
        
        Parameters:
            db_name (str):  Path to the SQLite database.
            grid_str (str): str representation of a grid.
        
        Returns:
            bool: True if db_name contains grid, False otherwise.
        """
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
    
        cursor.execute('SELECT 1 FROM s_grids WHERE value = ? LIMIT 1', (grid_str,))
        exists = cursor.fetchone() is not None
    
        conn.close()
        return exists

    
    @staticmethod
    def getSize_SQL_DB(db_name: str) -> int:
        """
        Returns the total number of entries in the SQLite database.
    
        Parameters
        ----------
        db_name : str
            Path to the SQLite database file.
    
        Returns
        -------
        int
            Number of entries in the 's_grids' table.
        """
        with sqlite3.connect(db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM s_grids")
            count = cursor.fetchone()[0]
        return count
        





##########################################################################################
##                                                                                      ##
##       S P L I T T I N G   A C T I V A T E D  C O L L E C T I O N S                   ##     
##                    into  T R A I N  /  T E S T  S E T S                              ##     
##                                                                                      ##
##########################################################################################

    def split_binary(self,
                      false_ratio: float = 0.5,
                      train_ratio: float = 0.8,
                      seed:        int   = None,
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        """
        Create a balanced binary classification dataset by mixing 'true' grids (active series)
        with 'false' grids (garbage collection), and split into training and testing subsets.
        
        This method facilitates supervised learning tasks where the model distinguishes valid
        Sudoku grids from invalid or discarded ones.
    
        Parameters
        ----------
        false_ratio : float, optional (default=0.5)
            The target ratio of negative ('false') samples relative to the positive ('true') samples.
            For example, false_ratio=0.5 aims for one false sample per two true samples.
            
        train_ratio : float, optional (default=0.8)
            Proportion of the combined dataset allocated for training; the rest is reserved for testing.
            
        seed : int or None, optional (default=None)
            Random seed for reproducibility of the dataset splits and sampling.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple containing `(X_train, X_test, y_train, y_test)` where:
              - `X_train`, `X_test` are arrays of Sudoku grids (both true and false),
              - `y_train`, `y_test` are corresponding binary labels (0 for false, 1 for true).
            
        Raises
        ------
        ValueError
            If the requested number of false samples exceeds the size of the available garbage collection.
        """        
        if seed is not None:
            np.random.seed(seed)

        true_ratio: float = 1.0 - false_ratio 
        size: int = int((1.0 / true_ratio) * self.SIZE_activeSeries) 
        false_count: int = size - self.SIZE_activeSeries
        
        if false_count > self.SIZE_garbageCollection:
            raise ValueError(f"Not enough garbage in collection; "
                             f"required: {false_count}, available: {self.SIZE_garbageCollection}")
        
        # get true and false grids 
        false_idx: np.ndarray = np.random.choice(np.arange(self.SIZE_garbageCollection),
                                                 false_count,
                                                 replace=False) 
        false_grids: np.ndarray = self.garbageCollection[false_idx]
        
        true_idx = np.random.permutation(self.SIZE_activeSeries)
        true_grids: np.ndarray = self.activeSeries[true_idx]

        # mix classes and build labels
        X_ = np.vstack(( false_grids, true_grids ))
        y_ = np.concatenate((
            np.zeros(false_grids.shape[0], dtype=np.uint8),
            np.ones(true_grids.shape[0], dtype=np.uint8)
        ))
    
        return _split_dataset(X_, y_, 
                              stratify_by=y_, 
                              train_ratio=train_ratio, 
                              seed=seed)

        # return self._split_dataset(X_, y_, 
        #                            stratify_by=y_, 
        #                            train_ratio=train_ratio, 
        #                            seed=seed)

    
    def split_multiclass(self, 
                         train_ratio=0.8, 
                         seed=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the currently active multiclass grid collection into training and testing subsets.
        
        This is applicable only when a multiclass activation series is active,
        where each grid has an associated label from multiple classes.
        
        Parameters
        ----------
        train_ratio : float, optional (default=0.8)
            Fraction of data to assign to the training set.
            
        seed : int or None, optional (default=None)
            Random seed for reproducibility.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            `(X_train, X_test, y_train, y_test)` arrays for multiclass classification.
        
        Raises
        ------
        ValueError
            If the current activation status is not 'multiclass'.
        AssertionError
            If the size of active series and labels differ.
        """        
        
        if self.activationStatus != "multiclass":
            raise ValueError(f"Muliclass series not activated; current status: {self.activationStatus}")
        assert self.SIZE_activeSeries == self.SIZE_labels
        return _split_dataset(self.activeSeries, self.labels, 
                              train_ratio=train_ratio, 
                              seed=seed)
#        return self._split_dataset(self.activeSeries, self.labels, train_ratio=train_ratio, seed=seed)


    
    def split_puzzle(self, train_ratio=0.8, seed=None): 
        """
        Split the currently active puzzle dataset into training and testing subsets,
        stratified by the number of blanks in the puzzles.
        
        This method is intended for Sudoku puzzle-solving tasks where the complexity
        (e.g., number of blanks) is a meaningful stratification factor.
        
        Parameters
        ----------
        train_ratio : float, optional (default=0.8)
            Proportion of data assigned to training.
            
        seed : int or None, optional (default=None)
            Random seed for reproducibility.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            `(X_train, X_test, y_train, y_test)` arrays with stratification.
        
        Raises
        ------
        ValueError
            If the puzzle series activation prefix is not found in the current dataset.
        AssertionError
            If the size of active series and labels differ.
        """        
        if "puzzle" not in self.prefix:
            raise ValueError(f"Puzzle series not activated; current status: {self.prefix + self.activationStatus}")
        assert self.SIZE_activeSeries == self.SIZE_labels
        z_ = np.array(self.__k_blanks)
        return _split_dataset(self.activeSeries, self.labels, 
                              stratify_by=z_, 
                              train_ratio=train_ratio, 
                              seed=seed)
#        return self._split_dataset(self.activeSeries, self.labels, stratify_by=z_, train_ratio=train_ratio, seed=seed)
    

    
    # def _split_dataset(self,
    #                    X: np.ndarray, 
    #                    y: np.ndarray,
    #                    stratify_by: Optional[np.ndarray] = None,
    #                    train_ratio: float = 0.8,
    #                    seed: Optional[int] = None
    #                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
    #     """
    #     General utility method to split datasets into training and testing sets,
    #     optionally stratifying by a given label or metadata array.
        
    #     Parameters
    #     ----------
    #     X : np.ndarray
    #         Dataset samples, e.g., Sudoku grids or feature arrays.
            
    #     y : np.ndarray
    #         Corresponding labels or target values.
            
    #     stratify_by : np.ndarray or None, optional (default=None)
    #         Array used for stratified sampling; must align with X and y in length.
    #         If None, stratification is not applied.
            
    #     train_ratio : float, optional (default=0.8)
    #         Fraction of data to use for training; remainder used for testing.
            
    #     seed : int or None, optional (default=None)
    #         Random seed for reproducibility of splits.
        
    #     Returns
    #     -------
    #     Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    #         `(X_train, X_test, y_train, y_test)` splits of the data.
    #     """                
    #     if seed is not None:
    #         np.random.seed(seed)
        
    #     perm_idx = np.random.permutation(len(X))
    #     X_ = X[perm_idx]
    #     y_ = y[perm_idx]
    #     z_ = stratify_by[perm_idx] if stratify_by is not None else None
    
    #     return train_test_split(
    #         X_, y_,
    #         test_size=(1.0 - train_ratio),
    #         random_state=seed,
    #         stratify=z_
    #     )
    



##########################################################################################
##                                                                                      ##
##                             T H E  (current)  E N D                                  ##     
##                                                                                      ##
##########################################################################################

##########################################################################################   
    
intro = ("\n\n\t\t====================================================\n"
         "\t\t*                                                  * \n"
         "\t\t*                  PSEUDO_Q  v2.1                  * \n"
         "\t\t*                                                  * \n"
         "\t\t*            @author: AlexPfaff (2023)             * \n"
         "\t\t*                                                  * \n"
         "\t\t*            The syntax of Sudoku Grids            * \n"
         "\t\t*                                                  * \n"
         "\t\t*              part 1b: Infrastructure             * \n"
         "\t\t*                   GridCollections                * \n"
         "\t\t*                                                  * \n"
         "\t\t==================================================== ")


if __name__ == '__main__': 
    print(intro)

    



