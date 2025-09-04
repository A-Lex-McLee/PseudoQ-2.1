#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 17:05:47 2025

@author: alexanderpfaff   
"""

from typing import List, Optional, Tuple, Callable, Set
import time
import numpy as np
from random import shuffle



class ModularKnight:
    KNIGHT_MOVES = [
        (+2, +1), (+2, -1), (-2, +1), (-2, -1),
        (+1, +2), (+1, -2), (-1, +2), (-1, -2),
    ]

    def __init__(
        self,
        size: int = 9,
        start_pos: Tuple[int, int] = (0, 0),
        box_jump_required: bool = False,
        random_move: bool = False,
        log_step_fn: Optional[Callable[['ModularKnight'], None]] = None
    ):
        self.N = size
        self._validate_grid_size()
        self.start_pos = self._wrap(*start_pos)
        self.box_jump_required = box_jump_required
        self.random_move = random_move
        self.log_step_fn = log_step_fn or (lambda k: None)  # No-op if None
        self.reset()

    def _validate_grid_size(self):
        root = self.N ** 0.5
        if int(root + 1e-5) ** 2 != self.N:
            raise ValueError(f"Grid size N={self.N} is not a perfect square.")

    def _wrap(self, r: int, c: int) -> Tuple[int, int]:
        return r % self.N, c % self.N

    def box_index(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        r, c = pos
        box_size = int(self.N ** 0.5)
        return (r // box_size, c // box_size)

    def get_legal_moves(self, 
                        from_pos: Optional[Tuple[int, int]] = None
                        ) -> List[Tuple[int, int]]:
        if from_pos is None:
            from_pos = self.pos
        current_box = self.box_index(from_pos)
        moves = []
        for dr, dc in self.KNIGHT_MOVES:
            new_r, new_c = self._wrap(from_pos[0] + dr, from_pos[1] + dc)
            new_pos = (new_r, new_c)
            if new_pos in self.visited:
                continue
            if self.box_jump_required and self.box_index(new_pos) == current_box:
                continue
            moves.append(new_pos)
        if self.random_move:
            shuffle(moves)
        return moves

    def is_legal_knight_move(self, to_pos: Tuple[int, int]) -> bool:
        to_pos = self._wrap(*to_pos)
        return to_pos in self.get_legal_moves()

    def move_to(self, new_pos: Tuple[int, int]) -> None:
        new_pos = self._wrap(*new_pos)
        if new_pos in self.visited:
            raise ValueError(f"Cell {new_pos} already visited.")
        if not self.is_legal_knight_move(new_pos):
            raise ValueError(
                f"Illegal move from {self.pos} to {new_pos}. "
                + ("(Box constraint violated)" if self.box_jump_required else "(Standard knight move only)")
            )
        self.pos = new_pos
        self.path.append(new_pos)
        self.visited.add(new_pos)

    def _undo(self) -> None:
        if len(self.path) <= 1:
            return  # can't undo the initial step
        last = self.path.pop()
        self.visited.remove(last)
        self.pos = self.path[-1]


    # no timeout
    def find_full_tour_(self, max_depth: Optional[int] = None) -> bool:
        def backtrack() -> bool:
            if len(self.visited) == self.N ** 2:
                return True
            if max_depth is not None and len(self.visited) >= max_depth:
                return False
            for next_pos in self.get_legal_moves():
                self.move_to(next_pos)
                self._log()
                if backtrack():
                    return True
                self._undo()
            return False

        return backtrack()
    
    
    
    def find_full_tour(self, max_depth: Optional[int] = None, timeout_sec: Optional[float] = None, use_heuristic: bool = False) -> bool:
        start_time = time.time()
    
        def backtrack() -> bool:
            if len(self.visited) == self.N ** 2:
                return True
            if max_depth is not None and len(self.visited) >= max_depth:
                return False
            if timeout_sec is not None and (time.time() - start_time) > timeout_sec:
                return False
    
            moves = self.get_legal_moves()
            if use_heuristic:
                # Warnsdorff’s heuristic: fewest onward moves first
                moves.sort(key=lambda pos: len(self.get_legal_moves(from_pos=pos)))
    
            for next_pos in moves:
                self.move_to(next_pos)
                self._log()
                if backtrack():
                    return True
                self._undo()
            return False
    
        return backtrack()
    
    

    def _log(self):
        self.log_step_fn(self)

    def reset(self, start_pos: Optional[Tuple[int, int]] = None) -> None:
        if start_pos is not None:
            self.start_pos = self._wrap(*start_pos)
        self.pos = self.start_pos
        self.path: List[Tuple[int, int]] = [self.start_pos]
        self.visited: Set[Tuple[int, int]] = {self.start_pos}

    def get_path_grid(self) -> List[List[Optional[int]]]:
        grid = [[None for _ in range(self.N)] for _ in range(self.N)]
        for step, (r, c) in enumerate(self.path):
            grid[r][c] = step
        return grid

    def print_path_grid(self) -> None:
        grid = self.get_path_grid()
        for row in grid:
            line = " ".join(
                f"{cell:2}" if cell is not None else " ."
                for cell in row
            )
            print(line)
        print()

    def __str__(self) -> str:
        return (
            f"ModularKnight(pos={self.pos}, visited={len(self.visited)}/{self.N**2}, "
            f"box_jump_required={self.box_jump_required})"
        )








class TourStats:
    """
    Track detailed statistics of the knight's tour search:
    steps, backtracks, forks, retries, time elapsed.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = None
        self.end_time = None
        self.steps = 0
        self.backtracks = 0
        self.forks = 0
        self.retries = 0
        self.last_visited_count = 0

    def track_tour(self, knight: ModularKnight) -> None:
        if self.start_time is None:
            self.start_time = time.time()

        current_visited = len(knight.visited)
        # Step increments when visited increases
        if current_visited > self.last_visited_count:
            self.steps += 1
        # Backtrack detected if visited shrinks (undo)
        elif current_visited < self.last_visited_count:
            self.backtracks += 1

        # Fork: when get_legal_moves > 1 after a step
        moves_count = len(knight.get_legal_moves())
        if moves_count > 1:
            self.forks += 1

        # retries: rough estimate as backtracks + forks
        self.retries = self.backtracks + self.forks

        self.last_visited_count = current_visited

    def finalize(self):
        self.end_time = time.time()

    def summary(self) -> str:
        duration = (self.end_time - self.start_time) if self.end_time and self.start_time else 0.0
        return (
            f"TourStats:\n"
            f"  Steps taken: {self.steps}\n"
            f"  Backtracks: {self.backtracks}\n"
            f"  Forks (branching): {self.forks}\n"
            f"  Retries (approx): {self.retries}\n"
            f"  Time elapsed: {duration:.3f} seconds"
        )


class TourRunner:
    """
    Run ModularKnight tours with visualization and stats tracking.
    Supports multiple tours with varying start positions and constraints.
    """

    def __init__(
        self,
        size: int = 9,
        box_jump_required: bool = False,
        random_move: bool = False,
        use_heuristic: bool = False,
        timeout_sec: Optional[float] = None
    ):
        self.size = size
        self.box_jump_required = box_jump_required
        self.random_move = random_move
        self.use_heuristic = use_heuristic
        self.timeout_sec = timeout_sec
        self.stats = TourStats()
        self.path_grid = None

    def run_tour(self, start_pos: Tuple[int, int]) -> bool:
        self.stats.reset()

        knight = ModularKnight(
            size=self.size,
            start_pos=start_pos,
            box_jump_required=self.box_jump_required,
            random_move=self.random_move,
            log_step_fn=self._log_step
        )
        success = knight.find_full_tour(
            max_depth=None,
            timeout_sec=self.timeout_sec,
            use_heuristic=self.use_heuristic
        )
        self.stats.finalize()
        self.path_grid = np.asarray(knight.get_path_grid()) + 1
        return success

    def _log_step(self, knight: ModularKnight) -> None:
        self.stats.track_tour(knight)

    def show_visualization(self) -> None:
        print(self.path_grid)





def all_grid_coords(size=9):
    idx = np.arange(size)  # [0, 1, ..., 8]
    rows, cols = np.meshgrid(idx, idx, indexing='ij')
    # rows and cols are 9x9 arrays of row and col indices
    
    # flatten both arrays and zip into (row, col) tuples
    coords = list(zip(rows.flatten(), cols.flatten()))
    return coords

coords = all_grid_coords()




runner = TourRunner(size=9, box_jump_required=True, use_heuristic=True, timeout_sec=10)



tours = []

for coord in coords:
    print(f"Start coordinate: ({coord[0]+1, coord[1]+1})")
    success = runner.run_tour(coord)
    tours.append(success)

    if success:
        print("Tour found!")
        runner.show_visualization()
    else:
        print("No tour found within constraints.")

    print()
    # print(runner.stats.summary())
    # print()
    input("Press <Enter> to continue \n")
    print()


















