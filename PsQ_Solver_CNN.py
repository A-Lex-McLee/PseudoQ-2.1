#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 15:28:05 2025

@author: alexanderpfaff
"""

import numpy as np
from tensorflow import keras
from keras import Model, Input
from keras.layers import Embedding, Conv2D, Activation, BatchNormalization #, Reshape, MultiHeadAttention, LayerNormalization   #, LSTM, Bidirectional, Dense, Reshape, Flatten, Lambda

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from PsQ_Grid import Grid
from PsQ_GridCollection import GridCollection as GC

from typing import Tuple # List, , Literal, Optional
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm 




def make_CNNModel() -> keras.Model: 
    """
    Creates a full (2D) Convolutional Neural Network (CNN); this architecture  
    processes and recognizes geometric fetures and structural dependencies, and
    is therefore highly suitable to train on (2D matrix) Sudoku grids. 
    Labels & output are themselves full specified 9 x 9 Sudoku grids.

    Returns
    -------
    model : keras.Model
        Full CNN to train on blanked Sudoku grids and output "filled" grids. 

    """
    vocab_size = 9  # digits 1-9
    embedding_dim = 16

    inputs = Input(shape=(9, 9), dtype='int32')  # (batch, 9, 9)

    # Learnable embedding for each digit 0-9 (0 = masked)
    x = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, mask_zero=False)(inputs)  # → (batch, 9, 9, embed_dim)

    # Convolutional feature extraction -- power3 vs. standard: e.g. 64
    x = Conv2D(81, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    
    x = Conv2D(135, (3, 3), padding='same', activation='relu')(x) 
    
    x = Conv2D(180, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 
   
    x = Conv2D(243, (3, 3), padding='same', activation='relu', dilation_rate=2)(x)     

    # Output: predict 1 of 9 digits at each cell (softmax over depth)
    x = Conv2D(vocab_size, (1, 1), padding='same')(x)
    outputs = Activation('softmax')(x)  # shape: (batch, 9, 9, 9)

    model = Model(inputs, outputs)
    model.summary()

    optimizer = Adam(
        learning_rate=0.0012,     # Good default starting point
        beta_1=0.9,               # Decay rate for the first moment (momentum)
        beta_2=0.999,             # Decay rate for the second moment (adaptive rate scaling)
        epsilon=1e-7              # Small constant for numerical stability
    )

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model




class Grid_CNN_Solver:
    """ Sudoku solver  based on a full CNN model.  """
    def __init__(self, 
                 data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                 ) -> None:  
        """
        Constructor: expects a qudrupel of np.ndarrays; this format corresponds to
        the return value of sklearn train_test_split (x_train, x_test, y_train, y_test). 
        
        Stores the data (x_train, x_test) and labels (y_train, y_test). 
        Allows inherent evaluation (keras.model.eval)

        Parameters
        ----------
        data : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Train and test data, format: (x_train, x_test, y_train, y_test).

        """
         
        self._prep_splitData(data)
        
        self.maskedGrids = np.concatenate([self.X_train, self.X_test]) 
        self.labels = np.concatenate([self.y_train, self.y_test]) 

        assert self.maskedGrids.shape[0] == self.labels.shape[0]

        self.model = make_CNNModel()


    @property 
    def gridsize(self) -> int:
        return self.maskedGrids.shape[1] * self.maskedGrids.shape[2]
    
    @property
    def datasize(self) -> int:
        return self.maskedGrids.shape[0] 
    
    @property
    def history(self) : #-> Model.history.history: 
        try:
            return self.model.history
        except Exception as e:
            raise NotImplementedError(f"Model has not been trained yet: {e}")


    def _prep_splitData(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):         
        self.X_train, self.X_test, y_train, y_test = (arr.reshape(-1, 9, 9) for arr in data)
        self.y_train, self.y_test = (y - 1 for y in (y_train, y_test))


    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the full dataset splits: (X_train, y_train, X_test, y_test)"""
        return self.X_train, self.y_train, self.X_test, self.y_test


    def fit(self):

        early_stop = EarlyStopping(
            monitor='val_loss',              
            patience=2,  
            restore_best_weights=True,
            verbose=2
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.51,          
            patience=2,        
            min_lr=1e-5,
            verbose=2
        )

        X_train, y_train, _, _ = self.get_data()

        print(X_train.shape, X_train.min(), X_train.max())
        print(y_train.shape, y_train.min(), y_train.max())

        self.history = self.model.fit(X_train, y_train,
                                    batch_size=256, #  128
                                    epochs=15,
                                    validation_split= 0.27,
                                    callbacks=[early_stop, reduce_lr],
                                    shuffle=True)
                                

    def evaluate(self): 
        self.model.evaluate(self.X_test, self.y_test)
        
    def evaluate_full(self): 
        self.model.evaluate(self.maskedGrids, self.labels)
        
        
    
    def save_model(self, filename: str = "solver") -> None:
        filename = filename + ".keras"
        self.model.save(filename) 
        
    def save_data(self, filename: str = "data") -> None:
        data = self.maskedGrids
        filename = filename + ".npy"
        np.save(filename, data)

    def save_labels(self, filename: str = "labels") -> None:
        labels = self.labels
        filename = filename + ".npy"
        np.save(filename, labels)


    def gimme_k_blanks(self, k):
        X = self.maskedGrids.copy()
        zero_counts = (X == 0).sum(axis=1)
        return np.where(zero_counts == k)[0]



class SudokuPlayer:  
    """
    AI-based Sudoku solver based on CNN model trained on blanked grids.
    
    Employs 'Best-Argmax-Wins' strategy: 
        run model.predict() on a given grid with k blanks; model predics full grid,
        i.e. 81 probability distributions. Predictions for cells with a nonzero digit
        are discarded; among the real predictions, each of which an argmax (over 
        probabilities for 9 digits), only the value with the highest probability
        is chosen, and inserted into the grid. 
        Recursive procedure: The resulting grid with k-1 blanks is passed into 
        model.predict(), same selection (best argmax) and insertion, until k = 0.
        
        Internally, the grid is checked for validity after every insertion.
    """
    def __init__(self, model, k_blanks, data): 
        self.model = model
        self.data = data
        self._grid = self.pick_k_blanks_grid(k_blanks)
        self.k = k_blanks 
        
    @property 
    def grid(self):
        return self._grid  
    
    @grid.setter 
    def grid(self, grd):
        self._grid = grd
    


    def pick_k_blanks_grid(self, k = 42): 
        zero_counts = (self.data == 0).sum(axis=(1, 2))   
        mask =  zero_counts == k 
        candidates = self.data[mask].copy()
        idx = np.random.randint(candidates.shape[0])
        return candidates[idx] 
    
    
    def _find_most_confident_prediction(self, x_pred, x_input):
        """ >> best argmax wins!  """
        # Create mask for positions where x_input == 0, then broadcast to shape (9, 9, 9)
        mask = (x_input == 0)[:, :, None]                     # shape (9, 9, 1)
        mask = np.broadcast_to(mask, x_pred.shape)           # shape (9, 9, 9)
    
        masked_probs = np.where(mask, x_pred, -np.inf)       # ignore positions with filled digits
        flat_index = np.argmax(masked_probs)
        i, j, d = np.unravel_index(flat_index, (9, 9, 9))
        return i, j, d + 1   # convert class index (0–8) to digit (1–9)
    

    def play_the_game(self, grid: np.array):
        if grid.min() > 0:
            return grid
    
        y_preds = self.model.predict(grid[None, ...])
        pred = y_preds[0]                      
        i, j, val = self._find_most_confident_prediction(pred, grid)
    
        grid[i, j] = val
        return ((i, j), val), grid


    def lets_play(self):
        g = Grid()
        game = True
        
        while game:
            if (self._grid == 0).sum() > 0:
                print(self._grid)
                nextMove, self._grid = self.play_the_game(self._grid)
            else:
                print("Congratulations! ") 
                self._grid = self.pick_k_blanks_grid(self.k)
                break 

            input("next move: ")
            
            print(f"Inserted: {nextMove}")
            g._quick_insert(self._grid.flatten())
            print("next move: ")
            g.showGrid()
            gok = g.gridCheckZero()
            print(gok)
            print()
            if not gok:
                print("Game Over!")
                game = False
            


##########################################################################################################################
##########################################################################################################################


###################################
###################################
##                               ##
##    version for use in  GUI!   ##
##                               ##
##################################
class GridSolver:  
    """
    AI-based Sudoku solver based on CNN model trained on blanked grids.
    
    Employs 'Best-Argmax-Wins' strategy: 
        run model.predict() on a given grid with k blanks; model predics full grid,
        i.e. 81 probability distributions. Predictions for cells with a nonzero digit
        are discarded; among the real predictions, each of which an argmax (over 
        probabilities for 9 digits), only the value with the highest probability
        is chosen, and inserted into the grid. 
        Recursive procedure: The resulting grid with k-1 blanks is passed into 
        model.predict(), same selection (best argmax) and insertion, until k = 0.
        
        Internally, the grid is checked for validity after every insertion.
    """
    def __init__(self, model, grid, parent): 
        self.model = model
        self._grid = grid,
        self.parent = parent
        
    @property 
    def grid(self):
        return self._grid  
    
    @grid.setter 
    def grid(self, grd):
        self._grid = grd
    
    
    def _find_most_confident_prediction(self, x_pred, x_input):
        """ >> best argmax wins!  """
        # Create mask for positions where x_input == 0, then broadcast to shape (9, 9, 9)
        mask = (x_input == 0)[:, :, None]                     # shape (9, 9, 1)
        mask = np.broadcast_to(mask, x_pred.shape)           # shape (9, 9, 9)
    
        masked_probs = np.where(mask, x_pred, -np.inf)       # ignore positions with filled digits
        flat_index = np.argmax(masked_probs)
        i, j, d = np.unravel_index(flat_index, (9, 9, 9))
        return i, j, d + 1   # convert class index (0–8) to digit (1–9)
    

    def nextMove(self, grid: np.array):
        if grid.min() == 0:    
            y_preds = self.model.predict(grid[None, ...])
            pred = y_preds[0]                      
            i, j, val = self._find_most_confident_prediction(pred, grid)        
            grid[i, j] = val
        return grid


    def lets_play(self):
        g = Grid()
        game = True
        
        while game:
            if (self._grid == 0).sum() > 0:
                print(self._grid)
                nextMove, self._grid = self.play_the_game(self._grid)
            else:
                print("Congratulations! ") 
                self._grid = self.pick_k_blanks_grid(self.k)
                break 

            input("next move: ")
            
            print(f"Inserted: {nextMove}")
            g._quick_insert(self._grid.flatten())
            print("next move: ")
            g.showGrid()
            gok = g.gridCheckZero()
            print(gok)
            print()
            if not gok:
                print("Game Over!")
                game = False
            




















