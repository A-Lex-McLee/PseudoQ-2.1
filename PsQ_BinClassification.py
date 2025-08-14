#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 21:01:53 2025

@author: alexanderpfaff    
"""


#import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten  
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.metrics import SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


import numpy as np

from PsQ_GridCollection import GridCollection as GC
from tqdm import tqdm



# create a simple cnn model that takes vectorized sudoku grids as input
# binary classifier, labels: valid grid (1) -- invalid grid (0)
def make_gridModel(one_hot: bool = False) -> Sequential: 
        
    gshape = (9, 9, 1) if not one_hot else (9, 9, 9)
    model = Sequential([
        Input(shape=gshape),
        Conv2D(32, kernel_size=3, activation='relu', padding='same'),
        Conv2D(64, kernel_size=3, activation='relu', padding='same'), ###### just for testin . . .  padding='same'
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(
        optimizer='adam',    
        loss='sparse_categorical_crossentropy',
        metrics=[SparseCategoricalAccuracy(name='accuracy')
                 # SparseTopKCategoricalAccuracy(k=1, name='top1_acc')   ####
    ]) 
    return model
    



class Binary_CNN_Classifier:
    """ Class that generates models (of same architecture) to be trained on distinct, but partially related datasets;
        {grids} x horizontal permutation series based on the same vertical permutation series,
        1 x the permutation series itself 
        n x random grid collections with no obvious / or explicit geometric relation (i.e. do not belong to some same series other than being valid grids)

        - trains a model m for each dataset m, and evaluates models on their own -- train & test -- datasets
        - comparison: models predict on the other datasets; 
          expectation for model i and dataset j: should predict all as 'false' (0) -- even those that have the label 'true' (1)
    """
    def __init__(self, 
                 grids: int = 5,
                 double: bool = False,
                 oneHot: bool = False
                 ) -> None:
        
        print("Generate a (valid) random grid, and produce 3!^8 = 1679616 permutations (= vertical series)")
        self.gc = GC.from_scratch()
        self.models: list = [make_gridModel() 
                             for t in range(grids)] 
        self.datasets: list = []
        self.candidateGrids = np.random.choice(range(self.gc.SIZE_collection), grids)  # random index to pick grids for horizontal permutation
        
        self._shape = (-1, 9, 9, 1) if not oneHot else (-1, 9, 9, 9)

        self.one_hot = oneHot

        self._prepData()
        
        self.modelStatus = ["horizontal" for i in range(grids)]
        
        self.models[0].summary()
        self.v_model = make_gridModel()        # for the vertical series
        self.rnd_model = make_gridModel()      # for the random series
        self.weird_model = make_gridModel()    # for the weird series (arbitrary ints; most likely not valid grids)

        self.early_stop = EarlyStopping(
            monitor='val_loss',              
            patience=3,  
            restore_best_weights=True,
            verbose=1
        )

        self.reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,          
            patience=2,        
            min_lr=1e-5,
            verbose=1
        ) 


    
    def _prepData(self) -> None:
        print()
        for grd in tqdm(self.candidateGrids): 
            self.gc.clear_all()
            self.gc.activate_horizontalSeries(grd)
            self.gc.makeFalseGrids_fromCurrent_seq() 
            self.gc.makeFalseGrids_arbitrary(how_many=363000) 

            dataset = self.gc.split_binary()
            
            dataset = list(dataset)
            dataset[0] = dataset[0].reshape(self._shape)
            dataset[1] = dataset[1].reshape(self._shape)

            dataset = tuple(dataset)
            self.datasets.append(dataset)
                    
    
    def fit(self):
        for idx, (X_train, _, y_train, _) in tqdm(enumerate(self.datasets)):
            print(f"\nFitting model {idx}\n-----------\n")
            self.models[idx].fit(X_train, 
                                 y_train, 
                                 epochs=10, 
                                 batch_size=256, 
                                 validation_split=0.2,
                                 verbose= 2,
                                 callbacks = [self.early_stop, self.reduce_lr] 
                                 )
            print()

    
    def evaluateAll(self): 
        for idx, (X_train, X_test, y_train, y_test) in enumerate(self.datasets):
            print(f"Model {idx}: -- {self.modelStatus[idx]} \n")
            print("===============")
            model = self.models[idx]
            
            self.evaluateGridModel(model=model, 
                                   X_train=X_train, 
                                   X_test=X_test,
                                   y_train=y_train,
                                   y_test=y_test)
    


    def evaluateGridModel(self, model, X_train, X_test, y_train, y_test):
            
        y_pred = model.predict(X_train) 
        y_pred = np.argmax(y_pred, axis=1)
        out = y_pred == y_train
        print()
        model.evaluate(X_train, y_train, batch_size=32, verbose=2) 
        print()
        print(f"Correctly predicted (train): {sum(out)}  out of  {out.shape[0]}  ==  {round(sum(out) / out.shape[0], 5)*100}%")

        y_pred = model.predict(X_test) 
        y_pred = np.argmax(y_pred, axis=1)
        out = y_pred == y_test
        print()
        model.evaluate(X_test, y_test, batch_size=32, verbose=2) 
        print()
        print(f"Correctly predicted (test): {sum(out)}  out of  {out.shape[0]}  ==  {round(sum(out) / out.shape[0], 5)*100}%")
        print("= " * 33)
        print()
        print()




    def cross_evaluate(self, mod_nr: int = 0):
        # get the actual X, y ...
        model = self.models[mod_nr] 
        print(f"Model {mod_nr} -- {self.modelStatus[mod_nr]}:\n")
        print("===============")
        for idx in range(len(self.datasets)):
            if idx == mod_nr:
                continue
            print(f"Testing dataset {idx} -- {self.modelStatus[idx]}:\n")
            
            X_true, X_false, y_true, y_false = self.split_datasets(self.datasets[idx])
            
            y_pred_true = model.predict(X_true)
            y_pred_true = np.argmax(y_pred_true, axis=1)
            out_true = y_pred_true == y_true
            
            y_pred_false = model.predict(X_false)
            y_pred_false = np.argmax(y_pred_false, axis=1)
            out_false = y_pred_false == y_false
            
            print() 
            
            model.evaluate(X_true, y_true, batch_size=32, verbose=2) 
            model.evaluate(X_false, y_false, batch_size=32, verbose=2) 
            
            print()
            print(f"Falsely predicted as 'true': {sum(out_true)}  out of  {out_true.shape[0]}  ==  {round(sum(out_true) / out_true.shape[0], 5)*100}%")
            print("- " * 33)
            print()
            
            print(f"Correctly predicted as 'false': {sum(out_false)}  out of  {out_false.shape[0]}  ==  {round(sum(out_false) / out_false.shape[0], 5)*100}%")
            print("- " * 33)
            print()

    def split_datasets(self, dataset): 
        X_train, X_test, y_train, y_test = dataset
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        idx_false = np.where(y == 0)[0]
        idx_true = np.where(y == 1)[0]
        
        X_true = X[idx_true]
        y_true = y[idx_true]
        X_false = X[idx_false]
        y_false = y[idx_false]
        
        return X_true, X_false, y_true, y_false

    def fit_verticalSeries(self):
        self.v_model.compile()
        self.gc.clear_all()
        self.gc.activate_verticalSeries()     
        self.gc.makeFalseGrids_fromCurrent_seq() 
        self.gc.makeFalseGrids_arbitrary(how_many=1600000)

        dataset = self.gc.split_binary()
        
        dataset = list(dataset)
        dataset[0] = dataset[0].reshape(self._shape)
        dataset[1] = dataset[1].reshape(self._shape)

        dataset = tuple(dataset)
        self.datasets.append(dataset)
        self.modelStatus.append(self.gc.activationStatus)
        self.models.append(self.v_model)
        
        X_train, _, y_train, _ = dataset

        self.v_model.fit(
            X_train, y_train, 
            epochs=10, 
            batch_size=256, 
            validation_split=0.2,
            verbose= 2 )
        print()




    def fit_randomSeries(self, how_many = 500000):
        self.rnd_model.compile()

        self.gc.clear_all()
        self.gc.activate_randomSeries(how_many=how_many, series=7)
        self.gc.makeFalseGrids_fromCurrent_seq() 
        self.gc.makeFalseGrids_arbitrary(how_many=how_many)
                
        dataset = self.gc.split_binary()
        
        dataset = list(dataset)
        dataset[0] = dataset[0].reshape(self._shape)
        dataset[1] = dataset[1].reshape(self._shape)

        dataset = tuple(dataset)
        self.datasets.append(dataset)
        self.modelStatus.append(self.gc.activationStatus)
        self.models.append(self.rnd_model)
        
        X_train, _, y_train, _ = dataset

        self.rnd_model.fit(
            X_train, y_train, 
            epochs=10, 
            batch_size=256, 
            validation_split=0.2,
            verbose= 2 )
        print()


    def fit_weirdSeries(self, how_many = 500000):
        self.weird_model.compile()

        X_weird = np.random.randint(low=1, high=10, size=(how_many, 81))

        gc = GC(X_weird) 
        gc.activate_thisSeries("weird")   
        gc.makeFalseGrids_fromCurrent_seq() 
        gc.makeFalseGrids_arbitrary(how_many=how_many)
        
        dataset = gc.split_binary()
        
        dataset = list(dataset)
        dataset[0] = dataset[0].reshape(self._shape)
        dataset[1] = dataset[1].reshape(self._shape)

        dataset = tuple(dataset)
        self.datasets.append(dataset)
        self.modelStatus.append(gc.activationStatus)
        self.models.append(self.weird_model)
        
        X_train, _, y_train, _ = dataset

        self.weird_model.fit(
            X_train, y_train, 
            epochs=10, 
            batch_size=256, 
            validation_split=0.2,
            verbose= 2 )
        print()





####################




























