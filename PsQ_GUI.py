#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 19:23:20 2025

@author: alexanderpfaff
"""

import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import simpledialog

from tensorflow import keras
import numpy as np

from PsQ_Grid import Grid, box, col, row, box_col, box_row
from PsQ_GridCollection import GridCollection as GC



# here you can insert your own model / dataset
myModel = "solver1.keras"
mySudokuData = "reduced_dataset_1.npy"





class PseudoQ_Window:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PseudoQ -- visual update ")
        self.root.minsize(1450, 640)  
        self.grid_frame: tk.Frame
        self.left_frame: tk.Frame
        self.right_frame: tk.Frame
        self.bottom_frame: tk.Frame
        
        self._prepare()
        
        self._vcmd_Dim = self.root.register(self._validate_input)
        self._vcmd_Pos = self.root.register(self._validate_posInput)
        self._vcmd_Pos_int = self.root.register(self._validate_IntInput)
        

        self._cells = self._create_grid()
        self._grid = Grid(3)
        
        self._grid.generate_rndGrid()  
#        self._grid.insert( _grd )
        self._insertGrid()
        self._permutationIdx = 1
        self._currentPermutations: tuple = self._grid.permutePos()
        self._isABC: bool = False
        
        self.model = keras.models.load_model(myModel)
        self.puzzleSet = np.load(mySudokuData)
        self._k_blanks = 42
        
        self.rotateButton()
        self.diaflectButton()
        self.nextPermutationButton()        
        self.alphabetizeButton()
        self.specifyRecoder()
        self.recodeButton()
        self.specifyDimension()
        self.specifyPosition()
        self.getPermutationsButton()
        self.gridProtocolButton()
        self.rndGridButton()
        self.fetchGameGridButton()
#        self.getGridCollButton()
        self.nextValPredictionButton()
        
        self.root.mainloop()
        
                
    def _prepare(self):
        self.root.grid_rowconfigure(0, weight=1, minsize=200)  # Top row (grid + right widgets)
        self.root.grid_rowconfigure(1, weight=0, minsize=100)  # Bottom row (future widgets)
        self.root.grid_columnconfigure(0, weight=1)  # Left column (grid)
        self.root.grid_columnconfigure(1, weight=1)  # Right column (additional widgets)
        self.root.grid_columnconfigure(2, weight=1)  # Right column (additional widgets)

        # top left section (functionalities)
        self.left_frame = tk.Frame(self.root, relief="sunken", bd=2, bg="#C1C5C0")
        self.left_frame.grid(row=0, column=0, padx=30, pady=10)

        # Grid section
        self.grid_frame = tk.Frame(self.root, relief="sunken", bd=2, bg="#9B7BB5")
        self.grid_frame.grid(row=0, column=1, padx=50, pady=40)

        self.right_frame = tk.Frame(self.root, relief="sunken", bd=2, bg="#C1C5C0")
        self.right_frame.grid(row=0, column=2, padx=30, pady=10)

        # --- Create Bottom Section (Future Widgets) ---
        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10)



    #Buttons middle 
    def rotateButton(self):
        button_rotate = tk.Button(self.left_frame, text="Rotate", command = self.rotate,
                                  font=("Arial", 18, "bold"), fg="#6D7F61", 
                                  highlightbackground="#9B7BB5", highlightcolor="#9B7BB5")
        button_rotate.grid(row=0, column=1, pady=30, padx=25)

        
    def diaflectButton(self):        
        button_diaflect = tk.Button(self.left_frame, text="Diaflect", command = self.diaflect,
                                    font=("Arial", 18, "bold"), fg="#6D7F61",
                                    highlightbackground="#9B7BB5", highlightcolor="#9B7BB5")
        button_diaflect.grid(row=1, column=1, pady=30, padx=25)
        
        
    def nextPermutationButton(self):        
        button_nextPermutation = tk.Button(self.left_frame, text="Next \nPermutation",
                                           command = self.nextPermutation, 
                                           font=("Arial", 18, "bold"), fg="#6D7F61",
                                           highlightbackground="#9B7BB5", highlightcolor="#9B7BB5")
        button_nextPermutation.grid(row=4, column=1, pady=20, padx=35)
        
        
    def alphabetizeButton(self):        
        button_nextPermutation = tk.Button(self.left_frame, text="Alphabetize",
                                           command = self.alphabetize, 
                                           font=("Arial", 18, "bold"), fg="#6D7F61", 
                                           highlightbackground="#9B7BB5", highlightcolor="#9B7BB5")        
        button_nextPermutation.grid(row=0, column=0, pady=30, padx=50)
        
        
        
#################################

    def specifyRecoder(self):
        text_recoder = tk.Text(self.left_frame, height=6, width=26,
                                 font=("Courier", 14), fg="black", bg="#9B7BB5")
        text_recoder.grid(row=2, column=0, padx=25, pady=20)        
        text_recoder.insert(tk.END, "Enter Recoder\n")
        text_recoder.insert(tk.END, 
                              " >> nine numbers from\n       {1,2,3,4,5,6,7,8,9}\n >> without repetition\n >> e.g.: 782163945")        
        text_recoder.tag_add("bold_font", "1.0", "1.16")  # Apply bold to the first part of the text
        text_recoder.tag_add("large_font", "2.0", "2.35")  # Apply larger font to the second part
        text_recoder.tag_configure("bold_font", font=("Courier", 20, "bold"))
        text_recoder.tag_configure("large_font", font=("Courier", 15))

        self.entry_gridInt = tk.Entry(self.left_frame, width=10, justify="left", font=("Courier", 16), 
                                        bg="#B8C9A6",
                                        validate='key', validatecommand=(self._vcmd_Pos_int, '%S'))
        self.entry_gridInt.grid(row=3, column=0)
                
        self.entry_gridInt.bind("<KeyRelease>", 
                                  lambda event: self._on_key_releaseInt(event, self.entry_gridInt))
        # self.entry_gridInt.insert(0, "1")

        
    def recodeButton(self):        
        button_getPermutations = tk.Button(self.left_frame, text="Recode",
                                           command = self.recode, 
                                           font=("Arial", 18, "bold"), fg="#6D7F61",
                                           highlightbackground="#9B7BB5", highlightcolor="#9B7BB5")        
        button_getPermutations.grid(row=4, column=0, pady=38, padx=50)


#################################
        
                
    def specifyDimension(self):
        text_dimension = tk.Text(self.right_frame, height=6.5, width=26,
                                 font=("Courier", 14), fg="black", bg="#A1B49C")
        text_dimension.grid(row=0, column=2, padx=25, pady=20)        
        text_dimension.insert(tk.END, "Enter Dimension\n")
        text_dimension.insert(tk.END, 
                              " possible values:\n >> r -- row\n >> c -- col\n >> R -- BoxRow\n >> C -- BoxCol")        
        text_dimension.tag_add("bold_font", "1.0", "1.16")  # Apply bold to the first part of the text
        text_dimension.tag_add("large_font", "2.0", "2.35")  # Apply larger font to the second part
        text_dimension.tag_configure("bold_font", font=("Courier", 20, "bold"))
        text_dimension.tag_configure("large_font", font=("Courier", 15))

        self.entry_dimension = tk.Entry(self.right_frame, width=2, font=("Courier", 16), 
                                        bg="#B8A1D4",
                                        validate='key', validatecommand=(self._vcmd_Dim, '%S'))
        self.entry_dimension.grid(row=1, column=2)
                
        self.entry_dimension.bind("<KeyRelease>", 
                                  lambda event: self._on_key_release(event, self.entry_dimension))
        self.entry_dimension.insert(0, "c")

        

    def specifyPosition(self):
        text_position = tk.Text(self.right_frame, height=5, width=26,
                                 font=("Courier", 14), fg="black", bg="#A1B49C")
        text_position.grid(row=3, column=2, padx=25, pady=20)        
        text_position.insert(tk.END, "Enter Position\n")
        text_position.insert(tk.END, 
                              " possible values:\n >> r/c: 1-3\n >> R/C: <1> (default)")        
        text_position.tag_add("bold_font", "1.0", "1.16")  # Apply bold to the first part of the text
        text_position.tag_add("large_font", "2.0", "2.35")  # Apply larger font to the second part
        text_position.tag_configure("bold_font", font=("Courier", 20, "bold"))
        text_position.tag_configure("large_font", font=("Courier", 15))

        self.entry_position = tk.Entry(self.right_frame, width=2, justify="left", font=("Courier", 16), 
                                        bg="#B8A1D4",
                                        validate='key', validatecommand=(self._vcmd_Pos, '%S'))
        self.entry_position.grid(row=4, column=2)
                
        self.entry_position.bind("<KeyRelease>", 
                                  lambda event: self._on_key_releasePos(event, self.entry_position))
        self.entry_position.insert(0, "1")

        
    def getPermutationsButton(self):        
        button_getPermutations = tk.Button(self.right_frame, text="Get \nPermutations",
                                           command = self.getPermutations, 
                                           font=("Arial", 18, "bold"), bg="#A1B49C", fg="#7E5A8F",
                                           highlightbackground="#A1B49C", highlightcolor="#A1B49C")        
        button_getPermutations.grid(row=5, column=2, pady=38, padx=50)



   
#########################################        
    def fetchGameGridButton(self):
          button_gridCollection = tk.Button(self.bottom_frame, text="FetchGameGrid\n for DEMO ", # load puzzle
                                             command = self.loadXGrid, 
                                             font=("Arial", 18, "bold"), bg="#A1B49C",
                                             highlightbackground="#9B7BB5", highlightcolor="#9B7BB5")        
          button_gridCollection.grid(row=0, column=1, pady=20, padx=50)

#################################################  
        
    def rndGridButton(self):
          button_rndGrid = tk.Button(self.bottom_frame, text="RandomGrid\nGenerator",
                                             command = self.generate_rndGrid, 
                                             font=("Arial", 18, "bold"), bg="#A1B49C",
                                             highlightbackground="#9B7BB5", highlightcolor="#9B7BB5")        
          button_rndGrid.grid(row=0, column=10, pady=20, padx=50)

  



    def getGridCollButton(self):
          close_button = tk.Button(self.bottom_frame, text="Get Collection",      
                                             command = self.getGridCollectionDialog,  ###
                                             font=("Arial", 18, "bold"), bg="#A1B49C",
                                             highlightbackground="#9B7BB5", highlightcolor="#9B7BB5")        
          close_button.grid(row=1, column=10, pady=20, padx=50)
  
        
    #def closeMainButton(self):        
    # close_button = tk.Button(root, text="Close", command=root.destroy)
    # close_button.pack(pady=20)
        

    def getGridCollectionDialog(self):
        
        puzzle = messagebox.askyesno("Game for the Game", "Do you want a blanked Collection?\n\n'Yes' --> Sudoku *puzzles* \n'No' --> regular GridCollection")
        if puzzle:
            self._makeBlankCollection()
            return

        series = self.ask_radio_choice("  Permutation Series   ", 
                                       "Choose Permutation Series:", 
                                       ["Horizontal Series", "Vertical Series"])
        trainTest = messagebox.askyesno("TrainTestSplit", "Create TrainTest dataset?") 
        if trainTest:
            ratio = simpledialog.askinteger("Training Ratio", "Training Ratio 10-90 (%)",
                                            minvalue=10, 
                                            maxvalue=90)



        ratio = simpledialog.askinteger("Training Ratio", 
                                        "Training Ratio 10-90 (%)",
                                        minvalue=10, 
                                        maxvalue=90) if trainTest else 0


        oneHot = messagebox.askyesno("OneHotEncoding", "To One_Hot?") 

        fileName = simpledialog.askstring("File Name", "Enter File Name") 
        if not fileName:
            return

        self._makeGridCollection(series, oneHot, trainTest, fileName, ratio)


    def _makeGridCollection(self, series, oneHot, trainTest, fileName, ratio): 
        gc = self._grid.permuteGrids(toCollection=True)
        if series == "Horizontal Series":
            gc.activate_HorizontalSeries()
        else:
            gc.activate_VerticalSeries()
        if trainTest:
            gc.makeFalseGrids_fromCurrent() 
            X_train, y_train, X_test, y_test = gc.train_test_gridSplit(train_ratio=ratio, to_oneHot=oneHot)
        
            np.save(fileName + "_X_train.npy", X_train)
            np.save(fileName + "_y_train.npy", y_train)
            np.save(fileName + "_X_test.npy", X_test)
            np.save(fileName + "_y_test.npy", y_test)
            return 
        
        outColl = gc.to_oneHot(gc.__activeSeries) if oneHot else gc.activeGrid 
        np.save(fileName + ".npy", outColl)



    def _makeBlankCollection(self): 
        fileName = simpledialog.askstring("File Name", "Enter File Name") 
        gc = self._grid.permuteGrids(toCollection=True)
        gc.activate_HorizontalSeries(42)
        gc = gc.activeGrid
        self.maskedGridCollection = np.empty((0, 81), dtype=gc.dtype) 
        self._mask_normal(gc)
        self._mask_largeGaps(gc)     
        np.save(fileName + "_psq_puzzle.npy", self.maskedGridCollection)
        del self.maskedGridCollection, gc


    def _mask_normal(self, gridcollection):
        n_samples = 1200000
        output = np.empty((n_samples, 81), dtype=gridcollection.dtype)
        
        for iteration in range(n_samples): 
            idx = iteration % 362880 
            k = iteration % 60 + 1 
            mask = np.random.choice(range(81), k, replace=False)
            grid = gridcollection[idx].copy()
            grid[mask] = 0 
            output[iteration] = grid
        self.maskedGridCollection = np.concatenate([self.maskedGridCollection, output])



    def _mask_largeGaps(self, gridcollection):
        n_samples = 1000000
        output = np.empty((n_samples, 81), dtype=gridcollection.dtype)

        for iteration in range(n_samples): 
            idx = iteration % 362880 
            low = int(60 *(3/5) + 2)
            diff = 60 - low + 1
            k = iteration % low + diff   

            mask = np.random.choice(range(81), k, replace=False)
            grid = gridcollection[idx].copy()
            grid[mask] = 0 
            output[iteration] = grid

        self.maskedGridCollection = np.concatenate([self.maskedGridCollection, output])



    def ask_radio_choice(self, title, question, options):
        result = None
    
        def on_submit():
            nonlocal result
            result = var.get()
            dialog.destroy()
    
        dialog = tk.Toplevel()
        dialog.title(title)
        dialog.grab_set()  # modal
    
        tk.Label(dialog, text=question, padx=20, pady=10).pack()
    
        var = tk.StringVar(value=options[0])  # default selection
    
        for option in options:
            rb = tk.Radiobutton(dialog, text=option, variable=var, value=option)
            rb.pack(anchor='w', padx=20)
    
        submit_btn = tk.Button(dialog, text="OK", command=on_submit)
        submit_btn.pack(pady=10)
    
        dialog.wait_window()
        return result
        



    def gridProtocolButton(self):
        button_gridProtocol = tk.Button(self.bottom_frame, text="Grid Coordinates",
                                           command = self.getProtocol, 
                                           font=("Arial", 18))        
        button_gridProtocol.grid(row=0, column=7, pady=20, padx=120)




    def _validate_IntInput(self, char) -> bool:        
        allowed_chars = "123456789"
        if char in allowed_chars:
            return True
        else:
            return False

    def _clear_and_insertInt(self, entry, new_char):
        lastPos: int = len(entry.get())-1
        entry.delete(lastPos, tk.END)
        entry.insert(tk.END, new_char)
        
    def _on_key_releaseInt(self, event, entry):
        if not event.char:
            return
        new_char = event.char
        if len(new_char) <= 9 and self._validate_IntInput(new_char):
            self._clear_and_insertInt(entry, new_char)
        else:
            lastPos: int = len(entry.get())-1
            entry.delete(lastPos, tk.END)
        

    def _validate_posInput(self, char) -> bool:        
        self.entry_position.delete(0, tk.END)
        allowed_chars = "123"
        if char in allowed_chars:
            if self.entry_dimension.get() in "CR":
                self.entry_position.insert(0, "1")
            else:
                self.entry_position.insert(0, char)
            return True
        else:
            self.entry_position.insert(0, "1")
            return False

    def _clear_and_insertPos(self, entry, new_char):
        entry.delete(0, tk.END)
        entry.insert(0, new_char)
        
    def _on_key_releasePos(self, event, entry):
        if not event.char:
            return
        new_char = event.char
        if len(new_char) == 1 and self._validate_posInput(new_char):
            self._clear_and_insert(entry, new_char)
        else:
            entry.delete(0, tk.END)
            self.entry_position.insert(0, "1")
        

    def _validate_input(self, char) -> bool:        
        self.entry_dimension.delete(0, tk.END)
        allowed_chars = "rcRC"
        if char in allowed_chars:
            self.entry_dimension.insert(0, char)
            return True
        else:
            self.entry_dimension.insert(0, "c")
            return False

    def _clear_and_insert(self, entry, new_char):
        entry.delete(0, tk.END)
        entry.insert(0, new_char)

    def _on_key_release(self, event, entry):
        if not event.char:
            return
        allowed_chars = "rcRC"
        new_char = event.char
        if len(new_char) == 1 and new_char in allowed_chars:
            self._clear_and_insert(entry, new_char)
        else:
            entry.delete(0, tk.END)
            self.entry_dimension.insert(0, "c")




    def _create_grid(self):  # to be attached to a separate widget (top left )
        cells = []
        # Create 9 boxs for the 3x3 boxes
        for boxRow in range(3):
            for boxCol in range(3):
                # Create a box for each 3x3 box
                box = tk.Frame(self.grid_frame, padx=5, pady=5, bg="#C1C5C0")  # Padding to separate boxes slightly
                box.grid(row=boxRow, column=boxCol, padx=5, pady=5)
    
                # Create 3x3 grid of Entry widgets inside the box
                for i in range(3):  # 3 rows per box
                    for j in range(3):  # 3 columns per box
                        cell = tk.Entry(box, width=2, justify="center", font=("Helvetica", 24, "bold"))  # Create Entry widget
                        cell.grid(row=i, column=j, padx=2, pady=2)  # Grid them inside the box
                        cells.append(cell)  # Add 
        return cells





    def _insertGrid(self):
        boxwiseGrid = self._grid.gridOut(box)
        for gridIdx in range(81):
            val = boxwiseGrid[gridIdx] 
            try:
                val = int(val)
            except Exception:
                val = str(val)
            self._cells[gridIdx].config(state=tk.NORMAL, fg="black" )
            self._cells[gridIdx].delete(0)
            self._cells[gridIdx].insert(0, val)
            self._cells[gridIdx].config(state=tk.DISABLED, disabledforeground="black")

        


    def diaflect(self):
        if self._isABC:
            return
        self._grid.diaflect()
        self._insertGrid()
        self.getPermutations()

    def rotate(self):
        if self._isABC:
            return
        self._grid.rotate()
        self._insertGrid()
        self.getPermutations()


    def getPermutations(self):
        if self._isABC:
            return
        if not self.entry_dimension.get() in "rcRC":
            self.entry_dimension.delete(0, tk.END)
            self.entry_dimension.insert(0, "r")
            return
        if not self.entry_position.get() in "123":
            self.entry_position.delete(0, tk.END)
            self.entry_position.insert(0, "1")
            return
        pos = 1 if not self.entry_dimension.get() in "rc" else int(self.entry_position.get()) 
        decoder = {"r": row, "c": col, "R": box_row, "C": box_col}
        func = decoder[self.entry_dimension.get()]
        self._currentPermutations = self._grid.permutePos(pos, func)
        self._permutationIdx = 1
        self.entry_position.delete(0, tk.END)
        self.entry_position.insert(0, pos)


        

    def nextPermutation(self):
        if self._isABC:
            return
        if self._permutationIdx > 5:
            self._permutationIdx = 0
        nextGrid = self._currentPermutations[self._permutationIdx]
        self._grid.insert(nextGrid)
        self._insertGrid()
        self._permutationIdx += 1



##  BIG TODO
####################################################################


    def blankGrid(self, gridarray: list) -> tuple:
        grid = [' ' if x == 0 else x for x in gridarray.flatten()]
        return np.array(tuple(grid))
        

    def loadXGrid(self): 
#        self.root.withdraw()
        self._k_blanks = simpledialog.askinteger(title="Number of Blanks", 
                                                 prompt=f"Please enter number of blanks (0-60)\n (currently: {self._k_blanks} blanks) ",
                                                 minvalue=0,
                                                 maxvalue=60,
                                                 initialvalue=self._k_blanks)
#        self.root.deiconify()
        self._currentGameGrid: np.ndarray = self._pick_k_blanks_grid(puzzles = self.puzzleSet, 
                                                                     k = self._k_blanks)  # 9 x 9 array -- ints
        self._currentGame = self.blankGrid(self._currentGameGrid)                     # list -- ints + ''


        self._currentGame = self._currentGame


        self.updateGrid() 
        

    def updateGrid(self):
        self._currentGame = self.blankGrid(self._currentGameGrid)                     # list -- ints + ''

        self.clearGrid()
        self._grid._quick_insert(self._currentGame)


        self._insertGrid()
        #self.checkGrid()



    def clearGrid(self):
        self._grid.setZero()
        self._insertGrid()


    
    def _pick_k_blanks_grid(self, puzzles: np.array, k = 42):  
        """
        Selects a game grid with k blanks at random 

        Parameters
        ----------
        puzzles : np.array
            DESCRIPTION.
        k : TYPE, optional
            DESCRIPTION. The default is 42.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        zero_counts = (puzzles == 0).sum(axis=(1, 2))   
        mask =  zero_counts == k 
        candidates = puzzles[mask].copy()
        idx = np.random.randint(candidates.shape[0])
        return candidates[idx].astype(np.uint8) 
    
    
    def _find_most_confident_prediction(self, x_pred, x_input):
        """
        Find the (i, j, d) with the highest confidence in x_pred,
        considering only those (i, j) where x_input[i, j] == 0.
        
        Parameters:
            x_pred : np.ndarray of shape (9, 9, 9)
                Model output probabilities.
            x_input : np.ndarray of shape (9, 9)
                Input Sudoku grid with 0s in cells to be filled.
    
        Returns:
            i, j, d : int
                Position (i, j) and digit d (1–9) of the most confident eligible prediction.
        """
        # Create mask for positions where x_input == 0, then broadcast to shape (9, 9, 9)
        mask = (x_input == 0)[:, :, None]                     # shape (9, 9, 1)
        mask = np.broadcast_to(mask, x_pred.shape)           # shape (9, 9, 9)
    
        masked_probs = np.where(mask, x_pred, -np.inf)       # ignore positions with filled digits
        flat_index = np.argmax(masked_probs)
        i, j, d = np.unravel_index(flat_index, (9, 9, 9))
        return i, j, d + 1   # convert class index (0–8) to digit (1–9)
    
    
    def play_the_game(self):
        grid = Grid()
        if self._currentGameGrid.min() > 0:
            grid._quick_insert(self._currentGameGrid.flatten())
            if grid.is_valid():
                messagebox.showinfo("Congratulations", "Well done, have a biscuit ;)")
            else:
                messagebox.showinfo("Loser", "You messed up, game over !)") 

            return 
    
        y_preds = self.model.predict(self._currentGameGrid[None, ...])
        pred = y_preds[0]                      
        i, j, val = self._find_most_confident_prediction(pred, self._currentGameGrid)
    
        self._currentGameGrid[i, j] = val
        self.updateGrid() 
        grid._quick_insert(self._currentGameGrid.flatten())
        if not grid.gridCheckZero():
            r_idx = grid.getRun(i + 1, j + 1)
            
            self._cells[r_idx-1].config(state=tk.NORMAL, fg="red" )

            
            messagebox.showinfo("Loser", "You messed up, game over !")
  #      return  (i, j), val, grid
    
    
    

        
    def nextValPredictionButton(self):
          button_rndGrid = tk.Button(self.bottom_frame, text="IA-Solver",  # predict next
                                             command = self.play_the_game, 
                                             font=("Arial", 18, "bold"), bg="#A1B49C",
                                             highlightbackground="#9B7BB5", highlightcolor="#9B7BB5")        
          button_rndGrid.grid(row=1, column=1, pady=20, padx=50)







    def checkGrid(self):
        self.root.attributes('-topmost', True)

        message = f"Grid is valid (so far): \n--{self._grid.gridCheckZero()}--" 
        title = "Valid Grid (so far)?" 
        
        # Show message box
        messagebox.showinfo(title, message, parent=self.root)

        # Restore normal window stacking
        self.root.attributes('-topmost', False)
     



        
    def alphabetize(self): 
        self._grid.to_abc_Grid()
        self._insertGrid()
        self._isABC = True



    def recode(self) -> None:
        self.entry_gridInt.focus()
        if not self._isABC:
            return
        if len(self.entry_gridInt.get()) != 9:
            messagebox.showerror("ERROR", f"""In order to recode the Grid,  
                                  you must enter\nEXACTLY NINE NUMBERS\n
                                  (numbers entered: {len(self.entry_gridInt.get())})""")
            return
        for i in range(1, 10):
            if self.entry_gridInt.get().count(str(i)) !=1:
                messagebox.showerror("ERROR", f"""EACH number (1-9) 
                                                 must be entered 
                                                 EXACTLY ONCE\n
                                                 ( [{i}] occurs {self.entry_gridInt.get().count(str(i))} times)""")
                return
        nextGrid = []
        for i in self.entry_gridInt.get():
            nextGrid.append(int(i))
        self._grid.recode(recoder=nextGrid) 
        self._isABC = False
        self._insertGrid()
        self.getPermutations()





    #######################################################################
    def generate_rndGrid(self):        
        goRnd = tk.messagebox.askokcancel("Warning", "Are you sure?\nCurrent grid will be lost! ") 
        if not goRnd:
            return

        self._isABC = False
        self._grid.generate_rndGrid()
        self._insertGrid()
        self.getPermutations()




    def openFile(self):
        file_path = filedialog.askopenfilename(title="Select a grid collection file", 
                                               filetypes=[("Text files", "*.txt"), ("All files", "*.*")])

        if file_path:
            print(f"Selected file: {file_path}")
        else:
            print("No file selected")
        


    def getProtocol(self):
        text_window = tk.Toplevel()
        text_window.title("Grid Coordinates")
        text_window.minsize(1207, 600)
        text_window.maxsize(1207, 600)
        text_widget = tk.Text(text_window, wrap=tk.WORD, height=35, width=300)
        text_widget.pack(padx=20, pady=20)
        
        text_widget.insert(tk.END, "GRID_as_TUPLE: \n\n< ")
        text_widget.insert(tk.END, self._grid.gridOut())
        text_widget.insert(tk.END, " > \n\n-----------------------------------------------------------\n\n")

        text_widget.insert(tk.END, "\nTHE FRAME: \n\n")
        for cell in self._grid.baseGrid:
            outStr: str = f'run: {cell.run:2};   '  \
                + f'row: {cell.row};   '            \
                + f'col: {cell.col};   '            \
                + f'box: {cell.box};   '            \
                + f'box_col: {cell.box_col};   '    \
                + f'box_row: {cell.box_row};   '    \
                + f'value: {cell.val} \n'    
            text_widget.insert(tk.END, outStr)
        def close_window():
            text_window.destroy()
        ok_button = tk.Button(text_window, text="OK", command=close_window)
        ok_button.pack(pady=10)


       


_grd = (3, 1, 7, 2, 4, 6, 5, 9, 8, 
        5, 8, 6, 7, 3, 9, 2, 1, 4, 
        4, 9, 2, 1, 8, 5, 6, 7, 3, 
        9, 3, 4, 5, 2, 8, 1, 6, 7, 
        7, 2, 8, 6, 9, 1, 4, 3, 5, 
        6, 5, 1, 4, 7, 3, 9, 8, 2, 
        8, 6, 5, 3, 1, 4, 7, 2, 9, 
        1, 7, 3, 9, 5, 2, 8, 4, 6, 
        2, 4, 9, 8, 6, 7, 3, 5, 1)



p = PseudoQ_Window()
