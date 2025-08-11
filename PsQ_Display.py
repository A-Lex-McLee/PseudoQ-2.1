#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 00:08:41 2025

@author: alexanderpfaff
"""

from PsQ_GridCollection import GridCollection as GC 
from PsQ_Grid import Grid, col, row, box_col, box_row, generate_validGrid # box,
from PsQ_Solver_CNN import GridSolver

from tensorflow import keras
import numpy as np

import sys
import io

from PyQt5.QtWidgets import ( 
    QDialog, QInputDialog, QFileDialog,  QLineEdit, QPushButton,  QMessageBox,  QTextEdit, 
    QApplication, QWidget, QGridLayout, QHBoxLayout, QVBoxLayout, QLabel, QCheckBox, # QSizePolicy 
    QSplitter
    )

from PyQt5.QtGui import QColor, QPalette, QFont, QIntValidator, QRegExpValidator, QFontMetrics
from PyQt5.QtCore import Qt, QRegExp  #, QTimer

import matplotlib
matplotlib.use("Agg")  # disable GUI backend
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure





# three strategies to make sure lineedit has been evaluated !!! check and compare!
# def on_permute_button_clicked(self):
#     self.enforce_pos_text()  # call it explicitly
#     # proceed with action
    
#     # give focus policy 
#     btn.setFocusPolicy(Qt.StrongFocus) 
    
    
#     self.some_lineedit.clearFocus()  # force it to lose focus manually





class SudokuMainWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        self._solver: keras.Model = None
        self._blankedGrid = None
        self._blankedDataset = None 
        # self._gc_active = False

        # ===== Set up window =====
        self.setWindowTitle("PseudoQ -- Main Display")
        self.setGeometry(100, 100, 900, 600)  # Window size + position 
        
        # ===== Grid Widget =====
        self.grid = SudokuGrid(base_number=3)
        self.grid.setFixedSize(500, 500)  # Fixed size (important for grid)
        self._set_background(self.grid, QColor(210, 210, 220))  # Slightly bluish gray

        # ===== Create main horizontal layout =====
        main_layout = QHBoxLayout(self) 
        main_layout.addStretch( )  
        
        # ===== Left panel ===== 
        self.left_panel = LeftPanel(parent=self)
        # self.left_panel.hide()  # for later use!!!!
        
        self.left_panel.setFixedWidth(500)  # Bounded resizing possible
        self._set_background(self.left_panel, QColor(180, 180, 180))  # Light gray for now 

        btn_layout = QVBoxLayout()
        # Hide/Show button
        self.btn_toggle_left = QPushButton("Hide")
        self.btn_toggle_left.clicked.connect(self._toggle_left_panel)
        self.left_panel_visible = True  # Track visibility state

        # check grid button
        self.btn_check = QPushButton("IsValid")
        self.btn_check.clicked.connect(self._isValidGrid)   
        
        self.btn_zeroCheck = QPushButton("ZeroCheck")        
        self.btn_zeroCheck.clicked.connect(self._gridCheckZero)   
        self.btn_zeroCheck.setToolTip("checks whether the grid is valid so far,\n"
                                      "i.e. every digit occurs no more than once\n"
                                      "in every row, column and box")
        
        # demo solve button
        self.btn_demoSolve = QPushButton("Demo")
        self.btn_demoSolve.clicked.connect(self._demoGame)   
        self.btn_demoSolve.setToolTip("uses a pretrained CNN model\nto solve Sudoku puzzle step by step")
        
        # generate new grid button
        self.btn_new = QPushButton("NewGrid")
        self.btn_new.clicked.connect(self._newGrid) 
        
        # open editor
        self.btn_editor = QPushButton("PyEditor")
        self.btn_editor.clicked.connect(self._open_pyEditor) 
        
        # close button
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close) 
        
        btn_layout.addWidget(self.btn_toggle_left)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_check)
        btn_layout.addWidget(self.btn_zeroCheck)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_demoSolve)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_new)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_editor)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_close)

        # ===== Right panel =====
        self.right_panel = QWidget()
        self.right_panel.setFixedWidth(200)
        self._set_background(self.right_panel, QColor(180, 180, 180))        

        main_layout.addWidget(self.left_panel)  
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.grid)
        main_layout.addWidget(self.right_panel) 

        self.setLayout(main_layout)
        self.main_layout = main_layout 
         
        
    def _set_background(self, widget, color: QColor):
        """Apply background color to a widget."""
        palette = widget.palette()
        palette.setColor(QPalette.Window, color)
        widget.setAutoFillBackground(True)
        widget.setPalette(palette) 
        
    def _toggle_left_panel(self):
        if self.left_panel_visible:
            self.main_layout.removeWidget(self.left_panel)
            self.left_panel.setParent(None)  # Detach from parent to fully collapse
            self.btn_toggle_left.setText("Show")
            self.left_panel_visible = False 
            
        else:
            self.main_layout.insertWidget(0, self.left_panel)
            self.btn_toggle_left.setText("Hide")
            self.left_panel_visible = True        
            
    def _newGrid(self):
        self.grid._grid.generate_rndGrid()
        grd = self.grid._grid.grid_toArray(self.grid._grid.SIZE)
        self.grid.insert(grd) 
        self.left_panel.btn_update.click() 
        
    def _isValidGrid(self):
        ok: bool = self.grid._grid.is_valid() 
        gridval = "valid" if ok else "invalid"
        QMessageBox.information(self, "Grid Check", f"Current grid is {gridval}.") 
        
    def _gridCheckZero(self):
        ok: bool = self.grid._grid.gridCheckZero() 
        gridval = "still valid" if ok else "invalid"
        QMessageBox.information(self, "Grid Check Zero", f"Current grid is {gridval}.") 
                
    def _demoGame(self): 
        k_blanks, ok = QInputDialog.getInt(self, "Input Needed", "How many blanks?\n(1-64)", min=1, max=64)
        if ok: 
            return GameDialog(k_blanks=k_blanks, parent=self)
        

    def _open_pyEditor(self):
        editor = PythonEditor(self)
        editor.exec_()



##########################################################
class GameDialog(QDialog):
    def __init__(self, k_blanks: int, parent=None):
        super().__init__(parent)  
        num_suff = "s" if k_blanks > 1 else ""
        self.setWindowTitle(f"really cool AI Sudoku solver for grid with {k_blanks} blank{num_suff}") 
        
        self.setGeometry(100, 100, 450, 300)  # Window size + position 
        self.setFixedSize(450, 300)
        
        self.parent = parent
        self.player = None 
        self.k_blanks = k_blanks
        self._keepGrid = False

        layout = QHBoxLayout()
        layout_setting = QVBoxLayout()
        
        self.btn_getModel = QPushButton("Load Model")
        self.btn_getDataset = QPushButton("Load Dataset")
        self.btn_externalGrid = QPushButton("Grid_fromDataset")
        self.btn_rndGrid = QPushButton("Grid_random")  
        
        self.btn_getModel.clicked.connect(self._set_Solver)
        self.btn_getModel.clicked.connect(self._updateStatus)
        
        self.btn_getDataset.clicked.connect(self._get_blankedData_fromFile)
        self.btn_getDataset.clicked.connect(self._updateStatus)
        
        self.btn_externalGrid.clicked.connect(self._get_gridFromFile)
        self.btn_externalGrid.clicked.connect(self._updateStatus)
        
        self.btn_rndGrid.clicked.connect(self._get_rndBlankedGrid)
        self.btn_rndGrid.clicked.connect(self._updateStatus)
        
        self.label_mod_active = QLabel("Model ")
        self.label_grid_active = QLabel("Grid ")
        self.label_dataset_active = QLabel("Dataset ")
        
        layout_setting.addWidget(self.btn_getModel)
        layout_setting.addWidget(self.label_mod_active) 
        layout_setting.addStretch(1)
        layout_setting.addWidget(self.btn_getDataset)
        layout_setting.addWidget(self.label_dataset_active) 
        layout_setting.addStretch(1)
        layout_setting.addWidget(self.btn_externalGrid)
        layout_setting.addWidget(self.btn_rndGrid)
        layout_setting.addWidget(self.label_grid_active)
        
        layout.addLayout(layout_setting)
        layout.addStretch(1)

        layout_play = QVBoxLayout()

        self.label = QLabel("") 
        metrics = QFontMetrics(self.label.font())
        char_width = metrics.horizontalAdvance("0")  # width of one character
        self.label.setMinimumWidth(char_width * 21) 
        line_height = metrics.lineSpacing()  # height of one line in the current font
        self.label.setMinimumHeight(line_height * 4)  # space        
        
        
        self.btn_next = QPushButton("Next Cell")
        self.btn_next.clicked.connect(self.next_item) 
        self.btn_next.setEnabled(False)

        checkbox = QCheckBox('get (g)rid of result?', self)
        checkbox.setChecked(False)
        checkbox.stateChanged.connect(self._checkbox_keepResult)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject) 

        layout_play.addWidget(checkbox)
        layout_play.addStretch(1) 
        layout_play.addWidget(self.label)
        layout_play.addWidget(self.btn_next, alignment=Qt.AlignHCenter) 
        layout_play.addStretch(2) 
        layout_play.addWidget(self.btn_cancel, alignment=Qt.AlignHCenter) 
        
        layout.addLayout(layout_play)
        
        self.setLayout(layout) 
        self._updateStatus()
        self.exec_()
        
            
    def _checkbox_keepResult(self, state): 
        # print(state) 
        self._keepGrid = bool(state) 
        
        
    def next_item(self):  
                
        if self.parent._solver is None or self.parent._blankedGrid is None:
            return 
        if self.player is None:
            self.player = GridSolver(self.parent._solver, 
                                     self.parent._blankedGrid, 
                                     self) 
        resultGrid = self.player.nextMove(self.parent._blankedGrid) 
        self._set_blankedGrid(resultGrid)  
        
        if not self.parent.grid._grid.gridCheckZero():
            QMessageBox.warning(self, "gameover", "G A M E  O V E R !") 
            self._resetGame()
            return
            
        if self.parent._blankedGrid.min() > 0:
            QMessageBox.information(self, "solved!","Congratulations, have a biscuit!")
            self._resetGame()
            
        
    def _resetGame(self):        
        self.player = None
        self.parent._blankedGrid = None
        self._updateStatus() 
        self.btn_next.setEnabled(False)
                

    def _set_Solver(self): 
        path = self._get_Solver()
        if path:
            self.parent._solver = keras.models.load_model(path)
            print("Selected:", path)
        else:
            if self.parent._solver is None:
                QMessageBox.warning(
                    self,
                    "NO MODEL SELECTED!",
                    "You have not selected a solver --\nwhich makes solving slightly complicated ..."
                )


    def _get_Solver(self):
        # app = QApplication.instance() or QApplication(sys.argv)  --> safety check for use outside widget
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select a Keras Model File",
            ".",                      # start directory
            "Keras Model Files (*.keras);;All Files (*)"
        )
    
        return file_path  # Returns "" if canceled
    
            
 
    def _get_gridFromFile(self):
        if self.parent._blankedDataset is None: 
            self._get_blankedData_fromFile()

        if self.parent._blankedDataset is None: 
            return     
        data = self.parent._blankedDataset.copy().reshape(-1, 
                                                          self.parent.grid.dimension,  
                                                          self.parent.grid.dimension) 
        zero_counts = (data == 0).sum(axis=(1, 2))   
        mask = zero_counts == self.k_blanks 
        candidates = data[mask] 
        idx = np.random.randint(candidates.shape[0]) 
        grid = candidates[idx].flatten() 
        self._set_blankedGrid(grid) 



    def _get_blankedData_fromFile(self):
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Select a blanked Dataset File",
            ".",                      # start directory
            "Numpy Dataset Files (*.npy);;All Files (*)"
        )
    
        if path: 
            self.parent._blankedDataset = np.load(path)
                        
        else:
            if self.parent._blankedDataset is None:
                QMessageBox.warning(
                    self,
                    "NO DATA SELECTED!",
                    "You have not selected a grid --\nwhich makes playing slightly complicated ..." 
                )

    
    def _get_rndBlankedGrid(self): 
        size = self.parent.grid.basenumber**4 
        grid, _ = generate_validGrid(base_number=self.parent.grid.basenumber) 
        grid = grid.reshape(size,) 
        rnd_idx = np.random.choice(np.arange(size), self.k_blanks, replace=False) 
        grid[rnd_idx] = 0

        self._set_blankedGrid(grid)


    def _set_blankedGrid(self, grid):        
        self.parent.grid._grid.insert(grid, automatic_check=False)
        self.parent.grid.insert(grid.flatten()) 
        self.parent._blankedGrid = grid.reshape(self.parent.grid.dimension, 
                                                self.parent.grid.dimension) 
        

    def _updateStatus(self):  
        self.label_mod_active.setText(f"Model activated: {self.parent._solver is not None}")
        self.label_grid_active.setText(f"Grid activated: {self.parent._blankedGrid is not None}") 
        self.label_dataset_active.setText(f"Dataset activated: {self.parent._blankedDataset is not None}") 

        if not (self.parent._solver is None or self.parent._blankedGrid is None):
            self.btn_next.setEnabled(True) 
            self.label.setText("Press 'Next Cell' \nto solve this amazing puzzle"
                               "\n\n\t. . . . or die trying!")
            

        
    # TODO -- some more exit logic
    def _handle_exit(self): 
        if not self._keepGrid:
            self.parent.grid._grid.generate_rndGrid() 
            grid = self.parent.grid._grid.grid_toArray().flatten()
            self.parent.grid.insert(grid) 
        self.parent.left_panel._updateGrid_pos()
        self.parent.left_panel._updateGrid_recoder()
        self.parent._blankedGrid = None 
        
    def closeEvent(self, event): 
        event.accept()
        self._handle_exit()
        super().closeEvent(event)

    def reject(self):
        self._handle_exit()
        super().reject()

        
        
        
        
 
###################################################################################
class SudokuGrid(QWidget):
    def __init__(self, base_number=3):
        super().__init__()
        self.basenumber = base_number
        self.dimension = self.basenumber * self.basenumber  # Grid dimension
        self.cells = [] 
        
        self._grid = Grid(baseNumber=base_number) 
        self._grid.generate_rndGrid() 
        
        # === Grid layout ===
        layout = QGridLayout()
        layout.setSpacing(0)  # Tight grid, spacing done via style

        font_size = {2: 28, 3: 24, 4: 16}[self.basenumber]  # Adjust font
        font = QFont("Courier New", font_size, QFont.Bold)
        
        max_val = self.dimension
        validator = QIntValidator(1, max_val)

        for i in range(self.dimension):
            _row = []
            for j in range(self.dimension):
                cell = QLineEdit()
                cell.setFont(font)
                cell.setMaxLength(2 if self.dimension > 9 else 1)
                cell.setValidator(validator)
                cell.setFixedSize(40, 40)  # May adjust later
                cell.setAlignment(Qt.AlignCenter)
                cell.setReadOnly(True)

                # Style borders to simulate Sudoku boxes
                cell.setStyleSheet(self._cell_style(i, j))
                layout.addWidget(cell, i, j)
                _row.append(cell)
            self.cells.append(_row)

        self.setLayout(layout) 
        
        self.insert(self._grid.grid_toArray()) 
        

    def _cell_style(self, i, j):
        """Returns CSS style string to draw Sudoku-style borders."""
        thick = 5
        thin = 1
        b = self.basenumber

        # Border logic: thicker borders between boxes
        top = thick if i % b == 0 else thin
        left = thick if j % b == 0 else thin        
        right = thick if j % b == 2 else thin
        bottom = thick if i % b == 2 else thin

        return f"""
            QLineEdit {{
                border-top: {top}px solid #53673E;
                border-left: {left}px solid #53673E;
                border-right: {right}px solid #53673E;
                border-bottom: {bottom}px solid #53673E;
                background-color: #F2F2F2;
            }}
        """

    def insert(self, flat_grid):
        """Insert values from a flat array into the grid (read-only by default)."""
        assert len(flat_grid) == self.dimension * self.dimension
        for idx, val in enumerate(flat_grid):
            i, j = divmod(idx, self.dimension)
            cell = self.cells[i][j]
            cell.setReadOnly(False)
            cell.setText(str(val) if val else "")
            cell.setReadOnly(True)

    def setReadOnly(self, flag: bool):
        """Toggle read-only state for all cells."""
        for _row in self.cells:
            for cell in _row:
                cell.setReadOnly(flag)

    def getVal(self, coord):
        """Get value from (i, j) or linear index."""
        if isinstance(coord, int):
            i, j = divmod(coord, self.dimension)
        else:
            i, j = coord
        return self.cells[i][j].text()

    def setVal(self, coord, val):
        """Temporarily unlock a cell, set value, and relock."""
        if isinstance(coord, int):
            i, j = divmod(coord, self.dimension)
        else:
            i, j = coord
        cell = self.cells[i][j]
        cell.setReadOnly(False)
        cell.setText(str(val))
        cell.setReadOnly(True)





###################################################################################
class LeftPanel(QWidget):
    def __init__(self, parent: QWidget):
        super().__init__()
        outer_layout = QHBoxLayout()
        outer_layout.setSpacing(30)  
        
        self.parent = parent 
                
        self.button_style = """
            QPushButton {
                background-color: #B5BFA1;
                border: 2px solid #5A6D4F;
                color: #2F332B;
                font-weight: bold;
                padding: 6px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #A5B391;
            }
        """        
        self.lineedit_style = """
            QLineEdit {
                background-color: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #F3F3E8, stop:1 #E0E0D0
                );
                border: 1px solid #9A9A7F;
                border-radius: 4px;
                padding: 4px 6px;
                font-size: 14px;
                color: #333;
            }
        """

        settings = self._make_setCol(parent)
        buttons = self._make_transCol(parent)
        
        # left grid related attributes
        parent._axis_keys = {'c' : col, 
                             'r' : row, 
                             'C' : box_col, 
                             'R' : box_row} 
        parent._g_axis = 'c'
        parent._g_pos = 1
        parent._recoder = "123456789" 
        parent._current_permutations = parent.grid._grid.permutePos() 
        parent._is_abc: bool = False 
        parent._perm_idx = 1
        
        outer_layout.addStretch(2)
        outer_layout.addWidget(buttons)
        outer_layout.addStretch(2)
        outer_layout.addWidget(settings)
        outer_layout.addStretch(2)
        self.setLayout(outer_layout)
        self.setAutoFillBackground(True)
#        self.setStyleSheet("background-color: #ff0000;") 

        self.btn_update.click()


    def _make_transCol(self, parent) -> QVBoxLayout:
        # === Section 1: transformations: geometry & permutations 
        col_layout = QVBoxLayout() 
        col_layout.setSpacing(25)  # 20 px between widgets

        label = QLabel("Transformations\n  ")
        label.setStyleSheet("font-weight: bold; font-size: 18px; color: #333;")
        col_layout.addWidget(label)
        col_layout.addStretch(1)
        
        # permute button
        self.btn_permute = QPushButton("Permute")
        self.btn_permute.setStyleSheet(self.button_style) 
        col_layout.addWidget(self.btn_permute)  
        self.btn_permute.clicked.connect(self.enforce_pos_text)
        self.btn_permute.clicked.connect(self._permuteGrid)

        col_layout.addStretch(3)
                
        # abc button
        self.btn_alphabetize = QPushButton("Alphabetize")
        self.btn_alphabetize.setStyleSheet(self.button_style) 
        col_layout.addWidget(self.btn_alphabetize) 
        self.btn_alphabetize.clicked.connect(self._to_abc)
        
        # general recoder
        self.btn_recode = QPushButton("Recode")
        self.btn_recode.setStyleSheet(self.button_style) 
        col_layout.addWidget(self.btn_recode) 
        self.btn_recode.clicked.connect(self.btn_update_rec.click)
        self.btn_recode.clicked.connect(self._recode)
        
        # rotate button
        self.btn_rotate = QPushButton("Rotate")
        self.btn_rotate.setStyleSheet(self.button_style) 
        col_layout.addWidget(self.btn_rotate) 
        self.btn_rotate.clicked.connect(self.btn_update_pos.click)
        self.btn_rotate.clicked.connect(self._rotate)
        
        # rotate button        
        self.btn_diaflect = QPushButton("Diaflect")
        self.btn_diaflect.setStyleSheet(self.button_style) 
        col_layout.addWidget(self.btn_diaflect) 
        self.btn_diaflect.clicked.connect(self.btn_update_pos.click)
        self.btn_diaflect.clicked.connect(self._diaflect)
        
        container = QWidget()
        container.setLayout(col_layout)
        container.setStyleSheet("background-color: #D6D8C9; border-radius: 8px;")
        return container
        
        


    def _make_setCol(self, parent) -> QVBoxLayout:
        # === Section 2: coordinate & recoding settings
        col_layout = QVBoxLayout() 
        col_layout.setSpacing(25)   

        label = QLabel("Current settings\n   ")
        label.setStyleSheet("font-weight: bold; font-size: 18px; color: #333;")
        col_layout.addWidget(label)
        
        ## coordinates
        coordinates = QVBoxLayout()
        coordinates.setSpacing(2)  

        label3 = QLabel("Axis") 
        label3.setStyleSheet("font-weight: bold; font-size: 16px; color: #333;")
        coordinates.addWidget(label3) 

        # axis entry
        self.axis_text = QLineEdit() 
        axis_validator = QRegExpValidator(QRegExp("[cCrR]"))
        self.axis_text.setValidator(axis_validator)
        self.axis_text.setText("c")
        self.axis_text.setStyleSheet(self.lineedit_style)
        coordinates.addWidget(self.axis_text)
        
        label4 = QLabel("Position") 
        label4.setStyleSheet("font-weight: bold; font-size: 16px; color: #333;")
        coordinates.addWidget(label4) 

        # position entry
        self.pos_text = QLineEdit()  
        
        def enforce_pos_text():
            axis = self.axis_text.text()
            pos = self.pos_text.text()
        
            # Determine allowed values
            if axis in ['C', 'R']:
                allowed = {'1', ''}
            else:
                allowed = {'1', '2', '3'}
        
            if pos not in allowed:
                self.pos_text.setText('1')  # Hard reset to default
        
        self.enforce_pos_text = enforce_pos_text
        self.pos_text.editingFinished.connect(enforce_pos_text)
        self.pos_text.setText("1")
        self.pos_text.setStyleSheet(self.lineedit_style)
        coordinates.addWidget(self.pos_text)  
        
        label1 = QLabel("\nAxis:\t\tPos:") 
        label1.setStyleSheet("font-size: 14px; color: #333;")
        coordinates.addWidget(label1) 
        label2 = QLabel("'c' - col\t\t1-3\n"
                        "'r' - row\t\t1-3\n"
                        "'C' - box_col\t1\n"
                        "'R' - box_row\t1") 
        label2.setStyleSheet("font-size: 14px; color: #333;")
        coordinates.addWidget(label2) 
        col_layout.addLayout(coordinates)

        recode_layout = QVBoxLayout()
        recode_layout.setSpacing(10)  
        recode_label = QLabel("Recoder") 
        recode_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #333;")
        recode_layout.addWidget(recode_label) 

        # recoder entry 
        self.recoder_text = QLineEdit()  
        def validate_recoder_text():
            text = self.recoder_text.text()
            if (
                len(text) != 9
                or not text.isdigit()
                or len(set(text)) != 9
            ):
                self.recoder_text.blockSignals(True)  # Prevent recursion
                QMessageBox.warning(
                    parent,
                    "Invalid Input",
                    f"Invalid recoder!\nPlease enter {parent.grid.dimension} distinct digits (1–{parent.grid.dimension})."
                )
                self.recoder_text.setText("123456789")
                self.recoder_text.blockSignals(False)
        
        self.recoder_text.editingFinished.connect(validate_recoder_text)
        self.recoder_text.setText("123456789")
        self.recoder_text.setStyleSheet(self.lineedit_style)
        recode_layout.addWidget(self.recoder_text)        
        label1 = QLabel(f"   --> {parent.grid._grid.DIMENSION} distinct digits") 
        label1.setStyleSheet("font-size: 14px; color: #333;")
        recode_layout.addWidget(label1) 
        col_layout.addLayout(recode_layout) 

        
        
        col_layout.addStretch(1)

        # update button(s)
        self.btn_update = QPushButton("Update") 
        self.btn_update_pos = QPushButton() 
        self.btn_update_pos.clicked.connect(enforce_pos_text) 
        self.btn_update_pos.clicked.connect(self._updateGrid_pos) 
        self.btn_update.clicked.connect(self.btn_update_pos.click)

        self.btn_update_rec = QPushButton()  
        self.btn_update_rec.clicked.connect(validate_recoder_text)                  
        self.btn_update_rec.clicked.connect(self._updateGrid_recoder)  
        self.btn_update.clicked.connect(self.btn_update_rec.click) 

        self.btn_update.setStyleSheet(self.button_style) 
        col_layout.addWidget(self.btn_update) 
        col_layout.addStretch(2)

        container = QWidget()
        container.setLayout(col_layout)
        container.setStyleSheet("background-color: #E8E8E8; border-radius: 8px;")
        return container
        

    def _diaflect(self) -> None:
        if self.parent._is_abc:
            return          
        if not self.parent.grid._grid.is_valid(): 
            QMessageBox.warning(self, "Grid Check", "No transformation on invalid grid.")
            return
        self.parent.grid._grid.diaflect()
        grd = self.parent.grid._grid.grid_toArray(self.parent.grid._grid.SIZE)
        self.parent.grid.insert(grd) 
        self.btn_update.click() 
        self.parent._perm_idx = 1
        

    def _rotate(self) -> None:
        if self.parent._is_abc:
            return 
        if not self.parent.grid._grid.is_valid(): 
            QMessageBox.warning(self, "Grid Check", "No transformation on invalid grid.")
            return
        self.parent.grid._grid.rotate()
        grd = self.parent.grid._grid.grid_toArray(self.parent.grid._grid.SIZE)
        self.parent.grid.insert(grd) 
        self.btn_update.click() 
        self.parent._perm_idx = 1
        

    def _recode(self) -> None: 
        if not self.parent.grid._grid.is_valid(): 
            QMessageBox.warning(self, "Grid Check", "No transformation on invalid grid.")
            return        
        recoder = self.recoder_text.text() 
        self.parent.grid._grid.recode(recoder) 
        grd = self.parent.grid._grid.grid_toArray(self.parent.grid._grid.SIZE)
        self.parent.grid.insert(grd) 
        self.btn_update.click() 
        self.parent._is_abc = False
        self.parent._perm_idx = 1


    def _to_abc(self) -> None: 
        if self.parent._is_abc:
            return 
        if not self.parent.grid._grid.is_valid(): 
            QMessageBox.warning(self, "Grid Check", "No transformation on invalid grid.")
            return        
        self.parent.grid._grid.to_abc_Grid() 
        grd = self.parent.grid._grid.canonical_abc_key()
        self.parent.grid.insert(grd) 
        self.parent._is_abc = True
            
        
    def _permuteGrid(self) -> None:  
        if self.parent._is_abc:
            return 
        if not self.parent.grid._grid.is_valid(): 
            QMessageBox.warning(self, "Grid Check", "No transformation on invalid grid.")
            return        
        if self.parent._perm_idx >= len(self.parent._current_permutations): 
            self.parent._perm_idx = 0 
                    
        grd = self.parent._current_permutations[self.parent._perm_idx] 
        self.parent.grid.insert(grd)
        self.parent.grid._grid.insert(grd) 
        self.parent._perm_idx += 1

            
    def _updateGrid_pos(self) -> None:
        """ Explicitly updates the settings as specified in left panel: pos, axis. """
        self.parent._g_axis = self.axis_text.text()
        self.parent._g_pos = int(self.pos_text.text())
        pos = self.parent._g_pos 
        coord = self.parent._axis_keys[self.parent._g_axis]
        self.parent._current_permutations = self.parent.grid._grid.permutePos(pos=pos, grdCoord=coord) 
        
    def _updateGrid_recoder(self) -> None:
        """ Explicitly updates the settings as specified in left panel: recoder. """
        self._recoder = self.recoder_text.text()
        



#######################################################################################

class PythonEditor(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent) 

        self.setGeometry(75, 75, 750, 500)  # Window size + position 
                
        self.setWindowTitle("PseudoQ -- Python Editor") 
        self.setModal(True)

        self._gc_active = False


        main_layout = QHBoxLayout(self)

        btn_GC_layout = QVBoxLayout() 
        
        self.btn_GC_init = QPushButton("GridCollection")
        self.btn_GC_init.clicked.connect(self._init_GC)

        self.btn_GC_activate = QPushButton("Activate Series")
        self.btn_GC_activate.clicked.connect(self._activate_series)

        self.btn_GC_garbage = QPushButton("Make Garbage")
        self.btn_GC_garbage.clicked.connect(self._makeGarbage)

        self.btn_GC_dataset = QPushButton("Get Dataset")
        self.btn_GC_dataset.clicked.connect(self._getDataset)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)

        btn_GC_layout.addWidget(self.btn_GC_init)
        btn_GC_layout.addWidget(self.btn_GC_activate)
        btn_GC_layout.addWidget(self.btn_GC_garbage)
        btn_GC_layout.addWidget(self.btn_GC_dataset)
        btn_GC_layout.addStretch(1)
        btn_GC_layout.addWidget(self.btn_close)
        
        main_layout.addLayout(btn_GC_layout)


        # --- Python editor layout: vertical ---
        python_layout = QVBoxLayout()

        # Splitter to hold editor + console vertically
        splitter = QSplitter(Qt.Vertical)

        # --- Code editor ---
        self.editor = QTextEdit()
        self.editor.setPlainText("# Python Editor for internal experimantation \n"
                                 "# PS: this is a preliminary version with reduced functionality \n"
                                 "#     try to keep your code simple -- runtimewise \n"
                                 "print('PseudoQ') \n"
                                 # "import matplotlib.pyplot as plt\n"
                                 # "plt.plot([1,2,3],[4,5,6])\n"
                                 # "plt.title('Test Plot')\n"
                                 # "plt.show()"
                                 )         
        self.editor.setTabStopDistance(4 * self.editor.fontMetrics().horizontalAdvance(' '))
        splitter.addWidget(self.editor)

        # --- Console output ---
        self.console = QTextEdit()
        self.console.setReadOnly(True) 
        self.console.setTabStopDistance(4 * self.console.fontMetrics().horizontalAdvance(' '))
        splitter.addWidget(self.console)

        splitter.setSizes([300, 150])
        python_layout.addWidget(splitter)

        # --- Button bar ---
        btn_layout = QHBoxLayout()
        run_btn = QPushButton("Run")
        run_btn.clicked.connect(self.run_code)
        clear_btn = QPushButton("Clear Output")
        clear_btn.clicked.connect(lambda: self.console.clear())
        btn_layout.addWidget(run_btn)
        btn_layout.addWidget(clear_btn)
        btn_layout.addStretch()
        python_layout.addLayout(btn_layout) 
        
        
        main_layout.addLayout(python_layout) 
        
        
        # # --- Matplotlib figure display ---
        # self.figure = Figure()
        # self.canvas = FigureCanvas(self.figure)
        # python_layout.addWidget(QLabel("Plot Output:"))
        # python_layout.addWidget(self.canvas)

    def run_code(self):
        code = self.editor.toPlainText()

        # Redirect stdout/stderr
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_buf, stderr_buf

        # # Clear previous plot
        # self.figure.clear()

        # # Replace plt.show() with a hook that draws to our canvas
        # def inline_show(*args, **kwargs):
        #     ax = self.figure.add_subplot(111)
        #     # Copy current state from pyplot into our Figure
        #     for fig in matplotlib._pylab_helpers.Gcf.get_all_fig_managers():
        #         src_fig = fig.canvas.figure
        #         for src_ax in src_fig.axes:
        #             dst_ax = self.figure.add_subplot(111)
        #             for line in src_ax.get_lines():
        #                 dst_ax.plot(line.get_xdata(), line.get_ydata())
        #             dst_ax.set_title(src_ax.get_title())
        #     self.canvas.draw()
        #     plt.close("all")  # clear pyplot state

        # plt.show = inline_show

        try:
            exec(code, globals())
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)

        # Restore stdout/stderr
        sys.stdout, sys.stderr = old_stdout, old_stderr

        # Append outputs to console
        out_text = stdout_buf.getvalue()
        err_text = stderr_buf.getvalue()
        if out_text:
            self.console.append(out_text)
        if err_text:
            self.console.append(f"<span style='color:red;'>{err_text}</span>") 
            
            
    def _init_GC(self):
        self._gc_init_code = ("# this is a default output -- something has gone wrong! \n" 
                              "gc GC.from_scratch() \n"
                              "print(gc) \n")  
        dlg_gridcollection = GridCollectionDialog(parent=self) 
        dlg_gridcollection.exec()
        self.editor.append("")                      
        self.editor.append(self._gc_init_code) 
        
        
    def _activate_series(self): 
        code = ("# make sure a GridCollection object gc has been instantiated \n" 
                "gc.activate_horizontalSeries() \n"
                "print(gc) \n") 
        self.editor.append("")                      
        self.editor.append(code) 
        
        
    def _makeGarbage(self):  
        code = ("# make sure a GridCollection object gc has been instantiated \n"
                "\tgc.makeFalseGrids_cardinality(how_many=123456) \n"
                "\tprint(gc) \n")  
        self.editor.append("")                      
        self.editor.append(code) 
        

    def _getDataset(self): 
        code = ("# make sure a GridCollection object gc has been instantiated \n"
                "X_train, X_test, y_train, y_test = gc.split_binary() \n")
        self.editor.append("")                      
        self.editor.append(code) 
        

    def _close(self):
        super().reject() 
        
        
        
        

##########################################################
class GridCollectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)  
        self.setWindowTitle("Initialize GridCollection") 
        
#        self.setGeometry(100, 100, 350, 240)  # Window size + position 
        self.setFixedSize(250, 300)
        
        layout = QVBoxLayout() 
        
        self.parent = parent
        
        self.btn_fromScratch = QPushButton("GC_fromScratch")
        self.btn_fromScratch.clicked.connect(self._gc_fromScratch) 
        self.btn_fromScratch.clicked.connect(self.close) 

        checkbox = QCheckBox('+ diaflect', self)
        checkbox.setChecked(False)
        checkbox.stateChanged.connect(self._checkbox_double)

        self.btn_fromSQL = QPushButton("GC_fromSQL")
        self.btn_fromSQL.clicked.connect(self.gc_getPath) 
        self.btn_fromSQL.clicked.connect(self.close) 

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.close) 

        layout.addWidget(self.btn_fromScratch, alignment=Qt.AlignHCenter)
        layout.addWidget(checkbox, alignment=Qt.AlignHCenter)
        layout.addStretch(1) 
        layout.addWidget(self.btn_fromSQL, alignment=Qt.AlignHCenter)
        layout.addStretch(2) 
        layout.addWidget(self.btn_cancel, alignment=Qt.AlignHCenter) 
        
        self._double = False
        self.setLayout(layout) 
        # self.exec_()
        

            
    def _checkbox_double(self, state): 
        self._double = bool(state) 

    def _gc_fromScratch(self):
        try:
            self.parent.gc = GC.from_scratch(double=self._double) 
            self.parent._gc_init_code = (f"gc = GC.from_scratch(double={self._double}) \n"
                                         "print(gc) \n") 
            self.parent._gc_active = True 
        except Exception as e:
            self.parent._gc_init_code = f"{e}"
            

    def gc_getPath(self): 
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Select a GridCollection sql DB-File",
            ".",                      # start directory
            "SQLite Files (*.db *.sqlite *.sqlite3);;All Files (*)"
        )
                
        if path: 
            try:
                self.parent.gc = GC.from_sql(db_name=path) 
                self.parent._gc_init_code = (f"gc = GC.from_sql(db_name='{path}') \n"
                                             "print(gc) \n") 
                self.parent._gc_active = True 
            except Exception as e:
                self.parent._gc_init_code = f"{e}"

        
    # TODO -- some more exit logic
    def _handle_exit(self): 
        pass
        
    def closeEvent(self, event): 
        event.accept()
        self._handle_exit()
        super().closeEvent(event)

    def reject(self):
        self._handle_exit()
        super().reject()

        



###################################        
###################################        



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SudokuMainWindow()
    window.show()
    sys.exit(app.exec_())



###################################                
###################################        
        









