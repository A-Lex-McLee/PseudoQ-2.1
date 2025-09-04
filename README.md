# PseudoQ v2.1

### Sudoku is really fun $-$ but it's also really hard, in fact, it is NP hard! 
This does not merely concern the solving of actual Sudoku puzzles, but also the grid architecture as a whole, as observed e.g. in the attempt to generate random (valid) grids, enumerate and construct valid permutations, . . . . and specifying the grammar of valid grid sequences in a manner that does not merely amount to restating and recreating the game rules. 


**PseudoQ** is the result of the (still ongoing) project *The Syntax of Sudoku* intended to address the above issues (and more). Version 2.1 now available in this repository provides the basic infrastructure and architecture for the manipulation of individual grids, processing large collections of grids, and creating datasets for various ML/DL tasks, among others.  

$\Rightarrow$ **Create millions of labeled data in a couple of minutes!**


**Central components**: 
- **Grid** (class): instantiate, generate, manipulate and permute (individual) Sudoku grids; generate grid permutation series to be instantiated as GridCollection. 
- **GridCollection** (class):
    * static constructor methods: from_scratch, from_sql 
    * transform grid collections in various ways
    * activate collections with (or without) specific geometric properties (horizontal permutations, vertical permutations, diaflection/transposition)
    * modify / encode the active series (one-hot, alphabetize, label-recoding); create multiclass collections; create puzzle collections (with k cells blanked = value replace by 0)
    * activate garbage collection:
        - create various kinds of (collections) of grids that violate Sudoku rules in specific ways (cardinality, local distortion, off-by-X etc.)
        - create valid grids (= 'guest grids') that belong to geometric/permutation families different from the active series
    * create train/test datasets (for ML/DL models) 
        - binary classification
        - multiclass classification
        - blanked grids/puzzles to train solvers 
    * SQL API to load / store datasets <br>


<pre> ```
# illustration: create a dataset for multiclass classification task  
	
import GridCollection as GC
gc = GC.from_scratch()
gc.makeFalseGrids_arbitrary() 
gc.makeFalseGrids_cardinality()
gc.makeFalseGrids_offBy_X(how_many=345678, X=2, how_far=4)
gc.activate_multiClassSeries(classes=8)  # since make_garbage=True (default) -> + 1 garbage class = 9 classes
gc.activate_oneHotSeries()	
. . . 
print(gc)	
>>> GridCollection[ 
                     internal shape: (1679616, 81)  
                     active series: multiclass 
   	                	size: 3265920  
                     labeled series: 9 classes  
                     labels: True  
	               		size: 3265920  
                     garbage: True 
	               		total size: 545678 
 					 		guest_grids              : 0
					 		false_fromCurrent_seq    : 0
					 		false_fromCurrent_switch : 0
					 		false_cardinality        : 100000
					 		false_off_by_X           : 345678
					 		false_arbitrary          : 100000 
                     one_hot encoding: True  
                     internal abc collection: False  
                   ] 	
. . . 
data = gc.split_multiclass(train_ratio=0.7, seed=42) 
# data = (X_train, X_test, y_train, y_test)  
for dataset in data:
    print(dataset.shape)
>>> (2286143, 9, 9, 9)
>>> (979777, 9, 9, 9)
>>> (2286143,)
>>> (979777,)	

``` </pre>
<br>
<br>

**GUI**: 
- tkinter: 'PsQ_GUI' - obsolete/deprecated (just in case you have not installed PyQt)
- PyQt: 'PsQ_Display: Grid permutations, recodings, rnd grid generation; AI Sudoku solver; integrated Python editor (preliminary version)
- 
Currently, the main purpose of the GUI is to illustrate the basic functionalities of Grid and GridCollection; further functionalities will be added.


**Auxiliary classes/functions**: prerequisites for notebooks
- PsQ_BinClassification
- PsQ_Solver_CNN 


**Stand-alone file**: modular knight travels acrosse Sudoku grid 
- PsQ_ModuloKnight

**Notebooks**: DL projects with data produced by GridCollection
- psq_binary_clf.ipynb
- psq_multi_clf.ipynb
- psq_solver_player.ipynb


**Also included**: (for application, see psq_solver_player; GUI)
- solver1.keras           (pretrained Sudoku solver)
- reduced_dataset_1.npy   (dataset for solver1)


<br> 

⚠️ Note: This project uses third-party libraries (NumPy, pandas, matplotlib, PyQt5, TensorFlow / keras, sklearn, tqdm). <br>
These are not included in the repository and must be installed separately. <br>
They are subject to their own licenses.









