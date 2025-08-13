# PseudoQ v2.1

### Sudoku is really fun $-$ but it's also really hard, in fact, it is NP hard! 
This does not merely concern the solving of actual Sudoku puzzles, but also the grid architecture as a whole, as observed e.g. in the attempt to generate random (valid) grids, enumerate and construct valid permutations, . . . . and specifying the grammar of valid grid sequences in a manner that does not merely amount to restating and recreating the game rules. 


**PseudoQ** is the result of the (still ongoing) project *The Syntax of Sudoku* intended to address the above issues (and more). Version 2.1 now available in this repository provides the basic infrastructure and architecture for the manipulation of individual grids, processing large collections of grids, and creating datasets for various ML/DL tasks, among others.  


**Central components**: 
- Grid (class): instantiate, generate, manipulate and permute (individual) Sudoku grids; generate grid permutation series to be instantiated as GridCollection.
- GridCollection (class):  


This repository is part of a side project of mine called "The syntax of Sudoko grids" (some might prefer the label 'geometry' instead of 'syntax', but I'm a linguist and I call the shots here!). It provides the basic architecture to instantiate, generate, manipulate and permute Sudoku grids.




* Grid
* GridCollection
* utilFunX

* 
