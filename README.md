# Overview

This is an imlpementation of a parallel algorithm to solve the Minimum Vertex Cover Problem and its Parameterized version on GPUs. This code is written using CUDA and C/C++. 

# Reference

The methods and algorithms used are discussed in the below paper :
- P. Yamout, K. Barada, A. Jaljuli, A. Mouawad, I. El Hajj. Parallel Vertex Cover Algorithms on GPUs. In Proceedings of the IEEE International Parallel & Distributed Processing Symposium (IPDPS), 2022.

Please cite this paper if you find this project useful.

# Folders
- data: Has a number of graphs used to run on the code
- src: Contains the source code of the project
- test: Has a number of python scripts used to test the code

# Instructions
All of the below commands should be executed in the src folder

- To compile:
```make```

- To compile enabling Counters (won't run efficiently):
```make USE_COUNTERS=1```

- To clean:
```make clean```

- To run:
```./output <args>```

- For help on how to configure arguments:
```./output -h ```

# Notes
- Data regarding your run will be appended as a row to the end of the a csv file Results.csv in src/Results/Results.csv
- If code is complied enabling counters then counter data about each block will be written in files in src/NODES_PER_SM and src/Counters for each graph
- Data in the src/NODES_PER_SM folder represents how many nodes from the work tree (Mentioned in Referenced Paper) each SM in your GPU solved.
- Data in the src/Counters folder represnets how much time was collectively spent on parts of the code
