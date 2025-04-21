# PythonProgram_DAEs
The Python program includes two sources files (main.py &amp; NSmodel.py) designed to do all the computations for the paper titled "DAEs-driven Dynamics and North-South Protection of IPRs" (by Hwan C. Lin). Main tasks are to solve a North-South dynamical system of differential algebraic equations (DAEs) and compute intertemporal welfare effects.

The Python program was coded using Anaconda Python 3.12.7.

To run this Python program, one should save the two source files into the same folder of your computer. The file of main.py is the driver, while the file of NSmodel.py declares a Python class including class members and class functions.

Specifically, the Python program accomplishes the following:

1) Solve the North-South dynamical system of DAEs with the Python solver "scipy.integrate.solve_bvp" for a nonlnear stable manifold and the time paths of variables that evolve in transition to steady state;
2) Compute steady state equlibirums for various scenarios;
3) Compute intertemporal welfare effects for the North and the South when Southern IPRs are tightened to match the Northen standard.
4) Robustness checks on the coefficient of creative destruction.

The Python program includes detailed remarks in order for the code to be reader-friendly.
