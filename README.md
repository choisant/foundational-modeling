# Foundational modeling
Project where I explore data driven inference with different algorithms.

## Toy experiment
The toy experiment explores important properties of the algorithms I have studied. Feel free to browse the notebooks, but I suggest starting with the ´introduction.ipynb´ notebook and then looking at the ´generating_process.ipynb´ notebook.

### Apptainer container
This container is used to enable the usage of the R-packages used in this project. It includes a new version of Ubuntu as well as an installation of R and some packages needed to run the 'inferno' software. To use it, make sure Apptainer is installed, then build the container with the command

´´´
apptainer build env.sif apptainer.def 
´´´
To install the R-package, open the container and install the package in R, either directly from github or from a local copy.

## Sources of information

The internet is broken and it is hard to find good sources of information.
