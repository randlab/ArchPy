# ArchPy
A hierarchical stochastic geological modeling tool in Python


## Installation

ArchPy can be installed with pip :

```
pip install geoarchpy
```

OR 

ArchPy can be installed with 
```
pip install .
```
when in the main directory.

Alternatively, it is possible to add ArchPy path directly in the python script with sys :
```
sys.path.append("path where ArchPy is") 
```
and then import ArchPy.

## Requirements
The following python packages are absolutely necessary:
   - [Geone](https://github.com/randlab/geone)
   - matplotlib
   - numpy
   - SciPy
   - sklearn
   - pandas
   - shapely < 2.0

These are not required but highly recommanded.
   - PyVista, ver. = 0.32 (ArchPy does not support 0.33 for the moment).
   - pyyaml (for export uses)
   - Rasterio (to use rasters)

 ## Examples
 There is some example notebooks :
 - 01_basic : a folder where simple and basics ArchPy functionnalities are described 
 - 02_3D_ArchPy : a complete 3D ArchPy model example
 - 03_Article_example : a synthetical example shown in ArchPy article
 - 04_hierarchies : an exemple with many hierarchical units to test ArchPy capabilities
 - 05_mps_surfaces : an example how to use MPS to simulate the units surfaces
 - 06_cross_validation : a notebook that present how to perform a cross-validation directly with ArchPy
 - 07_geological_map : this notebook presents how to integrate and use a geological in an ArchPy model
 - 08_inference : little guide how to use archpy inference tools to estimate surface parameters (no facies parameters for now)
 
 ## Paper
 A paper was published on the ArchPy concept and its different capabilities.
 The paper was written with the version 0.1 of ArchPy.
 It is available with the following [link](https://www.frontiersin.org/articles/10.3389/feart.2022.884075/).

 ## Contact
 For any questions regarding ArchPy, please contact me at <ludovic.schorpp@unine.ch>
