# ArchPy
[![Documentation Status](https://readthedocs.org/projects/archpy/badge/?version=latest)](https://archpy.readthedocs.io/en/latest/?badge=latest)
![logo](./sphinx_build/source/figures/logo_web.png)

A hierarchical stochastic geological modeling tool in Python


## Installation

ArchPy can be installed with pip :

```
pip install geoarchpy
```

This will install all ArchPy dependencies (even the optional ones) and the package itself.

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
and then import ArchPy. In such case, it is necessary to install all the dependencies manually.

:warning: **Issues with widgets**: For some reasons, widgets does not work properly, probably due to the recent version of jupyter-server. For now these issues are not solved and widgets are not working, except partially with inline backend. 

Concerning the interactivity of the plots, it is necessary to install trame as well as some trame subpackages. This can be done with the following command :
```
pip install trame
pip install trame-vuetify
pip install trame-vtk
```

## Requirements
Works and tested with 3.8 <= python <= 3.11

The following python packages are absolutely necessary:
   - [Geone](https://github.com/randlab/geone)
   - matplotlib
   - numpy (tested with 1.26.4)
   - SciPy (tested with 1.14.1) 
   - sklearn (tested with 1.5.1) 
   - pandas (tested with 2.2.2) 
   - shapely (tested with 2.0.6)

These are not required but highly recommanded. They are installed with ArchPy by default.
   - PyVista (tested with 0.44.1)
   - pyyaml (tested with 6.0.2, for export uses)
   - Rasterio (tested with 1.3.10, to use rasters)
   - Geopandas (tested with 1.0.1, to use shapefile)
   - ipywidgets
   
 ## Examples
 There is some example notebooks :
 - 01_basic : a folder where simple and basics ArchPy functionnalities are described 
 - 02_3D_ArchPy : a complete 3D ArchPy model example
 - 03_Article_example : a synthetical example shown in ArchPy article
 - 04_hierarchies : an exemple with many hierarchical units to test ArchPy capabilities
 - 05_mps_surfaces : an example how to use MPS to simulate the units surfaces
 - 06_cross_validation : a notebook that present how to perform a cross-validation directly with ArchPy
 - 07_geological_map : this notebook presents how to integrate and use a geological in an ArchPy model
 - 08_inference : little guide how to use archpy inference tools to estimate surface parameters (no facies parameters for now) --> Note that for now, the interface is not working due to incompatibilities issues with ipywidgets.
 - 09_interface : little exemple of an interface to call an preexisting archpy model as well as drawing a new model extension.
 - 10_rotation : an example of how to create a rotated model
 
 ## Paper
 A paper was published on the ArchPy concept and its different capabilities.
 The paper was written with the version 0.1 of ArchPy.
 It is available with the following [link](https://www.frontiersin.org/articles/10.3389/feart.2022.884075/).

 ## list of references using ArchPy
 Schorpp, L., Straubhaar, J., & Renard, P. (2024). From lithological descriptions to geological models: an example from the Upper Aare Valley. Frontiers in Applied Mathematics and Statistics, 10, 1441596 [link](https://doi.org/10.3389/fams.2024.1441596).

 Neven, A., & Renard, P. (2023). A novel methodology for the stochastic integration of geophysical and hydrogeological data in geologically consistent models. Water Resources Research, 59(7). [link](https://doi.org/10.1029/2023WR034992)
 
 Neven, A., Schorpp, L., & Renard, P. (2022). Stochastic multi-fidelity joint hydrogeophysical inversion of consistent geological models. Frontiers in Water, 4, 989440. [link](https://doi.org/10.3389/frwa.2022.989440)
 
 ## Contact
 For any questions regarding ArchPy, please contact me at <ludovic.schorpp@unine.ch>
