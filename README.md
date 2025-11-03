# ArchPy
[![Documentation Status](https://readthedocs.org/projects/archpy/badge/?version=latest)](https://archpy.readthedocs.io/en/latest/?badge=latest)
![logo](./sphinx_build/source/figures/logo_web.png)

A hierarchical stochastic geological modeling tool in Python


## Installation

ArchPy can be installed with pip :

```
pip install geoarchpy
```

This will install ArchPy and the necessary dependencies. To install all the dependencies (including optional ones) :

```
pip install geoarchpy[all]
```

ArchPy can also be installed from the source code. To do so, clone the repository and run the following command in the main directory :

```
pip install .
```

Alternatively, it is possible to add ArchPy path directly in the python script with sys :
```
sys.path.append("path where ArchPy is") 
```
and then import ArchPy. In such case, it is necessary to install all the dependencies manually.

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
   - matplotlib (tested with 3.10.7)
   - numpy (tested with 1.26.4)
   - scipy (tested with 1.16.3)
   - sklearn (tested with 1.7.2)
   - pandas (tested with 2.3.3)
   - shapely (tested with 2.1.2)

These are not required but highly recommanded. They are installed with ArchPy by default.
   - pyvista (tested with 0.46.4)
   - yaml (tested with 6.0.3)
   - rasterio (tested with 1.4.3)
   - geopandas (tested with 1.1.1)
   - ipywidgets (tested with 8.1.8)
   - flopy (tested with 3.9.5)
   
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
 - 11_modflow_coupling : an example of how to couple ArchPy with Modflow
 
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
