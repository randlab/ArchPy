Introduction
============

ArchPy package is a tool to create, manipulate and vizualize 3D geological and petro-physical models of the subsurface.
It relies on a hierarchical approach to generate geological and petrophyiscal models at different spatial scales.
Three main levels can be distinguished: 

   -  Units: Biggest geological features, this level corresponds to the 
      stratigraphical geological units that are distinct by their depositional time.
      They are defined by 2D surfaces that bound their bottom and top.
   -  Facies: Smaller and more scattered geological features, they represent the geological heterogeneity at a more local scale 
      and are generally simulated using 3D indicator simulations (SIS, TPGs, MPS, ...)
   -  Properties: these represent the physical properties of the subsurface such as permeability or electrical conductivity and are continuous.
      They represent a very local but critical heterogeneity that has to be taken into account into physical simulations (groundwater, geophysical, transport, etc.)

ArchPy offers a variety of capabilities such as:

   - blabla
   - blabla 2

ArchPy is a pure  Swiss product (|:fondue:|), produced by the `Randlab <https://www.unine.ch/philippe.renard/de/home.html>`_ at University of Neuchâtel.
 
Installation
------------

ArchPy is available on `Pypi <https://pypi.org/project/geoarchpy>`_ and can be installed with pip::

   pip install geoarchpy

OR 

ArchPy can be installed locally with::

   pip install .


when in the main directory.

Alternatively, it is possible to add ArchPy path directly in the python script with sys::

   sys.path.append("path where ArchPy is") 

and then import ArchPy.

Requirements
------------

The following python packages are absolutely necessary:

   - `Geone <https://github.com/randlab/geone>`_
   - matplotlib
   - numpy
   - SciPy
   - sklearn
   - pandas
   - shapely < 2.0

These are not required but highly recommanded:

   - PyVista (for 3D vizualisation)
   - pyyaml (for export uses)
   - Rasterio (to use rasters)
   - Geopandas (to use shapefile)
   

Notebook examples
----------------- 

 There is some example notebooks :
   
   - 01_basic : a folder where simple and basics ArchPy functionnalities are described 
   - 02_3D_ArchPy : a complete 3D ArchPy model example
   - 03_Article_example : a synthetical example shown in ArchPy article
   - 04_hierarchies : an example with many hierarchical units to test ArchPy capabilities
   - 05_mps_surfaces : an example how to use MPS to simulate the units surfaces
   - 06_cross_validation : a notebook that present how to perform a cross-validation directly with ArchPy
   - 07_geological_map : this notebook presents how to integrate and use a geological in an ArchPy model
   - 08_inference : little guide how to use archpy inference tools to estimate surface parameters (no facies parameters for now)
   - 09_interface : little example of an interface to call an preexisting archpy model.
 
How to cite
-----------

 A paper was published on the ArchPy concept and its different capabilities.
 The paper was written with the version 0.1 of ArchPy.
 It is available with the following `Link <https://www.frontiersin.org/articles/10.3389/feart.2022.884075/>`_.

Contact
-------
 
 For any questions regarding ArchPy, please feel free to contact me at <ludovic.schorpp@unine.ch>
