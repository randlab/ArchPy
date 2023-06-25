Introduction
============

`ArchPy` package is a tool to create, manipulate and vizualize 3D geological and petro-physical models of the subsurface.
It relies on a hierarchical approach to generate these models at different spatial scales.
The input data consists of a set of borehole data and a stratigraphic pile. 
The stratigraphic pile describes formally and in a compact manner how the model should be constructed,
such as the relations between the geological features or the modelling parameters.

ArchPy offers a variety of capabilities such as:

   - Various interpolation methods (GRF, Kriging, Spline, etc.)
   - Various simulation methods (MPS, SIS, TPGs, etc.)
   - Integration of hierarchy in the modelling process at different spatial scales
   - Input/Output capabilities to read and write borehole data, stratigraphic pile and models
   - Automatic inference of conditioning point 
   - Automatic inference of surface parameters
   - Can handle raster and shapefile data
   - Can handle 3D geological map
   - Cross-validation capabilities
   - and many more...

`ArchPy` is a pure  Swiss product (|:fondue:|), produced by the `Randlab <https://www.unine.ch/philippe.renard/de/home.html>`_ at University of Neuchâtel.
 
Installation
------------

`ArchPy` is available on `Pypi <https://pypi.org/project/geoarchpy>`_ and can be installed with pip::

   pip install geoarchpy

OR 

`ArchPy` can be installed locally with::

   pip install .


when in the main directory.

Alternatively, it is possible to add `ArchPy` path directly in the python script with sys::

   sys.path.append("path where ArchPy is") 

and then import `ArchPy`.

Requirements
------------
ArchPy has been tested with python>=3.8

ArchPy requires the following packages:

   - `Geone <https://github.com/randlab/geone>`_
   - matplotlib
   - numpy
   - SciPy
   - sklearn
   - pandas
   - shapely < 2.0
   - scikit-learn

The following packages are optional but are required for some functionalities:

   - PyVista (for 3D vizualisation)
   - pyyaml (for export uses)
   - Rasterio (to use rasters)
   - Geopandas (to use shapefile)
 
.. warning::
   **Issues with widgets**: For some reasons, widgets does not work properly with recent version of jupyter-server due to recent versions of tornardo.
   Ensure to have tornado==6.1 if you plan to use widgets and interactive functions with the matplotlib notebook backend. 


Members of the project
----------------------

ArchPy is developped, tested and supported by a group of people from the Randlab. These include:

   - Ludovic Schorpp 
   - Alexis Neven
   - Julien Straubhaar
   - Philippe Renard


How to cite
-----------

 A paper was published on the `ArchPy` concept and its different capabilities.
 The paper was written with the version 0.1 of `ArchPy`.
 It is available with the following `Link <https://www.frontiersin.org/articles/10.3389/feart.2022.884075/>`_.


Contact
-------
 
 For any questions regarding `ArchPy`, please feel free to contact me at <ludovic.schorpp@unine.ch>