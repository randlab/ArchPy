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
   - Coupling with groundwater flow using MODFLOW and flopy
   - Integration of hierarchy in the modelling process at different spatial scales
   - Input/Output capabilities to read and write borehole data, stratigraphic pile and models
   - Automatic inference of conditioning point 
   - Automatic and manual interface for inference of surface parameters
   - Can handle raster and shapefile data
   - Can handle geological maps
   - Cross-validation capabilities
   - and many more...

`ArchPy` is a pure  Swiss product (|:fondue:|), produced by the `Randlab <https://www.unine.ch/philippe.renard/de/home.html>`_ at University of Neuchâtel.
 
Installation
------------

`ArchPy` is available on `Pypi <https://pypi.org/project/geoarchpy>`_ and can be installed with pip::

   pip install geoarchpy[all]

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
   - matplotlib (tested with 3.10.0)
   - numpy (tested with 1.26.4)
   - scipy (tested with 1.15.1)
   - sklearn (tested with 1.6.1)
   - pandas (tested with 2.2.3)
   - shapely (tested with 2.0.6)

The following packages are optional but are required for some functionalities:

   - pyvista (tested with 0.44.0)
   - yaml (tested with 6.0.2)
   - rasterio (tested with 1.4.3)
   - geopandas (tested with 1.0.1)
   - ipywidgets (tested with 8.1.5)
   - flopy (tested with 3.9.1)


Members of the project
----------------------

ArchPy is developped, tested and supported by a group of people from the Randlab. These include:

   - Ludovic Schorpp 
   - Alexis Neven
   - Julien Straubhaar
   - Philippe Renard
   - Nina Egli


How to cite
-----------

A paper was published on the `ArchPy` concept and its different capabilities.
The paper was written with the version 0.1 of `ArchPy`.
It is available with the following `Link <https://www.frontiersin.org/articles/10.3389/feart.2022.884075/>`_.


List of references using ArchPy
-------------------------------

Schorpp, L., Straubhaar, J., & Renard, P. (2024). From lithological descriptions to geological models: an example from the Upper Aare Valley. Frontiers in Applied Mathematics and Statistics, 10, 1441596 [link](https://doi.org/10.3389/fams.2024.1441596).

Neven, A., & Renard, P. (2023). A novel methodology for the stochastic integration of geophysical and hydrogeological data in geologically consistent models. Water Resources Research, 59(7). [link](https://doi.org/10.1029/2023WR034992)

Neven, A., Schorpp, L., & Renard, P. (2022). Stochastic multi-fidelity joint hydrogeophysical inversion of consistent geological models. Frontiers in Water, 4, 989440. [link](https://doi.org/10.3389/frwa.2022.989440)


Contact
-------
 
 For any questions regarding `ArchPy`, please feel free to contact me at <ludovic.schorpp@unine.ch>