Tutorials
=========

Theoretical background
----------------------

The simulations in ArchPy are performed hierarchically, from the largest geological features to the smallest.
Three main levels can be distinguished: 

   -  Units: Biggest geological features, this level corresponds to the 
      stratigraphical geological units that are distinct by their depositional time.
      They are defined by 2D surfaces that bound their bottom and top.
   -  Facies: Smaller and more scattered geological features, they represent the geological heterogeneity at a more local scale 
      and are generally simulated using 3D indicator simulations (SIS, TPGs, MPS, ...)
   -  Properties: these represent the physical properties of the subsurface such as permeability or electrical conductivity and are continuous.
      They represent a very local but important heterogeneity that has to be taken into account into physical simulations (groundwater, geophysical, transport, etc.)

All the simulation parameters and geological rules are contained into one object that we call the stratigraphic pile (SP). 
The SP obviously contains the stratigraphic units and their stratigraphical relationships (e.g. unit B is above unit A) 
but also the facies and their spatial relationships (e.g. facies A and B are inside unit A) and the properties. 
On top of that, `ArchPy` also integrates the novel capability to simulate hierarchical units, i.e. units that are themselves composed of smaller units and so on. 
This is done by defining multiples SPs that are embedded into each other. For example, we define that the unit B, that is composed of 3 smaller units (B1, B2 and B3), 
is above unit A. This is done by defining a master SP (P1) for unit A and another one for unit B. The SP of unit B (PB) contains the SPs of its subunits (B1, B2 and B3). 
Proceeding this way, we can define a hierarchy of SPs that can be as deep as we want. You can find example of these hierarchical units in this  `notebook <notebooks/1_many_hierarchical_units.html>`_
or this `one <notebooks/3D_ArchPy_example.html>`_.

To explore the capabilities of `ArchPy`, we will now present a series of tutorials that will guide you through the different steps of the simulation process.

Notebooks
---------

.. nbgallery::
    :caption: Basics
    
    notebooks/1_the_very_basics
    notebooks/2_Hard_data
    notebooks/3_filling_and_saving
    notebooks/4_load_project


.. nbgallery::
   :caption: ArchPy capabilities
    
   Hierarchy <notebooks/1_many_hierarchical_units>
   notebooks/3D_ArchPy_example
   notebooks/mps_surfaces
   notebooks/3D_ArchPy_geological_map
   notebooks/Article_example
   notebooks/x_valid_exemple1
   notebooks/inference_surfaces
   notebooks/example

