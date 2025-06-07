"""
This module propose several functions a class to interface ArchPy with MODFLOW 6
"""

import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import geone
import geone.covModel as gcm
import geone.imgplot3d as imgplt3
import pyvista as pv
import sys
import os
import flopy as fp
import pandas as pd
import ArchPy
import ArchPy.base
import ArchPy.uppy
from ArchPy.uppy import upscale_k, rotate_point

# some functions
def mask_below_unit(T1, unit, iu=0):
    u = T1.get_unit(unit)
    
    mask_tot = np.zeros([T1.nz, T1.ny, T1.nx])
    mask_unit = T1.unit_mask(u.name, iu=iu)
    mask_tot[mask_unit.astype(bool)] = 1

    for unit_to_compare in T1.get_all_units():
        if unit_to_compare.name == unit:
            mask_unit = T1.unit_mask(unit_to_compare.name, iu=iu)
            mask_tot[mask_unit.astype(bool)] = 1
            continue
        if unit_to_compare > u:
            mask_unit = T1.unit_mask(unit_to_compare.name, iu=iu)
            mask_tot[mask_unit.astype(bool)] = 1
    
    return mask_tot

def cellidBD(idomain, layer=0):   
    
    """
    extract the cellids at the boundary of the domain at a given layer
    idomain : 3D array, idomain array which determine if a cell is active or not (1 active, 0 inactive)
    layer : int, layer on which the boundary cells are extract
    """
    lst_cellBD=[]

    for irow in range(idomain.shape[1]):
        for icol in range(idomain.shape[2]):
            if idomain[layer][irow,icol]==1:
                #check neighbours
                if np.sum(idomain[layer][irow-1:irow+2,icol-1:icol+2]==1) < 8:
                    lst_cellBD.append((layer,irow,icol))
    return lst_cellBD

def check_thk(top, botm):
    """
    check if all the cells in a modflow model (given top and botm array) which have a thickness <= 0 for and this for each layer
    input : top (the top surface) and botm (botom of each layer)
    output : lst of bool (false mean everything's okay in that specific layer !)
    """
    nlay = botm.shape[0]
    bol_lst=[]
    bol_lst.append(not ((top-botm[0])<=0).any())
    for ilay in range(nlay-1):
        bol_lst.append(not ((botm[ilay]-botm[ilay+1])<=0).any())
    return bol_lst

def array2cellids(array, idomain):

    """
    Convert an array to a list of cellids
    array : 3D array, array to convert
    idomain : 3D array, idomain array which determine if a cell is active or not (1 active, 0 inactive)
    """
    cellids = []

    idomain_copy = idomain.copy()
    idomain_copy[idomain == -1] = 0
    b = array & idomain_copy.astype(bool)
    for ilay in range(b.shape[0]):
        for irow in range(b.shape[1]):
            for icol in range(b.shape[2]):
                if b[ilay, irow, icol]:
                    cellids.append((ilay, irow, icol))
    return cellids

def plot_particle_facies_sequence(arch_table, df, plot_time=False, plot_distance=False, proportions=False,
                                   time_unit="s", resampling="d"):

    """
    Plot the facies sequence of a particle in time and distance

    Parameters
    ----------
    arch_table : :class:`base.Arch_table` object
        ArchPy table object
    df : pandas dataframe
        dataframe with the particle data. Must contain the columns "facies" or "facies_prop_i" where i is the facies id
        and "time" or "distance" or "cum_distance".
    plot_time : bool, optional
        if True, plot the facies sequence in time. The default is False.
    plot_distance : bool, optional
        if True, plot the facies sequence in distance. The default is False.    
    proportions : bool, optional
        if True, plot the proportions of each facies in time or distance. The default is False. 
        df must contain the columns "facies_prop_i" where i is the facies id.
    time_unit : str, optional
        unit of time in the column "time". The default is "s".
    resampling : str, optional
        resampling frequency for the time series. The default is "d". Check pandas documentation for more information.
    fac_time : float, optional
        factor to apply to time column in case number are too large and cause an overflow error in pandas
    """

    if plot_time and plot_distance:
        fig, ax = plt.subplots(2,1, figsize=(10, 1.5), dpi=200)
        axi = ax[0]
    elif plot_time or plot_distance:
        fig, axi = plt.subplots(1,1, figsize=(10, 0.5), dpi=200)
    plt.subplots_adjust(hspace=1.5)

    if proportions:
        colors_fa = []
        for col in df.columns:
            if col.split("_")[0] == "facies":
                id_fa = int(col.split("_")[-1])
                color_fa = arch_table.get_facies_obj(ID=id_fa, type="ID").c
                colors_fa.append(color_fa)

    if plot_time:

        if proportions:
            # df.set_index("time").iloc[:, -len(colors_fa):].plot(color=colors_fa, legend=False, ax=axi)
            df.time = pd.to_datetime(df.time,unit=time_unit)  # set datetime to resample
            df.set_index("time").resample(resampling).first().fillna(method="ffill").reset_index().iloc[:, -len(colors_fa):].plot(color=colors_fa, legend=False, ax=axi)
            axi.set_ylabel("Proportion")
            axi.set_xlabel("time [{}]".format(resampling))
            axi.set_ylim(-.1, 1.1)

        else:
            dt = df["dt"]
            for i, (facies, time) in enumerate(zip(df["facies"], df["time"])):
                if i > 0:
                    axi.barh(0, dt[i], left=df["time"].loc[i], color=arch_table.get_facies_obj(ID=facies, type="ID").c)
                else:
                    axi.barh(0, dt[i], left=0, color=arch_table.get_facies_obj(ID=facies, type="ID").c, label=arch_table.get_facies_obj(ID=facies, type="ID").name)
            
            axi.set_xlim(0, df["time"].iloc[-1])
            axi.set_xlabel("Time (days)")
            axi.set_yticks([])

        if plot_distance:
            axi = ax[1]
    
    # plot facies function of the distance traveled
    if plot_distance:
        all_dist = df["distance"].values
        all_cum_dist = df["cum_distance"].values

        if proportions:
            df.set_index("cum_distance").iloc[:, -len(colors_fa):].plot(color=colors_fa, legend=False, ax=axi)
            axi.set_ylabel("Proportion")
            axi.set_xlabel("Distance [m]")
            axi.set_ylim(-.1, 1.1)

        else:

            for i in range(int(len(df["facies"]) - 1)):
                facies = df["facies"].iloc[i]
                width = all_dist[i+1]
                distance = all_cum_dist[i]
                axi.barh(0, width, left=distance, color=arch_table.get_facies_obj(ID=facies, type="ID").c)

        axi.set_xlim(0, all_cum_dist[-1])
        axi.set_xlabel("Distance (m)")
        axi.set_yticks([])

def get_nodes(locs, nx, ny):
    nodes = []
    for k, i, j in locs:
        nodes.append(k * nx * ny + i * nx + j)
    return nodes

def get_locs(nodes, nx, ny):
    locs = []
    for node in nodes:
        k = node // (nx * ny)
        i = (node - k * nx * ny) // nx
        j = node - k * nx * ny - i * nx
        locs.append((k, i, j))
    return locs

def points2grid_index(points, grid):
    """
    Convert a list of points to a list of grid indices

    Parameters
    ----------
    points : list of tuples
        list of points to convert. Each tuple must have 3 values (xp, yp, zp) corresponding to the coordinates of the point
    grid : modflow grid object
        modflow grid object

    Returns
    -------
    list of tuples
        list of tuples with the indices of the points in the grid. Depends on the type of grid
        if grid is disv, return (ilay, icell)
        if grid is disu, return (icell)
    """
    # import packages
    from scipy.spatial import cKDTree
    from matplotlib.path import Path
    from flopy.utils.geometry import is_clockwise

    # functions to intersect the points with the grid
    def intersect_point_fast_ug(self, index_cells, x, y, z=None,
                                xv=None, yv=None, zv=None,
                                local=False, forgive=False):

        
        
        """
        Get the CELL2D number of a point with coordinates x and y

        When the point is on the edge of two cells, the cell with the lowest
        CELL2D number is returned.

        Parameters
        ----------
        x : float
            The x-coordinate of the requested point
        y : float
            The y-coordinate of the requested point
        z : float, None
            optional, z-coordiante of the requested point
        local: bool (optional)
            If True, x and y are in local coordinates (defaults to False)
        forgive: bool (optional)
            Forgive x,y arguments that fall outside the model grid and
            return NaNs instead (defaults to False - will throw exception)

        Returns
        -------
        icell2d : int
            The CELL2D number

        """
        if local:
            # transform x and y to real-world coordinates
            x, y = super().get_coords(x, y)
        
        if xv is None or yv is None or zv is None:
            # get the vertices of the grid cells
            xv, yv, zv = self.xyzvertices

        for icell2d in index_cells:
            
            xa = np.array(xv[icell2d])
            ya = np.array(yv[icell2d])
            # x and y at least have to be within the bounding box of the cell
            if (
                np.any(x <= xa)
                and np.any(x >= xa)
                and np.any(y <= ya)
                and np.any(y >= ya)
            ):
                if is_clockwise(xa, ya):
                    radius = -1e-9
                else:
                    radius = 1e-9
                path = Path(np.stack((xa, ya)).transpose())
                # use a small radius, so that the edge of the cell is included
                if path.contains_point((x, y), radius=radius):
                    if z is None:
                        return icell2d

                    for lay in range(self.nlay):
                        if lay != 0 and not self.grid_varies_by_layer:
                            icell2d += self.ncpl[lay - 1]
                        if zv[0, icell2d] >= z >= zv[1, icell2d]:
                            return icell2d

        if forgive:
            icell2d = np.nan
            return icell2d

        raise Exception("point given is outside of the model area")

    def intersect_point_fast_vg(self, index_cells, x, y, z=None, 
                        xv=None, yv=None, zv=None,
                        local=False, forgive=False):

        from matplotlib.path import Path
        from flopy.utils.geometry import is_clockwise
        """
        Get the CELL2D number of a point with coordinates x and y

        When the point is on the edge of two cells, the cell with the lowest
        CELL2D number is returned.

        Parameters
        ----------
        x : float
            The x-coordinate of the requested point
        y : float
            The y-coordinate of the requested point
        z : float, None
            optional, z-coordiante of the requested point will return
            (lay, icell2d)
        local: bool (optional)
            If True, x and y are in local coordinates (defaults to False)
        forgive: bool (optional)
            Forgive x,y arguments that fall outside the model grid and
            return NaNs instead (defaults to False - will throw exception)

        Returns
        -------
        icell2d : int
            The CELL2D number

        """
        if local:
            # transform x and y to real-world coordinates
            x, y = super().get_coords(x, y)
        
        if xv is None or yv is None or zv is None:
            # get the vertices of the grid cells
            xv, yv, zv = self.xyzvertices

        for icell2d in index_cells:
            xa = np.array(xv[icell2d])
            ya = np.array(yv[icell2d])
            # x and y at least have to be within the bounding box of the cell
            if (
                np.any(x <= xa)
                and np.any(x >= xa)
                and np.any(y <= ya)
                and np.any(y >= ya)
            ):
                path = Path(np.stack((xa, ya)).transpose())
                # use a small radius, so that the edge of the cell is included
                if is_clockwise(xa, ya):
                    radius = -1e-9
                else:
                    radius = 1e-9
                if path.contains_point((x, y), radius=radius):
                    if z is None:
                        return icell2d

                    for lay in range(self.nlay):
                        if (
                            self.top_botm[lay, icell2d]
                            >= z
                            >= self.top_botm[lay + 1, icell2d]
                        ):
                            return lay, icell2d

        if forgive:
            icell2d = np.nan
            if z is not None:
                return np.nan, icell2d

            return icell2d

        raise Exception("point given is outside of the model area")

    # load cell centers and vertices
    xc, yc, zc = grid.xyzcellcenters
    xv, yv, zv = grid.xyzvertices

    list_points = np.array(points)
    
    tree = cKDTree(np.array([xc, yc, zc[0]]).T)
    dist, index = tree.query(list_points, k=10)  # find the nearest cells for each point in list_points

    list_index = []
    for list_of_index_cells, point in zip(index, list_points):
        x, y, z = point
        if grid.grid_type == "vertex":
            p = intersect_point_fast_vg(grid, list_of_index_cells, x, y, z=z,
                                        xv=xv, yv=yv, zv=zv, 
                                        local=False, forgive=True
                                        )
        elif grid.grid_type == "unstructured":
            p = intersect_point_fast_ug(grid, list_of_index_cells, x, y, z=z,
                                        xv=xv, yv=yv, zv=zv,
                                        local=False, forgive=True
                                        )   
        else:
            raise ValueError("Grid type not supported")
        
        list_index.append(p)

    return np.array(list_index)

## archpy to modflow class ##
class archpy2modflow:

    """
    Class to convert an ArchPy table to a MODFLOW 6 model

    Parameters
    ----------
    T1 : :class:`base.Arch_table` object
        ArchPy table object to convert
    sim_name : str
        name of the simulation
    model_dir : str
        directory where the model will be saved
    model_name : str
        name of the model
    exe_name : str
        path to the mf6 executable
    """

    def __init__(self, T1, sim_name="sim_test", model_dir="workspace", model_name="test", exe_name="mf6"):
        self.T1 = T1
        self.sim_name = sim_name
        self.model_dir = model_dir
        self.model_name = model_name
        self.exe_name = exe_name
        self.sim = None
        self.sim_prt = None
        self.sim_e = None
        self.sim_t = None
        self.grid_mode = None
        self.layers_names = None
        self.list_active_cells = None
        self.mp = None
        self.factor_x = None
        self.factor_y = None
        self.factor_z = None

    def create_sim(self, grid_mode="archpy", iu=0, 
                   lay_sep=1,
                   modflowgrid_props=None, xorigin=0, yorigin=0, angrot=0,
                   factor_x=None, factor_y=None, factor_z=None, 
                   unit_limit=None):

        """
        Create a modflow simulation from an ArchPy table
        
        Parameters
        ----------
        grid_mode : str
            "archpy" : use the grid defined in the ArchPy table
            "layers" : use the surfaces of each unit to define the grid
            "new_resolution" : use factors to change the resolution of the grid
            In this case, factor_x, factor_y and factor_z must be provided
        iu : int
            index of the unit to use when grid_mode is "layers"
        lay_sep : int or list of int of size nlay
            if grid_mode is layers, lay_sep indicates the number of layers to separate each unit
        factor_x : float
            factor to change the resolution of the grid in the x direction. e.g. 2 means that the resolution will be divided by 2
        factor_y : float
            factor to change the resolution of the grid in the y direction.
        factor_z : float    
            factor to change the resolution of the grid in the z direction.
        unit_limit: unit name
            unit under which cells are considered as inactive
        """

        sim = fp.mf6.MFSimulation(sim_name=self.sim_name, version='mf6', exe_name=self.exe_name, 
                         sim_ws=self.model_dir)
        gwf = fp.mf6.ModflowGwf(sim, modelname=self.model_name,
                                model_nam_file='{}.nam'.format(self.model_name))

        # upscale idomain
        sx_grid, sy_grid, sz_grid = self.T1.get_sx(), self.T1.get_sy(), self.T1.get_sz()
        ox_grid, oy_grid, oz_grid = self.T1.get_ox(), self.T1.get_oy(), self.T1.get_oz()

        if grid_mode in ["disv", "disu"]:  

            if grid_mode == "disv":
                dis = fp.mf6.ModflowGwfdisv(gwf, **modflowgrid_props, xorigin=xorigin, yorigin=yorigin)
            else:
                dis = fp.mf6.ModflowGwfdisu(gwf, **modflowgrid_props, xorigin=xorigin, yorigin=yorigin)
            
            grid = gwf.modelgrid  # get the grid object
            # rotate grid around the origin of archpy model
            xorigin_rot, yorigin_rot = rotate_point((xorigin, yorigin), origin=(ox_grid, oy_grid), angle=-angrot) 
            grid.set_coord_info(xoff=xorigin_rot, yoff=yorigin_rot, angrot=-angrot)

            # idomain #
            # inactive cells below unit limit
            idomain = np.flip(np.flipud(self.T1.get_mask().astype(int)), axis=1)
            if unit_limit is not None:
                mask = mask_below_unit(self.T1, unit_limit, iu=iu)
                mask = np.flip(np.flipud(mask), axis=1)  # flip the array to have the same orientation as the ArchPy table
                idomain[mask == 1] = 0

            new_idomain = upscale_k(idomain, method="arithmetic",
                                    dx=sx_grid, dy=sy_grid, dz=sz_grid,
                                    ox=ox_grid, oy=oy_grid, oz=oz_grid,
                                    factor_x=factor_x, factor_y=factor_y, factor_z=factor_z,
                                    grid=grid)[0]

            # idomain --> superior to 0.5 set to 1
            new_idomain[new_idomain > 0.5] = 1
            new_idomain[new_idomain <= 0.5] = 0

            # set idomain
            # return dis, new_idomain
            if grid_mode == "disv":
                dis.idomain.set_layered_data(new_idomain)
            else:
                dis.idomain.set_data(new_idomain)
            
        else:
                
            #grid
            nlay, nrow, ncol = self.T1.get_nz(), self.T1.get_ny(), self.T1.get_nx()
            delr, delc = self.T1.get_sx(), self.T1.get_sy()
            xoff, yoff = self.T1.get_ox(), self.T1.get_oy()

            if grid_mode == "archpy":
                top = np.ones((nrow, ncol)) * self.T1.get_zg()[-1]
                botm = np.ones((nlay, nrow, ncol)) * self.T1.get_zg()[:-1].reshape(-1, 1, 1)
                botm = np.flip(np.flipud(botm), axis=1)  # flip the array to have the same orientation as the ArchPy table
                idomain = np.flip(np.flipud(self.T1.get_mask().astype(int)), axis=1)  # flip the array to have the same orientation as the ArchPy table

                # inactive cells below unit limit
                if unit_limit is not None:
                    mask = mask_below_unit(self.T1, unit_limit, iu=iu)
                    mask = np.flip(np.flipud(mask), axis=1)  # flip the array to have the same orientation as the ArchPy table
                    idomain[mask == 1] = 0

            elif grid_mode == "layers":
                
                n_units = len(self.T1.get_surface()[1])
                if isinstance(lay_sep, int):
                    lay_sep = [lay_sep] * n_units
                assert len(lay_sep) == n_units, "lay_sep must have the same length as the number of units"
                self.lay_sep = lay_sep  # save lay_sep

                def list_unit_below_unit(T1, unit):
                    u = T1.get_unit(unit)
                    units = []
                    for u_name in T1.get_surface()[1]:
                        unit_to_compare = T1.get_unit(u_name)
                        if u_name == unit:
                            units.append(unit_to_compare)
                            continue
                        if unit_to_compare > u or unit_to_compare in u.get_baby_units(vb=0):  # if the unit is below the unit or is a baby unit
                            units.append(unit_to_compare)
                    return units

                # determine which units will be inactive
                if unit_limit is not None:
                    units = list_unit_below_unit(self.T1, unit_limit)
                    n_units_removed = len(units)
                    n_units_removed = np.sum(lay_sep[-n_units_removed:])
                else:
                    n_units_removed = None

                # get surfaces of each unit
                top = self.T1.get_surface(typ="top")[0][0, iu].copy()
                top = np.flip(top, axis=0)
                botm = self.T1.get_surface(typ="bot")[0][:, iu].copy()
                botm = np.flip(botm, axis=1)

                botm_org = botm.copy()  # copy of botm to ensure that we only select original surfaces and not new sublayers

                # add sublayers to botm
                for ilay in range(n_units):
                    if ilay == 0:
                        s1 = top
                    else:
                        s1 = botm_org[ilay-1]
                    s2 = botm_org[ilay]
                    for isublay in range(1, lay_sep[ilay]):

                        smean = s1*(1-isublay/lay_sep[ilay]) + s2*(isublay/lay_sep[ilay])  # mean of the two surfaces
                        botm = np.insert(botm, sum(lay_sep[0:ilay])+isublay-1, smean, axis=0)  # insert the surface at the right place

                layers_names = self.T1.get_surface(typ="bot")[1]
                self.layers_names = layers_names
                nlay = botm.shape[0]

                # define idomain (1 if thickness > 0, 0 if nan, -1 if thickness = 0)
                idomain = np.ones((nlay, nrow, ncol))
                thicknesses = -np.diff(np.vstack([top.reshape(-1, nrow, ncol), botm]), axis=0)
                idomain[thicknesses == 0] = -1
                idomain[np.isnan(thicknesses)] = 0

                # set nan of each layer to the mean of the previous layer + 1e-2
                prev_mean = None
                for ilay in range(nlay-1, -1, -1):
                    mask = np.isnan(botm[ilay])
                    if ilay == nlay-1:
                        prev_mean = np.nanmean(botm[ilay])
                        botm[ilay][mask] = prev_mean
                    else:
                        prev_mean = max(np.nanmean(botm[ilay]), prev_mean + 1e-2)
                        botm[ilay][mask] = prev_mean

                # inactive cells below unit limit
                if n_units_removed is not None:
                    idomain[-n_units_removed:] = 0

                rtol = 1e-7
                # adapt botm in order that each layer has a thickness > 0 
                for i in range(-1, nlay-1):
                    if i == -1:
                        s1 = top
                    else:
                        s1 = botm[i]
                    s2 = botm[i+1]
                    # mask = np.abs(s2 - s1) < rtol
                    # s1[mask] += 1e-2    
                    mask = (s1 <= s2) | (np.abs(s2 - s1) < rtol)
                    s1[mask] = s2[mask] + 1e-2
                    # mask = ((s2 < (s1 + np.ones(s1.shape)*rtol)) & (s2 > (s1 - np.ones(s1.shape)*rtol)))  # mask to identify cells where the thickness is == 0 with some tolerance

                    # 2nd loop over previous layers to ensure that the thickness is > 0
                    for o in range(i, -1, -1):
                        s2 = botm[o]
                        if o == 0:
                            s1 = top
                        else:
                            s1 = botm[o-1]
                        mask = (s1 <= s2) | (np.abs(s2 - s1) < rtol)
                        # mask = (s1 <= s2) | ((s2 < (s1 + np.ones(s1.shape)*rtol)) & (s2 > (s1 - np.ones(s1.shape)*rtol)))  # mask to identify cells where the thickness is <= 0 with some tolerance
                        s1[mask] = s2[mask] + 1e-2

            elif grid_mode == "new_resolution":
                assert factor_x is not None, "factor_x must be provided"
                assert factor_y is not None, "factor_y must be provided"
                assert factor_z is not None, "factor_z must be provided"
                assert nrow % factor_y == 0, "nrow must be divisible by factor_y"
                assert ncol % factor_x == 0, "ncol must be divisible by factor_x"
                assert nlay % factor_z == 0, "nlay must be divisible by factor_z"
                nrow = int(nrow / factor_y)
                ncol = int(ncol / factor_x)
                nlay = int(nlay / factor_z)
                delr = delr * factor_x
                delc = delc * factor_y
                top = np.ones((nrow, ncol)) * self.T1.get_zg()[-1]
                botm = np.ones((nlay, nrow, ncol)) * self.T1.get_zg()[::-factor_z][1:].reshape(-1, 1, 1)
                
                # how to define idomain ?
                idomain = np.zeros((nlay, nrow, ncol))
                mask_org = self.T1.get_mask().astype(int)

                # modify mask_org to remove the inactive cells
                if unit_limit is not None:
                    mask = mask_below_unit(self.T1, unit_limit, iu=iu)
                    mask_org[mask == 1] = 0

                for ilay in range(0, self.T1.get_nz(), factor_z):
                    for irow in range(0, self.T1.get_ny(), factor_y):
                        for icol in range(0, self.T1.get_nx(), factor_x):
                            mask = mask_org[ilay:ilay+factor_z, irow:irow+factor_y, icol:icol+factor_x]
                            if mask.mean() >= 0.5:
                                idomain[ilay//factor_z, irow//factor_y, icol//factor_x] = 1
                
                idomain = np.flip(np.flipud(idomain), axis=1)  # flip the array to have the same orientation as the ArchPy table

                self.factor_x = factor_x
                self.factor_y = factor_y
                self.factor_z = factor_z
            
            assert (np.array(check_thk(top, botm))).all(), "Error in the processing of the surfaces, some cells have a thickness < 0"

            rot_angle = self.T1.get_rot_angle()
            dis = fp.mf6.ModflowGwfdis(gwf, nlay=nlay, nrow=nrow, ncol=ncol,
                                        delr=delr, delc=delc,
                                        top=top, botm=botm,
                                        xorigin=xoff, yorigin=yoff, 
                                        idomain=idomain, angrot=rot_angle)

        # save grid mode
        self.grid_mode = grid_mode

        perioddata = [(1, 1, 1.0)]
        tdis = fp.mf6.ModflowTdis(sim, time_units='SECONDS',perioddata=perioddata)

        ims = fp.mf6.ModflowIms(sim, complexity="simple")

        #Initial condition
        ic   = fp.mf6.ModflowGwfic(gwf, strt=1)

        # output control
        oc   = fp.mf6.ModflowGwfoc(gwf,budget_filerecord='{}.cbc'.format(self.model_name),
                                    head_filerecord='{}.hds'.format(self.model_name),
                                    saverecord=[('HEAD', 'LAST'),
                                                ('BUDGET', 'LAST')],
                                    printrecord=[('BUDGET', 'ALL')])
        
        # npf package
        # empty package
        npf = fp.mf6.ModflowGwfnpf(gwf, icelltype=0, k=1e-3, save_flows=True, save_saturation=True)

        self.sim = sim
        print("Simulation created")
        print("To retrieve the simulation, use the get_sim() method")

    def upscale_prop(self, prop_key, iu=0, ifa=0, ip=0, method="arithmetic", log=False):

        gwf = self.get_gwf()
        grid_mode = self.grid_mode
        if grid_mode == "archpy":
            prop = self.T1.get_prop(prop_key)[iu, ifa, ip]
            if log:
                prop = 10**prop
            new_prop = np.flip(np.flipud(prop), axis=1)  # flip the array to have the same orientation as the ArchPy table
        elif grid_mode == "layers":
            nrow, ncol, nlay = gwf.modelgrid.nrow, gwf.modelgrid.ncol, gwf.modelgrid.nlay
            
            # get prop
            prop = self.T1.get_prop(prop_key)[iu, ifa, ip]
            if log:
                prop = 10**prop
            new_prop = np.ones((nlay, nrow, ncol))

            # get existing mask units
            try:
                mask_units = self.mask_units
            except:
                raise ValueError("mask_units must be defined before upscaling a property --> try setting the hydraulic conductivity first")

            for irow in range(nrow):
                for icol in range(ncol):
                    for ilay in range(nlay):
                        mask_unit = mask_units[ilay]

                        # extract values in the new cell
                        prop_vals = prop[:, irow, icol][mask_unit[:, irow, icol]]

                        if method == "arithmetic":
                            new_prop[ilay, irow, icol] = np.mean(prop_vals)
                        elif method == "harmonic":
                            new_prop[ilay, irow, icol] = 1 / np.mean(1 / prop_vals)
                        else:
                            raise ValueError("method must be one of 'arithmetic' or 'harmonic'")

            # fill nan values with the mean of the layer
            for ilay in range(nlay):
                mask = np.isnan(new_prop[ilay])
                new_prop[ilay][mask] = np.nanmean(new_prop[ilay])

            new_prop = np.flip(new_prop, axis=1)  # we have to flip in order to match modflow grid

        elif grid_mode == "new_resolution":

            from ArchPy.uppy import upscale_k
            assert self.factor_x is not None, "factor_x must be defined, try setting the hydraulic conductivity first"
            assert self.factor_y is not None, "factor_y must be defined, try setting the hydraulic conductivity first"
            assert self.factor_z is not None, "factor_z must be defined, try setting the hydraulic conductivity first"

            prop = self.T1.get_prop(prop_key)[iu, ifa, ip]
            prop = np.flip(np.flipud(prop), axis=1)  # flip the array to have the same orientation as the ArchPy table
            if log:
                prop = 10**prop
            new_prop, _, _ = upscale_k(prop, method=method,
                                 dx=self.T1.get_sx(), dy=self.T1.get_sy(), dz=self.T1.get_sz(),
                                 factor_x=self.factor_x, factor_y=self.factor_y, factor_z=self.factor_z)

            # fill nan values
            new_prop[np.isnan(new_prop)] = np.nanmean(new_prop)

        elif grid_mode in ["disv", "disu"]:

            from ArchPy.uppy import upscale_k
            dx, dy, dz = self.T1.get_sx(), self.T1.get_sy(), self.T1.get_sz()
            ox, oy, oz = self.T1.get_ox(), self.T1.get_oy(), self.T1.get_oz()   

            prop = self.T1.get_prop(prop_key)[iu, ifa, ip]
            prop = np.flip(np.flipud(prop), axis=1)  # flip the array to have the same orientation as the ArchPy table
            if log:
                prop = 10**prop

            # get the grid object --> needs to be rotated
            import copy
            grid = gwf.modelgrid
            grid = copy.deepcopy(grid)  # create a copy of the grid to avoid modifying the original one

            # rotate grid around the origin of archpy model

            # retrieve origin of the new grid and archpy grid as well as the rotation angle 
            xorigin, yorigin = grid.xoffset, grid.yoffset
            ox_grid, oy_grid = self.T1.get_ox(), self.T1.get_oy()
            angrot = self.T1.get_rot_angle()
            xorigin_rot, yorigin_rot = rotate_point((xorigin, yorigin), origin=(ox_grid, oy_grid), angle=-angrot) 

            # rotation
            grid.set_coord_info(xoff=xorigin_rot, yoff=yorigin_rot, angrot=-angrot)

            # upscale
            new_prop, _, _ = upscale_k(prop, method=method,
                                 dx=dx, dy=dy, dz=dz, ox=ox, oy=oy, oz=oz,
                                 factor_x=self.factor_x, factor_y=self.factor_y, factor_z=self.factor_z,
                                 grid=grid)

        return new_prop

    def set_k(self, k_key="K",
              iu=0, ifa=0, ip=0,
              log=False, k=None, k22=None, k33=None, k_average_method="arithmetic", 
              upscaling_method="simplified_renormalization",
              xt3doptions=None):

        """
        Set the hydraulic conductivity for a specific facies
        """

        # write a function to get proportion of each value in the array
        def get_proportion(arr):
            unique, counts = np.unique(arr, return_counts=True)
            return dict(zip(unique, counts/np.sum(counts)))

        # remove the npf package if it already exists
        gwf = self.get_gwf()
        gwf.remove_package("npf")

        if k is None:
            grid_mode = self.grid_mode
            if grid_mode == "archpy":
                new_k22 = None
                new_k33 = None
                k = self.T1.get_prop(k_key)[iu, ifa, ip]
                k = np.flip(np.flipud(k), axis=1)  # flip the array to have the same orientation as the ArchPy table
                if log:
                    new_k = 10**k
                else:
                    new_k = k

            elif grid_mode == "layers":

                new_k22 = None
                nrow, ncol, nlay = gwf.modelgrid.nrow, gwf.modelgrid.ncol, gwf.modelgrid.nlay

                # initialize new_k and new_k33
                kh = self.T1.get_prop(k_key)[iu, ifa, ip] 
                if log:
                    kh = 10**kh
                new_k = np.ones((nlay, nrow, ncol))

                # initialize variable for facies upscaling
                facies_arr = self.T1.get_facies(iu, ifa, all_data=False)
                upscaled_facies = {}

                for ifa in np.unique(facies_arr):
                    # upscaled_facies[ifa] = np.zeros((facies_arr.shape[0], facies_arr.shape[1], facies_arr.shape[2]))
                    upscaled_facies[ifa] = np.zeros((nlay, nrow, ncol))

                if k_average_method == "anisotropic":
                    new_k33 = np.ones((nlay, nrow, ncol))
                else:
                    new_k33 = None

                botm = gwf.dis.botm.array.copy()
                botm = np.flip(botm, axis=1)

                # mask units (boolean mask of each layer to compute the average)
                # if lay sep is not 1, we need to compute new mask units for each sublayers
                layers = self.layers_names
                mask_units = []
                ilay = 0
                for l in layers:
                    if self.lay_sep[ilay] == 1:
                        mask_units.append(self.T1.unit_mask(l).astype(bool))
                    else:
                        for isublay in range(self.lay_sep[ilay]):
                            if ilay == 0 and isublay == 0:
                                s1 = np.flip(gwf.dis.top.array, axis=1)
                                s2 = botm[isublay]
                            else:
                                s1 = botm[sum(self.lay_sep[0:ilay])+isublay-1]
                                s2 = botm[sum(self.lay_sep[0:ilay])+isublay]
                            mask = self.T1.compute_domain(s1, s2)
                            mask *= self.T1.unit_mask(l).astype(bool)  # apply mask of the whole unit to ensure that we only consider the cells of the unit
                            mask_units.append(mask.astype(bool))
                    
                    ilay += 1
                self.mask_units = mask_units

                for irow in range(nrow):
                    for icol in range(ncol):
                        for ilay in range(nlay):
                            mask_unit = mask_units[ilay]

                            # extract values in the new cell
                            k_vals = kh[:, irow, icol][mask_unit[:, irow, icol]]

                            # mask_unit = mask_units[ilay]
                            if k_average_method == "arithmetic":
                                new_k[ilay, irow, icol] = np.mean(k_vals)
                            elif k_average_method == "harmonic":
                                new_k[ilay, irow, icol] = 1 / np.mean(1 / k_vals)
                            elif k_average_method == "anisotropic":
                                new_k[ilay, irow, icol] = np.mean(k_vals)
                                new_k33[ilay, irow, icol] = 1 / np.mean(1 / k_vals)
                            else:
                                raise ValueError("k_average_method must be one of 'arithmetic' or 'harmonic' or 'anisotropic'")
                            
                            # facies upscaling
                            arr = facies_arr[:, irow, icol][mask_unit[:, irow, icol]]  # array of facies values in the unit
                            prop = get_proportion(arr)
                            for ifa in np.unique(arr):
                                # upscaled_facies[ifa][:, irow, icol][mask_unit[:, irow, icol]] = prop[ifa]
                                upscaled_facies[ifa][ilay, irow, icol] = prop[ifa]
                
                # save upscaled facies
                self.upscaled_facies = upscaled_facies

                # fill nan values with the mean of the layer
                for ilay in range(nlay):
                    mask = np.isnan(new_k[ilay])
                    new_k[ilay][mask] = np.nanmean(new_k[ilay])

                new_k = np.flip(new_k, axis=1)  # we have to flip in order to match modflow grid

                if k_average_method == "anisotropic":
                    for ilay in range(nlay):
                        mask = np.isnan(new_k33[ilay])
                        new_k33[ilay][mask] = np.nanmean(new_k33[ilay])
                    
                    new_k33 = np.flip(new_k33, axis=1)

            elif grid_mode == "new_resolution":

                from ArchPy.uppy import upscale_k
                dx, dy, dz = self.T1.get_sx(), self.T1.get_sy(), self.T1.get_sz()
                
                factor_x = self.factor_x
                factor_y = self.factor_y
                factor_z = self.factor_z
                
                field = self.T1.get_prop(k_key)[iu, ifa, ip]
                field = np.flip(np.flipud(field), axis=1)  # flip the array to have the same orientation as the ArchPy table
                if log:
                    field = 10**field

                if upscaling_method == "standard_renormalization_center":
                    field_kxx, field_kyy, field_kzz = upscale_k(field, method="standard_renormalization", dx=dx, dy=dy, dz=dz, factor_x=factor_x, factor_y=factor_y, factor_z=factor_z, scheme="center")
                elif upscaling_method == "standard_renormalization_direct":
                    field_kxx, field_kyy, field_kzz = upscale_k(field, method="standard_renormalization", dx=dx, dy=dy, dz=dz, factor_x=factor_x, factor_y=factor_y, factor_z=factor_z, scheme="direct")
                elif upscaling_method == "tensorial_renormalization":
                    field_kxx, field_kyy, field_kzz = upscale_k(field, method="tensorial_renormalization", dx=dx, dy=dy, dz=dz, factor_x=factor_x, factor_y=factor_y, factor_z=factor_z)
                elif upscaling_method == "simplified_renormalization":
                    field_kxx, field_kyy, field_kzz = upscale_k(field, method="simplified_renormalization", dx=dx, dy=dy, dz=dz, factor_x=factor_x, factor_y=factor_y, factor_z=factor_z)

                new_k = field_kxx
                new_k22 = field_kyy
                new_k33 = field_kzz
                
                # fill nan values
                new_k[np.isnan(new_k)] = np.nanmean(new_k)
                new_k22[np.isnan(new_k22)] = np.nanmean(new_k22)
                new_k33[np.isnan(new_k33)] = np.nanmean(new_k33)

                # facies upscaling
                facies_arr = self.T1.get_facies(iu, ifa, all_data=False)
                upscaled_facies = {}
                for ifa in np.unique(facies_arr):
                    # upscaled_facies[ifa] = np.zeros((facies_arr.shape[0], facies_arr.shape[1], facies_arr.shape[2]))
                    upscaled_facies[ifa] = np.zeros((field_kxx.shape[0], field_kxx.shape[1], field_kxx.shape[2]))
                
                for ilay in range(0, self.T1.get_nz(), factor_z):
                    for irow in range(0, self.T1.get_ny(), factor_y):
                        for icol in range(0, self.T1.get_nx(), factor_x):
                            mask_unit = facies_arr[ilay:ilay+factor_z, irow:irow+factor_y, icol:icol+factor_x]
                            arr = mask_unit.flatten()
                            prop = get_proportion(arr)
                            for ifa in np.unique(arr):
                                # upscaled_facies[ifa][ilay:ilay+factor_z, irow:irow+factor_y, icol:icol+factor_x] = prop[ifa]
                                upscaled_facies[ifa][ilay//factor_z, irow//factor_y, icol//factor_x] = prop[ifa]

                self.upscaled_facies = upscaled_facies
            
            elif grid_mode in ["disv", "disu"]:

                from ArchPy.uppy import upscale_k
                dx, dy, dz = self.T1.get_sx(), self.T1.get_sy(), self.T1.get_sz()
                ox, oy, oz = self.T1.get_ox(), self.T1.get_oy(), self.T1.get_oz()

                new_k = None
                new_k22 = None
                new_k33 = None

                # get the grid object --> needs to be rotated
                grid = gwf.modelgrid  

                # rotate grid around the origin of archpy model

                # retrieve origin of the new grid and archpy grid as well as the rotation angle 
                xorigin, yorigin = grid.xoffset, grid.yoffset
                ox_grid, oy_grid = self.T1.get_ox(), self.T1.get_oy()
                angrot = self.T1.get_rot_angle()
                xorigin_rot, yorigin_rot = rotate_point((xorigin, yorigin), origin=(ox_grid, oy_grid), angle=-angrot) 

                # rotation
                grid.set_coord_info(xoff=xorigin_rot, yoff=yorigin_rot, angrot=-angrot)

                # get field
                field = self.T1.get_prop(k_key)[iu, ifa, ip]
                field = np.flip(np.flipud(field), axis=1)  # flip the array to have the same orientation as the ArchPy table
                if log:
                    field = 10**field

                field_kxx, field_kyy, field_kzz = upscale_k(field, method=upscaling_method,
                                                            dx=dx, dy=dy, dz=dz, ox=ox, oy=oy, oz=oz,
                                                            factor_x=self.factor_x, factor_y=self.factor_y, factor_z=self.factor_z,
                                                            grid=grid)

                new_k = field_kxx.astype(float)

                if upscaling_method in ["arithmetic", "harmonic", "geometric"]:
                    new_k22 = None
                    new_k33 = None

                    # fill nan
                    new_k[np.isnan(new_k)] = np.nanmean(new_k)

                else:
                    new_k22 = field_kyy
                    new_k33 = field_kzz
                
                    # fill nan values
                    new_k[np.isnan(new_k)] = np.nanmean(new_k)
                    new_k22[np.isnan(new_k22)] = np.nanmean(new_k22)
                    new_k33[np.isnan(new_k33)] = np.nanmean(new_k33)
                    
                # facies upscaling
                facies_arr = self.T1.get_facies(iu, ifa, all_data=False)
                facies_arr = np.flip(np.flipud(facies_arr), axis=1)  # flip the array to have the same orientation as the ArchPy table

                upscaled_facies = {}
                for ifa in [fa.ID for fa in self.T1.get_all_facies()]:

                    upscaled_facies[ifa] = np.zeros((facies_arr.shape[0], facies_arr.shape[1], facies_arr.shape[2]))

                    field = facies_arr.copy()
                    field[facies_arr == ifa] = 1
                    field[facies_arr != ifa] = 0
                    

                    field_up, _, _ = upscale_k(field, method="arithmetic", 
                                               dx=dx, dy=dy, dz=dz,
                                               ox=ox, oy=oy, oz=oz,
                                               grid=grid)
                    
                    upscaled_facies[ifa] = field_up.astype(float)
                    upscaled_facies[ifa][np.isnan(upscaled_facies[ifa])] = 0

                    self.upscaled_facies = upscaled_facies

        else:
            new_k = k
            new_k22 = k22
            new_k33 = k33

        # new_k = np.flip(new_k, axis=1)  # we have to flip in order to match modflow grid
        npf = fp.mf6.ModflowGwfnpf(gwf, icelltype=0, k=new_k, k22=new_k22, k33=new_k33, save_flows=True, save_specific_discharge=True, save_saturation=True, xt3doptions=xt3doptions)
        # npf.write()

    def set_strt(self, heads=None):

        """
        Set the starting heads
        """

        gwf = self.get_gwf()
        gwf.remove_package("ic")
        ic = fp.mf6.ModflowGwfic(gwf, strt=heads)
        ic.write()

    def get_list_active_cells(self):
        """
        Get the list of active cells in the modflow model
        """

        grid_type = self.get_gwf().dis.package_type
        if self.list_active_cells is None:
            gwf = self.get_gwf()
            if grid_type == "dis":
                idomain = gwf.dis.idomain.array
                list_active_cells = []
                for ilay in range(idomain.shape[0]):
                    for irow in range(idomain.shape[1]):
                        for icol in range(idomain.shape[2]):
                            if idomain[ilay, irow, icol] == 1:
                                list_active_cells.append((ilay, irow, icol))
                self.list_active_cells = list_active_cells
            elif grid_type == "disv":
                idomain = gwf.dis.idomain.array  # (nlay, ncells)
                list_active_cells = []
                for ilay in range(idomain.shape[0]):
                    for icell in range(idomain.shape[1]):
                        if idomain[ilay, icell] == 1:
                            list_active_cells.append((ilay, icell))
                self.list_active_cells = list_active_cells
            elif grid_type == "disu":
                idomain = gwf.dis.idomain.array  # (ncells)
                list_active_cells = []
                for icell in range(idomain.shape[0]):
                    if idomain[icell] == 1:
                        list_active_cells.append((icell))
                self.list_active_cells = list_active_cells
        return self.list_active_cells

    # get functions
    def get_sim(self):
        """
        Get the MODFLOW 6 simulation object
        """
        assert self.sim is not None, "You need to create the simulation first"
        return self.sim
    
    def get_gwf(self):
        """
        Get the MODFLOW 6 groundwater flow object
        """
        assert self.sim is not None, "You need to create the simulation first"
        return self.sim.get_model()

    # outputs
    def get_heads(self, kstpkper=(0, 0)):
        """
        Get the heads of the simulation
        """
        gwf = self.get_gwf()
        head = gwf.output.head().get_data(kstpkper=kstpkper)
        return head
    
    # plots
    def plot_3D_heads(self, kstpkper=(0, 0), ax=None, **kwargs):

        """
        Plot the heads of the simulation
        """
        assert self.grid_mode == "archpy", "This function only works with the 'archpy' grid mode"
        
        head = self.get_heads(kstpkper=kstpkper)
        head[head == 1e30] = np.nan

        self.T1.plot_arr(np.flipud(np.flip(head, axis=1)), "head", 2)

    def mp_create(self, mpexe, trackdir="forward",
                  locs=None, rowcelldivisions=1, columncelldivisions=1, layercelldivisions=1,
                  list_p_coords=None):
        """
        Create a modpath simulation from an ArchPy table

        Parameters
        ----------
        mpexe : str
            path to the modpath executable
        trackdir : str
            direction of tracking
        locs : list of tuples
            list of cells to put particles. List of int where values correspond to the indices of the cells.
            Each cell in locs will be then spatially divided and will output rowcelldivisions * columncelldivisions * layercelldivisions particles
        rowcelldivisions : int
            number of row divisions. If 1, no division, if 2, divide each cell in 2 parts, etc.
        columncelldivisions : int
            number of column divisions
        layercelldivisions : int
            number of layer divisions
        list_p_coords : list of tuples
            list of particles coordinates. Each tuple must have 3 values (xp, yp, zp) corresponding to the coordinates of the particle
        """

        gwf = self.get_gwf()
        workspace = self.model_dir
        model_name = gwf.name
        mpnamf = f"{model_name}_mp_forward"
        
        if locs is not None:
            nodes = get_nodes(locs, gwf.modelgrid.ncol, gwf.modelgrid.nrow)
            if len(nodes) == 0:
                print("No particles to track")
                return
            elif len(nodes) == 1:
                nodes = nodes[0]
            mp = fp.modpath.Modpath7.create_mp7(
                                                modelname=mpnamf,
                                                trackdir=trackdir,
                                                flowmodel=gwf,
                                                model_ws=workspace,
                                                rowcelldivisions=rowcelldivisions,
                                                columncelldivisions=columncelldivisions,
                                                layercelldivisions=layercelldivisions,
                                                nodes = nodes,
                                                exe_name=mpexe,
                                            )
        else:
            if list_p_coords is not None:
                # write a function to find the modlfow cellids as well as localx, localy and localz from a coordinate
                from shapely.geometry import Point, MultiPoint

                grid = self.get_gwf().modelgrid
                ix = fp.utils.gridintersect.GridIntersect(mfgrid=grid)

                list_p = []
                list_cellids = []
                for pi in list_p_coords:
                    p1 = Point(pi)
                    list_cellids.append(ix.intersect(p1).cellids[0])

                cellids = np.array([cids for cids in list_cellids])

                # multp = MultiPoint(list_p)

                # cellids = ix.intersect(multp).cellids
                # cellids = np.array([np.array(cids) for cids in cellids])
                cellids[:, 0] += 1

                l = []
                for i in range(len(cellids)):
                    cid = cellids[i]
                    exem_i = list_p_coords[i]
            
                    local_dx = (grid.xvertices[cid[1], cid[2]+1] - grid.xvertices[cid[1], cid[2]]) / 2
                    local_dy = - (grid.yvertices[cid[1]+1, cid[2]] - grid.yvertices[cid[1], cid[2]]) / 2
                    # local_dz = - (grid.zcellcenters[cid[0]+1, cid[1], cid[2]] - grid.zcellcenters[cid[0], cid[1], cid[2]]) / 2
                    local_dz = (grid.botm[cid[0]-1, cid[1], cid[2]] - grid.botm[cid[0], cid[1], cid[2]]) / 2
                    localx = (exem_i[0] - (grid.xcellcenters[cid[1], cid[2]] - local_dx)) / (2*local_dx)
                    localy = (exem_i[1] - (grid.ycellcenters[cid[1], cid[2]] - local_dy)) / (2*local_dy)
                    localz = (exem_i[2] - (grid.zcellcenters[cid[0], cid[1], cid[2]] - local_dz)) / (2*local_dz)

                    p1 = fp.modpath.mp7particledata.ParticleData([tuple(cid)], structured=True, localx=localx, localy=localy, localz=localz)
                    pg = fp.modpath.mp7particlegroup.ParticleGroup(particledata=p1)

                    l.append(pg)
                    
                # create a modpath simulation
                mp = fp.modpath.Modpath7(
                    modelname=mpnamf,
                    flowmodel=gwf,
                    exe_name=mpexe,
                    model_ws=workspace,
                    verbose=0,
                )

                # basic package
                fp.modpath.Modpath7Bas(mp)

                fp.modpath.Modpath7Sim(
                    mp,
                    simulationtype="combined",
                    trackingdirection=trackdir,
                    weaksinkoption="pass_through",
                    weaksourceoption="pass_through",
                    referencetime=0.0,
                    stoptimeoption="extend",
                    particlegroups=l,
                )

        self.mp = mp  # save the modpath object
        self.mpnamf = mpnamf  # save the name of the modpath file


    def prt_create(self, prt_name="test", workspace="./", trackdir="forward", list_p_coords=None):
        
        """
        Create a particle tracking simulation from a list of coordinates

        Parameters
        ----------
        prt_name : str
            name of the particle tracking simulation
        workspace : str
            path to the workspace where the simulation will be saved
        trackdir : str
            tracking direction. Can be "forward" or "backward"
        list_p_coords : list of tuples
            list of particles coordinates. Each tuple must have 3 values (xp, yp, zp) corresponding to the coordinates of the particle
        """

        sim_prt = fp.mf6.MFSimulation(sim_name=prt_name, exe_name=self.exe_name, sim_ws=workspace, version="mf6")

        tdis_perioddata = self.get_sim().tdis.perioddata.array
        tdis = fp.mf6.ModflowTdis(sim_prt, pname="tdis", time_units="DAYS", nper=len(tdis_perioddata), perioddata=tdis_perioddata)

        # Create PRT model
        prt = fp.mf6.ModflowPrt(sim_prt, modelname=prt_name, model_nam_file="{}.nam".format(prt_name))

        # create prt input package
        mip = fp.mf6.ModflowPrtmip(prt, pname="mip", porosity=0.2)

        # Create grid discretization
        dis = self.get_gwf().dis

        # determine grid type
        grid_type = dis.package_type
        if grid_type == "dis":
            dis_prt = fp.mf6.ModflowGwfdis(prt, nlay=dis.nlay.array, nrow=dis.nrow.array, ncol=dis.ncol.array,
                                            delr=dis.delr.array, delc=dis.delc.array, top=dis.top.array, botm=dis.botm.array,
                                            xorigin=dis.xorigin.array, yorigin=dis.yorigin.array,
                                            idomain=dis.idomain.array)
        elif grid_type == "disv":
            dis_prt = fp.mf6.ModflowGwfdisv(prt, nlay=dis.nlay.array, ncpl=dis.ncpl.array, 
                                            top=dis.top.array, botm=dis.botm.array,
                                            xorigin=dis.xorigin.array, yorigin=dis.yorigin.array,
                                            vertices=dis.vertices.array, cell2d=dis.cell2d.array, nvert=dis.nvert.array,
                                            idomain=dis.idomain.array)
        
        elif grid_type == "disu":
            raise ValueError("disu grid type is not compatible with particle tracking yet")  # to do: add disu grid type
            # dis_prt = fp.mf6.ModflowGwfdisu(prt, nodes=dis.nodes.array, nja=dis.nja.array, area=dis.area.array,
            #                                 iac=dis.iac.array, jac=dis.jac.array, ihc=dis.ihc.array, cl12=dis.cl12.array, hwa=dis.hwa.array,
            #                                 top=dis.top.array, botm=dis.bot.array,
            #                                 xorigin=dis.xorigin.array, yorigin=dis.yorigin.array,
            #                                 vertices=dis.vertices.array, cell2d=dis.cell2d.array, nvert=dis.nverts.array,
            #                                 idomain=dis.idomain.array)


        # construct package data for the particles
        from shapely.geometry import Point, MultiPoint

        grid = self.get_gwf().modelgrid
        ix = fp.utils.gridintersect.GridIntersect(mfgrid=grid)

        if grid_type == "dis":
            list_cellids = []
            for pi in list_p_coords:
                p1 = Point(pi)
                result = ix.intersect(p1)
                if len(result) > 0:
                    list_cellids.append(result.cellids[0])
                else:
                    p1 = Point((pi[0], pi[1]))
                    result = ix.intersect(p1)
                    list_cellids.append((-1, result.cellids[0][0], result.cellids[0][1]))

            cellids = np.array([cids for cids in list_cellids])

            cellids[:, 0] += 1

            # ensure that particles are in active cells, if not move them to the nearest vertical active cell
            idomain = dis.idomain.array
            new_cellids = []
            for cid in cellids:
                if idomain[cid[0], cid[1], cid[2]] == -1:
                    o = 1
                    while True:
                        if cid[0] - o >= 0:
                            if idomain[cid[0] - o, cid[1], cid[2]] == 1:
                                flag_neg = True
                                break
                            
                        elif cid[0] + o < idomain.shape[0]:
                            if idomain[cid[0] + o, cid[1], cid[2]] == 1:
                                flag_neg = False
                                break

                        else:
                            break
                        o += 1

                    if flag_neg:
                        cid[0] -= o
                    else:
                        cid[0] += o

                new_cellids.append(tuple(cid))

            cellids = np.array(new_cellids)

            # package data (irptno, cellid, x, y, z)
            package_data = []
            for i in range(len(cellids)):
                package_data.append((i, cellids[i], list_p_coords[i][0] -grid.xoffset, list_p_coords[i][1] - grid.yoffset, list_p_coords[i][2]))

        elif grid_type in ["disv", "disu"]:
            
            # rotate the grid
            grid.set_coord_info(xoff=grid.xoffset, yoff=grid.yoffset, angrot=0)

            # convert coordinates into cellids
            cellids = points2grid_index(list_p_coords, grid)

            # package data (irptno, cellid, x, y, z)
            package_data = []
            for i in range(len(cellids)):
                package_data.append((i, cellids[i], list_p_coords[i][0] - grid.xoffset, list_p_coords[i][1] - grid.yoffset, list_p_coords[i][2]))

        else:
            raise ValueError("grid type not supported")

        # period data (when to release the particles)
        period_data = {0: ["FIRST"]}

        prp = fp.mf6.ModflowPrtprp(prt, pname="prp", filename="{}.prp".format(prt_name),
                                        packagedata=package_data,
                                        perioddata=period_data,
                                        nreleasepts=len(package_data),
                                        drape=True,
                                        exit_solve_tolerance=1e-5, extend_tracking=True)

        # output control package
        budgetfile_prt_name = "{}.bud".format(prt_name)
        trackfile_name = "{}.trk".format(prt_name)
        trackcsvfile_name = "{}.trk.csv".format(prt_name)
        budget_record = [budgetfile_prt_name]
        track_record = [trackfile_name]
        trackcsv_record = [trackcsvfile_name]
        oc = fp.mf6.ModflowPrtoc(
            prt,
            pname="oc",
            budget_filerecord=budget_record,
            track_filerecord=track_record,
            trackcsv_filerecord=trackcsv_record,
            saverecord=[("BUDGET", "ALL")])

        # load head and budget files
        gwf_ws = self.get_gwf().simulation_data.mfpath.get_sim_path()
        head_file_path = os.path.join(gwf_ws, "{}.hds".format(self.model_name))
        head_file = fp.utils.HeadFile(head_file_path, tdis=sim_prt.tdis)
        budget_file_path = os.path.join(gwf_ws, "{}.cbc".format(self.model_name))
        budget_file = fp.utils.CellBudgetFile(budget_file_path, precision="double", tdis=sim_prt.tdis)

        if trackdir == "backward":
            headfile_bkwd_name = f"{self.model_name}_hds_bkwd"
            budgetfile_bkwd_name = f"{self.model_name}_cbc_bkwd"

            # reverse head and budget files to get backward tracking
            head_file.reverse(workspace + "/" + headfile_bkwd_name)
            budget_file.reverse(workspace + "/" + budgetfile_bkwd_name)

            fmi = fp.mf6.ModflowPrtfmi(prt, packagedata=[
                ("GWFHEAD", headfile_bkwd_name),
                ("GWFBUDGET", budgetfile_bkwd_name),
            ])

        else:
            fmi = fp.mf6.ModflowPrtfmi(prt, packagedata=[
                ("GWFHEAD", head_file_path),
                ("GWFBUDGET", budget_file_path),
            ])

        ems = fp.mf6.ModflowEms(sim_prt, pname="ems", filename="{}.ems".format(prt_name))
        sim_prt.register_solution_package(ems, [prt.name])

        self.sim_prt = sim_prt

    def prt_run(self, silent=False):
        """
        Write PRT files and run the particle tracking simulation using MODFLOW 6
        """

        # write simulation to disk
        self.sim_prt.write_simulation()
        success, msg = self.sim_prt.run_simulation(silent=silent)
        if not success:
            print("particle tracking did not run successfully")
            print(msg)

    def mp_run(self, silent=False):
        """
        Write MODPATH files and run the particle tracking simulation using MODPATH 7
        """
        self.mp.write_input()
        success, msg = self.mp.run_model(silent=silent)
        if not success:
            print("modpath did not run successfully")
            print(msg)

    # get results
    def get_mp(self):
        """
        Get the modpath object
        """
        return self.mp
    
    def mp_get_pathlines_object(self):
        """
        Get the pathlines object from the modpath simulation
        """
        fpth = os.path.join(self.model_dir, f"{self.mpnamf}.mppth")
        p = fp.utils.PathlineFile(fpth)
        return p

    def mp_get_endpoints_object(self):
        """
        Get the endpoints object from the modpath simulation
        """
        fpth = os.path.join(self.model_dir, f"{self.mpnamf}.mpend")
        e = fp.utils.EndpointFile(fpth)
        return e

    def mp_get_facies_path_particle(self, i_particle, fac_time = 1/86400, iu = 0, ifa = 0):
        """
        Function to retrieve the facies sequence along a pathline
        """
        grid_mode = self.grid_mode
        p = self.mp_get_pathlines_object()
        pathline = p.get_data(i_particle)
        df = pd.DataFrame(pathline)
        cells_path = np.array((((df["z"].values-self.T1.zg[0])//self.T1.sz).astype(int),
                                ((df["y"].values-self.T1.yg[0])//self.T1.sy).astype(int),
                                ((df["x"].values-self.T1.xg[0])//self.T1.sx).astype(int))).T
        

        time_ordered = df["time"].values.copy()
        time_ordered *= fac_time
        dt = np.diff(time_ordered)

        # add a column to track distance traveled
        distances = ((df[["x", "y", "z"]].iloc[1:].values - df[["x", "y", "z"]].iloc[:-1].values)**2).sum(1)**0.5

        # store everything in a new dataframe
        df_all = pd.DataFrame(columns=["dt", "time", "distance", "cum_distance", "x", "y", "z"])
        df_all["dt"] = dt
        df_all["time"] = time_ordered[1:]
        df_all["distance"] = distances
        df_all["cum_distance"] = df_all["distance"].cumsum()
        df_all["x"] = (df["x"].values[1:] + df["x"].values[:-1]) / 2
        df_all["y"] = (df["y"].values[1:] + df["y"].values[:-1]) / 2
        df_all["z"] = (df["z"].values[1:] + df["z"].values[:-1]) / 2

        if grid_mode in ["layers", "new_resolution"]:
            dic_facies_path = {}
            # retrieve lithologies along the pathlines
            for fa in self.T1.get_all_facies():
                id_fa = fa.ID
                prop_fa = self.upscaled_facies[id_fa]

                facies_along_path = prop_fa[cells_path[:, 0], cells_path[:, 1], cells_path[:, 2]]
                dic_facies_path[fa.ID] = facies_along_path
            colors_fa = []
            for k, v in dic_facies_path.items():
                df_all["facies_prop_"+ str(k)] = v[:-1]
                colors_fa.append(self.T1.get_facies_obj(ID=k, type="ID").c)

        elif grid_mode == "archpy":
            facies = self.T1.get_facies(iu, ifa, all_data=False)
            facies_along_path = facies[cells_path[:, 0], cells_path[:, 1], cells_path[:, 2]]
            df_all["facies"] = facies_along_path[:-1]

        else:
            raise ValueError
        return df_all

    def prt_get_pathlines(self, i_particle=None):
        """
        Retrieve the raw pathlines from the particle tracking simulation
        """
        sim_dir = self.sim_prt.sim_path
        csv_name = self.sim_prt.prt[0].oc.trackcsv_filerecord.array[0][0]
        csv_path = os.path.join(sim_dir, csv_name)
        df = pd.read_csv(csv_path)

        if i_particle is not None:
            df = df.loc[df["irpt"] == i_particle]

            # filter particle results to remove issues along the paths
            # df = df.groupby("icell").first().sort_values("t").reset_index()
            pd.concat((df.iloc[:2], df.iloc[2:].groupby("icell").first().sort_values("t").reset_index()), axis=0)
        else:
            # filter all particles results to remove issues along the paths (and keep first two lines which are always true ?)
            # df = df.groupby(["irpt", "icell"]).first().sort_values(["irpt", "t"]).reset_index()
            df = pd.concat((df.groupby(["irpt", "icell"]).first().reset_index(), df.groupby(["irpt"]).nth(1))).sort_values(["irpt", "t"]).reset_index(drop=True)
        
        # we also need to add origin to coordinate as it is not considerd in the csv file
        ox = self.get_gwf().modelgrid.xoffset
        oy = self.get_gwf().modelgrid.yoffset

        df["x"] += ox
        df["y"] += oy

        return df

    def prt_get_facies_path_particle(self, i_particle=1, fac_time = 1/86400, iu = 0, ifa = 0):
        """
        Function to retrieve the facies sequences along a pathline
        """
        grid_mode = self.grid_mode
        df = self.prt_get_pathlines(i_particle)

        if df.shape[0] == 0:
            print("No particle found")
            return None
            
        time_ordered = df["t"].values.copy()
        time_ordered *= fac_time
        dt = np.diff(time_ordered)

        # add a column to track distance traveled
        distances = ((df[["x", "y", "z"]].iloc[1:].values - df[["x", "y", "z"]].iloc[:-1].values)**2).sum(1)**0.5

        # store everything in a new dataframe
        df_all = pd.DataFrame(columns=["dt", "time", "distance", "cum_distance", "x", "y", "z"])
        df_all["dt"] = dt
        df_all["time"] = time_ordered[:-1]
        df_all["distance"] = distances
        df_all["cum_distance"] = df_all["distance"].cumsum()
        df_all["x"] = (df["x"].values[1:] + df["x"].values[:-1]) / 2
        df_all["y"] = (df["y"].values[1:] + df["y"].values[:-1]) / 2
        df_all["z"] = (df["z"].values[1:] + df["z"].values[:-1]) / 2

        if grid_mode == "disv":
            cells_path = points2grid_index(df_all[["x", "y", "z"]].values, self.get_gwf().modelgrid)
            cells_path = np.array(cells_path)
            # cells_path[:, 0] += 1  # add 1 to the layer index to match modflow convention
        else:
            # cells_path = np.array((((df_all["z"].values-self.T1.zg[0])//self.T1.sz).astype(int),
            #                         ((df_all["y"].values-self.T1.yg[0])//self.T1.sy).astype(int),
            #                         ((df_all["x"].values-self.T1.xg[0])//self.T1.sx).astype(int))).T
            nx = self.get_gwf().modelgrid.ncol
            ny = self.get_gwf().modelgrid.nrow
            cells_path = get_locs(df.icell-1, nx, ny)
            cells_path = np.array(cells_path)[1:]

            # check that no cells path exceed the grid
            cells_path[:, 0][cells_path[:, 0] >= self.T1.nz] = self.T1.nz - 1
            cells_path[:, 1][cells_path[:, 1] >= self.T1.ny] = self.T1.ny - 1
            cells_path[:, 2][cells_path[:, 2] >= self.T1.nx] = self.T1.nx - 1

        if grid_mode in ["layers", "new_resolution"]:
            dic_facies_path = {}
            # retrieve lithologies along the pathlines
            for fa in self.T1.get_all_facies():

                id_fa = fa.ID
                prop_fa = self.upscaled_facies[id_fa]
                prop_fa = np.flip(prop_fa, axis=1)

                facies_along_path = prop_fa[cells_path[:, 0], cells_path[:, 1], cells_path[:, 2]]
                dic_facies_path[fa.ID] = facies_along_path
            colors_fa = []
            for k, v in dic_facies_path.items():
                df_all["facies_prop_"+ str(k)] = v
                colors_fa.append(self.T1.get_facies_obj(ID=k, type="ID").c)

        elif grid_mode == "archpy":
            facies = self.T1.get_facies(iu, ifa, all_data=False)
            facies = np.flip(np.flipud(facies), axis=1)
            facies_along_path = facies[cells_path[:, 0], cells_path[:, 1], cells_path[:, 2]]
            df_all["facies"] = facies_along_path
        
        elif grid_mode == "disv":
            dic_facies_path = {}
            # retrieve lithologies along the pathlines
            for fa in self.T1.get_all_facies():
                id_fa = fa.ID
                prop_fa = self.upscaled_facies[id_fa]

                facies_along_path = prop_fa[cells_path[:, 0], cells_path[:, 1]]
                dic_facies_path[fa.ID] = facies_along_path

            colors_fa = []
            for k, v in dic_facies_path.items():
                df_all["facies_prop_"+ str(k)] = v
                colors_fa.append(self.T1.get_facies_obj(ID=k, type="ID").c)
        else:
            raise ValueError
        
        return df_all

    ## transport model ##
    def create_sim_transport(self, strt_conc=0.0, porosity=0.2, diff=1e-9, 
                            adv_scheme="upstream",
                            alh=None, ath1=None,
                            decay=0.0, decay_0=False, decay_1=False):

        """
        Create a simulation object for a transport model with some default values
        """

        # paths
        sim_name = self.sim_name
        workspace = self.model_dir
        exe_name = self.exe_name
        gwf = self.get_gwf()

        # create simulation
        gwtname = "gwt-" + sim_name
        sim_ws = os.path.join(workspace, gwtname)
        sim_t = fp.mf6.MFSimulation(sim_name=sim_name, sim_ws=sim_ws, exe_name=exe_name)

        # Instantiating MODFLOW 6 groundwater transport model
        gwt = fp.mf6.MFModel(sim_t, model_type="gwt6", modelname=gwtname, model_nam_file=f"{gwtname}.nam" )

        # TDIS
        perioddata = [(1, 100, 1.0)]
        tdis = fp.mf6.ModflowTdis(sim_t, time_units='SECONDS', perioddata=perioddata)

        # IMS
        imsgwt = fp.mf6.ModflowIms(sim_t, print_option="SUMMARY", complexity="moderate")

        # DIS
        dis = self.get_gwf().dis

        grid_type = dis.package_type
        if grid_type == "dis":
            dis_t = fp.mf6.ModflowGwfdis(gwt, nlay=dis.nlay.array, nrow=dis.nrow.array, ncol=dis.ncol.array,
                                            delr=dis.delr.array, delc=dis.delc.array, top=dis.top.array, botm=dis.botm.array,
                                            xorigin=dis.xorigin.array, yorigin=dis.yorigin.array,
                                            idomain=dis.idomain.array)
        elif grid_type == "disv":
            dis_t = fp.mf6.ModflowGwfdisv(gwt, nlay=dis.nlay.array, ncpl=dis.ncpl.array, 
                                            top=dis.top.array, botm=dis.botm.array,
                                            xorigin=dis.xorigin.array, yorigin=dis.yorigin.array,
                                            vertices=dis.vertices.array, cell2d=dis.cell2d.array, nvert=dis.nvert.array,
                                            idomain=dis.idomain.array)
        
        elif grid_type == "disu":
            dis_t = fp.mf6.ModflowGwfdisu(gwt, nodes=dis.nodes.array, nja=dis.nja.array, area=dis.area.array,
                                            iac=dis.iac.array, jac=dis.jac.array, ihc=dis.ihc.array, cl12=dis.cl12.array, hwa=dis.hwa.array,
                                            top=dis.top.array, botm=dis.bot.array,
                                            xorigin=dis.xorigin.array, yorigin=dis.yorigin.array,
                                            vertices=dis.vertices.array, cell2d=dis.cell2d.array, nvert=dis.nverts.array,
                                            idomain=dis.idomain.array)
        
        # IC
        ict = fp.mf6.ModflowGwtic(gwt, strt=strt_conc, filename=f"{gwtname}.ic")

        # ADV (advection)
        adv = fp.mf6.ModflowGwtadv(gwt, scheme=adv_scheme, filename=f"{gwtname}.adv")

        # DSP (dispersion)$
        if alh is None:
            alh = max(max(dis.delr.array), max(dis.delc.array))
        
        if ath1 is None:
            ath1 = alh

        dsp = fp.mf6.ModflowGwtdsp(gwt, alh=alh, ath1=ath1, diffc=diff, pname="DSP", filename=f"{gwtname}.dsp")
        
        # MST
        mst = fp.mf6.ModflowGwtmst(
            gwt,
            porosity=porosity,
            zero_order_decay=decay_0,
            first_order_decay=decay_1,
            decay=decay,
            decay_sorbed=None,
            sorption=None,
            bulk_density=None,
            distcoef=None,
            filename=f"{gwtname}.mst",
        )

        # OC --> Instantiating MODFLOW 6 transport output control package
        oc_t = fp.mf6.ModflowGwtoc(
            gwt,
            budget_filerecord=f"{gwtname}.cbc",
            concentration_filerecord=f"{gwtname}.ucn",
            concentrationprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
            saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "ALL")],
            printrecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
        )

        # # Instantiating MODFLOW 6 flow-transport exchange mechanism
        # gwfname = self.get_gwf().name
        # sim = self.get_sim()
        # fp.mf6.ModflowGwfgwt(
        #     sim,
        #     exgtype="GWF6-GWT6",
        #     exgmnamea=gwfname,
        #     exgmnameb=gwtname,
        #     filename=f"{sim.name}.gwfgwt",
        # )

        # Instantiating MODFLOW 6 Flow-Model Interface package
        pd = [
            ("GWFHEAD", f"../{gwf.name}.hds", None),
            ("GWFBUDGET", f"../{gwf.name}.cbc", None),
        ]
        fmi = fp.mf6.ModflowGwtfmi(gwt, packagedata=pd)

        # register the packages
        self.sim_t = sim_t
        self.gwt = gwt


    def create_ssm_t(self, sourcerecarray):
        """
        Create the ssm package for the transport model

        Parameters
        ----------
        sourcerecarray : list of tuples
            list of tuples where each indicate a gwf package which has auxiliar variables (package name, AUX, auxname)
        """

        sim_t = self.get_sim_transport()
        gwt = self.get_gw_transport()
        gwtname = gwt.name

        # remove existing ssm package
        gwt.remove_package("ssm")

        # Source and sink mixing package --> make the link with the groundwater model
        ssm = fp.mf6.ModflowGwtssm(gwt, sources=sourcerecarray , filename=f"{gwtname}.ssm", save_flows=True)

    def get_sim_transport(self):
        """
        Get the MODFLOW 6 energy simulation object
        """
        return self.sim_t
    
    def get_gw_transport(self):
        """
        Get the MODFLOW 6 groundwater energy object
        """
        return self.gwt

   # set exisiting packages #
    def set_imsgwt(self, **kwargs):
        """
        Create the ims package for the transport model
        """
        sim_t = self.get_sim_transport()

        # remove existing ims package
        sim_t.remove_package("ims")

        imsgwt = fp.mf6.ModflowIms(sim_t, print_option="SUMMARY", **kwargs)

    def set_tdisgwt(self, perioddata):
        """
        Create the tdis package for the transport model

        Parameters
        ----------
        perioddata : list of tuples
            list of tuples where each tuple is a period data (perlen, nstp, tsmult)
        """
        sim_t = self.get_sim_transport()
        
        # remove existing tdis package
        sim_t.remove_package("tdis")

        # TDIS
        tdis = fp.mf6.ModflowTdis(sim_t, time_units='SECONDS', perioddata=perioddata)

    def set_strt_conc(self, strt_conc):
        """
        Set the initial concentration for the transport model

        Parameters
        ----------
        strt_conc : float or array of model dimension
            initial concentration
        """
        gwt = self.get_gw_transport()
        gwt.remove_package("ic")
        eic = fp.mf6.ModflowGweic(gwt, strt=strt_conc, filename=f"{gwt.name}.ic")

    def set_oc_t(self, saverecord, printrecord, concentrationprintrecord):
        """
        Set the output control for the transport model

        Parameters
        ----------
        budget_filerecord : str
            name of the budget file
        concentration_filerecord : str
            name of the concentration file
        """
        gwt = self.get_gw_transport()
        gwt.remove_package("oc")
        oc_t = fp.mf6.ModflowGwtoc(
            gwt,
            budget_filerecord=f"{gwt.name}.cbc",
            concentration_filerecord=f"{gwt.name}.ucn",
            concentrationprintrecord=concentrationprintrecord,
            saverecord=saverecord,
            printrecord=printrecord,
        )

    def set_dsp(self, alh=None, ath1=None, alv=None, ath2=None, atv=None,
                diffc=1e-9,
                xt3d_off=False, xt3d_rhs=False,
                vb=1):
        """
        Set the dispersion package for the transport model
        Note that this package reinitalize pre assigned values

        Parameters
        ----------
        alh : float
            longitudinal dispersivity if flow is horizontal
        ath1 : float
            transverse dispersivity (in y direction) if flow is horizontal
        ath2 : float
            transverse dispersivity (in z direction) if flow is horizontal
        alv : float
            longitudinal dispersivity if flow is vertical
        atv : float
            transverse dispersivity (in xy direction) if flow is vertical
        diffc : float
            molecular diffusion coefficient
        xt3d_off : bool 
            xt3d formulation
        """

        assert self.sim_t is not None, "You need to create a transport simulation first, see :meth:`create_sim_transport`"

        gwt = self.get_gw_transport()
        gwtname=gwt.name
        if xt3d_off is None:
            xt3d_off = False

        gwt = self.get_gw_transport()

        # get upscaled properties if necessary
        new_par = []
        for par in [alh, alv, ath1, ath2, atv, diffc]:
            if isinstance(par, str):
                par = self.upscale_prop(par)
            new_par.append(par)
        alh, alv, ath1, ath2, atv, diffc = new_par
        
        # remove existing cnd package
        gwt.remove_package("dsp")

        # ensure that alh and ath1 are not None
        if gwt.dis.package_type == "dis":
            if alh is None:
                alh = max(max(gwt.dis.delr.array), max(gwt.dis.delc.array))
            
            if ath1 is None:
                ath1 = alh
        else:
            pass
            # TO DO

        # dispersion package
        dsp = fp.mf6.ModflowGwtdsp(gwt, alh=alh, ath1=ath1, alv=alv, ath2=ath2, atv=atv,
                                    diffc=diffc, pname="DSP", filename=f"{gwtname}.dsp",
                                    xt3d_off=xt3d_off)

        if vb:
            print("dsp package updated")

    def set_mst(self, porosity=None, 
                decay=None, decay_0=False, decay_1=False,
                decay_sorbed=None,
                sorption=None, bulk_density=None, distcoef=None, sp2=None,
                vb=1):

        """
        Set the mass storage package for the transport model
        To assign a property from an Archpy model, use the name of the property as a string
        
        Parameters
        ----------
        porosity : float, array or string
            porosity of the medium
        decay : float, array or string
            decay coefficient. A negative value indicates production
        decay_0 : bool
            if True, decay is a zero order decay
        decay_1 : bool
            if True, decay is a first order decay
        decay_sorbed : float, array or string
            decay coefficient for the sorbed phase
        sorption : string
            sorption keyword to indicate that soprtion is used.
            Options are: linear, langmuir, freundlich
        bulk_density : float, array or string
            bulk density of the medium in mass/length^3
            Bulk density corresponds to the amount of solid per unit volume of the medium
        distcoef : float, array or string
            distribution coefficient for the equilibrium sorption. unit is length^3/mass
        sp2 : float, array or string
            exponent for the freundlich isotherm
        """

        assert self.sim_t is not None, "You need to create a transport simulation first, see :meth:`create_sim_transport`"

        gwt = self.get_gw_transport()
        gwtname=gwt.name

        gwt = self.get_gw_transport()
        
        # if porosity is None, keep previous value
        if porosity is None:
            porosity = gwt.mst.porosity.array
        else:
            # get upscaled properties if necessary
            if isinstance(porosity, str):
                porosity = self.upscale_prop(porosity)
            

        # get upscaled properties if necessary
        new_par = []
        for par in [decay, decay_sorbed, bulk_density, distcoef, sp2]:
            if isinstance(par, str):
                par = self.upscale_prop(par)
            new_par.append(par)
        decay, decay_sorbed, bulk_density, distcoef, sp2 = new_par
        
        # remove existing mst package
        gwt.remove_package("mst")

        # mass storage package
        mst = fp.mf6.ModflowGwtmst(
            gwt,
            porosity=porosity,
            zero_order_decay=decay_0,
            first_order_decay=decay_1,
            decay=decay,
            decay_sorbed=decay_sorbed,
            sorption=sorption,
            bulk_density=bulk_density,
            distcoef=distcoef,
            sp2=sp2,
            filename=f"{gwtname}.mst",
        )

        if vb:
            print("mst package updated")

    ## energy model ##
    def create_sim_energy(self,
                          strt_temp = 10,  # initial temperature
                          ktw = 0.56,  # thermal conductivity of water W/mK
                          kts=2.5,  # thermal conductivity of material W/mK
                          al = 1,  # dispersivity in longitudinal direction m
                          ath1 = 1,  # dispersivity in transverse direction m 
                          prsity = 0.2,  # porosity
                          cpw = 4186,  # specific heat capacity of water J/kg·K
                          cps = 840,  # specific heat capacity of solid J/kg·K
                          rhow = 1000,  # density of water kg/m3
                          rhos = 2650,  # density of solid kg/m3
                          lhv = 2.26e6  # latent heat of vaporization J/kg
                          ):

        """
        Create a simulation object for an energy model with some default values
        """

        # paths
        sim_name = self.sim_name
        workspace = self.model_dir
        exe_name = self.exe_name
        gwf = self.get_gwf()

        # create simulation
        gwename = "gwe-" + sim_name
        sim_ws = os.path.join(workspace, gwename)
        sim_e = fp.mf6.MFSimulation(sim_name=sim_name, sim_ws=sim_ws, exe_name=exe_name)

        # Instantiating MODFLOW 6 groundwater transport model
        gwe = fp.mf6.MFModel(sim_e, model_type="gwe6", modelname=gwename, model_nam_file=f"{gwename}.nam")

        # TDIS
        perioddata = [(1, 1, 1.0)]
        tdis = fp.mf6.ModflowTdis(sim_e, time_units='SECONDS', perioddata=perioddata)

        # IMS
        imsgwe = fp.mf6.ModflowIms(sim_e, print_option="SUMMARY", complexity="moderate")
        # sim_e.register_ims_package(imsgwe, [gwe.name])

        # DIS 
        dis = self.get_gwf().dis
        # dis_e = fp.mf6.ModflowGwfdis(gwe, nlay=dis.nlay.array, nrow=dis.nrow.array, ncol=dis.ncol.array,
        #                                 delr=dis.delr.array, delc=dis.delc.array, top=dis.top.array, botm=dis.botm.array,
        #                                 idomain=dis.idomain.array)

        # determine grid type
        grid_type = dis.package_type
        if grid_type == "dis":
            dis_e = fp.mf6.ModflowGwfdis(gwe, nlay=dis.nlay.array, nrow=dis.nrow.array, ncol=dis.ncol.array,
                                            delr=dis.delr.array, delc=dis.delc.array, top=dis.top.array, botm=dis.botm.array,
                                            xorigin=dis.xorigin.array, yorigin=dis.yorigin.array,
                                            idomain=dis.idomain.array)
        elif grid_type == "disv":
            dis_e = fp.mf6.ModflowGwfdisv(gwe, nlay=dis.nlay.array, ncpl=dis.ncpl.array, 
                                            top=dis.top.array, botm=dis.botm.array,
                                            xorigin=dis.xorigin.array, yorigin=dis.yorigin.array,
                                            vertices=dis.vertices.array, cell2d=dis.cell2d.array, nvert=dis.nvert.array,
                                            idomain=dis.idomain.array)
        
        elif grid_type == "disu":
            # raise ValueError("disu grid type is not compatible with particle tracking yet")  # to do: add disu grid type
            dis_e = fp.mf6.ModflowGwfdisu(gwe, nodes=dis.nodes.array, nja=dis.nja.array, area=dis.area.array,
                                            iac=dis.iac.array, jac=dis.jac.array, ihc=dis.ihc.array, cl12=dis.cl12.array, hwa=dis.hwa.array,
                                            top=dis.top.array, botm=dis.bot.array,
                                            xorigin=dis.xorigin.array, yorigin=dis.yorigin.array,
                                            vertices=dis.vertices.array, cell2d=dis.cell2d.array, nvert=dis.nverts.array,
                                            idomain=dis.idomain.array)

        # Instantiating MODFLOW 6 heat transport initial temperature
        eic = fp.mf6.ModflowGweic(gwe, strt=strt_temp, filename=f"{gwename}.ic")

        # Instantiating MODFLOW 6 heat transport advection package
        eadv = fp.mf6.ModflowGweadv(gwe, scheme="TVD", filename=f"{gwename}.adv")

        # conduction and dispersion 
        cnd = fp.mf6.ModflowGwecnd(gwe, alh=al, ath1=ath1, ktw=ktw, kts=kts, pname="CND", filename=f"{gwename}.dsp")

        # Instantiating MODFLOW 6 heat transport mass storage package (consider renaming to est)
        est = fp.mf6.ModflowGweest(gwe, porosity=prsity, heat_capacity_water=cpw, density_water=rhow,
                                   latent_heat_vaporization=lhv, heat_capacity_solid=cps, density_solid=rhos,
                                   pname="EST", filename=f"{gwename}.est", 
                                   save_flows=True,
                                   )

        # Instantiating MODFLOW 6 heat transport output control package
        oc = fp.mf6.ModflowGweoc(
            gwe,
            budget_filerecord=f"{gwename}.cbc",
            temperature_filerecord=f"{gwename}.ucn",
            saverecord=[("TEMPERATURE", "ALL"), ("BUDGET", "ALL")],
            printrecord=[("BUDGET", "LAST")],
        )

        # Instantiating MODFLOW 6 Flow-Model Interface package
        pd = [
            ("GWFHEAD", f"../{gwf.name}.hds", None),
            ("GWFBUDGET", f"../{gwf.name}.cbc", None),
        ]
        fmi = fp.mf6.ModflowGwefmi(gwe, packagedata=pd)

        # register the packages
        self.sim_e = sim_e
        self.gwe = gwe

    def get_sim_energy(self):
        """
        Get the MODFLOW 6 energy simulation object
        """
        return self.sim_e
    
    def get_gw_energy(self):
        """
        Get the MODFLOW 6 groundwater energy object
        """
        return self.gwe

    def get_sim_prt(self):
        """
        Get the MODFLOW 6 particle tracking simulation object
        """
        return self.sim_prt

    # create additional packages (ssm, ctp, ...) #
    def create_ssm_e(self, sourcerecarray):
        """
        Create the ssm package for the energy model

        Parameters
        ----------
        sourcerecarray : list of tuples
            list of tuples where each indicate a gwf package which has auxiliar variables (package name, AUX, auxname)
        """

        sim_e = self.get_sim_energy()
        gwe = self.get_gw_energy()
        gwename = gwe.name

        # remove existing ssm package
        gwe.remove_package("ssm")

        # Source and sink mixing package --> make the link with the groundwater model
        ssm = fp.mf6.ModflowGwessm(gwe, sources=sourcerecarray , filename=f"{gwename}.ssm", save_flows=True)

    # set exisiting packages #
    def set_imsgwe(self, **kwargs):
        """
        Create the ims package for the energy model
        """
        sim_e = self.get_sim_energy()
        # IMS

        # remove existing ims package
        sim_e.remove_package("ims")

        imsgwe = fp.mf6.ModflowIms(sim_e, print_option="SUMMARY", **kwargs)

    def set_tdisgwe(self, perioddata):
        """
        Create the tdis package for the energy model

        Parameters
        ----------
        perioddata : list of tuples
            list of tuples where each tuple is a period data (perlen, nstp, tsmult)
        """
        sim_e = self.get_sim_energy()
        
        # remove existing tdis package
        sim_e.remove_package("tdis")

        # TDIS
        tdis = fp.mf6.ModflowTdis(sim_e, time_units='SECONDS', perioddata=perioddata)

    def set_strt_temp(self, strt_temp):
        """
        Set the initial temperature of the energy model

        Parameters
        ----------
        strt_temp : float or array of model dimension
            initial temperature
        """
        gwe = self.get_gw_energy()
        gwe.remove_package("ic")
        eic = fp.mf6.ModflowGweic(gwe, strt=strt_temp, filename=f"{gwe.name}.ic")
    
    def set_porosity(self, 
                     iu = 0, ifa = 0, ip = 0, 
                     porosity=None, prop_key="porosity",
                     packages=None):

        """
        Set the porosity of the model

        Parameters
        ----------
        iu : int
            unit simulation index
        ifa : int
            facies simulation index
        ip : int
            property simulation index
        porosity : float
            porosity value, if None, the porosity is taken from the table according to the prop_key
        prop_key : str
            key of the property in the table
        packages : list
            list of packages to update the porosity. Can be "mp", "prt" and "energy".
            If None, all existing packages are updated
        """
        
        gwf = self.get_gwf()
        if porosity is None:
            new_porosity = self.upscale_prop(prop_key, iu, ifa, ip)
        else:
            new_porosity = porosity

        # update porosity in required packages
        if self.mp is not None:
            mpbas = fp.modpath.Modpath7Bas(self.get_mp(), porosity=new_porosity)
        
        if self.sim_prt is not None:
            mip = self.get_sim_prt().get_model().mip
            mip.porosity.set_data(new_porosity)

        if self.sim_e is not None:
            self.get_gw_energy().get_package("est").porosity.set_data(new_porosity)

        if self.sim_t is not None:
            self.get_gw_transport().get_package("mst").porosity.set_data(new_porosity)

    def set_cnd(self, xt3d_off=None, xt3d_rhs=None,
                alh=1, alv=None, 
                ath1=1, ath2=None, atv=None, 
                ktw=0.56, kts=2.5, vb=1, **kwargs):
        """
        Set the conduction and dispersion package

        Parameters
        ----------
        xt3d_off : bool
            flag indicating whether the XT3D package is active
        xt3d_rhs : bool
            flag indicating whether the XT3D package is using the right-hand side formulation
        alh : float or array of model dimension or string
            longitudinal dispersivity (if flow is horizontal)
            If a string is provided, it is assumed to be the name of the property in the ArchPy table
        alv : float or array of model dimension or string
            longitudinal dispersivity (if flow is vertical)
        ath1 : float or array of model dimension or string
            horizontal (y direction) transverse dispersivity (if flow is directed in the x-direction)
        ath2 : float or array of model dimension or string
            horizontal (z direction) transverse dispersivity (if flow is directed in the x-direction)
        atv : float or array of model dimension or string
            transverse (x-y plane) dispersivity (if flow is vertical)
        ktw : float or array of model dimension or string
            thermal conductivity of water
        kts : float or array of model dimension or string
            thermal conductivity of solid
        vb : int (0 or 1)
            flag to print messages
        """
        
        assert self.sim_e is not None, "You need to create an energy simulation first, see :meth:`create_sim_energy`"
        gwe = self.get_gw_energy()

        # get upscaled properties if necessary
        new_par = []
        for par in [alh, alv, ath1, ath2, atv, ktw, kts]:
            if isinstance(par, str):
                par = self.upscale_prop(par)
            new_par.append(par)
        alh, alv, ath1, ath2, atv, ktw, kts = new_par
        
        # remove existing cnd package
        gwe.remove_package("cnd")

        # conduction and dispersion
        cnd = fp.mf6.ModflowGwecnd(gwe, alh=alh, alv=alv, ath1=ath1, ath2=ath2, atv=atv, ktw=ktw, kts=kts, **kwargs)

        if vb:
            print("cnd package updated")

    def set_est(self, cpw=4184.0, cps=840,
                rhow=1000, rhos=2500,
                lhv=2453500.0,
                porosity=None,
                save_flows=True, vb=1, **kwargs):
        """
        Set the heat transport mass storage package

        Parameters
        ----------
        cpw : float
            specific heat capacity of the fluid
        cps : float or array of model dimension or string
            specific heat capacity of the solid
            if a string is provided, it is assumed to be the name of the property object (see :class:`base.Prop`)
        rhow : float
            density of the fluid
        rhos : float or array of model dimension or string
            density of the solid
            if a string is provided, it is assumed to be the name of the property in the ArchPy table
        lhv : float
            latent heat of vaporization
        porosity : float or array of model dimension or string
            porosity. Note that it is preferable to update 
            the porosity using the :meth:`set_porosity` method.
            As this method only update porosity in the est package but 
            not in the other packages (mp, prt, ...)
            if a string is provided, it is assumed to be the name of the property in the ArchPy table
        save_flows : bool
            flag to save flows in the cell budget file
        vb : int (0 or 1)
            flag to print messages
        """

        assert self.sim_e is not None, "You need to create an energy simulation first, see :meth:`create_sim_energy`"
        gwe = self.get_gw_energy()

        # get upscaled properties if necessary
        new_par = []
        for par in [porosity, cps, rhos]:
            if isinstance(par, str):
                par = self.upscale_prop(par)
            new_par.append(par)

        porosity, cps, rhos = new_par

        # update values
        est = gwe.get_package("est")

        est.heat_capacity_water.set_data(cpw)
        est.heat_capacity_solid.set_data(cps)
        est.density_solid.set_data(rhos)
        est.density_Water.set_data(rhow)
        est.latent_heat_vaporization.set_data(lhv)

        if porosity is not None:
            est.porosity.set_data(porosity)
        
        if vb:
            print("est package updated")



