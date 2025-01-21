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

# some functions

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

def plot_particle_facies_sequence(arch_table, df, plot_time=False, plot_distance=False):

    if plot_time and plot_distance:
        fig, ax = plt.subplots(2,1, figsize=(10, 1.5), dpi=200)
        axi = ax[0]
    elif plot_time or plot_distance:
        fig, axi = plt.subplots(1,1, figsize=(10, 0.5), dpi=200)
    plt.subplots_adjust(hspace=1.5)


    if plot_time:
        dt = df["dt"]
        for i, (facies, time) in enumerate(zip(df["facies"], df["time"])):
            if i > 0:
                axi.barh(0, dt[i], left=df["time"].loc[i-1], color=arch_table.get_facies_obj(ID=facies, type="ID").c)
            else:
                axi.barh(0, dt[i], left=0, color=arch_table.get_facies_obj(ID=facies, type="ID").c, label=arch_table.get_facies_obj(ID=facies, type="ID").name)
        
        axi.set_xlim(0, df["time"].iloc[-1])
        axi.set_xlabel("Time (days)")
        axi.set_yticks([])

        if plot_distance:
            axi = ax[1]
    
    if plot_distance:

        # plot facies function of the distance traveled
        all_dist = df["distance"].values
        all_cum_dist = df["cum_distance"].values

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

## archpy to modflow class ##
class archpy2modflow:

    """
    Class to convert an ArchPy table to a MODFLOW 6 model

    Parameters
    ----------
    T1 : Arch_table
        ArchPy table to convert
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
        self.grid_mode = None
        self.layers_names = None
        self.list_active_cells = None
        self.mp = None
        self.factor_x = None
        self.factor_y = None
        self.factor_z = None

    def create_sim(self, grid_mode="archpy", iu=0, factor_x=None, factor_y=None, factor_z=None):

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
        factor_x : float
            factor to change the resolution of the grid in the x direction. e.g. 2 means that the resolution will be divided by 2
        factor_y : float
            factor to change the resolution of the grid in the y direction.
        factor_z : float    
            factor to change the resolution of the grid in the z direction.
        """

        sim = fp.mf6.MFSimulation(sim_name=self.sim_name, version='mf6', exe_name=self.exe_name, 
                         sim_ws=self.model_dir)
        gwf = fp.mf6.ModflowGwf(sim, modelname=self.model_name,
                                model_nam_file='{}.nam'.format(self.model_name))

        #grid
        nlay, nrow, ncol = self.T1.get_nz(), self.T1.get_ny(), self.T1.get_nx()
        delr, delc = self.T1.get_sx(), self.T1.get_sy()
        xoff, yoff = self.T1.get_ox(), self.T1.get_oy()

        if grid_mode == "archpy":
            top = np.ones((nrow, ncol)) * self.T1.get_zg()[-1]
            botm = np.ones((nlay, nrow, ncol)) * self.T1.get_zg()[:-1].reshape(-1, 1, 1)
            botm = np.flip(np.flipud(botm), axis=1)  # flip the array to have the same orientation as the ArchPy table
            idomain = np.flip(np.flipud(self.T1.get_mask().astype(int)), axis=1)  # flip the array to have the same orientation as the ArchPy table

        elif grid_mode == "layers":
            # get surfaces of each unit
            top = self.T1.get_surface(typ="top")[0][0, iu]
            top = np.flip(top, axis=1)
            botm = self.T1.get_surface(typ="bot")[0][:, iu]
            botm = np.flip(botm, axis=1)
            layers_names = self.T1.get_surface(typ="bot")[1]
            self.layers_names = layers_names
            nlay = botm.shape[0]

            # define idomain (1 if thickness > 0, 0 if nan, -1 if thickness = 0)
            idomain = np.ones((nlay, nrow, ncol))
            thicknesses = -np.diff(np.vstack([top.reshape(-1, nrow, ncol), botm]), axis=0)
            idomain[thicknesses == 0] = -1
            idomain[np.isnan(thicknesses)] = 0

            # adapt botm in order that each layer has a thickness > 0 
            for i in range(-1, nlay-1):
                if i == -1:
                    s1 = top
                else:
                    s1 = botm[i]
                s2 = botm[i+1]
                mask = s1 == s2
                s1[mask] += 1e-2

                # 2nd loop over previous layers to ensure that the thickness is > 0
                for o in range(i, -1, -1):
                    s2 = botm[o]
                    if o == 0:
                        s1 = top
                    else:
                        s1 = botm[o-1]
                    mask = s1 <= s2
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
            # botm = np.flip(np.flipud(botm), axis=1)  # flip the array to have the same orientation as the ArchPy table
            
            # how to define idomain ?
            idomain = np.zeros((nlay, nrow, ncol))
            mask_org = self.T1.get_mask().astype(int)
            for ilay in range(0, self.T1.get_nz(), factor_z):
                for irow in range(0, self.T1.get_ny(), factor_y):
                    for icol in range(0, self.T1.get_nx(), factor_x):
                        mask = mask_org[ilay:ilay+factor_z, irow:irow+factor_y, icol:icol+factor_x]
                        if mask.mean() >= 0.5:
                            idomain[ilay//factor_z, irow//factor_y, icol//factor_x] = 1
            
            self.factor_x = factor_x
            self.factor_y = factor_y
            self.factor_z = factor_z
            
        else:
            raise ValueError("grid_mode must be one of 'archpy', 'layers' or 'new_resolution'")

        # save grid mode
        self.grid_mode = grid_mode
        
        assert (np.array(check_thk(top, botm))).all(), "Error in the processing of the surfaces, some cells have a thickness < 0"

        dis = fp.mf6.ModflowGwfdis(gwf, nlay=nlay, nrow=nrow, ncol=ncol,
                                    delr=delr, delc=delc,
                                    top=top, botm=botm,
                                    xorigin=xoff, yorigin=yoff, 
                                    idomain=idomain)

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
        npf = fp.mf6.ModflowGwfnpf(gwf, icelltype=0, k=1, save_flows=True, save_saturation=True)

        self.sim = sim
        print("Simulation created")
        print("To retrieve the simulation, use the get_sim() method")
    
    def set_k(self, k_key="K",
              iu=0, ifa=0, ip=0,
              log=False, k=None, k22=None, k33=None, k_average_method="arithmetic", 
              upscaling_method="simplified_renormalization"):

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
                new_k = np.ones((nlay, nrow, ncol))

                # initialize variable for facies upscaling
                facies_arr = self.T1.get_facies(iu, ifa, all_data=False)
                upscaled_facies = {}

                for ifa in np.unique(facies_arr):
                    upscaled_facies[ifa] = np.zeros((facies_arr.shape[0], facies_arr.shape[1], facies_arr.shape[2]))

                if k_average_method == "anisotropic":
                    new_k33 = np.ones((nlay, nrow, ncol))
                else:
                    new_k33 = None
                layers = self.layers_names
                mask_units = [self.T1.unit_mask(l).astype(bool) for l in layers]

                for irow in range(nrow):
                    for icol in range(ncol):
                        for ilay in range(nlay):
                            mask_unit = mask_units[ilay]
                            if k_average_method == "arithmetic":
                                new_k[ilay, irow, icol] = np.mean(kh[:, irow, icol][mask_unit[:, irow, icol]])
                            elif k_average_method == "harmonic":
                                new_k[ilay, irow, icol] = 1 / np.mean(1 / kh[:, irow, icol][mask_unit[:, irow, icol]])
                            elif k_average_method == "anisotropic":
                                new_k[ilay, irow, icol] = np.mean(kh[:, irow, icol][mask_unit[:, irow, icol]])
                                new_k33[ilay, irow, icol] = 1 / np.mean(1 / kh[:, irow, icol][mask_unit[:, irow, icol]])
                            else:
                                raise ValueError("k_average_method must be one of 'arithmetic' or 'harmonic'")
                            
                            # facies upscaling
                            arr = facies_arr[:, irow, icol][mask_unit[:, irow, icol]]  # array of facies values in the unit
                            prop = get_proportion(arr)
                            for ifa in np.unique(arr):
                                upscaled_facies[ifa][:, irow, icol][mask_unit[:, irow, icol]] = prop[ifa]
                
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

                if log:
                    new_k = 10**new_k
                    if k_average_method == "anisotropic":
                        new_k33 = 10**new_k33

            elif grid_mode == "new_resolution":

                from ArchPy.uppy import upscale_k
                dx, dy, dz = self.T1.get_sx(), self.T1.get_sy(), self.T1.get_sz()
                
                factor_x = self.factor_x
                factor_y = self.factor_y
                factor_z = self.factor_z
                
                field = self.T1.get_prop(k_key)[iu, ifa, ip]
                field = np.flip(np.flipud(field), axis=1)  # flip the array to have the same orientation as the ArchPy table
                
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

                if log:
                    new_k = 10**new_k
                    new_k22 = 10**new_k22
                    new_k33 = 10**new_k33

                # facies upscaling
                facies_arr = self.T1.get_facies(iu, ifa, all_data=False)
                upscaled_facies = {}
                for ifa in np.unique(facies_arr):
                    upscaled_facies[ifa] = np.zeros((facies_arr.shape[0], facies_arr.shape[1], facies_arr.shape[2]))
                
                for ilay in range(0, self.T1.get_nz(), factor_z):
                    for irow in range(0, self.T1.get_ny(), factor_y):
                        for icol in range(0, self.T1.get_nx(), factor_x):
                            mask_unit = facies_arr[ilay:ilay+factor_z, irow:irow+factor_y, icol:icol+factor_x]
                            arr = mask_unit.flatten()
                            prop = get_proportion(arr)
                            for ifa in np.unique(arr):
                                upscaled_facies[ifa][ilay:ilay+factor_z, irow:irow+factor_y, icol:icol+factor_x] = prop[ifa]

                self.upscaled_facies = upscaled_facies
        else:
            new_k = k
            new_k22 = k22
            new_k33 = k33

        # new_k = np.flip(new_k, axis=1)  # we have to flip in order to match modflow grid
        npf = fp.mf6.ModflowGwfnpf(gwf, icelltype=0, k=new_k, k22=new_k22, k33=new_k33, save_flows=True, save_specific_discharge=True, save_saturation=True)
        npf.write()


    def set_strt(self, heads=None):
        """
        Set the starting heads
        """
        gwf = self.get_gwf()
        gwf.remove_package("ic")
        ic = fp.mf6.ModflowGwfic(gwf, strt=heads)
        ic.write()

    def get_list_active_cells(self):

        if self.list_active_cells is None:
            gwf = self.get_gwf()
            idomain = gwf.dis.idomain.array
            list_active_cells = []
            for ilay in range(idomain.shape[0]):
                for irow in range(idomain.shape[1]):
                    for icol in range(idomain.shape[2]):
                        if idomain[ilay, irow, icol] == 1:
                            list_active_cells.append((ilay, irow, icol))
            self.list_active_cells = list_active_cells
            
        return self.list_active_cells

    # get functions
    def get_sim(self):
        assert self.sim is not None, "You need to create the simulation first"
        return self.sim
    
    def get_gwf(self):
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

        # Create grid discretization
        dis = self.get_gwf().dis
        dis_prt = fp.mf6.ModflowGwfdis(prt, nlay=dis.nlay.array, nrow=dis.nrow.array, ncol=dis.ncol.array,
                                        delr=dis.delr.array, delc=dis.delc.array, top=dis.top.array, botm=dis.botm.array,
                                        idomain=dis.idomain.array)

        # create prt input package
        mip = fp.mf6.ModflowPrtmip(prt, pname="mip", porosity=0.2)

        from shapely.geometry import Point, MultiPoint

        grid = self.get_gwf().modelgrid
        ix = fp.utils.gridintersect.GridIntersect(mfgrid=grid)

        list_cellids = []
        for pi in list_p_coords:
            p1 = Point(pi)
            result = ix.intersect(p1)
            if len(result) > 0:
                list_cellids.append(result.cellids[0])
            else:
                p1 = Point((pi[0], pi[1]))
                result = ix.intersect(p1)
                list_cellids.append((-1, result.cellids[0][1], result.cellids[0][2]))

        cellids = np.array([cids for cids in list_cellids])

        cellids[:, 0] += 1

        # package data (irptno, cellid, x, y, z)
        package_data = []
        for i in range(len(cellids)):
            package_data.append((i, cellids[i], list_p_coords[i][0], list_p_coords[i][1], list_p_coords[i][2]))

        # period data (when to release the particles)
        period_data = {0: ["FIRST"]}

        prp = fp.mf6.ModflowPrtprp(prt, pname="prp", filename="{}.prp".format(prt_name),
                                        packagedata=package_data,
                                        perioddata=period_data,
                                        nreleasepts=len(package_data),
                                        exit_solve_tolerance=1e-5)


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

    def set_porosity(self, 
                     iu = 0, ifa = 0, ip = 0, 
                     porosity=None, k_key="porosity"):

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
            porosity value, if None, the porosity is taken from the table according to the k_key
        k_key : str
            key of the property in the table
        """
        
        gwf = self.get_gwf()
        if porosity is None:
            
            # check if the porosity is already in the table
            assert k_key in self.T1.get_prop_names(), "The property {} is not in the table".format(k_key)

            grid_mode = self.grid_mode
            if grid_mode == "archpy":

                k = self.T1.get_prop(k_key)[iu, ifa, ip]
                k = np.flip(np.flipud(k), axis=1)  # flip the array to have the same orientation as the ArchPy table
                new_k = k

            elif grid_mode == "layers":

                nrow, ncol, nlay = gwf.modelgrid.nrow, gwf.modelgrid.ncol, gwf.modelgrid.nlay

                kh = self.T1.get_prop(k_key)[iu, ifa, ip] 
                new_k = np.ones((nlay, nrow, ncol))

                layers = self.layers_names
                mask_units = [self.T1.unit_mask(l).astype(bool) for l in layers]

                for irow in range(nrow):
                    for icol in range(ncol):
                        for ilay in range(nlay):
                            mask_unit = mask_units[ilay]
                            new_k[ilay, irow, icol] = np.mean(kh[:, irow, icol][mask_unit[:, irow, icol]])

                # fill nan values with the mean of the layer
                for ilay in range(nlay):
                    mask = np.isnan(new_k[ilay])
                    new_k[ilay][mask] = np.nanmean(new_k[ilay])

                new_k = np.flip(new_k, axis=1)  # we have to flip in order to match modflow grid

            elif grid_mode == "new_resolution":
                pass
            
        else:
            new_k = porosity

        mpbas = fp.modpath.Modpath7Bas(self.get_mp(), porosity=new_k)

    def prt_run(self, silent=False):
        # write simulation to disk
        self.sim_prt.write_simulation()
        success, msg = self.sim_prt.run_simulation(silent=silent)
        if not success:
            print("particle tracking did not run successfully")
            print(msg)

    def mp_run(self, silent=False):
        self.mp.write_input()
        success, msg = self.mp.run_model(silent=silent)
        if not success:
            print("modpath did not run successfully")
            print(msg)

    # get results
    def get_mp(self):
        return self.mp
    
    def mp_get_pathlines_object(self):

        fpth = os.path.join(self.model_dir, f"{self.mpnamf}.mppth")
        p = fp.utils.PathlineFile(fpth)
        return p

    def mp_get_endpoints_object(self):

        fpth = os.path.join(self.model_dir, f"{self.mpnamf}.mpend")
        e = fp.utils.EndpointFile(fpth)
        return e

    def mp_get_facies_path_particle(self, i_particle, fac_time = 1/86400, iu = 0, ifa = 0):

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
        df["distance"] = 0
        for i in range(1, df.shape[0]):
            x0, y0, z0 = df[["x", "y", "z"]].iloc[i-1]
            x1, y1, z1 = df[["x", "y", "z"]].iloc[i]
            distance = np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
            df.loc[df.index[i], "distance"] = distance

        df["cum_distance"] = df["distance"].cumsum()

        # store everything in a new dataframe
        df_all = pd.DataFrame(columns=["dt", "time", "distance", "cum_distance", "x", "y", "z"])
        df_all["dt"] = dt
        df_all["time"] = time_ordered[:-1]
        df_all["distance"] = df["distance"].values[:-1]
        df_all["cum_distance"] = df["cum_distance"].values[:-1]
        df_all["x"] = df["x"].values[:-1]
        df_all["y"] = df["y"].values[:-1]
        df_all["z"] = df["z"].values[:-1]

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
        sim_dir = self.sim_prt.sim_path
        csv_name = self.sim_prt.prt[0].oc.trackcsv_filerecord.array[0][0]
        csv_path = os.path.join(sim_dir, csv_name)
        df = pd.read_csv(csv_path)

        if i_particle is not None:
            df = df.loc[df["irpt"] == i_particle]
        return df

    def prt_get_facies_path_particle(self, i_particle=1, fac_time = 1/86400, iu = 0, ifa = 0):

        grid_mode = self.grid_mode
        df = self.prt_get_pathlines(i_particle)

        time_ordered = df["t"].values.copy()
        time_ordered *= 1/86400
        dt = np.diff(time_ordered)

        # add a column to track distance traveled
        distances = ((df[["x", "y", "z"]].iloc[1:].values - df[["x", "y", "z"]].iloc[:-1].values)**2).sum(1)

        # store everything in a new dataframe
        df_all = pd.DataFrame(columns=["dt", "time", "distance", "cum_distance", "x", "y", "z"])
        df_all["dt"] = dt
        df_all["time"] = time_ordered[1:]
        df_all["distance"] = distances
        df_all["cum_distance"] = df_all["distance"].cumsum()
        df_all["x"] = (df["x"].values[1:] + df["x"].values[:-1]) / 2
        df_all["y"] = (df["y"].values[1:] + df["y"].values[:-1]) / 2
        df_all["z"] = (df["z"].values[1:] + df["z"].values[:-1]) / 2

        cells_path = np.array((((df_all["z"].values-self.T1.zg[0])//self.T1.sz).astype(int),
                                ((df_all["y"].values-self.T1.yg[0])//self.T1.sy).astype(int),
                                ((df_all["x"].values-self.T1.xg[0])//self.T1.sx).astype(int))).T

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
                df_all["facies_prop_"+ str(k)] = v
                colors_fa.append(self.T1.get_facies_obj(ID=k, type="ID").c)

        elif grid_mode == "archpy":
            facies = self.T1.get_facies(iu, ifa, all_data=False)
            facies_along_path = facies[cells_path[:, 0], cells_path[:, 1], cells_path[:, 2]]
            df_all["facies"] = facies_along_path
        else:
            raise ValueError
        
        return df_all


