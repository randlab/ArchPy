import numpy as np
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
import pyvista as pv
import scipy
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import uniform_filter
import copy
import time
import shapely.geometry
import sys
import numba
from numba import jit

#geone
import geone
import geone.covModel as gcm
import geone.grf as grf
from geone import img
import geone.imgplot as imgplt
import geone.imgplot3d as imgplt3
import geone.deesseinterface as dsi
import geone.geosclassicinterface as gci

#ArchPy modules
from ArchPy.ineq import *
from ArchPy.data_transfo import *
from ArchPy.tpgs import * #truncated plurigraussian
from ArchPy.inputs import * # inputs utilities


##### functions ######
def Arr_replace(arr, dic):

    """
    Replace value in an array using a dictionnary
    linking actual values to new values

    #inputs#
    arr: any nd.array with values
    dic: a dictionnary with arr values as keys and
          new values as dic values

    #output#
    A new array with values replaced
    """

    arr_old=arr.flatten().copy()
    arr_new=np.zeros([arr_old.shape[0]])
    for i, x in enumerate(arr_old):
        if x==x:
            arr_new[i]=dic[x]
        else:
            arr_new[i]=np.nan
    return arr_new.reshape(arr.shape)

def get_size(obj, seen=None):
    """Recursively finds size of objects,
       function taken from stack overflow by Aaron Hall"""
    size=sys.getsizeof(obj)
    if seen is None:
        seen=set()
    obj_id=id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def resample_to_grid(xc, yc, rxc, ryc, raster_band, method="nearest"):

    """
    !! function taken from flopy (3.3.4) !!
    Method to resample the raster data to a
    user supplied grid of x, y coordinates.

    x, y coordinate arrays should correspond
    to grid vertices

    Parameters
    ----------
    xc: np.ndarray or list
        an array of x-cell centers
    yc: np.ndarray or list
        an array of y-cell centers
    rxc: ndarray
         raster xcell centers
    ryc: ndarray
         raster ycell centers
    raster_band: 2D ndarray
        raster band to re-sample
    method: str
        scipy interpolation method options

        "linear" for bi-linear interpolation
        "nearest" for nearest neighbor
        "cubic" for bi-cubic interpolation

    Returns
    -------
        np.array
    """

    from scipy.interpolate import griddata

    #get some info and output grid
    data_shape=xc.shape
    xc=xc.flatten()
    yc=yc.flatten()

    # step 2: flatten raster grid
    rxc=rxc.flatten()
    ryc=ryc.flatten()

    #flatten array
    arr=raster_band.flatten()

    # interpolation
    data=griddata((rxc, ryc), arr, (xc, yc), method=method)

    #rearrange shape
    data=data.reshape(data_shape)

    return data


##### ArchPy functions #####
def interp2D(litho, xg, yg, xu, verbose=0, ncpu=1, mask2D=None, seed=123456789, **kwargs):

    """
    function to realize a 2D interpolation based on a
    multitude of methods (scipy.interpolate, geone (kriging, MPS, ...))

    #####
    inputs:
    litho: Surface object
    xg, yg: 1 ndarray of size nx, ny, central coordinates
             vectors of the simulation grid
    xu   : ndarray of size (n, 2), position at which
             we want to know the estimation
             (not used for every interp method)
    kwargs: different parameters for surface interpolation
        - nit: number of iterations for gibbs sampler
                 (depends on the number of data)
        - nmax: number of neighbours in grf
                 with inequalities (to speed up the simulations)
        - krig_type: ordinary or simple kriging for kriging
                 interpolation and grf inequalities
                 calculation of weights
        - mean: values or 2D array, mean value for the simulation if unco flag is set to True or no HD are found
        - unco: bool, unconditional or not

    ######
    outputs:
    s: array of same size as x, interpolated values
    """

    if hasattr(litho, "sto_x") and hasattr(litho, "sto_y") and hasattr(litho, "sto_z") and hasattr(litho, "sto_ineq"):
        # merge sto and real hd
        l_x = litho.x + litho.sto_x
        l_y = litho.y + litho.sto_y
        l_z = litho.z + litho.sto_z
        l_ineq = litho.ineq + litho.sto_ineq
    else:
        l_x = litho.x
        l_y = litho.y
        l_z = litho.z
        l_ineq = litho.ineq

    xp=np.array(l_x)
    yp=np.array(l_y)
    zp=np.array(l_z)
    ineq_data=l_ineq  #inequality data
    method=litho.int_method

    ##grid
    xg.sort()
    yg.sort()
    nx=len(xg)-1
    ny=len(yg)-1
    sx=xg[1] - xg[0]
    sy=yg[1] - yg[0]
    ox=xg[0]
    oy=yg[0]

    ##kwargs
    kwargs_def_grf={"nit": 50, "nmax": 20, "krig_type": "simple_kriging",  # number of gibbs sampler iterations, number of neigbors, krig type
                      "grf_method": "fft", "mean": None,"unco": False}  # and grf method, mean and unconditional flag

    kwargs_def_MPS={"unco": False,
                      "xr": 1, "yr": 1, "zr": 1, "maxscan": 0.25, "neig": 24, "thresh": 0.05, "xloc": False, "yloc": False, "zloc": False,
                      "homo_usage": 1, "rot_usage": 1, "rotAziLoc": False, "rotAzi": 0, "rotDipLoc": False, "rotDip": 0, "rotPlungeLoc": False, "rotPlunge": 0,
                      "radiusMode": "large_default", "rx": nx*sx, "ry": ny*sy, "rz": 1, "anisotropyRatioMode": "one", "ax": 1, "ay": 1, "az": 1,
                      "angle1": 0, "angle2": 0, "angle3": 0,
                      "relativeDistanceFlag": False, "rescalingMode": 'min_max', "TargetMin": None, "TargetMax": None, "TargetMean": None, "TargetLength": None} #continous params

    kw={}
    #assign default values
    if method in ["kriging", "grf", "grf_ineq"]:
        kw=kwargs_def_grf
    elif method.lower() == "mps":
        kw=kwargs_def_MPS

    for k, v in kw.items():
        if k not in kwargs.keys():
            kwargs[k]=v

    #if no data --> unco set to True
    if len(xp)+len(ineq_data) == 0:
        kwargs["unco"]=True
    else:
        kwargs["unco"]=False

    #mask
    if "mask" in kwargs.keys():
        mask2D = kwargs["mask"] & mask2D

    ##DATA
    # handle inequalities (setup equality points to lower/upper bounds of inequalities if krig ineq or GRF ineq are not used for the interpolation)

    if len(litho.ineq) == 0 and method == "grf_ineq":
        method = "grf"

    x_in=[]
    y_in=[]
    z_in=[]
    if method in ["kriging", "cubic", "linear", "nearest", "grf"]:
        if litho.get_surface_covmodel(vb=0) is not None and len(ineq_data) > 0:

            # gibbs sampler to estimate values at inequality point, requires a covmodel
            eq_d=np.array([xp, yp, zp]).T
            dmy=np.nan*np.ones([2, len(litho.x)+len(litho.sto_x)]).T
            eq_d=np.concatenate([eq_d, dmy], 1)  # append all data together in right format
            all_data=np.concatenate([eq_d, np.array(ineq_data)])
            all_data=ArchPy.ineq.Gibbs_estimate(all_data, litho.get_surface_covmodel(), krig_type="simple_kriging", nit=50)  # Gibbs sampler

            xp=all_data[:, 0]
            yp=all_data[:, 1]
            zp=all_data[:, 2]

        else:
            for in_data in ineq_data:  # handle inequality with non ineq methods
                if (in_data[3] == in_data[3]) & (in_data[4] != in_data[4]): # inf ineq
                    x_in.append(in_data[0])
                    y_in.append(in_data[1])
                    z_in.append(in_data[3])

                elif (in_data[3] != in_data[3]) & (in_data[4] == in_data[4]): # sup ineq
                    x_in.append(in_data[0])
                    y_in.append(in_data[1])
                    z_in.append(in_data[4])

                elif (in_data[3] == in_data[3]) & (in_data[4] == in_data[4]): # sup and inf ineq
                    x_in.append(in_data[0])
                    y_in.append(in_data[1])
                    z_in.append((in_data[3]+in_data[4])/2)

            # append data
            xp=np.concatenate((xp, x_in))
            yp=np.concatenate((yp, y_in))
            zp=np.concatenate((zp, z_in))

        data=np.concatenate([xp.reshape(-1, 1), yp.reshape(-1, 1)], axis=1)

    ## dealt with inequality data in the right format
    elif method.lower() in ["grf_ineq", "mps"]:

        # equality data
        x_eq=np.array([xp, yp]).T
        v_eq=zp

        if len(litho.ineq) == 0:
            xIneq_min = None
            vIneq_min = None
            xIneq_max = None
            vIneq_max = None

        elif len(litho.ineq) > 0:
            #ineq
            ineq_data=np.array(litho.ineq)
            mask=(ineq_data[:, 3] == ineq_data[:, 3]) # inf boundary
            xIneq_min=ineq_data[:,: 2][mask]
            vIneq_min=ineq_data[:, 3][mask]

            mask=(ineq_data[:, 4] == ineq_data[:, 4]) # sup boundary
            xIneq_max=ineq_data[:,: 2][mask]
            vIneq_max=ineq_data[:, 4][mask]


    ### interpolations methods ###
    if method.lower() in ["linear", "cubic", "nearest"]: # spline methods
        if kwargs["unco"] == False:
            s=scipy.interpolate.griddata(np.array([xp, yp]).T, zp, xu, method=method, fill_value=np.mean(zp))
            s=s.reshape(ny, nx)
        else:
            raise ValueError ("Error: No data point found or unconditional spline interpolation requested")

    ## MULTI-GAUSSIAN ### covmodel required
    elif method.lower() in ["kriging", "grf", "grf_ineq"]:
        covmodel=copy.deepcopy(litho.get_surface_covmodel())
        if method.lower() == "kriging":
            if kwargs["unco"] == False:
                s, var=gcm.krige(data, zp, xu, covmodel, method=kwargs["krig_type"])
                #s=gci.estimate2D(covmodel, [nx, ny], [sx, sy], [ox, oy], x=data, v=zp,
                #                   method="ordinary_kriging", nneighborMax=10, searchRadiusRelative=1.0)["image"].val[0]
                s=s.reshape(ny, nx)
            else:
                raise ValueError ("Error: No data point found or unconditional kriging requested")

        elif method.lower() == "grf":
            if kwargs["unco"] == False: #conditional
                #transform data into normal distr
                if litho.N_transfo:

                    if hasattr(litho, "distribution"):
                        di = litho.distribution
                    else:
                        di=store_distri(zp, t=kwargs["tau"])
                    norm_zp=NScore_trsf(zp, di)

                    # need to recompute variogram TO DO

                    if kwargs["grf_method"] == "fft":
                        np.random.seed(int(seed))  # set seed for fft
                        sim=geone.grf.grf2D(covmodel, [nx, ny], [sx, sy], [ox, oy], x=data, v=norm_zp, nreal=1, mean=0, var=1, printInfo=False)
                        s=NScore_Btrsf(sim[0].flatten(), di)# back transform
                        s=s.reshape(ny, nx)
                    elif kwargs["grf_method"] == "sgs":
                        sim=gci.simulate2D(covmodel, [nx, ny], [sx, sy], [ox, oy], x=data, v=norm_zp, nreal=1, mean=0, var=1, verbose=verbose, nthreads=ncpu, seed=seed, mask=mask2D)
                        s=NScore_Btrsf(sim["image"].val[0,0].flatten(), di)# back transform
                        s=s.reshape(ny, nx)

                else: # no normal score
                    if "mean" not in kwargs.keys():
                        mean = np.mean(zp)
                    else:
                        mean = kwargs["mean"]

                    if kwargs["grf_method"] == "fft":
                        np.random.seed(int(seed))  # set seed for fft
                        sim=geone.grf.grf2D(covmodel, [nx, ny], [sx, sy], [ox, oy], x=data, v=zp, nreal=1, mean=mean, printInfo=False)
                        s=sim[0]
                    elif kwargs["grf_method"] == "sgs":
                        sim=gci.simulate2D(covmodel, [nx, ny], [sx, sy], [ox, oy], x=data, v=zp, nreal=1, mean=mean, verbose=verbose, nthreads=ncpu, seed=seed, mask=mask2D)
                        s=sim["image"].val[0,0]

            else:  # unconditional
                if kwargs["grf_method"] == "fft":
                    np.random.seed(int(seed))  # set seed for fft
                    sim=geone.grf.grf2D(covmodel, [nx, ny], [sx, sy], [ox, oy], nreal=1, mean=kwargs["mean"], printInfo=False)
                    s=sim[0]
                elif kwargs["grf_method"] == "sgs":
                        sim=gci.simulate2D(covmodel, [nx, ny], [sx, sy], [ox, oy], nreal=1, mean=kwargs["mean"], verbose=verbose, nthreads=ncpu, seed=seed, mask=mask2D)
                        s=sim["image"].val[0,0]

        elif method.lower() == "grf_ineq":
            # Normal transform
            if litho.N_transfo:

                if hasattr(litho, "distribution"):
                    di = litho.distribution
                else:
                    di=store_distri(v_eq, t=kwargs["tau"])

                v_eq=NScore_trsf(v_eq, di)
                vIneq_min=NScore_trsf(vIneq_min, di)
                vIneq_max=NScore_trsf(vIneq_max, di)
                var=1
                mean=0

                # need to recompute variogram TO DO

            else:
                var=covmodel.sill()
                # define mean
                if "mean" not in kwargs:
                    if len(v_eq) == 0: # if only inequality data what to do ??
                        mean=np.mean(np.concatenate((vIneq_max, vIneq_min)))
                    else:
                        mean=(np.mean(zp))
                else:
                    mean = kwargs["mean"]

            #return x_eq, v_eq, xIneq_min, vIneq_min,xIneq_max,vIneq_max, mask2D, mean, (nx, ny), (sx, sy), (ox, oy)
            sim=gci.simulate2D(covmodel, (nx, ny), (sx, sy), (ox, oy), method=kwargs["krig_type"], mean=mean,
                                x=x_eq, v=v_eq,
                                xIneqMin=xIneq_min, vIneqMin=vIneq_min,
                                xIneqMax=xIneq_max, vIneqMax=vIneq_max,
                                searchRadiusRelative=1, verbose=verbose,
                                nGibbsSamplerPathMin=kwargs["nit"],nGibbsSamplerPathMax=2*kwargs["nit"],
                                 seed=seed, nneighborMax=kwargs["nmax"], nthreads=ncpu, mask=mask2D)["image"].val[0, 0]

            """
            eq_d=np.concatenate([x_eq, v_eq.reshape(-1, 1), np.nan*np.ones([v_eq.shape[0], 2])], axis=1)
            all_data=np.concatenate([eq_d, np.array(litho.ineq)])
            sim=run_sim_2d(all_data, xg, yg, covmodel, nsim=1, nit=kwargs["nit"], var=var, mean=mean, nmax=kwargs["nmax"], krig_type=kwargs["krig_type"], grf_method=kwargs["grf_method"], ncpu=ncpu, seed=seed)[0]
            """

            if litho.N_transfo:
                s=NScore_Btrsf(sim.flatten(), di)
                s=s.reshape(ny, nx)
            else:
                s=sim

    elif method.lower() == "mps":

        assert isinstance(kwargs["TI"], geone.img.Img), "TI is not a geone image object"

        #load parameters
        TI=kwargs["TI"] #get TI

        #extract hard data
        eq_d=np.concatenate([x_eq, 0.5*np.ones([v_eq.shape[0], 1]), v_eq.reshape(-1, 1), np.nan*np.ones([v_eq.shape[0], 2])], axis=1)
        if len(litho.ineq) == 0:
            sup_d = None
            inf_d = None
            all_data = eq_d

            varname=['x', 'y', 'z', 'code'] # list of variable names
            hd=all_data.T
            pt=img.PointSet(npt=hd.shape[1], nv=4, val=hd, varname=varname)

        else:
            sup_d=np.concatenate([xIneq_max, 0.5*np.ones([vIneq_max.shape[0], 1]), np.nan*np.ones([vIneq_max.shape[0], 2]), vIneq_max.reshape(-1, 1)], axis=1)
            inf_d=np.concatenate([xIneq_min, 0.5*np.ones([vIneq_min.shape[0], 1]), np.nan*np.ones([vIneq_min.shape[0], 1]), vIneq_min.reshape(-1, 1), np.nan*np.ones([vIneq_min.shape[0], 1])], axis=1)
            all_data=np.concatenate([eq_d, sup_d, inf_d])

            varname=['x', 'y', 'z', 'code', 'code_min', 'code_max'] # list of variable names
            hd=all_data.T
            pt=img.PointSet(npt=hd.shape[1], nv=6, val=hd, varname=varname)

            #define mode (only rescaling min-max)
            if kwargs["TargetMin"] is None:
                kwargs["TargetMin"]=np.nanmin(all_data[:,3: ])
            if kwargs["TargetMax"] is None:
                kwargs["TargetMin"]=np.nanmax(all_data[:,3: ])

        #DS research
        snp=dsi.SearchNeighborhoodParameters(
            radiusMode=kwargs["radiusMode"], rx=kwargs["rx"], ry=kwargs["ry"], rz=kwargs["rz"],
            anisotropyRatioMode=kwargs["anisotropyRatioMode"], ax=kwargs["ax"], ay=kwargs["ay"], az=kwargs["az"],
            angle1=kwargs["angle1"], angle2=kwargs["angle2"], angle3=kwargs["angle3"])

        #DS input
        deesse_input=dsi.DeesseInput(
            nx=nx, ny=ny, nz=1,       # dimension of the simulation grid (number of cells)
            sx=sx, sy=sy, sz=1,     # cells units in the simulation grid (here are the default values)
            ox=ox, oy=oy, oz=0.5,     # origin of the simulation grid (here are the default values)
            nv=1, varname='code',       # number of variable(s), name of the variable(s)
            nTI=1, TI=TI,             # number of TI(s), TI (class dsi.Img)
            dataPointSet=pt,            # hard data (optional)
            searchNeighborhoodParameters=snp,
            homothetyUsage=kwargs["homo_usage"],
            homothetyXLocal=kwargs["xloc"],
            homothetyXRatio=kwargs["xr"],
            homothetyYLocal=kwargs["yloc"],
            homothetyYRatio=kwargs["yr"],
            homothetyZLocal=kwargs["zloc"],
            homothetyZRatio=kwargs["zr"],
            rotationUsage=kwargs["rot_usage"],            # tolerance or not
            rotationAzimuthLocal=kwargs["rotAziLoc"], #    rotation according to azimuth: global
            rotationAzimuth=kwargs["rotAzi"],
            rotationDipLocal=kwargs["rotDipLoc"],
            rotationDip=kwargs["rotDip"],
            rotationPlungeLocal=kwargs["rotPlungeLoc"],
            rotationPlunge=kwargs["rotPlunge"],
            distanceType='continuous', # distance type: proportion of mismatching nodes (categorical var., default)
            relativeDistanceFlag=kwargs["relativeDistanceFlag"],
            rescalingMode=kwargs["rescalingMode"],
            rescalingTargetMin=kwargs["TargetMin"], # min of the target interval
            rescalingTargetMax=kwargs["TargetMax"], #max of the target interval
            rescalingTargetMean=kwargs["TargetMean"],
            rescalingTargetLength=kwargs["TargetLength"],
            nneighboringNode=kwargs["neig"],        # max. number of neighbors (for the patterns)
            distanceThreshold=kwargs["thresh"],     # acceptation threshold (for distance between patterns)
            maxScanFraction=kwargs["maxscan"],       # max. scanned fraction of the TI (for simulation of each cell)
            seed=np.random.randint(1e6),
            npostProcessingPathMax=1,   # number of post-processing path(s)
            nrealization=1)          # number of realization(s))

        # Run deesse
        deesse_output=dsi.deesseRun(deesse_input, nthreads=ncpu, verbose=2)
        sim=deesse_output["sim"][0]
        s=sim.val[0,0]

    else:
        raise ValueError ("choose proper interpolation method")

    return s

def split_logs(bh):

    """
    Take a raw borehole with hierarchical units mixed
    and sort them by group and hierarchy.

    #input#
    bh, a borehole object

    #output#
    list of new boreholes
    """

    l_bhs=[]
    l_logs=[] #list of logs
    bhID=bh.ID
    bhx, bhy, bhz, depth=bh.x, bh.y, bh.z, bh.depth
    log_s=bh.log_strati


    l_logs=[]

    #logs stratis
    h_max=max([i[0].get_h_level() for i in log_s if i[0] is not None])
    for i in range(h_max):
        l_logs.append([])

    def bidule(s, h_lev):

        """
        Recursive operation to identifiy log strati at lower order level from higher levels.
        If in a borehole with a certain subsubunit (e.g. B11), this information must be also transfer to unit B1 and B
        This function does that (I guess...)
        """
        
        if h_lev > 1:
            if s in [i[0] for i in l_logs[h_lev-1]]:  # sub_unit already present
                if s == l_logs[h_lev-1][-1][0]:  # last unit added is the same --> adapt bot
                    l_logs[h_lev-1][-1][-1]=bot  # change bot
                elif s.mummy_unit.SubPile.nature == "3d_categorical":
                    l_logs[h_lev-1].append([s, top, bot])  # unit already present but
            elif s not in [i[0] for i in l_logs[h_lev-1]]:
                l_logs[h_lev-1].append([s, top, bot])
            h_lev -= 1
            bidule(s.mummy_unit, h_lev)
        elif h_lev == 1:
            if s not in [i[0] for i in l_logs[h_lev-1]]:  # add unit only if it's not in log
                l_logs[h_lev-1].append([s, top])

    for i in range(len(log_s)):
        s=log_s[i] #get unit and contact
        if i == len(log_s)-1: #if last unit
            bot = bhz - depth
        else:
            s_aft = log_s[i+1]
            bot = s_aft[1]
        unit=s[0] #get unit
        if unit is not None:
            h_lev=unit.get_h_level()
        else:
            h_lev=1
        top =s[1]

        if h_lev == 1:
            l_logs[h_lev-1].append([unit, top])
        elif h_lev > 1:
            bidule(unit, h_lev)

    #1st order log
    bh=borehole(bhID, bhID, bhx, bhy, bhz, depth, log_strati=l_logs[0], log_facies=bh.log_facies) # first borehole
    l_bhs.append(bh)

    #2 and more order logs
    for log in l_logs[1: ]: #loop over different logs hierarchy levels
        i_0=0
        if len(log) > 1: #more than 1 unit is present
            for i in range(1, len(log)):
                unit=log[i-1][0]
                unit_after=log[i][0]
                if unit.mummy_unit != unit_after.mummy_unit:  # check that two successive units does not belong to the same hierarchic group
                    new_log=log[i_0: i]
                    depth=new_log[0][1] - new_log[-1][2]
                    log_strati=[s[: -1] for s in new_log]
                    bh=borehole(str(bhID)+"_"+unit.name, str(bhID)+"_"+unit.name, bhx, bhy, log_strati[0][1], depth, log_strati=log_strati)
                    l_bhs.append(bh)
                    i_0=i
                if i == len(log)-1:
                    new_log=log[i_0: ]
                    depth=new_log[0][1] - new_log[-1][2]
                    log_strati=[s[: -1] for s in new_log]
                    bh=borehole(str(bhID)+"_"+unit_after.name, str(bhID)+"_"+unit_after.name, bhx, bhy, log_strati[0][1],
                                  depth, log_strati=log_strati)
                    l_bhs.append(bh)

        else:
            unit=log[0][0]
            depth=log[0][1] - log[-1][2]
            log_strati=[s[: -1] for s in log]
            bh=borehole(str(bhID)+"_"+unit.name, str(bhID)+"_"+unit.name, bhx, bhy, log_strati[0][1],
                          depth, log_strati=log_strati)
            l_bhs.append(bh)

    return l_bhs

def running_mean_2D(x, N):

    """
    smooth a 2d surface
    x: 2D nd.array
    N: window semi-size
    """

    s=x.copy()
    s=uniform_filter(s, size=N, mode="reflect")

    return s

####### CLASSES ########
class Arch_table():

    """
    Major class of ArchPy. Arch_table is the central object
    that can be assimilated as a "project".
    Practically every operations are done using an Arch_table

    Attributes:
    name            : string, name of the project
    working directory: folder where to create and save files
                        if it doesn't exist, it will be created
    seed            : int, numerical seed for stochastic applications
    verbose         : 0 or 1, if 0, ArchPy will print nothing
    fill_flag       : bool, flag to perform filling of the top unit after
                      units have been simulated
    ncpu            : int, number of cpus to use if
                        mulithread operations are available.
                        -1 for all cpus - 1.
    """


    def __init__(self, name, working_directory="ArchPy_workspace", seed=np.random.randint(1e6), write_results=False, fill_flag=False, verbose=1, ncpu=-1):

        assert name is not None, "A name must be provided"
        # put assert seed

        self.name=name
        self.ws=working_directory  #working directy where files will be created
        self.list_all_units=[]
        self.list_all_facies=[]
        self.list_bhs=[] # list of boreholes for hd
        self.list_fake_bhs=[] # list of "fake" boreholes
        self.list_map_bhs=[]
        self.sto_hd = []  # stochastic hard data
        self.list_props=[]
        self.seed=int(1e6*seed)
        self.verbose=verbose  # 0: print (quasi)-nothing, 1: print everything
        self.ncpu=ncpu
        self.xg=None
        self.yg=None
        self.zg=None
        self.xgc =None
        self.ygc=None
        self.zgc=None
        self.sx=None
        self.sy=None
        self.sz=None
        self.ox=None
        self.oy=None
        self.oz=None
        self.nx=None
        self.ny=None
        self.nz=None
        self.Pile_master=None
        self.geol_map=None
        self.write_results=write_results
        self.bhs_processed=0  # flag to know if boreholes have been processed
        self.surfaces_computed=0
        self.facies_computed=0
        self.prop_computed=0
        self.nreal_units=0
        self.nreal_fa=0
        self.nreal_prop=0
        self.fill_flag = fill_flag
        self.Geol=Geol()

    #get functions
    def get_pile_master(self):
        if self.Pile_master is None:
            raise ValueError ("No Pile master defined for Arch Table {}".format(self.name))
        return self.Pile_master

    def get_xg(self):
        if self.xg is None:
            assert 0, ('Error: Grid was not added')
        return self.xg

    def get_yg(self):
        if self.yg is None:
            assert 0, ('Error: Grid was not added')
        return self.yg

    def get_zg(self):
        if self.zg is None:
            assert 0, ('Error: Grid was not added')
        return self.zg

    def get_xgc(self):
        if self.xgc is None:
            assert 0, ('Error: Grid was not added')
        return self.xgc

    def get_ygc(self):
        if self.ygc is None:
            assert 0, ('Error: Grid was not added')
        return self.ygc

    def get_zgc(self):
        if self.zg is None:
            assert 0, ('Error: Grid was not added')
        return self.zgc
    def get_nx(self):
        if self.nx is None:
            assert 0, ('Error: Grid was not added')
        return self.nx

    def get_ny(self):
        if self.ny is None:
            assert 0, ('Error: Grid was not added')
        return self.ny

    def get_nz(self):
        if self.nz is None:
            assert 0, ('Error: Grid was not added')
        return self.nz
    def get_sx(self):
        if self.sx is None:
            assert 0, ('Error: Grid was not added')
        return self.sx

    def get_sy(self):
        if self.sy is None:
            assert 0, ('Error: Grid was not added')
        return self.sy

    def get_sz(self):
        if self.sz is None:
            assert 0, ('Error: Grid was not added')
        return self.sz

    def get_ox(self):
        if self.ox is None:
            assert 0, ('Error: Grid was not added')
        return self.ox

    def get_oy(self):
        if self.oy is None:
            assert 0, ('Error: Grid was not added')
        return self.oy

    def get_oz(self):
        if self.oz is None:
            assert 0, ('Error: Grid was not added')
        return self.oz

    # def get_facies(self):

    #     # if self.Geol.facies_domains is None:
    #     #     assert 0, ('Error: facies domains not computed')

    #     facies =self.Geol.facies_domains.copy()
    #     return facies

    def get_facies(self, iu=0, ifa=0, all_data=True):

        """
        Return a numpy array of 1 or all facies realization(s).
        iu      : int, unit index
        ifa     : int, facies index
        all_data: bool, return all the units simulations
        """

        if self.write_results:
            if "fd" not in [i.split(".")[-1] for i in os.listdir(self.ws)]:
                raise ValueError("Facies have not been computed yet")

        if all_data:
            # get all real
            if self.write_results:
                nreal_fa=self.nreal_fa
                nreal_u=self.nreal_units
                nx=self.get_nx()
                ny=self.get_ny()
                nz=self.get_nz()
                fd=np.zeros([nreal_u, nreal_fa, nz, ny, nx], dtype=np.int8)
                for iu in range(nreal_u):
                    for ifa in range(nreal_fa):
                        fname=self.name+"_{}_{}.fd".format(iu, ifa)
                        fpath=os.path.join(self.ws, fname)
                        with open(fpath, "rb") as f:
                            fd[iu, ifa]=pickle.load(f)
            else:
                fd=self.Geol.facies_domains.copy()
        else:
            if self.write_results:
                fname=self.name+"_{}_{}.fd".format(iu, ifa)
                fpath=os.path.join(self.ws, fname)
                with open(fpath, "rb") as f:
                    fd=pickle.load(f)
            else:
                fd=self.Geol.facies_domains[iu, ifa].copy()

        return fd



    def get_surfaces_unit(self, unit, typ="top"):

        """
        Return a 3D array of computed surfaces for a specific unit

        #inputs#
        unit: a Unit object contain inside the master pile
               or another subunit
        typ: string, (top, bot or original),
              specify which type of surface to return, top,
              bot or original (surfaces before applying erosion
              and stratigrapic rules)

        #ouputs#
        All computed surfaces in a nd.array of
        size (nreal_units, ny, nx)
        """

        assert unit in self.get_all_units(), "Unit must be included in a pile related to the master pile"
        assert self.Geol.surfaces_by_piles is not None, "Surfaces not computed"
        assert typ in ("top", "bot", "original"), "Surface type {} doesnt exist".format(typ)
        hl=unit.get_h_level()
        if hl == 1:
            P=self.get_pile_master()
        elif hl > 1:
            P=unit.mummy_unit.SubPile

        if P.nature == "surfaces":
            if typ =="top":
                s=self.Geol.surfaces_by_piles[P.name][:, unit.order-1].copy()
            elif typ =="bot":
                s=self.Geol.surfaces_bot_by_piles[P.name][:, unit.order-1].copy()
            elif typ == "original":
                s=self.Geol.org_surfaces_by_piles[P.name][:, unit.order-1].copy()
            return s
        else:
            return None

    def get_surface(self, h_level="all"):

        """
        Return a 4D array of multiple surfaces according to
        the hierarchical level desired, by default ArchPy try
        to return the highest hierarchical unit's surface

        #inputs#
        h_level: int or string, maximum level of hierarchy
                  desired to return, "all" indicates to return
                  the highest hierarchical level possible for each unit

        #outputs#
        - a 4D (nlayer, nsim, ny, nx) arrays with the surfaces
        - a list (nlayer) of units name corresponding to the
          surfaces to distinguish wich surface correspond to which unit
        """

        if h_level == "all":
            l=[]
            def fun(pile):
                for u in pile.list_units:
                    if u.f_method != "SubPile":
                        l.append(u)
                    else:
                        fun(u.SubPile)

        elif isinstance(h_level, int) and h_level > 0:
            l=[]
            def fun(pile):
                for u in pile.list_units:
                    if u.f_method == "SubPile":
                        if u.get_h_level() < h_level:
                            fun(u.SubPile)
                        elif u.get_h_level() == h_level:
                            l.append(u)
                        else:
                            pass
                    else:
                        l.append(u)

        fun(self.get_pile_master())
        nlay=len(l)
        unit_names=[i.name for i in l]
        for i in range(nlay):
            u=l[i]
            s=self.get_surfaces_unit(u)
            if i == 0:
                nreal, ny, nx=s.shape
                surfs=np.zeros([nlay, nreal, ny, nx], dtype=np.float32)
                surfs[0]=s
            else:
                surfs[i]=s
        return surfs, unit_names

    def get_unit(self, name="A", ID=1, type="name", all_strats=True, vb=1):

        """
        Return the unit in the Pile with the associated name or ID.

        #inputs#
        name     : string, name of the strati to retrieve
        ID       : int, ID of the strati to retrieve
        type     : str, (name or ID), retrieving method
        all_strats: bool, flag to indicate to also search in sub-units,
                     if false research will be restricted to units
                     directly in the Pile master


        #outputs#
        A unit object
        """

        assert isinstance(name, str), "Name must be a string"

        if all_strats:
            l=self.get_all_units()
        else:
            l=self.get_pile_master().list_units()
        for s in l:
            if type == "name":
                if s.name == name:
                    return s
            elif type == "ID":
                if s.ID == ID:
                    return s

        if type=="name":
            var=name
        elif type=="ID":
            var=ID
        if vb:
            print ("No unit with that name/ID {}".format(var))
        return None

    def getbhindex(self, ID):

        """
        Return the index corresponding to a certain borehole ID"""

        interest=None
        for i in range(len(self.list_bhs)):
            if self.list_bhs[i].ID == ID:
                interest=i
        if interest is None:
            assert 1, ('the propriety '+ID+' was not found')
        else:
            return interest

    def getbh(self, ID):

        """
        Return the borehole object given its ID
        """
        index=self.getbhindex(ID)
        return self.list_bhs[index]

    def get_facies_obj(self, name="A", ID=1, type="name", vb=1):

        """
        Return the facies in the Pile with the associated name or ID.

        #inputs#
        name      : string, name of the strati to retrieve
        ID        : int, ID of the strati to retrieve
        type      : str, (name or ID), retrieving method
        all_strats: bool, flag to indicate to also search in sub-units,
                      if false research will be restricted to units
                      directly in the Pile master

        #outputs#
        A facies object
        """

        assert isinstance(name, str), "Name must be a string"

        l=self.get_all_facies()
        for s in l:
            if type == "name":
                if s.name == name:
                    return s

            elif type == "ID":
                if s.ID == ID:
                    return s

        if type=="name":
            var=name
        elif type=="ID":
            var=ID
        if vb:
            print ("No facies with that name/ID {}".format(var))
        return None

    def pointToIndex(self, x, y, z):

        cell_x=np.array((x-self.ox)/self.sx).astype(int)
        cell_y=np.array((y-self.oy)/self.sy).astype(int)
        cell_z=np.array((z-self.oz)/self.sz).astype(int)

        return cell_x, cell_y, cell_z


    def get_all_units(self, recompute=True):

        """
        return a list of all units, even sub-units

        recompute: bool, if False, the list all units attribute
                    will be simply retrieve. Even if changes have
                    been made.

        #outputs#
        List of all units
        """

        if len(self.list_all_units) == 0:
            recompute=True
        if self.list_all_units is None or recompute: # recompute if wanted of if there is no list_all_stratis

            def list_all_units(all_stratis, subpile_stratis):

                all_stratis=all_stratis + subpile_stratis
                for s in subpile_stratis:
                    if s.f_method == "SubPile":
                        all_stratis=list_all_units(all_stratis, s.SubPile.list_units)
                return all_stratis

            lau=list_all_units([], self.get_pile_master().list_units)
            #check that stratis have different names
            l=[]
            for s in lau:
                if s.name not in l:
                    l.append(s.name)
                else:
                    raise ValueError("Some units have the same name (Unit {}".format(s.name))
            self.list_all_units=lau

        return self.list_all_units

    def get_all_facies(self, recompute=True):

        """
        return a list of all facies

        recompute: bool, if False, the list all facies attribute
                    will be simply retrieve. Even if changes have
                    been made on the project.

        #outputs#
        List of all facies
        """

        if len(self.list_all_facies) == 0:
            recompute=True

        if recompute:
            l=[]
            def l_fa(pile):
                for s in pile.list_units:
                    for fa in s.list_facies:
                        if fa not in l:
                            l.append(fa)
                    if s.f_method == "SubPile":
                        l_fa(s.SubPile)

            l_fa(self.get_pile_master())
            self.list_all_facies=l
            return l
        else:
            if self.list_all_facies is not None:
                return self.list_all_facies
            else:

                self.get_all_facies(recompute=True)
                return self.list_all_facies

    def get_piles(self):

        """Return a list of all the subpiles"""

        l=[]

        def func(pile):
            l.append(pile)
            for u in pile.list_units:
                if u.f_method == "SubPile":
                    func(u.SubPile)
        func(self.Pile_master)
        return l

    def getpropindex(self, name):
        """ Return the index corresponding to a certain prop name"""
        interest=None
        for i in range(len(self.list_props)):
            if self.list_props[i].name == name:
                interest=i
        if interest is None:
            raise ValueError('the propriety '+name+' was not found')
        return interest

    def getprop(self, name, iu=None, ifa=None, ip=None, all_data=True):

        """
        Return a numpy array of 1 or all facies realization(s).
        iu      : int, unit index
        ifa     : int, facies index
        ip      : int, property index
        all_data: bool, return all the units simulations
        """

        if self.write_results:
            if "pro" not in [i.split(".")[-1] for i in os.listdir(self.ws)]:
                raise ValueError("Properties have not been computed yet")

        l=[i.name for i in self.list_props]
        if name not in l:
            raise ValueError('The propriety "'+name+'"" was not found \n available Property names are: {}'.format(l))

        nreal_prop=self.nreal_prop
        nreal_fa=self.nreal_fa
        nreal_u=self.nreal_units
        nx=self.get_nx()
        ny=self.get_ny()
        nz=self.get_nz()

        if all_data:
            prop=np.zeros([nreal_u, nreal_fa, nreal_prop, nz, ny, nx], dtype=np.float32)

            if self.write_results:
                # get all real
                for iu in range(nreal_u):
                    for ifa in range(nreal_fa):
                        for ip in range(nreal_prop):
                            fname=self.name+"{}_{}_{}_{}.pro".format(name, iu, ifa, ip)
                            fpath=os.path.join(self.ws, fname)
                            with open(fpath, "rb") as f:
                                prop[iu, ifa, ip]=pickle.load(f)
            else:
                prop=self.Geol.prop_values[name]

        else:
            if self.write_results:
                fname=self.name+"{}_{}_{}_{}.pro".format(name, iu, ifa, ip)
                fpath=os.path.join(self.ws, fname)
                with open(fpath, "rb") as f:
                    prop=pickle.load(f)
            else:
                prop=self.Geol.prop_values[name][iu, ifa, ip]

        return prop

    def get_bounds(self):

        """Return bounds of the simulation domain (xmin, xmax, ymin, ymax, zmin, zmax)"""

        bounds=[self.get_ox(), self.get_xg()[-1], self.get_oy(), self.get_yg()[-1], self.get_oz(), self.get_zg()[-1]]
        return bounds

    def check_units_ID(self):

        l=[]
        for i in self.get_all_units(recompute=True):
            if i.ID not in l:
                l.append(i.ID)
            else:
                print("Sorry, unit {} has the same ID {} than another unit. ID must be differents for each units".format(i.name, i.ID))
                return None
        return 1

    def check_piles_name(self):

        """
        check names of subpile --> must be different
        """

        l_names=[]

        l=self.get_piles()
        for p in l:
            if p.name not in l_names:
                l_names.append(p.name)
            else:
                print("Pile name '{}' is attributed to more than one pile, verboten !".format(p.name))
                return 0

        return 1

    def set_Pile_master(self, Pile_master):

        """
        Define a pile object as the main pile of the project
        """

        assert isinstance(Pile_master, Pile), "Pile master is not an ArchPy Pile object"
        self.Pile_master=Pile_master
        if self.check_piles_name():
            if self.verbose:
                print("Pile sets as Pile master")
        else:
            self.Pile_master=None
            if self.verbose:
                print("Pile not sets")

    def indextocell(self, x, y, z): #duplicate with pointoindex --> remove one
        '''cell number to cell indexing'''
        cell_x=(x/self.get_sx() - min(self.get_xgc())).astype(int)
        cell_y=(y/self.get_sy() - min(self.get_ygc())).astype(int)
        cell_z=(z/self.get_sz() - min(self.get_zgc())).astype(int)

        return cell_x, cell_y, cell_z

    def celltoindex(self,cell_x,cell_y,cell_z):
        '''cell index to cell position'''
        x=self.get_sx()* cell_x + min(self.get_xgc())
        y=self.get_sy()* cell_y + min(self.get_ygc())
        z=self.get_sz()* cell_z + min(self.get_zgc())
        return x,y,z

    def reprocess(self):
        self.bhs_processed=0
        self.erase_hd()
        self.process_bhs()
        self.seed= int(self.seed + 1e6)

    def resample2grid(self, raster_path, band=None, rspl_method="nearest"):
        """
        resample a raster to the size of the simulation grid.

        ### inputs ###
        raster_path : str, path to the raster file
        band        : int, raster band to use, if None 0 is used
        rspl_method : str, resampling method to use.
                     availables are : nearest, linear, cubic

        ### outputs ###
        2D array of size (self.ny, self.nx)
        """

        #import rasterio
        import rasterio

        #open raster and extract cell centers
        DEM=rasterio.open(raster_path)
        x0, y0, x1, y1=DEM.bounds
        rxlen=DEM.read().shape[2]
        rylen=DEM.read().shape[1]
        x=np.linspace(x0, x1, rxlen)
        y=np.linspace(y1, y0, rylen)
        rxc, ryc=np.meshgrid(x, y)

        #take grid cell centers
        xc=self.xcellcenters
        yc=self.ycellcenters


        if band is None:
            ib = 0
        else:
            ib = band
        return resample_to_grid(xc, yc, rxc, ryc, DEM.read()[ib], method=rspl_method)  # resampling

    def add_grid(self, dimensions, spacing, origin, top=None, bot=None, rspl_method="nearest", polygon=None, mask=None):

        """
        Method to add/change simulation grid, regular grid.

        # parameters #
        dimensions: sequence of size 3,
                     number of cells in x, y and direction (nx, ny, nz)
        spacing: sequence of size 3,
                 spacing of the cells in x, y and direction (sx, sy, sz)
        origin: sequence of size 3,
                origin of the simulation grid of the cells
                in x, y and direction (ox, oy, oz)
        top, bot: 2D ndarray of dimensions (ny, nx)
                  float or raster file, top and bottom of the simulation domain
        rspl_method: string
                    scipy resampling method (nearest, linear
                    and cubic --> nearest is generally sufficient)
        polygon: 2D ndarray of dimensions (ny, nx)
                boolean array to indicate where the simulation is active (1)
                or inactive (0). Polygon can also be a Shapely (Multi) - polygon.
        mask: 3D ndarray of dimensions (nz, ny, nx)
                3D boolean array to indicate where the simulation
                is active (1) or inactive (0).
                If given, top, bot and polygon are ignored.
        """

        if self.verbose:
            print("## Adding Grid ##")
        ## cell centers ##
        sx=spacing[0]
        sy=spacing[1]
        sz=spacing[2]

        nx=dimensions[0]
        ny=dimensions[1]
        nz=dimensions[2]

        ox=origin[0]
        oy=origin[1]
        oz=origin[2]

        xg=np.arange(ox, ox+nx*sx+sx, sx, dtype=np.float32)
        yg=np.arange(oy, oy+ny*sy+sy, sy, dtype=np.float32)
        zg=np.arange(oz, oz+nz*sz+sz, sz, dtype=np.float32)

        xgc=xg[: -1]+sx/2
        ygc=yg[: -1]+sy/2
        zgc=zg[: -1]+sz/2

        self.sx=sx
        self.sy=sy
        self.sz=sz

        self.nx=nx
        self.ny=ny
        self.nz=nz

        self.ox=ox
        self.oy=oy
        self.oz=oz

        self.xg=xg
        self.yg=yg
        self.zg=zg

        self.xgc=xgc  # xg_cell_centers
        self.ygc=ygc  # yg_cell_centers
        self.zgc=zgc  # zg_cell_centers

        self.xcellcenters, self.ycellcenters=np.meshgrid(xgc, ygc) # cell centers coordinates

        z_tree=KDTree(zg.reshape(-1, 1))
        self.z_tree=z_tree
        self.zc_tree=KDTree(zgc.reshape(-1, 1))
        self.xc_tree=KDTree(xgc.reshape(-1, 1))
        self.yc_tree=KDTree(ygc.reshape(-1, 1))


        ## resample top and bot if needed
        if isinstance(top, str) or isinstance(bot, str):

            #import rasterio
            import rasterio

            #open raster and extract cell centers
            DEM=rasterio.open(top)
            x0, y0, x1, y1=DEM.bounds
            rxlen=DEM.read().shape[2]
            rylen=DEM.read().shape[1]
            x=np.linspace(x0, x1, rxlen)
            y=np.linspace(y1, y0, rylen)
            rxc, ryc=np.meshgrid(x, y)

            #take grid cell centers
            xc=self.xcellcenters
            yc=self.ycellcenters

            if isinstance(top, str):
                if self.verbose:
                    print("Top is a raster - resampling activated")
                top=resample_to_grid(xc, yc, rxc, ryc, DEM.read()[0], method=rspl_method)  # resampling

            if isinstance(bot, str):
                if self.verbose:
                    print("Bot is a raster - resampling activated")
                rast=rasterio.open(bot)
                x0, y0, x1, y1=rast.bounds
                rxlen=rast.read().shape[2]
                rylen=rast.read().shape[1]
                x=np.linspace(x0, x1, rxlen)
                y=np.linspace(y1, y0, rylen)
                rxc, ryc=np.meshgrid(x, y)
                bot=resample_to_grid(xc, yc, rxc, ryc, rast.read()[0], method=rspl_method)  # resampling

        if top is not None:
            assert top.shape == (ny, nx), "Top shape is not adequat respectively to coordinate vectors. \n Must be have a size of -1 respectively to coordinate vectors xg and yg (which are the vectors of edge cells)"
        if bot is not None:
            assert bot.shape == (ny, nx), "Bot shape is not adequat respectively to coordinate vectors. \n Must be have a size of -1 respectively to coordinate vectors xg and yg (which are the vectors of edge cells)"

        #define top/bot
        if (top is None) and (mask is None):
            top=np.ones([ny, nx], dtype=np.float32)*np.max(zg)

        elif (top is None) and (mask is not None):  # if mask is provided but not top
            top =np.zeros([ny, nx], dtype=np.float32)*np.nan
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz-1, -1, -1):
                        if mask[iz, iy, ix]:
                            top[iy, ix]=zgc[iz]
                            break

        # cut top
        top[top>zg[-1]]=zg[-1]
        top[top<zg[0]]=zg[0]
        self.top=top.astype(np.float32)

        if (bot is None) and (mask is None):
            bot=np.ones([ny, nx], dtype=np.float32)*np.min(zg)

        elif (bot is None) and (mask is not None):  # if mask is provided but not top
            bot =np.zeros([ny, nx], dtype=np.float32)*np.nan
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        if mask[iz, iy, ix]:
                            bot[iy, ix]=zgc[iz]
                            break
        # cut bot
        bot[bot>zg[-1]]=zg[-1]
        bot[bot<zg[0]]=zg[0]
        self.bot=bot.astype(np.float32)

        # create mask from top and bot if none
        if mask is None:
            mask=np.zeros([nz, ny, nx], dtype=bool)
            iu=z_tree.query(top.reshape(-1, 1), return_distance=False).reshape(ny, nx)
            il=z_tree.query(bot.reshape(-1, 1), return_distance=False).reshape(ny, nx)

            for ix in range(len(xgc)):
                for iy in range(len(ygc)):
                    mask[il[iy, ix]: iu[iy, ix], iy, ix]=1

        # list of coordinates 2D and 3D
        X, Y=np.meshgrid(xgc, ygc)
        self.xu2D=np.array([X.flatten(), Y.flatten()], dtype=np.float32).T

        #X, Y, Z=np.meshgrid(xgc, ygc, zgc)
        #self.xu3D=np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

        #apply polygon

        # if polygon is a shapefile
        if isinstance(polygon, str):
            if polygon.split(".")[-1] == "shp":
                import geopandas as gp 
                poly = gp.read_file(polygon)
                polygon = Polygon(poly.geometry[0])

        #if polygon is shapely Polygon
        if isinstance(polygon, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
            if self.verbose:
                print("Polygon is a shapely instance - discretization activated")

            polygon_array=np.zeros([ny*nx], dtype=bool) #2D array simulation domain
            cell_l=[]
            for i,cell in enumerate(self.xu2D):

                xy=((cell[0]-sx/2, cell[1]-sy/2),(cell[0]-sx/2, cell[1]+sy/2),
                      (cell[0]+sx/2, cell[1]+sy/2),(cell[0]+sx/2, cell[1]-sy/2))
                p=shapely.geometry.Polygon(xy)
                p.name=i
                cell_l.append(p)

            l=[]#list of intersected cells
            for cell in cell_l:
                if cell.intersects(polygon):
                    l.append(cell.name)
            polygon_array[np.array(l)]=1
            polygon=polygon_array.reshape(ny, nx)

        if polygon is not None:
            polygon=polygon.astype(bool) #ensure polygon is a boolean array
            for ix in range(nx):
                for iy in range(ny):
                    if ~polygon[iy, ix]:
                        mask[:, iy, ix]=0

        self.mask=mask
        self.mask2d = mask.any(0)

        if self.verbose:
            print("## Grid added and is now simulation grid ##")


    def hierarchy_relations(self, vb=1):

        """Method that sets the hierarchical relations between units"""

        def h_relations(pile):
            """
            # calculate mummy unit for each sub unit (to which unit a sub-unit is related)
            """
            for s in pile.list_units: # loop over units of the master pile
                if s.f_method == "SubPile":
                    for ssub in s.SubPile.list_units: # loop over units of the sub pile if filling method is SubPile
                        ssub.mummy_unit=s
                    h_relations(s.SubPile)

        h_relations(self.get_pile_master())

        #set bb units
        for unit in self.get_all_units():
            unit.get_baby_units(recompute=True, vb=0)

        if vb:
            print("hierarchical relations set")


    def check_bh_inside(self, bh):

        """
        Check if a borehole is inside the simulation domain
        and cut boreholes if needed
        """

        if bh.depth <= 0:
            if self.verbose:
                print("borehole depth is not positive")
            return 0


        #fun to check botom
        def cut_botom(bh):
            z0_bh=bh.z
            zbot_bh=z0_bh - bh.depth

            bot_z = self.bot[self.coord2cell(bh.x, bh.y)]
            # if (zbot_bh < zg[0]):  # modification to cut below bot and not below simulation grid
            if (zbot_bh < bot_z):
                bh.depth=(z0_bh - bot_z) - sz/2
                if self.verbose:
                    print("Borehole {} goes below model limits, borehole {} depth cut".format(bh.ID, bh.ID))
                if bh.log_strati is not None: # update log strati
                    new_log=[]
                    for s in bh.log_strati:
                        if s[1] > bot_z:
                            new_log.append(s)
                    bh.log_strati=new_log
                if bh.log_facies is not None: # update log facies
                    new_log=[]
                    for s in bh.log_facies:
                        if s[1] > bot_z:
                            new_log.append(s)
                    bh.log_facies=new_log

        zg=self.get_zg() # z vector
        sz=self.get_sz()

        #check botom
        if bh.log_strati is not None:
            z0_bh=bh.log_strati[0][1] # borehole altitude top
        elif bh.log_facies is not None:
            z0_bh=bh.log_facies[0][1]
        else:
            if self.verbose:
                print("no log found in bh {}".format(bh.ID))
            return 0

        #check inside mask and adapt z borehole to DEM
        for iz in np.arange(z0_bh, z0_bh-bh.depth, -sz):
            if self.coord2cell(bh.x, bh.y, iz) is not None: # check if inside mask
                if self.verbose:
                    #print("Borehole {} inside of the simulation zone".format(bh.ID))
                    pass
                if iz == bh.z:
                    cut_botom(bh)
                    if bh.depth < 0:
                        return 0
                    else:
                        return 1
                elif iz < bh.z: # if the borehole is above DEM and must be cut
                    bh.z=iz  #update borehole altitude
                    # update log strati
                    if bh.log_strati:
                        if len(bh.log_strati) > 1:
                            new_log=[]
                            for i in range(len(bh.log_strati)-1):
                                s=bh.log_strati[i]
                                s2=bh.log_strati[i+1]
                                if s[1] > iz and s2[1] < iz: # unit cut by the dem
                                    new_log.append((s[0], iz))
                                elif s[1] <= iz:
                                    new_log.append(s)
                                else:
                                    pass
                            #last unit in log
                            if s2[1] <= iz:
                                new_log.append(s2)
                            elif s2[1] > iz:
                                new_log.append((s2[0], iz))
                            bh.log_strati=new_log
                        else:
                            pass
                    # update log facies
                    if bh.log_facies:
                        if len(bh.log_facies)>1:
                            new_log=[]
                            for i in range(len(bh.log_facies)-1):
                                s=bh.log_facies[i]
                                s2=bh.log_facies[i+1]
                                if s[1] > iz and s2[1] < iz: # unit cut by the dem
                                    new_log.append((s[0], iz))
                                elif s[1] <= iz:
                                    new_log.append(s)
                                else:
                                    pass
                            if s2[1] <= iz:
                                new_log.append(s2)
                            elif s2[1] > iz:
                                new_log.append((s2[0], iz))
                            bh.log_facies=new_log
                        else:
                            pass
                    cut_botom(bh)
                    if bh.depth < 0:
                        return 0
                    else:
                        return 1

        if self.verbose:
            print("Borehole {} outside of the simulation zone - not added -".format(bh.ID))
        return 0

    def coord2cell(self, x, y, z=None):

        """
        Method that returns the cell in which are the given coordinates
        """

        assert y == y, "coordinates contain NaN"
        assert x == x, "coordinates contain NaN"

        if z is not None:
            assert z == z, "coordinates contain NaN"

        xg=self.get_xg()
        yg=self.get_yg()
        zg=self.get_zg()
        sx=self.get_sx()
        sy=self.get_sy()
        sz=self.get_sz()
        nz=self.get_nz()

        # check point inside simulation block
        if (x <= xg[0]) or (x >= xg[-1]):
            if self.verbose:
                print("point outside of the grid in x")
            return None
        if (y <= yg[0]) or (y >= yg[-1]):
            if self.verbose:
                print("point outside of the grid in y")
            return None

        if z is not None:
            if (z <= zg[0]) or (z > zg[-1]):
                if self.verbose:
                    print("point outside of the grid in z")
                return None

        ix=((x-xg[0])//sx).astype(int)
        iy=((y-yg[0])//sy).astype(int)

        if z is not None:
            iz=((z-zg[0])//sz).astype(int)
            if iz > nz-1:
                iz=nz-1

            cell=(iz, iy, ix)
            if self.mask[iz, iy, ix]:
                return cell
            else:
                #print("Point outside of the simulation domain")
                return None
        else:
            cell = (iy, ix)
            return cell

    def add_prop(self, prop):

        """
        Add a property to the Arch_Table
        """

        try:
            for i in prop:
                if (isinstance(i, Prop)) and (i not in self.list_props):
                    self.list_props.append(i)
                    if self.verbose:
                        print("Property {} added".format(i.name))
                else:
                    if self.verbose:
                        print("object isn't a Property object or it is already in the list")
        except: # strati not in a list
            if (isinstance(prop, Prop)) and (prop not in self.list_props):
                self.list_props.append(prop)
                if self.verbose:
                    print("Property {} added".format(prop.name))
            else:
                if self.verbose:
                    print("object isn't a Property object")

    def add_bh(self, bhs):

        """
        Method to add borehole, list of boreholes if multiples

        bhs: breohole or list of borehole objects
        """

        if hasattr(bhs, "__iter__"):
            for i in bhs:
                if (isinstance(i, borehole)) and (i not in self.list_bhs):
                    if self.check_bh_inside(i):
                        self.list_bhs.append(i)
                        if self.verbose:
                            print("Borehole {} added".format(i.ID))
                else:
                    if self.verbose:
                        print("object isn't a borehole object or object is already in the list")
        else: # boreholes not in a list
            if (isinstance(bhs, borehole)) and (bhs not in self.list_bhs):
                if self.check_bh_inside(bhs):
                    self.list_bhs.append(bhs)
                    if self.verbose:
                        print("Borehole {} added".format(bhs.ID))
            else:
                if self.verbose:
                    print("object isn't a borehole object or object is already in the list")

    def make_fake_bh(self, positions_x, positions_y, units=None, facies=None, stratIndex=0, faciesIndex=0, extractUnits=True, extractFacies=True, vb=1):

        """
        Create fake boreholes from realization of the Arch_table.
        # inputs #
        positions_x  : seqence of numbers, indicate the x positions 
                       of the borehole to create bhs
        positions_y  : seqence of numbers, indicate the y positions 
                       of the borehole to create bhs
        units        : optional argument, 3D array, unit array to use 
                       to create bhs
        facies       : optional argument, 3D array, facies array to use 
                       to create bhs
        stratIndex   : int or sequence of int, unit index to sample
        faciesIndex  : int or sequence of int, facies index to sample 
        extractUnits : bool, flag to indicate to sample units or not
        extractFacies: bool, flag to indicate to sample facies or not
        # outputs #
        a list of borehole objects
        """ 

        # get 3D arrays to sample
        if units is None and extractUnits:
            units = self.get_units_domains_realizations()
            # units_data[np.isnan(units_data)]=-99  # default ID to indicate no data 
            units_data=units.copy()
        if facies is None and extractFacies:
            facies = self.get_facies()
            facies_data=facies.copy()
            # facies_data[np.isnan(facies_data)]=-99  # same for facies

        #change into array

        if type(positions_x) is not np.ndarray:
            positions_x = np.array([positions_x])
        if type(positions_y) is not np.ndarray:
            positions_y = np.array([positions_y])
        if type(stratIndex) is int:
            stratIndex = np.array([stratIndex])
        if type(faciesIndex) is int:
            faciesIndex = np.array([faciesIndex])


        l_pos = []
        for x,y in zip(positions_x,positions_y):

            cell_x, cell_y, z = self.pointToIndex(x,y,1)
            l_pos.append((cell_x, cell_y, z))
        
        u_list = []
        for iu in stratIndex:
            fa_list = []
            for ifa in faciesIndex:
                fake_bh = []
                for i in range(len(positions_x)):

                    cell_x, cell_y, z = l_pos[i] 
                    x = positions_x[i]
                    y = positions_y[i]
                    
                    if extractUnits:
                        #unit log
                        unit_log = []
                        unit_idx = units_data[iu, :, cell_y, cell_x]
                        if unit_idx[-1] != 0:
                            unit_log.append((self.get_unit(ID = unit_idx[-1], type='ID', vb=0), np.round(self.zg[-1], 2)))

                        for index_trans in reversed(np.where(np.diff(unit_idx) != 0)[0]):
                            unit_log.append((self.get_unit(ID =unit_idx[index_trans], type='ID', vb=0), np.round(self.zg[index_trans+1], 2)))
                    else:
                        unit_log=None

                    #facies log
                    if extractFacies:
                        facies_log = []
                        facies_idx = facies_data[iu, ifa, :, cell_y, cell_x]
                        #If model has no free space above, we need to add the transition Surface - first layer
                        if facies_idx[-1] != 0:
                            facies_log.append((self.get_facies_obj(ID = facies_idx[-1], type='ID', vb=0), np.round(self.zg[-1], 2)))

                        for index_trans in reversed(np.where(np.diff(facies_idx) != 0)[0]):
                            fa_obj=self.get_facies_obj(ID =facies_idx[index_trans], type='ID', vb=0)
                            iz=np.round(self.zg[index_trans+1], 2)
                            facies_log.append((fa_obj, iz))

                    else:
                        facies_log=None

                    # merge None if same unit above and below
                    if unit_log is not None:
                        c=0
                        for i in range(len(unit_log)):
                            if i == len(unit_log)-1 and unit_log[i][0] is None:
                                unit_log = unit_log[:-1]
                                break
                            i-=c
                            if unit_log[i][0] is None and i > 0:
                                if unit_log[i-1][0] == unit_log[i+1][0]:
                                    unit_log=unit_log[:i] + unit_log[(i+2):]
                                c+=2

                    # if no info set logs to None
                    if facies_log is not None:
                        if len(facies_log) == 0:
                            facies_log=None

                    if unit_log is not None:
                        if unit_log:            
                            unit_log[0] = (unit_log[0][0], self.top[cell_y,cell_x]-self.get_sz()/10)
                        else:
                            unit_log = None

                    if unit_log is not None or facies_log is not None:
                        fake_bh.append(borehole("fake","fake",x=x,y=y,z=self.top[cell_y,cell_x]-self.get_sz()/10,
                                                depth=self.top[cell_y,cell_x]-self.bot[cell_y,cell_x],log_strati=unit_log,log_facies=facies_log))
                    else:
                        if vb:
                            print("Borehole at positon ({}, {}) is outside of a simulation zone".format(x, y))
                
                fa_list.append(fake_bh)
            u_list.append(fa_list)
        return u_list

    def add_fake_bh(self, bhs):

        """
        Method to add a fake borehole, list if multiples
        Use for inversion purposes
        """

        try:
            for i in bhs:
                if (isinstance(i, borehole)) and (i not in self.list_fake_bhs):
                    if self.check_bh_inside(i):
                        self.list_fake_bhs.append(i)
                        if self.verbose:
                            print("Borehole {} added".format(i.ID))
                else:
                    if self.verbose:
                        print("object isn't a borehole object or object is already in the list")
        except: # boreholes not in a list
            if (isinstance(bhs, borehole)) and (bhs not in self.list_fake_bhs):
                if self.check_bh_inside(bhs):
                    self.list_fake_bhs.append(bhs)
                    if self.verbose:
                        print("Borehole {} added".format(bhs.ID))
            else:
                if self.verbose:
                    print("object isn't a borehole object or object is already in the list")

    ## geological map functions
    def compute_geol_map(self, iu=0, color=False):
        
        """
        Compute and return the geological map for given unit realization
        iu  : int, unit realization index
        """

        ny = self.get_ny()
        nx = self.get_nx()
        nz = self.get_nz()
        arr = self.get_units_domains_realizations(iu)
        geol_map = np.zeros([ny, nx], dtype=np.int8)
        for iz in np.arange(nz-1, 0, -1):
            iy,ix=np.where(arr[iz] != 0)
            slic = geol_map[iy, ix]
            slic[slic == 0] = arr[iz, iy, ix][slic == 0]
            geol_map[iy, ix] = slic

        if color:
            arr_plot = np.ones([ny, nx, 4])

            for iv in np.unique(geol_map):
                
                if iv != 0:
                    arr_plot[geol_map == iv, :] = matplotlib.colors.to_rgba(self.get_unit(ID = iv, type="ID").c)
                else:
                    arr_plot[geol_map == iv, :] = (1, 1, 1, 1)

            return arr_plot

        return geol_map

    def add_geological_map(self, raster):

        """
        Add a geological map to Arch_table

        ### inputs ###
        raster : 2D nd array of size (ny, nx). Values are units IDs
        """


        self.geol_map = raster
        if self.verbose:
            print("Geological map added")

    def geol_contours(self, step = 5):
        
        """
        This function extract information at the boundaries
        between units from the given geological map 
        (see. Arch_table.add_geological_map)        
        """

        # some functions
        def unit_contact(u1_id, u2_id, geol_map):
        
            from skimage import measure
            
            arr = geol_map.astype(np.float32).copy()
            arr[geol_map == u1_id] = 1
            arr[geol_map == u2_id] = 2
            arr[(geol_map != u1_id) & (geol_map != u2_id)] = np.nan
            
            contours = measure.find_contours(arr, 1)
        
            return contours


        def combi_p2(lst):

            res = []
            for i in range(len(lst) - 1):

                o2 = i+1
                for o in range(i, len(lst)-1):
                    res.append((lst[i], lst[o2]))
                    o2 += 1   
            return np.array(res)  


        # retrieve some things
        geol_map = self.geol_map  
        top = self.top
        sz = self.get_sz()
        xgc = self.get_xgc()
        ygc = self.get_ygc()
        
        ids_in_geol_map = np.unique(geol_map)
        
        combis = combi_p2(ids_in_geol_map)  # possible combinations of unit ids
        lst = []
        for u1_id, u2_id in combis:
            
            u1 = self.get_unit(ID=u1_id, type="ID", vb=0)
            u2 = self.get_unit(ID=u2_id, type="ID", vb=0)
            
            if u1 is not None and u2 is not None:
                conts_u1_u2 = unit_contact(u1_id, u2_id, geol_map)  # extract contacts between u1 and u2
                
                if u1.mummy_unit != u2.mummy_unit:  # units are not from the same (sub)-pile
                    if u1.get_big_mummy_unit() < u2.get_big_mummy_unit():
                        u1_above = True
                    else:
                        u1_above = False
                else:  # units are from the same pile
                    if u1 < u2:
                        u1_above=True
                    else:
                        u1_above=False
                
                for cont in conts_u1_u2:
                    for iy, ix in cont[::step]:
                        iy = int(iy)
                        ix = int(ix)
                        z = top[iy ,ix]-1e-3
                        if u1_above:
                            bh = borehole("contact_bh", "contact_bh", xgc[ix], ygc[iy], z, sz/2, [(u1, z),(u2, z-sz/2)])
                        else : 
                            bh = borehole("contact_bh", "contact_bh", xgc[ix], ygc[iy], z, sz/2, [(u2, z),(u1, z-sz/2)])
                        lst.append(bh)
            
        return lst

    def process_geological_map(self, typ="all", step = 5):
        
        """
        Process the geological map attributed to ArchTable model. 
        This function creates fake boreholes from a given geological map (raster) and them to list_map_bhs

        ### inputs ###
        typ    : str, flag to indicate what information to take,
                 "uniform" for only superficial information (no contact or boundaries)
                 "boundaries" for only the contact between the units
                 "all" for both
        step   : int, step for sampling the geological map, small values implies 
                that much more data are sampled from the raster but this increases
                the computational burden.

        """
        
        xg = self.get_xg()
        yg = self.get_yg()
        xgc = self.get_xgc()
        ygc = self.get_ygc()
        sz  = self.get_sz()
        
        raster = self.geol_map
        assert raster.shape == (self.get_ny(), self.get_nx()), "invalid shape for geological map, should be ({}, {})".format(self.get_ny(), self.get_nx())
        
        self.list_map_bhs=[]

        sample_raster = False
        sample_boundaries = False

        if typ=="all":
            sample_raster = True
            sample_boundaries = True
        elif typ == "uniform":
            sample_raster = True
        elif typ == "boundaries":
            sample_boundaries = True

        if sample_raster:
            mask2d = self.mask.any(0)
            bhs_map = []
            for ix in np.arange(0, len(xg)-1, step):
                for iy in np.arange(0, len(yg)-1, step):
                    if mask2d[iy, ix]:
                        unit_id = raster[iy, ix]
                        unit = self.get_unit(ID=unit_id, type="ID", vb=0)
                        if unit is not None:
                            z = self.top[iy, ix]-1e-3
                            bh = borehole("raster_bh", "raster_bh", xgc[ix], ygc[iy], z, sz/4, [(unit, z)])

                            bhs_map.append(bh)
            
            self.list_map_bhs += bhs_map

        if sample_boundaries:
            bhs = self.geol_contours(step=step)
            self.list_map_bhs += bhs

        if self.verbose:
            print("Geological map extracted - processus ended successfully")

    # remove boreholes
    def rem_all_bhs(self, fake_only=False, geol_map_only=False):

        """Remove all boreholes from the list"""

        if fake_only:
            self.list_fake_bhs=[]
            if self.verbose:
                print("Fake boreholes removed")
        if geol_map_only:
            self.list_map_bhs=[]
            if self.verbose:
                print("Boreholes from geological map removed")
        else:
            self.list_fake_bhs=[]
            self.list_bhs=[]
            self.list_map_bhs=[]
            if self.verbose:
                print("boreholes removed")

    def rem_bh(self, bh):

        """
        Remove a given bh from the list of boreholes
        """

        if bh in self.list_bhs:
            self.list_bhs.remove(bh)
            if self.verbose:
                print("Borehole {} removed".format(bh.ID))
        else:
            if self.verbose:
                print("Borehole {} not in the list".format(bh.ID))

    def rem_fake_bh(self, bh):

        if bh in self.list_fake_bhs:
            self.list_fake_bhs.remove(bh)
            if self.verbose:
                print("Borehole {} removed".format(bh.ID))
        else:
            if self.verbose:
                print("Borehole {} not in the list".format(bh.ID))

    def erase_hd(self):

        """
        Erase the hard data from all surfaces and facies
        """

        for unit in self.get_all_units():
            s=unit.surface
            s.x=[]
            s.y=[]
            s.z=[]
            s.ineq=[]
        for fa in self.get_all_facies():
            fa.x=[]
            fa.y=[]
            fa.z=[]
        if self.verbose:
            print("Hard data reset")

    def rem_all_facies_from_units(self):

        """
        To remove facies from all units
        """

        for unit in self.get_all_units():
            unit.rem_facies(all_facies=True)

        if self.verbose:
            print("All facies have been removed from units")

    def order_Piles(self):

        """
        Order all the units in all the piles according to order attribute
        """
        if self.verbose:
            print("##### ORDERING UNITS ##### ")

        def ord_fun(pile):
            pile.order_units(vb=self.verbose) # organize the pile
            for s in pile.list_units: #check subunits
                if s.f_method == "SubPile":
                    ord_fun(s.SubPile)
        ord_fun(self.get_pile_master())


    def add_prop_hd(self,prop,x,v):

        """
        Add Hard data to the property "prop"

        #inputs#
        prop : string, property name
        x    : ndarray of size (ndata, 3), x, y and z coordinates of hd points
        v    : array of size (ndata), HD property values at x position
        """

        prop=self.list_props(self.getpropindex(prop))
        prop.add_hd(x,v)

    def hd_un_in_unit(self, unit, iu=0):

        """
        Extract sub-units hard data for a unit
        """

        mask = self.Geol.units_domains[iu] == unit.ID 
        hd=[]
        sub_units=[]
        for un in unit.SubPile.list_units:
            for ix,iy,iz in zip(un.x, un.y,un.z):
                cell=self.coord2cell(ix,iy,iz)
                if cell is not None:
                    if mask[cell]:
                        hd.append((ix, iy, iz))
                        sub_units.append(un.ID)
                    else:
                        pass

        return hd, sub_units

    def hd_fa_in_unit(self, unit, iu=0):

        """
        Extract facies hard data for a unit and send warning
        if a hard data should not be in the unit
        """

        mask=self.unit_mask(unit.name, iu=iu)
        hd=[]
        facies=[]
        errors={}
        for fa in self.get_all_facies():
            fa_err=0
            for ix,iy,iz in zip(fa.x, fa.y,fa.z):
                cell=self.coord2cell(ix,iy,iz)
                if mask[cell]:
                    if fa not in unit.list_facies:
                        fa_err += 1
                        #if self.verbose:
                            #print("Warning facies {} have been found inside unit {}".format(fa.name, unit.name))
                    else:
                        hd.append((ix, iy, iz))
                        facies.append(fa.ID)
                else:
                    pass

            errors[fa.name]=fa_err

        #print errors
        if np.sum(list(errors.values())) > 0:
            if self.verbose:
                print("Some errors have been found \nSome facies were found inside units where they shouldn't be \n\n### List of errors ####")
                for k,v in errors.items():
                    if v > 0:
                        print("Facies {}: {} points".format(k,v))

                print("\n")

        return hd, facies

    def compute_distribution(self):

        if self.verbose:
            print("\n ## Computing distributions for Normal Score Transform ##\n")

        for unit in self.get_all_units(recompute=True):

            if unit.surface.N_transfo:

                data = np.array(unit.surface.z)
                tau = unit.surface.dic_surf["tau"]
                bandwidth_mult = unit.surface.dic_surf["bandwidth_mult"]
                n = len(data)

                if n > 10:

                    if bandwidth_mult > 0:

                        from sklearn.neighbors import KernelDensity
                        from scipy.stats import iqr

                        bandwidth = bandwidth_mult* 0.9 * min (np.std(data), iqr(data)) * n**(-1/5)
                        if bandwidth > 0:

                            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data.reshape(-1, 1))
                            data_kern = kde.sample(1000)
                            di = store_distri(data_kern, t=0)
                            unit.surface.distribution = di
                        else:
                            pass
                    else:

                        di = store_distri(data_kern, t=0)
                        unit.surface.distribution = di

                else:
                    if self.verbose:
                        print("Not enough data points to estimate a cdf and use Normal Score Transform for unit {} \n".format(unit.name))
                    unit.surface.N_transfo = False
                    unit.surface.dic_surf["N_transfo"] = False
           

    def estimate_surf_params(self, default_covmodel=None, auto=False, **kwargs):

        """
        Alias for infer surface in ArchPy.infer

        auto   : bool, to automatically infer parameter (True) or not (False)
        kwargs : various kwargs and parameters that can be passed to ArchPy.infer.infer_surface or ArchPy.infer.fit_surfaces
        """

        import ArchPy.infer as api
        # default surface covmodel
        if auto:
            print("### SURFACE PARAMETERS ESTIMATION ### \n")
            for u in self.list_all_units:
                print("### UNIT : {} ### \n".format(u.name))
                api.infer_surface(self, u, default_covmodel=default_covmodel, **kwargs)
        else:
            api.fit_surfaces(self, default_covmodel=default_covmodel, **kwargs)
        


    # def estimate_facies_params(self, **kwargs):

    #     return

    # def get_prop_units(self, depth_max=np.inf, depth_min=0, ignore_units=[], mask=None):

    #     """
    #     Function that returns the proportions of the units in the boreholes

    #     # inputs #
    #     depth_max    : float, maximum depth of investigation in the boreholes
    #     depth_min    : float, minimum depth of investigation in the boreholes
    #     ignore_units : list of str, units name to ignore during the analysis
    #     mask         : 2D ndarray of size (ny, nx), mask where to analyse 
    #                    the borholes
                       
    #     # output #
    #     dictionnary of units proportions
    #     """
        
    #     list_bhs = self.list_bhs
        
    #     if mask is not None:
    #         new_l_bhs = []
    #         for bh in list_bhs:
    #             iy, ix = self.coord2cell(bh.x, bh.y)
    #             if mask[iy, ix]:
    #                 new_l_bhs.append(bh)
    #         list_bhs = new_l_bhs
        
        
    #     meters_units = {}

    #     for bh in list_bhs:

    #         if bh.log_strati is not None:

    #             thk = 0

    #             n_units = len(bh.log_strati)

    #             for i in range(n_units-1):
    #                 s2 = bh.log_strati[i+1]
    #                 s = bh.log_strati[i]

    #                 if s[0] is not None:
    #                     if s[0].name not in meters_units.keys():
    #                         meters_units[s[0].name] = 0

    #                 if bh.z - s2[1] < depth_max:

    #                     if s[0] is not None:
    #                         thk += s[1] - s2[1]
    #                         meters_units[s[0].name] += s[1] - s2[1]
    #                 else:
    #                     if s[0] is not None:
    #                         bot = bh.z - depth_max
    #                         top = s[1]
    #                         if bh.z - top > depth_max:
    #                             top = bh.z - depth_max

    #                         thk += top - bot
    #                         meters_units[s[0].name] += top - bot

    #             # last unit
    #             if n_units > 1:
    #                 s = bh.log_strati[i+1]
    #             elif n_units == 1:
    #                 s = bh.log_strati[0]

    #                 if s[0] is not None: 

    #                     if s[0].name not in meters_units.keys():
    #                         meters_units[s[0].name] = 0

    #                     bot = bh.z - bh.depth
    #                     top = s[1]

    #                     if bh.z - bot > depth_max:
    #                         bot = bh.z - depth_max
    #                     if bh.z - top > depth_max:
    #                         top = bh.z - depth_max

    #                     meters_units[s[0].name] += top - bot
    #                     thk += top - bot

    #         prop_units = {}

    #         tot = 0
    #         for k,v in meters_units.items():
    #             if k not in ignore_units:
    #                 tot += v

    #         for k,v in meters_units.items():
    #             if k not in ignore_units:
    #                 prop_units[k] = v/tot

    #     return prop_units

    def get_proportions(self, type="units", depth_min=0, depth_max=np.inf, ignore_units=[], mask=None):

        """
        Function that returns the proportions of the units in the boreholes

        # inputs #
        depth_max    : float, maximum depth of investigation in the boreholes
        depth_min    : float, minimum depth of investigation in the boreholes
        ignore_units : list of str, units name to ignore during the analysis
        mask         : 2D ndarray of size (ny, nx), mask where to analyse 
                       the borholes
                       
        # output #
        dictionnary of units proportions
        """
        
        list_bhs = self.list_bhs

        if mask is not None:
            new_l_bhs = []
            for bh in list_bhs:
                iy, ix = self.coord2cell(bh.x, bh.y)
                if mask[iy, ix]:
                    new_l_bhs.append(bh)
            list_bhs = new_l_bhs


        meters_units = {}

        for bh in list_bhs[:]:

            z_min = bh.z - depth_max  # minimum altitude of investigation
            z_max = bh.z - depth_min  # maximum altitude of investigation

            if type == "units":
                log = bh.log_strati
            elif type == "facies":
                log = bh.log_facies

            if log is not None:

                thk = 0

                n_units = len(log)

                for i in range(n_units):

                    analysis = True

                    if i == n_units-1:
                        s = log[-1]
                        top = s[1]
                        bot = bh.z - bh.depth

                    else:

                        s2 = log[i+1]
                        s = log[i]

                        top = s[1]
                        bot = s2[1]
                    
                    if bot >= z_max:  # do nothing because interval above investigation altitudes
                        analysis = False

                    elif bot < z_max and top > z_max and bot >= z_min:  # case where top is above depth min
                        top = z_max
                        
                    elif top > z_max and bot < z_min:  # case where top above depth min and bot below depth max
                        top = z_max
                        bot = z_min

                    elif bot < z_max and top < z_max and bot > z_min and top > z_min:  # inside
                        pass

                    elif top <= z_max and bot < z_min and top > z_min:  # 
                        bot = z_min

                    elif top <= z_min:
                        analysis = False
                    
                    else:
                        pass

                    if analysis:
                        if s[0] is not None:
                            if s[0].name not in meters_units.keys():
                                meters_units[s[0].name] = 0

                            thk += top - bot
                            meters_units[s[0].name] += top - bot

            prop_units = {}

            tot = 0
            for k,v in meters_units.items():
                if k not in ignore_units:
                    tot += v

            for k,v in meters_units.items():
                if k not in ignore_units:
                    prop_units[k] = v/tot

        return prop_units

    def process_bhs(self, step=None, facies=True, stop_condition=False):

        """
        ArchPy Pre-processing algorithm
        Extract hard data from boreholes for all units given
        the Piles defined in the Arch_table object.

        #parameters#
        step : float, vertical interval for extracting borehole
                 facies information, default is sz from simulation grid
        facies: bool, flag to indicate to process facies data or not
        stop_condition: flag bool, flag to indicate if the process must be
                         aborted if a inconsistency in bhs is found (False)
                         or bhs will be simply ignored (True)

        """

        # functions
        def check_bhs_consistency(bhs_lst, stop_condition=False):

            """
            Check that boreholes are not in the same cell,
            have correct units/facies info, etc.
            """

            bhs_lst_cop=bhs_lst.copy()
            stop=0
            no_gap = True
            ## check consistency ##
            for bh in bhs_lst_cop:
                s_bef=0
                i=0
                if bh.log_strati is not None:

                    ## determine pile nature
                    try:
                        nature = bh.log_strati[0][0].mummy_unit.SubPile.nature
                    except:
                        nature = "surfaces"

                    for s in bh.log_strati:

                        if s[0] is not None or s[0] == "comf":  # if there is unit info (not a gap)
                            if s_bef == 0 and no_gap:
                                s_bef=s
                                if s[1] != bh.z:
                                    if self.verbose:
                                        print("First altitude in log strati of bh {} is not set at the top of the borehole, altitude changed".format(bh.ID))
                                    bh.log_strati[0]=(s[0], bh.z)
                                    s_bef=s
                            elif s_bef != 0:
                                # check if unit appear only one time
                                c=0
                                for s2 in bh.log_strati:
                                    if s2[0] is not None:
                                        c+= (s2[0] == s[0])

                                #check consistency with pile
                                if s[0].order < s_bef[0].order:
                                    if nature == "surfaces":
                                        if self.verbose:
                                            print("borehole {} not consistent with the pile".format(bh.ID)) # remove and not stop
                                        if stop_condition:
                                            stop=1
                                        else:
                                            #remove borehole from lists
                                            remove_bh(bh, bhs_lst)
                                            break
                                #check height
                                elif s[1] > s_bef[1]:
                                    if self.verbose:
                                        print("Height in log_strati in borehole {} must decrease with depth".format(bh.ID))
                                    if stop_condition:
                                        stop=1
                                    else:
                                        #remove borehole from lists
                                        remove_bh(bh, bhs_lst)
                                        break
                                #check if unit appear only one time
                                elif c > 1:
                                    if nature == "surfaces":
                                        if self.verbose:
                                            print("Unit {} appear more than one time in log_strati of borehole {}".format(s[0].name, bh.ID))
                                        if stop_condition:
                                            stop=1
                                        else:
                                            #remove borehole from lists
                                            remove_bh(bh, bhs_lst)
                                            break
                            s_bef=s
                        else:
                            if i == 0:
                                no_gap=False
                        i += 1

                if bh.log_facies is not None:
                    i=0
                    for fa in bh.log_facies:
                        if i == 0:
                            fa_bef=fa
                            if fa[1] != bh.z:
                                if self.verbose:
                                    print("First altitude in log facies of bh {} is not set at the top of the borehole, altitude changed".format(bh.ID))
                                bh.log_facies[0]=(fa[0], bh.z)
                                fa_bef=fa
                        else:
                            if fa[1] > fa_bef[1]:
                                if self.verbose:
                                    print("Height in log_facies in borehole {} must decrease with depth".format(bh.ID))
                                if stop_condition:
                                    stop=1
                                else:
                                    #remove borehole from lists
                                    remove_bh(bh, bhs_lst)
                        fa_bef=fa
                        i += 1

            if stop:
                if self.verbose:
                    print("Process bhs aborted")
                    return None

        def remove_bh(bh, bhs_lst):
            #if bh in self.list_bhs:
            #    self.rem_bh(bh)
            #elif bh in self.list_fake_bhs:
            #    self.rem_fake_bh(bh)
            if bh in bhs_lst:
                bhs_lst.remove(bh) # remove borehole from current list

        #merge lists
        bhs_lst=self.list_bhs + self.list_fake_bhs + self.list_map_bhs

        #get grid
        xg=self.get_xg()
        yg=self.get_yg()
        zg=self.get_zg()

        #order units
        self.order_Piles()

        #set hierarchy_relations
        self.hierarchy_relations(vb=self.verbose)

        if len(self.list_bhs + self.list_fake_bhs + self.list_map_bhs) == 0:
            if self.verbose:
                print("No borehole found - no hd extracted")
            #self.bhs_processed=1
            return

        ### extract strati units and facies info from borehole data
        if self.bhs_processed == 0:

            self.sto_hd = []
            ### check multiple boreholes in the same cell ###
            t=KDTree(self.xu2D)

            xbh=[i.x for i in bhs_lst]
            ybh=[i.y for i in bhs_lst]
            c_bh=np.array([xbh, ybh]).T  # boreholes coordinates

            idx=t.query(c_bh, return_distance=False)
            bh_arr=np.array(bhs_lst) # array of boreholes
            idx_removed=[]
            for idx1 in idx:
                if idx1 not in idx_removed:
                    mask=(idx1 == idx).reshape(-1)
                    bhs_cell=bh_arr[mask] #boreholes in the same cell

                    if len(bhs_cell) > 1: # more than one borehole in one cell
                        if self.verbose:
                            print("Multiples boreholes {} were found inside the same cell, the deepest will be kept".format([i.ID for i in bhs_cell]))
                        depths=(np.array([i.depth for i in bhs_cell]))
                        mask2=(depths == max(depths))
                        if sum(mask2) == 1:
                            for (bh, chk) in zip(bhs_cell, mask2):
                                if ~chk:
                                    #remove borehole from lists
                                    remove_bh(bh, bhs_lst)
                                    idx_removed.append(idx1)
                        elif sum(mask2) > 1: #bhs have the same depths --> remove others
                            #remove borehole from lists
                            for i in range(1, len(bhs_cell)):
                                remove_bh(bhs_cell[i], bhs_lst)
                            idx_removed.append(idx1)
                        else:
                            if self.verbose:
                                print("Error this shouldn't happen :)")

            if len(bhs_lst) == 0:
                if self.verbose:
                    print("No valid borehole, processus aborted")

                return None

            ## SPLIT BOREHOLES ##
            # logs must be splitted by hierarchy group and linked with assigned Pile
            new_bh_lst=[]
            for bh in bhs_lst:
                if bh.log_strati is not None:
                    if len([i[0] for i in bh.log_strati if i[0] is not None]) > 0:
                        for new_bh in split_logs(bh):
                            new_bh_lst.append(new_bh)
                elif bh.log_facies is not None:
                    new_bh_lst.append(bh)

            #check consistency
            check_bhs_consistency(new_bh_lst)

            def add_contact(s, x, y, z, type="equality", z2=None): #function to add an hd point to a surface
                if self.coord2cell(x, y, z) is not None:
                    if type == "equality":
                        s.surface.x.append(x)
                        s.surface.y.append(y)
                        s.surface.z.append(z)
                    elif type == "ineq_inf":
                        s.surface.ineq.append([x, y, 0, z, np.nan])
                    elif type == "ineq_sup":
                        s.surface.ineq.append([x, y, 0, np.nan, z])
                    elif type == "double_ineq":
                        s.surface.ineq.append([x, y, 0, z, z2])  # inferior and upper ineq

            if len(new_bh_lst) == 0:
                if self.verbose:
                    print("No valid borehole, processus aborted")
                return None


            #### PROCESSING (ArchPy Algorithm) ######
            #### Facies ####
            if facies:
                if step is None:
                    step=np.abs(zg[1] - zg[0]) # interval spacing to sample data
                for bh in new_bh_lst:
                    if bh.log_facies is not None:
                        for zi in np.arange(bh.z+step/2, bh.z-bh.depth, -step): # loop over all the borehole from top to botom
                            if zi < bh.z:
                                idx=np.where(zi < np.array(bh.log_facies)[:, 1])[0][-1]
                                fa=bh.log_facies[idx][0]
                                if fa is not None: # if there is data
                                    if self.coord2cell(bh.x, bh.y, zi) is not None:
                                        fa.x.append(bh.x)
                                        fa.y.append(bh.y)
                                        fa.z.append(zi)

            # # remove None at end of bh
            # while [bh for bh in new_bh_lst if bh.log_strati[-1][0] is None]:
            #     for bh in new_bh_lst:
            #         if bh.log_strati[-1][0] is None:
            #             new_bh_lst.remove(bh)
            #             new_bh = bh
            #             new_bh.log_strati = new_bh.log_strati[:-1]
            #             new_bh_lst.append(new_bh)

            # print([i.log_strati for i in new_bh_lst])

            #### Units ####
            for bh in new_bh_lst:
                if bh.log_strati is not None:
                    for i, s in enumerate(bh.log_strati):  # loop over borehole (i: index, s: tuple (strati, altitude of contact))
                        if s[0] is not None:  # if there is unit info
                            s1=s[0]  # unit of interest (first element is strati object)

                            # get pile
                            if s1.mummy_unit is not None:
                                Pile=s1.mummy_unit.SubPile  # Link with pile
                            elif s1 in self.Pile_master.list_units:
                                Pile=self.Pile_master

                            if Pile.nature == "surfaces":
                                if (i == 0) and (s1.order == 1):  # first unit encountered is also first unit in pile
                                    pass

                                else:
                                    non_unit=True  # flag to know if above unit is None or not
                                    if i == 0: # first unit in the log
                                        s_above_order=1  # second unit in pile (first is ignored)
                                        non_unit=False  # flag

                                    elif i <= len(bh.log_strati):  # others units encountered except first one
                                        s_above=bh.log_strati[i-1]  # unit just above unit of interest in bh
                                        # check above is not None
                                        if s_above[0] is not None:
                                            s_above_order=s_above[0].order
                                            non_unit=False  # flag

                                    if non_unit == False:
                                        if s1.contact == "comf":
                                            for il in range(s_above_order, s1.order-1):
                                                s2=Pile.list_units[il]
                                                if s2.surface.contact == "erode":
                                                    add_contact(s2, bh.x, bh.y, s[1], "ineq_inf")
                                                elif s2.surface.contact == "onlap":
                                                    add_contact(s2, bh.x, bh.y, s[1], "ineq_sup")

                                        elif s1.contact != "comf":
                                            erod_lst=[]  # list of erode units
                                            erod_lst_2=[]  # list of erode surfaces (no unit, don't have volume)
                                            for il in range(s_above_order, s1.order-1): # check if overlaying layers are erode
                                                s2=Pile.list_units[il]
                                                if s2.surface.contact == "erode" :
                                                    if s2.contact == "onlap":
                                                        erod_lst.append(s2)
                                                    elif s2.contact == "erode":
                                                        erod_lst_2.append(s2)

                                            if erod_lst:  # if at least one erode layer exists above --> add equality point to erode
                                                s2=min(erod_lst)  # select higher one
                                                if i == 0:  # first unit at topography must be ineq_inf --> all erosions must go above
                                                    add_contact(s2, bh.x, bh.y, s[1], "ineq_inf")
                                                else:
                                                    add_contact(s2, bh.x, bh.y, s[1], "equality")

                                                add_contact(s1, bh.x, bh.y, s[1], "ineq_inf")

                                                # supplementary info, layer above erosion layer must go below and other erosion layer must go above
                                                for il in range(s_above_order, s1.order-1):
                                                    s_erod=Pile.list_units[il]
                                                    if s_erod.order < s2.order and i > 0:  # non eroded layers --> not deposited a checker
                                                        add_contact(s_erod, bh.x, bh.y, s[1], "ineq_sup")
                                                    if s_erod.surface.contact == "erode":
                                                        add_contact(s_erod, bh.x, bh.y, s[1], "ineq_inf")

                                            elif erod_lst_2 and len(erod_lst) == 0:  # only erode surfaces
                                                l = []
                                                # case no erosion
                                                case = []
                                                if i == 0:
                                                    case.append((s1, "ineq_inf"))  # unit at the topography must go above topo
                                                else:
                                                    case.append((s1, "equality"))  # else equality

                                                for il in range(s_above_order, s1.order-1):  # add inequality sup to other above layers
                                                    s_12=Pile.list_units[il]  # unit above s1
                                                    if s_12.surface.contact == "onlap" and i > 0:
                                                        case.append((s_12, "ineq_sup"))
                                                    elif s_12.contact == "erode":
                                                        case.append((s_12, "ineq_inf"))

                                                l.append(case)        
                                                for er in erod_lst_2[::-1] :  # cases of erosion (one for each er. surface)
                                                    case = []
                                                    case.append((er, "equality"))
                                                    case.append((s1, "ineq_inf"))
                                                    for il in range(s_above_order, s1.order-1):
                                                        s_erod=Pile.list_units[il]

                                                        if s_erod.order < er.order:
                                                            if s_erod.surface.contact == "onlap":
                                                                case.append((s_erod, "ineq_sup"))
                                                            elif s_erod.surface.contact == "erode":
                                                                case.append((s_erod, "ineq_inf"))
                                                        elif s_erod.order > er.order :
                                                            if s_erod.surface.contact == "erode":
                                                                case.append((s_erod, "ineq_inf"))  # erode layers below er goes above

                                                    l.append(case)
                                                p = np.ones(len(l))*(1/len(l))  # user input ? 
                                                self.sto_hd.append(((bh.x, bh.y, s[1]), l, p))

                                            elif erod_lst and erod_lst_2:  # both (TO DO)
                                                pass

                                            else: # no erode layer --> exact data on surface of interest (s1) and ineq sup to others above
                                                if i == 0:
                                                    add_contact(s1, bh.x, bh.y, s[1], "ineq_inf")  # unit at the topography must go above topo
                                                else:
                                                    add_contact(s1, bh.x, bh.y, s[1], "equality")  # else equality

                                                for il in range(s_above_order, s1.order-1): # add inequality sup to other above layers
                                                    s_12=Pile.list_units[il] # unit above s1
                                                    if s_12.surface.contact == "onlap" and i > 0:
                                                        add_contact(s_12, bh.x, bh.y, s[1], "ineq_sup")

                                    else:  # unit above is a gap  TO DO to update with new approach
                                        if i == 1:  # if second unit in log and above is None
                                            add_contact(s1, bh.x, bh.y, s[1], "ineq_inf")  # unit must go above
                                        else:
                                            s_gap=s_above  # store gap information
                                            s_above=bh.log_strati[i-2]  # take unit above gap
                                            s_above_order=s_above[0].order

                                            erod_lst=[]
                                            for il in range(s_above_order, s1.order-1):  # check if overlaying layers are erode
                                                s2=Pile.list_units[il]
                                                if s2.surface.contact == "erode":
                                                    erod_lst.append(s2)

                                            if erod_lst: # if at least one erode layer exists above
                                                s2=min(erod_lst)  # select higher one
                                                add_contact(s2, bh.x, bh.y, s[1], z2=s_gap[1], type="double_ineq")  # erode surface must go between gap
                                                add_contact(s1, bh.x, bh.y, s[1], "ineq_inf")

                                                # supplementary info, layer above erosion layer must go below and other erosion layer must go above
                                                for il in range(s_above_order, s1.order-1):
                                                    s_erod=Pile.list_units[il]
                                                    if s_erod.order < s2.order and i > 0:  # non eroded layers --> not deposited a checker
                                                        add_contact(s_erod, bh.x, bh.y, s[1], "ineq_sup")
                                                    elif s_erod.order > (s2.order):  # layers below erosion horizon --> must go above
                                                        if s_erod.surface.contact == "erode":
                                                            add_contact(s_erod, bh.x, bh.y, s[1], "ineq_inf")

                                            else:  # no erode layer
                                                add_contact(s1, bh.x, bh.y, s[1], z2=s_gap[1], type="double_ineq")  # surface must go between gap

                                                # add upper bound to other above units in pile
                                                for il in range(s_above_order, s1.order-1): # add inequality sup to other above layers
                                                    s_12=Pile.list_units[il]  # unit above s1
                                                    if s_12.surface.contact == "onlap" and i > 0:
                                                        add_contact(s_12, bh.x, bh.y, s_gap[1], "ineq_sup")

                                if (i == len(bh.log_strati)-1) & (s1.order < Pile.list_units[-1].order):  # if last unit is not last in the pile --> below layers must go below
                                    for il in range(s1.order, Pile.list_units[-1].order):
                                        s_12=Pile.list_units[il]

                                        add_contact(s_12, bh.x, bh.y, bh.z-bh.depth, "ineq_sup")

                            elif Pile.nature == "3d_categorical":
                                ## get top and bottom
                                top = s[1]
                                if i != len(bh.log_strati) - 1:
                                    bot = bh.log_strati[i+1][1]
                                else:
                                    bot = bh.z - bh.depth

                                if step is None:
                                    step = self.get_sz() # interval spacing to sample data

                                for iz in np.arange(top, bot, -step):
                                    s1.x.append(bh.x)
                                    s1.y.append(bh.y)
                                    s1.z.append(iz)

                        else:  # if there is a gap ignore and pass
                            pass


            # compute distribution for each unit if necessary
            self.compute_distribution()

            self.bhs_processed=1
            if self.verbose:
                print("Processing ended successfully")
        elif self.bhs_processed == 1:
            if self.verbose:
                print("Boreholes already processed")

    def compute_surf(self, nreal=1, fl_top=True, rm_res_files=True):

        """
        Performs the computation of the surfaces

        #Inputs#
        nreal: int, number of realization
        rm_res_files : bool, flag to remove previous existing resulting files
               in working directory
        fl_top: bool, assign first layer to top of the domain (True by default)
        """

        start=time.time()
        np.random.seed(self.seed)  # set seed


        if nreal == 0:
            if self.verbose:
                print("Warning: nreal is set to 0.")
            return

        if self.bhs_processed == 0:
            print("Boreholes not processed, fully unconditional simulations will be tempted")
        #assert self.bhs_processed == 1, "Boreholes not processed"

        # create work directory if it doesn't exist
        if self.ws not in os.listdir():
            os.mkdir(self.ws)

        # remove preexisting results files in ws
        if rm_res_files:
            for file in os.listdir(self.ws):
                if file.split(".")[-1] in ("unt", "fac", "pro"):
                    fpath=os.path.join(self.ws, file)
                    os.remove(fpath)

        self.Geol.units_domains=np.zeros([nreal, self.get_nz(), self.get_ny(), self.get_nx()], dtype=np.int8)  # initialize tmp result array

        if self.check_units_ID() is None:  # check if units have correct IDs
            return None
        self.get_pile_master().compute_surf(self, nreal, fl_top, vb=self.verbose)  # compute surfs of the first pile

        ## stochastic hard data
        #hierarchies
        def fun(pile):  # compute surf hierarchically
            i=0
            for unit in pile.list_units:
                if unit.f_method == "SubPile":
                    if unit.SubPile.nature == "surfaces":

                        tops=self.Geol.surfaces_by_piles[pile.name][:, i]
                        bots=self.Geol.surfaces_bot_by_piles[pile.name][:, i]
                        unit.SubPile.compute_surf(self, nreal, fl_top=True, subpile=True, tops=tops, bots=bots, vb=self.verbose)
                        fun(unit.SubPile)

                    elif unit.SubPile.nature == "3d_categorical":
                        unit.compute_facies(self, nreal=1, mode="units", verbose=self.verbose)
                        fun(unit.SubPile)
                i += 1

        fun(self.get_pile_master())  # run function

        if self.fill_flag:
            self.fill_top_unit()

        end=time.time()
        if self.verbose:
            print("\n### {}: Total time elapsed for computing surfaces ###".format(end - start))


        # write results
        if self.write_results:
            units_domains=self.Geol.units_domains
            for ireal in range(nreal):

                ud=units_domains[ireal]
                fname=self.name+"_{}.ud".format(ireal)
                fpath=os.path.join(self.ws, fname)
                with open(fpath, "wb") as f:
                    pickle.dump(ud, f)

            #delet units domains in Geol object to for memory space
            del(self.Geol.units_domains)

        self.surfaces_computed=1  # flag


    def define_domains(self, surfaces, fl_top=True):

        """
        Performs the computation of the units domains when surfaces are provided

        #Inputs#
        surfaces : dictionary of surfaces as values and pile name as key
        rm_res_files : bool, flag to remove previous existing resulting files
               in working directory
        fl_top: bool, assign first layer to top of the domain (True by default)
        """

        #np.random.seed(self.seed)  # set seed

        nreal = surfaces[self.get_pile_master().name].shape[0]  # get number of realizations
        self.Geol.units_domains=np.zeros([nreal, self.get_nz(), self.get_ny(), self.get_nx()], dtype=np.int8)  # initialize tmp result array


        if self.check_units_ID() is None:  # check if units have correct IDs
            return None

        self.get_pile_master().define_domains(self, surfaces[self.get_pile_master().name], vb=self.verbose, fl_top=fl_top)

        #hierarchies
        def fun(pile):  # compute surf hierarchically
            i=0
            for unit in pile.list_units:
                if unit.f_method == "SubPile":
                    tops=surfaces[pile.name][:, i]
                    bots=surfaces[pile.name][:, i+1]
                    unit.SubPile.define_domains(self, surfaces[unit.SubPile.name], fl_top=True, subpile=True, tops=tops, bots=bots, vb=self.verbose)
                    fun(unit.SubPile)
                i += 1

        fun(self.get_pile_master())  # run function

        self.surfaces_computed=1  # flag


    def fill_ID(self, arr, ID=0):
    
        """
        Fill ID values in an 3D array given surroundings values using nearest neighbors

        # inputs #
        arr : ndarray of size (nz, ny, nx) --> simulation grid size
        ID  : int, ID to replace
        """

        nx = self.get_nx()
        ny = self.get_ny()
        nz = self.get_nz()

        from sklearn.neighbors import NearestNeighbors

        X = np.ones([nz, ny, nx])* self.xgc
        Y = np.ones([nz, ny, nx])
        Y[:] = np.ones([nx, ny]).T * self.ygc.reshape(-1, 1)
        Z = np.ones([nz, ny, nx])
        Z[:, :] =( np.ones([nz, nx]) * self.zgc.reshape(-1, 1)).reshape(nz, 1, nx)

        xu3D = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

        mask = (arr != ID) * (arr != 0)  # mask of data
        X_fit = xu3D[mask.flatten()]
        y_fit = arr.flatten()[mask.flatten()]
        mask = (arr == ID)  # mask where to fill

        if mask.any():
            X_pred = xu3D[mask.flatten()]

            # fit
            nn = NearestNeighbors(n_neighbors=1).fit(X_fit)

            #pred
            res = nn.kneighbors(X_pred, return_distance=False, n_neighbors=1)

            # assign
            y_pred = y_fit[res]
            arr_test = arr.copy()
            arr_test[mask] = y_pred[:, 0]  # reassign values

        return arr_test


    def fill_top_unit(self, method = "nearest_neighbors"):  # to remove
    
        """
        Function to fill each cells simulated top unit given their nearest neighbour.
        """

        nx = self.get_nx()
        ny = self.get_ny()
        nz = self.get_nz()
        
        # get top unit ID
        top_unit = self.get_all_units(1)[0]
        ID = top_unit.ID
        
        if method == "nearest_neighbors":
            
            from sklearn.neighbors import NearestNeighbors
            
            X = np.ones([nz, ny, nx])* self.xgc
            Y = np.ones([nz, ny, nx])
            Y[:] = np.ones([nx, ny]).T * self.ygc.reshape(-1, 1)
            Z = np.ones([nz, ny, nx])
            Z[:, :] =( np.ones([nz, nx]) * self.zgc.reshape(-1, 1)).reshape(nz, 1, nx)

            xu3D = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
            
            for iu in range(self.nreal_units):
                arr = self.get_units_domains_realizations(iu)
                
                mask = (arr != ID) * (arr != 0)
                X_fit = xu3D[mask.flatten()]
                y_fit = arr.flatten()[mask.flatten()]
                mask = (arr == ID)
                
                if mask.any():
                    X_pred = xu3D[mask.flatten()]

                    # fit
                    nn = NearestNeighbors(n_neighbors=1).fit(X_fit)

                    #pred
                    res = nn.kneighbors(X_pred, return_distance=False, n_neighbors=1)

                    # assign
                    y_pred = y_fit[res]
                    arr_test = arr.copy()
                    arr_test[mask] = y_pred[:, 0]  # reassign values

                    self.Geol.units_domains[iu] = arr_test
        else:
            if self.vb:
                print("Invalid method")

    def compute_facies(self, nreal=1, verbose_methods=0):

        """
        Performs the computation of the facies

        nreal: int, number of realizations
        c_all: bool, compute facies in each units if True.
                If false units must be passed by args and nreal
                will be ignored and taken from previous simulation is possible
        verbose_methods: int (0 or 1), verbose for the facies methods, 0 by default
        """

        if nreal==0:
            if self.verbose:
                print("Warning: nreal set to 0")
            return
        #asserts
        all_ids=[]
        for fa in self.get_all_facies():
            if fa.ID not in all_ids:
                all_ids.append(fa.ID)
            else:
                raise ValueError ("{} facies index has been defined on multiple units")

        #start
        start_tot=time.time()
        np.random.seed (self.seed) # set seed

        #grid
        xg=self.xg
        yg=self.yg
        zg=self.zg

        #initialize array and set number of realization
        self.nreal_fa=nreal
        self.Geol.facies_domains=np.zeros([self.nreal_units, nreal, self.get_nz(), self.get_ny(), self.get_nx()], dtype=np.int8)

        for strat in self.get_all_units():  # loop over strati
            if strat.contact == "onlap":
                if self.verbose:
                    print("\n### Unit {}: facies simulation with {} method ####".format(strat.name, strat.f_method))
                start=time.time()
                strat.compute_facies(self, nreal, verbose=verbose_methods)
                end=time.time()

                if self.verbose:
                    print("Time elapsed {} s".format(np.round((end - start), decimals=2)))

        end=time.time()
        if self.verbose:
            print("\n### {}: Total time elapsed for computing facies ###".format(np.round((end - start_tot), decimals=2)))

        if self.write_results:
            # write results
            fa_domains=self.Geol.facies_domains
            for iu in range(self.nreal_units):
                for ifa in range(self.nreal_fa):
                    fd=fa_domains[iu, ifa]
                    fname=self.name+"_{}_{}.fd".format(iu, ifa)
                    fpath=os.path.join(self.ws, fname)
                    with open(fpath, "wb") as f:
                        pickle.dump(fd, f)

            del(self.Geol.facies_domains)  # delete facies domain to free some memory

        self.facies_computed=1

    def compute_domain(self, s1, s2):

        """
        Return a bool 2D array that define the domain where the units
        exist (between two surfaces, s1 and s2)

        s1, s2: 2D arrays, two given surfaces over simulation domain size: (ny, nx)),
        s1 is top surface, s2 is bot surface
        """

        zg=self.get_zg()
        xg=self.get_xg()
        yg=self.get_yg()
        sz=self.get_sz()
        nx=self.get_nx()
        ny=self.get_ny()
        nz=self.get_nz()

        top=self.top
        bot=self.bot


        z0=zg[0]
        z1=zg[-1]

        s1[s1 < z0]=z0
        s1[s1 > z1]=z1
        s2[s2 < z0]=z0
        s2[s2 > z1]=z1

        idx_s1=(np.round((s1-z0)/sz)).astype(int)
        idx_s2=(np.round((s2-z0)/sz)).astype(int)

        #domain
        a=np.zeros([nz, ny, nx], dtype=bool)
        for iy in range(ny):
            for ix in range(nx):
                a[idx_s2[iy, ix]: idx_s1[iy, ix], iy, ix]=1

        return a


    def compute_prop(self, nreal=1):
        #TO DO --> add an option to resimulate only certain properties (e.g. *args)

        """
        Performs the computation of the properties added to the ArchTable
        nreal: int, number of realizations
        """

        assert len(self.list_props) > 0, "No property have been added to Arch_table object"
        assert nreal > -1, "You cannot make a negative number of realizations, nice try tho"
        if nreal==0:
            if self.verbose:
                print("Warning: nreal is set to 0")
            return
        np.random.seed (self.seed) # set seed
        self.nreal_prop=nreal
        xg=self.xgc
        yg=self.ygc
        zg=self.zgc
        nx=self.get_nx()
        ny=self.get_ny()
        nz=self.get_nz()
        x0=self.get_ox()
        y0=self.get_oy()
        z0=self.get_oz()
        sx=self.get_sx()
        sy=self.get_sy()
        sz=self.get_sz()
        nreal_units=self.nreal_units
        nreal_fa=self.nreal_fa
        self.Geol.prop_values={} #remove previous prop simulations


        for prop in self.list_props:
            counter=0
            if self.verbose:
                print ("### {} {} property models will be modeled ###".format(nreal_units*nreal_fa*nreal, prop.name))
            prop_values=np.zeros([nreal_units, nreal_fa, nreal, nz, ny, nx], dtype=np.float32) # property values

            #HD
            x=prop.x
            v=prop.v

            #loop
            for iu in range(nreal_units): # loop over units models
                for ifa in range(nreal_fa): # loop over lithological models
                    K_fa=np.zeros([nreal, nz, ny, nx], dtype=np.float32)
                    for strat in self.get_all_units():
                        if strat.f_method != "Subpile": #discard units filled by subunits
                            # mask_strat=(self.get_units_domains_realizations(iu) == strat.ID)
                            mask_strat=(self.Geol.units_domains[iu] == strat.ID)  # mask unit
                            for ite,fa in enumerate(strat.list_facies):

                                #create mask facies for prop simulation
                                # facies_domain=self.get_facies(iu, ifa, all_data=False)
                                facies_domain=self.Geol.facies_domains[iu, ifa].copy()  # gather a realization of facies domain
                                facies_domain[facies_domain != fa.ID]=0  # set 0 to other facies
                                facies_domain[~mask_strat]=0  # set 0 outside of the strati to simulate same facies but in other strat independantely
                                mask_facies=facies_domain.copy()
                                mask_facies[mask_facies != 0]=1  # set to 1 for mask

                                #simulate
                                if (fa in prop.facies) and (fa.ID in facies_domain): # simulation of the property (check if facies is inside strat unit)
                                    i=prop.facies.index(fa) # retrieve index
                                    m=prop.means[i] #mean value
                                    covmodel=prop.covmodels[i] # covariance model used
                                    method=prop.int[i] #method of interpolation (sgs or fft)

                                    if method == "fft":
                                        sims=grf.grf3D(covmodel, [nx, ny, nz], [sx, sy, sz], [x0, y0, z0],
                                                         nreal=nreal, mean=m, x=x, v=v, printInfo=False)
                                    elif method == "homogenous":
                                        sims=np.ones([nreal, nz, ny, nx])*m
                                        if x is not None:
                                            if self.verbose:
                                                print("homogenous method chosen ! Warning: Some HD can be not respected")

                                    elif method == "homogenous_uniform":
                                        dat = np.random.uniform(m[0], m[1], nreal)
                                        sims=np.ones([nreal, nz, ny, nx]) * dat[:,np.newaxis,np.newaxis,np.newaxis]
                                        if x is not None:
                                            if self.verbose:
                                                print("homogenous method chosen ! Warning: Some HD can be not respected")

                                    elif method == "sgs":

                                        sims = gci.simulate3D(covmodel, [nx,  ny,  nz],  [sx,  sy,  sz],  [x0,  y0,  z0],
                                                             nreal=nreal, mean=m, mask=mask_facies, x=x, v=v, verbose=0, nthreads=self.ncpu, seed=self.seed + iu*10000 + ite*100 + ifa)["image"].val

                                        sims=np.nan_to_num(sims) #remove nan

                                    elif method == "mps":
                                        #TO DO
                                        pass
                                    else: # not define --> homogenous and default mean value
                                        m=prop.def_mean
                                        sims=np.ones([nreal, nz, ny, nx])*m
                                else:
                                    m=prop.def_mean
                                    sims=np.ones([nreal, nz, ny, nx])*m

                                for ir in range(nreal):
                                    sim=sims[ir]
                                    sim[facies_domain != fa.ID]=0
                                    K_fa[ir] += sim

                    K_fa[:, ~self.mask]=np.nan  #default value is 0 --> warning ?
                    # ma=((self.get_facies(iu, ifa, all_data=False) == 0) * self.mask)
                    ma=((self.Geol.facies_domains[iu, ifa] == 0) * self.mask)
                    K_fa[:, ma]=prop.def_mean

                    if prop.vmin is not None:
                        K_fa[K_fa < prop.vmin]=prop.vmin
                    if prop.vmax is not None:
                        K_fa[K_fa > prop.vmax]=prop.vmax

                    if self.write_results:
                        for ireal in range(nreal):
                            fname=self.name+"{}_{}_{}_{}.pro".format(prop.name, iu, ifa, ireal)
                            fpath=os.path.join(self.ws, fname)
                            with open(fpath, "wb") as f:
                                pickle.dump(K_fa[ireal], f)
                    else:
                        prop_values[iu, ifa]=K_fa

                    counter += nreal
                    if self.verbose:
                        print("### {} {} models done".format(counter, prop.name))

            if ~self.write_results:
                self.Geol.prop_values[prop.name]=prop_values

        self.prop_computed=1

    def physicsforward(self, method, positions, stratIndex=None, faciesIndex=None, propIndex=None, idx=0, cpuID=0):

        import ArchPy.forward as fd
        
        """Alias to the function ArchPy.forward.physicsforward.
        method: string. method to Forward
        position: position of the forward"""
        return fd.physicsforward(self, method, positions, stratIndex, faciesIndex, propIndex, idx, cpuID)

    def unit_mask(self, unit_name, iu=0, all_real=False):

        """
        Return the mask of the given unit for a realization iu

        #inputs#
        unit_name: string, unit name defined when creating the object
                    for more details: ArchPy.base.Unit
        iu      : int, unit realization index (0, 1, ..., Nu)
        all_real: bool, flag to know if a mask of all realizations
                    must be returned

        #outputs#
        3D (or 4D) nd.array of 1 (present) and 0 (absent)
        """

        if all_real: #all realizations
            unit=self.get_unit(name=unit_name, type="name")
            l=[]
            def fun(unit): # Hierarchy unit
                if unit.f_method == "SubPile":
                        for su in unit.SubPile.list_units:
                            l.append(su.ID)
                        if su.f_method == "SubPile":
                            fun(su)
                else:
                    l.append(unit.ID)
            fun(unit)

            arr=np.zeros([self.nreal_units, self.get_nz(), self.get_ny(), self.get_nx()])  # mask with 0 everywhere
            for idx in l:
                arr[self.get_units_domains_realizations() == idx]=1  # set 1 where unit or sub-units are present

        else:
            unit=self.get_unit(name=unit_name, type="name")
            l=[]
            def fun(unit):
                if unit.f_method == "SubPile":
                    for su in unit.SubPile.list_units:
                        l.append(su.ID)
                        if su.f_method == "SubPile":
                            fun(su)
                else:
                    l.append(unit.ID)
            fun(unit)

            arr=np.zeros([self.get_nz(), self.get_ny(), self.get_nx()]) #mask with 0 everywhere
            for idx in l:
                arr[self.get_units_domains_realizations(iu) == idx]=1  #set 1 where unit or sub-units are present

        return arr


    ### others funs ###
    def orientation_map(self, unit, method="simple", iu=0, smooth=2):

        """
        Compute an orientation map for a certain Unit object
        from which we want to have the differents orientations

        #inputs#
        method: str, method to use to infer orientation map
                (simple: vertically interpolate top/bot layer orientation)
        unit : unit object (units simulations must have been performed)
        iu   : idx units (if multiple realization of units)
        smooth: int, half-size windows for rolling mean
        """

        def azi_dip(s):
            dy=-np.diff(s, axis=0)[:, 1: ]/sy
            dx=-np.diff(s, axis=1)[1:,: ]/sx
            e2=(0, 1) # principal direction (North)
            ang=np.ones([dy.shape[0]*dy.shape[1]], dtype=np.float32)
            i=-1
            for ix, iy in zip(dx.flatten(), dy.flatten()):
                i+= 1
                v=(ix, iy)
                if np.sqrt(v[0]**2+v[1]**2) == 0: #divide by 0
                    val=0
                else:
                    val=(np.rad2deg(np.arccos(np.dot(e2, v)/np.sqrt(v[0]**2+v[1]**2))))
                if v[0]<0:
                    val *= -1

                ang[i]=val

            dip=np.rad2deg(np.arctan(-dy/np.cos(np.deg2rad(ang).reshape(ny-1, nx-1))))

            return ang.reshape(ny-1, nx-1), dip

        xg=self.get_xgc()
        yg=self.get_ygc()
        zg=self.get_zgc()
        nx=self.get_nx()
        ny=self.get_ny()
        nz=self.get_nz()
        sx=self.get_sx()
        sy=self.get_sy()
        sz=self.get_sz()

        if unit.get_h_level() == 1:
            pile=self.get_pile_master()
        else:
            pile=unit.mummy_unit.SubPile
        if method == "simple":
            azi_bot=np.ones([ny, nx])
            azi_top=np.ones([ny, nx])
            dip_bot=np.ones([ny, nx])
            dip_top=np.ones([ny, nx])
            top    =running_mean_2D(self.Geol.surfaces_by_piles[pile.name][iu, unit.order-1], N=smooth)
            bot    =running_mean_2D(self.Geol.surfaces_bot_by_piles[pile.name][iu, unit.order-1], N=smooth)

            azi_top[1:, 1: ], dip_top[1:, 1: ]=azi_dip(top)
            azi_bot[1:, 1: ], dip_bot[1:, 1: ]=azi_dip(bot)
            azi_top[0,: ]=azi_top[1,: ]
            azi_top[:, 0]=azi_top[:, 1]
            azi_bot[0,: ]=azi_bot[1,: ]
            azi_bot[:, 0]=azi_bot[:, 1]
            dip_top[0,: ]=dip_top[1,: ]
            dip_top[:, 0]=dip_top[:, 1]
            dip_bot[0,: ]=dip_bot[1,: ]
            dip_bot[:, 0]=dip_bot[:, 1]

            azi=np.zeros([nz, ny, nx], dtype=np.float32)
            mask=self.unit_mask(unit_name=unit.name, iu=iu)
            for ix in range(nx):
                for iy in range(ny): # TO FINISH
                    t=np.where(mask[:, iy, ix]) # retrieve idx to indicate where layer exist vertically at certain location x

                    if len(t[0])>0: # if layer exists at position ix iy
                        i2=np.min(t)
                        i1=np.max(t)
                        vbot=azi_bot[iy, ix]
                        vtop=azi_top[iy, ix]

                        if (np.abs(np.min((vbot, vtop)) - np.max((vbot, vtop)))) < (np.abs(360 + np.min((vbot, vtop)) - np.max((vbot, vtop)))):
                            azi[:, iy, ix]=np.interp(zg, [zg[i2], zg[i1]], [vbot, vtop])
                        else:
                            if vbot < vtop:
                                azi[:, iy, ix]=np.interp(zg, [zg[i2], zg[i1]], [vbot+360, vtop])
                            else:
                                azi[:, iy, ix]=np.interp(zg, [zg[i2], zg[i1]], [vbot, vtop+360])
            azi[mask!=1]=0

            dip=np.zeros([nz, ny, nx], dtype=np.float32)
            for ix in range(nx):
                for iy in range(ny):
                    t=np.where(mask[:, iy, ix]) # retrieve idx to indicate where layer exist vertically at certain location x

                    if len(t[0])>0: # if layer exists at position ix iy
                        i2=np.min(t)
                        i1=np.max(t)
                        dip[:, iy, ix]=np.interp(zg, [zg[i2], zg[i1]], [dip_bot[iy, ix], dip_top[iy, ix]])

            dip[mask!=1]=0

        return azi, dip

    def extract_log_facies_bh(self, facies, bhx, bhy, bhz, depth):

        """
        Extract the log facies in the ArchPy format at the specific location

        #inputs#
        facies: 3D array, a facies realization
        bhx, bhy, bhz: borehole location
        depth: depth of investigation

        #output#
        a log facies (list of facies object with elevations)
        """

        bh_cell=self.coord2cell(bhx, bhy, bhz)
        bh_bot_cell=self.coord2cell(bhx, bhy, bhz-depth)
        if bh_bot_cell is not None:
            bh_bot_cell=bh_bot_cell[0]
        else:
            bh_bot_cell=0

        id_facies=facies[bh_bot_cell: bh_cell[0+1], bh_cell[1], bh_cell[2]][:: -1]
        z_facies=self.zgc[bh_bot_cell: bh_cell[0]+1][:: -1]

        i=0
        log_facies=[]
        fa_prev=0
        for fa in id_facies:

            if fa != fa_prev and fa != 0:
                log_facies.append((self.get_facies_obj(ID =fa, type="ID"), np.round(z_facies[i], 2)))
                fa_prev=fa

            i += 1

        return log_facies

    def get_entropy(self, typ = "units", h_level = None, recompute=False):

        """
        Compute the Shannon entropy for units or facies and return it.

        ## inputs ##
        typ      : str, type of models to use to comppute the entropy,
                   valid values are "units" and "facies"
        h_level  : int, hiearchical level to use to compute the entropy,
                   only used if typ is "units"
        recompute: bool

        ## output ##
        3D nd.array of size (nz, ny, nx)
        """

        if typ == "units":
            string = "units_entropy"
        elif typ== "facies":
            string = "units_facies"

        if hasattr(self.Geol, string) and recompute == False:

            if recompute == "false":
                if typ == "units":
                    return self.Geol.units_entropy
                elif typ == "facies":
                    return self.Geol.facies_entropy
 
        elif recompute or not hasattr(self.Geol, string):

            nz = self.get_nz()
            ny = self.get_ny()
            nx = self.get_nx()

            arr = np.zeros([nz, ny, nx])
            if typ =="units":
                units = self.get_units_domains_realizations(h_level = h_level)
                list_units_ids = np.unique(units[units != 0])

                data = units[:, self.mask]
                SE = np.zeros([data.shape[1]])
                #b=len(self.get_all_units())
                b = len(list_units_ids)
                nreal=units.shape[0]

                for idx in list_units_ids:
                    unit = self.get_unit(ID=idx, type="ID")
                    Pi = (data == unit.ID).sum(0)/nreal
                    pi_mask = (Pi != 0)
                    SE[pi_mask] += Pi[pi_mask] * (np.log(Pi[pi_mask])/np.log(b))
                    arr[self.mask] =- SE

                arr[~self.mask]=np.nan

                self.Geol.units_entropy = arr    
                return arr

            elif typ == "facies":

                facies_domains = self.get_facies().reshape(-1, self.nz, self.ny, self.nx)
                data = units[:, self.mask]
                SE = np.zeros([data.shape[1]])  # shannon entropy
                b = len(self.get_all_facies())
                nreal=facies_domains.shape[0]

                for facies in self.get_all_facies():
                    Pi = (data == facies.ID).sum(0)/nreal
                    pi_mask = (Pi != 0)
                    SE[pi_mask] += Pi[pi_mask] * (np.log(Pi[pi_mask])/np.log(b))
                    arr[self.mask] =- SE

                arr[~self.mask]=np.nan

                self.Geol.facies_entropy = arr

            else:
                print("Choose between units or facies")


    def realizations_aggregation(self, method="basic",
                             depth=100, ignore_units=None,
                             units_to_fill=[],
                             n_iter = 50):
    
        """
        Method to aggregate multiple ArchPy realizations into one for units and facies (TO DO)
        
        # inputs #
        method   : str, method to use to aggregate the real.
                   valid method are: 
                   - basic, return a model with the most probable 
                            units/facies in each cells
                   - probas_prop, return a model constructed sequentially
                                  by ensuring that proportions are respected (at best...)
                   - mean_surfs,  return a model created by meaning the surface elevation
                                  if units were simulated with categorical method,
                                  basic method is used for these units
                            
        depth : 
                   float, probas_prop parameter, maximum depth of investigation 
                   to compute probas and proportions. 
                   Should be around the median depth of the boreholes
                       
        ignore_units: 
                   list, probas_prop parameter, units name to ignore.
                   These units will not be aggregated in the final model.
                   
        units_to_fill:
                   list, mean_surfs parameter, units name to fill with NN at the end
                   Should not be used except to fill the top unit.      
                   
        """

        if ignore_units is None:
            ignore_units = []

        ign_un = []
        for un in ignore_units:
            ign_un.append(un)

        nz = self.get_nz()
        ny = self.get_ny()
        nx = self.get_nx()

        if method == "basic":
            
            units = self.get_units_domains_realizations()

            most_prob = np.zeros((nz, ny, nx), dtype=np.int8)

            for iz in range(self.get_nz()):
                for iy in range(self.get_ny()):
                    for ix in range(self.get_nx()):
                        if self.mask[iz, iy, ix]:
                            occ = np.bincount(units[:, iz, iy, ix])
                            idx = np.where(occ==max(occ))
                            most_prob[iz, iy, ix] = idx[0][0]

            return most_prob
        
        elif method == "probas_prop":

            ## tirer les proportions de chaque units
            d_prop_units = self.get_proportions(depth_min = 0, depth_max = depth, ignore_units = ign_un)

            # create mask
            inter = int(depth/self.get_sz())
            mask_depth = np.zeros([nz, ny, nx], dtype=bool)
            for iy in range(ny):
                for ix in range(nx):
                    a = np.where(self.mask[:, iy, ix])
                    if len(a[0]) > 0:

                        v = int(max(a[0]) - min(a[0]))
                        if v > inter:
                            mask_depth[max(a[0])-inter:max(a[0]), iy, ix] = True
                        else:
                            mask_depth[min(a[0]):max(a[0]), iy, ix] = True

            # compute probas
            d_sorted = {k: v for k, v in sorted(d_prop_units.items(), key=lambda item: item[1])}
            units=self.get_units_domains_realizations(all_data=True)
            tot_cells = mask_depth.sum()

            best_model = np.zeros([nz, ny, nx], dtype=np.int8)

            for k,v in d_sorted.items():
        #         print(k)

                # compute prop of unit
                #d = self.get_proportions(depth_min = 0, depth_max = depth)
                #v = d[k]

                # compute proba unit
                arr=np.zeros([self.nz, self.ny, self.nx])

                # compute probabilities
                arr += (units == self.get_unit(k).ID).sum(0)
                arr /= units.shape[0]
                
                # loop probas
                for iv in np.arange(1, 0, -0.01):
                    prop = (arr[mask_depth] >= iv).sum() / mask_depth.sum()
        #             print(prop, v)
                    if prop > v:
                        break

        #         print(np.round(iv, 3))
                mask_sim = (best_model==0) & (arr >= iv)
                best_model[mask_sim] = mask_sim[mask_sim].astype(int) * self.get_unit(k).ID
                
                #tot_cells -= mask_sim[mask_depth].sum()  # remove attributed cells from total
                # units[:, mask_sim] = 0  # remove simulated units 

                # ign_un.append(k)

            mask = (best_model == 0) & (self.mask)
            best_model[mask] = -99  # non filled units

            ## fill last cells with nearest neighbors 
            res = self.fill_ID(best_model, ID=-99)

            return res

        elif method == "mean_surfs":
            
            best_model = np.zeros([nz, ny, nx], dtype=np.int8)

            def fun(pile, bot=None, mask_unit=None):

                if pile.nature == "surfaces":

                    mean_surfs = []

                    # units, get means surfaces
                    for un_p in pile.list_units:
                        mean_surfs.append(np.nanmean(self.get_surfaces_unit(un_p), 0))

                    for i in range(len(mean_surfs)):
                        un = pile.list_units[i]
                        s = mean_surfs[i]

                        if i == len(mean_surfs)-1:
                            s2 = bot
                        else:
                            s2 = mean_surfs[i+1]

                        a = self.compute_domain(s, s2)
                        best_model[a] = un.ID * self.mask[a]

                elif pile.nature == "3d_categorical":
                    ids = [u.ID for u in pile.list_units]

                    h_lev = pile.list_units[0].get_h_level()

                    units = self.get_units_domains_realizations(h_level=h_lev)
                    units = units[:, mask_unit]

                    l = []
                    # get most probable units by cells
                    for i in range(units.shape[1]):
                        a = np.bincount(units[:, i])
                        idxs= np.where(a == max(a))[0]

                        if len(idxs)> 1:
                            idx = np.random.choice(idxs)
                        elif len(idxs) == 1:
                            idx = idxs[0]
                        else:
                            idx = 0
                            pass

                        l.append(idx)

                    l = np.array(l)

                    best_model[mask_unit] = l  # assign
                    # nearest neighbor ? 

                for i in range(len(pile.list_units)):
                    un_p = pile.list_units[i]
                    if un_p.SubPile is not None:

                        if pile.nature == "surface":
                            bot = mean_surfs[i+1]  # get bottom of un_p
                        else:
                            bot = None
                        mask = (best_model == un_p.ID)

                        fun(un_p.SubPile, bot, mask)

            fun(self.get_pile_master(), self.bot, self.mask)

            ids = [self.get_unit(u).ID for u in units_to_fill]
            for iv in ids + [0]:
                best_model[(best_model==iv) & (self.mask)] = -99
                best_model = self.fill_ID(best_model, -99)

            return best_model
        
        elif method == "MDS_errors":  # a discuter avec Philippe
            
            from sklearn.manifold import MDS
            
            # matrix of distances between simulations
            M = np.zeros([self.nreal_units, self.nreal_units])
            for ireal in range(len(self.Geol.units_domains)):
                for oreal in range(ireal):
                    s1 = self.Geol.units_domains[ireal]
                    s2 = self.Geol.units_domains[oreal]

                    M[ireal, oreal] = np.sum(s1 != s2)
                    M[oreal, ireal] = M[ireal, oreal]
                    
            # mds = MDS(random_state=None)
            # M_transform = mds.fit_transform(M)
            
            # from sklearn.cluster import k_means

            # centr, code, jsp = k_means(M_transform, 1)
            # # plt.scatter(centr[:, 0], centr[:, 1], c="r", s=100, marker="x")
            # # plt.scatter(M_transform[:, 0], M_transform[:, 1], c=code)

            # dist_sim = np.sqrt((M_transform[:, 0] - centr[0][0])**2 + (M_transform[:, 1] - centr[0][1])**2)

            # idx = np.where(dist_sim==min(dist_sim))[0][0]
            
            # loop to find the most representative simulation
            l = []
            for i in range(n_iter):

                mds = MDS(random_state=None)
                M_transform = mds.fit_transform(M)

                from sklearn.cluster import k_means

                centr, code, jsp = k_means(M_transform, 1)
                dist_sim = np.sqrt((M_transform[:, 0] - centr[0][0])**2 + (M_transform[:, 1] - centr[0][1])**2)

                idx = np.where(dist_sim==min(dist_sim))[0][0]
                l.append(idx)

            l = np.array(l)
            res = np.bincount(l)
            idx = np.where(np.bincount(l) == max(np.bincount(l)))[0][0]
            best_model = self.Geol.units_domains[idx]
            
            return best_model

        else:
            print("help")

    ### plotting ###
    def plot_bhs(self, log="strati", plotter=None, v_ex=1, plot_top=False, plot_bot=False):

        """
        Plot the boreholes of the Arch_table project.

        #parameters#
        log   : string, which log to plot --> strati or facies
        plotter: pyvista plotter
        v_ex  : float, vertical exaggeration
        plot_top: bool, if the top of the simulation domain must be plotted
        plot_bot: bool, if the bot of the simulation domain must be plotted
        """

        z0=self.get_oz()

        def lines_from_points(points):
            """Given an array of points, make a line set"""
            poly=pv.PolyData()
            poly.points=points
            cells=np.full((len(points)-1, 3), 2, dtype=np.int_)
            cells[:, 1]=np.arange(0, len(points)-1, dtype=np.int_)
            cells[:, 2]=np.arange(1, len(points), dtype=np.int_)
            poly.lines=cells
            return poly

        if plotter is None:
            p=pv.Plotter()
        else:
            p=plotter

        if log == "strati":
            for bh in self.list_bhs:
                if bh.log_strati is not None:
                    for i in range(len(bh.log_strati)):

                        l=[]
                        st=bh.log_strati[i][0]
                        l.append(bh.log_strati[i][1])
                        if i < len(bh.log_strati)-1:
                            l.append(bh.log_strati[i+1][1])

                        if i == len(bh.log_strati)-1:
                            l.append(bh.z-bh.depth)

                        pts=np.array([np.ones([len(l)])*bh.x, np.ones([len(l)])*bh.y, l]).T
                        line=lines_from_points(pts)
                        line.points[:, -1]=(line.points[:, -1] - z0)*v_ex+z0

                        if st is not None:
                            color=st.c
                            opacity=1
                        else:
                            color="white"
                            opacity=0
                        p.add_mesh(line, color=color, interpolate_before_map=True, render_lines_as_tubes=True, line_width=15, opacity=opacity)

        elif log == "facies":
            for bh in self.list_bhs:
                if bh.log_facies is not None:
                    for i in range(len(bh.log_facies)):

                        l=[]
                        st=bh.log_facies[i][0]
                        l.append(bh.log_facies[i][1])
                        if i < len(bh.log_facies)-1:
                            l.append(bh.log_facies[i+1][1])

                        if i == len(bh.log_facies)-1:
                            l.append(bh.z-bh.depth)

                        pts=np.array([np.ones([len(l)])*bh.x, np.ones([len(l)])*bh.y, l]).T
                        line=lines_from_points(pts)
                        line.points[:, -1]=(line.points[:, -1] - z0)*v_ex+z0
                        if st is not None:
                            color=st.c
                            opacity=1
                        else:
                            color="white"
                            opacity=0
                        p.add_mesh(line, color=color, interpolate_before_map=True, render_lines_as_tubes=True, line_width=15, opacity=opacity)

        if plot_top:
            X, Y=np.meshgrid(self.get_xgc(), self.get_ygc())
            grid=pv.StructuredGrid(X, Y, (self.top-z0)*v_ex+z0)
            p.add_mesh(grid, opacity=0.8, color="white")

        if plot_bot:
            X, Y=np.meshgrid(self.get_xgc(), self.get_ygc())
            grid=pv.StructuredGrid(X, Y, (self.bot-z0)*v_ex+z0)
            p.add_mesh(grid, opacity=0.8, color="red")

        if plotter is None:
            p.add_bounding_box()
            p.show_axes()
            p.show()

    def plot_geol_map(self, plotter=None, v_ex=1, up=0):
    
        nx = self.get_nx()
        ny = self.get_ny()
        
        # plot geol
        X,Y = np.meshgrid(self.xgc, self.ygc)
        grid = pv.StructuredGrid(X, Y, v_ex*(self.top - self.oz)+self.oz + up)
        
        geol_map = self.geol_map
        
        arr = np.ones([ny, nx, 3])*np.nan
        for iy in range(geol_map.shape[0]):
            for ix in range(geol_map.shape[1]):
                unit = self.get_unit(ID=geol_map[iy, ix], type="ID", vb=0)
                if unit is not None:
                    arr[iy, ix] = matplotlib.colors.to_rgb(unit.c)

        grid.point_arrays['my point values'] = arr.reshape((nx*ny, 3),order="F")
        
        if plotter is None:
            p = pv.Plotter()

            p.add_mesh(grid,"red",scalars="my point values",opacity=.5, rgb=True)
            p.add_bounding_box()
            p.show()
        
        else:
            plotter.add_mesh(grid,"red",scalars="my point values",opacity=.5, rgb=True)

    def get_units_domains_realizations(self, iu=None, all_data=True, fill="ID", h_level="all"):

        """
        Return a numpy array of 1 or all units realization(s).
        iu     : int, simulation to return
        all_data: bool, return all the units simulations,
                   in that case, iu is ignored
        fill   : string, ID or color are possible, to return
                   realizations with unit ID or RGBA color
                   (for plotting purpose e.g. with plt.imshow)
        h_level: string or int, hierarchical level to plot.
                   A value of 1 indicates that only unit of the
                   master pile will be plotted. "all" to plot all possible units
        """

        # if self.Geol.units_domains is None:
        #     raise ValueError("Units have not been computed yet")
        if self.write_results:
            if "ud" not in [i.split(".")[-1] for i in os.listdir(self.ws)]:
                raise ValueError("Units have not been computed yet")

        else:
             if self.Geol.units_domains is None:
                raise ValueError("Units have not been computed yet")


        if isinstance(iu, int):
            all_data=False

        nreal=self.nreal_units
        nx=self.get_nx()
        ny=self.get_ny()
        nz=self.get_nz()

        if all_data:
            ud=np.zeros([nreal, nz, ny, nx], dtype=np.int8)

            if self.write_results:
                # get all real
                for ireal in range(nreal):
                    fname=self.name+"_{}.ud".format(ireal)
                    fpath=os.path.join(self.ws, fname)
                    with open(fpath, "rb") as f:
                        ud[ireal]=pickle.load(f)
            else:
                ud=self.Geol.units_domains.copy()

            if fill == "ID":
                units_domains=ud
                if h_level == "all":
                    pass

                elif isinstance(h_level, int) and h_level > 0:
                    lst_ID=np.unique(units_domains)
                    for idx in lst_ID:
                        if idx != 0:
                            s=self.get_unit(ID=idx, type="ID")
                            h_lev=s.get_h_level()  # hierarchical level of unit
                            if h_lev > h_level:  # compare unit level with level to plot
                                for i in range(h_lev - h_level):
                                    s=s.mummy_unit
                                if s is None:
                                    raise ValueError("Error: parent unit return is None, hierarchy relations are inconsistent with Pile and simulations")
                                units_domains[units_domains == idx]=s.ID  # change ID values to Mummy ID
                
            elif fill == "color":
                units_domains=np.zeros([nreal, nz, ny, nx, 4], dtype=np.float32)
                for unit in self.get_all_units():
                    if unit.f_method != "Subpile":
                        mask=(ud == unit.ID)
                        units_domains[mask,: ]=matplotlib.colors.to_rgba(unit.c)

        else:
            if self.write_results:
                fname=self.name+"_{}.ud".format(iu)
                fpath=os.path.join(self.ws, fname)
                with open(fpath, "rb") as f:
                    ud=pickle.load(f)
            else:
                ud=self.Geol.units_domains.copy()[iu]

            if fill == "ID":
                units_domains=ud
                if h_level == "all":
                    pass
                elif isinstance(h_level, int) and h_level > 0:
                    lst_ID=np.unique(units_domains)
                    for idx in lst_ID:
                        if idx != 0:
                            s=self.get_unit(ID=idx, type="ID")
                            h_lev=s.get_h_level()  # hierarchical level of unit
                            if h_lev > h_level:  # compare unit level with level to plot
                                for i in range(h_lev - h_level):
                                    s=s.mummy_unit
                                if s is None:
                                    raise ValueError("Error: parent unit return is None, hierarchy relations are inconsistent with Pile and simulations")
                                units_domains[units_domains == idx]=s.ID  # change ID values to Mummy ID


            elif fill == "color":
                nz, ny, nx=ud.shape
                units_domains=np.zeros([nz, ny, nx, 4])
                for unit in self.get_all_units():
                    if unit.f_method != "Subpile":
                        mask=(ud == unit.ID)
                        units_domains[mask,: ]=matplotlib.colors.to_rgba(unit.c)

        return units_domains


    def plot_units(self, iu=0, v_ex=1, plotter=None, h_level="all", slicex=None, slicey=None, slicez=None,
                   filtering_value=None, scalar_bar_kwargs=None, show_scalar_bar=True):

        """
        Plot units domain for a specific realization iu

        iu    : int, unit index to plot
        v_ex  : float, vertical exageration
        h_level: string or int, hierarchical level to plot. A value
                  of 1 indicates that only unit of the master pile will
                  be plotted. "all" to plot all possible units

        For other parameters see plot_arr function.
        """

        #ensure hierarchy_relations have been set
        self.hierarchy_relations(vb=0)

        colors=[]
        d={}
        nx=self.get_nx()
        ny=self.get_ny()
        nz=self.get_nz()
        sx=self.get_sx()
        sy=self.get_sy()
        sz=self.get_sz()
        x0=self.get_ox()
        y0=self.get_oy()
        z0=self.get_oz()

        stratis_domain=self.get_units_domains_realizations(iu=iu, fill="ID", h_level=h_level).astype(np.float32)
        lst_ID=np.unique(stratis_domain)

        ## change values
        new_id=1
        for i in lst_ID:
            if i != 0:
                s=self.get_unit(ID=i, type="ID")
                stratis_domain[stratis_domain == i]=new_id
                colors.append(s.c)
                d[new_id +0.5]=s.name
                new_id += 1

        #plot
        stratis_domain[stratis_domain==0]=np.nan  # remove where no formations are present
        if plotter is None:
            p=pv.Plotter()
        else:
            p=plotter

        im=geone.img.Img(nx, ny, nz, sx, sy, sz*v_ex, x0, y0, z0, nv=1, val=stratis_domain, varname="Units")

        if slicex is not None:
            slicex=np.array(slicex)
            slicex[slicex<0]=0
            slicex[slicex>1]=1
            cx=im.ox + slicex * im.nx * im.sx  # center along x
        else:
            cx=None
        if slicey is not None:
            slicey=np.array(slicey)
            slicey[slicey<0]=0
            slicey[slicey>1]=1
            cy=im.oy + slicey * im.ny * im.sy  # center along x
        else:
            cy=None
        if slicez is not None:
            slicez=np.array(slicez)
            slicez[slicez<0]=0
            slicez[slicez>1]=1
            cz=im.oz + slicez * im.nz * im.sz  # center along x
        else:
            cz=None

        if slicex is not None or slicey is not None or slicez is not None:
            imgplt3.drawImage3D_slice(im, plotter=p, slice_normal_x=cx, slice_normal_y=cy, slice_normal_z=cz,
                                    custom_scalar_bar_for_equidistant_categories=True,
                                    custom_colors=colors, scalar_bar_annotations=d, filtering_value=filtering_value,
                                    scalar_bar_kwargs=scalar_bar_kwargs, show_scalar_bar = show_scalar_bar)
        else:
            imgplt3.drawImage3D_surface(im, plotter=p, custom_scalar_bar_for_equidistant_categories=True,
                                                custom_colors=colors, scalar_bar_annotations=d, filtering_value=filtering_value,
                                                scalar_bar_kwargs=scalar_bar_kwargs, show_scalar_bar = show_scalar_bar)

        if plotter is None:
            p.add_bounding_box()
            p.show_axes()
            p.show()


    def plot_proba(self, obj, v_ex=1, plotter=None, slicex=None, slicey=None, slicez=None, filtering_interval=[0.01, 1], scalar_bar_kwargs=None):

        """
        Plot the probability of occurence of a specific unit or facies
        (can be passed by a name or the object directly)

        #parameters#
        obj: Unit/facies object or string name of the unit/facies
        filtering_interval: interval of values to plot, values are variable
                             and does not depend on the facies/unit IDs
        for others params: see plot_arr function
        """
        nx=self.get_nx()
        ny=self.get_ny()
        nz=self.get_nz()
        sx=self.get_sx()
        sy=self.get_sy()
        sz=self.get_sz()
        x0=self.get_ox()
        y0=self.get_oy()
        z0=self.get_oz()

        if isinstance(obj, str):
            if self.get_unit(obj, vb=0) is not None:
                u=self.get_unit(obj)
                i=u.ID
                hl=u.get_h_level()
                typ="unit"
            elif self.get_facies_obj(obj, vb=0) is not None:
                fa=self.get_facies_obj(obj)
                i=fa.ID
                typ="facies"
        elif isinstance(obj, Unit):
            i=obj.ID
            hl=obj.get_h_level()
            typ="unit"
        elif isinstance(obj, Facies):
            i=obj.ID
            typ="facies"
        else:
            raise ValueError ("unit/facies must be a string name of a unit or a unit object that is contained inside the Master Pile")

        if typ == "unit":
            arr=np.zeros([self.nz, self.ny, self.nx])
            # compute probabilities
            for iu in range(self.nreal_units):
                units=self.get_units_domains_realizations(iu=iu, h_level=hl, all_data=False)
                arr+=(units == i)
            arr/=self.nreal_units

        elif typ == "facies":
            arr=np.zeros([self.nz, self.ny, self.nx])
            # compute probabilities
            for iu in range(self.nreal_units):
                for ifa in range(self.nreal_fa):
                    facies=self.get_facies(iu=iu, ifa=ifa, all_data=False)
                    arr+=(facies == i)
            arr/=(self.nreal_units*self.nreal_fa)
        im=geone.img.Img(nx, ny, nz, sx, sy, sz*v_ex, x0, y0, z0, nv=1, val=arr, varname="P [-]") #create img object

        #create slices
        if slicex is not None:
            slicex=np.array(slicex)
            slicex[slicex<0]=0
            slicex[slicex>1]=1
            cx=im.ox + slicex * im.nx * im.sx  # center along x
        else:
            cx=None
        if slicey is not None:
            slicey=np.array(slicey)
            slicey[slicey<0]=0
            slicey[slicey>1]=1
            cy=im.oy + slicey * im.ny * im.sy  # center along y
        else:
            cy=None
        if slicez is not None:
            slicez=np.array(slicez)
            slicez[slicez<0]=0
            slicez[slicez>1]=1
            cz=im.oz + slicez * im.nz * im.sz  # center along z
        else:
            cz=None

        if arr.any(): #if values are found
            if plotter is None:
                p=pv.Plotter()
            else:
                p=plotter

            if slicex is not None or slicey is not None or slicez is not None:
                imgplt3.drawImage3D_slice(im, plotter=p, slice_normal_x=cx, slice_normal_y=cy, slice_normal_z=cz, filtering_interval=filtering_interval, scalar_bar_kwargs=scalar_bar_kwargs)
            else:
                imgplt3.drawImage3D_surface(im, plotter=p, filtering_interval=filtering_interval, scalar_bar_kwargs=scalar_bar_kwargs)

            if plotter is None:
                p.add_bounding_box()
                p.show_axes()
                p.show()

        else:
            print("No values found for this unit")


    def plot_facies(self, iu=0, ifa=0, v_ex=1, inside_units=None,
                    plotter=None, slicex=None, slicey=None, slicez=None, filtering_value=None,
                    scalar_bar_kwargs=None, show_scalar_bar=True):

        """
        Plot the facies realizations over the domain with the colors attributed to facies

        #parameters#
        iu: int, indice of units realization
        ifa: int, indice of facies realziation
        v_ex: int or float, vertical exageration
        inside_units: array-like, list of units inside which
                       we want to have the plot.
                       if None --> all
        plotter: pyvista plotter if wanted
        slicex, slicey, slicez: array-like or number(s) between
                                   0 and 1. fraction in x, y or z
                                   direction where a plot of slices
                                   is desired.
                                   slicex=0.5 will mean a slice at
                                   the middle of the x axis
        filtering_value: array-like, values to plot, these values
                          DOES NOT correspond to facies IDs
        """

        fa_domains=self.get_facies(iu, ifa, all_data=False).astype(np.float32)

        #keep facies in only wanted units
        if inside_units is not None:
            mask_all=np.zeros([self.get_nz(), self.get_ny(), self.get_nx()])
            for u in inside_units:
                if isinstance(u, str):
                    mask=self.unit_mask(u, iu= iu)
                elif isinstance(u, Unit) and u in self.get_all_units():
                    mask=self.unit_mask(u.name, iu= iu)
                else:
                    raise ValueError ("Unit passed in inside_units must be a unit name or a unit object contained in the ArchTable")
                mask_all[mask == 1]=1

            fa_domains[mask_all != 1]=0

        d={}
        d_ID={} # dic of values given ID
        colors=[]
        nx=self.get_nx()
        ny=self.get_ny()
        nz=self.get_nz()
        sx=self.get_sx()
        sy=self.get_sy()
        sz=self.get_sz()
        x0=self.get_ox()
        y0=self.get_oy()
        z0=self.get_oz()
        lst_ID=np.unique(fa_domains)

        new_id=1
        for i in lst_ID:
            if i != 0:
                fa=self.get_facies_obj(ID=i, type="ID")
                fa_domains[fa_domains == fa.ID]=new_id
                d_ID[fa.ID]=new_id
                colors.append(fa.c)
                d[new_id + 0.5]=fa.name
                new_id += 1

        #remove 0 occurence (where no facies are present)
        fa_domains[fa_domains==0]=np.nan

        if plotter is None:
            p=pv.Plotter()
        else:
            p=plotter
        im=geone.img.Img(nx, ny, nz, sx, sy, sz*v_ex, x0, y0, z0, nv=1, val=fa_domains, varname="Lithologies")

        if slicex is not None:
            slicex=np.array(slicex)
            slicex[slicex<0]=0
            slicex[slicex>1]=1
            cx=im.ox + slicex * im.nx * im.sx  # center along x
        else:
            cx=None
        if slicey is not None:
            slicey=np.array(slicey)
            slicey[slicey<0]=0
            slicey[slicey>1]=1
            cy=im.oy + slicey * im.ny * im.sy  # center along y
        else:
            cy=None
        if slicez is not None:
            slicez=np.array(slicez)
            slicez[slicez<0]=0
            slicez[slicez>1]=1
            cz=im.oz + slicez * im.nz * im.sz  # center along z
        else:
            cz=None

        #filtering values, change values to have right numbers
        if filtering_value is not None:
            ft_val=[d_ID[i] for i in filtering_value]
        else:
            ft_val=None

        if slicex is not None or slicey is not None or slicez is not None:
            imgplt3.drawImage3D_slice(im, plotter=p, slice_normal_x=cx, slice_normal_y=cy, slice_normal_z=cz,
                                    custom_scalar_bar_for_equidistant_categories=True,
                                    custom_colors=colors, scalar_bar_annotations=d, filtering_value=ft_val,
                                    scalar_bar_kwargs=scalar_bar_kwargs, show_scalar_bar = show_scalar_bar)
        else:
            imgplt3.drawImage3D_surface(im, plotter=p, custom_scalar_bar_for_equidistant_categories=True,
                                    custom_colors=colors, scalar_bar_annotations=d, filtering_value=ft_val,
                                    scalar_bar_kwargs=scalar_bar_kwargs, show_scalar_bar = show_scalar_bar)

        if plotter is None:
            p.add_bounding_box()
            p.show_axes()
            p.show()


    def plot_prop(self, property, iu=0, ifa=0, ip=0, v_ex=1, inside_units=None, inside_facies=None,
                    plotter=None, slicex=None, slicey=None, slicez=None, cmin=None, cmax=None, filtering_interval=None, scalar_bar_kwargs=None):

        """
        Plot the facies realizations over the domain with the colors attributed to facies

        #parameters#
        property: string, property name
        iu          : int, indice of units realization
        ifa          : int, indice of facies realization
        ip           : int, indice of property realization
        v_ex         : int or float, vertical exageration
        inside_units: array-like of Unit objects or unit names
                        (string), list of units inside which we
                        want to have the plot. By default all
        inside_facies: array-like of Facies objects or facies
                        names (string), list of facies inside
                        which we want to have the plot. By default all
        plotter     : pyvista plotter if wanted
        slicex, slicey, slicez: array-like or number(s) between
                                   0 and 1. fraction in x, y or z
                                   direction where a plot of slices
                                   is desired. slicex=0.5 will mean
                                   a slice at the middle of the x axis
        filtering_interval: array-like of 2 values, interval to plot,
                             values out of it will be set to nan

        others params see plot_arr function
        """

        prop=self.getprop(property, iu, ifa, ip, all_data=False)
        facies=self.get_facies(iu, ifa, all_data=False)

        #keep values in only wanted units
        if inside_units is not None:
            mask_all=np.zeros([self.get_nz(), self.get_ny(), self.get_nx()])
            for u in inside_units:
                if isinstance(u, str):
                    mask=self.unit_mask(u, iu=iu)
                elif isinstance(u, Unit) and u in self.get_all_units():
                    mask=self.unit_mask(u.name, iu=iu)
                else:
                    raise ValueError ("Unit passed in inside_units must be a unit name or a unit object contained in the ArchTable")
                mask_all[mask == 1]=1

            prop[mask_all != 1]=np.nan

        #keep values in only wanted facies
        if inside_facies is not None:
            mask_all=np.zeros([self.get_nz(), self.get_ny(), self.get_nx()])
            for fa in inside_facies:
                if isinstance(fa, str):
                    mask=(facies == self.get_facies_obj(fa).ID)
                elif isinstance(fa, Facies) and fa in self.get_all_facies():
                    mask=(facies == fa.ID)
                else:
                    raise ValueError ("Unit passed in inside_units must be a unit name or a unit object contained in the ArchTable")
                mask_all[mask == 1]=1

            prop[mask_all != 1]=np.nan

        if ~prop.any():
            raise ValueError ("Error: No values found")

        nx=self.get_nx()
        ny=self.get_ny()
        nz=self.get_nz()
        sx=self.get_sx()
        sy=self.get_sy()
        sz=self.get_sz()
        x0=self.get_ox()
        y0=self.get_oy()
        z0=self.get_oz()

        if plotter is None:
            p=pv.Plotter()
        else:
            p=plotter
        im=geone.img.Img(nx, ny, nz, sx, sy, sz*v_ex, x0, y0, z0, nv=1, val=prop, varname=property)

        if slicex is not None:
            slicex=np.array(slicex)
            slicex[slicex<0]=0
            slicex[slicex>1]=1
            cx=im.ox + slicex * im.nx * im.sx  # center along x
        else:
            cx=None
        if slicey is not None:
            slicey=np.array(slicey)
            slicey[slicey<0]=0
            slicey[slicey>1]=1
            cy=im.oy + slicey * im.ny * im.sy  # center along y
        else:
            cy=None
        if slicez is not None:
            slicez=np.array(slicez)
            slicez[slicez<0]=0
            slicez[slicez>1]=1
            cz=im.oz + slicez * im.nz * im.sz  # center along z
        else:
            cz=None


        if slicex is not None or slicey is not None or slicez is not None:
            imgplt3.drawImage3D_slice(im, plotter=p, slice_normal_x=cx, slice_normal_y=cy, slice_normal_z=cz, cmin=cmin, cmax=cmax,
                                    filtering_interval=filtering_interval, scalar_bar_kwargs=scalar_bar_kwargs)
        else:
            imgplt3.drawImage3D_surface(im, plotter=p, cmin=cmin, cmax=cmax,
                                    filtering_interval=filtering_interval, scalar_bar_kwargs=scalar_bar_kwargs)

        if plotter is None:
            p.add_bounding_box()
            p.show_axes()
            p.show()



    def plot_mean_prop(self, property, type="arithmetic", v_ex=1, inside_units=None, inside_facies=None,
                    plotter=None, slicex=None, slicey=None, slicez=None, cmin=None, cmax=None, filtering_interval=None, scalar_bar_kwargs=None):

        #TO DO --> to optimize
        """
        Function that plots the arithmetic mean of a property at every cells
        of the simulation domain given all the property simulations

        #inputs#
        property: str, property name
        type: str, to specify the type of function to apply,
               "arithmetic" stands for arithmetic mean,
               "std" for standard deviation,
               "median" is for plotting the median value

        Others arguments --> see plot_arr() function

        #outputs#
        A plot
        """

        #load property array and facies array
        prop=self.getprop(property)  # to modify
        prop_shape=prop.shape
        facies=self.get_facies()
        facies_shape=facies.shape

        #keep values in only wanted units
        if inside_units is not None:
            nreal_units=self.nreal_units  # number of units real
            mask_all=np.zeros(prop_shape)
            for iu in range(nreal_units):
                for u in inside_units:
                    if isinstance(u, str):
                        mask=self.unit_mask(u, iu=iu)
                    elif isinstance(u, Unit) and u in self.get_all_units():
                        mask=self.unit_mask(u.name, iu=iu)
                    else:
                        raise ValueError ("Unit passed in inside_units must be a unit name or a unit object contained in the ArchTable")
                    mask_all[iu,:,:,mask == 1]=1

            prop[mask_all != 1]=np.nan

        #keep values in only wanted facies
        if inside_facies is not None:
            mask_all=np.zeros(prop_shape)
            for iu in range(nreal_units):
                for ifa in range(self.nreal_fa):
                    for fa in inside_facies:
                        if isinstance(fa, str):
                            mask=(facies[iu,ifa] == self.get_facies_obj(fa).ID)
                        elif isinstance(fa, Facies) and fa in self.get_all_facies():
                            mask=(facies[iu,ifa] == fa.ID)
                        else:
                            raise ValueError ("Unit passed in inside_units must be a unit name or a unit object contained in the ArchTable")
                        mask_all[iu,ifa,:,mask == 1]=1

            prop[mask_all != 1]=np.nan

        if ~prop.any():
            raise ValueError ("Error: No values found")

        if type == "arithmetic":
            arr=np.nanmean(prop.reshape(-1,self.get_nz(),self.get_ny(),self.get_nx()),axis=0)

        elif type == "std":
            arr=np.nanstd(prop.reshape(-1,self.get_nz(),self.get_ny(),self.get_nx()),axis=0)

        elif type == "median":
            arr=np.nanmedian(prop.reshape(-1,self.get_nz(),self.get_ny(),self.get_nx()),axis=0)

        self.plot_arr(arr,property,v_ex=v_ex, plotter=plotter, slicex=slicex, slicey=slicey, slicez=slicez,
                      cmin=cmin, cmax=cmax, filtering_interval=filtering_interval, filtering_value=None, scalar_bar_kwargs=scalar_bar_kwargs)



    def plot_arr(self,arr,var_name ="V0",v_ex=1, plotter=None, slicex=None, slicey=None, slicez=None,
                 cmin=None, cmax=None, filtering_interval=None, filtering_value=None, scalar_bar_kwargs=None):

        """
        This function plot a 3D array with the same size of the simulation domain

        #parameters#
        arr      : 3D array to plot. Size of the simulation
                     domain (nx, ny, nz)
        var_name : str, variable names to plot and to show
                     in the pyvista plot
        v_ex     : float, vertical exaggeration
        plotter  : pyvista external plotter
        cmin, cmax: floats, min and max values for colorbar
        filtering_interval: sequence of two values to plot,
                             values outside this range will
                             be discarded
        filtering_value  : sequence of values, values passed
                             will be plotted, others not
                             (for categorical plot for example)
        scalar_bar_kwargs: pyvista scalar bar kwargs
                             (see pyvista colorbar documentation)
        slicex, slicey, slicez: array-like or number(s) between
                                   0 and 1. fraction in x, y or z
                                   direction where a plot of slices
                                   is desired. slicex=0.5 will mean
                                   a slice at the middle of the x axis
"""

        nx=self.get_nx()
        ny=self.get_ny()
        nz=self.get_nz()
        sx=self.get_sx()
        sy=self.get_sy()
        sz=self.get_sz()
        x0=self.get_ox()
        y0=self.get_oy()
        z0=self.get_oz()

        assert arr.shape == (nz, ny, nx), "Invalid shape for array, must be equal to {}".format(nz, ny, nx)

        if plotter is None:
            p=pv.Plotter()
        else:
            p=plotter
        im=geone.img.Img(nx, ny, nz, sx, sy, sz*v_ex, x0, y0, z0, nv=1, val=arr, varname=var_name)

        if slicex is not None:
            slicex=np.array(slicex)
            slicex[slicex<0]=0
            slicex[slicex>1]=1
            cx=im.ox + slicex * im.nx * im.sx  # center along x
        else:
            cx=None
        if slicey is not None:
            slicey=np.array(slicey)
            slicey[slicey<0]=0
            slicey[slicey>1]=1
            cy=im.oy + slicey * im.ny * im.sy  # center along y
        else:
            cy=None
        if slicez is not None:
            slicez=np.array(slicez)
            slicez[slicez<0]=0
            slicez[slicez>1]=1
            cz=im.oz + slicez * im.nz * im.sz  # center along z
        else:
            cz=None


        if slicex is not None or slicey is not None or slicez is not None:
            imgplt3.drawImage3D_slice(im, plotter=p, slice_normal_x=cx, slice_normal_y=cy, slice_normal_z=cz, cmin=cmin, cmax=cmax,
                                    filtering_interval=filtering_interval, filtering_value= filtering_value, scalar_bar_kwargs=scalar_bar_kwargs)
        else:
            imgplt3.drawImage3D_surface(im, plotter=p, cmin=cmin, cmax=cmax,
                                    filtering_interval=filtering_interval, filtering_value= filtering_value, scalar_bar_kwargs=scalar_bar_kwargs)

        if plotter is None:
            p.add_bounding_box()
            p.show_axes()
            p.show()


    #cross sections
    def draw_cross_section(self, background="units", iu=0, ifa=0, **kwargs):

        extent  = [self.get_ox(), self.get_xg()[-1], self.get_oy(), self.get_yg()[-1]]

        if background == "units":  # add option to pass rasters
            back_map = self.compute_geol_map(0, color = True)
            plt.imshow(back_map, origin="lower", extent=extent)

        else:  
            pass
        
        p_list = plt.ginput(n=-1, timeout=0)
        plt.close()

        ## draw cross-section position
        plt.imshow(back_map, origin="lower", extent=extent)
        plt.plot([i[0] for i in p_list], [i[1] for i in p_list], c="red")
        plt.show()

        return p_list


    def cross_section(self, arr_to_plot, p_list, esp=None):

        """
        Return a cross section along the points pass in p_list

        #params#
        arr_to_plot: 3D or 4D array of dimension nz, ny, nx(, 4)
                      of which we want a cross section.
                      This array will be considered being part
                      of the ArchPy simulation domain.
        p_list    : list or array of tuple containing x and
                      y coordinates
                      (e.g. p_list=[(100, 200), (300, 200)]
                      --> draw a cross section between these two points)
        esp       : float, spacing to use when sampling
                      the array along the cross section

        return
        An array x_sec ready to plot and total distance of the cross section
        """

        ox=self.get_ox()
        oy=self.get_oy()
        sx=self.get_sx()
        sy=self.get_sy()

        if esp is None:
            esp=(self.get_sx()**2+self.get_sy()**2)**0.5

        dist_tot=0
        for ip in range(len(p_list)-1): # loop over points
            p1=np.array(p_list[ip])
            p2=np.array(p_list[ip+1])
            d1, d2=p2-p1
            dist=np.sqrt(d1**2 + d2**2)
            lam=esp/dist

            x_d, y_d=p1  # starting point

            f=arr_to_plot.copy()
            no_color=False
            if len(f.shape) == 4:
                no_color=False
                x_sec_i=np.zeros([f.shape[0], int(dist/esp)+1, 4])
                if ip == 0:
                    x_sec=np.zeros([f.shape[0], int(dist/esp)+1, 4])
            elif len(f.shape) == 3:
                no_color=True
                x_sec_i=np.zeros([f.shape[0],int(dist/esp)+1])
                if ip == 0:
                    x_sec=np.zeros([f.shape[0],int(dist/esp)+1])

            if no_color:
                i=0
                for o in np.arange(0,dist,esp):
                    x_d += d1*lam
                    y_d += d2*lam
                    ix=int((x_d - ox)/sx)
                    iy=int((y_d - oy)/sy)
                    fp=f[:,iy,ix]
                    if ip == 0:
                        x_sec[:,i]=fp
                    else:
                        x_sec_i[:,i]=fp
                    i += 1
            else:
                i=0
                for o in np.arange(0,dist,esp):
                    x_d += d1*lam
                    y_d += d2*lam
                    ix=int((x_d - ox)/sx)
                    iy=int((y_d - oy)/sy)
                    fp=f[:,iy,ix]
                    if ip == 0:
                        x_sec[:,i,: ]=fp
                    else:
                        x_sec_i[:,i,: ]=fp
                    i += 1

            dist_tot += dist
            #append xsections
            if ip > 0:
                x_sec=np.concatenate((x_sec, x_sec_i), axis=1)

        return x_sec, dist_tot



    ### BUG TO CORRECT
    """
    Boreholes appear multiple times on the cross-section, have to think about that
    """
    def plot_cross_section(self, p_list, typ="units", arr=None, iu=0, ifa=0, ip=0,
                           property=None, esp=None, ax=None, colorbar=False,
                           ratio_aspect=2, i=0,
                           dist_max = 100, width=.5,
                           vmax=None, vmin=None):

        """
        Plot a cross section along the points given in
        p_list with a spacing defined (esp)

        #inputs#
        p_list : list or array of tuple containing
                   x and y coordinates
                  (e.g. p_list=[(100, 200), (300, 200)]
                  --> draw a cross section between these two points)
        typ    : string, units, facies or prop
                   (if typ is "prop" then a property name
                   should be given in the property argument)
        iu        : int, units index realization
        ifa    : int, facies index realization
        ip          : int, property index realization
        property: str, property name that have been computed
        esp    : float, spacing to use when sampling the
                   array along the cross section
        ax     : matplotlib axes if desired
        ratio_aspect: float, ratio between y and x axis
                       to adjust vertical exaggeration
        """

        def plot_bh(bh, x=None, width=width, typ="units"):

            if typ == "units":
                if bh.log_strati is not None:
                    if x is None:
                        ix = bh.x
                    else:
                        ix = x

                    i = -1
                    for i in range(len(bh.log_strati)-1):
                        s = bh.log_strati[i][1]
                        unit = bh.log_strati[i][0]

                        if i < len(bh.log_strati):
                            s2 = bh.log_strati[i+1][1]

                        if unit is not None:
                            plt.bar(ix, s - s2, bottom=s2, color=unit.c, alpha=1, edgecolor = 'black', width=width)

                    s = bh.log_strati[i+1][1]
                    unit = bh.log_strati[i+1][0]
                    s2 = bh.z - bh.depth
                    if unit is not None:
                        plt.bar(ix, s - s2, bottom=s2, color=unit.c, alpha=1, edgecolor = 'black', width=width)

        if typ == "units":
            arr=self.get_units_domains_realizations(iu=iu, fill="color", all_data=False)

        elif typ == 'proba_units':
            units=self.get_units_domains_realizations()
            nreal=units.shape[0]
            arr=(units == i).sum(0)/nreal
            del(units)

        elif typ == 'proba_facies':
            facies=self.get_facies()
            nreal=facies.shape[0] * facies.shape[1]
            arr=(facies == i).sum(0).sum(0) /nreal
            del(facies)
            
        elif typ =="facies":
            arr=self.get_facies(iu, ifa, all_data=False)
            #change values to have colors directly
            new_arr=np.zeros([arr.shape[0], arr.shape[1], arr.shape[2], 4])
            list_fa=np.unique(arr)
            for IDfa in list_fa:
                if IDfa != 0:
                    fa=self.get_facies_obj(ID=IDfa, type="ID")  # get facies object
                    mask=(arr == IDfa)
                    new_arr[mask,: ]=colors.to_rgba(fa.c)
            arr=new_arr

        elif typ =="prop":
            assert isinstance(property, str), "property should be given in a property name --> string"
            arr=self.getprop(property, iu, ifa, ip, all_data=False)

        elif typ =="entropy_units":
            units=self.get_units_domains_realizations()
            SE=np.zeros(self.mask.shape)  # shannon entropy
            b=len(self.get_all_units())
            nreal=units.shape[0]
            for unit in self.get_all_units():
                print(unit)
                Pi=(units==unit.ID).sum(0)/nreal
                pi_mask=(self.mask) & (Pi!=0)
                SE[pi_mask] += Pi[pi_mask]*(np.log(Pi[pi_mask])/np.log(b))
                arr=-SE
            arr[~self.mask]=np.nan
            del(units)

        elif typ == "entropy_facies":
            facies_domains=self.get_facies().reshape(-1, self.nz, self.ny, self.nx)
            SE=np.zeros(self.mask.shape)  # shannon entropy
            b=len(self.get_all_facies())
            nreal=facies_domains.shape[0]
            for facies in self.get_all_facies():
                Pi=(facies_domains==facies.ID).sum(0)/nreal
                pi_mask=(self.mask) & (Pi!=0)
                SE[pi_mask] += Pi[pi_mask]*(np.log(Pi[pi_mask])/np.log(b))
                arr=-SE
            arr[~self.mask]=np.nan
            del(facies_domains)

        elif typ =="arr":
            pass
        else:
            assert 'Typ unknown'
            return

        #extract cross section
        xsec, dist=self.cross_section(arr, p_list, esp=esp)

        if ax is None:
            fig, ax=plt.subplots(figsize=(10, 10))

        extent=[0, dist, self.get_oz(), self.get_zg()[-1]]
        a=ax.imshow(xsec, origin="lower", extent=extent, interpolation="none", vmax=vmax, vmin=vmin)
        if colorbar:
            plt.colorbar(a, ax=ax, orientation='horizontal')
        ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/ratio_aspect)

        del(arr)

        #get boreholes
        dist_tot = 0
        for ip in range(len(p_list)-1):

            p1 = p_list[ip]
            p2 = p_list[ip+1]
            xmin = min(p1[0], p2[0])
            xmax = max(p1[0], p2[0])
            Lx = xmax - xmin
            xmin -= max(0.2*Lx, dist_max/2)
            xmax += max(0.2*Lx, dist_max/2)

            ymin = min(p1[1], p2[1])
            ymax = max(p1[1], p2[1])
            Ly = ymax - ymin
            ymin -= max(0.2*Ly, dist_max/2)
            ymax += max(0.2*Ly, dist_max/2)

            # select bh inside p1 and p2
            sel_bhs = [bh for bh in self.list_bhs if bh.x > xmin and bh.x < xmax and bh.y > ymin and bh.y < ymax]

            # compute dist of sel_bhs to line

            # proj bhs
            for bh in sel_bhs:
                a = p2[0] - p1[0]
                b = p2[1] - p1[1]
                x_proj = (a**2 * bh.x + b**2 * p2[0] + a * b * (bh.y - p2[1])) / (a**2 + b**2)
                y_proj = (b * x_proj - b * p2[0] + a * p2[1]) / a

                dist_line = ((bh.x - x_proj) ** 2 + (bh.y - y_proj) ** 2)**0.5

                if dist_line < dist_max:
                    dist = ((x_proj - p1[0]) ** 2 + (y_proj - p1[1]) ** 2)**0.5
                    dist_to_plot = dist_tot + dist  # distance where to plot the bh

                    # plot bh
                    if typ == "units":
                        plot_bh(bh, dist_to_plot)

            # increment total distance
            dist_points = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)**0.5
            dist_tot += dist_points

        


    def plot_lines(self, list_lines, names=None, ax=None, legend=True):

        """
        Plot cross-sections lines given in  the
        "list_lines" argument on a 2D top view of the domain

        list_lines: list or array-like of point lists as
                     defined in plot_cross_section
        names: array-like of string to pass as label
                to the plot function
        """

        if ax is None:
            fig, ax=plt.subplots()

        ox=self.ox
        oy=self.oy
        x1=self.xg[-1]
        y1=self.yg[-1]
        ax.plot((ox, ox, x1, x1, ox),(oy, y1, y1, oy, oy), c="k") #plot domain
        ax.set_aspect("equal")

        i=1
        for line in list_lines:
            if names is None:
                label="cross " + str(i)
            else:
                label=names[i-1]
            line=np.array(line)
            ax.plot(line[:,0], line[:,1], label=label)
            i += 1

        if legend:
            ax.legend()


# CLASSES PILE + UNIT + SURFACE

class Pile():

    """
    This class is a major object the Stratigraphic Pile (SP).
    It contains all the units objects
    and allow to know the stratigraphical relations between the units.
    One Pile object must be defined for each subpile + 1 "master" pile.

    ### init params ###
    name    : str, name of the pile
    nature  : str, units type of interpolation, can be "surfaces" or "3d_categorical"
              if "surfaces" is chosen (default), 2D surfaces interpolations of 
              the surfaces are performed. The surfaces are then used to delimit unit domains.
              if "3d_categorical", a facies method is used to simulate position of the units.
              The available methods are the same to simulate the facies.
    """

    def __init__(self, name, nature="surfaces", verbose=1, seed=1):

        assert isinstance(name, str), "A name should be provided and must be a string"
        self.list_units=[]
        self.name=name
        self.verbose=verbose
        self.seed=seed
        self.nature = nature


    def __repr__(self):
        return self.name

    def add_unit(self, unit):

        try: #iterable
            for i in unit:
                if (isinstance(i, Unit)) and (i not in self.list_units):
                    i.verbose = self.verbose
                    self.list_units.append(i)
                    if self.verbose:
                        print("Stratigraphic unit {} added".format(i.name))
                elif (isinstance(i, Unit)):
                    if self.verbose:
                        print("object is already in the list")
                else:
                    if self.verbose:
                        print("object isn't a unit object")
        except: # unit not in a list
            if (isinstance(unit, Unit)) and (unit not in self.list_units):
                self.list_units.append(unit)
                unit.verbose = self.verbose
                if self.verbose:
                    print("Stratigraphic unit {} added".format(unit.name))
            elif (isinstance(unit, Unit)):
                if self.verbose:
                    print("object is already in the list")
                else:
                    if self.verbose:
                        print("object isn't a unit object")

    def remove_unit(self, unit_to_rem):

        """Remove the given unit object from Pile object"""

        if len(self.list_units) > 0:
            if unit_to_rem in self.list_units:
                self.list_units.remove(unit_to_rem)
                if self.verbose:
                    print("Unit {} removed from Pile {}".format(unit_to_rem.name, self.name))

        else:
            if self.verbose:
                print("no units found in pile {}".format(self.name))

    def order_units(self, vb=1):

        """
        Order list_liths according the order attributes of each lithologies
        """

        if vb:
            print("Pile {}: ordering units".format(self.name))

        self.list_units.sort(key=lambda x: x.order)
        if vb:
            print("Stratigraphic units have been sorted according to order")

        i=0
        flag=0
        # check orders of unit
        for s in self.list_units:
            if i == 0:
                s_bef=s
                if s_bef.order != 1:
                    flag=1
                    if vb:
                        print("first unit order is not equal to 1")

            else:
                if s.order == s_bef.order:
                    flag=1
                    if vb:
                        print("units {} and {} have the same order".format(s.name, s_bef.name))

                elif s.order != s_bef.order + 1:
                    flag=1
                    if vb:
                        print("Discrepency in the orders for units {} and {}".format(s.name, s_bef.name))

                s_bef=s
            i += 1

        if flag:
            if vb:
                print("Changing orders for that they range from 1 to n")
            # check order ranging from 1 to n
            for i in range(len(self.list_units)):
                if self.list_units[i].order != i+1:
                    self.list_units[i].order=i+1

    def compute_surf(self, ArchTable, nreal=1, fl_top=False, subpile=False, tops=None, bots=None, vb=1):

        """
        Compute the elevation of the surfaces units (1st hierarchic order)
        contained in the Pile object.

        #inputs#
        nreal    : int, number of realization
        fl_top   : bool, to not interpolate first layer and assign it=top
        subpile  : pile object, if the pile is a subpile
        tops, bots: sequence of arrays, of top/bot for subpile surface
        vb       : bool, verbose 1 for all and 0 for nothing
        """

        def add_sto_contact(s, x, y, z, type="equality", z2=None): #function to add a stochastic hd point to a surface
            # warning no check that point is inside domain
            if type == "equality":
                s.surface.sto_x.append(x)
                s.surface.sto_y.append(y)
                s.surface.sto_z.append(z)
            elif type == "ineq_inf":
                s.surface.sto_ineq.append([x, y, 0, z, np.nan])
            elif type == "ineq_sup":
                s.surface.sto_ineq.append([x, y, 0, np.nan, z])
            elif type == "double_ineq":
                s.surface.sto_ineq.append([x, y, 0, z, z2])  # inferior and upper ineq


        if ArchTable.check_piles_name() == 0: # check consistency in unit
            return None

        ArchTable.nreal_units=nreal  # store number of realizations

        #simulation grid
        xg=ArchTable.get_xgc()
        nx=xg.shape[0]
        yg=ArchTable.get_ygc()
        ny=yg.shape[0]
        zg=ArchTable.get_zg()
        nz=zg.shape[0] - 1
        if ~subpile:
            top=ArchTable.top
            bot=ArchTable.bot
        mask=ArchTable.mask  # simulation mask
        mask2D = ArchTable.mask2d  # mask 2D
        nlay=len(self.list_units)

        if vb:
            print("########## PILE {} ##########".format(self.name))
        #make sure to have ordered lithologies
        self.order_units(vb=vb)

        ### initialize surfaces by setting to 0

        surfs=np.zeros([nreal, nlay, ny, nx], dtype=np.float32)
        surfs_bot=np.zeros([nreal, nlay, ny, nx], dtype=np.float32)
        org_surfs=np.zeros([nreal, nlay, ny, nx], dtype=np.float32)
        real_domains=np.zeros([nreal, nlay, nz, ny, nx], dtype=np.int8)

        for ireal in range(nreal): # loop over real

            # erase stochastic data
            for s in self.list_units:
                s.surface.sto_x = []
                s.surface.sto_y = []
                s.surface.sto_z = []
                s.surface.sto_ineq = []

            # compute sto data
            for g in ArchTable.sto_hd:
                coord = g[0]
                x,y,z = coord
                hd = g[1]
                p = g[2]
                i = np.random.choice(range(len(p)), p=p)
                for s, typ in np.array(hd[i]):
                    add_sto_contact(s, x, y, z, typ)

            #top/bot
            if subpile:
                top=tops[ireal]
                bot=bots[ireal]

            # counter for current simulated surface
            i=-1

            for litho in self.list_units[:: -1]:
                if vb:
                    print("\n#### COMPUTING SURFACE OF UNIT {}".format(litho.name))
                i += 1  # index for simulated surface
                start=time.time()
                if (fl_top) and (i == nlay-1): # first layer assign to top
                    s1=top
                elif (litho.surface.int_method in ["kriging", "linear", "cubic", "nearest"]) and (ireal > 0): # determinist method
                    s1=org_surfs[0, (nlay-1)-i].copy()
                    if vb:
                        print("{}: determinist interpolation method, reuse the first surface".format(litho.name))
                else:

                    # change mean if thickness mode is activated
                    if "thickness" in litho.surface.dic_surf.keys():
                        if litho.surface.dic_surf["thickness"] is not None:
                            if i == 0:
                                litho.surface.dic_surf["mean"] = ArchTable.bot + litho.surface.dic_surf["thickness"]
                            else:
                                litho.surface.dic_surf["mean"] = s1 + litho.surface.dic_surf["thickness"]

                    s1=interp2D(litho.surface, ArchTable.get_xg(), ArchTable.get_yg(), ArchTable.xu2D,
                                 seed=ArchTable.seed + litho.ID * 1e3 + ireal, verbose=ArchTable.verbose, ncpu=ArchTable.ncpu, mask2D=mask2D,
                                 **litho.surface.dic_surf) # simulation

                    # remove mean
                    if "thickness" in litho.surface.dic_surf.keys():
                        if litho.surface.dic_surf["thickness"] is not None:
                            litho.surface.dic_surf["mean"] = None

                end=time.time()
                if vb:
                    print("{}: time elapsed for computing surface {} s".format(litho.name, (end - start)))

                ## if nan inside the domain (non simulated area) --> set values from surface below
                mask_nan = np.zeros([ny, nx], dtype=bool)
                mask_nan[mask2D] = np.isnan(s1[mask2D])
                if mask_nan.any():
                    if i > 0:
                        s1[mask_nan] = org_surfs[ireal, (nlay-1)-i+1][mask_nan]
                    else:
                        s1[mask_nan] = bot[mask_nan]

                org_surfs[ireal, (nlay-1)-i]=s1.copy()

                # adapt altitude if above/below top/bot
                s1[s1>top]=top[s1>top]
                s1[s1<bot]=bot[s1<bot]

                #strati consistency and erosion rules
                if i > 0:
                    for o in range(i): # index for checking already simulated surfaces
                        litho2=self.list_units[:: -1][o]
                        s2=surfs[ireal, (nlay-1)-o] #idx from nlay-1 to nlay-i-1
                        if litho != litho2:
                            if (litho.order < litho2.order) & (litho.surface.contact == "onlap"): # si couche simulée sup et onlap
                                s1[s1 < s2]=s2[s1 < s2]
                            elif (litho.order < litho2.order) & (litho.surface.contact == "erode"): # si couche simulée érode une ancienne
                                s2[s2 > s1]=s1[s2 > s1]
                                if o == i-1 and litho.contact == "erode": # if index "o" is underlying layer
                                    s1[s1 > s2]=s2[s1 > s2] # prevent erosion layers having volume

                surfs[ireal, (nlay-1)-i]=s1  #add surface

            #compute domains
            start=time.time()
            surfs_ir=surfs[ireal]

            for i in range(surfs_ir.shape[0]):
                if i < surfs_ir.shape[0]-1:
                    s_bot=surfs_ir[i+1] # bottom
                else:
                    s_bot=bot
                s_top=surfs_ir[i] # top
                a=ArchTable.compute_domain(s_top, s_bot)
                real_domains[ireal, i]=a*mask*self.list_units[i].ID
                surfs_bot[ireal, i]=s_bot

            end=time.time()
            if vb:
                print("\nTime elapsed for getting domains {} s".format((end - start)))

        #only 1 big array
        a=np.sum(real_domains, axis=1)
        ArchTable.Geol.units_domains[a != 0]=a[a != 0]
        ArchTable.Geol.surfaces_by_piles[self.name]=surfs
        ArchTable.Geol.surfaces_bot_by_piles[self.name]=surfs_bot
        ArchTable.Geol.org_surfaces_by_piles[self.name]=org_surfs

        if vb:
            print("##########################\n")


    def define_domains(self, ArchTable, surfaces, tops=None, bots=None, subpile=False, vb=0, fl_top=True):

        assert surfaces.shape[1] == len(self.list_units), "the number of surfaces {} provided is not equal to the number of units {}".format(surfaces.shape[1], len(self.list_units))

        # nreal
        nreal=surfaces.shape[0]

        ArchTable.nreal_units=nreal  # store number of realizations

        #simulation grid
        xg=ArchTable.get_xgc()
        nx=xg.shape[0]
        yg=ArchTable.get_ygc()
        ny=yg.shape[0]
        zg=ArchTable.get_zg()
        nz=zg.shape[0] - 1
        if ~subpile:
            top=ArchTable.top
            bot=ArchTable.bot
        mask=ArchTable.mask
        nlay=len(self.list_units)

        #make sure to have ordered lithologies
        self.order_units(vb=vb)

        ### initialize surfaces by setting to 0

        surfs=np.zeros([nreal, nlay, ny, nx])
        surfs_bot=np.zeros([nreal, nlay, ny, nx])
        org_surfs=surfaces
        real_domains=np.zeros([nreal, nlay, nz, ny, nx])

        for ireal in range(nreal): # loop over real

            #top/bot
            if subpile:
                top=tops[ireal]
                bot=bots[ireal]

            # counter for current simulated surface
            i=-1

            for litho in self.list_units[:: -1]:

                i += 1  # index for simulated surface
                start=time.time()
                if (fl_top) and (i == nlay-1): # first layer assign to top
                    s1=top
                else:
                    s1=org_surfs[ireal, (nlay-1)-i].copy()

                # adapt altitude if above/below top/bot
                s1[s1>top]=top[s1>top]
                s1[s1<bot]=bot[s1<bot]

                # strati consistency and erosion rules
                if i > 0:
                    for o in range(i): # index for checking already simulated surfaces
                        litho2=self.list_units[:: -1][o]
                        s2=surfs[ireal, (nlay-1)-o] #idx from nlay-1 to nlay-i-1
                        if litho != litho2:
                            if (litho.order < litho2.order) & (litho.surface.contact == "onlap"): # si couche simulée sup et onlap
                                s1[s1 < s2]=s2[s1 < s2]
                            elif (litho.order < litho2.order) & (litho.surface.contact == "erode"): # si couche simulée érode une ancienne
                                s2[s2 > s1]=s1[s2 > s1]
                                if o == i-1 and litho.contact == "erode": # if index "o" is underlying layer
                                    s1[s1 > s2]=s2[s1 > s2] # prevent erosion layers having volume

                surfs[ireal, (nlay-1)-i]=s1  #add surface

            #compute domains

            surfs_ir=surfs[ireal]

            for i in range(surfs_ir.shape[0]):
                if i < surfs_ir.shape[0]-1:
                    s_bot=surfs_ir[i+1] # bottom
                else:
                    s_bot=bot
                s_top=surfs_ir[i] # top
                a=ArchTable.compute_domain(s_top, s_bot)
                real_domains[ireal, i]=a*mask*self.list_units[i].ID
                surfs_bot[ireal, i]=s_bot


        #only 1 big array
        a=np.sum(real_domains, axis=1)
        ArchTable.Geol.units_domains[a != 0]=a[a != 0]
        del(real_domains)
        ArchTable.Geol.surfaces_by_piles[self.name]=surfs
        ArchTable.Geol.surfaces_bot_by_piles[self.name]=surfs_bot
        ArchTable.Geol.org_surfaces_by_piles[self.name]=org_surfs

        if vb:
            print("##########################\n")


class Unit():

    def __init__(self, name, order, color, surface=None, ID=None,
                dic_facies={"f_method": "homogenous", "f_covmodel": None, "TI": None, "SubPile": None, "Flag": None, "G_cm": None},
                contact="onlap", verbose=1):

        """
        ###Description###
        name     : string, name of the unit
        order    : int, order of the unit in the pile (1 (top) to n (bottom),
                     where n is the total number of units)
        color    : string, color to use for plotting and representation
        contact  : string, onlap or erode, if the unit is defined as erode
                     then the unit will only act as an erosion surface and will
                     fill nothing
        ineq_data: inequality data of the following format: [(x1, y1, z1, lb, ub)]
                     lb, ub: lower/upper boundary of the inequality, np.nan if none
        dic_facies: parameters for facies filling
            f_method : string, valable method are: homogenous, SubPile, SIS,
                         MPS and TPGs
            f_covmodel: 3D geone covmodel or list of covmodels
                         (see geone documentation) for facies interpolation
                         with SIS
            TI       : geone image, Training image for MPS simulation
                         of the filling the units
            SubPile  : Pile object, is used to fill the unit
                         if the f_method is "SubPile"
            **kwargs
              ||
              ¦¦
              \/
        Facies keyword arguments to pass to the facies methods (these should be pass through dic_facies !):
            - SIS:
                - neig : number of neighbours
                - r    : relative radius of research (default is 1)
    
            - MPS:
                - TI                    : geone img, Training image(s) to use
                - mps "classic" parameters (maxscan, thresh, neig (number of neighbours))
                - npost                 : number of path postprocessing, default 1
                - radiusMode            : radius mode to use (check deesse manual)
                - anisotropyRatioMode   : Anisotropy ratio to use in reasearch of neighbours (check deesse manual)
                - rx, ry, rz            : radius in x, y and z direction
                - angle1, angle2, angle3: ellipsoid of research rotation angles
                - ax, ay, az            : anisotropy ratio for research ellipsoid
                - rot_usage             : 0, 1 or 2 (check deesse manual)
                - rotAziLoc, rotDipLoc, : local rotation, True or False ?
                  rotPlungeLoc
                - rotAzi, rotDip,       : global rotation angles: values, min-max, maps, see deesse_input doc
                  rotPlunge                
                - xr, yr, zr            : ratio for geom transformation
                - xloc, yloc, zloc      : local or not transformation
                - homo_usage            : homothety usage
                - probaUsage            : probability constraint usage, 0 for no proba constraint,
                                          1 for global proportion defined in globalPdf,
                                          2 for local proportion defined in localPdf
                - globalPdf             : array-like of float of length equal to the number of class,
                                          proportion for each class
                - localPdf              : (nclass, nz, ny, nx) array of floats probability for each class,
                                          localPdf[i] is the "map defined on the simulation grid
                - localPdfRadius        : support radius for local pdf, default is 2
                - deactivationDistance  : float, distance at which localPdf are deactivated (see Deesse doc)
                - constantThreshold     : float, threshold value for pdf's comparison
                - dataImage             : geone img used as data, see deesse/geone documentation
                - outputVarFlag         : bool or list of bool of size nv, to output or not the variables
                - distanceType          : string or list of strings, "categorical" or "continuous"

            - TPGs:
                - flag: dictionary of limits of cuboids domains in Gaussian space, check ... for the right format.
                - Ik_cm: indicator covmodels
                - G_cm: list of Gaussian covmodels for the two gaussian fields, can be infered from Ik_cm
                - various parameters: (du, dv --> precision of integrals for G_cm inference),
                                      (dim: dimension wanted for G_cm inference,
                                      (c_reg: regularisation term for G_cm inference),
                                      (n: number of control points for G_cm inference))
        """

        assert ID != 0, "ID cannot be 0"
        assert name is not None, "A name must be provided"
        assert isinstance( surface, Surface), "A surface object must be provided for each unit"


        self.name=name
        self.order=order
        self.contact=contact
        self.c=color

        if ID is None:
            self.ID=order
        else:
            self.ID=ID
        self.SubPile=None
        self.verbose=verbose
        self.list_facies=[]
        self.bb_units = []
        self.x = []  # data coordinates for categorical simulations
        self.y = []  # data coordinates for categorical simulations
        self.z = []  # data coordinates for categorical simulations
        self.mummy_unit=None
        self.set_dic_facies(dic_facies) #set dic facies to unit

        if surface is not None:
            self.set_Surface(surface)


    def set_dic_facies(self, dic_facies):

        """
        Set a dictionary facies to Unit object
        """

        assert dic_facies["f_method"] in ("MPS", "SIS", "homogenous", "TPGs", "SubPile"), 'Filling method unknown, valids are "MPS", "SIS", "homogenous", "TPGs", "SubPile'

        self.f_method=dic_facies["f_method"]
        # check important input for facies methods
        #MPS
        if self.f_method == "MPS":
            if "TI" not in dic_facies.keys():
                if self.verbose:
                    print("WARNING NO TI PASSED FOR MPS SIMULATION")
            else:
                self.set_f_TI(dic_facies["TI"])

        #Subpile
        elif self.f_method == "SubPile":
            if "SubPile" not in dic_facies.keys():
                if self.verbose:
                    print("No SubPile passed for SubPile filling, consider adding one with set_SubPile() before proceeding")
            else:
                self.set_SubPile(dic_facies["SubPile"])

            if "units_fill_method" in dic_facies.keys():
                if dic_facies["units_fill_method"] == "SIS":
                    self.set_f_covmodels(dic_facies["f_covmodel"])

                elif dic_facies["units_fill_method"] == "MPS":
                    if "TI" not in dic_facies.keys():
                        if self.verbose:
                            print("WARNING NO TI PASSED FOR MPS SIMULATION")
                    else:
                        self.set_f_TI(dic_facies["TI"])

                elif dic_facies["units_fill_method"] == "TPGs":
                    if "Flag" in dic_facies.keys():
                        self.flag = dic_facies["Flag"]
                    if "G_cm" in dic_facies.keys():
                        self.G_cm = dic_facies["G_cm"]

            # TO COMPLETE

        #Truncated Plurigaussians CREATE CHECK FUNCTIONS TO DO
        elif self.f_method == "TPGs":
            #assert
            self.flag=dic_facies["Flag"]
            self.G_cm=dic_facies["G_cm"]

        #SIS
        elif self.f_method == "SIS":
            if "f_covmodel" not in dic_facies.keys():
                if self.verbose:
                    print("Unit {}: WARNING NO COVMODELS PASSED FOR SIS SIMULATION".format(self.name))
            else:
                self.set_f_covmodels(dic_facies["f_covmodel"])

        self.dic_facies=dic_facies


    ## magic fun
    def __repr__(self):
        s=self.name
        return s

    def __call__(self):
        return print("Unit {}".format(self.name))

    def __eq__(self, other):
        """
        for comparison
        """

        if other is not None:
            if (self.name == other.name) & (self.order == other.order):
                return True
            else:
                return False
        else:
            return False

    def __it__(self, other): # inequality comparison
        if (self.order < other.order):
            return True
        else:
            return False

    def __gt__(self, other): # inequality comparison
        if (self.order > other.order):
            return True
        else:
            return False

    def __str__(self):
        return self.name

    #copy fun
    def copy(self):
        return copy.deepcopy(self)

    def set_SubPile(self, SubPile):

        """
        Change or define a subpile for filling
        """
        if isinstance(SubPile, Pile):
            if self.f_method == "SubPile":
                self.SubPile=SubPile
            else:
                if self.verbose:
                    print("Really ? Filling method is not SubPile")
        else:
            if self.verbose:
                print("The SubPile object is not an Arch_table object")

    def set_f_TI(self, TI):

        """
        Change training image for a strati unit

        #input#
        TI: Training image MUST be a geone img
        """

        if isinstance(TI, geone.img.Img):
            self.f_TI=TI
            if self.verbose:
                print("Unit {}: TI added".format(self.name))
        else:
            if self.verbose:
                print("TI NOT A GEONE IMAGE")

    def set_Surface(self, surface):

        """
        Change or add a top surface for Unit object.
        This will define how the top of the formation will be modelled.

        #input#
        surface: a ArchPy surface object
        """

        if isinstance(surface, Surface):
            self.surface=surface
            if self.verbose:
                print("Unit {}: Surface added for interpolation".format(self.name))
        else:
            raise ValueError("Surface object must be an object of the class Surface from ArchPy")

    def set_f_covmodels(self, f_covmodel):

        """
        Remove existing facies covmodels and one or more
        depending of what is passed (array like or only one object)

        #input#
        f_covmodel: geone.covModel object that will be used
                     for the interpolation of the facies if method is SIS,
                     can be a list or only one object
        """

        self.list_f_covmodel=[]
        covmodels_class=(geone.covModel.CovModel3D)
        try:
            for covmodel in f_covmodel:
                if isinstance(covmodel, covmodels_class):
                    f=1
                else:
                    f=0
                    break
            if f:
                self.list_f_covmodel=f_covmodel

            else:
                if self.verbose:
                    print("at least one covmodel is not a 3D covmodel geone, nothing has changed")

        except: # only one object passed
            if isinstance(f_covmodel, covmodels_class):
                self.list_f_covmodel.append(f_covmodel)
                if self.verbose:
                    print("Unit {}: covmodel for SIS added".format(self.name))
            else:
                if self.verbose:
                    print("Unit {}: covmodel not a geone 3D covmodel".format(self.name))

    def get_h_level(self):

        """
        Return hierarchical level of the unit
        """

        def func(unit, h_lev):
            if unit.mummy_unit is not None:
                h_lev += 1
                h_lev=func(unit.mummy_unit, h_lev)

            return h_lev
        h_lev=1
        h_lev=func(self, h_lev)
        return h_lev

    def goes_up_until(self, unit_target):
        
        """
        Climb the hierarchical tree of self unit until finding a specific unit and return it
        """
        
        unit = self
        for i in range(self.get_h_level()):
            
            if unit != unit_target:
                unit = unit.mummy_unit
                if unit is None:
                    return None
            else:
                return unit
            
        return unit

    def get_big_mummy_unit(self):

        """
        Return lowest unit (main unit) in hierarchical order
        """

        unit = self
        for i in range(self.get_h_level()):

            if unit.mummy_unit is not None:
                unit = unit.mummy_unit
            
        return unit


    def get_baby_units(self, recompute=False, vb=1):


        """
        
        """

        
        if self.SubPile is None:
            if vb:
                print("Unit has no hierarchy")
            return []

        if recompute:
            self.bb_units = []

            def fun(unit):

                if unit.SubPile is not None:
                    l=unit.SubPile.list_units
                    self.bb_units += l

                    for unit in l:
                        fun(unit)

            fun(self)
        return self.bb_units

    def add_facies(self, facies):

        """
        Add facies object to strati
        facies: facies object
        """

        if type(facies) == list:
            for i in facies:
                if (isinstance(i, Facies)) and (i not in self.list_facies): # check Facies object belong to Facies class
                    self.list_facies.append(i)
                    if self.verbose:
                        print("Facies {} added to unit {}".format(i.name, self.name))
                else:
                    if self.verbose:
                        print("object isn't a Facies object or Facies object has already been added")
        else: # facies not in a list
            if (isinstance(facies, Facies))  and (facies not in self.list_facies):
                self.list_facies.append(facies)
                if self.verbose:
                    print("Facies {} added to unit {}".format(facies.name, self.name))
            else:
                if self.verbose:
                    print("object isn't a Facies object or Facies object has already been added")

        return

    def rem_facies(self, facies=None, all_facies=True):

        """
        To remove facies from unit, by default all facies are removed
        """

        if all_facies:
            self.list_facies=[]
        else:
            self.list_facies.remove(facies)

    def compute_facies(self, ArchTable, nreal=1, mode="facies", verbose=0):

        """
        Compute facies domain for the specific unit

        #inputs#
        ArchTable: ArchTable object containing units, surface,
                    facies and at least a Pile (see example on the github)
        nreal   : int, number of realization (per unit realizations) to make
        verbose : 0 or 1.
        """

        xg=ArchTable.get_xg()
        yg=ArchTable.get_yg()
        zg=ArchTable.get_zg()

        seed=ArchTable.seed
        np.random.seed(seed)

        ## grid parameters
        nx=len(xg)-1
        ny=len(yg)-1
        nz=len(zg)-1
        dimensions=(nx, ny, nz)
        ox=np.min(xg[0])
        oy=np.min(yg[0])
        oz=np.min(zg[0])
        origin=(ox, oy, oz)
        sx=np.diff(xg)[0]
        sy=np.diff(yg)[0]
        sz=np.diff(zg)[0]
        spacing=(sx, sy, sz)

        nreal_units = ArchTable.nreal_units

        facies_domains=np.zeros([nreal_units, nreal, nz, ny, nx], dtype=np.int8)

        if mode == "facies":
            method=self.f_method  # method of simulation
        elif mode == "units":
            if "units_fill_method" in self.dic_facies.keys():
                method = self.dic_facies["units_fill_method"]
            else:
                method = "homogenous"

        kwargs=self.dic_facies  # retrieve keyword arguments

        #default kwargs for SIS
        kwargs_def_SIS={"neig": 10, "r": 1, "probability": None,"SIS_orientation": False,"azimuth": 0,"dip": 0,"plunge": 0}

        kwargs_def_MPS={"varname":"code", "nv":1, "dataImage":None, "distanceType":["categorical"], "outputVarFlag":None,
                         "xr": 1, "yr": 1, "zr": 1, "maxscan": 0.25, "neig": 24, "thresh": 0.05, "xloc": False, "yloc": False, "zloc": False,
                         "homo_usage": 1, "rot_usage": 1, "rotAziLoc": False, "rotAzi": 0, "rotDipLoc": False, "rotDip": 0, "rotPlungeLoc": False, "rotPlunge": 0,
                          "radiusMode": "large_default", "rx": nx*sx, "ry": ny*sy, "rz": nz*sz, "anisotropyRatioMode": "one", "ax": 1, "ay": 1, "az": 1,
                          "angle1": 0, "angle2": 0, "angle3": 0,
                          "globalPdf": None, "localPdf": None, "probaUsage": 0, "localPdfRadius": 12., "deactivationDistance": 4., "constantThreshold": 1e-3, "npost":1}

        kwargs_def_TPGs={"neig": 20, "nit": 100, "grf_method": "fft"}

        methods=["SIS", "MPS", "TPGs", "SubPile"]
        kwargs_def=[kwargs_def_SIS, kwargs_def_MPS, kwargs_def_TPGs]

        for f_method, kw in zip(methods, kwargs_def):
            if method == f_method:
                if len(kwargs) == 0:
                    kwargs=kw
                else:
                    for k, v in kw.items():
                        if k not in kwargs.keys():
                            kwargs[k]=v

        # list objects
        if mode == "facies":
            list_obj = self.list_facies
        elif mode == "units":
            list_obj = self.SubPile.list_units

        #check if only 1 facies --> homogenous
        if len(list_obj) < 2 and method != "homogenous":
            method ="homogenous"
            if self.verbose:
                print("Unit {} has only one facies, facies method sets to homogenous".format(self.name))

        ### Simulations ###
        if method != "SubPile" or mode == "units":
            for iu in range(nreal_units): # loop over existing surfaces
                if ArchTable.verbose:
                    print("### Unit {} - realization {} ###".format(self.name, iu))
                if self.contact == "onlap":

                    mask=ArchTable.get_units_domains_realizations(iu) == self.ID

                    ## HOMOGENOUS ##
                    if method.lower() == "homogenous":

                        if len(list_obj) > 1: # more than one
                            if ArchTable.verbose:
                                print("WARNING !! More than one facies has been passed to homogenous unit {}\nFirst in the list is taken".format(self.name))
                        elif len(list_obj) < 1: # no facies added
                            raise ValueError ("No facies passed to homogenous unit {}".format(self.name))

                        ## setup a 3D array of the simulation grid size and assign one facies to the unit###
                        facies=list_obj[0]
                        for ireal in range(nreal):
                            facies_domains[iu, ireal][mask]=facies.ID

                    ## SIS ##
                    elif method.upper() == "SIS":
                        ## setup SIS ##

                        if mode == "facies":
                            hd, facies = ArchTable.hd_fa_in_unit(self, iu=iu)
                        elif mode == "units":
                            hd, facies = ArchTable.hd_un_in_unit(self, iu=iu)

                        cat_values=[i.ID for i in list_obj]  # ID values
                        
                        hd=np.array(hd)
                        facies=np.array(facies)
                            
                        ### orientation map ###
                        if (kwargs["SIS_orientation"]) == "follow_surfaces": # if orientations must follow surfaces
                            # Warning: This option changes alpha and beta angles assuming that rz is smaller than rx and ry. Moreover, rx and ry must be similar.

                            azi,dip=ArchTable.orientation_map(self, smooth=5)
                            for cm in self.list_f_covmodel: # iterate through all facies covmodels and change angles
                                cm.alpha=azi
                                cm.beta=dip

                        elif kwargs["SIS_orientation"]: # if set True, change angles with inputs angles
                            al=kwargs["alpha"]
                            be=kwargs["beta"]
                            ga=kwargs["gamma"]
                            if type(al) is list and type(be) is list and type(ga) is list:
                            #if hasattr(al,"__iter__") and hasattr(be,"__iter__") and hasattr(ga,"__iter__"): #if a list is given
                                assert len(al) == len(be) == len(ga), "Error: number of given values/arrays in alpha, beta and gamma must be the same"
                                for i in range(len(al)): # loop over
                                    for cm in self.list_f_covmodel:
                                            cm.alpha=al[i]
                                            cm.beta=be[i]
                                            cm.gamma=ga[i]
                            else: # if only a ndarray or a value are given
                                for cm in self.list_f_covmodel:
                                    cm.alpha=al
                                    cm.beta=be
                                    cm.gamma=ga


                        #Modify sill of covmodel if there is only one
                        if len(self.list_f_covmodel) == 1:
                            if ArchTable.verbose:
                                print("Only one facies covmodels for multiples facies, adapt sill to right proportions")

                            cm=self.list_f_covmodel[0]

                            sill=cm.sill() #get sill

                            self.list_f_covmodel=[] #reset list

                            ifa=0
                            for fa in list_obj: #loop over facies
                                
                                cm_copy=copy.deepcopy(cm) #make a copy

                                if "probability" in kwargs.keys():
                                    if kwargs["probability"] is not None:
                                        p = kwargs["probability"][ifa]
                                    else:
                                        p=np.sum(facies == fa.ID)/len(facies == fa.ID) #calculate proportion of facies
                                else:
                                    p=np.sum(facies == fa.ID)/len(facies == fa.ID) #calculate proportion of facies

                                if p > 0:
                                    var=p*(1-p) #variance
                                    ratio=var/sill  #ratio btw covmodel and real variance

                                    for e in cm_copy.elem:
                                        e[1]["w"] *= ratio
                                else:
                                    for e in cm_copy.elem:
                                        e[1]["w"] *= 1
                                self.list_f_covmodel.append(cm_copy)
                                ifa += 1

                        ## Simulation
                        simus = gci.simulateIndicator3D(cat_values, self.list_f_covmodel, dimensions, spacing, origin,
                                                        nreal=nreal, method="simple_kriging", x=hd, v=facies, mask=mask,
                                                        searchRadiusRelative=kwargs["r"], nneighborMax=kwargs["neig"],
                                                        probability=kwargs["probability"], verbose=verbose, nthreads = ArchTable.ncpu, seed=seed+iu)["image"].val

                        ### rearrange data into a 2D array of the simulation grid size ###
                        for ireal in range(nreal):
                            grid=simus[ireal]
                            grid[grid==0]=np.nan
                            grid[mask==0]=np.nan  # extract only the part on the real domain
                            facies_domains[iu, ireal][mask]=grid[mask]

                    elif method.upper() == "MPS":
                        ## assertions ##
                        assert  isinstance(self.f_TI, geone.img.Img), "TI is not a geone image object"

                        #load parameters
                        TI=self.f_TI

                        #facies IDs
                        IDs=np.unique(TI.val)
                        nclass=len(IDs)
                        classInterval=[]
                        for c in IDs:
                            classInterval.append([c-0.5, c+0.5])

                        # extract hard data
                        hd=np.ones([4, 1])
                        for fa in list_obj: #facies by facies in unit
                            fa_x=np.array(fa.x)
                            fa_y=np.array(fa.y)
                            fa_z=np.array(fa.z)
                            hdi=np.zeros([4, fa_x.shape[0]])
                            hdi[0]= fa_x
                            hdi[1]=fa_y
                            hdi[2]=fa_z
                            hdi[3]=np.ones(fa_x.shape)*fa.ID
                            hd=np.concatenate([hd, hdi], axis=1)
                        hd=np.delete(hd, 0, axis=1) # remove 1st point (I didn't find a clever way to do it...)
                        pt=img.PointSet(npt=hd.shape[1], nv=4, val=hd)
                        pt.set_varname("code")

                        #DS research
                        snp=dsi.SearchNeighborhoodParameters(
                            radiusMode=kwargs["radiusMode"], rx=kwargs["rx"], ry=kwargs["ry"], rz=kwargs["rz"],
                            anisotropyRatioMode=kwargs["anisotropyRatioMode"], ax=kwargs["ax"], ay=kwargs["ay"], az=kwargs["az"],
                            angle1=kwargs["angle1"], angle2=kwargs["angle2"], angle3=kwargs["angle3"])

                        snp_l = []
                        for iv in range(kwargs["nv"]):
                            snp_l.append(snp)

                        #DS softproba
                        sp=dsi.SoftProbability(
                            probabilityConstraintUsage=kwargs["probaUsage"],   # probability constraints method (1 for globa, 2 for local)
                            nclass=nclass,                  # number of classes of values
                            classInterval=classInterval,  # list of classes
                            localPdf= kwargs["localPdf"],             # local target PDF
                            globalPdf=kwargs["globalPdf"],
                            localPdfSupportRadius=kwargs["localPdfRadius"],      # support radius
                            comparingPdfMethod=5,           # method for comparing PDF's (see doc: help(gn.deesseinterface.SoftProbability))
                            deactivationDistance=kwargs["deactivationDistance"],       # deactivation distance (checking PDF is deactivated for narrow patterns)
                            constantThreshold=kwargs["constantThreshold"])        # acceptation threshold

                        sp_l = []
                        for iv in range(kwargs["nv"]):
                            sp_l.append(sp)

                        #DS input
                        deesse_input=dsi.DeesseInput(
                            nx=nx, ny=ny, nz=nz,       # dimension of the simulation grid (number of cells)
                            sx=sx, sy=sy, sz=sz,     # cells units in the simulation grid (here are the default values)
                            ox=ox, oy=oy, oz=oz,     # origin of the simulation grid (here are the default values)
                            nv=kwargs["nv"], varname=kwargs["varname"],       # number of variable(s), name of the variable(s)
                            nTI=1, TI=TI,             # number of TI(s), TI (class dsi.Img)
                            dataPointSet=pt,            # hard data (optional)
                            dataImage = kwargs["dataImage"],
                            outputVarFlag = kwargs["outputVarFlag"],
                            distanceType = kwargs["distanceType"],  # distance type: proportion of mismatching nodes (categorical var., default)
                            softProbability=sp_l,
                            searchNeighborhoodParameters=snp_l,
                            homothetyUsage=kwargs["homo_usage"],
                            homothetyXLocal=kwargs["xloc"],
                            homothetyXRatio=kwargs["xr"],
                            homothetyYLocal=kwargs["yloc"],
                            homothetyYRatio=kwargs["yr"],
                            homothetyZLocal=kwargs["zloc"],
                            homothetyZRatio=kwargs["zr"],
                            rotationUsage=kwargs["rot_usage"],            # tolerance or not
                            rotationAzimuthLocal=kwargs["rotAziLoc"], #    rotation according to azimuth: global
                            rotationAzimuth=kwargs["rotAzi"],
                            rotationDipLocal=kwargs["rotDipLoc"],
                            rotationDip=kwargs["rotDip"],
                            rotationPlungeLocal=kwargs["rotPlungeLoc"],
                            rotationPlunge=kwargs["rotPlunge"],
                            nneighboringNode=kwargs["neig"],        # max. number of neighbors (for the patterns)
                            distanceThreshold=kwargs["thresh"],     # acceptation threshold (for distance between patterns)
                            maxScanFraction=kwargs["maxscan"],       # max. scanned fraction of the TI (for simulation of each cell)
                            npostProcessingPathMax=kwargs["npost"],   # number of post-processing path(s)
                            seed=seed,                   # seed (initialization of the random number generator)
                            nrealization=nreal,         # number of realization(s)
                            mask=mask)                 # ncpu

                        deesse_output=dsi.deesseRun(deesse_input, nthreads=ArchTable.ncpu, verbose=verbose)

                        simus=deesse_output["sim"]
                        for ireal in range(nreal):
                            sim=simus[ireal]
                            #self.facies_domains[iu, ireal]=sim.val[0] #output in facies_domains
                            facies_domains[iu, ireal][mask]=sim.val[0][mask]

                    elif method == "TPGs": #truncated (pluri)gaussian

                        #load params#
                        flag=self.flag
                        G_cm=self.G_cm
                        ## data format ##
                        # data=(x, y, z, g1, g2, v), where x, y, z are the cartesian coordinates, g1 and g2 are the values of 
                        # first/second gaussian fields and v is the facies value

                        ## setup and get hard data
                        hd=np.array([])
                        for fa in list_obj:
                            #ndarray
                            nd=len(fa.x)
                            fa_x=np.array(fa.x)
                            fa_y=np.array(fa.y)
                            fa_z=np.array(fa.z)
                            facies=fa.ID*np.ones(nd)# append facies IDs

                            hd=np.concatenate([hd.reshape(-1, 6), np.concatenate([[fa_x], [fa_y], [fa_z], [np.zeros(nd)], [np.zeros(nd)], [facies]], axis=0).T]) # append data for input TPGs

                        if hd.shape[0] == 0:
                            hd=None
                        simus=run_tpgs(nreal, xg, yg, zg, hd, G_cm, flag, nmax=kwargs["neig"], grf_method=kwargs["grf_method"], mask=mask)
                        for ireal in range(nreal):
                            grid=simus[ireal]
                            grid[mask==0]=np.nan  # extract only the part on the real domain
                            facies_domains[iu, ireal][mask]=grid[mask]

                    elif method == "nearest":
                        if ArchTable.verbose:
                            print("<===Nearest neighbors interpolation===>")

                        from sklearn.neighbors import NearestNeighbors

                        X = np.ones([nz, ny, nx])* ArchTable.xgc
                        Y = np.ones([nz, ny, nx])
                        Y[:] = np.ones([nx, ny]).T * ArchTable.ygc.reshape(-1, 1)
                        Z = np.ones([nz, ny, nx])
                        Z[:, :] =( np.ones([nz, nx]) * ArchTable.zgc.reshape(-1, 1)).reshape(nz, 1, nx)
                        xu3D = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

                        # get hard data
                        if mode == "facies":
                            hd, facies = ArchTable.hd_fa_in_unit(self, iu=iu)
                        elif mode == "units":
                            hd, facies = ArchTable.hd_un_in_unit(self, iu=iu)

                        hd=np.array(hd)
                        facies=np.array(facies)

                        X_fit = hd
                        y_fit = facies

                        X_pred = xu3D[mask.flatten()]

                        # fit
                        nn = NearestNeighbors(n_neighbors=1).fit(X_fit)

                        #pred
                        res = nn.kneighbors(X_pred, return_distance=False, n_neighbors=1)

                        # assign
                        y_pred = y_fit[res]
                        simus = np.zeros([nz, ny, nx])
                        simus[mask] = y_pred[:, 0]  # reassign values

                        for ireal in range(nreal):
                            facies_domains[iu, ireal][mask]=simus[mask]


                if mode == "facies":
                    ArchTable.Geol.facies_domains[iu,:, mask]=facies_domains[iu,:, mask]  # store results
                elif mode == "units":
                    ArchTable.Geol.units_domains[iu, mask] = facies_domains[iu, 0, mask]  # store results
        elif method == "SubPile": # Hierarchical filling
            if ArchTable.verbose:
                print("SubPile filling method, nothing happened")
            pass

class Surface():

    def __init__(self, name="Surface_1",
                dic_surf={"int_method": "nearest", "covmodel": None, "N_transfo": False},
                contact="onlap"):

        """
        Class Surface, must be linked to a Unit object
        ###Description###
        name          : string to identify the surface for debugging purpose
        contact         : string, onlap or erode. Onlap indicates
                         that this surface cannot erode older surfaces,
                         on contrary to erode surfaces
        dic_surf        : parameters for surface interpolation
            int_method: method of interpolation, possible methods are:
                         kriging, MPS, grf, grf_ineq, linear, nearest,
            covmodel : geone covariance model (see doc Geone for more information)
                         if a multi-gaussian method is used
            N_transfo: bool, Normal-score transform. Flag to apply or not a
                         Normal Score on the data for the interpolation
            kwargs:
              ||
              ¦¦
              \/
        Units kwargs  (passed in dic_surf directly):
            for the search ellipsoid (kriging, grf, ...):
                - r, relative (to covariance model) radius of research (default is 1)
                - neig, number of neighbours
                - krig_type, string, kriging method (ordinary_kriging, simple_kriging)
            for MPS:
                - TI
                - ...
            if N_transfo is True:
                tau: number between 0 and 0.5, threshold to apply at the distribution
                      implying that 2*tau of the data are not "present"
                      --> avoids extreme values effects and
                      allows simulations to go higher/lower than data. Default 0.
                bandwidth_mult: ]0 to 1] multiplier to apply on the bandwidth to 
                                compute the kernel distribution. Default 1.

        """

        assert contact in ["erode", "onlap", "comf"], "contact must be erode or onlap or comf"
        assert dic_surf["int_method"] is not None, "An interpolation method must be provided"
        assert dic_surf["int_method"] in ["linear", "cubic", "nearest", "kriging", "grf", "grf_ineq", "MPS"], "Unknown interpolation method"

        #default values dic surf
        kwargs_def_surface={"covmodel": None, "N_transfo": False, "bandwidth_mult": 1, "tau": 0}
        for k, v in kwargs_def_surface.items():
            if k not in dic_surf.keys():
                dic_surf[k]=v

        self.name=name
        self.x=[]
        self.y=[]
        self.z=[]
        self.ineq=[]
        self.int_method=dic_surf["int_method"]
        self.N_transfo=dic_surf["N_transfo"]
        self.covmodel=dic_surf["covmodel"]
        self.contact=contact

        #check inputs for surface methods
        if self.int_method in ["kriging", "grf", "grf_ineq"]:
            self.get_surface_covmodel()

        if self.covmodel is not None and self.N_transfo and self.int_method in ["kriging", "grf", "grf_ineq"]:
            if self.get_surface_covmodel().sill() != 1:
                print("Unit {}: !! Normal transformation is applied but the variance of the Covariance model is not equal to 1 !!".format(self.name))

        self.dic_surf=dic_surf

    def copy(self):
        return copy.deepcopy(self)

    def set_covmodel(self, covmodel):
        """
        change or add a covmodel for surface interpolation of self unit.
        covmodel: geone.covModel2D that will serve for interpolation if the chosen method is grf, grf ineq or kriging
        """
        if isinstance(covmodel, geone.covModel.CovModel2D):
            self.covmodel=covmodel
            # self.dic_surf["covmodel"]=covmodel
            print("Surface {}: covmodel added".format(self.name))
        else:
            print("Surface {}: covmodel not a geone 2D covmodel".format(self.name))

    def get_surface_covmodel(self, vb=1):
        if self.covmodel is None:
            if vb:
                print ("Warning: Unit '{}' have no Covmodel for surface interpolation".format(self.name))
            return None
        else:
            return self.covmodel

class Facies():

    """
    class for facies (2nd level of hierarchy)

    """

    def __init__(self, ID, name, color):

        self.x=[]
        self.y=[]
        self.z=[]
        if ID != 0:
            self.ID=ID
        else:
            raise ValueError("ID facies cannot be equal to 0")
        self.name=name
        self.c=color

    def __eq__(self, other):
        if (self.name == other.name) & (self.ID == other.ID):
            return True
        else:
            return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

class Prop():

    """
    Class for defining a propriety to simulate (3rd level)
    """

    def __init__(self, name, facies, covmodels, means, int_method="sgs", x=None, v= None, def_mean=1, vmin=None, vmax=None):

        """
        name     : name of the propriety
        facies   : list, facies in which we want to simulate
                     the property (in the others the propriety will
                     be set homogenous with a default value
        covmodels: list, covmodels for the simulation
                     (same size of facies or only 1)
        means    : list, mean of the property in each
                     facies (same size of facies)
        x        : ndarray of size (n, 3 --> x, y and value)
                     hard data (if there is some)
        int_method: string, interpolation method
                     --> sgs, fft or homogenous, default is sgs
        def_mean : float, default mean to used if none is passed in means array
        vmin, vmax: float, min resp. max value of the property.
                     Values below (resp. above) will be set to
                     the min (resp. max) value
        """

        assert isinstance(facies, list), "Facies must be a list of facies, even there is only one"
        assert isinstance(vmin, float) or isinstance(vmin, int) or vmin is None, "Vmin error"
        assert isinstance(vmax, float) or isinstance(vmax, int) or vmax is None, "Vmax error"

        self.name=name
        self.facies=facies
        n_facies=len(facies)
        self.n_facies=n_facies

        self.vmin=vmin
        self.vmax=vmax
        self.x=x
        self.v=v

        #covmodels
        try:
            for cm in covmodels:
                pass
            self.covmodels=covmodels
        except:
            if isinstance(covmodels, gcm.CovModel3D):
                self.covmodels=[covmodels]*n_facies
            else:
                raise ValueError("{} is not a valid CovModel3D".format(cm))

        #means
        self.means=means
        try:
            for m in means:
                pass
        except:
            self.means=[means]*n_facies

        #interpolation methods
        self.int=int_method
        try:
            for i_method in int_method:
                if i_method in ("sgs", "fft", "homogenous", "mps", "homogenous_uniform"):
                    pass
                else:
                    raise ValueError("{} is not a valid inteprolation method".format(i_method))
        except:
            if int_method in ("sgs", "fft", "homogenous", "mps","homogenous_uniform"):
                self.int=[int_method]*n_facies
            else:
                raise ValueError("{} is not a valid inteprolation method".format(i_method))

        self.def_mean=def_mean

        def __eq__(self, other):
            if (self.name == other.name):
                return True
            else:
                return False

        def __repr__(self):
            return self.name


    def add_hd(self, x,v):

        assert x.shape[1] == 3, "invalid shape for hd position (x), must be (ndata, 3)"
        assert v.shape[0] == x.shape[0], "invalid number of data points between v and x"

        self.x=x
        self.v=v


class borehole():
    def __init__(self, name, ID, x, y, z, depth, log_strati, log_facies=None):

        self.name=name  # name of lithology
        self.ID=ID  # ID of the boreholes
        self.x=x  # x coordinate of borehole
        self.y=y  # y coordinate of borehole
        self.z=z  # altitude of the bh (côte terrain)
        self.depth=depth
        self.log_strati=log_strati
        self.log_facies=log_facies

        if log_strati is not None:
            self.list_stratis=[s for s, d in self.log_strati]
            if len(self.list_stratis) == 0:
                self.log_strati=None

        if log_facies is not None:
            self.list_facies =[s for s, d in self.log_facies]
            if len(self.log_facies) == 0:
                self.log_facies=None


    def get_list_stratis(self):
        self.list_stratis=[s for s, d in self.log_strati]
        return self.list_stratis

    def get_list_facies(self):
        self.list_facies =[s for s, d in self.log_facies]
        return self.list_facies

    def __eq__(self, other):
        if  (self.ID == other.ID) & \
        (self.x == other.x) & (self.y == other.y) & (self.z == other.z):
            return True
        else:
            return False

    def prop_units(self):

        """
        Return a dictionnary of the proportion of the units in the borehole
        """

        d = {}

        alt_prev = self.z
        unit_prev = None
        for s in self.log_strati:
            if s[0] is not None:
                if s[0].name not in d.keys():
                    d[s[0].name] = 0
                
                if unit_prev is not None:
                    thk = unit_prev[1] - s[1]
                    d[unit_prev[0].name] += thk
                    
                unit_prev = s
        thk = unit_prev[1] - (self.z - self.depth)
        d[unit_prev[0].name] += thk 

        for k,v in d.items():  # mean
            d[k] /= self.depth

        return d

    def extract(self, z, vb=1):

        """extract the units and facies information at specified altitude z"""

        ls=False
        lf=False
        unit=None
        facies=None

        if z > self.z or z < self.z - self.depth:
            if vb:
                print("borehole have no information at this altitude")
            return None

        if self.log_strati is not None:
            ls=True
            for i in range(len(self.log_strati)):
                if i < len(self.log_strati) - 1:
                    s1 = self.log_strati[i]
                    s2 = self.log_strati[i+1]
                else:
                    s1 = self.log_strati[i]
                    s2 = (None, self.z - self.depth)
                if s1[1] >= z and s2[1] < z:
                    unit=s1[0]
                    break


        if self.log_facies is not None:
            lf=True
            for i in range(len(self.log_facies)):
                if i < len(self.log_facies) - 1:
                    f1 = self.log_facies[i]
                    f2 = self.log_facies[i+1]
                else:
                    f1 = self.log_facies[i]
                    f2 = (None, self.z - self.depth)
                if f1[1] >= z and f2[1] < z:
                    facies=f1[0]
                    break
            

        if ls and lf:
            return (unit, facies)
        elif ls:
            return unit
        elif lf:
            return facies


class Geol():

    """
    ArchPy output class which contain the results for the geological simulations
    """

    def __init__(self):

        self.surfaces=None
        self.surfaces_bot=None
        self.org_surfaces=None
        self.surfaces_by_piles={}
        self.surfaces_bot_by_piles={}
        self.org_surfaces_by_piles={}
