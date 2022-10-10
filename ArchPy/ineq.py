import numpy as np
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import scipy
import geone
import geone.covModel as gcm
import geone.geosclassicinterface as gci
import sys
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.neighbors import KDTree

from ArchPy.data_transfo import *


### functions ###
def krige(x, v, xu, cov_model, method='simple_kriging', mean=None):
    """
    Performs kriging - interpolates at locations xu the values v measured at locations x.
    Covariance model given should be:
        - in same dimension as dimension of locations x, xu
        - in 1D, it is then used as an omni-directional covariance model
    (see below).

    :param x:       (2-dimensional array of shape (n, d)) coordinates
                        of the data points (n: number of points, d: dimension)
                        Note: for data in 1D, it can be a 1-dimensional array of shape (n,)
    :param v:       (1-dimensional array of shape (n,)) values at data points

    :param xu:      (2-dimensional array of shape (nu, d)) coordinates
                        of the points where the interpolation has to be done
                        (nu: number of points, d: dimension same as for x),
                        called unknown points
                        Note: for data in 1D, it can be a 1-dimensional array of shape (nu,)

    :param cov_model:
                    covariance model:
                        - in same dimension as dimension of points (d), i.e.:
                            - CovModel1D class if data in 1D (d=1)
                            - CovModel2D class if data in 2D (d=2)
                            - CovModel3D class if data in 3D (d=3)
                        - or CovModel1D whatever dimension of points (d):
                            - used as an omni-directional covariance model

    :param method:  (string) indicates the method used:
                        - 'simple_kriging': interpolation by simple kriging
                        - 'ordinary_kriging': interpolation by ordinary kriging

    :param mean:    (None or float or ndarray) mean of the simulation
                        (for simple kriging only):
                            - None   : mean of hard data values (stationary),
                                       i.e. mean of v
                            - float  : for stationary mean (set manually)
                            - ndarray: of of shape (nu,) for non stationary mean,
                                mean at point xu
                        For ordinary kriging (method = 'ordinary_kriging'),
                        this parameter ignored (not used)

    :return:        (vu, vu_std) with:
                        vu:     (1-dimensional array of shape (nu,)) kriged values (estimates) at points xu
                        vu_std: (1-dimensional array of shape (nu,)) kriged standard deviation at points xu
    """

    # Prevent calculation if covariance model is not stationary
    if not cov_model.is_stationary():
        print("ERROR: 'cov_model' is not stationary: krige can not be applied")
        return None, None

    # Get dimension (d) from x
    if np.asarray(x).ndim == 1:
        # x is a 1-dimensional array
        x = np.asarray(x).reshape(-1, 1)
        d = 1
    else:
        # x is a 2-dimensional array
        d = x.shape[1]

    # Get dimension (du) from xu
    if np.asarray(xu).ndim == 1:
        # xu is a 1-dimensional array
        xu = np.asarray(xu).reshape(-1, 1)
        du = 1
    else:
        # xu is a 2-dimensional array
        du = xu.shape[1]

    # Check dimension of x and xu
    if d != du:
        print("ERROR: 'x' and 'xu' do not have same dimension")
        return None, None

    # Check dimension of cov_model and set if used as omni-directional model
    if cov_model.__class__.__name__ != 'CovModel{}D'.format(d):
        if isinstance(cov_model, gcm.CovModel1D):
            omni_dir = True
        else:
            print("ERROR: 'cov_model' is incompatible with dimension of points")
            return None, None
    else:
        omni_dir = False

    # Number of data points
    n = x.shape[0]
    # Number of unknown points
    nu = xu.shape[0]

    # Check size of v
    v = np.asarray(v).reshape(-1)
    if v.size != n:
        print("ERROR: size of 'v' is not valid")
        return None, None

    # Method
    ordinary_kriging = False
    if method == 'simple_kriging':
        if mean is None:
            mean = np.mean(v)
        else:
            mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if mean.size not in (1, nu):
                print("ERROR: size of 'mean' is not valid")
                return None, None
        nmat = n # order of the kriging matrix
    elif method == 'ordinary_kriging':
        ordinary_kriging = True
        nmat = n+1 # order of the kriging matrix
    else:
        print("ERROR: 'method' is not valid")
        return None, None

    # Covariance function
    cov_func = cov_model.func() # covariance function
    if omni_dir:
        # covariance model in 1D is used
        cov0 = cov_func(0.) # covariance function at origin (lag=0)
    else:
        cov0 = cov_func(np.zeros(d)) # covariance function at origin (lag=0)

    # Set
    #   - kriging matrix (mat) of order nmat
    #   - right hand side of the kriging system (b),
    #       matrix of dimension nmat x nu
    mat = np.ones((nmat, nmat))
    for i in range(n-1):
        # lag between x[i] and x[j], j=i+1, ..., n-1
        h = x[(i+1):] - x[i]
        if omni_dir:
            # compute norm of lag
            h = np.sqrt(np.sum(h**2, axis=1))
        cov_h = cov_func(h)
        mat[i, (i+1):n] = cov_h
        mat[(i+1):n, i] = cov_h
        mat[i,i] = cov0

    b = np.ones((nmat, nu))
    for i in range(n):
        # lag between x[i] and every xu
        h = xu - x[i]
        if omni_dir:
            # compute norm of lag
            h = np.sqrt(np.sum(h**2, axis=1))
        b[i,:] = cov_func(h)

    if ordinary_kriging:
        mat[-2,-2] = cov0
        mat[-1,-1] = 0.0
    else:
        mat[-1,-1] = cov0

    # Solve the kriging system
    w = np.linalg.solve(mat, b) # w: matrix of dimension nmat x nu


    # Kriged standard deviation at unknown points
    vu_std = np.sqrt(np.maximum(0, cov0 - np.array([np.dot(w[:,i], b[:,i]) for i in range(nu)])))
    if ordinary_kriging:
        w = w[:n]
        #vu_std = values_std[:n]
    return w,vu_std


###
def run_sim(data_org,xg,covmodel,var,nsim = 10,nit=20):

    """
    ### inputs ###
    data_org :
    xg : 1d grid (np.arange(x0,x1+sx,sx), must be sorted
    nsim :
    nit :
    covmodel :

    output : array of size (nsim,nx), list of all simulations in one 2D array
    """

    # grid
    nx = xg.shape[0]
    sx = np.diff(xg)[0]
    ox = xg[0]

    # copy data
    data = data_org.copy()
    mask_eq = (data[:,2] == data[:,2]) | (data[:,3] == data[:,3])

    #split data
    ineq_data = data[mask_eq]
    eq_data = data[~mask_eq]
    if eq_data.shape[0] > 0 :
        mean = np.mean(eq_data[:,1]) # what to do if no equality point ???
    else:
        mean = np.mean(ineq_data[:,1]) # mean on inequality points ? or propose another alternative ?

    #calculation weights
    weight_arr = np.zeros([ineq_data.shape[0],data.shape[0]-1])
    std_arr = np.zeros([ineq_data.shape[0]])
    for i in range(ineq_data.shape[0]):
        xu = ineq_data[i]
        x = np.concatenate([eq_data,np.delete(ineq_data,i,axis=0)]) # append remaining data (without inequality where we simulate)

        w,vu_std = simple_kriging(x[:,0],xu[0].reshape(-1,1),covmodel,mean=mean)
        weight_arr[i] = w
        std_arr[i] = vu_std

    #loop over inequality data
    lst = np.ones([nsim,nx])
    for isim in range(nsim):
        for it in range(nit):
            for i in range(ineq_data.shape[0]):
                xu = ineq_data[i]
                x = np.concatenate([eq_data,np.delete(ineq_data,i,axis=0)]) # append remaining data
                m = np.dot(weight_arr[i],x[:,1]) + (1-np.sum(weight_arr[i],axis=0))*mean
                s = std_arr[i]

                if np.abs(m) != np.inf and s > 0:

                    ## truncation
                    if (xu[2] != xu[2] ) & (xu[3] == xu[3]):
                        myclip_a = -np.inf
                        myclip_b = xu[3]

                    elif (xu[2] == xu[2]) & (xu[3] != xu[3]):
                        myclip_a = xu[2]
                        myclip_b = np.inf

                    elif (xu[2] == xu[2]) & (xu[3] == xu[3]):
                        myclip_a = xu[2]
                        myclip_b = xu[3]

                    a, b = (myclip_a - m) / s, (myclip_b - m) / s
                    xu[1] = truncnorm.rvs(a,b,loc=m,scale=s) # draw and update value

        #simulate and save it
        data_all = np.concatenate([eq_data,ineq_data]) #reappendd data
        sim = geone.grf.grf1D(covmodel,nx,sx,ox,mean=mean,var=var,x=data_all[:,0],v=data_all[:,1],printInfo=False) # grf with all data
        if sim is not None:
            lst[isim] = sim[0]

    return np.array(lst)

def Gibbs_estimate(data_org, covmodel, nit=50, krig_type="simple_kriging", mean=None, var=None, nmax=20):

    # copy data and rearrange data
    data = data_org.copy()

    # default values
    for idata in data:
        if (idata[3] != idata[3]) and (idata[4] == idata[4]): # inf ineq
            idata[2] = idata[4]
        elif(idata[3] == idata[3]) and (idata[4] != idata[4]): # sup ineq
            idata[2] = idata[3]
        elif (idata[3] == idata[3]) and (idata[4] == idata[4]): # sup and inf ineq
            assert idata[3] <= idata[4], "inf ineq must be inferior or equal to sup ineq in point {}".format(idata)
            idata[2] = (idata[3]+idata[4])/2
            if idata[3] == idata[4]: # if both ineq are equal
                idata[3] = np.nan
                idata[4] = np.nan
            
    #split data 
    mask_eq = (data[:,3] == data[:,3]) | (data[:,4] == data[:,4])
    ineq_data = data[mask_eq]
    eq_data = data[~mask_eq]
    neq = eq_data.shape[0]
    data = np.concatenate((eq_data, ineq_data))

    if krig_type == "ordinary_kriging":
        mean = None #mean set to None

    ## kriging to test and remove some ineq data 
    ##(an ineq is removed if the ineq is above/below kriging estimate +- 4 times krig standard deviation)
    #change ini data to krige estimates only with eq data
    """
    if neq > 1:
        ini_v,ini_s = gcm.krige(eq_data[:,:2], eq_data[:,2], ineq_data[:,:2].reshape(-1,2), cov_model=covmodel, method=krig_type, mean=mean)
        arr_sup = ini_v+4*ini_s
        arr_inf = ini_v-4*ini_s
        mask_inf = (ineq_data[:,3] < arr_inf)
        mask_sup = (ineq_data[:,4] > arr_sup)
        
        #remove ineq_
        ineq_data[mask_inf, 3] = np.nan
        ineq_data[mask_sup, 4] = np.nan
        super_mask = (ineq_data[:, 3] == ineq_data[:, 3]) | (ineq_data[:, 4] == ineq_data[:, 4])
        ineq_data = ineq_data[super_mask]
        
        data = np.concatenate((eq_data, ineq_data))
    """
    
    #if no variance are given
    if var is None:
        var = covmodel.sill()

    if (eq_data.shape[0] > 0) & (mean is None) :
        mean = np.mean(eq_data[:,2]) # what to do if no equality point ???
    elif (eq_data.shape[0] == 0) & (mean is None):
        mean = (np.nanmean(ineq_data[:,3]) + np.nanmean(ineq_data[:,4]))/2
        krig_type = "simple_kriging" #set to simple kriging to avoid surface to go at infinity if no equality points
    if krig_type == "ordinary_kriging":
        mean = None #mean set to None

    #calculation weights --> add nmax neighbours
    nineq = ineq_data.shape[0]
    neq = eq_data.shape[0]
    ndata = data.shape[0]
    weight_arr = np.zeros([nineq,ndata])
    std_arr = np.zeros(nineq)
    
    #select nearest neighbours using a kdtree
    tree = KDTree(data[:,:2])
    if nmax > ndata:
        nmax = ndata
    for i in range(nineq):
        xu = ineq_data[i]
        idx = tree.query(xu[:2].reshape(1,-1),k=nmax,return_distance=False)[0] # search nearest neighbours
        x = data[idx[1:]] # select nearest neig
        w,vu_std = krige(x[:,:2],x[:,3],xu[:2].reshape(-1,2),cov_model = covmodel,method = krig_type,mean=mean)
        weight_arr[i,idx[1:]] = w[:,0]
        std_arr[i] = vu_std
        
    ### loop over inequality data, Gibbs sampler ###
    
    vals=np.zeros([nineq, nit+1])
    if neq > 1:
        #initial values
        x_tmp = []
        v_tmp = []
        for ieq in eq_data:
            x_tmp.append(tuple(ieq[0:2]))
            v_tmp.append(ieq[2])
        for i in range(nineq):
            xu = ineq_data[i]
            idx = tree.query(xu[:2].reshape(1,-1),k=ndata,return_distance=False)[0] # search nearest neighbours
            idx2 = [i for i in range(len(data)) if tuple(data[i,:2]) in x_tmp]
            idx2 = idx2[:nmax]
            x = np.array(x_tmp)[idx2]
            v = np.array(v_tmp)[idx2]
            m,s = gcm.krige(x, v, xu[:2].reshape(-1,2), cov_model=covmodel, method=krig_type, mean=mean)
            if np.abs(m) != np.inf and s > 0:
                ## truncation
                if (xu[3] != xu[3]) and (xu[4] == xu[4]):
                    myclip_a = -np.inf
                    myclip_b = xu[4]
                elif (xu[3] == xu[3]) and (xu[4] != xu[4]):
                    myclip_a = xu[3]
                    myclip_b = np.inf
                elif (xu[3] == xu[3]) and (xu[4] == xu[4]):
                    myclip_a = xu[3]
                    myclip_b = xu[4]
                a, b = (myclip_a - m) / s, (myclip_b - m) / s
                val = truncnorm.rvs(a,b,loc=m,scale=s) # draw and update value
                if ~np.isinf(val) and ~np.isnan(val):
                    xu[2] = val
                    vals[i, 0]=val
                    x_tmp.append(tuple(xu[:2]))
                    v_tmp.append(val)
    for it in range(nit):
        for i in range(nineq):
            xu = ineq_data[i]
            v_data = np.concatenate([eq_data,ineq_data]) #reappend data
            if krig_type == "simple_kriging":
                m = np.dot(weight_arr[i], v_data[:,2]) + (1-np.sum(weight_arr[i],axis=0))*mean #compute expected value using simple kriging
            elif krig_type == "ordinary_kriging":
                m = np.dot(weight_arr[i], v_data[:,2])
            s = std_arr[i]
            #draw using truncated gaussians
            if np.abs(m) != np.inf and s > 0:
                ## truncation
                if (xu[3] != xu[3]) and (xu[4] == xu[4]):
                    myclip_a = -np.inf
                    myclip_b = xu[4]
                elif (xu[3] == xu[3]) and (xu[4] != xu[4]):
                    myclip_a = xu[3]
                    myclip_b = np.inf
                elif (xu[3] == xu[3]) and (xu[4] == xu[4]):
                    myclip_a = xu[3]
                    myclip_b = xu[4]
                a, b = (myclip_a - m) / s, (myclip_b - m) / s
                val = truncnorm.rvs(a,b,loc=m,scale=s) # draw and update value
                if ~np.isinf(val) and ~np.isnan(val):
                    xu[2] = val
                    vals[i, it+1]=val
    
    ineq_vals=vals.mean(1)  #  estimate best value
    o=0
    for idata in data[neq:]:
        idata[2]=ineq_vals[o]
        o+=1
    
    return data
            
            
def run_sim_2d(data_org,xg,yg,covmodel,var=None,mean=None,krig_type = "simple_kriging",nsim = 1,nit=20,nmax=20, grf_method = "fft", ncpu=-1, seed=123456789):


    """
    #####
    inputs
    #####
    data_org : 2D array-like of size nd x 6
               [[x1, y1, z1, vineq_min1, vineq_max1],
                [x2, y2, z2, vineq_min2, vineq_max2],
                ...,
                [xnd, ynd, znd, vineq_minnd, vineq_maxnd],]
    xg : x coordinate vector (np.arange(x0,x1+sx,sx)
    yg : y coordinate vector (np.arange(y0,y1+sy,sy)
    nsim : int, number of simulations
    nit : int, number of Gibbs iterations
    covmodel : covariance model (see geone.CovModel documentation)
    #####
    output : array of size (nsim,ny,nx), array of all simulations
    #####
    """
    # grid
    nx = xg.shape[0]-1
    sx = np.diff(xg)[0]
    ox = xg[0]
    ny = yg.shape[0]-1
    sy = np.diff(yg)[0]
    oy = yg[0]
    # copy data and rearrange data
    data = data_org.copy()

    """ SHIFTED TO BASE. A supprimer
    # default values
    for idata in data:
        if (idata[3] != idata[3]) and (idata[4] == idata[4]): # inf ineq
            idata[2] = idata[4]
        elif(idata[3] == idata[3]) and (idata[4] != idata[4]): # sup ineq
            idata[2] = idata[3]
        elif (idata[3] == idata[3]) and (idata[4] == idata[4]): # sup and inf ineq
            assert idata[3] <= idata[4], "inf ineq must be inferior or equal to sup ineq in point {}".format(idata)
            idata[2] = (idata[3]+idata[4])/2
            if idata[3] == idata[4]: # if both ineq are equal
                idata[3] = np.nan
                idata[4] = np.nan
    """
    #split data
    mask_eq = (data[:,3] == data[:,3]) | (data[:,4] == data[:,4])
    ineq_data = data[mask_eq]
    eq_data = data[~mask_eq]
    neq = eq_data.shape[0]
    data = np.concatenate((eq_data, ineq_data))

    if krig_type == "ordinary_kriging":
        mean = None #mean set to None

    ## kriging to test and remove some ineq data 
    ##(an ineq is removed if the ineq is above/below kriging estimate +- 4 times krig standard deviation)
    #change ini data to krige estimates only with eq data
    if neq > 1:
        ini_v,ini_s = gcm.krige(eq_data[:,:2], eq_data[:,2], ineq_data[:,:2].reshape(-1,2), cov_model=covmodel, method=krig_type, mean=mean)
        arr_sup = ini_v+4*ini_s
        arr_inf = ini_v-4*ini_s
        mask_inf = (ineq_data[:,3] < arr_inf)
        mask_sup = (ineq_data[:,4] > arr_sup)

        #remove ineq_
        ineq_data[mask_inf, 3] = np.nan
        ineq_data[mask_sup, 4] = np.nan
        super_mask = (ineq_data[:, 3] == ineq_data[:, 3]) | (ineq_data[:, 4] == ineq_data[:, 4])
        ineq_data = ineq_data[super_mask]

        data = np.concatenate((eq_data, ineq_data))
    
    #if no variance are given
    if var is None:
        var = covmodel.sill()

    if (eq_data.shape[0] > 0) & (mean is None) :
        mean = np.mean(eq_data[:,2]) # what to do if no equality point ???
    elif (eq_data.shape[0] == 0) & (mean is None):
        mean = (np.nanmean(ineq_data[:,3]) + np.nanmean(ineq_data[:,4]))/2
        krig_type = "simple_kriging" #set to simple kriging to avoid surface to go at infinity if no equality points
    if krig_type == "ordinary_kriging":
        mean = None #mean set to None

    #calculation weights --> add nmax neighbours
    nineq = ineq_data.shape[0]
    neq = eq_data.shape[0]
    ndata = data.shape[0]
    weight_arr = np.zeros([nineq,ndata])
    std_arr = np.zeros(nineq)
    #select nearest neighbours using a kdtree
    tree = KDTree(data[:,:2])
    if nmax > ndata:
        nmax = ndata
    for i in range(nineq):
        xu = ineq_data[i]
        idx = tree.query(xu[:2].reshape(1,-1),k=nmax,return_distance=False)[0] # search nearest neighbours
        x = data[idx[1:]] # select nearest neig
        w,vu_std = krige(x[:,:2],x[:,3],xu[:2].reshape(-1,2),cov_model = covmodel,method = krig_type,mean=mean)
        weight_arr[i,idx[1:]] = w[:,0]
        std_arr[i] = vu_std

    ### loop over inequality data, Gibbs sampler ###
    lst = np.ones([nsim,ny,nx])
    for isim in range(nsim):
        if neq > 1:
            #initial values
            x_tmp = []
            v_tmp = []
            for ieq in eq_data:
                x_tmp.append(tuple(ieq[0:2]))
                v_tmp.append(ieq[2])
            for i in range(nineq):
                xu = ineq_data[i]
                idx = tree.query(xu[:2].reshape(1,-1),k=ndata,return_distance=False)[0] # search nearest neighbours
                idx2 = [i for i in range(len(data)) if tuple(data[i,:2]) in x_tmp]
                idx2 = idx2[:nmax]
                x = np.array(x_tmp)[idx2]
                v = np.array(v_tmp)[idx2]
                m,s = gcm.krige(x, v, xu[:2].reshape(-1,2), cov_model=covmodel, method=krig_type, mean=mean)
                if np.abs(m) != np.inf and s > 0:
                    ## truncation
                    if (xu[3] != xu[3]) and (xu[4] == xu[4]):
                        myclip_a = -np.inf
                        myclip_b = xu[4]
                    elif (xu[3] == xu[3]) and (xu[4] != xu[4]):
                        myclip_a = xu[3]
                        myclip_b = np.inf
                    elif (xu[3] == xu[3]) and (xu[4] == xu[4]):
                        myclip_a = xu[3]
                        myclip_b = xu[4]
                    a, b = (myclip_a - m) / s, (myclip_b - m) / s
                    val = truncnorm.rvs(a,b,loc=m,scale=s) # draw and update value
                    if ~np.isinf(val) and ~np.isnan(val):
                        xu[2] = val
                        x_tmp.append(tuple(xu[:2]))
                        v_tmp.append(val)
        for it in range(nit):
            for i in range(nineq):
                xu = ineq_data[i]
                v_data = np.concatenate([eq_data,ineq_data]) #reappend data
                if krig_type == "simple_kriging":
                    m = np.dot(weight_arr[i], v_data[:,2]) + (1-np.sum(weight_arr[i],axis=0))*mean #compute expected value using simple kriging
                elif krig_type == "ordinary_kriging":
                    m = np.dot(weight_arr[i], v_data[:,2])
                s = std_arr[i]
                #draw using truncated gaussians
                if np.abs(m) != np.inf and s > 0:
                    ## truncation
                    if (xu[3] != xu[3]) and (xu[4] == xu[4]):
                        myclip_a = -np.inf
                        myclip_b = xu[4]
                    elif (xu[3] == xu[3]) and (xu[4] != xu[4]):
                        myclip_a = xu[3]
                        myclip_b = np.inf
                    elif (xu[3] == xu[3]) and (xu[4] == xu[4]):
                        myclip_a = xu[3]
                        myclip_b = xu[4]
                    a, b = (myclip_a - m) / s, (myclip_b - m) / s
                    val = truncnorm.rvs(a,b,loc=m,scale=s) # draw and update value
                    if ~np.isinf(val) and ~np.isnan(val):
                        xu[2] = val
        #simulate and save it
        data_all = np.concatenate([eq_data,ineq_data]) #reappend data
        if grf_method == "fft":
            sim = geone.grf.grf2D(covmodel, [nx,ny], [sx,sy], [ox,oy], mean=mean, var=var, x=data_all[:,:2], v=data_all[:,2], printInfo=False) # grf with all data
            if sim is not None:
                lst[isim] = sim[0]
            else:
                print("Simulation failed")
                return None
        elif grf_method == "sgs":
            sim = gci.simulate2D(covmodel, [nx,ny], [sx,sy], [ox,oy], mean=mean, var=var, x=data_all[:,:2], v=data_all[:,2],
                                nthreads = ncpu, seed=seed, method=krig_type)
            lst[isim] = sim["image"].val[0,0]
    return np.array(lst)
