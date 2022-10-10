import numpy as np
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import truncnorm
from sklearn.neighbors import KDTree # kdtree for nearest neighbours
import copy
import time
import os
import inspect

#geone
import geone
import geone.covModel as gcm
import geone.grf as grf
from geone import img
import geone.imgplot as imgplt
import geone.deesseinterface as dsi
import geone.geosclassicinterface as gci


#### C Library

import os
from ctypes import CDLL,c_int,c_double,Structure
dirname = os.path.dirname(__file__)
rel_path = "libraries/cov_facies.dll"
#C_fun = CDLL(os.path.join(dirname,rel_path)) # import library


## structures --> initialization of the structures and functions with ctypes
class VarStruc2D(Structure):
    _fields_ = [("nstruc",c_int),
                ("var_type",c_int*10),
                ("rx",c_double*10),
                ("ry",c_double*10),
                ("alpha",c_double),
                ("c",c_double*10)]

class VarStruc3D(Structure):
    _fields_ = [("nstruc",c_int),
                ("var_type",c_int*10),
                ("rx",c_double*10),
                ("ry",c_double*10),
                ("rz",c_double*10),
                ("alpha",c_double),
                ("beta",c_double),
                ("gamma",c_double),
                ("c",c_double*10)]

class fa_domains(Structure):
    _fields_ = [("ncuboids",c_int),
                ("bnd_vector",c_double*100)]


## import functions (we must define ctypes for arguments and output)
"""
C_fun.P_ij.argtypes = (c_double,c_double,(VarStruc2D * 2),
                        fa_domains,fa_domains,c_double,c_double)
C_fun.P_ij.restype = c_double

C_fun.P_ij3D.argtypes = (c_double,c_double,c_double,(VarStruc3D * 2),
                        fa_domains,fa_domains,c_double,c_double)
C_fun.P_ij3D.restype = c_double
P_ij3D = C_fun.P_ij3D # store function inside a variable
"""

def geoCm2Cstruc2D(covmodel):

    """
    Pass from a geone covmodel to a C structure in ctypes that will be used in C functions
    """

    nstruc = len(covmodel.elem)

    s = VarStruc2D()
    s.nstruc = nstruc
    s.alpha = covmodel.alpha

    for i in range(nstruc):
        struc = covmodel.elem[i]
        if struc[0] == "nugget":
            vartype = -1
        elif struc[0] == "gaussian":
            vartype = 0
        elif struc[0] == "spherical":
            vartype = 1
        elif struc[0] == "exponential":
            vartype = 2
        elif struc[0] == "cubic":
            vartype = 3
        s.var_type[i] = vartype

        for k,v in struc[1].items():
            if k == "w":
                s.c[i] = v
            elif k =="r":
                s.rx[i] = v[0]
                s.ry[i] = v[1]

    return s

def Cstruc2geoCm2D(Cstruc):

    """
    Inverse of geoCm2Cstruc
    """

    nstruc = Cstruc.nstruc
    elem=[]
    for i in range(nstruc):

        struc = Cstruc.var_type
        if struc[i] == -1:
            vartype = "nugget"
        elif struc[i] == 0:
            vartype = "gaussian"
        elif struc[i] == 1:
            vartype = "spherical"
        elif struc[i] == 2:
            vartype = "exponential"
        elif struc[i] == 3:
            vartype = "cubic"

        r = [Cstruc.rx[i],Cstruc.ry[i]]
        w = Cstruc.c[i]
        if vartype != "nugget":
            elem.append((vartype,{"w":w,"r":r}))
        else:
            elem.append((vartype,{"w":w}))

    new_cm = gcm.CovModel2D(elem,alpha=Cstruc.alpha)
    return new_cm

def geoCm2Cstruc3D(covmodel):

    """
    Pass from a geone covmodel to a C structure in ctypes that will be used in C functions
    """

    nstruc = len(covmodel.elem)


    s = VarStruc3D()
    s.nstruc = nstruc
    s.alpha = covmodel.alpha
    s.beta = covmodel.beta
    s.gamma = covmodel.gamma

    for i in range(nstruc):
        struc = covmodel.elem[i]
        if struc[0] == "nugget":
            vartype = -1
        elif struc[0] == "gaussian":
            vartype = 0
        elif struc[0] == "spherical":
            vartype = 1
        elif struc[0] == "exponential":
            vartype = 2
        elif struc[0] == "cubic":
            vartype = 3
        s.var_type[i] = vartype

        for k,v in struc[1].items():
            if k == "w":
                s.c[i] = v
            elif k =="r":
                s.rx[i] = v[0]
                s.ry[i] = v[1]
                s.rz[i] = v[2]

    return s

def Cstruc2geoCm3D(Cstruc):

    """
    inverse of geoCm2Cstruc3D
    """

    nstruc = Cstruc.nstruc
    elem=[]
    for i in range(nstruc):

        struc = Cstruc.var_type
        if struc[i] == -1:
            vartype = "nugget"
        elif struc[i] == 0:
            vartype = "gaussian"
        elif struc[i] == 1:
            vartype = "spherical"
        elif struc[i] == 2:
            vartype = "exponential"
        elif struc[i] == 3:
            vartype = "cubic"

        r = [Cstruc.rx[i],Cstruc.ry[i],Cstruc.rz[i]]
        w = Cstruc.c[i]
        if vartype != "nugget":
            elem.append((vartype,{"w":w,"r":r}))
        else:
            elem.append((vartype,{"w":w}))

    new_cm = gcm.CovModel3D(elem,alpha=Cstruc.alpha,beta=Cstruc.beta,gamma=Cstruc.gamma)
    return new_cm

def flag2Cflag(flag):

    """
    From a dictionary of facies to dic of C structures for the flag of the Truncated Plurigau
    """

    dic = {}
    for k,v in flag.items():
        fa = fa_domains()
        fa.ncuboids = len(v)

        for i,d in enumerate(np.asarray(np.array(v).flatten())):
            if d == np.inf:
                d = 3
            elif d == -np.inf:
                d = -3
            fa.bnd_vector[i] = d

        dic[k] = fa
    return dic

#functions
def Ivario(x,v,icat,ncla=10,dim = 1,alpha=0.0,hmax=np.nan):

    """
    compute indicator variogram using geone variogramExp
    x : ndarray of size (n,k) coordinates of points, where n is the number of points and k is the dimension
    v : ndarray of size n, facies vector containing facies ID
    icat : int, ID facies to indicate which facies to analyze
    dim : int, wanted dimension (1,2 only)
    alpha : float, direction if dim = 2
    hmax : int or tuple of size 2 to indicate maximum distance to investigate
    """

    f = v.copy()
    f[f!=icat]=0
    f[f==icat]=1

    if dim == 1:
        return geone.covModel.variogramExp1D(x,f,ncla=ncla,hmax=hmax,make_plot=False)
    elif dim == 2:
        return geone.covModel.variogramExp2D(x,f,alpha=alpha,ncla=ncla,hmax=hmax,make_plot=False)


def zone_facies(g1v,g2v,flag_facies):

    """
    Given gaussian values for gaussian fields (g1v,g2v)
    check if the point is inside a certain facies and in which interval
    flag_facies : list of thresholds for a certain facies

    return : index position to indicate which interval of the facies or None if the point is not inside
    """

    i = -1
    for di in flag_facies:

        i += 1
        big1,bsg1 = di[0]
        big2,bsg2 = di[1]

        if (g1v >= big1) & (g1v < bsg1) & (g2v >= big2) & (g2v < bsg2):

            return i

    return None

def infacies(g1v,g2v,flag):

    """
    return the facies in which a point of coordinate (g1v, g2v) is.

    """

    for k in flag.keys():
        flag_facies = flag[k]
        v = zone_facies(g1v,g2v,flag_facies)

        if v is not None:
            return k

    return print("error point {} outside domain (flag not correctly defined)".format(g1v,g2v))


def pfa(flag):

    """
    Determine the probability of each facies according to the flag for TPGs
    """

    l = []
    for k in flag.keys():

        pf = 0
        for di in flag[k]:
            dig1 = di[0] # domain for g1
            dig2 = di[1:] # domain(s) for g2

            pfg1 = norm.cdf(dig1[1]) - norm.cdf(dig1[0])

            pfg2 = 0
            for dig2i in dig2:
                pfg2 += norm.cdf(dig2i[1]) - norm.cdf(dig2i[0])

            pf += pfg1*pfg2
        l.append(pf)
    return np.array(l)

def p1zone(flag_facies):

    """
    Calculate the probability of a facies to belongs to a certain part of the facies
    (on the flag --> facies are separated by thresholds and it is possible to have multiple zones for 1 facies)
    flag_facies : list of the thresholds for the facies
                (format : [[(x0,x1),(y0,y1)],[(x1,x3),(y1,y4)]],
                 for some thresholds xi,yi for the two gaussian fields)
    """

    proba_zones = []
    for di in flag_facies:
        dig1 = di[0] # domain for g1
        dig2 = di[1:] # domain(s) for g2

        pfg1 = norm.cdf(dig1[1]) - norm.cdf(dig1[0])

        pfg2 = 0
        for dig2i in dig2:
            pfg2 += norm.cdf(dig2i[1]) - norm.cdf(dig2i[0])
        proba_zones.append(pfg1*pfg2)

    pz = np.array(proba_zones)
    pz /= pz.sum()
    return pz

def plot_flag(flag,**kwargs):


    l= []
    for v in flag.values():
        for sb in v:
            for lim in sb:
                l.append(lim[0])
                l.append(lim[1])
    l = np.array(l)
    if np.max(l) > 1:
        space = "Gspace"
    elif (np.max(l) <= 1) & (np.min(l)>=0):
        space = "Pspace"
    else :
        raise "error flag not correctly defined, not in gaussian or proba space"


    if "alpha" not in kwargs.keys():
        kwargs["alpha"] = 1

    if space == "Gspace":
        g1 = np.arange(-3,3,0.1)
        g2 = np.arange(-3,3,0.1)

    elif space == "Pspace":
        g1 = np.arange(0,1,0.01)
        g2 = np.arange(0,1,0.01)

    FLAG = np.ones([g2.shape[0],g1.shape[0]])
    for ix,ig1 in enumerate(g1):
        for iy,ig2 in enumerate(g2):
            FLAG[iy,ix] = infacies(ig1,ig2,flag)
    plt.imshow(FLAG,cmap="plasma",alpha=kwargs["alpha"],extent=[g1[0],g1[-1],g2[0],g2[-1]],origin="lower")
    plt.colorbar()

def Gspace2Pspace(flag):

    """
    Return a flag in the probability space given a flag in gaussian space
    """

    d = {}
    for k in flag:
        l = []
        for isb,sb in enumerate(flag[k]):
            lsb = []
            for ilim,lim in enumerate(sb):
                lsb.append((norm.cdf(lim[0]),norm.cdf(lim[1])))
            l.append(lsb)
        d[k] = l

    return d

def Truncation2D(nx,ny,sims,flag):

    """
    Function that operates the truncation process for TPGs

    """

    nk = len(flag.keys())

    facies_position = np.ones([nk,sims[0].shape[0],ny,nx])
    for ik,k in enumerate(flag.keys()):
        t = np.zeros(sims[0].shape)
        for di in flag[k]:
            big1,bsg1 = di[0]

            for dig2i in di[1:]:
                trunc = np.ones(sims[0].shape)
                big2,bsg2 = dig2i
                trunc *= ((sims[0] > big1) & (sims[0] < bsg1) & (sims[1] > big2) & (sims[1] < bsg2))

            t += trunc

        facies_position[ik] = t*k

    return facies_position.sum(0)


def Truncation3D(nx,ny,nz,sims,flag):

    """
    Function that operates the truncation process for TPGs in 3D
    nx, ny, nz : number of cell in x,y and z direction
    sims : ndarray of size (nsim,nz,ny,nx) where nsim is the number of realizations
    flag   : dictionnary containing for each facies a list of the thresholds (in gaussian space) for the two gaussian fields
             exemple with 3 facies of ID : (1,2,3) :
           {1: [[(-inf, -0.3), (-inf, 0)], [(0.3, inf), (-inf, 0.5)]],
            2: [[(-inf, -0.3), (0, inf)]],
            3: [[(-0.3, 0.3), (-inf, inf)], [(0.3, inf), (0.5, inf)]]}.
    """

    nk = len(flag.keys())
    facies_position = np.ones([nk,sims[0].shape[0],nz,ny,nx])
    for ik,k in enumerate(flag.keys()):
        t = np.zeros(sims[0].shape)
        for di in flag[k]:
            big1,bsg1 = di[0]

            for dig2i in di[1:]:
                trunc = np.ones(sims[0].shape)
                big2,bsg2 = dig2i
                trunc *= ((sims[0] > big1) & (sims[0] < bsg1) & (sims[1] > big2) & (sims[1] < bsg2))

            t += trunc

        facies_position[ik] = t*k

    return facies_position.sum(0)


##### OPTIMIZATION #####
def opti_vario(IK_covmodels,covmodels_to_fit,pk,flag,n=8,du=0.03,dv=0.03,c_reg=0.001,print_infos = False,
                min_method = "Nelder-mead",ftol = 0.2,xtol = 0.2):

    """
    Optimize parameters of the two covmodels (covmodels_to_fit) of the tpgs
    in order to reproduce the IK_covmodels using a least-square method
    Very slow --> To improve

    ## inputs ##
    IK_covmodels : list of k indicator covmodels (geone.covModel) where k is the number of facies
                   order in the list must be the same than the keys of the flag
    covmodels_to_fit : list of 2 covmodels (geone.covModel) to infer. Parameters to infer should be specified with a string.
                --> G1_to_opt = gcm.CovModel3D(elem=[("gaussian",{"w":1,"r":["rx1","rx1","rz1"]})],
                                                alpha="alpha",beta=0,gamma=0)
                    G2_to_opt = gcm.CovModel3D(elem=[("gaussian",{"w":1,"r":["rx2","rx2","rz2"]})],
                                                alpha="alpha2",beta="beta2",gamma=0)
                    covmodels_to_fit = [G1_to_opt,G2_to_opt]
    pk : array-like of proportion for each facies (order in the array same that IK_covmodels)
    flag : dictionnary containing for each facies a list of the thresholds (in gaussian space) for the two gaussian fields
             exemple with 3 facies of ID : (1,2,3) :
           {1: [[(-inf, -0.3), (-inf, 0)], [(0.3, inf), (-inf, 0.5)]],
            2: [[(-inf, -0.3), (0, inf)]],
            3: [[(-0.3, 0.3), (-inf, inf)], [(0.3, inf), (0.5, inf)]]}
    n : int, number of points to use for inversion along each axis (3*n points will be used for each misfit calculation)
    du,dv : float, precision to use in the calcul of the probability btw facies, values of 0.05 are generally enough.
    c_reg : regularization coefficient to apply on radius parameters (radius of covmodels only), problem dependent.
    print_infos : bool, print misfit and regularization objective functions each iteration
    min_method : method to use for the minimization with minimize (only Nelder-mead is available)
    ftol, xtol : tolerance ratio for convergence of objective function and parameter resp.

    ## outputs ##
    results of the minimization.
    covmodels_to_fit has also been updated (by reference) with best parameters
    """

    C_flag = flag2Cflag(flag) # get C_flag

    def chk_par(par,n_p,typ="radius"): # to check if a parameter is already known or not and add it to lists
        if isinstance(par,str) and (par not in d_pars.keys()):
            if typ == "angle":
                x.append(45)
            elif typ == "radius":
                x.append(mea_r)
            elif typ == "c":
                x.append(1)
            d_pars[par] = n_p
            d_par_type[par] = typ
            n_p += 1

        return n_p

    l = []
    for cm in IK_covmodels:
        for el in cm.elem:
            if "r" in el[1].keys():
                l.append(np.mean(el[1]["r"]))

    ranges = np.asarray(l)
    mea_r = np.mean(ranges)
    max_r = np.max(ranges)

    #initial values and create dictionnary of parameters
    d_pars = {}
    d_par_type = {}
    x = []
    n_p = 0

    for g in covmodels_to_fit:
        n_p = chk_par(g.alpha,n_p,typ="angle")
        n_p = chk_par(g.beta,n_p,typ="angle")
        n_p = chk_par(g.gamma,n_p,typ="angle")

        for i in range(len(g.elem)):
            n_p = chk_par(g.elem[i][1]["r"][0],n_p,typ="radius")
            n_p = chk_par(g.elem[i][1]["r"][1],n_p,typ="radius")
            n_p = chk_par(g.elem[i][1]["r"][2],n_p,typ="radius")

            n_p = chk_par(g.elem[i][1]["w"],n_p,typ="c")

    x = np.array(x) # transform into array

    G1_t = copy.deepcopy(covmodels_to_fit[0])
    G2_t = copy.deepcopy(covmodels_to_fit[1])
    cov_model_to_fit_template = [G1_t,G2_t]

    def up_par(par_t,x):

        """
        Update parameter par with string par_t
        """

        if isinstance(par_t,str):
            par = x[d_pars[par_t]]
        else:
            par = par_t
        return par

    def misfit_3d(x,c_reg):

        #update parameters
        for g_t,g in zip(cov_model_to_fit_template,covmodels_to_fit):
            g.alpha = up_par(g_t.alpha,x)
            g.beta = up_par(g_t.beta,x)
            g.gamma = up_par(g_t.gamma,x)

            for i in range(len(g_t.elem)): # loop over structures
                if "r" in g.elem[i][1].keys():
                    g.elem[i][1]["r"][0] = up_par(g_t.elem[i][1]["r"][0],x) # rx
                    g.elem[i][1]["r"][1] = up_par(g_t.elem[i][1]["r"][1],x) # ry
                    g.elem[i][1]["r"][2] = up_par(g_t.elem[i][1]["r"][2],x) # rz

                g.elem[i][1]["w"] = up_par(g_t.elem[i][1]["w"],x)

        # convert covmodels to C structure
        G1 = geoCm2Cstruc3D(covmodels_to_fit[0])
        G2 = geoCm2Cstruc3D(covmodels_to_fit[1])
        Gk = (VarStruc3D * 2)()
        Gk[0] = G1
        Gk[1] = G2

        # misfit calculation
        misfit=0
        for i,ifacies in enumerate(C_flag.keys()): # iteration through facies

            #corr = 0 - (P_ij3D(max_r*10,0,0,Gk,C_flag[ifacies],C_flag[ifacies],du,dv)- \
            #            pk[i]*pk[i]) # correction coeff to correct deviation from integral
            corr = 0
            k_covmodel = IK_covmodels[i] # retrieve experimental indicator covmodel
            k_func = k_covmodel.vario_func() # vario func

            # determine hx, hy and hz #
            rx_1,ry_1,rz_1 =  k_covmodel.mrot()[:,0]*k_covmodel.r123()[0]
            rx_2,ry_2,rz_2 =  k_covmodel.mrot()[:,1]*k_covmodel.r123()[1]
            rx_3,ry_3,rz_3 =  k_covmodel.mrot()[:,2]*k_covmodel.r123()[2]
            hx_1,hy_1,hz_1 = np.linspace(du,rx_1,n),np.linspace(du,ry_1,n),np.linspace(du,rz_1,n)
            hx_2,hy_2,hz_2 = np.linspace(du,rx_2,n),np.linspace(du,ry_2,n),np.linspace(du,rz_2,n)
            hx_3,hy_3,hz_3 = np.linspace(du,rx_3,n),np.linspace(du,ry_3,n),np.linspace(du,rz_3,n)
            hx = np.concatenate((hx_1,hx_2,hx_3))
            hy = np.concatenate((hy_1,hy_2,hy_3))
            hz = np.concatenate((hz_1,hz_2,hz_3))

            start = time.time()
            var=[]
            cov0 = pk[i]*(1-pk[i])
            for ihx,ihy,ihz in zip(hx,hy,hz): # along new axis according to covmodel
                #covii = P_ij3D(ihx,ihy,ihz,Gk,C_flag[ifacies],C_flag[ifacies],du,dv)-pk[i]*pk[i]+corr
                covii = corr
                vh = cov0 - covii
                var.append(vh)
            t1 = time.time()
            #print(t1-start)
            #Misfit update
            real_var = k_func(np.array((hx,hy,hz)).T) # variance from indicator variograms
            var = np.array(var) # variance computed by TPGs
            misfit += np.sum((var - real_var)**2)


        #reg term for radius
        #TO IMPROVE
        reg = 0
        for p in d_pars.keys():
            if d_par_type[p] == "radius":
                reg += (mea_r - x[d_pars[p]])**2

        reg *= c_reg
        if print_infos:
            print("misfit : {}".format(misfit))
            print("reg : {}".format(reg))
        return misfit+reg

    if min_method == "Nelder-mead":
        res = minimize(misfit_3d,x,args=c_reg,method="Nelder-mead",options={'xatol': xtol, 'fatol': ftol})
    return res

def coord2cell(xg,yg,zg,x,y,z):

    sx = np.diff(xg)[0]
    sy = np.diff(yg)[0]
    sz = np.diff(zg)[0]
    # check point inside simulation block
    if (x <= xg[0]) or (x >= xg[-1]):
        if self.verbose:
            print("point outside of the grid in x")
        return None
    if (y <= yg[0]) or (y >= yg[-1]):
        if self.verbose:
            print("point outside of the grid in y")
        return None
    if (z <= zg[0]) or (z >= zg[-1]):
        if self.verbose:
            print("point outside of the grid in z")
        return None

    ix = ((x-xg[0])//sx).astype(int)
    iy = ((y-yg[0])//sy).astype(int)
    iz = ((z-zg[0])//sz).astype(int)

    cell = (iz,iy,ix)
    return cell


# RUN SIM #
def run_tpgs(nsim,xg,yg,zg,data,Gk,flag,nit=100,nmax = 24,grf_method="fft",mask=None):

    """
    Run simulations using the Truncated plurigaussian (2 gaussian fields) methods : Covmodels must be provided !
    nsim   : number of realizations
    xg,yg,zg : 1D vector of edges coordinates
    data   :(x,y,z,g1,g2,v), where x,y,z are the cartesian coordinates,
             g1 and g2 are the values of first/second gaussian fields and v is the facies value
    Gk     : list of 2 3D covmodels (geone object)
    flag   : dictionnary containing for each facies a list of the thresholds (in gaussian space) for the two gaussian fields
             exemple with 3 facies :
           {1: [[(-inf, -0.3), (-inf, 0)], [(0.3, inf), (-inf, 0.5)]],
            2: [[(-inf, -0.3), (0, inf)]],
            3: [[(-0.3, 0.3), (-inf, inf)], [(0.3, inf), (0.5, inf)]]}
    grf_method :  string, geostatistical method to realize gaussin fields (fft or sgs)
    mask : bool, to delimit where to simulate (no computational effect if grf_method is fft)
    """

    ## grid parameters
    nx = len(xg)-1
    ny = len(yg)-1
    nz = len(zg)-1
    dimensions = (nx,ny,nz)
    ox = np.min(xg[0])
    oy = np.min(yg[0])
    oz = np.min(zg[0])
    origin = (ox,oy,oz)
    sx = np.diff(xg)[0]
    sy = np.diff(yg)[0]
    sz = np.diff(zg)[0]
    spacing = (sx,sy,sz)

    G1,G2 = Gk
    output = np.ones([nsim,nz,ny,nx])

    if data is not None:  # conditional simulation

        #keep data inside mask
        if mask is not None:
            new_data = []
            for d in data:
                if mask[coord2cell(xg,yg,zg,d[0],d[1],d[2])]:
                    new_data.append(d)
            data = np.array(new_data)

        ## initial values ##
        for idata in data:

            flag_f = flag[idata[5]]
            p = p1zone(flag_f)
            zc = np.random.choice(range(p.shape[0]),p=p) # zone choice

            m = 0
            s = 1

            #G1
            big1,bsg1 = flag_f[zc][0]

            myclip_a = big1
            myclip_b = bsg1
            val = np.random.rand(1)*myclip_b + myclip_a
            a, b = (myclip_a - m) / s, (myclip_b - m) / s
            val = truncnorm.rvs(a,b,loc=m,scale=s) # draw initial value
            idata[3] = val

            #G2
            big2,bsg2 = flag_f[zc][1]

            myclip_a = big2
            myclip_b = bsg2
            a, b = (myclip_a - m) / s, (myclip_b - m) / s
            val = truncnorm.rvs(a,b,loc=m,scale=s) # draw initial value
            idata[4] = val

        ## weights
        tree = KDTree(data[:,:3])
        ndata_p = data.shape[0]
        if nmax > ndata_p:
            nmax = ndata_p

        weight_arr = np.zeros([2,ndata_p,ndata_p])
        std_arr = np.zeros([2,ndata_p])
        for i in range(ndata_p):
            # G1
            xu = data[i]
            idx = tree.query(xu[:3].reshape(1,-1),k=nmax,return_distance=False)[0] # search nearest neighbours
            x = data[idx[1:]] # select nearest neig

            covmodel = G1
            mean = np.mean(x[:,3])
            w,vu_std = simple_kriging(x[:,:3],xu[0:3].reshape(-1,3),covmodel,mean=mean)
            weight_arr[0,i,idx[1:]] = w
            std_arr[0,i] = vu_std

            #G2
            covmodel = G2
            mean = np.mean(x[:,4])
            w,vu_std = simple_kriging(x[:,:3],xu[0:3].reshape(-1,3),covmodel,mean=mean)
            weight_arr[1,i,idx[1:]] = w
            std_arr[1,i] = vu_std

        #gibbs sampler
        for ireal in range(nsim):
            for it in range(nit):
                for i in range(data.shape[0]):
                    xu = data[i]

                    #G1
                    mean = np.mean(data[:,3])
                    m = np.dot(weight_arr[0,i],data[:,3]) + (1-np.sum(weight_arr[0,i],axis=0))*mean
                    s = std_arr[0,i]
                    ## draw
                    g1val = m+s*norm.rvs()

                    #G2
                    mean = np.mean(data[:,4])
                    m = np.dot(weight_arr[1,i],data[:,4]) + (1-np.sum(weight_arr[1,i],axis=0))*mean
                    s = std_arr[1,i]

                    ## draw
                    g2val = m+s*norm.rvs()

                    ## check point inside zone ##
                    chk = zone_facies(g1val,g2val,flag[xu[-1]])
                    if chk is not None:
                        xu[3] = g1val
                        xu[4] = g2val

            if grf_method == "fft":
                G1sim = grf.grf3D(G1,dimensions,spacing,origin,x = data[:,:3],v = data[:,3],nreal=1,printInfo=False)
                G2sim = grf.grf3D(G2,dimensions,spacing,origin,x = data[:,:3],v = data[:,4],nreal=1,printInfo=False)
            elif grf_method == "sgs":
                G1sim = gci.simulate3D(G1,dimensions,spacing,origin,x = data[:,:3],v = data[:,3], nreal=1, verbose=0,mask=mask)["image"].val
                G2sim = gci.simulate3D(G2,dimensions,spacing,origin,x = data[:,:3],v = data[:,4], nreal=1, verbose=0,mask=mask)["image"].val

            sims = [G1sim,G2sim]
            output[ireal] = Truncation3D(nx,ny,nz,sims,flag)


    elif data is None:# unco
        if grf_method == "fft":
            G1sim = grf.grf3D(G1,dimensions,spacing,origin,nreal=nsim,printInfo=False)
            G2sim = grf.grf3D(G2,dimensions,spacing,origin,nreal=nsim,printInfo=False)
        elif grf_method == "sgs":
            G1sim = gci.simulate3D(G1,dimensions,spacing,origin, nreal = nsim, verbose=0,mask=mask)["image"].val
            G2sim = gci.simulate3D(G2,dimensions,spacing,origin, nreal = nsim, verbose=0,mask=mask)["image"].val

        sims = [G1sim,G2sim]
        output = Truncation3D(nx,ny,nz,sims,flag)

    return output





    ## kriging functions  ##
def simple_kriging(x, xu, cov_model, mean):
    """
    Simple kriging - interpolates at locations xu the values v measured at locations x.
    Covariance model given should be:
        - in same dimension as dimension of locations x, xu
        - in 1D, it is then used as an omni-directional covariance model
    (see below).

    :param x:       (2-dimensional array of shape (n, d)) coordinates
                        of the data points (n: number of points, d: dimension)
                        Note: for data in 1D, it can be a 1-dimensional array of shape (n,)
    :param xu:      (2-dimensional array of shape (nu, d)) coordinates
                        of the points where the interpolation has to be done
                        (nu: number of points, d: dimension same as for x),
                        called unknown points
                        Note: for data in 1D, it can be a 1-dimensional array of shape (nu,)

    :param cov_model:   covariance model:
                            - in same dimension as dimension of points (d), i.e.:
                                - CovModel1D class if data in 1D (d=1)
                                - CovModel2D class if data in 2D (d=2)
                                - CovModel3D class if data in 3D (d=3)
                            - or CovModel1D whatever dimension of points (d):
                                - used as an omni-directional covariance model

    :return:        (w, vu_std) with:
                        w:     (1-dimensional array of shape (x,)) weights at position xu
                        vu_std: (1-dimensional array of shape (nu,)) kriged standard deviation at points xu
    """
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
        return (None, None)

    # Check dimension of cov_model and set if used as omni-directional model
    if cov_model.__class__.__name__ != 'CovModel{}D'.format(d):
        if cov_model.__class__.__name__ == 'CovModel1D':
            omni_dir = True
        else:
            print("ERROR: 'cov_model' is incompatible with dimension of points")
            return (None, None)
    else:
        omni_dir = False

    # Number of data points
    n = x.shape[0]
    # Number of unknown points
    nu = xu.shape[0]

    # Covariance function
    cov_func = cov_model.func() # covariance function
    if omni_dir:
        # covariance model in 1D is used
        cov0 = cov_func(0.) # covariance function at origin (lag=0)
    else:
        cov0 = cov_func(np.zeros(d)) # covariance function at origin (lag=0)

    # Fill matrix of ordinary kriging system (matOK)
    nSK = n # order of the matrix
    matSK = np.ones((nSK, nSK))
    for i in range(n):
        # lag between x[i] and x[j], j=i+1, ..., n-1
        h = x[(i+1):] - x[i]
        if omni_dir:
            # compute norm of lag
            h = np.sqrt(np.sum(h**2, axis=1))
        cov_h = cov_func(h)
        matSK[i, (i+1):] = cov_h
        matSK[(i+1):, i] = cov_h
        matSK[i,i] = cov0

    # Right hand side of the ordinary kriging system (b):
    #   b is a matrix of dimension nOK x nu
    b = np.ones((nSK, nu))
        # lag between x[i] and every xu
    h = xu - x
    if omni_dir:
        # compute norm of lag
        h = np.sqrt(np.sum(h**2, axis=1))
    b = cov_func(h)

    # Solve the kriging system
    w = np.linalg.solve(matSK,b) # w: matrix of dimension nSK x nu

    # Kriged values at unknown points
    #vu = v.dot(w) + (1-np.sum(w,axis=0))*mean

     # Kriged standard deviation at unknown points
    vu_std = np.sqrt(np.maximum(0, cov0 - np.array([w.dot(b)])))
    return w,vu_std
