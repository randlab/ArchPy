"""
This module contains various functions to infer parameters of archpy model. For now, only surface parameters can be inferred.
"""

import numpy as np
import matplotlib.pyplot as plt
import geone
import geone.covModel as gcm
import sys
import ipywidgets as widgets
from ipywidgets import interact, fixed, Button
from IPython.display import display

#my modules
import ArchPy

# functions
def cm_any_nan(cm):
    
    """
    Detect if there is a nan inside a covmodel

    Parameters
    ----------
    cm: :class:`geone.covModel.CovModel1D`, :class:`geone.covModel.CovModel2D` or :class:`geone.covModel.CovModel3D`
        covmodel to check
    """
    
    yes = False

    if isinstance(cm, geone.covModel.CovModel2D):
        dim = 2
    if isinstance(cm, geone.covModel.CovModel1D):
        dim = 1
    if isinstance(cm, geone.covModel.CovModel3D):
        dim = 3

    elems = [i[1] for i in cm.elem]
    for el in elems:
        for k,v in el.items():
            if k in ("w", "nu", "s"):
                if np.isnan(v):
                    yes = True
                    break
            if k == "r":
                if dim > 1:
                    for ri in v:
                        if np.isnan(ri):
                            yes = True
                            break
                            
                            
def infer_surface(ArchTable, unit, hmax=np.nan, cm_to_fit=None, auto=True, dim=1, plot=True,
                  npoints_min=20, max_nugget=1, bounds = None, default_covmodel=None, vb=1, **kwargs):
    
    """
    Infer surface parameters of a unit in an :class:`base.Arch_table`. Fitting can be done automatically or manually.

    Parameters
    ----------
    ArchTable: :class:`base.Arch_table`
        ArchTable to infer surface parameters
    unit: :class:`base.Unit`
        unit to infer surface parameters
    hmax: float, optional  
        maximum distance to consider for variogram inference
    cm_to_fit: :class:`geone.covModel.CovModel1D`, :class:`geone.covModel.CovModel2D` or :class:`geone.covModel.CovModel3D`, optional
        covmodel to fit
    auto: bool, optional
        if True, automatic inference of the covmodel
    dim: int, optional
        dimension of the covmodel to fit
    plot: bool, optional
        if True, plot the variogram
    npoints_min: int, optional
        minimum number of points to infer the covmodel
    max_nugget: float, optional
        maximum nugget effect to infer
    bounds: list, optional  
        bounds for the parameters of the covmodel to fit
        list containing two lists, one for the lower bounds and one for the upper bounds of each parameter

    Returns
    -------
    cm: :class:`geone.covModel.CovModel1D`, :class:`geone.covModel.CovModel2D` or :class:`geone.covModel.CovModel3D`
        inferred covmodel
    """
    
    if plot:
        plt.cla()
        plt.close()
        global fig, ax  # global figure to allow class to access to the plot
        fig,ax=plt.subplots()
        
    surf = unit.surface

    x = np.array([surf.x, surf.y]).T
    v = np.array(surf.z)
    Lz = ArchTable.get_zg()[-1] - ArchTable.get_zg()[0]
    Lx = ArchTable.get_xg()[-1] - ArchTable.get_xg()[0]
    Ly = ArchTable.get_yg()[-1] - ArchTable.get_yg()[0]
    
    dmax = np.sqrt(Lx**2 + Ly**2)

    if len(surf.z) > npoints_min:
        
        if auto:  # automatic inference
            
            if dim == 1:
                
                # compute var exp
                h,exp_var, p = gcm.variogramExp1D(x, v, hmax=hmax, ncla=20, make_plot = False, **kwargs)  # just make a plot of the exp variogram

                if plot:
                    plt.scatter(h, exp_var)
                
                # boundaries
                vmax = np.var(v)*2
                rmax = dmax*2                
                
                # covmodel to fit
                if cm_to_fit is None:
                    if cm_any_nan(surf.covmodel):  # check if there is any nan inside cm
                        cm_to_fit = surf.covmodel  # if so it will be passed to fit function
                    else:
                        # default covmodel to fit
                        cm_to_fit = gcm.CovModel1D(elem=[("cubic",{"w":np.nan,"r":np.nan}),
                                                         ("exponential",{"w":np.nan,"r":np.nan}),
                                                         ("spherical",{"w":np.nan,"r":np.nan}),
                                                         ("nugget",{"w":np.nan})])
                        
                        bounds = ((0, 0, 0, 0, 0, 0, 0),  # min bounds
                                  (np.var(v), dmax*2, np.var(v), dmax*2, np.var(v), dmax*2, max_nugget)) # max bounds
                     
                else:
                    # ensure that bounds exist
                    if bounds is None:
                        bmin = []
                        bmax = []

                        for el in cm_to_fit.elem:
                            if el[0] is not "nugget":
                                for k in el[1].keys():
                                    if k == "w":
                                        bmin.append(0)
                                        bmax.append(vmax)
                                    elif k == "r":
                                        bmin.append(0)
                                        bmax.append(rmax)
                            else:
                                bmin.append(0)
                                bmax.append(max_nugget)
                        bounds = (bmin, bmax)
                
                # automatic fitting
                cm_fitted = gcm.covModel1D_fit(x, v, cm_to_fit, 
                                               hmax=hmax, bounds=bounds, make_plot=False)[0]
                
                
                #TO DO : add a check to see if fitting is good enough

                #remove unutilized structures
                l = cm_fitted.elem.copy()
                sum_w = 0
                for e in l:
                    if e[1]["w"] > sum_w:
                        sum_w += e[1]["w"]

                for e in l:
                    if (e[1]["w"]/sum_w) < 0.01:
                        cm_fitted.elem.remove(e)    

                # finalization                              
                surf.set_covmodel(gcm.covModel1D_to_covModel2D(cm_fitted))
                if plot:
                    plt.title(unit.name)
                    cm_fitted.plot_model(vario=True, c="k")
                    if ~np.isnan(hmax):
                        plt.xlim(-hmax*0.1, hmax)
                    plt.xlabel("h [L]")
                    plt.legend() 
                    plt.show() 
                

            elif dim == 2: 
                
                # boundaries
                vmax = np.var(v)*2
                rmax = dmax*2  
                
                # covmodel to fit
                if cm_to_fit is None:
                    if cm_any_nan(surf.covmodel):  # check if there is any nan inside cm
                        cm_to_fit = surf.covmodel  # if so it will be passed to fit function
                    else:
                        # default covmodel to fit
                        cm_to_fit = gcm.CovModel2D(elem=[("cubic",{"w":np.nan,"r":[np.nan, np.nan]}),
                                                         ("exponential",{"w":np.nan,"r":[np.nan, np.nan]}),
                                                         ("spherical",{"w":np.nan,"r":[np.nan, np.nan]}),
                                                         ("nugget",{"w":np.nan})], alpha=np.nan)
                        
                        bounds = ((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -45),  # min bounds
                                  (vmax, rmax, rmax,
                                   vmax, rmax, rmax,
                                   vmax, rmax, rmax,
                                   max_nugget,
                                   45)) # max bounds
                     
                else:
                    # ensure that bounds exist
                    if bounds is None:
                        bmin = []
                        bmax = []

                        for el in cm_to_fit.elem:
                            if el[0] is not "nugget":
                                for k in el[1].keys():
                                    if k == "w":
                                        bmin.append(0)
                                        bmax.append(vmax)
                                    elif k == "r":
                                        for o in range(2):
                                            bmin.append(0)
                                            bmax.append(rmax)
                            else:
                                bmin.append(0)
                                bmax.append(max_nugget)
                        
                        # alpha
                        bmin.append(-45)
                        bmax.append(45)
                        bounds = (bmin, bmax)
                
                # automatic fitting
                cm_fitted = gcm.covModel2D_fit(x, v, cm_to_fit, 
                                               hmax=hmax, bounds=bounds, make_plot=False)[0]
                
                (h1, v1, p1), (h2, v2, p2) = gcm.variogramExp2D(x, v, cm_fitted.alpha, hmax=(hmax, hmax), 
                                                                make_plot=False, **kwargs)
                
                if plot:
                    plt.scatter(h1, v1, color="lightblue", label="x : var-exp")
                    plt.scatter(h2, v2, color="orange", label="y : var-exp")
                    
                    plt.title(r"{} : $\alpha$ {}".format(unit.name, np.round(cm_fitted.alpha, 2)))
                    
                    cm_fitted.plot_model_one_curve(1, 1, hmax=2*dmax, c="lightblue", label= "x' model")
                    cm_fitted.plot_model_one_curve(2, 1, hmax=2*dmax, c="orange", label="y' model")
                
                    if ~np.isnan(hmax):
                        plt.xlim(-hmax*0.1, hmax)
                    plt.xlabel("h [L]")
                    plt.legend()  
                    
                # set model
                surf.set_covmodel(cm_fitted)
                
        else : # manual fitting
            
            #create a continue Button
            button = Button(description="Continue")  # continue button
            output = widgets.Output()
            display(button, output)
            
            # function to separate exp var and fitting
            def on_button_clicked(b):
                global i
                with output:
                    if i == 0:
                        plt.grid()
                        dim = ev.dim  # which dimension ? 
                        hmax = ev.hmax
                        cm_to_fit.h_max = np.max(hmax)
                        cm_to_fit.alpha = ev.alpha
                        ev.clear(all=True)
                        cm_to_fit.fit(dim=dim)

                        plt.xlim(0, right = 1.1*np.nanmax(ev.h))
                        plt.ylim(0, top = 1.1*np.nanmax(ev.val))
                    elif i > 0:
                        print("Finished and variogram saved")
                        cm_to_fit.clear(all=True)
                        button.close()
                        if isinstance(cm_to_fit.cm, gcm.CovModel1D):
                            surf.set_covmodel(gcm.covModel1D_to_covModel2D(cm_to_fit.cm))
                        else:
                            surf.set_covmodel(cm_to_fit.cm)
                    i+=1
            
            
            ev = Var_exp(x, v, hmax_lim=dmax*2)  # object for experimental variogram setting
            ev.fit()
            if np.isnan(hmax):
                cm_to_fit = Cm2fit(h_max=np.max((Lx*2, Ly*2, Lz*2)), r_max=2*np.max((Lx*2, Ly*2, Lz*2)), w_max=2*np.var(v))  # object for fitting a covmodel object (see geone doc.)
            else:
                cm_to_fit = Cm2fit(h_max=hmax, r_max=2*np.max((Lx*2, Ly*2, Lz*2)), w_max=2*np.var(v))  # object for fitting a covmodel object (see geone doc.)
            global i
            i = 0
            button.on_click(on_button_clicked)
            
            # display(button)  # Important !! --> you have to use display to show the button
            
    else :  # not enough data for a variogram

        if vb:
            print("Not enough data points")

        if not auto:
            def on_button_clicked(b):
                # create default

                if default_covmodel is not None:
                    surf.set_covmodel(default_covmodel)
                else:
                    cm_def = gcm.CovModel2D(elem=[("exponential", {"w":(Lz/4)**2/(128), "r":[min(Lx, Ly)/4, min(Lx,Ly)/4]})])
                    surf.set_covmodel(cm_def)

            button = Button(description="add default covmodel")  # continue button
            display(button)
            button.on_click(on_button_clicked)

        else:
            if vb:
                print("Default covmodel added")
            if default_covmodel is not None:
                surf.set_covmodel(default_covmodel)
            else:
                if surf.covmodel is None:
                    cm_def = gcm.CovModel2D(elem=[("exponential", {"w":(Lz/4)**2/(128), "r":[min(Lx, Ly)/4, min(Lx,Ly)/4]})])
                    surf.set_covmodel(cm_def)
                else:
                    if vb:
                        print("Nothing has changed")

    
                                               
def fit_surfaces(self, default_covmodel=None, **kwargs):

    """
    Function to fit a covmodel to each surface of an :class:`Arch_table` object

    Parameters
    ----------
    self : :class:`base.Arch_table` object
        Arch_table object to fit
    default_covmodel : :class:`geone.CovModel2D` object, optional
        Default covmodel to add if not enough data for a variogram
    **kwargs :
        Arguments for :func:`infer_surface`. See :func:`infer_surface` for more details
    """

    print(default_covmodel)
    # create a nested function to set default argument (ArchTable and surface)
    def f(unit, auto, default_covmodel, **kwargs):
        unit = self.get_unit(unit)
        def _f():
            infer_surface(self, unit, auto=auto, default_covmodel=default_covmodel, **kwargs)
        widgets.interact(_f, **kwargs)
        
    units_n = [i.name for i in self.get_all_units()]

    widgets.interact(f, unit=units_n, auto=False, dim=[1, 2], default_covmodel=fixed(default_covmodel), **kwargs)
    
# Classes

# var exp
## Class for estimate var exp
def plot_var_exp(h1, h2, v1 ,v2, p1, p2, print_pairs=False, ax=None):
    
    """
    Function to plot experimental variogram

    Parameters
    ----------
    h1 : array
        spacing lag between points in x direction
    h2 : array
        spacing lag between points in y direction
    v1 : array
        variance between points in x direction
    v2 : array
        variance between points in y direction
    p1 : array
        number of pairs between points in x direction
    p2 : array
        number of pairs between points in y direction
    print_pairs : bool, optional
        print number of pairs on the plot. The default is False.
    """
    
    if ax is None:
        axi = plt.gca()
    else:
        axi = ax

    axi.scatter(h1, v1, label="x'")
    axi.scatter(h2, v2, label="y'")
    axi.plot(h1, v1)
    axi.plot(h2, v2)   
    
    if print_pairs:
        for i in range(len(h1)):
            axi.text(h1[i],v1[i],p1[i]) 
        for i in range(len(h2)):
            axi.text(h2[i], v2[i], p2[i]) 
    
    axi.legend()
    axi.grid()
    
class Var_exp():
    
    """
    Class to estimate experimental variogram

    Parameters
    ----------
    x : array
        x coordinates of the points. size = (n, dim) with dim = 1, 2 or 3
    v : array
        values
    hmax_lim : float
        maximum distance to consider for the variogram
    ax : matplotlib axis
        axis to plot the variogram
    """

    def __init__(self, x, v, hmax_lim=1000, ax=None):
        
        self.x = np.array(x)
        self.v = np.array(v)
        self.h = None # results
        self.val = None # results
        self.hmax_lim = hmax_lim
        self.dim = 1
        self.hmax = 0
        self.alpha = 0
        self.ax = ax
        self.w = []
        self.w2 = []
        
    # fit function
    def fit(self, **kwargs):
        
        """
        Function to fit the experimental variogram

        Parameters
        ----------
        kwargs : dict
            dictionnary of parameters to pass to make_exp_var
        """

        if self.ax is None:
            axi = plt.gca()
        else:
            axi = self.ax
        [l.remove() for l in axi.lines]

        # print("General parameters - Do not touch except dim if you want to specify the dimension")
        # w=widgets.interact(self.make_exp_var, dim=(1, 3), **kwargs)
        # self.w = w

        def on_change(change_dim):

            self.clear()
            
            self.make_exp_var(change_dim["new"])

        change_dim = widgets.ToggleButtons(options=[1, 2], description="Dimension")
        change_dim.observe(on_change, names='value')
        self.w2.append(change_dim)
        display(change_dim)

    
    def clear(self, all=False):

        """
        :meta_private:
        """
        for w in self.w:
            w.widget.close()
        if all:
            for w in self.w2:
                w.close()
    
    
    # functions
    def make_exp_var(self, dim=1, **kwargs):

        """
        Function to estimate the experimental variogram

        Parameters
        ----------
        dim : int
            dimension of the problem
        kwargs : dict
            dictionnary of parameters to pass to make_exp_var_1D, make_exp_var_2D or make_exp_var_3D
        """

        x = self.x
        v = self.v
        self.dim = dim  # store dimension
        if dim == 1:
            wid = widgets.interact(self.make_exp_var_1D, x=fixed(x), v=fixed(v),
                             ncla=(1, 200, 5), hmax=(0, self.hmax_lim, 1), **kwargs)
            self.w.append(wid)
            
        elif dim == 2:
            wid = widgets.interact(self.make_exp_var_2D, x=fixed(x), v=fixed(v),
                     ncla=(1, 50, 1), 
                     hmax=(0, self.hmax_lim, 1),
                     alpha=(-90, 90, 5),
                     make_plot=fixed(False), **kwargs)
            self.w.append(wid)
            
        elif dim == 3:
            # TO DO
            pass

    def make_exp_var_1D(self, x, v, hmax, ncla=10, **kwargs):

        """
        Function to estimate the experimental variogram in 1D

        Parameters
        ----------

        x : array
            x coordinates of the points. size = (n, dim) with dim = 1, 2 or 3
        v : array
            values
        hmax : float
            maximum distance to consider for the variogram
        ncla : int
            number of classes to consider
        kwargs : dict
            dictionnary of parameters to pass to geone.covModel.variogramExp1D
        """

        plt.cla()
        
        if hmax <= 0:
            hmax = np.nan

        h,v,p = gcm.variogramExp1D(x, v, hmax=hmax, ncla=ncla, make_plot=False, **kwargs)
        
        self.h = h
        self.val = v
        self.hmax = hmax
        plt.scatter(h, v, label="var_exp")
        plt.plot(h, v)
        #plt.bar(h, p, alpha=0.3, width=max(h)/len(h))
        plt.legend()
        plt.grid()
        
    def make_exp_var_2D(self, x, v, hmax, ncla, alpha=0, **kwargs):
        
        """
        Function to estimate the experimental variogram in 2D

        Parameters
        ----------
        x : array
            x coordinates of the points. size = (n, dim) with dim = 1, 2 or 3
        v : array
            values
        hmax_x : float
            maximum distance to consider for the variogram in x direction
        hmax_y : float
            maximum distance to consider for the variogram in y direction
        ncla_x : int
            number of classes to consider in x direction
        ncla_y : int   
            number of classes to consider in y direction
        alpha : float
            angle of the direction of the variogram
        kwargs : dict
            dictionnary of parameters to pass to geone.covModel.variogramExp2D
        """

        plt.cla()
        
        if hmax <= 0:
            hmax = np.nan

        tol_dist = hmax / ncla        
        ncla = (ncla, ncla)
        hmax=(hmax, hmax)

        self.hmax = hmax
        (h1, v1, p1), (h2, v2, p2) = gcm.variogramExp2D(x, v, alpha, tol_dist=tol_dist,
                                                        hmax=hmax, ncla=ncla,
                                                        **kwargs)
        
        
        self.h = np.array((h1, h2))
        self.val = np.array((v1, v2))
        self.alpha = alpha
        plot_var_exp(h1, h2, v1 ,v2, p1, p2, ax=self.ax)
        plt.legend()
        
    def make_exp_var_3D(self, x, v, ncla=(10, 10, 10), hmax=None, alpha=0, beta=0, gamma=0, **kwargs):
        # TO DO
        (h1, v1, p1), (h2, v2, p2), (h3, v3, p3) = gcm.variogramExp3D(x, v,
                                                                      alpha, beta, gamma,
                                                                      hmax=hmax, ncla=ncla,
                                                                      **kwargs)

class Cm2fit():
    
    """
    Class for adjusting variograms

    Parameters
    ----------
    h_max : float
        maximum distance for the variogram
    w_max : float
        maximum variance for the variogram
    r_max : float
        maximum range for the variogram
    nu_max : float
        maximum smoothness (nu parameter) to consider for the variogram, only for Matern covariance
    alpha : float
        angle of the direction of the variogram, 0 implies that x axis os oriented West-East and y axis South-North
        This is just initial value for the fit. Only used for 2D variograms
    ax : matplotlib.axes, optional
        axes to plot the variogram
    """


    # 1D estimation
    def __init__(self, h_max=100, w_max=10, r_max = 100, nu_max=10, alpha=0, ax=None):
        
        self.h_max = h_max
        self.w_max = w_max
        self.r_max = r_max
        self.nu_max = nu_max
        self.alpha = alpha
        self.l = []
        self.w = []
        self.w2 = []
        self.ax = ax

    # update functions 
    def update_hmax(self, h_max):
        
        """
        update the maximum distance for the variogram

        :meta private:
        """
        #remove
        if self.ax is None:
            axi = plt.gca()
        else:
            axi = self.ax

        [l.remove() for l in axi.lines]

        #update params
        self.h_max = h_max
        
        # plot
        h = np.arange(0, self.h_max, 0.1)
        f = self.cm.vario_func()
        axi.plot(h, f(h), c="k")
    
    def update_cm2D(self, h_max, alpha):
        
        """
        Update the maximum distance for the variogram and the angle of the direction of the variogram

        :meta private:
        """

        #remove
        if self.ax is None:
            axi = plt.gca()
        else:
            axi = self.ax

        [l.remove() for l in axi.lines]
        
        #update params
        self.h_max = h_max
        self.cm.alpha = alpha
        
        # plot
        self.cm.plot_model_one_curve(1, True, hmax=self.h_max, color="blue")
        self.cm.plot_model_one_curve(2, True, hmax=self.h_max, color="orange")  
    
    def make_update(self):
        
        """
        Private function to update the variogram

        :meta private:
        """

        def update(i, typ, **kwargs):
            
            if self.ax is None:
                axi = plt.gca()
            else:
                axi = self.ax

            [l.remove() for l in axi.lines]
            
            if typ == "nugget":
                dic = {"w": kwargs["w"]}
            elif typ == "matern":
                dic = {"w": kwargs["w"], "r":kwargs["r"], "nu":kwargs["nu"]}
            elif typ in ("gamma", "power", "exponential_generalized"):
                pass
                # dic = {"w": kwargs["w"], "r":kwargs["r"], "s":kwargs["s"]}
            else:
                dic = {"w": kwargs["w"], "r":kwargs["r"]}
            self.cm.elem[i] = (typ, dic)
            
            f = self.cm.vario_func()
            h = np.arange(0, self.h_max, 0.1)
            axi.plot(h, f(h), c="k")
            
            
        return update

    def make_update_2D(self):

        """
        Private function to update the variogram

        :meta private:
        """

        def update(i, typ, **kwargs):
            
            # remove previous lines
            if self.ax is None:
                axi = plt.gca()
            else:
                axi = self.ax

            [l.remove() for l in axi.lines]

            # if self.ax is None:
            #     for o in range(2):
            #         [l.remove() for l in ax.lines]
            #     # ax.texts.clear()
            # else:
            #     for o in range(2):
            #         [l.remove() for l in self.ax.lines]
                # self.ax.texts.clear()
            
            if typ == "nugget":
                dic = {"w": kwargs["w"]}
            elif typ == "matern":
                dic = {"w": kwargs["w"], "r":[kwargs["rx"], kwargs["ry"]], "nu":kwargs["nu"]}
            elif typ in ("gamma", "power", "exponential_generalized"):
                print("not implemented yet")
                pass
                # dic = {"w": kwargs["w"], "r":kwargs["r"], "s":kwargs["s"]}
            else:
                dic = {"w": kwargs["w"], "r":[kwargs["rx"], kwargs["ry"]]}
            self.cm.elem[i] = (typ, dic)  # change elements
            
            self.cm.plot_model_one_curve(1, True, hmax=self.h_max, color="blue", label="x'")
            self.cm.plot_model_one_curve(2, True, hmax=self.h_max, color="orange", label="y'")
            plt.legend()

        return update
    
    # display widgets
    def choose_struc(self, n_struc=1, dim=1):
        
        """
        Methods to add widgets depending on the numbers of structures to consider

        :meta private:

        Parameters
        ----------
        n_struc : int
            number of structures to consider
        dim : int
            dimension of the problem
        """

        if dim == 1:
            #empty covmodel
            elem = []
            for i in range(n_struc):
                elem.append(("spherical", {"w":1, "r":1}))
            self.cm = gcm.CovModel1D(elem = elem)

            update = self.make_update()
            w_hmax = widgets.interact(self.update_hmax, h_max=(0, 2*self.h_max))
            self.w.append(w_hmax)
            for i in range(n_struc):
                update = self.make_update()

                wid = widgets.interact(update, i=fixed(i), typ=["spherical", "exponential", "cubic", "gaussian", "matern", "nugget"],
                                          r=(0, self.r_max, self.r_max/1000),
                                          w=(0, self.w_max, self.w_max/1000),
                                          nu=(0, self.nu_max, 0.1))
                self.w.append(wid)
        
        elif dim == 2:  #  in 2D
            elem = []
            for i in range(n_struc):
                elem.append(("spherical", {"w":1, "r":[1, 1]}))
            self.cm = gcm.CovModel2D(elem = elem, alpha=self.alpha)
            
            update = self.make_update_2D()
            w_hmax = widgets.interact(self.update_cm2D, h_max=(0, 2*self.h_max), alpha=fixed(self.alpha)) 
            self.w.append(w_hmax) 
            for i in range(n_struc):
                update = self.make_update_2D()
                wid = widgets.interact(update, i=fixed(i), typ=["spherical", "exponential", "cubic","gaussian", "matern", "nugget"],
                                          rx=(0, self.r_max, self.r_max/1000),
                                          ry=(0, self.r_max, self.r_max/1000),
                                          w=(0, self.w_max, self.w_max/1000),
                                          nu=(0, self.nu_max, 0.1))
                self.w.append(wid)

    def fit(self, dim=1):
        
        """
        :meta private:
        """

        plt.grid()
        if self.ax is None:
            axi = plt.gca()
        else:
            axi = self.ax
        [l.remove() for l in axi.lines]
        
        def on_change(_):

            # remove legend if exists
            if axi.get_legend() is not None:
                axi.get_legend().remove()

            self.clear()
            
            self.choose_struc(change_w.value, change_dim.value)

        change_w = widgets.IntSlider(value=1, min=1, max=4, description="Number of structures")
        change_dim = widgets.ToggleButtons(options=[1, 2], description="Dimension")
        self.w2.append(change_w)
        self.w2.append(change_dim)
        change_w.observe(on_change, names='value')
        change_dim.observe(on_change, names='value')
        display(change_dim)
        display(change_w)

    def clear(self, all=False):

        """
        :meta private:
        """

        for w in self.w:
            w.widget.close()

        if all:
            for w in self.w2:
                w.close()