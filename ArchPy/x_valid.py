import numpy as np
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
import pyvista as pv
import copy
import time
import sys

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
import ArchPy


# functions

def brier(p, i):
    
    """
    p : vector of probabilities
    i : index of the true answer in vector
    """
    
    s=0
    M=len(p)
    for j in range(M):
        
        if i == j:
            s+= (1 - p[j])**2
        else:
            s+= (-p[j])**2
    
    return -s


def img_to_3D_colors(arr, dic):

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
    arr_new=np.ones([arr_old.shape[0], 3])*np.nan
    for i, x in enumerate(arr_old):
        if x==x:
            arr_new[i]=dic[x]
        else:
            arr_new[i]=(np.nan, np.nan, np.nan)
    return arr_new.reshape(*arr.shape, 3)


def bh2array(ArchTable, bh, typ="units"):

    """ from ArchPy borehole to array of IDs"""
    
    step=ArchTable.get_sz()  # vertical resolution
    nz=ArchTable.get_nz()  # nz
    zg=ArchTable.get_zgc()  # z cell centers

    
    arr_un=np.ones([nz])*np.nan  # initialize units/facies array
    arr_fa=np.ones([nz])*np.nan
    
    if bh.log_facies is not None: 
        fa_flag=True
    else:
        fa_flag=False
    if bh.log_strati is not None:
        un_flag=True
    else:
        un_flag=False
    

    for iz in range(nz):
        z=zg[iz]
        extract = bh.extract(z, vb=0)
        if extract is None:
            pass
        elif fa_flag and un_flag:
            if extract[0] is not None:
                arr_un[iz] = extract[0].ID
                arr_fa[iz] = extract[1].ID
        elif fa_flag:
            if extract is not None:
                arr_fa[iz] = extract.ID
        elif un_flag:
            if extract is not None:
                arr_un[iz] = extract.ID
        else:
            raise ValueError("No facies or unit data")
    
    if typ=="units":
        return arr_un
    elif typ == "facies": 
        return arr_fa
    else: 
        return (arr_un, arr_fa)
    
    
def test_bh_1fold(ArchTable, bhs_real, bhs_test, weighting_method="same_weight", dic_weights=None, plot=True):
    
    """
    a X-validation on one fold
    
    ### inputs ###
    
    bhs_real    : seq of ArchPy boreholes of realizations
    bhs_test    : seq of ArchPy boreholes to test (test set)
    same_weight : bool, mean weight to each facies/unit class
    plot        : bool, plot or not
    
    """
    
    reals = bhs_real
    scores_un = []
    scores_fa = []

    for i in range(len(bhs_test)):
        bh_test = bhs_test[i]
        bh_real = reals[:, :, i]
           
        ## change units in realization logs to match unit in reference log
        for bh in bh_real:
            bh = bh[0]  # nested list
            for i in range(len(bh.log_strati)):
                s = bh.log_strati[i][0]
                something=0
                if s not in bh_test.get_list_stratis(): # if unit in real log is not in ref log
                    for unit in bh_test.get_list_stratis():  
                        if s in unit.get_baby_units(True, 0):  # check if there is a baby unit of any unit in ref log
                            bh.log_strati[i] = (unit, bh.log_strati[i][1])  # if so change baby to mummy unit

        ## list unit ids
        l = []
        def fun(pile, l):
            for un in pile.list_units:
                if un.f_method == "SubPile":
                    fun(un.SubPile, l)
                else:
                    l.append(un)
        fun(ArchTable.get_pile_master(), l)
        list_units_real = l  # list of all units that are simulated by ArchPy (highest level of hierarchy)

        list_ids = []  # list of possible ids for this reference log
        for i in range(len(list_units_real)):
            s = list_units_real[i]
            something=0
            if s not in bh_test.get_list_stratis(): # if unit in real log is not in ref log
                for unit in bh_test.get_list_stratis():  
                    if s in unit.get_baby_units(True, 0):  # check if there is a baby unit of any unit in ref log
                        if unit.ID not in list_ids:
                            list_ids.append(unit.ID)
                        something=1
                        break  

            if something == 0:  # if s (unit in real bh) is not at all in ref log then add it to list of possible outcomes
                if s.ID not in list_ids:
                        list_ids.append(s.ID)

        ## list facies ids
        list_facies_ids = [i.ID for i in ArchTable.get_all_facies()]

        # transform bh to array
        lu = []
        lfa = []
        for iu in range(ArchTable.nreal_units): 
            ifa = 0
            bh = bh_real[iu, ifa]
            g = bh2array(ArchTable, bh, typ="both")
            lu.append(g[0])
            lfa.append(g[1])
            
            for ifa in range(1, ArchTable.nreal_fa):
                bh = bh_real[iu, ifa]
                g = bh2array(ArchTable, bh, typ="facies")
                lfa.append(g)

        res_un = np.array(lu)
        res_fa = np.array(lfa)
        
        # personal WARNING when check borehole for X-valid - attention to hierarchical units !!!

        # ref
        ref_arr = bh2array(ArchTable, bh_test, typ="both") 
        ref_arr_un = ref_arr[0]
        ref_arr_fa = ref_arr[1]
        
        if plot:
            ## plot unit

            # dictionary for ID to rgb colors
            d = {}
            for unit in ArchTable.list_all_units:
                d[unit.ID] = matplotlib.colors.to_rgb(unit.c)


            # create rgb array for plotting
            res_plot = res_un.T.copy()
            res_plot = img_to_3D_colors(res_plot, d)
            ref_arr_plot = ref_arr_un.reshape(1, -1).T.copy()
            ref_arr_plot = img_to_3D_colors(ref_arr_plot, d)

            # 1st plot
            fig = plt.figure(figsize=(0.7*ArchTable.nreal_units, 3))
            ax1 = plt.subplot(1,2, 1)
            ax1.imshow(res_plot, extent=[0, ArchTable.nreal_units, ArchTable.zg[0], ArchTable.zg[-1]], origin="lower", interpolation="nearest")
            plt.title("real")
            plt.xlabel("Realizations")
            plt.ylabel("Z [m]")

            ax2 = plt.subplot(1,2, 2, sharey = ax1)
            ax2.imshow(ref_arr_plot, extent=[0, ArchTable.nreal_units, ArchTable.zg[0], ArchTable.zg[-1]],
                       origin="lower", interpolation="nearest")
            plt.title("ref")
            plt.show()
            
            if ArchTable.nreal_fa > 0:
                # facies
                dfa_c = {}
                for fa in ArchTable.get_all_facies():
                    dfa_c[fa.ID] = matplotlib.colors.to_rgb(fa.c)

                 # create rgb array for plotting
                res_plot = res_fa.T.copy()
                res_plot = img_to_3D_colors(res_plot, dfa_c)
                ref_arr_plot = ref_arr_fa.reshape(1, -1).T.copy()
                ref_arr_plot = img_to_3D_colors(ref_arr_plot, dfa_c)

                # 1st facies plot
                fig = plt.figure(figsize=(0.7*(ArchTable.nreal_fa+ArchTable.nreal_units), 1.5))
                ax1 = plt.subplot(1,2, 1)
                ax1.imshow(res_plot, extent=[0, (ArchTable.nreal_fa*ArchTable.nreal_units), ArchTable.zg[0], ArchTable.zg[-1]],
                           origin="lower", interpolation="nearest")
                plt.title("real")
                plt.xlabel("Realizations")
                plt.ylabel("Z [m]")

                ax2 = plt.subplot(1,2, 2, sharey = ax1)
                ax2.imshow(ref_arr_plot, extent=[0, ArchTable.nreal_units, ArchTable.zg[0], ArchTable.zg[-1]],
                           origin="lower", interpolation="nearest")
                plt.title("ref")
                plt.show()

        
        # probability vectors
        p_real_un = np.array([np.count_nonzero(res_un == i, 0)/ArchTable.nreal_units for i in list_ids]).T
        p_real_fa = np.array([np.count_nonzero(res_fa == i, 0)/(ArchTable.nreal_units*ArchTable.nreal_fa) for i in list_facies_ids]).T
        
        
        ## compute brier scores
        
        # units
        d={}
        for i,k in enumerate(list_ids):
            d[k] = i

        l = []
        ref = []
        for i in range(ref_arr_un.shape[0]):
            if ~np.isnan(ref_arr_un[i]):
                ref.append(ref_arr_un[i])
                br = brier(p_real_un[i], d[ref_arr_un[i]])
                l.append(br)

        ref = np.array(ref)
        l = np.array(l)
        

        # user defined weights
        if weighting_method == "user_weights":

            ## compute mean
            lst_fac = np.unique(ref)
            n_fac = len(lst_fac)  # number of unit/facies to test
            weights = [dic_weights[i] for i in lst_fac]  # weights for each class, 1st order unit have a weight of 1

            res = 0
            for i in range(n_fac):
                u_id = lst_fac[i]
                res += weights[i]*np.mean(l[ref==u_id])

            res /= np.sum(weights)  


        # weights adjusted proportionally to the hierarchy
        elif weighting_method == "prop_weights":
            def weight_calc(unit):
                
                l = []
                def fun(unit, w):
                    
                    if unit.mummy_unit is not None:

                        m_unit = unit.mummy_unit
                        w *= 1/len(m_unit.SubPile.list_units)
                        print(w)
                        fun(m_unit, w)
                    else:
                        l.append(w)

                weight = 1
                fun(unit, weight)
                return l[0]

            ## compute mean
            lst_fac = np.unique(ref)
            n_fac = len(lst_fac)  # number of unit/facies to test
            weights = [weight_calc(ArchTable.get_unit(ID=i, type="ID")) for i in lst_fac]  # weights for each class, 1st order unit have a weight of 1

            res = 0
            for i in range(n_fac):
                u_id = lst_fac[i]
                res += weights[i]*np.mean(l[ref==u_id])

            res /= np.sum(weights)  


        # same weight to each class
        elif weighting_method == "same_weights":
        
            ## compute mean
            lst_fac = np.unique(ref)
            n_fac = len(lst_fac)  # number of unit/facies to test
            weights = np.ones(n_fac)*1  # weights of each unit/facies

            res = 0
            for i in range(n_fac):
                u_id = lst_fac[i]
                res += weights[i]*np.mean(l[ref==u_id])

            res /= n_fac
        else:
            res = np.mean(l)

        scores_un.append(res)  # add score for this borehole
        if ArchTable.verbose:
            print(res)
    
        # compute facies score
        d={}
        for i,k in enumerate(list_facies_ids):
            d[k] = i
        
        l = []
        ref = []
        for i in range(ref_arr_fa.shape[0]):

            if ~np.isnan(ref_arr_fa[i]):
                ref.append(ref_arr_fa[i])
                br = brier(p_real_fa[i], d[ref_arr_fa[i]])
                l.append(br)

        ref = np.array(ref)
        l = np.array(l)
        
        # same weight to each class
        ## compute mean
        lst_fac = np.unique(ref)
        n_fac = len(lst_fac)  # number of unit/facies to test
        weights = np.ones(n_fac)*1  # weights of each unit/facies

        res = 0
        for i in range(n_fac):
            u_id = lst_fac[i]
            res += weights[i]*np.mean(l[ref==u_id])
    
        res /= n_fac
            
        #res = np.mean(l)
        scores_fa.append(res)
        
                
    return (scores_un, scores_fa)


# X_validation
def X_valid(ArchTable, k=3, nreal_un=5, nreal_fa=2,plot=True,
            weighting_method="same_weight", dic_weights=None,
            seed=15, verbose=1):
    
    """
    Perform a Cross-validation on the given ArchTable

    ### inputs ###
    k     : int, number of folds
    nreal_un : int, number of unit realizations to estimate score
    nreal_fa : int, number of facies realiations to estimate score
    same_weight : bool, to apply same weight to each facies/unit
    plot        : bool, display plots
    seed        : seed for reproducibility
    verbose     : 0 or 1

    ### outputs ###
    list of tuples, each containing 2 scores,
    unit and facies. The number of tuples = k.
    """

    np.random.seed(seed)
    assert len(ArchTable.list_bhs) > 1, "there is not enough boreholes to perform K-validation"
    
    
    Lx = ArchTable.xg[-1] - ArchTable.xg[0]
    arch_table_dummy = copy.deepcopy(ArchTable)

    
    if nreal_fa == 0:
        extractFacies = False
    elif nreal_fa > 0:
        extractFacies = True
    else:
        extractFacies = False
        nreal_fa = 0

    # modify verbose
    ini_verb = arch_table_dummy.verbose
    arch_table_dummy.verbose = verbose
    arch_table_dummy.nreal_fa = nreal_fa
    
    l_bhs = arch_table_dummy.list_bhs
    n_bh = len(l_bhs)


    ## Folding method --> TO DO add other methods ##
    if k > n_bh:
        k = n_bh
        
    # determine size of a fold
    size_fold = int(n_bh/k)

    # shuffle bh ids
    ids = np.arange(n_bh)
    np.random.shuffle(ids)

    # get folds
    folds = np.ones([k, size_fold])
    o_b = 0
    o_a = 0
    for i in range(k):

        o_a += size_fold
        folds[i] = ids[o_b:o_a]
        o_b = o_a
    
    ## 
    score_folds = []
    for ifold in range(k):

             
        idx_to_remove = folds[ifold]  # fold

        new_fold=[bh for i,bh in enumerate(l_bhs) if i not in idx_to_remove]  # train set
        bh_rm = [bh for i,bh in enumerate(l_bhs) if i in idx_to_remove]  # test set
        xy_fold = np.array([(i.x, i.y) for i in new_fold])  # coordinates
        xy_rm = np.array([(i.x, i.y) for i in bh_rm])

        # initialize
        arch_table_dummy.Geol = ArchPy.base.Geol()  # initialize results to nothing  
        arch_table_dummy.rem_all_bhs()  # remove previous boreholes
        arch_table_dummy.add_bh(new_fold)
        
        # reprocess
        arch_table_dummy.bhs_processed=0
        arch_table_dummy.erase_hd()
        arch_table_dummy.process_bhs()
        
        # simulations
        arch_table_dummy.compute_surf(nreal_un)
        arch_table_dummy.compute_facies(nreal_fa)
        
        # extract the realizations at test bh locations
        res = np.array([(i.x, i.y) for i in bh_rm])
        pos_x = res[:, 0]
        pos_y = res[:, 1]
        
        # extract bh at locations
        a = arch_table_dummy.make_fake_bh(pos_x, pos_y, stratIndex=range(arch_table_dummy.nreal_units),
                              faciesIndex=range(max(1, arch_table_dummy.nreal_fa)),
                              extractFacies=extractFacies)
        a = np.array(a)

        # test this fold
        scores = test_bh_1fold(arch_table_dummy, a, bh_rm, plot=plot, weighting_method=weighting_method, dic_weights=dic_weights)
        scores = np.array(scores)
        if plot:
            plt.scatter(xy_rm[:, 0], xy_rm[:, 1])
            plt.scatter(xy_fold[:, 0], xy_fold[:, 1])
            for i in range(xy_rm.shape[0]):
                plt.text(xy_rm[i, 0], xy_rm[i, 1], s=np.round(scores[0, i], 3))
                if nreal_fa > 0:
                    plt.text(xy_rm[i, 0] - Lx/10, xy_rm[i, 1], s=np.round(scores[1, i], 3), c="r")        
                
                
        final_score_un = 0
        final_score_fa = 0
        total_depth = 0
        for i in range(len(scores[0])):
            de = bh_rm[i].depth
            total_depth += de
            final_score_un += de*scores[0][i]
            final_score_fa += de*scores[1][i]
            
        final_score_un /= total_depth
        final_score_fa /= total_depth
        
        score_folds.append((final_score_un, final_score_fa))
          
    # re add boreholes
#     arch_table_dummy.rem_all_bhs() 
#     arch_table_dummy.add_bh(l_bhs)
#     arch_table_dummy.verbose = ini_verb

    return np.array(score_folds)