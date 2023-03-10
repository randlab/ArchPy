import numpy as np
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
import pyvista as pv
import copy
import time
import sys
import pandas as pd

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
import ArchPy.base
from ArchPy.base import Geol

# functions

def brier_func(p, i):
    
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
            #arr_new[i]=(np.nan, np.nan, np.nan)
            arr_new[i]=(1, 1, 1)
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
            if extract[1] is not None:
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
    
    
def test_bh_1fold(ArchTable, bhs_real, bhs_test, weighting_method = "same_weight", dic_weights = None,
                  plot = True, brier = True, proba_correct = False, aspect="auto"):
    
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
    scores_correct_un = []
    scores_correct_fa = []

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
                        if unit is not None:  
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
                    if unit is not None: 
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
            fig,ax = plt.subplots(1,2, figsize=(.5*ArchTable.nreal_units, 5), sharey=True)
            ax[0].imshow(res_plot, extent=[0, ArchTable.nreal_units, ArchTable.zg[0], ArchTable.zg[-1]], origin="lower", interpolation="nearest", aspect=aspect)
            ax[0].set_title("real")
            ax[0].set_xlabel("Realizations")
            ax[0].set_ylabel("Z [m]")

            ax[1].imshow(ref_arr_plot, extent=[0, ArchTable.nreal_units, ArchTable.zg[0], ArchTable.zg[-1]],
                       origin="lower", interpolation="nearest", aspect=aspect)
            ax[1].set_title("ref")
            plt.subplots_adjust(wspace=1/ArchTable.nreal_units)
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


        if proba_correct:
            ## compute_proba_correct

            # units
            model_res = []
            for ireal in range(res_un.shape[0]):
                imodel = res_un[ireal]
                mask = (ref_arr_un == ref_arr_un)
                n_correct = np.sum(ref_arr_un[mask] == imodel[mask]) / np.sum(mask)
                model_res.append(n_correct)

            scores_correct_un.append(np.mean(model_res))

            # facies
            model_res = []
            for ireal in range(res_fa.shape[0]):
                imodel = res_fa[ireal]
                mask = (ref_arr_fa == ref_arr_fa)
                n_correct = np.sum(ref_arr_fa[mask] == imodel[mask]) / np.sum(mask)
                model_res.append(n_correct)

            scores_correct_fa.append(np.mean(model_res))

        if brier:
            ## compute brier scores

            # probability vectors
            p_real_un = np.array([np.count_nonzero(res_un == i, 0)/ArchTable.nreal_units for i in list_ids]).T
            p_real_fa = np.array([np.count_nonzero(res_fa == i, 0)/(ArchTable.nreal_units*ArchTable.nreal_fa) for i in list_facies_ids]).T

            # units
            d={}
            for i,k in enumerate(list_ids):
                d[k] = i

            l = []
            ref = []
            for i in range(ref_arr_un.shape[0]):
                if ~np.isnan(ref_arr_un[i]):  # ignore nan in test bhs
                    ref.append(ref_arr_un[i])
                    br = brier_func(p_real_un[i], d[ref_arr_un[i]])
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
                if n_fac > 0:
                    weights = np.ones(n_fac)*1  # weights of each unit/facies

                    res = 0
                    for i in range(n_fac):
                        u_id = lst_fac[i]
                        res += weights[i]*np.mean(l[ref==u_id])

                    res /= n_fac

                else:

                    res = -1
            else:
                res = np.mean(l)

            if np.isnan(res) or res is None:
                res = -1

            scores_un.append(res)  # add score for this borehole
            #if ArchTable.verbose:
            #    print(res)

            # compute facies score
            d={}
            for i,k in enumerate(list_facies_ids):
                d[k] = i

            l = []
            ref = []
            for i in range(ref_arr_fa.shape[0]):

                if ~np.isnan(ref_arr_fa[i]):
                    ref.append(ref_arr_fa[i])
                    br = brier_func(p_real_fa[i], d[ref_arr_fa[i]])
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

            #res /= n_fac

            res = np.mean(l)
            scores_fa.append(res)

    d = {"proba_correct":(scores_correct_un, scores_correct_fa), "brier":(scores_un, scores_fa)}
    return d

# X_validation
def X_valid(ArchTable, k=3, nreal_un=5, nreal_fa=2,plot=True,
            brier = True, proba_correct = True,
            aggregate_method = None,
            weighting_method = "same_weights", dic_weights = None, parallel=False,
            seed = 15, folding_method = "random", aspect="auto",
             verbose = 1, **kwargs):

    """
    Perform a Cross-validation on the given ArchTable

    ## inputs ##
    k     : int, number of folds
    nreal_un : int, number of unit realizations to estimate score
    nreal_fa : int, number of facies realiations to estimate score
    brier    : bool, return brier scores
    proba_correct : bool, return proportion of correct cells per units/facies
    aggregate_method : str or None, to perform X-valid on mean model rather than on realizations
                       This parameter is pass to the realizations_aggregation function
                       Ignored if None
    weighting_method : str, which method to use for applying the weights
                 possible values are : same_weights, prop_weights, user_weights
    folding_method : string, folding method to use to separate the data
                     methods availables : "random", "k_means" (to implement), "stratified" (to implement)
    plot        : bool, display plots
    seed        : int, seed 
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

    if nreal_un == 0:
        extractUnits = False
    elif nreal_un > 0:
        extractUnits = True
    else:
        extractUnits = False
        nreal_un = 0

    if not extractUnits and not extractFacies:
        print("Number of units and facies realizations are equal to 0, Cross-validation aborted")
        return 

    # modify verbose
    ini_verb = arch_table_dummy.verbose
    arch_table_dummy.verbose = verbose
    arch_table_dummy.nreal_fa = nreal_fa
    
    l_bhs = arch_table_dummy.list_bhs

    # keep bhs with necessary info
    new_l_bhs = []
    if extractFacies:
        for bh in l_bhs:
            if bh.log_facies is not None:
                new_l_bhs.append(bh)

    if extractUnits:   
        for bh in l_bhs:
            if bh.log_strati is not None:
                if bh not in new_l_bhs:
                    new_l_bhs.append(bh)

    l_bhs = new_l_bhs
    n_bh = len(l_bhs)

    if k > n_bh:
        k = n_bh    

    ## Folding method --> TO DO add other methods ##
    if folding_method == "random":
            
        # determine size of a fold
        size_fold = int(n_bh/k)

        # shuffle bh ids
        ids = np.arange(n_bh)
        np.random.shuffle(ids)

    elif folding_method == "stratified":
        pass

    elif folding_method == "kmeans":
        pass

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

        new_fold=[bh for i,bh in enumerate(l_bhs) if (i not in idx_to_remove)]  # train set
        bh_rm = [bh for i,bh in enumerate(l_bhs) if (i in idx_to_remove)]  # test set
        xy_fold = np.array([(i.x, i.y) for i in new_fold])  # coordinates
        xy_rm = np.array([(i.x, i.y) for i in bh_rm])

        # initialize
        arch_table_dummy.Geol = Geol()  # initialize results to nothing  
        arch_table_dummy.rem_all_bhs()  # remove previous boreholes
        arch_table_dummy.add_bh(new_fold)
        arch_table_dummy.seed = np.random.randint(1e6, 10e6)

        # reprocess
        arch_table_dummy.bhs_processed=0
        arch_table_dummy.erase_hd()
        if arch_table_dummy.geol_map is not None:
            arch_table_dummy.process_geological_map()
        arch_table_dummy.process_bhs()

        # simulations
        if parallel:
            import ArchPy.parallel
            ArchPy.parallel.parallel_compute(arch_table_dummy, nreal_un, nreal_fa, 0)
        else:
            arch_table_dummy.compute_surf(nreal_un)
            arch_table_dummy.compute_facies(nreal_fa)


        if aggregate_method is not None:  # only works on units...

            res = arch_table_dummy.realizations_aggregation(method=aggregate_method, **kwargs)

            arch_table_dummy.Geol.units_domains = res.reshape(1, arch_table_dummy.get_nz(), arch_table_dummy.get_ny(), arch_table_dummy.get_nx())
            arch_table_dummy.nreal_units= 1
            nreal_un = 1

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
        d = test_bh_1fold(arch_table_dummy, a, bh_rm, plot=plot, weighting_method=weighting_method, dic_weights=dic_weights, 
                          proba_correct = proba_correct, brier=brier, aspect=aspect)
        scores = d["brier"]
        scores = np.array(scores)

        if plot:
            plt.scatter(xy_rm[:, 0], xy_rm[:, 1])
            plt.scatter(xy_fold[:, 0], xy_fold[:, 1])
            for i in range(xy_rm.shape[0]):
                plt.text(xy_rm[i, 0], xy_rm[i, 1], s=np.round(scores[0, i], 3))
                if nreal_fa > 0:
                    plt.text(xy_rm[i, 0] - Lx/10, xy_rm[i, 1], s=np.round(scores[1, i], 3), c="r")        
            
            plt.show()
            
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
        
        score_folds.append((final_score_un, final_score_fa, scores, np.array(d["proba_correct"]), bh_rm, a))  


    ### Begin summary ###
    print("@@@ CONFUSION MATRIX: UNITS @@@ \n")

    res = score_folds.copy()

    y_actu = []
    y_pred = []
    y_actu_fa = []
    y_pred_fa = []

    if nreal_fa > 0:
        fa_flag = True
    else:
        fa_flag = False

    n_bh = len(res[0][-2])
    for ik in range(k):
        for ifa in range(max(1, nreal_fa)):
            for ibh in range(n_bh):
                bh_reals = res[ik][-1][:, ifa, ibh]
                bh_true = res[ik][-2][ibh]

                if fa_flag:
                    a = bh2array(ArchTable, bh_true, typ=None)
                    arr_true = a[0]
                    arr_true_fa = a[1]

                    mask_fa = arr_true_fa == arr_true_fa
                else:
                    arr_true = bh2array(ArchTable, bh_true)
                mask = arr_true == arr_true

                for ireal in range(nreal_un):

                    if fa_flag:
                        a = bh2array(ArchTable, bh_reals[ireal], typ=None)
                        arr_real = a[0]
                        arr_real_fa = a[1]

                        for i in range(len(arr_true_fa[mask_fa])):
                            v_fa = arr_real_fa[mask][i]
                            v2_fa = arr_true_fa[mask][i]

                            if v_fa == v_fa and v2_fa == v2_fa:
                                y_actu_fa.append(v2_fa)
                                y_pred_fa.append(v_fa)

                    else:
                        arr_real = bh2array(ArchTable, bh_reals[ireal])

                    for i in range(len(arr_true[mask])):
                        v = arr_real[mask][i]
                        v2 = arr_true[mask][i]

                        if v == v and v2 == v2:
                            y_actu.append(v2)
                            y_pred.append(v)

    df_conf_norm_un = None
    df_conf_norm_fa = None

    # matrix units     
    y_actu = np.array(y_actu)
    y_pred = np.array(y_pred)

    y_actu = pd.Series(y_actu, name='Actual')
    y_pred = pd.Series(y_pred, name='Predicted')
    df_confusion = pd.crosstab(y_pred, y_actu, rownames=['Predicted'], colnames=['Actual'])

    for ival in df_confusion.index:
        if ival not in df_confusion.columns:
            df_confusion[ival] = np.ones([df_confusion.shape[0]])*np.nan

    for ival in df_confusion.columns:
        if ival not in df_confusion.index:
            df_confusion.loc[ival] = np.ones([df_confusion.shape[1]])*np.nan

    df_conf_norm = df_confusion.div(df_confusion.sum(axis=1), axis="index")
    df_conf_norm.columns = [ArchTable.get_unit(ID=i, type="ID").name[:] for i in df_conf_norm.columns]
    df_conf_norm.columns.name = df_confusion.columns.name
    df_conf_norm.index = [ArchTable.get_unit(ID=i, type="ID").name[:] for i in df_conf_norm.index]
    df_conf_norm.index.name = df_confusion.index.name

    figsize = (int(df_conf_norm.shape[0]/3), int(df_conf_norm.shape[0]/3))
    plt.figure(figsize=figsize)
    plot_confusion_matrix(df_conf_norm, title="Normalized Confusion matrix")
    df_conf_norm_un = df_conf_norm.copy()

    #matrix facies
    if fa_flag:
        print("@@@ CONFUSION MATRIX: FACIES @@@ \n")
        y_actu_fa = np.array(y_actu_fa)
        y_pred_fa = np.array(y_pred_fa)

        y_actu = pd.Series(y_actu_fa, name='Actual')
        y_pred = pd.Series(y_pred_fa, name='Predicted')
        df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'])
        df_conf_norm = df_confusion.div(df_confusion.sum(axis=1), axis="index")
        df_conf_norm.columns = [ArchTable.get_facies_obj(ID=i, type="ID").name[:] for i in df_conf_norm.columns]
        df_conf_norm.columns.name = df_confusion.columns.name
        df_conf_norm.index = [ArchTable.get_facies_obj(ID=i, type="ID").name[:] for i in df_conf_norm.index]
        df_conf_norm.index.name = df_confusion.index.name
        figsize = (int(df_conf_norm.shape[0]/3), int(df_conf_norm.shape[0]/3))
        plt.figure(figsize=figsize)
        plot_confusion_matrix(df_conf_norm, title="Normalized Confusion matrix")
        df_conf_norm_fa = df_conf_norm.copy()

# other things to add ?

    return (np.array(score_folds), df_conf_norm_un, df_conf_norm_fa)


def plot_confusion_matrix(df, title='Confusion matrix', cmap="plasma"):
    plt.imshow(df, cmap=cmap) # imshow
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(df.columns))
    plt.xticks(tick_marks, df.columns, rotation=90)
    plt.yticks(tick_marks, df.index)
    #plt.tight_layout()
    plt.ylabel(df.index.name)
    plt.xlabel(df.columns.name)
    plt.show()