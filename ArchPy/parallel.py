import numpy as np
import os
import copy

#my modules
from ArchPy.base import *


from multiprocessing import Pool


def compute_surf_1table(Arch_Table):

    Arch_Table.compute_surf(1)
    res_un = Arch_Table.get_units_domains_realizations()[0]
    surf = Arch_Table.Geol.surfaces_by_piles
    surf_bot = Arch_Table.Geol.surfaces_bot_by_piles
    del(Arch_Table)
    
    return (res_un, surf, surf_bot)


def compute_fa_1table(Arch_Table):
    
    Arch_Table.compute_facies(1)
    res_fa = Arch_Table.get_facies()[0]
    del(Arch_Table)
    
    return res_fa


def compute_prop_1table(Arch_Table):
    
    Arch_Table.compute_prop(1)
    res_prop = Arch_Table.Geol.prop_values
    del(Arch_Table)
    
    return res_prop

def parallel_compute(Arch_Table, n_real = 1, n_real_fa = 0, n_real_prop = 0, l_u = None):
    
    nx = Arch_Table.get_nx()
    ny = Arch_Table.get_ny()
    nz = Arch_Table.get_nz()
    
    if Arch_Table.ncpu < 0:
        ncpu = os.cpu_count() + Arch_Table.ncpu
    elif Arch_Table.ncpu > 0:
        ncpu = Arch_Table.ncpu
    
    Arch_Table.nreal_units = n_real
    
    ## units ##
    l = []
    ncpu_units_real = max(int(ncpu/n_real), 1)
    left_cpu = int(ncpu - ncpu_units_real*n_real)
    
    if l_u is not None:
        l = l_u
    else:
        for iu in range(n_real):
            
            # make a copy
            T1_copy = copy.deepcopy(Arch_Table)

            T1_copy.seed  = int(Arch_Table.seed + iu)
            if left_cpu > 0:
                cpu_bonus = 1
                left_cpu -= 1
            else:
                cpu_bonus = 0
            T1_copy.ncpu = ncpu_units_real + cpu_bonus

            T1_copy.verbose = 3
            l.append(T1_copy)
    
    with Pool(ncpu) as p:
        res = p.map(compute_surf_1table, l)

    # get unit results and assemble them
    u_domains = np.array([i[0] for i in res]).reshape(n_real, nz, ny, nx)
    list_surf_dic = np.array([i[1] for i in res])
    # combine dictionaries
    d = list_surf_dic[0]
    for k in d.keys():
        d[k] = np.concatenate([d[k] for d in list_surf_dic], axis=0)
    surf = d

    list_surf_bot_dic = np.array([i[2] for i in res])
    # combine dictionaries
    d = list_surf_bot_dic[0]
    for k in d.keys():
        d[k] = np.concatenate([d[k] for d in list_surf_bot_dic], axis=0)
    surf_bot = d

    if n_real_fa > 0:
        ## facies ##
        l_fa = []  # list to contain Arch_tables
        Arch_Table.nreal_fa = n_real_fa
        n_real_tot_fa = n_real * n_real_fa
        ncpu_fa_real = max(int(ncpu/(n_real_tot_fa)), 1)
        left_cpu = int(ncpu - ncpu_fa_real*n_real_tot_fa)
        for iu in range(n_real):
            for ifa in range(n_real_fa):
                # make a copy
                T1_copy = copy.deepcopy(Arch_Table)
                T1_copy.seed  = int(Arch_Table.seed + iu*n_real_fa + ifa)
                if left_cpu > 0:
                    cpu_bonus = 1
                    left_cpu -= 1
                else:
                    cpu_bonus = 0
                T1_copy.ncpu = ncpu_fa_real + cpu_bonus
                T1_copy.nreal_units = 1
                T1_copy.verbose = 0
                T1_copy.Geol.units_domains = u_domains[iu].reshape(1, nz, ny, nx)
                l_fa.append(T1_copy)
        
        with Pool(ncpu) as p:
            res_fa = p.map(compute_fa_1table, l_fa)

        fa_domains = np.array(res_fa).reshape(n_real, n_real_fa, nz, ny, nx)
        
    if n_real_prop > 0:                   
        ## prop ##
        l_prop = []  # list to contain Arch_tables
        Arch_Table.nreal_prop = n_real_prop    
        n_real_tot_prop = n_real * n_real_fa * n_real_prop  # total number of simulations
        ncpu_prop_real = max(int(ncpu/(n_real_tot_prop)), 1)
        left_cpu = int(ncpu - ncpu_prop_real*n_real_tot_prop)

        for iu in range(n_real):
            for ifa in range(n_real_fa):
                for ip in range(n_real_prop):
                    # make a copy
                    T1_copy = copy.deepcopy(Arch_Table)
                    T1_copy.seed  = int(Arch_Table.seed + iu*n_real_prop*n_real_fa + ifa*n_real_prop + ip)
                    if left_cpu > 0:
                        cpu_bonus = 1
                        left_cpu -= 1
                    else:
                        cpu_bonus = 0
                    T1_copy.nreal_units = 1
                    T1_copy.nreal_fa = 1
                    T1_copy.ncpu = ncpu_prop_real + cpu_bonus
                    T1_copy.verbose = 0
                    T1_copy.Geol.units_domains = u_domains[iu].reshape(1, nz, ny, nx)  # assign right unit domains
                    T1_copy.Geol.facies_domains = fa_domains[iu, ifa].reshape(1, 1, nz, ny, nx)  # assign right facies domains
                    l_prop.append(T1_copy)

        with Pool(ncpu) as p:
            res_prop = p.map(compute_prop_1table, l_prop)
    
    # store results 
    Arch_Table.Geol.units_domains = u_domains
    Arch_Table.Geol.surfaces_by_piles = surf
    Arch_Table.Geol.surfaces_bot_by_piles = surf_bot
    Arch_Table.surfaces_computed = 1
    if n_real_fa > 0:
        # put results in facies_domains
        Arch_Table.Geol.facies_domains = fa_domains
        Arch_Table.facies_computed = 1
            
    if n_real_prop > 0:
        # put results in prop_values
        new_d = {}
        for prop_obj in Arch_Table.list_props:
            l = []
            for iprop in res_prop:
                l.append(iprop[prop_obj.name][0])
            new_d[prop_obj.name] = np.array(l).reshape(n_real, n_real_fa, n_real_prop, nz, ny, nx)
        Arch_Table.Geol.prop_values = new_d
        Arch_Table.prop_computed = 1
               
    # print("done")

