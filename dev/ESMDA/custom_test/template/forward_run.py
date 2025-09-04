import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import pyemu
def main():

    try:
       os.remove(r'head_obs_ref.csv')
    except Exception as e:
       print(r'error removing tmp file:head_obs_ref.csv')
    import sys
    sys.path.append(os.path.join('..','..', '..', '..'))
    import ArchPy
    pyemu.helpers.apply_list_and_array_pars(arr_par_file='mult2model_info.csv',chunk_len=50)
    T1 = ArchPy.inputs.import_project("P1", "../ArchPy_workspace")
    surf_C = np.loadtxt("template/surf_C.txt")
    pyemu.os_utils.run(r'mf6')


if __name__ == '__main__':
    mp.freeze_support()
    main()

