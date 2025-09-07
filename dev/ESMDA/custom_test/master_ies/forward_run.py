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
    pyemu.helpers.apply_list_and_array_pars(arr_par_file='mult2model_info.csv',chunk_len=50)
    import sys
    sys.path.append('..')
    from ArchPy_update import update_gw_model
    update_gw_model()
    pyemu.os_utils.run(r'..\..\..\..\..\..\exe\mf6')


if __name__ == '__main__':
    mp.freeze_support()
    main()

