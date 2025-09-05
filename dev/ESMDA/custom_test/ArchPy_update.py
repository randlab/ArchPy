import sys
import numpy as np
import os

sys.path.append(os.path.join('..','..', '..', '..'))
import ArchPy
import flopy as fp
from ArchPy.ap_mf import archpy2modflow, array2cellids


def update_gw_model():
    T1 = ArchPy.inputs.import_project("P1", "../ArchPy_workspace")
    surf_C = np.loadtxt("surf_C.txt")
    surfaces_by_piles = {}
    surfaces_by_piles["P1"] = T1.Geol.surfaces_by_piles["P1"].copy()
    surfaces_by_piles["PD"] = T1.Geol.surfaces_by_piles["PD"].copy()
    surfaces_by_piles["P1"][0, 1] = surf_C  # update surface C

    T1.define_domains(surfaces_by_piles)

    # archpy2modflow --> apply modifications and retrieve new botoms and idomain
    archpy_flow = archpy2modflow(T1, exe_name="./", model_dir="./")
    archpy_flow.create_sim(grid_mode="layers", iu=0)
    gwf = archpy_flow.get_gwf()

    # import modflow model
    tmp_model_ws = "./"
    sim = fp.mf6.MFSimulation.load(sim_ws=tmp_model_ws, load_only=["obs, chd"])
    gwf_ref = sim.get_model()

    # update
    gwf_ref.dis.botm.set_data(gwf.dis.botm.array)
    gwf_ref.dis.idomain.set_data(gwf.dis.idomain.array)
    gwf_ref.dis.write()  # write dis file

    # CHD
    gwf_ref.chd.remove()
    # BC # add BC at left and right on all layers
    h1 = 1
    h2 = 0
    chd_data = []

    a = np.zeros((gwf_ref.modelgrid.nlay, gwf_ref.modelgrid.nrow, gwf_ref.modelgrid.ncol), dtype=bool)
    a[:, :, 0] = 1
    lst_chd = array2cellids(a, gwf_ref.dis.idomain.array)
    for cellid in lst_chd:
        chd_data.append((cellid, h1))

    a = np.zeros((gwf_ref.modelgrid.nlay, gwf_ref.modelgrid.nrow, gwf_ref.modelgrid.ncol), dtype=bool)
    a[:, :, -1] = 1
    lst_chd = array2cellids(a, gwf_ref.dis.idomain.array)
    for cellid in lst_chd:
        chd_data.append((cellid, h2))

    chd = fp.mf6.ModflowGwfchd(gwf_ref, stress_period_data=chd_data, save_flows=True, maxbound=len(chd_data))
    chd.write() # write chd file

    # OBS to do
    lst_obs =  gwf_ref.obs.continuous.get_data("head_obs_ref.csv")

    new_lst_obs = []
    # check in active cells
    idomain = gwf_ref.dis.idomain.array
    for cell in lst_obs:
        cell_coord = cell[2]
        # print(idomain[cell_coord])
        if idomain[cell_coord] != 1:
            # print("asdfasdf")
            il, ir, ic = cell_coord
            if idomain[il+1, ir, ic] == 1:
                new_cell = (il + 1, ir, ic)
                print("modif")
            elif idomain[il-1, ir, ic] == 1:
                new_cell = (il - 1, ir, ic)
                print("modif")
            else:
                new_cell = cell_coord
        else:
            new_cell = cell_coord
        new_obs = (cell[0], cell[1], new_cell)
        new_lst_obs.append(new_obs)

    obs_recarray = {
        f"head_obs_ref.csv": new_lst_obs
    }

    obs_package = fp.mf6.ModflowUtlobs(
        gwf_ref,
        pname="head_obs",
        filename="{}.obs".format(gwf_ref.name),
        digits=5,
        print_input=True,
        continuous=obs_recarray,
    )

    # gwf_ref.obs.remove()
    obs_package.write()  # write obs file