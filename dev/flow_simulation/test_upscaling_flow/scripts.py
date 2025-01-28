import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from numba import jit
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import cdist
import flopy as fp
import ArchPy.ap_mf
from ArchPy.ap_mf import archpy2modflow, array2cellids


# some functions
def add_chd(archpy_flow, h1=100, h2=0):

    # add BC at left and right on all layers
    chd_data = []

    gwf = archpy_flow.get_gwf()
    
    a = np.zeros((gwf.modelgrid.nlay, gwf.modelgrid.nrow, gwf.modelgrid.ncol), dtype=bool)
    a[:, :, 0] = 1
    lst_chd = array2cellids(a, gwf.dis.idomain.array)
    for cellid in lst_chd:
        chd_data.append((cellid, h1))

    a = np.zeros((gwf.modelgrid.nlay, gwf.modelgrid.nrow, gwf.modelgrid.ncol), dtype=bool)
    a[:, :, -1] = 1
    lst_chd = array2cellids(a, gwf.dis.idomain.array)
    for cellid in lst_chd:
        chd_data.append((cellid, h2))

    chd = fp.mf6.ModflowGwfchd(gwf, stress_period_data=chd_data, save_flows=True)


@jit()
def DTW(path1, path2, dist):
    """
    Compute the Frechet distance between two pathlines
    """
    
    # compute the distance between each point of the two pathlines
    # dist = distance.cdist(path1, path2)

    # compute frechet matrix
    MF = np.zeros(dist.shape)
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            if i == 0 and j == 0:
                MF[i, j] = dist[i, j]
            elif i == 0:
                MF[i, j] = dist[i, j]
            elif j == 0:
                MF[i, j] = dist[i, j]
            else:
                MF[i, j] = dist[i, j] + min(MF[i-1, j], MF[i, j-1], MF[i-1, j-1])

    return MF[-1, -1]
    
def frechet_distance(path1, path2, dist):

    # compute the distance between each point of the two pathlines
    # dist = distance.cdist(path1, path2)

    # compute frechet matrix
    MF = np.zeros(dist.shape)
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            if i == 0 and j == 0:
                MF[i, j] = dist[i, j]
            elif i == 0:
                MF[i, j] = max(MF[i, j-1], dist[i, j])
            elif j == 0:
                MF[i, j] = max(MF[i-1, j], dist[i, j])
            else:
                MF[i, j] = max(min(MF[i-1, j], MF[i, j-1], MF[i-1, j-1]), dist[i, j])
    
    return MF[-1, -1]

def hausdorff_distance(path1, path2):
    """
    Compute the Hausdorff distance between two pathlines
    """

    d1 = directed_hausdorff(path1, path2)
    return d1[0]

def full_script_one_model(T1, grid_modes, mode_names,
                         factor_x, factor_y, factor_z,
                         iu=0, ifa=0, ip=0, n_loc=100, 
                         modflow_path = "mf6.exe", experience_name="test1", prt_name="test", 
                         compute_models=True, write_results=True, l_df_pi=None):
    
    # create results folder
    if not os.path.exists(f"results/{experience_name}"):
        os.makedirs(f"results/{experience_name}")

    # write a txt file to resume the experience
    with open(f"results/{experience_name}/resume.txt", "w") as f:
        f.write(f"Experience name: {experience_name}\n")
        f.write(f"grid_modes: {grid_modes}\n")
        f.write(f"mode_names: {mode_names}\n")
        f.write(f"factor_x: {factor_x}\n")
        f.write(f"factor_y: {factor_y}\n")
        f.write(f"factor_z: {factor_z}\n")
        f.write(f"number of boreholes: {len(T1.list_bhs)}\n")
        f.write(f"number of particles: {n_loc}\n")

    # plot pile and save it
    T1.plot_pile()
    plt.savefig(f"results/{experience_name}/pile.png", bbox_inches='tight')
    plt.close()

    # create a table with surface and facies parameters
    df_un, df_fa = T1.get_sp()
    df_un.to_excel(f"results/{experience_name}/units_params.xlsx")
    df_fa.to_excel(f"results/{experience_name}/facies_params.xlsx")

    if compute_models:

        #####################################################################################
        ############################# compute flow models ###################################
        #####################################################################################

        l_archpy_flow_models = []
        for o in range(len(grid_modes)):
            grid_mode = grid_modes[o]
            print(f"grid_mode: {grid_mode}")
            model_name = "model"
            model_dir = f"workspace_{grid_mode}_{iu}_{ifa}_{ip}"

            try :
                int(grid_mode.split("_")[-1])
                grid_mode = "_".join(grid_mode.split("_")[:-1])
            except:
                pass

            archpy_flow = archpy2modflow(T1, exe_name=modflow_path, model_name=model_name, model_dir=model_dir)
            archpy_flow.create_sim(grid_mode=grid_mode, iu=iu, factor_x=factor_x[o], factor_y=factor_y[o], factor_z=factor_z[o])
            archpy_flow.set_k("K", iu, ifa, ip, log=True)

            sim = archpy_flow.get_sim()
            gwf = archpy_flow.get_gwf()

            # add chd
            add_chd(archpy_flow, 100, 0)  # create some boundary conditions (fixed head on the left and right) 

            # ims and IC
            if grid_mode == "archpy":

                sim.ims.complexity = "complex"

                # set IC
                dis = gwf.get_package("DIS")
                nx, ny, nz = dis.nrow.get_data(), dis.ncol.get_data(), dis.nlay.get_data()
                strt = np.linspace(100, 0, nx)
                strt = np.tile(strt, (ny, 1))
                strt = np.tile(strt, (nz, 1, 1))

                archpy_flow.set_strt(strt)
            else:
                sim.ims.complexity = "moderate"

            sim.write_simulation()
            res = sim.run_simulation()
            # if run failed, print error message and stop
            if res[0] == False:
                print(f"Error in simulation {grid_mode}")

                if grid_mode != "archpy":
                    # retry with higher complexity
                    sim.ims.complexity = "complex"
                    sim.write_simulation()
                    res = sim.run_simulation()
                    if res[0] == False:
                        print(f"Error in simulation {grid_mode} with complexity complex")
                        # write a txt file to resume the experience
                        with open(f"results/{experience_name}/resume.txt", "a") as f:
                            f.write(f"Error in simulation {grid_mode} with complexity complex and moderate\n")

            l_archpy_flow_models.append(archpy_flow)
                

        #####################################################################################
        ########################### compute particle tracking ###############################
        #####################################################################################

        np.random.seed(1)

        particles_loc_x = np.random.uniform(T1.get_xg()[len(T1.get_xg())//2], T1.get_xg()[-1], n_loc)
        particles_loc_y = np.random.uniform(0, T1.get_yg()[-1], n_loc)
        particles_loc_z = np.random.uniform(-10, -6, n_loc)
        particles_loc = list(zip(particles_loc_x, particles_loc_y, particles_loc_z))

        l_df_pi = []
        for o in range(len(mode_names)):

            archpy_flow_o = l_archpy_flow_models[o]
            # Particle Tracker mode #
            ws_prt = f"workspace_prt_{iu}_{ifa}_{ip}"
            archpy_flow_o.prt_create(prt_name=prt_name, workspace=ws_prt, list_p_coords=particles_loc, trackdir="backward")
            archpy_flow_o.prt_run()

            l = []
            for i in range(n_loc):
                df_particle = archpy_flow_o.prt_get_facies_path_particle(i+1)
                l.append(df_particle)

            # store results
            l_df_pi.append(l)

            # copy test.trk.csv file to results folder
            import shutil
            shutil.copy(f"{ws_prt}/{prt_name}.trk.csv", f"results/{experience_name}/pathlines_{mode_names[o]}_{prt_name}.csv")

        # write a pickle file to store l_df_pi
        import pickle
        with open(f"results/{experience_name}/l_df_pi.pkl", "wb") as f:
            pickle.dump(l_df_pi, f)
        
        # remove all workspaces
        for o in range(len(grid_modes)):
            grid_mode = grid_modes[o]
            model_dir = f"workspace_{grid_mode}_{iu}_{ifa}_{ip}"
            shutil.rmtree(model_dir)
        
        shutil.rmtree(ws_prt)

    if write_results:
        #####################################################################################
        ##################### compute metrics and results, save figures #####################
        #####################################################################################

        if l_df_pi is None:
            import pickle
            with open(f"results/{experience_name}/l_df_pi.pkl", "rb") as f:
                l_df_pi = pickle.load(f)
        
        ####### compare max distance and time #######
        fig, ax = plt.subplots(1, 2, figsize=(11, 3), dpi=200)

        ax[1].set_title("Total distance traveled by the particles")

        for o in range(len(mode_names)):
            ax[1].boxplot([l_df_pi[o][i].cum_distance.iloc[-1] for i in range(n_loc) if l_df_pi[o][i] is not None], positions=[o])

        ax[1].set_xticklabels(mode_names, rotation=45)  # put mode names as xticks  
        ax[1].set_ylabel("Distance [m]")

        ax[0].set_title("Total time traveled by the particles")

        for o in range(len(mode_names)):
            ax[0].boxplot([l_df_pi[o][i].time.iloc[-1] for i in range(n_loc) if l_df_pi[o][i] is not None], positions=[o])
        
        ax[0].set_xticklabels(mode_names, rotation=45)  # put mode names as xticks
        ax[0].set_yscale("log")  # set y axis to log scale
        ax[0].set_ylabel("Time [d]")
        
        plt.savefig(f"results/{experience_name}/distance_time_boxplot.png", bbox_inches='tight')
        plt.close()

        # compare distribution of max time and distance
        def kl_divergence(p, q):
            return np.sum(np.where(p != 0, p * np.log(p / q), 0))
        import scipy.stats as stats

        fig, ax = plt.subplots(2, 2, figsize=(11, 7), dpi=200)

        # TIME #
        for o in range(len(mode_names)):
            ax[0, 0].hist(np.log10([l_df_pi[o][i].time.iloc[-1] for i in range(n_loc) if l_df_pi[o][i] is not None]), bins=20, alpha=0.5, label=mode_names[o])

        ax[0, 0].set_xlabel("log10(time [d])")
        ax[0, 0].set_ylabel("count")

        # fit a kde to obtain probability distribution of each mode
        kde = []
        for o in range(len(mode_names)):
            kde.append(stats.gaussian_kde(np.log10([l_df_pi[o][i].time.iloc[-1] for i in range(n_loc) if l_df_pi[o][i] is not None])))

        x = np.linspace(-2, 10, 2000)
        kl = []

        for o in range(1, len(mode_names)):
            tot_kde = kde[o](x).sum()
            kl.append(kl_divergence(kde[0](x), kde[o](x)) / tot_kde)

        ax[1, 0].bar(mode_names[1:], kl)
        ax[1, 0].set_ylabel("KL divergence")

        ax[0, 0].set_title("Time")
        ax[0, 1].set_title("Distance")

        # save kl
        kl_df = pd.DataFrame(kl, index=mode_names[1:], columns=["KL divergence"])
        kl_df.to_csv(f"results/{experience_name}/kl_divergence_time.csv")

        # DISTANCE #
        for o in range(len(mode_names)):
            ax[0, 1].hist([l_df_pi[o][i].cum_distance.iloc[-1] for i in range(n_loc) if l_df_pi[o][i] is not None], bins=20, alpha=0.5, label=mode_names[o])

        ax[0, 1].set_xlabel("Distance [m]")
        ax[0, 1].set_ylabel("count")

        kde = []
        for o in range(len(mode_names)):
            kde.append(stats.gaussian_kde([l_df_pi[o][i].cum_distance.iloc[-1] for i in range(n_loc) if l_df_pi[o][i] is not None]))

        x = np.linspace(0, 500, 2000)
        kl = []

        for o in range(1, len(mode_names)):
            tot_kde = kde[o](x).sum()
            kl.append(kl_divergence(kde[0](x), kde[o](x)) / tot_kde)

        ax[1, 1].bar(mode_names[1:], kl)
        ax[1, 1].set_ylabel("KL divergence")

        plt.savefig(f"results/{experience_name}/time_distance_dist.png", bbox_inches='tight')
        plt.close()

        # save kl
        kl_df = pd.DataFrame(kl, index=mode_names[1:], columns=["KL divergence"])
        kl_df.to_csv(f"results/{experience_name}/kl_divergence_distance.csv")

        ####### facies score #######
        ## RMSE ##
        def get_facies_prop_time(df):
            o = 0
            for col in df.columns:
                if col.split("_")[0] == "facies":
                    break
                o += 1

            return ((df.iloc[:, -o:].T * df["dt"].values).T).sum(axis=0) / df["dt"].sum()

        all_particles_prop = {}

        for i_particle in range(n_loc):
                
            df1 = l_df_pi[0][i_particle]
            facies_prop = {}
            for ifacies in [i.ID for i in T1.get_all_facies()]:
                df1.loc[df1.facies == ifacies] 
                facies_prop["facies_prop_" + str(ifacies)] = df1.loc[df1.facies == ifacies, "dt"].sum() / df1["dt"].sum()

            df1_prop = pd.Series(facies_prop)
            df_prop = pd.DataFrame(df1_prop)
            for imode in range(1, len(mode_names)):
                df_pi = l_df_pi[imode][i_particle]
                if df_pi is not None:
                    df_i_prop = get_facies_prop_time(df_pi)
                else:
                    df_i_prop = pd.Series(np.nan, index=df1_prop.index)
                df_prop = pd.concat([df_prop, df_i_prop], axis=1)

            df_prop.columns = mode_names
            df_res = np.sqrt(np.mean((df_prop.iloc[:, 1:].T - df_prop.iloc[:, 0])**2, axis=1))
            all_particles_prop[i_particle] = df_res

        fig, ax = plt.subplots(figsize=(1.5*len(mode_names)-1, 2), dpi=200)
        # pd.DataFrame(all_particles_prop).plot(kind="bar", legend=False, ax=ax)
        df_rmse_mean = pd.DataFrame(all_particles_prop).mean(axis=1)
        df_rmse_mean.plot(kind="bar", ax=ax)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_ylabel("RMSE")
        ax.set_title("RMSE facies proportion")

        plt.savefig(f"results/{experience_name}/facies_proportion.png", bbox_inches='tight')
        plt.close()
        
        # save results
        df_rmse_mean.to_csv(f"results/{experience_name}/facies_proportion.csv")

        ## Brier score ##
        brier_all = {}

        for imode in range(1, len(mode_names)):
            l_brier_p = []
            for i_particle in range(n_loc):
                
                # one hot encoding of the facies
                path1 = l_df_pi[0][i_particle]
                path2 = l_df_pi[imode][i_particle]

                if path2 is None:
                    l_brier_p.append(np.nan)
                    continue

                n_fa = 0
                for col in path2.columns:
                    if col.split("_")[0] == "facies":
                        n_fa += 1

                df_merge = pd.merge(path1, path2, on="time", suffixes=("_1", "_2"), how="outer").fillna(method="ffill").set_index("time")
                path1 = df_merge[["facies"]].fillna(method="bfill")
                path2 = df_merge.iloc[:, -n_fa:].fillna(method="bfill")

                # one hot encoding of the facies
                oh = np.zeros((path1.shape[0], n_fa))
                for i in path2.columns:
                    if i.split("_")[0] == "facies":
                        i_facies = int(i.split("_")[-1]) -1
                        oh[:, i_facies][path1["facies"].values == i_facies] = 1

                # brier score
                path2 = path2[sorted(path2.columns, key=lambda x: int(x.split("_")[-1]))]  # reorganize columns in path2 in order that facies are in order
                brier_p = ((path2.values - oh)**2).sum(axis=1).mean()
                
                l_brier_p.append(brier_p)
            
            brier_all[mode_names[imode]] = l_brier_p

        brier_all = pd.DataFrame(brier_all)

        # plot
        for col in brier_all.columns:
            plt.boxplot(brier_all[col], positions=[mode_names.index(col)])

        plt.xticks(range(1, len(mode_names)), mode_names[1:])

        plt.ylabel("Brier score")
        plt.savefig(f"results/{experience_name}/brier_scores.png")
        plt.close()

        # save brier scores
        brier_all.to_csv(f"results/{experience_name}/brier_scores.csv")

        ####### distance metrics #######
        all_l_frec = []
        all_l_haus = []
        all_l_dtw = []
        for imode in range(1, len(mode_names)):
            l_frechet = []
            l_hausdorff = []
            l_dtw = []
            for i_particle in range(n_loc):
                path1 = l_df_pi[0][i_particle][["x", "y", "z"]].values
                if l_df_pi[imode][i_particle] is None:
                    l_frechet.append(np.nan)
                    l_hausdorff.append(np.nan)
                    l_dtw.append(np.nan)
                    continue

                path2 = l_df_pi[imode][i_particle][["x", "y", "z"]].values

                dist = distance.cdist(path1, path2)
                # frechet
                l_frechet.append(frechet_distance(path1, path2, dist))

                # hausdorff
                l_hausdorff.append(hausdorff_distance(path1, path2))

                path1 = l_df_pi[0][i_particle][["x", "y", "z", "time"]]
                path2 = l_df_pi[imode][i_particle][["x", "y", "z", "time"]]

                # dtw
                df_merge = pd.merge(path1, path2, on="time", suffixes=("_1", "_2"), how="outer").fillna(method="ffill").set_index("time")
                path1 = df_merge[["x_1", "y_1", "z_1"]].fillna(method="bfill")
                path2 = df_merge[["x_2", "y_2", "z_2"]].fillna(method="bfill")
                dist = distance.cdist(path1.values, path2.values)

                l_dtw.append(DTW(path1.values, path2.values, dist))
            
            all_l_frec.append(l_frechet)
            all_l_haus.append(l_hausdorff)
            all_l_dtw.append(l_dtw)

        # plot boxplots of the frechet, hausdorff and DTW distances
        fig, ax = plt.subplots(1, 3, figsize=(6*(len(grid_modes)-1), 4))
        plt.subplots_adjust(wspace=0.3)

        ax[0].boxplot([all_l_frec[i] for i in range(len(grid_modes) - 1)])
        ax[0].set_xticklabels(mode_names[1:], rotation=45)
        ax[0].set_ylabel("Frechet distance")

        ax[1].boxplot([all_l_haus[i] for i in range(len(grid_modes) - 1)])
        ax[1].set_xticklabels(mode_names[1:], rotation=45)
        ax[1].set_ylabel("Hausdorff distance")

        ax[2].boxplot([all_l_dtw[i] for i in range(len(grid_modes) - 1)])
        ax[2].set_xticklabels(mode_names[1:], rotation=45)
        ax[2].set_ylabel("DTW distance")

        plt.savefig(f"results/{experience_name}/curves_distances.png", bbox_inches='tight')
        plt.close()

        # save results (all distances in the same dataframe)
        df_distances = pd.DataFrame(all_l_frec).T
        df_distances = pd.concat([df_distances, pd.DataFrame(all_l_haus).T], axis=1)
        df_distances = pd.concat([df_distances, pd.DataFrame(all_l_dtw).T], axis=1)

        df_distances.columns = [f"frechet_{mode_names[i]}" for i in range(1, len(mode_names))] + [f"hausdorff_{mode_names[i]}" for i in range(1, len(mode_names))] + [f"dtw_{mode_names[i]}" for i in range(1, len(mode_names))]
        df_distances.to_csv(f"results/{experience_name}/distances.csv")
    

# parallel computing
import multiprocessing
from multiprocessing import Pool

def run_script(l):
    T1, iu, ifa, ip = l
    # hard code parameters
    experience_name = "exp1_{}_{}_{}".format(iu, ifa, ip)
    modflow_path = "../../../../../exe/mf6.exe"
    grid_modes = ["archpy", "layers", "new_resolution", "new_resolution_2"]
    mode_names = ["reference", "layers", "new_resolution - 2", "new_resolution - 4"]
    factor_x = [None, None, 2, 4]
    factor_y = [None, None, 2, 4]
    factor_z = [None, None, 2, 4]

    full_script_one_model(T1, grid_modes, mode_names, factor_x, factor_y, factor_z, iu=iu, ifa=ifa, ip=ip, n_loc=200, 
                    modflow_path = modflow_path, experience_name=experience_name, write_results=True, compute_models=True)

def parallel(nthreads, l=None):
    
    pool = Pool(nthreads)
    pool.map(run_script, l)
    pool.close()
    pool.join()
