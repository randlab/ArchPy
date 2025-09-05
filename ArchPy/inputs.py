"""
This module contains functions to load and save ArchPy models and results.
"""

import os
import sys
import numpy as np
import matplotlib
from matplotlib.widgets import PolygonSelector, RectangleSelector
import matplotlib.pyplot as plt
import pandas as pd
import copy
import pickle
import yaml

#geone
import geone
import geone.covModel as gcm

#ArchPy
import ArchPy


###utils_inputs####
class drawPoly:
    
    """
    Class to draw a polygon on a specified figure (ax) and return the coordinates of the polygon.
    First the object must be create and after the draw function can be called.
    It takes as inputs a figure (ax) and a type of drawing (rectangle or polygon)
    """

    def __init__(self):
        
        self.list_p=[]
    
    def draw(self, ax, typ="rectangle"):

        """
        Draw a polygon on a specified figure (ax) and return the coordinates of the polygon.
        It takes as inputs a figure (ax) and a type of drawing (rectangle or polygon)

        Parameters
        ----------
        ax : matplotlib axes object
            axes on which the polygon will be drawn
        typ : str, optional
            type of drawing, by default "rectangle".
            Can be "rectangle" or "polygon".
        """

        print("Select your region")
        print("Use left button to select you area(s)")
        if typ=="rectangle":
            self.poly=RectangleSelector(ax, self.f_rect, interactive=True)
        elif typ=="polygon":
            self.poly=PolygonSelector(ax, self.f)
            
    def f_rect(self, eclick, erelease):
        
        self.list_p=[] #remove previous coordinates if any to ensure to have only one rectangle
        x1, y1=eclick.xdata, eclick.ydata
        x2, y2=erelease.xdata, erelease.ydata      
        self.list_p.append( [(x1, y1),(x1, y2),(x2, y2),(x2, y1)])
        
        return 1
    
    def f(self, x):

        if x not in self.list_p:
            self.list_p.append(x)
            print("Polygon added - please press esc if you want to draw another or close the window when you have finished")
        
        return 1


#### Load files ####
def imp_cm(covmodel_dic, ws=None):
    
    """
    Function to import a geone covmodel given the dictionnary written in the yaml file

    Note
    ----
    The dictionnary must contain the following keys:
    - elem : list of tuples containing the elementary structures
    - alpha : list of floats, anisotropy ratios
    - beta : list of floats, anisotropy ratios
    - gamma : list of floats, anisotropy ratios

    Warning
    -------
    Does not work if the variogram is a pure nugget

    Parameters
    ----------
    covmodel_dic : dict
        dictionnary containing the covmodel parameters
        
    Returns
    -------
    geone.covModel
    """

    # check dimension
    n=len(covmodel_dic["elem"][0][1]["r"])
    if n == 1:
        cm=gcm.CovModel1D(elem=covmodel_dic["elem"])
    elif n == 2:
        if isinstance(covmodel_dic["alpha"], str):
            alpha = np.load(os.path.join(ws, covmodel_dic["alpha"]))
        else:
            alpha = covmodel_dic["alpha"]
        cm=gcm.CovModel2D(elem=covmodel_dic["elem"], alpha=alpha)
    elif n == 3:
        if isinstance(covmodel_dic["alpha"], str):
            alpha = np.load(os.path.join(ws, covmodel_dic["alpha"]))
        else:
            alpha = covmodel_dic["alpha"]
        if isinstance(covmodel_dic["beta"], str):
            beta = np.load(os.path.join(ws, covmodel_dic["beta"]))
        else:
            beta = covmodel_dic["beta"]
        if isinstance(covmodel_dic["gamma"], str):
            gamma = np.load(os.path.join(ws, covmodel_dic["gamma"]))
        else:
            gamma = covmodel_dic["gamma"]
        
        cm=gcm.CovModel3D(elem=covmodel_dic["elem"], alpha=alpha, beta=beta, gamma=gamma)
    
    return cm

       
def load_results(ArchTable, surfs=None, surfs_bot=None, units=None, facies=None, props=None):
    
    """
    Load ArchTable results. 
    If external results must be imported, path of these files (e.g. surf argument) can be passed directly
    All files must be pickle binary files with appropriate format (see below).

    Parameters
    ----------
    ArchTable : :class:`base.Arch_table`
                ArchTable object
    surf      : str,
                Path to the surface results file
                The surfaces original format must be a dictionary with 2D arrays containing surfaces
                as values and associated pile objects as keys.
    surf_bot  : str,
                Path to the surface bot results file
                The surfaces original format must be a dictionary with 2D arrays containing surfaces
                as values and associated pile objects as keys.
    units     : str,
                Path to the units results file
                The units original format must be a 4D array of size (nreal, nz, ny, nx)
                with unit IDs as values.
    facies    : str,
                Path to the facies results file
                The facies original format must be a 5D array of size
                (nreal_units, nreal_facies, nz, ny, nx) with facies IDs as values.
    props     : str,
                Path to the properties results file
                The properties original format must be a dictionary with property names as keys and
                6D array of size (nreal_units, nreal_facies, nreal_prop, nz, ny, nx)
                with prop values as values.

    Returns
    -------
    ArchTable : :class:`base.Arch_table`
                ArchTable object with results loaded
    """
       
    if surfs is None:
        #surfaces
        try:
            with open(os.path.join(ArchTable.ws, ArchTable.name + ".sf"), "rb") as f:
                ArchTable.Geol.surfaces_by_piles=pickle.load(f)
                ArchTable.surfaces_computed=1
        except:
            print("Surface results file not found")
    if surfs_bot is None:
        try:
            with open(os.path.join(ArchTable.ws, ArchTable.name + ".sfb"), "rb") as f:
                ArchTable.Geol.surfaces_bot_by_piles=pickle.load(f)
            print("#### Surfaces loaded ####")
            ArchTable.surfaces_computed=1
        except:
            print("Surface bot results file not found")
            
    if ~ArchTable.write_results:
        if units is None:
            #units
            try:
                with open(os.path.join(ArchTable.ws, ArchTable.name + ".unt"), "rb") as f:
                    ArchTable.Geol.units_domains=pickle.load(f)
                print("#### Units loaded ####")
                ArchTable.surfaces_computed=1
            except: 
                print("Unit results file not found")
        if facies is None:
            #facies
            try:
                with open(os.path.join(ArchTable.ws, ArchTable.name + ".fac"), "rb") as f:
                    ArchTable.Geol.facies_domains=pickle.load(f)
                print("#### Facies loaded ####")
                ArchTable.facies_computed=1
            except:
                print("Facies results file not found")
        if props is None:
            #prop
            try:
                with open(os.path.join(ArchTable.ws, ArchTable.name + ".pro"), "rb") as f:
                    ArchTable.Geol.prop_values=pickle.load(f)
                print("#### Properties loaded ####")
                ArchTable.prop_computed=1
            except:
                print("Property results file not found")
            
    if surfs is not None:
        try:
            with open(surfs, "rb") as f:
                ArchTable.Geol.surfaces_by_piles=pickle.load(f)
                print("#### Surfaces loaded ####")
                ArchTable.surfaces_computed=1
        except:
            print("Surface results file not found")
    if surfs_bot is not None:
        try:
            with open(surfs_bot, "rb") as f:
                ArchTable.Geol.surfaces_bot_by_piles=pickle.load(f)      
                print("#### Bottom surfaces loaded ####")      
                ArchTable.surfaces_computed=1
        except:
            print("Surface bot results file not found")

    if units is not None:
        try:
            with open(units, "rb") as f:
                ArchTable.Geol.units_domains=pickle.load(f)  
                print("#### Units loaded ####")
                ArchTable.surfaces_computed=1
        except:
            print("Units results file not found")
    if facies is not None:
        try:
            with open(facies, "rb") as f:
                ArchTable.Geol.facies_domains=pickle.load(f)
                print("#### Facies loaded ####")
                ArchTable.facies_computed=1
        except:
            print("Facies results file not found")
    if props is not None:
        try:
            with open(props, "rb") as f:
                ArchTable.Geol.prop_values=pickle.load(f)  
                print("#### Properties loaded ####")
                ArchTable.prop_computed=1
        except:
            print("Property results file not found")

def load_bh_files(list_bhs, facies_data, units_data,
                  lbhs_bh_id_col="bh_ID", u_bh_id_col="bh_ID", fa_bh_id_col="bh_ID",
                  u_top_col="top",u_bot_col="bot",u_ID="Strat",
                  fa_top_col="top",fa_bot_col="bot",fa_ID="facies_ID",
                  bhx_col="bh_x", bhy_col='bh_y', bhz_col='bh_z', bh_depth_col='bh_depth',
                  dic_units_names={},
                  dic_facies_names={}, altitude=True, vb=1):

    """
    This function merges unit and facies
    databases in only one, 
    each row of the output dataframe corresponds
    to a layer information about units and facies
    
    Parameters
    ----------
    list_bhs : panda dataframe
        list of boreholes in ArchPy format
    facies_data : panda dataframe
        facies data file in ArchPy format
    units_data : panda dataframe
        unit data file in ArchPy format
    lbhs_bh_id_col : str, optional
        column name of borehole identifier in list_bhs data file, by default "bh_ID"
    u_bh_id_col : str, optional
        column name of borehole identifier in unit data file, by default "bh_ID"
    fa_bh_id_col : str, optional
        column name of borehole identifier in facies data file, by default "bh_ID"
    u_top_col : str, optional
        column name of top elevation info in unit data file, by default "top"
    u_bot_col : str, optional
        column name of bot elevation info in unit data file, by default "bot"
    u_ID : str, optional
        column name of unit identifier info in unit data file, by default "Strat"
    fa_top_col : str, optional  
        column name of top elevation info in facies data file, by default "top"
    fa_bot_col : str, optional  
        column name of bot elevation info in facies data file, by default "bot"
    fa_ID : str, optional
        column name of top elevation info in facies data file, by default "facies_ID"
    dic_units_names : dictionary, optional
        dictionary of units names, by default {}
    dic_facies_names : dictionary, optional
        dictionary of facies names, by default {}
    altitude : bool, optional
        if True, the elevation of the boreholes is taken into account, by default True
    vb : int, optional
        verbose, by default 1

    Returns
    -------
    panda dataframe
        dataframe with unit and facies information merged
    """

    def merge_dbs (fa_data, s_data):

        strat=[]
        for bh_id in list_bhs.index:
            
            if bh_id in fa_data.index and bh_id in s_data.index:
                fa_idata=fa_data.loc[[bh_id]].reset_index()
                s_idata=s_data.loc[[bh_id]].reset_index()
                
                #first top and bots
                ifa_top=fa_idata.top.values[0]
                ifa_bot=fa_idata.bot.values[0]
                is_top=s_idata.top.values[0]
                is_bot=s_idata.bot.values[0]
                
                #set top of log to top of bh
                if fa_idata.top.values[0] != list_bhs.loc[bh_id]["bh_z"]:
                    fa_idata.top.values[0]=list_bhs.loc[bh_id]["bh_z"]
                if s_idata.top.values[0] != list_bhs.loc[bh_id]["bh_z"]:
                     s_idata.top.values[0]=list_bhs.loc[bh_id]["bh_z"]
                     
                i_fa=0
                i_s=0
                ibot=None
                itop=None
                for i in range(fa_idata.shape[0]+s_idata.shape[0]):  # loop over layers in borehole

                    #top, bot and id from facies and unit data 
                    ifa_top=fa_idata.top.values[i_fa]
                    ifa_bot=fa_idata.bot.values[i_fa]
                    ifa_id=fa_idata.facies_ID.values[i_fa]

                    is_top=s_idata.top.values[i_s]
                    is_bot=s_idata.bot.values[i_s]
                    is_id=s_idata.Strat.values[i_s]
                    
                    if itop is None:
                        itop=is_top
                    else:
                        itop=ibot
                        
                    ibot=max(is_bot, ifa_bot) #take highest bot
                    strat.append((bh_id, is_id, ifa_id, itop, ibot)) #add interval
                    
                    if ifa_bot > is_bot:
                        i_fa += 1

                    elif ifa_bot < is_bot:
                        i_s += 1

                    else: #if both bot match
                        i_s += 1
                        i_fa += 1

                    if i_s == s_idata.shape[0]: #end of the borehole
                            break
                    if i_fa == fa_idata.shape[0]:
                            break

            elif bh_id in s_data.index: # if there is info about unit but not facies --> add none to facies
                
                s_idata=s_data.loc[[bh_id]].reset_index()
                
                for i in range(s_idata.shape[0]):
                    is_top=s_idata.top.values[i]
                    is_bot=s_idata.bot.values[i]
                    is_id=s_idata.Strat.values[i]
                    
                    strat.append((bh_id, is_id, None, is_top, is_bot))

            elif bh_id in fa_data.index:# if there is info about faces but not about unit --> add none to unit
                
                fa_idata=fa_data.loc[[bh_id]].reset_index()
                
                for i in range(fa_idata.shape[0]):
                    ifa_top=fa_idata.top.values[i]
                    ifa_bot=fa_idata.bot.values[i]
                    ifa_id=fa_idata.facies_ID.values[i]
                    
                    strat.append((bh_id, None, ifa_id, ifa_top, ifa_bot))
            

        all_strats=pd.DataFrame(strat, columns=["bh_ID","Strat_ID","Facies_ID","top","bot"]).set_index("bh_ID")
        return all_strats

    #fill gaps
    def add_gap(data, bh_id, top, bot):
        if vb:
            print("Gap encountered - creation of a gap interval")
            
        idata=data.loc[[bh_id]]
        lay= copy.deepcopy(idata.iloc[0])
        lay.top=top
        lay.bot=bot
        lay.facies_ID=None
        lay.Strat=None
        data = pd.concat([data, pd.DataFrame(lay).T])
        return data

    #loading files
    #fa_data=pd.read_csv(facies_data)
    fa_data=facies_data.copy()
    fa_data.rename(columns={fa_bh_id_col:"bh_ID", fa_top_col:"top",fa_bot_col:"bot",fa_ID:"facies_ID"},inplace=True)
    if altitude:
        fa_data.sort_values(by=["bh_ID","top"], ascending=False, inplace=True)
    else:
        fa_data.sort_values(by=["bh_ID","top"], ascending=True, inplace=True)

    fa_data.set_index("bh_ID", inplace=True)
    
    #s_data=pd.read_csv(units_data)
    s_data=units_data.copy()
    s_data.rename(columns={u_bh_id_col:"bh_ID", u_top_col:"top",u_bot_col:"bot",u_ID:"Strat"},inplace=True)
    if altitude:
        s_data.sort_values(by=["bh_ID","top"], ascending=False, inplace=True)
    else:
        s_data.sort_values(by=["bh_ID","top"], ascending=True, inplace=True)

    s_data.set_index("bh_ID", inplace=True)
    
    #list_bhs=pd.read_csv(list_bhs)
    list_bhs=list_bhs.copy()
    list_bhs.rename(columns={lbhs_bh_id_col:"bh_ID", bhx_col:"bh_x", bhy_col:"bh_y", bhz_col:"bh_z", bh_depth_col:"bh_depth"}, inplace=True)
    list_bhs=list_bhs[["bh_ID", "bh_x", "bh_y", "bh_z", "bh_depth"]].copy() #keep only wanted columns
    list_bhs.set_index("bh_ID", inplace=True)

    # check that two boreholes doesnt have the same name
    assert list_bhs.index.is_unique, "Boreholes names are not unique. Please check your list_bhs file."

    #apply dictionnaries to changes identifier 
    fa_data.replace(dic_facies_names, inplace=True)
    s_data.replace(dic_units_names, inplace=True)

    #change depth into altitude
    if not altitude:
        for bh_id in list_bhs.index:
            top_0=list_bhs.loc[bh_id, "bh_z"] #borehole altitude
            if bh_id in fa_data.index:
                fa_data.loc[bh_id, "top"]=top_0 - fa_data.loc[bh_id, "top"]
                fa_data.loc[bh_id, "bot"]=top_0 - fa_data.loc[bh_id, "bot"]

            if bh_id in s_data.index:
                s_data.loc[bh_id, "top"]=top_0 - s_data.loc[bh_id, "top"]
                s_data.loc[bh_id, "bot"]=top_0 - s_data.loc[bh_id, "bot"]   

    #fill gaps
    for bh_id in list_bhs.index:

        bh=list_bhs.loc[bh_id]

        if bh_id in fa_data.index:
            fa_idata=fa_data.loc[[bh_id]].reset_index()
        else:
            fa_idata=np.array(())

        if bh_id in s_data.index:
            s_idata=s_data.loc[[bh_id]].reset_index()
        else:
            s_idata =np.array(())

        top_0=bh["bh_z"]
        depth=bh["bh_depth"]

        #facies data
        for i in range(fa_idata.shape[0]):
            #top

            top=fa_idata.loc[i, "top"]
            if top_0 != top and i == 0:

                if vb:
                    print("Top altitude of first facies ({0}) does not match borehole ({2}) altitude ({1}), altitudes modified.\n".format(top_0, top, bh_id))
                    
                #fa_idata.loc[i,"top"]=top_0
                if top_0 > top:
                    fa_data=add_gap(fa_data, bh_id, top_0, top)
                else:
                    fa_idata.loc[i, "top"]=top_0

            if i > 0 and bot != top:  # if there is a gap in the data
                fa_data=add_gap(fa_data, bh_id, bot, top)

            bot=fa_idata.loc[i, "bot"]

            if i == fa_idata.shape[0] - 1:  # last lay
                if (bot-1e-5 > top_0 - depth):  # if bot above maximum depth of borehole --> add a gap
                    fa_data=add_gap(fa_data, bh_id, bot, top_0 - depth)

        #units data
        for i in range(s_idata.shape[0]):

            #top
            top=s_idata.loc[i, "top"]

            if top_0 != top and i == 0:
                if vb:
                    print("Top altitude of first unit ({0}) does not match borehole ({2}) altitude ({1}), altitudes modified.\n".format(top_0, top, bh_id)) 
                #s_idata.loc[i,"top"]=top_0
                if top_0 > top:  # top of borehole above top of first unit --> add a gap
                    s_data=add_gap(s_data, bh_id, top_0, top)
                else:  # top of borehole below top of first unit --> change top of first unit
                    s_idata.loc[i, "top"]=top_0

            if i > 0 and bot != top:  # if there is a gap in the data
                s_data=add_gap(s_data, bh_id, bot, top)

            bot=s_idata.loc[i,"bot"]

            if i == s_idata.shape[0] - 1:  # last lay
                if (bot-1e-5 > top_0 - depth):  # if bot above maximum depth of borehole --> add a gap
                    s_data=add_gap(s_data, bh_id, bot, top_0 - depth)

    fa_data.index.rename("bh_ID", inplace=True)
    s_data.index.rename("bh_ID", inplace=True)

    #sort everything
    fa_data=fa_data.reset_index().sort_values(by=["bh_ID","top"],ascending=False).set_index("bh_ID")
    s_data=s_data.reset_index().sort_values(by=["bh_ID","top"],ascending=False).set_index("bh_ID")

    #merge
    final_db=merge_dbs(fa_data, s_data)

    return final_db, list_bhs


def extract_bhs(df, list_bhs, ArchTable, units_to_ignore=(), facies_to_ignore=(), extract_units=True, extract_facies=True, ignore2None=True, fill_gaps=True, vb=0):
     
    """
    Return a list of ArchTable boreholes from final database of load_bh_files

    Parameters
    ----------
    df : pandas dataframe
        final database of load_bh_files
    list_bhs : pandas dataframe
        list of boreholes dataframe with appropriate column name (generally use second output of :func:`load_bh_files`)
    ArchTable : :class:`base.Arch_table` object
        Arch_table object
    units_to_ignore : sequence of string, optional
        unit identifiers to ignore in the database df. The default is ().
    facies_to_ignore : sequence of string, optional
        facies identifiers to ignore in the database df. The default is ().
    extract_units : bool, optional
        extracting unit info. The default is True.
    extract_facies : bool, optional
        extracting facies info. The default is True.
    ignore2None : bool, optional
        Ignoring units and facies are considered as None (i.e. gaps). The default is True.
    fill_gaps : bool, optional
        if possible, gaps are filled. It means that if you have a gap in the data, 
        the unit/facies is considered as the one above. The default is True.
    vb : int, optional
        verbose. The default is 0.

    Returns
    -------
    list of :class:`base.borehole` objects 
        list of boreholes objects

    """
 
    #list_bhs=pd.read_csv(list_bhs)

    #reset index for getting boreholes objects
    df=df.reset_index()
    list_bhs=list_bhs.reset_index()
    
    #get boreholes
    i=0
    idepth=0
    bh_ID_prev=""
    unit_name_prev=""
    facies_name_prev= ""
    prev_ignore_flag=False
    l_bhs=[]
    log_strati=[]
    log_facies=[]

    altitude=True

    for idx in df.index:
    
        bh_ID=df.loc[idx,"bh_ID"]
        

        #store borehole
        if bh_ID != bh_ID_prev:  # new borehole --> save old one
            if log_strati or log_facies:  # if data about units or facies 
                
                # check if there is at least one info in each log
                if len([s[0] for s in log_strati if s[0] is not None]) == 0:
                    log_strati=None
                if len([s[0] for s in log_facies if s[0] is not None]) == 0:
                    log_facies=None
                
                if log_strati is not None or log_facies is not None: 
                   
                    # get borehole info
                    bh=list_bhs.loc[list_bhs["bh_ID"] == bh_ID_prev] 
                    x=bh["bh_x"].values[0]
                    y=bh["bh_y"].values[0]
                    z=bh["bh_z"].values[0]
                    depth=bh["bh_depth"].values[0]
                    if idepth != depth:
                        if vb:
                            print("end of data does not match end of borehole. \nBorehole depth adapted to the data")
                        depth=idepth
                    ID=bh["bh_ID"].values[0]
                    l_bhs.append(ArchPy.base.borehole(i, ID, x, y, z, depth, log_strati, log_facies))
                    i += 1

                    #reinitialize lith and facies names
                    unit_name_prev=""
                    facies_name_prev= ""
                    prev_ignore_flag=False

                    #reini idepth
                    idepth=0
                
            bh_ID_prev=bh_ID

            #reset logs
            log_strati=[]
            log_facies=[]

        if extract_units:
            unit_name=df.loc[idx,"Strat_ID"]
            #log strati
            if unit_name not in units_to_ignore:
                prev_ignore_flag=False
                if unit_name in [i.name for i in ArchTable.get_all_units()] or unit_name is None:# new bh that have strati info
                    #if unit_name not in [i[0].name for i in log_strati if i[0] is not None]: # if this encountered unit is not already in log
                    if unit_name_prev != unit_name:
                        if altitude:
                            z=df.loc[idx,"top"] 
                        else:
                            z=list_bhs.loc[list_bhs["bh_ID"] == bh_ID]["bh_z"].values[0] - df.loc[idx,"top"] 
                        z=np.round(z, 2)

                        if unit_name is not None:
                            log_strati.append((ArchTable.get_unit(unit_name),z))
                        else:
                            log_strati.append((None, z))
                        unit_name_prev=unit_name
                        
                    #idepth --> variable to ensure that the depth of the bh will be reached
                    if (list_bhs.loc[list_bhs["bh_ID"] == bh_ID]["bh_z"].values[0] - df.loc[idx,"bot"]) > idepth:
                        idepth=list_bhs.loc[list_bhs["bh_ID"] == bh_ID]["bh_z"].values[0] - df.loc[idx,"bot"]

                        
                else:
                    if vb:
                        print("{} unit name not found in ArchTable".format(unit_name))
                    pass
            else:  # unit to ignore
                if ignore2None:
                    if prev_ignore_flag:
                        pass
                    else:
                        if altitude:
                            z=df.loc[idx,"top"] 
                        else:
                            z=list_bhs.loc[list_bhs["bh_ID"] == bh_ID]["bh_z"].values[0] - df.loc[idx,"top"]
                        z=np.round(z, 2)

                        if log_strati and fill_gaps:
                            if log_strati[-1][0] is None:  # if latest entry is None
                                log_strati.pop(-1)  # remove latest None entry

                        log_strati.append((None, z))
                        prev_ignore_flag = True

                        if not fill_gaps:
                            unit_name_prev = unit_name

        if extract_facies:
            facies_name=df.loc[idx,"Facies_ID"]
       
            #log_facies
            if facies_name not in facies_to_ignore:
                prev_ignore_flag = False
                if facies_name in [i.name for i in ArchTable.get_all_facies()] or facies_name is None:# new bh that have strati info
                    if facies_name_prev != facies_name:
                        if altitude:
                            z=df.loc[idx,"top"] 
                        else:
                            z=list_bhs.loc[list_bhs["bh_ID"] == bh_ID]["bh_z"].values[0] - df.loc[idx,"top"] 
                        z=np.round(z, 2)
                        
                        #idepth --> variable to ensure that the depth of the bh will be reached
                        if list_bhs.loc[list_bhs["bh_ID"] == bh_ID]["bh_z"].values[0] - df.loc[idx,"bot"] > idepth:
                            idepth=list_bhs.loc[list_bhs["bh_ID"] == bh_ID]["bh_z"].values[0] - df.loc[idx,"bot"]
                            
                        if facies_name is not None:
                            log_facies.append((ArchTable.get_facies_obj(facies_name),z))
                        else:
                            log_facies.append((None, z))
                        facies_name_prev=facies_name
                else:
                    if vb:
                        print("{} facies name not found in ArchTable".format(facies_name))
            else:  # unit to ignore
                if ignore2None:
                    if prev_ignore_flag:
                        pass
                    else:
                        if altitude:
                            z=df.loc[idx,"top"] 
                        else:
                            z=list_bhs.loc[list_bhs["bh_ID"] == bh_ID]["bh_z"].values[0] - df.loc[idx,"top"]
                        z=np.round(z, 2)

                        if log_facies and fill_gaps:
                            if log_facies[-1][0] is None:  # if latest entry is None
                                log_facies.pop(-1)  # remove latest None entry

                        log_facies.append((None, z))
                        prev_ignore_flag = True

                        if not fill_gaps:
                            facies_name_prev = facies_to_ignore
                

    #store final borehole
    if log_strati or log_facies: #if data about units or facies (to do)
        
        #check if there is at least one info in each log
        if len([s[0] for s in log_strati if s[0] is not None]) == 0:
            log_strati=None
        if len([s[0] for s in log_facies if s[0] is not None]) == 0:
            log_facies=None
        
        # get borehole info
        bh=list_bhs.loc[list_bhs["bh_ID"] == bh_ID_prev] 
        x=bh["bh_x"].values[0]
        y=bh["bh_y"].values[0]
        z=bh["bh_z"].values[0]
        depth=bh["bh_depth"].values[0]
        ID=bh["bh_ID"].values[0]
        l_bhs.append(ArchPy.base.borehole(i, ID, x, y, z, depth, log_strati, log_facies))
        i += 1
    
    return l_bhs

### import ###
def import_d_facies(dic_project):
    
    """
    Import a dictionary of facies objects from the dictionary of
    the project that was imported from the yaml file

    Parameters
    ----------
    dic_project : dict
        Dictionary of the project imported from the yaml file

    Returns
    -------
    dict
        Dictionary of facies objects
    """
    d={}

    if "Facies" not in dic_project.keys():
        print("Facies field not defined in the project yaml file")
        return d
    
    dic_facies=dic_project["Facies"]
    for k, v in dic_facies.items():
        d[k]=ArchPy.base.Facies(v["ID"], k, v["color"])
        
    return d


def import_d_units(dic_project, all_facies, ws=None):

    """
    Import a dictionary of unit objects

    Parameters
    ----------
    dic_project : dict
        Dictionary of the project imported from the yaml file
    all_facies : dict
        Dictionary of all facies objects
    ws : str, optional
        Path to the working directory, by default None

    Returns
    -------
    dict
        Dictionary of units objects   
    """

    d={}
    dic_units=dic_project["Units"]
    for name, para in dic_units.items():

        #surface
        dic_s={}
        for k, v in para["surface"]["dic_surf"].items():
            if k == "covmodel":
                if v is not None:
                    dic_s[k]=ArchPy.inputs.imp_cm(v, ws=ws)
                else:
                    dic_s[k]=v
            else:
                if isinstance(v, str) and v.split(".")[-1] == "npy":
                    print(v)
                    dic_s[k]=np.load(os.path.join(ws, v))
                else:
                    dic_s[k]=v
        S=ArchPy.base.Surface(name=para["surface"]["name"],
                                contact= para["surface"]["contact"],
                                dic_surf=dic_s)    

        #dic_facies
        d_fa={}
        for k, v in para["dic_facies"].items():

            if k =="TI":
                if isinstance(v, str):
                    d_fa[k]=geone.img.readImageGslib(os.path.join(ws, v))
                else: #if not a path
                    d_fa[k]=v
            elif k == "G_cm" or k=="f_covmodel":
                if v is not None:
                    l_cms=[]
                    try:
                        for cm in v:
                            l_cms.append(ArchPy.inputs.imp_cm(cm))
                    except:
                        l_cms.append(ArchPy.inputs.imp_cm(v))

                    d_fa[k]=l_cms
                else:
                    d_fa[k]=v

            elif isinstance(v, str) and v.split(".")[-1] == "npy":
                d_fa[k]=np.load(os.path.join(ws, v))

            else:
                d_fa[k]=v
        unit=ArchPy.base.Unit(para["name"], order=para["order"], color=para["color"], ID=para["ID"], surface=S, dic_facies=d_fa)
        for fa in para["list_facies"]:
            unit.add_facies(all_facies[fa]) # add facies
        d[name]=unit
                
    return d

def import_d_prop(dic_project, d_fa):
    
    """
    Import a dictionary of property objects

    Parameters
    ----------
    dic_project : dict
        Dictionary of the project imported from the yaml file
    d_fa : dict
        Dictionary of facies objects (see :func:`import_d_facies`)
    """

    assert "properties" in dic_project.keys(), '"properties" field not defined in the project yaml file'
    
    d={}
    for name, para in dic_project["properties"].items(): 
        
        if para["covmodels"] is None:
            l_cms = None
        else:
            l_cms=[]
            for cm in para["covmodels"]:
                if cm is not None:
                    l_cms.append(imp_cm(cm))
                else:
                    l_cms.append(cm)

        l_facies=[d_fa[i] for i in para["facies"]] #get list of facies object
        
        if "x" not in para.keys():
            para["x"]=None
        if "v" not in para.keys():
            para["v"]=None
        Prop=ArchPy.base.Prop(para["name"], facies=l_facies, covmodels=l_cms,
                                means=para["means"], int_method=para["int_method"], def_mean=para["def_mean"], 
                                vmin=para["vmin"], vmax=para["vmax"], x=para["x"], v=para["v"])
    
        d[name]=Prop
    return d
                

def import_d_piles(dic_project, d_units):
    
    """
    Import a dictionary of pile objects from the given dictionary project

    Parameters
    ----------
    dic_project : dict
        Dictionary of the project imported from the yaml file
    d_units : dict
        Dictionary of units objects (see :func:`import_d_units`)
    """

    d={}
    
    dic_piles=dic_project["Piles"]
    for k, v in dic_piles.items():
        pile=ArchPy.base.Pile(v["name"], verbose=v["verbose"], seed=v["seed"])
        for unit in v["list_units"]:
            pile.add_unit([d_units[unit]]) # to finish
        d[k]=pile
    return d


def import_project(project_name, ws, import_bhs=True, import_results=True, import_grid=True):
    
    """
    Load an ArchPy project (Arch_table) 
    with the specified name (project_name) 
    and in the working_directory (ws) 
    
    Note
    ----
    All the configurations files must be in the working directory

    Parameters
    ----------
    project_name : str
        Name of the project --> name of the yaml file
    ws : str
        Path to the working directory where all files are stored
    import_bh : bool, optional
        Flag to indicate to import borehole files, by default True
    import_results : bool, optional
        Flag to indicate to import result files, by default True
    import_grid : bool, optional
        Flag to indicate to import grid, by default True

    Returns
    -------
    :class:`ArchPy.base.Arch_table` object
    """
    
    print("### IMPORTING PROJECT {} IN {} DIRECTORY ### \n".format(project_name, ws))
    
    with open(os.path.join(ws, project_name+".yaml"), "r") as f:
        dic_project=yaml.load(f, Loader=yaml.Loader)  # bug fix latest yaml version
    
    # load archpy objects
    d_fa=import_d_facies(dic_project)  # facies
    d_un=import_d_units(dic_project, all_facies=d_fa, ws=ws)  # unit
    d_pro=import_d_prop(dic_project, d_fa=d_fa)  # prop
    d_piles=import_d_piles(dic_project, d_units=d_un)  # piles
    
    
    # add subpiles to units
    for unit in d_un.values():
        if "SubPile" in unit.dic_facies.keys():
            if unit.dic_facies["SubPile"] is not None:
                unit.set_SubPile(d_piles[unit.dic_facies["SubPile"]])
    
    ArchTable=ArchPy.base.Arch_table(dic_project["name"], working_directory=ws, seed = min(int(dic_project["seed"] / 1e6), 1),
                                      verbose=dic_project["verbose"], ncpu=dic_project["ncpu"])
    
    ArchTable.set_Pile_master(d_piles[dic_project["Pile_master"]])
    
    # nreal_units, nreal_fa, nreal_prop
    if "nreal_units" in dic_project.keys():
        ArchTable.nreal_units=dic_project["nreal_units"]
    if "nreal_fa" in dic_project.keys():
        ArchTable.nreal_fa=dic_project["nreal_fa"]
    if "nreal_prop" in dic_project.keys():
        ArchTable.nreal_prop=dic_project["nreal_prop"]

    # add properties
    for prop in d_pro.values():
        ArchTable.add_prop(prop)

    if import_grid:
        #grid
        assert "grid" in dic_project.keys(), "Grid was not defined for project {}".format(project_name)
        di=dic_project["grid"]["dimensions"]
        spa=dic_project["grid"]["spacing"]
        ori=dic_project["grid"]["origin"]
        
        # rotation
        if "rotation_angle" in dic_project["grid"].keys():
            rotation_angle = dic_project["grid"]["rotation_angle"]
        else:
            rotation_angle = 0
        #top
        top=dic_project["grid"]["top"]
        if isinstance(top, str): #path
            if top.split(".")[-1] == "top":
                path=os.path.join(ws, top+".npy")
                if os.path.exists(path):
                    top=np.load(path)
                else:
                    print("Top array has not been found")

        #bot
        bot=dic_project["grid"]["bot"]
        if isinstance(bot, str): #path
            if bot.split(".")[-1] == "bot":
                path=os.path.join(ws, bot+".npy")
                if os.path.exists(path):
                    bot=np.load(path)
                else:
                    print("bot array has not been found")
        
        # add polygon
        # mask
        mask = dic_project["grid"]["mask"]
        if mask.split(".")[-1] == "msk":
            path=os.path.join(ws, mask+".npy")
            if os.path.exists(path):
                mask=np.load(path)

        di=(int(di[0]), int(di[1]), int(di[2]))
        ArchTable.add_grid(di, spa, ori, top=top, bot=bot, mask=mask, rotation_angle=rotation_angle)
    
    if import_bhs:
        # boreholes
        if len(dic_project["boreholes"]) > 0:  # if there is borehole info

            # load true boreholes
            l_bh_path=os.path.join(ws, dic_project["boreholes"]["list_bhs"])
            fd_path=os.path.join(ws, dic_project["boreholes"]["facies_data"])
            un_path=os.path.join(ws, dic_project["boreholes"]["units_data"])
            l_bhs=pd.read_csv(l_bh_path)
            fa_data=pd.read_csv(fd_path)
            un_data=pd.read_csv(un_path)
            db, l_bhs=ArchPy.inputs.load_bh_files(l_bhs, fa_data, un_data, altitude=True)
            
            boreholes=extract_bhs(db, l_bhs, ArchTable)
            ArchTable.add_bh(boreholes)

            # fake boreholes
            l_bh_path=os.path.join(ws, dic_project["boreholes"]["list_fake_bhs"])
            fd_path=os.path.join(ws, dic_project["boreholes"]["facies_fake_data"])
            un_path=os.path.join(ws, dic_project["boreholes"]["units_fake_data"])
            l_bhs=pd.read_csv(l_bh_path)
            fa_data=pd.read_csv(fd_path)
            un_data=pd.read_csv(un_path)
            db, l_bhs=ArchPy.inputs.load_bh_files(l_bhs, fa_data, un_data, altitude=True)

            boreholes=extract_bhs(db, l_bhs, ArchTable, extract_facies=False)
            ArchTable.add_fake_bh(boreholes)
        else:
            print("No borehole data found")
    
    if "geol_map" in dic_project.keys():
        path=os.path.join(ws, dic_project["geol_map"]+".npy")
        if os.path.exists(path):
            geol_map=np.load(path)
            ArchTable.add_geological_map(geol_map)

    if import_results:
        print("\n\n ##LOADING RESULTS## \n\n")
        # load results 
        if "Results" in dic_project.keys():
            if dic_project["Results"] is not None:
                d_res=dic_project["Results"]
                if "surfaces" in d_res.keys():
                    surfs=os.path.join(ws, d_res["surfaces"])
                else:
                    surfs=None
                if "surfaces_bot" in d_res.keys():
                    surfs_bot=os.path.join(ws, d_res["surfaces_bot"])
                else:
                    surfs_bot=None
                # if "units" in d_res.keys():
                #     units=os.path.join(ws, d_res["units"])
                # else:
                #     units=None
                # if "facies" in d_res.keys():
                #     facies=os.path.join(ws, d_res["facies"])
                # else:
                #     facies=None
                # if "properties" in d_res.keys():
                #     props=os.path.join(ws, d_res["properties"])
                # else:
                #     props=None

                load_results(ArchTable, surfs, surfs_bot)
            else:
                print("No results files provided in the yaml file \n")
                print("Trying to find results files in working_directory")
                load_results(ArchTable)

    # Set hierarchical relations
    ArchTable.hierarchy_relations()        
    
    print("\n\n ### SUCCESSFUL IMPORT ### \n")
    return ArchTable


#### Write files ####
def write_bh_files(ArchTable, fake_bhs=False, vb=0):
    
    """
    Write input files from an existing ArchTable project
    """

    # dictionnaries
    l_bh={}
    stratis={}
    facies={}
    
    # empty lists
    l_id=[]
    l_name=[]
    l_bhx=[]
    l_bhy=[]
    l_bhz=[]
    l_depth=[]
    
    strati_bhid=[]
    strati_id=[]
    strati_top=[]
    strati_bot=[]
    
    facies_bhid=[]
    facies_id=[]
    facies_top=[]
    facies_bot=[]
    
    if fake_bhs:
        list_of_bhs=ArchTable.list_fake_bhs
        true_name = ArchTable.name
        ArchTable.name = "fake_" + true_name
    else:
        list_of_bhs=ArchTable.list_bhs

    for bh in list_of_bhs:
        l_id.append(bh.ID)
        l_name.append(bh.name)

        # if rotation --> change x and y
        if ArchTable.get_rot_angle() != 0:
            new_x, new_y = ArchTable.rotate(np.array([bh.x, bh.y]), angle=+ArchTable.get_rot_angle())
            l_bhx.append(new_x)
            l_bhy.append(new_y)
        else:
            l_bhx.append(bh.x)
            l_bhy.append(bh.y)

        l_bhz.append(bh.z)
        l_depth.append(bh.depth)
        
        #extract log_strati
        if bh.log_strati is not None:
            for i, s in enumerate(bh.log_strati):
                unit=s[0]
                top=s[1]
                if i < len(bh.log_strati) - 1:
                    bot=bh.log_strati[i+1][1]
                else:
                    bot=bh.z - bh.depth
                
                strati_bhid.append(bh.ID)
                if unit is not None:
                    strati_id.append(unit.name)
                else:
                    strati_id.append(None)
                strati_top.append(top)
                strati_bot.append(bot)
                
        if bh.log_facies is not None:
            for i, s in enumerate(bh.log_facies):
                fa=s[0]
                top=s[1]
                if i < len(bh.log_facies) - 1:
                    bot=bh.log_facies[i+1][1]
                else:
                    bot=bh.z - bh.depth
                
                facies_bhid.append(bh.ID)
                if fa is not None:
                    facies_id.append(fa.name)
                else:
                    facies_id.append(None)
                facies_top.append(top)
                facies_bot.append(bot)  
    
    #dic list boreholes
    l_bh["bh_ID"]=l_id
    l_bh["bh_x"]=l_bhx
    l_bh["bh_y"]=l_bhy
    l_bh["bh_z"]=l_bhz
    l_bh["bh_depth"]=l_depth
    
    stratis["bh_ID"]=strati_bhid
    stratis["Strat"]=strati_id
    stratis["top"]=strati_top
    stratis["bot"]=strati_bot

    facies["bh_ID"]=facies_bhid
    facies["facies_ID"]=facies_id
    facies["top"]=facies_top
    facies["bot"]=facies_bot    
    
    pd.DataFrame(l_bh).set_index("bh_ID").to_csv(os.path.join(ArchTable.ws, ArchTable.name+".lbh")) #list boreholes
    pd.DataFrame(stratis).set_index("bh_ID").to_csv(os.path.join(ArchTable.ws, ArchTable.name+".ud")) #units data
    pd.DataFrame(facies).set_index("bh_ID").to_csv(os.path.join(ArchTable.ws, ArchTable.name+".fd")) #facies data

    lbh_name = ArchTable.name+".lbh"
    ud_name = ArchTable.name+".ud"
    fd_name = ArchTable.name+".fd"

    if fake_bhs:
        ArchTable.name = true_name

    return lbh_name, ud_name, fd_name


def save_results(ArchTable):

    """
    Save ArchTable results
    """

    d={}
    if ArchTable.surfaces_computed:
        #surfaces
        fn=ArchTable.name + ".sf"
        d["surfaces"]=fn
        with open(os.path.join(ArchTable.ws, fn), "wb") as f:
            pickle.dump(ArchTable.Geol.surfaces_by_piles, f)

        #surfaces bot   
        fn=ArchTable.name + ".sfb"
        d["surfaces_bot"]=fn
        with open(os.path.join(ArchTable.ws, fn), "wb") as f:
            pickle.dump(ArchTable.Geol.surfaces_bot_by_piles, f)
    
    if ~ArchTable.write_results:
        #units
        fn=ArchTable.name + ".unt"
        d["units"]=fn
        with open(os.path.join(ArchTable.ws, fn), "wb") as f:
            pickle.dump(ArchTable.Geol.units_domains, f)

        if ArchTable.facies_computed:
            #facies
            fn=ArchTable.name + ".fac"
            d["facies"]=fn
            with open(os.path.join(ArchTable.ws, fn), "wb") as f:
                pickle.dump(ArchTable.Geol.facies_domains, f)

        if ArchTable.prop_computed:
            #prop
            fn=ArchTable.name + ".pro"
            d["properties"]=fn 
            with open(os.path.join(ArchTable.ws, fn), "wb") as f:
                pickle.dump(ArchTable.Geol.prop_values, f)

    return d

# Create data in a format acceptable for yaml#
def Cm2YamlCm (cm):

    """
    Convert covmodel numbers to float format for yaml compatibility

    :meta private:
    """

    cm_tr=[]
    for el in cm.elem:

        #new d
        new_d={}
        for k, v in el[1].items():
            v=np.array(v)
            if len(v.shape) > 0: 
                n_l=[]
                for iv in v:
                    n_l.append(float(iv))
                new_d[k]=n_l
            else:
                new_d[k]=float(v)     
        cm_tr.append((el[0],new_d))
    
    cm.elem=cm_tr
    if isinstance(cm, geone.covModel.CovModel2D):
        if isinstance(cm.alpha, (int, np.float32, np.float64)):
            cm.alpha=float(cm.alpha)
    if isinstance(cm, geone.covModel.CovModel3D):
        if isinstance(cm.alpha, (int, np.float32, np.float64)):
            cm.alpha=float(cm.alpha)
        if isinstance(cm.beta, (int, np.float32, np.float64)):
            cm.beta=float(cm.beta)
        if isinstance(cm.gamma, (int, np.float32, np.float64)):
            cm.gamma=float(cm.gamma)
    return cm


#create dic funs
#l_num_types= (int, float) #all number types
#np_num_types=(np.float16, np.float32, np.float64, np.int0, np.int8, np.int16, np.int32, np.int64)

def create_d_facies(self):
    l={i.name: {"ID": i.ID, "color": i.c} for i in self.get_all_facies()}
    
    return l

def create_d_f_covmodels(unit):

    l=[{"alpha": i.alpha, "beta": i.beta, "gamma": i.gamma, "elem": Cm2YamlCm(i).elem} for i in unit.list_f_covmodel]
    
    return l

def create_d_surface(surface, unit, ws):
    
    """
    create a yaml dic for a surface object

    :meta private:
    """
    d={}
    d["name"]=surface.name
    d["contact"]=surface.contact
    
    d_dic_s={}
    for k, v in surface.dic_surf.items():
        
        if isinstance(v, str) or isinstance(v, (int, float)) or v is None:
            d_dic_s[k]=v
        elif isinstance(v, gcm.CovModel2D):
            cm=Cm2YamlCm(v)
            if isinstance(cm.alpha, np.ndarray):
                f_name= unit + "_dic_surf_" + str(k) + ".npy"
                np.save(os.path.join(ws, f_name), cm.alpha)
                d_dic_s[k]={"alpha": f_name, "elem": cm.elem}
            else:
                d_dic_s["covmodel"]={"alpha": float(cm.alpha), "elem": cm.elem}
        
        elif hasattr(v, "__iter__") and isinstance(v[0], (int, float)): #sequence of numbers
            new_v=[]
            for iv in v:
                new_v.append(float(iv))  # change data format cause yaml doesn't support np.float so to be sure...
            d_dic_s[k]=new_v
        
        elif isinstance(v, np.ndarray):  # numpy array --> store it in binary file
            f_name=surface.name + "_dic_surf_" + str(k) + ".npy" 
            np.save(os.path.join(ws, f_name),v)
            d_dic_s[k]=f_name
            
        elif isinstance(v, geone.img.Img):  # geone image --> just save file in ws
            f_name=surface.name + "_dic_surf_" + str(k) + ".gslib"
            geone.img.writeImageGslib(v, os.path.join(ws, f_name))  # write file
            d_dic_s[k]=f_name
            
        else:  # warning some arguments cannot work with the export
            print("Warning: {} argument of surface {} is possibly not compatible with the export".format(k, surface.name))
            d_dic_s[k]=v
            
    d["dic_surf"]=d_dic_s
    return d

def create_d_unit(unit, ws):
    
    d={}
    d["name"]=unit.name
    d["order"]=unit.order
    
    if isinstance(unit.c, str):
        c=unit.c
        d["color"]=c
    elif isinstance(unit.c, tuple) or isinstance(unit.c, np.ndarray) or isinstance(unit.c, list):
        new_c=[]
        for ic in unit.c:
            new_c.append(float(ic)) #change data format cause yaml doesn't support np.float
        d["color"]=new_c
    
    d["ID"]=unit.ID
    d["surface"]=create_d_surface(unit.surface, unit.name, ws)
    d["list_facies"]=[i.name for i in unit.list_facies]

    #dic facies
    d_dic_f={}
    for k, v in unit.dic_facies.items():
        if isinstance(v, (str, int, bool)) or v is None:
            d_dic_f[k]=v
        elif isinstance(v, (float, np.float32, np.float64)): #only a string, float or int
            d_dic_f[k]=float(v)
        elif isinstance(v, gcm.CovModel3D): #f covmodel
            cm=Cm2YamlCm(v)
            d_dic_f[k]={"alpha": cm.alpha, "beta": cm.beta, "gamma": cm.gamma, "elem": cm.elem}
        elif isinstance(v, (list, tuple)) and isinstance(v[0],gcm.CovModel3D): #list of fcovmodels
            l=[]
            for iv in v:
                cm=Cm2YamlCm(iv)
                l.append({"alpha": cm.alpha, "beta": cm.beta, "gamma": cm.gamma, "elem": cm.elem})
            d_dic_f[k]=l
        elif isinstance(v, geone.img.Img): #geone image    
            f_name=unit.name + "_dic_facies_" + str(k) + ".gslib"
            geone.img.writeImageGslib(v, os.path.join(ws, f_name)) #write file
            d_dic_f[k]=f_name
        
        elif isinstance(v, np.ndarray): #numpy array --> store it in binary file
            f_name=unit.name + "_dic_facies_" + str(k)
            np.save(os.path.join(ws, f_name),v) #create file
            d_dic_f[k]=f_name+".npy"

        elif isinstance(v, (tuple, list)) and isinstance(v[0], (int, float)): #sequence like of numbers
            l=[]
            for iv in v:
                l.append(float(iv))
            d_dic_f[k]=l
        
        elif k == "SubPile":
            if v is not None:
                d_dic_f[k]=v.name
            else:
                 d_dic_f[k]=v
        else:
            print("Warning: {} argument of unit {} is possibly not compatible with the export".format(k, unit.name))
            d_dic_f[k]=v
            
    d["dic_facies"]=d_dic_f
    
    return d

def create_d_pile(pile):


    d={}
    d["name"]=pile.name
    d["list_units"]=[i.name for i in pile.list_units]
    d["verbose"]=pile.verbose
    d["seed"]=pile.seed

    return d

def create_d_prop(prop):

    def fun(seq):
        l=[]
        for i in seq:
            if hasattr(i, "__iter__"):
                l.append([float(o) for o in i])
            else:
                l.append(float(i))
        return l

    def fun2(seq):
        l=[]
        for i in seq:
            if i is not None:
                l.append(float(i))
            else:
                l.append(i)
        return l
    d={}
    d["name"]=prop.name
    d["facies"]=[i.name for i in prop.facies]

    if prop.covmodels is not None:
        l=[]
        for i in prop.covmodels:
            if i is not None:
                l.append({"alpha": i.alpha, "beta": i.beta, "gamma": i.gamma, "elem": Cm2YamlCm(i).elem})
            else:
                l.append(None)
    else:
        l=None
    d["covmodels"]=l

    d["means"]=fun2(prop.means)
    d["int_method"]=prop.int
    d["def_mean"]=prop.def_mean
    d["vmin"]=prop.vmin
    d["vmax"]=prop.vmax
    if prop.x is not None:
        d["x"]=fun(prop.x)
    if prop.v is not None:
        d["v"]=fun(prop.v)


    return d

def create_d_units(ArchTable):

    d={}
    for unit in ArchTable.get_all_units():
        d[unit.name]=ArchPy.inputs.create_d_unit(unit, ArchTable.ws)

    return d


def create_d_piles(ArchTable):

    d={}
    for pile in ArchTable.get_piles():
        d[pile.name]=create_d_pile(pile)

    return d


def create_d_properties(ArchTable):

    d={}
    for prop in ArchTable.list_props:
        d[prop.name]=create_d_prop(prop)

    return d

def save_project(ArchTable, results=True):
    
    """
    Save an Arch_table object in a yaml file and other supplementary files 
    (images, numpy arrays, gslib files) in a folder.

    Parameters
    ----------
    ArchTable : :class:`base.Arch_table`
        Arch_table object to save
    results : bool, optional
        flag to save results or not
    """

    d={}
    d["name"]=ArchTable.name
    d["ws"]=ArchTable.ws
    if ArchTable.ws not in os.listdir():
        os.makedirs(ArchTable.ws)
    d["seed"] = ArchTable.seed
    d["verbose"]=ArchTable.verbose
    d["Pile_master"]=ArchTable.get_pile_master().name
    d["ncpu"]=ArchTable.ncpu
    d["surfaces_computed"]=ArchTable.surfaces_computed
    d["facies_computed"]=ArchTable.facies_computed
    d["prop_computed"]=ArchTable.prop_computed
    d["nreal_units"]=ArchTable.nreal_units
    d["nreal_fa"]=ArchTable.nreal_fa
    d["nreal_prop"]=ArchTable.nreal_prop

    ##grid##
    gr={}
    gr["dimensions"]=(int(ArchTable.get_nx()), int(ArchTable.get_ny()), int(ArchTable.get_nz()))
    gr["spacing"]=(float(ArchTable.get_sx()), float(ArchTable.get_sy()), float(ArchTable.get_sz()))
    gr["origin"]=(float(ArchTable.get_ox()), float(ArchTable.get_oy()), float(ArchTable.get_oz()))
    gr["rotation_angle"]=ArchTable.get_rot_angle()
    
    #top -> save top with numpy
    file=ArchTable.name+".top"
    np.save(os.path.join(ArchTable.ws, file), ArchTable.top)
    gr["top"]=file
    
    #bot -> save bot with numpy
    file=ArchTable.name+".bot"
    np.save(os.path.join(ArchTable.ws, file), ArchTable.bot)
    gr["bot"]=file
    
    #mask --> save mask with numpy
    file=ArchTable.name+".msk"
    np.save(os.path.join(ArchTable.ws, file), ArchTable.mask)
    gr["mask"]=file
    
#    file=ArchTable.name+".poly"
#    np.save(os.path.join(ArchTable.ws, file), ArchTable.mask.any(0))
#    gr["polygon"]=file

    #save grid
    d["grid"]=gr
    
    #geol map
    if ArchTable.geol_map is not None:
        file=ArchTable.name+".gmap"
        np.save(os.path.join(ArchTable.ws, file), ArchTable.geol_map)
        d["geol_map"]=file

    #save boreholes
    d_bh={}
    l_bh, units_data, facies_data=write_bh_files(ArchTable, fake_bhs=False)
    d_bh["list_bhs"]=l_bh
    d_bh["units_data"]=units_data
    d_bh["facies_data"]=facies_data

    l_bh, units_data, facies_data=write_bh_files(ArchTable, fake_bhs=True)
    d_bh["list_fake_bhs"]=l_bh
    d_bh["units_fake_data"]=units_data
    d_bh["facies_fake_data"]=facies_data    
    
    d["boreholes"]=d_bh
    
    #piles
    d["Piles"]=create_d_piles(ArchTable)

    #units
    d["Units"]=create_d_units(ArchTable)

    #facies
    d["Facies"]=create_d_facies(ArchTable)

    #properties
    d["properties"]=create_d_properties(ArchTable)

    #save results
    if results:
        d_res=save_results(ArchTable)
        if len(d_res) > 0:
            d["Results"]=d_res
        else:
            d["Results"]=None
    else:
        d["Results"]=None

    #create file
    with open(os.path.join(ArchTable.ws, ArchTable.name + ".yaml"), 'w') as file:
        documents=yaml.dump(d, file)

    print("Project saved successfully")

    return True 


def bhs_analysis(db,  Strat_ID="Strat_ID", Facies_ID="Facies_ID", top_col="top", bot_col="bot", ax=None):
    
    """
    Function to plot the facies proportions in boreholes and return the values in a dataframe

    Parameters
    ----------
    db : pandas dataframe
        dataframe containing the geological database, each row is an interval observed
        The dataframe is composed of at least the following columns:
            - Strat_ID: name of the column containing the stratigraphic ID
            - Facies_ID: name of the column containing the facies ID
            - top: name of the column containing the top altitude of the interval
            - bot: name of the column containing the bot altitude of the interval
    Strat_ID : str, optional
        name of the column containing the name of the stratigraphic unit
    Facies_ID : str, optional
        name of the column containing the name of the facies
    top_col : str, optional
        name of the column containing the top of the interval
    bot_col : str, optional
        name of the column containing the bottom of the interval
    ax : matplotlib axis, optional
        axis to plot the results

    Returns
    -------
    pandas dataframe   
        dataframe containing the facies proportions in each stratigraphic unit
    """

    t=db.copy()
    ##Facies
    t["thickness"]=t[top_col] - t[bot_col]
    
    
    print("### Units proportion in boreholes")
    print(t.groupby(["Strat_ID"])["thickness"].sum() / t["thickness"].sum())
    
    
    if ax is None:
        fig, ax=plt.subplots(figsize=(15, 10))
    
    df=pd.DataFrame(t.groupby([Strat_ID, Facies_ID])["thickness"].sum()) #group thickness
    for s_id in df.index.get_level_values("Strat_ID").unique():
        df.loc[s_id]=df.loc[s_id].values/df.groupby("Strat_ID").sum().loc[s_id].values[0] #get proportions
    df.unstack()["thickness"].plot(kind="bar", stacked=True, legend=False, ax=ax)
    ax.set(ylabel="Facies proportion")
    ax.set_ylim(0, 1.5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1),
              ncol=3, fancybox=True, shadow=True)
    plt.tight_layout()
    
    return df
    
class GeolInconsistencyError(Exception):
    pass