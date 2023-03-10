import numpy as np


# automatic inference of the pile
def infer_pile(l_bhs):
    
    """
    Determine all possible stratigraphic piles compatible with a given set of boreholes
    
    # inputs #
    l_bhs    : list of lists, list of the boreholes. Boreholes are simply
               lists of int values ranging from 0 to n, where n is the number of
               different units. In the list number are from younger to older, meaning that
               first number is the top unit (youngest) of the bh, last is the last (oldest) unit.
    
    # output #
    results  : (list, float), pile are returned in the same format
               as boreholes (list of ids from top (youngest) 
               to bottom(oldest)). The float is the percentage 
               of boreholes in aggreement with this pile
    """
    
    def create_table(l_bhs, n_units):
    
        M = np.zeros((n_units, n_units), dtype=int)

        for ibh in range(len(l_bhs)):

            bh = l_bhs[ibh]
            compatible = True

            #check compatibility first
            for i in range(len(bh) - 1, 0, -1):

                s1 = bh[i]
                s2 = bh[i-1]  # s2 is above s1
                if M[s2, s1] <= -1 or (M[M[s1] >= 1, s2] > 0).any():  # incorrect info
                    compatible = False
                    break

            if compatible :
                # loop over the log
                for i in range(len(bh) - 1, 0, -1):

                    s1 = bh[i]
                    s2 = bh[i-1]  # s2 is above s1

                    if M[s2, s1] >= 0:

                        M[s2, s1] += 1
                        M[s1, s2] += -1
                        M[s2, M[s1] >= 1] += 1
                        M[M[s1] >= 1, s2] += -1   

                M[0, 0] += 1
            else:
                pass
    
        return M
        
        
    ## algo
    l_M = []  # list of all the possible connexion tables

    # how many units present
    list_ids = []
    for ibh in l_bhs:
        for un in ibh:
            if un not in list_ids:
                list_ids.append(un)
    n_units = len(list_ids)


    #matrice
    M = np.zeros((n_units, n_units), dtype=int)
    l_M.append(M)

    for ibh in range(len(l_bhs[:])):

        bh = l_bhs[ibh]
        compatible = True

        all_incompatible = False
        
        iM = -1
        for Mi in l_M:
            
            iM += 1
            compatible = True   

            #check compatibility first
            for i in range(len(bh) - 1, 0, -1):  # loop in the log
                s1 = bh[i]
                s2 = bh[i-1]  # s2 is above s1
                if Mi[s2, s1] <= -1 or (Mi[Mi[s1] >= 1, s2] > 0).any():  # incorrect info
                    compatible = False  # not compatible
                    break  # bh not compatible with Mi --> break and go to next table

            if compatible:
                # loop over the log
                for i in range(len(bh) - 1, 0, -1):

                    s1 = bh[i]
                    s2 = bh[i-1]  # s2 is above s1
                    
                    Mi[s2, s1] += 1
                    Mi[s1, s2] += -1

                    Mi[s2, Mi[s1] >= 1] += 1
                    Mi[Mi[s1] >= 1, s2] += -1                       
        
                Mi[0, 0] += 1
                all_incompatible = True  # at least one table match the borehole

            else:
                pass

        if not all_incompatible:  # if no table match the borehole, create a new one with previous bhs
            new_M = create_table([bh] + l_bhs[:ibh-1], n_units)  
            l_M.append(new_M)  # append the new table

    
    # process tables to get piles
    n_piles = len(l_M)
    
    # order piles by % of bhs
    ids_piles = np.arange(n_piles, dtype=int)
    nbh_piles = np.array(l_M)[:, 0, 0]
    a = np.array((ids_piles, nbh_piles)).T
    a = a[a[:, 1].argsort()[::-1]]
    
    mask = ~np.eye(n_units, dtype=bool)  # mask non-diagonal elements
    results = []
    new_l_M = []
    for id_table in a[:, 0]:
        definite = True
        Mi = l_M[id_table]
        new_l_M.append(Mi)
        
        if sum((Mi[mask] == 0)) > 2:
            definite = False
            
        b = np.zeros(n_units, dtype=int)
        b[0] -= 1

        order_max = (Mi > 0).sum(0) + b 
        order_min = n_units - (Mi > 0).sum(1) + b 

        # determine position(s) of each unit
        d_pos = {}
        unit = 0
        for lb, ub in zip(((Mi > 0).sum(0)), (n_units - (Mi > 0).sum(1))):
            if unit == 0:
                lb -= 1
                ub += 1
            pos = []
            for o in range(lb, ub):
                pos.append(o)
            d_pos[unit] = pos
            unit+=1

        pile_mi = []
        # ## defined units

        ### clean d_pos
        for k,values in d_pos.items():

            if len(values) == 1:
                v = values[0]

                for k2, values2 in d_pos.items():
                    if k != k2:
                        if v in values2:
                            values2.remove(v)


        for ipos in range(n_units):  # loop over each position of the pile
            c = 0
            val = None
            units_pos_i = []
            for k,values in d_pos.items():
                for v in values:
                    if v == ipos:
                        c += 1
                        k_pos = k


            if c == 1:
                pile_mi.append(k_pos)
                d_pos[k_pos] = []

            else:
                units_pos_i = []
                for k,values in d_pos.items():
                    if values:
                        for v in values:
                            if v == ipos:
                                units_pos_i.append(k)
                pile_mi.append(tuple(units_pos_i))   
                
#         for ipos in range(n_units):  # loop over each position of the pile
#             units_pos_i = []
#             for k,values in d_pos.items():
#                 for v in values:
#                     if v == ipos:
#                         units_pos_i.append(k)
#             pile_mi.append(tuple(units_pos_i))        

        results.append((pile_mi, np.round(100*(Mi[0, 0]/len(l_bhs)), decimals=2), "Definite = {}".format(definite)))
        i += 1
            
    return results, new_l_M