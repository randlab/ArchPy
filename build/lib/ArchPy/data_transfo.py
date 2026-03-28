import numpy as np
import scipy
from scipy.stats import norm
from scipy.stats import uniform

## Functions from ESMDA paper
def ecdf(data):
    # Compute ECDF
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)

from scipy.stats import norm
from scipy.interpolate import interp1d

class N_transform():

    def __init__(self, min_val = None, max_val = None):
        self.min_val = min_val
        self.max_val = max_val


    def fit(self, data):
        size = data.shape

        x,y = ecdf(data)
        y[y>0.9999] = 0.9999
        y[y<0.0001] = 0.0001

        x, idx = np.unique(x,return_index=True)
        y = y[idx]

        distrib = norm()
        ppf = distrib.ppf(y)

        self.func = interp1d(x,ppf,kind='linear',fill_value="extrapolate",bounds_error=False)

        if self.min_val is not None:
            ppf = np.insert(ppf,0,min(ppf) *20)
            x = np.insert(x,0,self.min_val)

        if self.max_val is not None:
            ppf = np.append(ppf,max(ppf)*20)
            x = np.append(x,self.max_val)

        self.func_inv = interp1d(ppf,x,kind='linear',fill_value="extrapolate",bounds_error=False)

## Old functions
class distri():

    """
    Distribution : must have a direct function (cdf) and its inverse (ppf)

    Parameters
    ----------
    f : function
        cdf of the distribution
    f_1 : function
        inverse of the cdf of the distribution

    """

    def __init__(self,f,f_1):

        self.f = f
        self.f_1 = f_1


### N_transfo ###
def store_distri(data,t=0):

    """

    Parameters
    ----------
    data : nd.array
        data that we want to estimate the distribution
    t : float
        threshold to take extreme values not in dataset into account (percentage of values that doesn't appear in the dataset)
        For a dataset of ten values ranging from 1 to 10, a value of 0.2 indicates that
        the distribution takes into account the values 0 and 11, even they are not in the dataset.
    return : distri object

    """

    class distri():

        def __init__(self,f,f_1):

            self.f = f
            self.f_1 = f_1

    sorted_data = np.sort(data)
    n = data.shape[0]

    if t> 0:
        #ecdf
        cumsum = np.ones(n+2)
        ini_val = np.ones(n+2)

        # first and last values
        cumsum[0] = 0
        ini_val[0] = sorted_data[0] - t*(np.max(data)-np.min(data))
        cumsum[-1] = 1
        ini_val[-1] = sorted_data[-1] + t*(np.max(data)-np.min(data))

        cum = t/2
        for i,idata in enumerate(np.unique(data)):
            frac = np.sum(data==idata)/(n+t*n)
            cum += frac
            cumsum[i+1] = cum
            ini_val[i+1] = idata

    elif t==0:
        #ecdf
        cumsum = np.ones(n)
        ini_val = np.ones(n)
        cum = 0
        for i,idata in enumerate(np.unique(data)):
            frac = np.sum(data==idata)/(n)
            cum += frac
            cumsum[i] = cum
            ini_val[i] = idata

    #plt.plot(ini_val,cumsum)
    f = scipy.interpolate.interp1d(ini_val,cumsum,fill_value="extrapolate")
    f_1 = scipy.interpolate.interp1d(cumsum,ini_val,fill_value="extrapolate")

    di = distri(f,f_1)
    return di

def NScore_trsf(data,di):

    """
    Transform data distributed as di into a normal distribution N(0,1)

    Parameters
    ----------

    data: nd.array
        data that we want to transform
    di: :class:`distri`
        distribution of the data

    Return
    ------
    norm_val: nd.array
        data transformed into a normal distribution N(0,1)
    """

    f = di.f

    norm_val = np.ones([data.shape[0]])
    for i,idata in enumerate(data):
        G = f(idata)
        val = norm.ppf(G)
        if G > 0.999:
            norm_val[i] = norm.ppf(0.999) # youpi
        elif G < 0.001:
            norm_val[i] = norm.ppf(0.001) # youpi
        else :
            norm_val[i] = norm.ppf(f(idata))

    return norm_val

def NScore_Btrsf(data_transformed,di):

    """
    Back transform normal distributed data into original distribution di

    Parameters
    ----------
    data_transformed : nd.array
        data normally distributed that we want to back transform
    di: :class:`distri`
        distribution of the data

    Return
    ------
    x_retransformed : nd.array  
        data back transformed into original distribution di
    """

    f_1 = di.f_1
    norm_cdf = norm.cdf(data_transformed)
    x_retransformed = f_1(norm_cdf)

    return x_retransformed

