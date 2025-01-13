# Python module to upscale hydraulic conductivity fields
import numpy as np
import matplotlib.pyplot as plt
import geone
import geone.covModel as gcm
from numba import jit

## 2D algorithms ##
# TO DO

## 3D algorithms ##
# Simplified renormalization algorithm

# mean and harmonic mean functions
@jit()
def hmean(v1, v2, w1=1, w2=1):
    return (w1 + w2) / (w1 / v1 + w2 / v2)

@jit()
def mean(v1, v2, w1=1, w2=1):
    return (v1 * w1 + v2 * w2) / (w1 + w2)

def gmean(a):
    if np.all(a > 0):
        return np.exp(np.mean(np.log(a)))
    elif np.all(a < 0):
        a = np.abs(a)
        return -np.exp(np.mean(np.log(a)))
    else:
        raise ValueError("Array must be all positive or all negative")

def fill_nan_values_with_gmean(a):
    
    """
    Function to fill nan values with the geometric mean of the non-nan values
    """

    a = np.copy(a)
    mask = np.isnan(a)
    a[mask] = gmean(a[~mask])
    return a

# function to operate one mean operation
def f_3D(a, dx=None, dy=None, dz=None, direction="x", typ="x"):

    """
    Function to operate a mean between two 2D fields from a 3D field

    Parameters
    ----------
    a : 3D numpy array
        The 3D field
    direction : str
        The direction of the mean to compute. Can be "x", "y" or "z"
    type : str
        The direction where fields are selected. Can be "x", "y" or "z"
    """

    # initialize weights
    w1 = 1
    w2 = 1
    new_dx = None
    new_dy = None
    new_dz = None

    if typ == "y":

        v1 = a[:, ::2, :]
        v2 = a[:, 1::2, :]
        if dy is not None:
            w1 = dy[:, ::2, :]
            w2 = dy[:, 1::2, :]
            new_dy = w1 + w2
            new_dx = dx[:, ::2, :]
            new_dz = dz[:, ::2, :]

    elif typ == "x":

        v1 = a[:, :, ::2]
        v2 = a[:, :, 1::2]
        if dx is not None:
            w1 = dx[:, :, ::2]
            w2 = dx[:, :, 1::2]
            new_dx = w1 + w2
            new_dy = dy[:, :, ::2]
            new_dz = dz[:, :, ::2]

    elif typ == "z":
            
        v1 = a[::2, :, :]
        v2 = a[1::2, :, :]
        if dz is not None:
            w1 = dz[::2, :, :]
            w2 = dz[1::2, :, :]
            new_dz = w1 + w2
            new_dx = dx[::2, :, :]
            new_dy = dy[::2, :, :]


    if typ == direction:  # series
        res = hmean(v1, v2, w1=w1, w2=w2)
    else:
        res = mean(v1, v2, w1=w1, w2=w2)
    
    return res, new_dx, new_dy, new_dz

def test_f():

    """
    Test the function f
    """
    a = np.array([[[0.51394334, 0.77316505],
                   [0.87042769, 0.00804695]],
                  [[0.30973593, 0.95760374],
                   [0.51311671, 0.31828442]]])

    a1 = f_3D(a, direction="x", typ="x")[0]
    a2 = f_3D(a1, direction="x", typ="z")[0]
    a3 = f_3D(a2, direction="x", typ="y")[0]
    res = a3[0, 0, 0]
    assert np.allclose(res, 0.3735857397539791)
    
    a1 = f_3D(a, direction="x", typ="y")[0]
    a2 = f_3D(a1, direction="x", typ="z")[0]
    a3 = f_3D(a2, direction="x", typ="x")[0]
    res = a3[0, 0, 0]
    assert np.allclose(res, 0.5323798506575812)

    print("Test passed")

def merge(v1, v2, w1, w2, normalize=True):
    
    """
    Function to merge two vectors v1 and v2 with weights w1 and w2
    Is used here to merge two columns of different sizes
    """

    if normalize:
        sum_w = w1 + w2
        w1 = w1 / sum_w
        w2 = w2 / sum_w

    return w1*v1 + w2*v2 - (w1*w2*(v1 - v2)**2)/(w1*v2 + w2*v1)


def find_c_3D(k_field, direction="x", typ="min", dx=None, dy=None, dz=None):
    
    if typ == "min":
        if direction == "x":
            l_typ = ["x", "y", "z"]
        elif direction == "y":
            l_typ = ["y", "x", "z"]
        elif direction == "z":
            l_typ = ["z", "x", "y"]
    elif typ == "max":
        if direction == "x":
            l_typ = ["y", "z", "x"]
        elif direction == "y":
            l_typ = ["x", "z", "y"]
        elif direction == "z":
            l_typ = ["x", "y", "z"]

    # if dx, dy and dz are None, set them to 1
    if dx is None:
        dx = 1
    if dy is None:
        dy = 1
    if dz is None:
        dz = 1

    # if dx and dy are provided as a scalar, set them to an array of the same shape as k_field
    if isinstance(dx, (int, float)):
        dx = np.ones(k_field.shape) * dx
    if isinstance(dy, (int, float)):
        dy = np.ones(k_field.shape) * dy
    if isinstance(dz, (int, float)):
        dz = np.ones(k_field.shape) * dz

    a = np.copy(k_field)
    o = 0
    while a.flatten().shape[0] > 1:
        # print("dx:", dx)
        # print("dy:", dy)
        # print("dz:", dz)
        # print("asdf", a)
        # check if the number of elements is odd
        rm_col = False
        rm_row = False
        rm_lay = False
        rm_dx = None
        rm_dy = None
        rm_dz = None
        sim = True
        if l_typ[o] == "x":
            if a.shape[2] > 1:
                if a.shape[2] % 2 == 1:
                    # remove the last column
                    rm_col = a[:, :, -1]
                    a = a[:, :, :-1]
                    rm_dx = dx[:, :, -1]
                    dx = dx[:, :, :-1]
                    rm_dy = dy[:, :, -1]
                    dy = dy[:, :, :-1]
                    rm_dz = dz[:, :, -1]
                    dz = dz[:, :, :-1]
            else:
                sim = False

        elif l_typ[o] == "y":
            if a.shape[1] > 1:
                if a.shape[1] % 2 == 1:
                    # remove the last row
                    rm_row = a[:, -1, :]
                    a = a[:, :-1, :]
                    rm_dx = dx[:, -1, :]
                    dx = dx[:, :-1, :]
                    rm_dy = dy[:, -1, :]
                    dy = dy[:, :-1, :]
                    rm_dz = dz[:, -1, :]
                    dz = dz[:, :-1, :]
            else:
                sim = False

        elif l_typ[o] == "z":
            if a.shape[0] > 1:
                if a.shape[0] % 2 == 1:
                    # remove the last layer
                    rm_lay = a[-1, :, :]
                    a = a[:-1, :, :]
                    rm_dx = dx[-1, :, :]
                    dx = dx[:-1, :, :]
                    rm_dy = dy[-1, :, :]
                    dy = dy[:-1, :, :]
                    rm_dz = dz[-1, :, :]
                    dz = dz[:-1, :, :]
            else:
                sim = False
        if sim:
            a, dx, dy, dz = f_3D(a, direction=direction, typ=l_typ[o], dx=dx, dy=dy, dz=dz)

        # merge the removed column, row or layer
        if rm_col is not False:
            # new_col = merge(rm_col, a[:, :, -1], 1, 2)
            a = np.dstack((a, rm_col.reshape(a.shape[0], a.shape[1], 1)))
            dx = np.dstack((dx, rm_dx.reshape(dx.shape[0], dx.shape[1], 1)))
            dy = np.dstack((dy, rm_dy.reshape(dy.shape[0], dy.shape[1], 1)))
            dz = np.dstack((dz, rm_dz.reshape(dz.shape[0], dz.shape[1], 1)))
        elif rm_row is not False:
            # new_row = merge(rm_row, a[:, -1, :], 1, 2)
            a = np.hstack((a, rm_row.reshape(a.shape[0], 1, a.shape[2])))
            dx = np.hstack((dx, rm_dx.reshape(dx.shape[0], 1, dx.shape[2])))
            dy = np.hstack((dy, rm_dy.reshape(dy.shape[0], 1, dy.shape[2])))
            dz = np.hstack((dz, rm_dz.reshape(dz.shape[0], 1, dz.shape[2])))
        elif rm_lay is not False:
            # new_lay = merge(rm_lay, a[-1, :, :], 1, 2)
            a = np.vstack((a, rm_lay.reshape(1, a.shape[1], a.shape[2])))
            dx = np.vstack((dx, rm_dx.reshape(1, dx.shape[1], dx.shape[2])))
            dy = np.vstack((dy, rm_dy.reshape(1, dy.shape[1], dy.shape[2])))
            dz = np.vstack((dz, rm_dz.reshape(1, dz.shape[1], dz.shape[2])))

        o += 1
        if o == 3:
            o = 0

    return a[0, 0, 0]

def test_find_c():

    b = np.array([[[0.51394334, 0.77316505, 0.87042769],
        [0.00804695, 0.30973593, 0.95760374],
        [0.51311671, 0.31828442, 0.53919994]],

       [[0.22125494, 0.80648136, 0.34225463],
        [0.53888885, 0.00587379, 0.67315248],
        [0.21002426, 0.93255759, 0.37424475]],

       [[0.75241892, 0.763139  , 0.87049946],
        [0.11011118, 0.30009198, 0.47490577],
        [0.67293672, 0.25759243, 0.70115132]]])

    cmin = find_c_3D(b, direction="x", typ="min", dx=1, dy=1, dz=1)
    cmax = find_c_3D(b, direction="x", typ="max", dx=1, dy=1, dz=1)
    assert np.allclose(cmin, 0.3914114)
    assert np.allclose(cmax, 0.47437664)

    print("Test passed")

# now keq is a function of cmin and cmax according to keq = cmax**alpha*(cmin)**(1-alpha)
def get_alpha(dx, dy, dz=None, direction="x"):
    
    def f_u(t):
        return np.arctan(np.sqrt(t)) / (np.pi/2)

    if dz is None:
        if direction == "x":
            a = dx/dy
        elif direction == "y":
            a = dy/dx
        
        u = f_u(a)
        return u

    if direction == "x":
        a_1 = dx/dy
        a_2 = dx/dz
    elif direction == "y":
        a_1 = dy/dx
        a_2 = dy/dz
    elif direction == "z":
        a_1 = dz/dx
        a_2 = dz/dy
        
    u_1 = f_u(a_1)
    u_2 = f_u(a_2)

    return ((u_1 + u_2) - 2*u_1*u_2) / (1 - u_1*u_2)
    
def keq(cmin, cmax, alpha):

    res = (np.abs(cmax)**alpha)*np.abs(cmin)**(1-alpha)
    if cmax and cmin < 0:
        res = -res

    return res

def simplified_renormalization(field, dx, dy, dz, direction="x"):

    cx_min, cx_max = find_c_3D(field, direction=direction, typ="min", dx=dx, dy=dy, dz=dz), find_c_3D(field, direction=direction, typ="max", dx=dx, dy=dy, dz=dz)
    return keq(cx_min, cx_max, get_alpha(dx, dy, dz, direction=direction)) 

# Standard renormalization algorithm
def get_idx (i):
    
    if i == 1:
        idx = (0, 0, 0)
    elif i == 2:
        idx = (0, 0, 1)
    elif i == 3:
        idx = (0, 1, 0)
    elif i == 4:
        idx = (0, 1, 1)
    elif i == 5:
        idx = (1, 0, 0)
    elif i == 6:
        idx = (1, 0, 1)
    elif i == 7:
        idx = (1, 1, 0)
    elif i == 8:
        idx = (1, 1, 1)
    
    return idx

def f_sr(kxx, kyy, kzz, dx, dy, dz, direction="x"):

    if direction == "x":    
        pass
    elif direction == "y":
        # rotate the fields
        kxx = np.rot90(kxx, k=-1, axes=(1, 2))
        kyy = np.rot90(kyy, k=-1, axes=(1, 2))
        kzz = np.rot90(kzz, k=-1, axes=(1, 2))
        # adjust the dx and dy
        dx, dy = dy, dx
    elif direction == "z":
        # rotate the fields
        kxx = np.rot90(kxx, k=-1, axes=(0, 2))
        kyy = np.rot90(kyy, k=-1, axes=(0, 2))
        kzz = np.rot90(kzz, k=-1, axes=(0, 2))
        # adjust the dx and dz
        dx, dz = dz, dx
        
    def b1(i, u):

        idx = get_idx(i)

        if u == "x":
            du = dx
            ku = kxx[idx]
        elif u == "y":
            du = dy
            ku = kyy[idx]
        elif u == "z":
            du = dz
            ku = kzz[idx]

        return 2 * ku * (dx*dy*dz) / du**2

    def b2(i, j, u):
            
        idx = get_idx(i)
    
        idx2 = get_idx(j)
    
        if u == "x":
            du = dx
            ku = kxx[idx]
            ku2 = kxx[idx2]
        elif u == "y":
            du = dy
            ku = kyy[idx]
            ku2 = kyy[idx2]
        elif u == "z":
            du = dz
            ku = kzz[idx]
            ku2 = kzz[idx2]
    
        return ((2 * ku * ku2) /(ku + ku2)) * ((dx*dy*dz) / du**2)

    def d(i, j, k, l):
        return b1(i, "x") + b2(i, j, "x") + b2(i, k, "y") + b2(i, l, "z")


    A = np.array(
        [
            [d(1, 2, 3, 5), -b2(1, 2, "x"), -b2(1, 3, "y"), 0, -b2(1, 5, "z"), 0, 0, 0],
            [-b2(2, 1, "x"), d(2, 1, 4, 6), 0, -b2(2, 4, "y"), 0, -b2(2, 6, "z"), 0, 0],
            [-b2(3, 1, "y"), 0, d(3, 4, 1, 7), -b2(3, 4, "x"), 0, 0, -b2(3, 7, "z"), 0],
            [0, -b2(4, 2, "y"), -b2(4, 3, "x"), d(4, 3, 2, 8), 0, 0, 0, -b2(4, 8, "z")],
            [-b2(5, 1, "z"), 0, 0, 0, d(5, 6, 7, 1), -b2(5, 6, "x"), -b2(5, 7, "y"), 0],
            [0, -b2(6, 2, "z"), 0, 0, -b2(6, 5, "x"), d(6, 5, 8, 2), 0, -b2(6, 8, "y")],
            [0, 0, -b2(7, 3, "z"), 0, -b2(7, 5, "y"), 0, d(7, 8, 5, 3), -b2(7, 8, "x")],
            [0, 0, 0, -b2(8, 4, "z"), 0, -b2(8, 6, "y"), -b2(8, 7, "x"), d(8, 7, 6, 4)]
        ])
    
    b = np.array([b1(1, "x"), 0, b1(3, "x"), 0, b1(5, "x"), 0, b1(7, "x"), 0])	

    h = np.linalg.solve(A, b)
    K = (h[1::2] * kxx[:, :, -1].flatten()).sum()

    return K

# standard renormalization with direct scheme
def f_sr_center(kxx, kyy, kzz, dx, dy, dz):

    def b1(i, u):

        idx = get_idx(i)

        if u == "x":
            du = dx
            ku = kxx[idx]
        elif u == "y":
            du = dy
            ku = kyy[idx]
        elif u == "z":
            du = dz
            ku = kzz[idx]

        return 2 * ku * (dx*dy*dz) / du**2

    def b2(i, j, u):
            
        idx = get_idx(i)
    
        idx2 = get_idx(j)
    
        if u == "x":
            du = dx
            ku = kxx[idx]
            ku2 = kxx[idx2]
        elif u == "y":
            du = dy
            ku = kyy[idx]
            ku2 = kyy[idx2]
        elif u == "z":
            du = dz
            ku = kzz[idx]
            ku2 = kzz[idx2]
    
        return (2 * ku * ku2) /(ku + ku2) * ((dx*dy*dz) / du**2)

    # Kxx
    def d(i, j, k, l):
        return b1(i, "x") + b1(j, "x") + b1(k, "y") + b1(l, "z")

    A = np.array([[d(2, 1, 2, 2), -b1(2, "y"), -b1(2, "z"), 0],
                  [-b1(2, "y"), d(4, 3, 2, 4), 0, -b1(4, "z")],
                  [-b1(2, "z"), 0, d(6, 5, 6, 2), -b1(6, "y")],
                  [0, -b1(4, "z"), -b1(6, "y"), d(8, 7, 6, 4)]])

    b = np.array([b1(1, "x"), b1(3, "x"), b1(5, "x"), b1(7, "x")])

    h = np.linalg.solve(A, b)
    Kxx = 0.5 * (h * kxx[:, :, -1].flatten()).sum()

    # Kyy
    def d(i, j, k, l):
        return b1(i, "y") + b1(j, "x") + b1(k, "y") + b1(l, "z")
    
    A = np.array([[d(3, 3, 1, 3), -b1(3, "x"), -b1(3, "z"), 0],
                    [-b1(3, "x"), d(2, 3, 2, 4), 0, -b1(4, "z")],
                    [-b1(3, "z"), 0, d(7, 7, 5, 3), -b1(7, "x")],
                    [0, -b1(4, "z"), -b1(7, "x"), d(8, 7, 6, 4)]])

    b = np.array([b1(1, "y"), b1(2, "y"), b1(5, "y"), b1(6, "y")])

    h = np.linalg.solve(A, b)
    Kyy = 0.5 * (h * kyy[:, -1, :].flatten()).sum()

    # Kzz
    def d(i, j, k, l):
        return b1(i, "z") + b1(j, "x") + b1(k, "y") + b1(l, "z")

    A = np.array([[d(5, 5, 5, 1), -b1(5, "x"), -b1(5, "y"), 0],
                    [-b1(5, "x"), d(6, 5, 6, 2), 0, -b1(6, "y")],
                    [-b1(5, "y"), 0, d(7, 7, 5, 3), -b1(7, "x")],
                    [0, -b1(6, "y"), -b1(7, "x"), d(8, 7, 6, 4)]])

    b = np.array([b1(1, "z"), b1(2, "z"), b1(3, "z"), b1(4, "z")])

    h = np.linalg.solve(A, b)
    Kzz = 0.5 * (h * kzz[-1, :, :].flatten()).sum()

    return Kxx, Kyy, Kzz

def standard_renormalization(field_xx, field_yy, field_zz, dx, dy, dz, niter=1, scheme="direct"):
    
    assert field_xx.shape == field_yy.shape == field_zz.shape, "Field shapes must be the same"
    assert np.log2(field_xx.shape[0]).is_integer(), "Field size must be a power of 2"
    assert np.log2(field_xx.shape[1]).is_integer(), "Field size must be a power of 2"
    assert np.log2(field_xx.shape[2]).is_integer(), "Field size must be a power of 2"

    def one_iteration(field_xx, field_yy, field_zz):
        new_field_xx = np.zeros((field_xx.shape[0]//2, field_xx.shape[1]//2, field_xx.shape[2]//2))
        new_field_yy = np.zeros((field_xx.shape[0]//2, field_xx.shape[1]//2, field_xx.shape[2]//2))
        new_field_zz = np.zeros((field_xx.shape[0]//2, field_xx.shape[1]//2, field_xx.shape[2]//2))
        for i in range(0, field_xx.shape[0], 2):
            for j in range(0, field_xx.shape[1], 2):
                for k in range(0, field_xx.shape[2], 2):
                    block_xx = field_xx[i:i+2, j:j+2, k:k+2]
                    block_yy = field_yy[i:i+2, j:j+2, k:k+2]
                    block_zz = field_zz[i:i+2, j:j+2, k:k+2]

                    # fill nan values with the geometric mean
                    block_xx = fill_nan_values_with_gmean(block_xx)
                    block_yy = fill_nan_values_with_gmean(block_yy)
                    block_zz = fill_nan_values_with_gmean(block_zz)

                    if scheme == "direct":
                        kxx = f_sr(block_xx, block_yy, block_zz, dx, dy, dz, direction="x")
                        kyy = f_sr(block_yy, block_xx, block_zz, dy, dx, dz, direction="y")
                        kzz = f_sr(block_zz, block_xx, block_yy, dz, dx, dy, direction="z")
                    elif scheme == "center":
                        kxx, kyy, kzz = f_sr_center(block_xx, block_yy, block_zz, dx, dy, dz)
                    
                    new_field_xx[i//2, j//2, k//2] = kxx
                    new_field_yy[i//2, j//2, k//2] = kyy
                    new_field_zz[i//2, j//2, k//2] = kzz

        return new_field_xx, new_field_yy, new_field_zz

    # list_field_xx = [field_xx]
    # list_field_yy = [field_yy]
    # list_field_zz = [field_zz]

    for _ in range(niter):
        field_xx, field_yy, field_zz = one_iteration(field_xx, field_yy, field_zz)
        # list_field_xx.append(field_xx)
        # list_field_yy.append(field_yy)
        # list_field_zz.append(field_zz)

    return field_xx, field_yy, field_zz

# Tensorial renormalization algorithm
def f_rt_simple(kxx, kyy, kzz, dx, dy, dz):
    
    def b(i, u, v):

        idx = get_idx(i)
        if u != v:
            return 0

        elif u == "x" and v == "x":
            kuv = kxx
            du = dx
            dv = dx
        elif u == "y" and v == "y":
            kuv = kyy
            du = dy
            dv = dy
        elif u == "z" and v == "z":
            kuv = kzz
            du = dz
            dv = dz
        else:
            raise ValueError("Invalid indices")
        
        return kuv[idx] * (dx*dy*dz) / (du*dv)
    
    
    def d(i, j, k, l):
        return b(i, "x", "x") + b(i, "y", "y") + b(i, "z", "z") + b(j, "x", "x") + b(k, "y", "y") + b(l, "z", "z")

    def m1(i, j, u):
        return -b(i, u, u) - b(j, u, u)
    

    A = np.array(
        [
            [d(1, 2 ,3 ,5), m1(1, 2, "x"), m1(1, 3, "y"), 0, m1(1, 5, "z"), 0, 0],
            [m1(2, 1, "x"), d(2, 1, 4, 6), 0, m1(2, 4, "y"), 0, m1(2, 6, "z"), 0],
            [m1(3, 1, "y"), 0, d(3, 4, 1, 7), m1(3, 4, "x"), 0, 0, m1(3, 7, "z")],
            [0, m1(4, 2, "y"), m1(4, 3, "x"), d(4, 3, 2, 8), 0, 0, 0],
            [m1(5, 1, "z"), 0, 0, 0, d(5, 6, 7, 1), m1(5, 6, "x"), m1(5, 7, "y")],
            [0, m1(6, 2, "z"), 0, 0, m1(6, 5, "x"), d(6, 5, 8, 2), 0],
            [0, 0, m1(7, 3, "z"), 0, m1(7, 5, "y"), 0, d(7, 8, 5, 3)]
        ]
        )
    
    B = np.array(
        [
            [ b(2, "x", "x"),  b(3, "y", "y"),  b(5, "z", "z")],
            [-b(2, "x", "x"),  b(4, "y", "y"),  b(6, "z", "z")],
            [ b(4, "x", "x"), -b(3, "y", "y"),  b(7, "z", "z")],
            [-b(4, "x", "x"),- b(4, "y", "y"),  b(8, "z", "z")],
            [ b(6, "x", "x"),  b(7, "y", "y"), -b(5, "z", "z")],
            [-b(6, "x", "x"),  b(8, "y", "y"), -b(6, "z", "z")],
            [ b(8, "x", "x"), -b(7, "y", "y"), -b(7, "z", "z")]
        ]
        )

    T = np.array(
        [
            [b(2, "x", "x") + b(4, "x", "x") + b(6, "x", "x") + b(8, "x", "x"), 0, 0],
            [0, b(3, "y", "y") + b(4, "y", "y") + b(7, "y", "y") + b(8, "y", "y"), 0],
            [0, 0, b(5, "z", "z") + b(6, "z", "z") + b(7, "z", "z") + b(8, "z", "z")]    
        ])
    
    # solve the system of equations
    P = np.linalg.solve(A, B)
    dxdydz = np.array([[dx**2, 0, 0], [0, dy**2, 0], [0, 0, dz**2]])
    K = 0.5 * ((-np.dot(B.T, P) + T) @ dxdydz) * (1/dx * 1/dy * 1/dz) 
    return K

def tensorial_renormalization(field_xx, field_yy, field_zz, dx, dy, dz, niter=1):
    
    # assert that field is 2**n
    assert field_xx.shape == field_yy.shape == field_zz.shape, "Field shapes must be the same"
    # assert field_xx.shape[0] == field_xx.shape[1] == field_xx.shape[2]
    assert np.log2(field_xx.shape[0]).is_integer(), "Field size must be a power of 2"
    assert np.log2(field_xx.shape[1]).is_integer(), "Field size must be a power of 2"
    assert np.log2(field_xx.shape[2]).is_integer(), "Field size must be a power of 2"

    # assert np.log2(field.shape[0]).is_integer(), "Field size must be a power of 2
    
    def one_iteration(field_xx, field_yy, field_zz):
        new_field_xx = np.zeros((field_xx.shape[0]//2, field_xx.shape[1]//2, field_xx.shape[2]//2))
        new_field_yy = np.zeros((field_xx.shape[0]//2, field_xx.shape[1]//2, field_xx.shape[2]//2))
        new_field_zz = np.zeros((field_xx.shape[0]//2, field_xx.shape[1]//2, field_xx.shape[2]//2))
        for i in range(0, field_xx.shape[0], 2):
            for j in range(0, field_xx.shape[1], 2):
                for k in range(0, field_xx.shape[2], 2):
                    block_xx = field_xx[i:i+2, j:j+2, k:k+2]
                    block_yy = field_yy[i:i+2, j:j+2, k:k+2]
                    block_zz = field_zz[i:i+2, j:j+2, k:k+2]

                    # fill nan values with the geometric mean
                    block_xx = fill_nan_values_with_gmean(block_xx)
                    block_yy = fill_nan_values_with_gmean(block_yy)
                    block_zz = fill_nan_values_with_gmean(block_zz)

                    K = f_rt_simple(block_xx, block_yy, block_zz, dx, dy, dz)
                    kxx = K[0, 0]
                    kyy = K[1, 1]
                    kzz = K[2, 2]
                    
                    new_field_xx[i//2, j//2, k//2] = kxx
                    new_field_yy[i//2, j//2, k//2] = kyy
                    new_field_zz[i//2, j//2, k//2] = kzz

        return new_field_xx, new_field_yy, new_field_zz

    # list_field_xx = [field_xx]
    # list_field_yy = [field_yy]
    # list_field_zz = [field_zz]

    for _ in range(niter):
        field_xx, field_yy, field_zz = one_iteration(field_xx, field_yy, field_zz)
        # list_field_xx.append(field_xx)
        # list_field_yy.append(field_yy)
        # list_field_zz.append(field_zz)

    return field_xx, field_yy, field_zz



def upscale_k(field, dx=1, dy=1, dz=1, method="simplified_renormalization", factor_x=2, factor_y=2, factor_z=2, grid=None, scheme="center"):

    """
    Function to upscale a field using renormalization methods and average methods

    Parameters
    ----------
    field : np.ndarray
        3D isotropic hydraulic conductivity field
    dx : float or np.ndarray
        Cell size in the x direction
    dy : float or np.ndarray
        Cell size in the y direction
    dz : float or np.ndarray
        Cell size in the z direction
    method : str
        Renormalization method. Options are simplified_renormalization, standard_renormalization, tensorial_renormalization, arithmetic, harmonic and geometric
    factor_x : int
        Upscaling factor in the x direction. This means that the new field will have a shape divided by factor_x in the x direction
        Must be a power of 2 for standard and tensorial renormalization. It should be noted that field shape in the x direction must be a multiple of factor_x
    factor_y : int
        Upscaling factor in the y direction. This means that the new field will have a shape divided by factor_y in the y direction
        Must be a power of 2 for standard and tensorial renormalization. It should be noted that field shape in the y direction must be a multiple of factor_y
    factor_z : int
        Upscaling factor in the z direction. This means that the new field will have a shape divided by factor_z in the z direction
        Must be a power of 2 for standard and tensorial renormalization. It should be noted that field shape in the z direction must be a multiple of factor_z
    grid : np.ndarray
        Modflow dis grid. Only works with simplified_renormalization, arithmetic, harmonic and geometric. 
        Not implemented yet.
    scheme : str
        Scheme for standard renormalization. Options are direct and center. Center is generally more accurate and faster. Default is center
    """

    assert field.shape[1] % factor_y == 0, "factor_y must be a divisor of the field size"
    assert field.shape[2] % factor_x == 0, "factor_x must be a divisor of the field size"
    assert field.shape[0] % factor_z == 0, "factor_z must be a divisor of the field size"

    if method == "simplified_renormalization":
        
        if grid is None:
            new_field_kxx = np.zeros((field.shape[0]//factor_z, field.shape[1]//factor_y, field.shape[2]//factor_x))
            new_field_kyy = np.zeros((field.shape[0]//factor_z, field.shape[1]//factor_y, field.shape[2]//factor_x))
            new_field_kzz = np.zeros((field.shape[0]//factor_z, field.shape[1]//factor_y, field.shape[2]//factor_x))

            for i in range(0, field.shape[0], factor_z):
                for j in range(0, field.shape[1], factor_y):
                    for k in range(0, field.shape[2], factor_x):
                        selected_area = field[i:i+factor_z, j:j+factor_y, k:k+factor_z]

                        # fill nan values with the geometric mean of the selected area
                        selected_area = fill_nan_values_with_gmean(selected_area)

                        Kxx = simplified_renormalization(selected_area, dx, dy, dz, direction="x")
                        Kyy = simplified_renormalization(selected_area, dx, dy, dz, direction="y")
                        Kzz = simplified_renormalization(selected_area, dx, dy, dz, direction="z")

                        new_field_kxx[i//factor_z, j//factor_y, k//factor_x] = Kxx
                        new_field_kyy[i//factor_z, j//factor_y, k//factor_x] = Kyy
                        new_field_kzz[i//factor_z, j//factor_y, k//factor_x] = Kzz

        return new_field_kxx, new_field_kyy, new_field_kzz

    elif method == "standard_renormalization":
        assert factor_x == factor_y == factor_z, "factor_x, factor_y and factor_z must be equal for standard renormalization"
        assert np.log2(factor_x).is_integer(), "factor_x must be a power of 2"

        field_kxx, field_kyy, field_kzz = standard_renormalization(field, field, field, dx, dy, dz, niter=int(np.log2(factor_x)), scheme=scheme)
        return field_kxx, field_kyy, field_kzz
    
    elif method == "tensorial_renormalization":
        assert factor_x == factor_y == factor_z, "factor_x, factor_y and factor_z must be equal for tensorial renormalization"
        assert np.log2(factor_x).is_integer(), "factor_x must be a power of 2"

        field_kxx, field_kyy, field_kzz = tensorial_renormalization(field, field, field, dx, dy, dz, niter=int(np.log2(factor_x)))
        return field_kxx, field_kyy, field_kzz

    elif method in ["arithmetic", "harmonic", "geometric"]:

        if grid is None:
            new_field = np.zeros((field.shape[0]//factor_z, field.shape[1]//factor_y, field.shape[2]//factor_x))

            for i in range(0, field.shape[0], factor_z):
                for j in range(0, field.shape[1], factor_y):
                    for k in range(0, field.shape[2], factor_x):
                        selected_area = field[i:i+factor_z, j:j+factor_y, k:k+factor_x]

                        # remove nan values
                        selected_area = selected_area[~np.isnan(selected_area)]

                        if method == "arithmetic":
                            K = np.mean(selected_area.flatten())
                        elif method == "harmonic":
                            K = 1 / np.mean(1 / selected_area.flatten())
                        elif method == "geometric":
                            if np.all(selected_area < 0):
                                selected_area = np.abs(selected_area)
                                K = -np.exp(np.mean(np.log(selected_area.flatten())))
                            else:
                                K = np.exp(np.mean(np.log(selected_area.flatten())))
                        new_field[i//factor_z, j//factor_y, k//factor_x] = K

        return new_field

    else:
        raise ValueError("method must be simplified_renormalization, standard_renormalization, tensorial_renormalization, arithmetic, harmonic or geometric")