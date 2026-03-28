#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <simplified_renorm.hpp>

namespace py = pybind11;  // alias for convenience

// Forward declaration of the main function
double simplified_renorm2D(
    const std::vector<double>& arr,
    size_t nrow,
    size_t ncol,
    const std::vector<double>& dx,
    const std::vector<double>& dy,
    bool is_kx
);

double simplified_renorm2D_py(  // Wrapper function for Python
    py::array_t<double, py::array::c_style | py::array::forcecast> arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> dx,
    py::array_t<double, py::array::c_style | py::array::forcecast> dy,
    bool is_kx
){
    auto buf_arr = arr.request();
    auto buf_dx  = dx.request();
    auto buf_dy  = dy.request();

    if (buf_arr.ndim != 2)
        throw std::runtime_error("arr must be 2D");

    size_t nrow = buf_arr.shape[0];
    size_t ncol = buf_arr.shape[1];

    // Convert to std::vector
    std::vector<double> arr_vec(
        (double*)buf_arr.ptr,
        (double*)buf_arr.ptr + nrow * ncol
    );

    std::vector<double> dx_vec(
        (double*)buf_dx.ptr,
        (double*)buf_dx.ptr + ncol
    );

    std::vector<double> dy_vec(
        (double*)buf_dy.ptr,
        (double*)buf_dy.ptr + nrow
    );

    return simplified_renorm2D(arr_vec, nrow, ncol, dx_vec, dy_vec, is_kx);
}

// 3D
double simplified_renorm3D(
    const std::vector<double>& arr,
    size_t nrow,
    size_t ncol,
    size_t nz,
    const std::vector<double>& dx,
    const std::vector<double>& dy,
    const std::vector<double>& dz,
    int axis // 0=x, 1=y, 2=z
);

double simplified_renorm3D_py(  // Wrapper function for Python
    py::array_t<double, py::array::c_style | py::array::forcecast> arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> dx,
    py::array_t<double, py::array::c_style | py::array::forcecast> dy,
    py::array_t<double, py::array::c_style | py::array::forcecast> dz,
    int axis
){
    auto buf_arr = arr.request();
    auto buf_dx  = dx.request();
    auto buf_dy  = dy.request();
    auto buf_dz  = dz.request();

    if (buf_arr.ndim != 3)
        throw std::runtime_error("arr must be 3D");

    size_t nz = buf_arr.shape[0];
    size_t nrow = buf_arr.shape[1];
    size_t ncol = buf_arr.shape[2];

    // Convert to std::vector
    std::vector<double> arr_vec(
        (double*)buf_arr.ptr,
        (double*)buf_arr.ptr + nz * nrow * ncol
    );

    std::vector<double> dx_vec(
        (double*)buf_dx.ptr,
        (double*)buf_dx.ptr + ncol
    );

    std::vector<double> dy_vec(
        (double*)buf_dy.ptr,
        (double*)buf_dy.ptr + nrow
    );

    std::vector<double> dz_vec(
        (double*)buf_dz.ptr,
        (double*)buf_dz.ptr + nz
    );

    return simplified_renorm3D(arr_vec, nrow, ncol, nz, dx_vec, dy_vec, dz_vec, axis);
}

PYBIND11_MODULE(simplified_renorm_C, m) {

    m.def("simplified_renorm2D", &simplified_renorm2D_py,
          "Compute renormalized 2D value",
          py::arg("arr"),
          py::arg("dx"),
          py::arg("dy"),
          py::arg("is_kx"));

    m.def("simplified_renorm3D", &simplified_renorm3D_py,
          "Compute renormalized 3D value",
            py::arg("arr"),
            py::arg("dx"),
            py::arg("dy"),
            py::arg("dz"),
            py::arg("axis"));
}