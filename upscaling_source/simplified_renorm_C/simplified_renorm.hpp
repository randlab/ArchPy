#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>

inline size_t idx(size_t i, size_t j, size_t ncol);

void f_new2(
    const std::vector<double>& arr,
    size_t nrow,
    size_t ncol,
    const std::vector<double>& dx,
    const std::vector<double>& dy,
    bool is_x_axis,
    bool is_arithmetic,
    size_t& out_rows,
    size_t& out_cols,
    std::vector<double>& out_arr,   // now output buffer
    std::vector<double>& out_dx,
    std::vector<double>& out_dy);

void f_new(
    const std::vector<double>& arr,
    size_t nrow,
    size_t ncol,
    const std::vector<double>& dx,
    const std::vector<double>& dy,
    bool is_x_axis,
    bool is_arithmetic,
    size_t& out_rows,
    size_t& out_cols,
    std::vector<double>& out_arr,   // now output buffer
    std::vector<double>& out_dx,
    std::vector<double>& out_dy);

double find_c2D(
    const std::vector<double>& arr,
    size_t nrow,
    size_t ncol,
    const std::vector<double>& dx,
    const std::vector<double>& dy,
    bool is_x_direction,
    bool is_arithmetic
);

double get_alpha_2D(
    double dx,
    double dy,
    bool is_x_direction
);

double keq(
    double cmin,
    double cmax,
    double alpha
    );

double simplified_renorm2D(
    const std::vector<double>& arr,
    size_t nrow,
    size_t ncol,
    const std::vector<double>& dx,
    const std::vector<double>& dy,
    bool is_kx);

inline size_t idx3(size_t k, size_t i, size_t j,
                   size_t nrow, size_t ncol);

void f_new_3D(
    const std::vector<double>& arr,
    size_t nz, size_t nrow, size_t ncol,
    const std::vector<double>& dx,
    const std::vector<double>& dy,
    const std::vector<double>& dz,
    int axis, // 0=x, 1=y, 2=z
    bool is_arithmetic,
    size_t& out_nz,
    size_t& out_rows,
    size_t& out_cols,
    std::vector<double>& out_arr,
    std::vector<double>& out_dx,
    std::vector<double>& out_dy,
    std::vector<double>& out_dz);

double find_c3D(
    const std::vector<double>& arr,
    size_t nz,
    size_t nrow, 
    size_t ncol,
    const std::vector<double>& dx,
    const std::vector<double>& dy,
    const std::vector<double>& dz,
    int direction, // direction to obtain c (0, 1, 2) ie (x, y, z)
    bool is_max);

double get_alpha_3D(double dx, double dy, double dz, int direction);

double simplified_renorm3D(const std::vector<double>& arr,
    size_t nrow,
    size_t ncol,
    size_t nlay,
    const std::vector<double>& dx,
    const std::vector<double>& dy,
    const std::vector<double>& dz,
    int direction);