#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include "simplified_renorm.hpp"

// 2D functions

inline size_t idx(size_t i, size_t j, size_t ncol) {
    return i * ncol + j;
}


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
    std::vector<double>& out_dy)
{

    /*
    This function computes the next level of the renormalization process by averaging pairs of values along either the x or y axis, depending on the is_x_axis flag. 
    The type of averaging (arithmetic or harmonic) is determined by the is_arithmetic flag.
     The function also computes the new grid spacings (dx and dy) for the next level.
      The output is stored in out_arr, out_dx, and out_dy, and the dimensions of the output array are returned via out_rows and out_cols.
    */

    if (is_x_axis) {
        out_rows = nrow;
        out_cols = (ncol + 1) / 2;
    } else {
        out_rows = (nrow + 1) / 2;
        out_cols = ncol;
    }

    // reuse memory
    out_arr.resize(out_rows * out_cols);

    if (is_x_axis) {
        
        out_dx.resize(out_cols);
        out_dy = dy;  // could also swap if desired

        // arithmetic
        if (is_arithmetic){

            for (size_t i = 0; i < nrow; ++i) {

                size_t base_in  = i * ncol;
                size_t base_out = i * out_cols;

                for (size_t j = 0, k = 0; j < ncol; j += 2, ++k) {

                    if (j == ncol - 1) {
                        out_arr[base_out + k] = arr[base_in + j];
                        if (i == 0) out_dx[k] = dx[j];
                        continue;
                    }

                    double w1 = dx[j];
                    double w2 = dx[j+1];
                    double v1 = arr[base_in + j];
                    double v2 = arr[base_in + j + 1];

                    out_arr[base_out + k] = (w1*v1 + w2*v2)/(w1+w2);
                    
                    if (i == 0) out_dx[k] = w1 + w2;
                }
            }
        } else{  // harmonic

            for (size_t i = 0; i < nrow; ++i) {

                size_t base_in  = i * ncol;
                size_t base_out = i * out_cols;

                for (size_t j = 0, k = 0; j < ncol; j += 2, ++k) {

                    if (j == ncol - 1) {
                        out_arr[base_out + k] = arr[base_in + j];
                        if (i == 0) out_dx[k] = dx[j];
                        continue;
                    }

                    double w1 = dx[j];
                    double w2 = dx[j+1];
                    double v1 = arr[base_in + j];
                    double v2 = arr[base_in + j + 1];

                    out_arr[base_out + k] = (w1+w2)/((w1/v1)+(w2/v2));

                    if (i == 0) out_dx[k] = w1 + w2;
                }
            }
        }

    } else {

        out_dy.resize(out_rows);
        out_dx = dx;

        if (is_arithmetic){

            for (size_t i = 0, k = 0; i < nrow; i += 2, ++k) {

            size_t base_out = k * out_cols;
            size_t base_in1 = i * ncol;

            bool last_row = (i == nrow - 1);

            if (!last_row) {  // not last row, do averaging
                size_t base_in2 = (i+1) * ncol;

                for (size_t j = 0; j < ncol; ++j) {

                    double w1 = dy[i];
                    double w2 = dy[i+1];
                    double v1 = arr[base_in1 + j];
                    double v2 = arr[base_in2 + j];

                        out_arr[base_out + j] =
                            (w1*v1 + w2*v2)/(w1+w2);
                }

                out_dy[k] = dy[i] + dy[i+1];

            } else {  // last row (copy)
                
                    for (size_t j = 0; j < ncol; ++j) {
                        out_arr[base_out + j] = arr[base_in1 + j];
                    }
                    out_dy[k] = dy[i];
                }
            }
        } else { // harmonic

            for (size_t i = 0, k = 0; i < nrow; i += 2, ++k) {

                size_t base_out = k * out_cols;
                size_t base_in1 = i * ncol;

                bool last_row = (i == nrow - 1);

                if (!last_row) {  // not last row, do averaging
                    size_t base_in2 = (i+1) * ncol;

                    for (size_t j = 0; j < ncol; ++j) {

                        double w1 = dy[i];
                        double w2 = dy[i+1];
                        double v1 = arr[base_in1 + j];
                        double v2 = arr[base_in2 + j];

                            out_arr[base_out + j] =
                                (w1+w2)/((w1/v1)+(w2/v2));
                    }

                    out_dy[k] = dy[i] + dy[i+1];

                } else {  // last row (copy)
                    
                    for (size_t j = 0; j < ncol; ++j) {
                        out_arr[base_out + j] = arr[base_in1 + j];
                    }
                    out_dy[k] = dy[i];
                }
            }
        }
    }
}

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
    std::vector<double>& out_dy)
{

    /*
    This function computes the next level of the renormalization process by averaging pairs of values along either the x or y axis, depending on the is_x_axis flag. 
    The type of averaging (arithmetic or harmonic) is determined by the is_arithmetic flag.
     The function also computes the new grid spacings (dx and dy) for the next level.
      The output is stored in out_arr, out_dx, and out_dy, and the dimensions of the output array are returned via out_rows and out_cols.
    */

    if (is_x_axis) {
        out_rows = nrow;
        out_cols = (ncol + 1) / 2;
    } else {
        out_rows = (nrow + 1) / 2;
        out_cols = ncol;
    }

    // reuse memory
    out_arr.resize(out_rows * out_cols);

    if (is_x_axis) {
        
        out_dx.resize(out_cols);
        out_dy = dy;  // could also swap if desired

        for (size_t i = 0; i < nrow; ++i) {

            size_t base_in  = i * ncol;
            size_t base_out = i * out_cols;

            for (size_t j = 0, k = 0; j < ncol; j += 2, ++k) {

                if (j == ncol - 1) {
                    out_arr[base_out + k] = arr[base_in + j];
                    if (i == 0) out_dx[k] = dx[j];
                    continue;
                }

                double w1 = dx[j];
                double w2 = dx[j+1];
                double v1 = arr[base_in + j];
                double v2 = arr[base_in + j + 1];

                if (is_arithmetic) {
                    out_arr[base_out + k] =
                        (w1*v1 + w2*v2)/(w1+w2);
                } else {
                    // out_arr[base_out + k] =
                        // (w1+w2)/((w1/v1)+(w2/v2));
                        double inv = w1 / v1 + w2 / v2;
                        out_arr[base_out + k] = (w1 + w2) / inv;
                }

                if (i == 0) out_dx[k] = w1 + w2;
            }
        }

    } else {

        out_dy.resize(out_rows);
        out_dx = dx;

        for (size_t i = 0, k = 0; i < nrow; i += 2, ++k) {

            size_t base_out = k * out_cols;
            size_t base_in1 = i * ncol;

            bool last_row = (i == nrow - 1);

            if (!last_row) {  // not last row, do averaging

                double w1 = dy[i];
                double w2 = dy[i+1];
                
                size_t base_in2 = (i+1) * ncol;

                for (size_t j = 0; j < ncol; ++j) {


                    double v1 = arr[base_in1 + j];
                    double v2 = arr[base_in2 + j];

                    if (is_arithmetic) {
                        out_arr[base_out + j] =
                            (w1*v1 + w2*v2)/(w1+w2);
                    } else {
                        out_arr[base_out + j] =
                            (w1+w2)/((w1/v1)+(w2/v2));
                        // double inv = w1 / v1 + w2 / v2;
                        // out_arr[base_out + k] = (w1 + w2) / inv;
                    }
                }

                out_dy[k] = dy[i] + dy[i+1];

            } else {  // last row (copy)
                
                for (size_t j = 0; j < ncol; ++j) {
                    out_arr[base_out + j] = arr[base_in1 + j];
                }
                out_dy[k] = dy[i];
            }
        }
    }
}


double find_c2D(
    const std::vector<double>& arr,
    size_t nrow,
    size_t ncol,
    const std::vector<double>& dx,
    const std::vector<double>& dy,
    bool is_x_direction,
    bool is_arithmetic
)
{
    /*
    This function performs the iterative renormalization process on a 2D array until it reduces to a single value.
     The process alternates between averaging along the x and y directions, and between arithmetic and harmonic means, as determined by the is_x_direction and is_arithmetic flags.
      The function uses the f_new function to compute the next level of the renormalization, and it continues until the array is reduced to a single value, which is then returned as the result.

      in 2D two possible values can be obtained: cmin and cmax, which are obtained by starting with arithmetic or harmonic averaging along x or y direction. 
      The final value is then computed as a mix of these two values based on the grid spacing (dx and dy) at the center of the field see function get_alpha_2D and keq_2D.
    */

    size_t new_rows, new_cols;

    std::vector<double> current = arr;
    std::vector<double> next;

    std::vector<double> dx_current = dx, dy_current = dy;
    std::vector<double> dx_next, dy_next;

    while (nrow > 1 || ncol > 1) {

        f_new(
            current,
            nrow,
            ncol,
            dx_current,
            dy_current,
            is_x_direction,
            is_arithmetic,
            new_rows,
            new_cols,
            next,
            dx_next,
            dy_next
        );

        // swap instead of copy
        current.swap(next);
        dx_current.swap(dx_next);
        dy_current.swap(dy_next);

        nrow = new_rows;
        ncol = new_cols;

        is_x_direction = !is_x_direction;
        is_arithmetic = !is_arithmetic;
    }

    // std::cout << "Final value: " << current[0] << std::endl;
    return current[0];
    
}


// double find_c2D(
//     const std::vector<double>& arr,
//     size_t nrow,
//     size_t ncol,
//     const std::vector<double>& dx,
//     const std::vector<double>& dy,
//     bool is_x_direction,
//     bool is_arithmetic
// )
// {
//     size_t new_rows, new_cols;

//     std::vector<double> buffer1, buffer2;
//     std::vector<double>* current = &buffer1;
//     std::vector<double>* next = &buffer2;

//     std::vector<double> dx_current = dx, dy_current = dy;
//     std::vector<double> dx_next, dy_next;

//     *current = arr;  // single copy at start

//     while (nrow > 1 || ncol > 1) {

//         f_new(
//             *current,
//             nrow,
//             ncol,
//             dx_current,
//             dy_current,
//             is_x_direction,
//             is_arithmetic,
//             new_rows,
//             new_cols,
//             *next,
//             dx_next,
//             dy_next
//         );

//         // swap pointers instead of vectors
//         std::swap(current, next);
//         dx_current.swap(dx_next);
//         dy_current.swap(dy_next);

//         nrow = new_rows;
//         ncol = new_cols;

//         is_x_direction = !is_x_direction;
//         is_arithmetic = !is_arithmetic;
//     }

//     return (*current)[0];
// }


double f_u(
    double t
){
    return std::atan(std::sqrt(t)) / (3.14159 / 2);
}

double get_alpha_2D(
    double dx,
    double dy,
    bool is_x_direction
){

    // function to obtain alpha parameter to determine how to compute Keq based on cmin and cmax
    double a, u;
    if (is_x_direction){
        a = dx/dy;
    }
    else {
        a = dy/dx;
    }
    u = f_u(a);
    return u;
}

double keq(
    double cmin,
    double cmax,
    double alpha
    ){

    /*
    This function computes the equivalent value (keq) based on the minimum and maximum values obtained from the renormalization process (cmin and cmax) and the alpha parameter that determines how to mix these two values. 
    The keq is computed as a weighted geometric mean of cmin and cmax, where the weights are determined by alpha. 
    If both cmin and cmax are negative, the result is negated to ensure that keq has the same sign as the original values.
    */

    double res;
    
    res = std::pow(std::abs(cmax), alpha)
        * std::pow(std::abs(cmin), (1.0 - alpha));
    if (cmax < 0 && cmin < 0){
        res = -res;
    }
    return res;
}

double simplified_renorm2D(
    const std::vector<double>& arr,
    size_t nrow,
    size_t ncol,
    const std::vector<double>& dx,
    const std::vector<double>& dy,
    bool is_kx){
    
    // variables
    double cmax, cmin, dx_center, dy_center;

    // first compute cmin and cmax
    if (is_kx){
        // cxmax --> direction y arithmetic
        cmax = find_c2D(arr, nrow, ncol, dx, dy, false, true);
        // cxmax --> direction x harmonic
        cmin = find_c2D(arr, nrow, ncol, dx, dy, true, false);
    }
    else{
    // cymax --> direction x arithmetic
    cmax = find_c2D(arr, nrow, ncol, dx, dy, true, true);
    // cymin --> direction y harmonic
    cmin = find_c2D(arr, nrow, ncol, dx, dy, false, false);
    }
    // std::cout << "cmax: " << cmax << std::endl;
    // std::cout << "cmin: " << cmin << std::endl;

    // determine alpha --> first determine dx and dy at the center of the field
    int nrow_half = nrow / 2;
    int ncol_half = ncol / 2;
    dx_center = dx[ncol_half];
    dy_center = dy[nrow_half];

    double alpha = get_alpha_2D(dx_center, dy_center, is_kx);

    // compute keq
    double keq_val = keq(cmin, cmax, alpha);

    return keq_val;
    }


// 3D functions

inline void reduce_x(
    const double* __restrict in,
    size_t nz, size_t nrow, size_t ncol,
    const double* __restrict dx,
    bool is_arithmetic,
    double* __restrict out,
    double* __restrict out_dx,
    size_t out_cols)
{
    for (size_t k = 0; k < nz; ++k) {
        for (size_t i = 0; i < nrow; ++i) {

            const double* row = in + (k * nrow + i) * ncol;
            double* out_row   = out + (k * nrow + i) * out_cols;

            size_t j = 0, jj = 0;

            // main loop (no branch)
            for (; j + 1 < ncol; j += 2, ++jj) {
                double w1 = dx[j];
                double w2 = dx[j+1];
                double v1 = row[j];
                double v2 = row[j+1];

                double wsum = w1 + w2;

                out_row[jj] = is_arithmetic
                    ? (w1*v1 + w2*v2) / wsum
                    : wsum / ((w1/v1) + (w2/v2));

                if (k == 0 && i == 0)
                    out_dx[jj] = wsum;
            }

            // tail (rare, no branch inside loop)
            if (j < ncol) {
                out_row[jj] = row[j];
                if (k == 0 && i == 0)
                    out_dx[jj] = dx[j];
            }
        }
    }
}

inline void transpose_y_to_x(
    const std::vector<double>& in,
    size_t nz, size_t nrow, size_t ncol,
    std::vector<double>& out)
{
    out.resize(nz * ncol * nrow);

    for (size_t k = 0; k < nz; ++k) {
        for (size_t i = 0; i < nrow; ++i) {
            for (size_t j = 0; j < ncol; ++j) {
                out[(k * ncol + j) * nrow + i] =
                    in[(k * nrow + i) * ncol + j];
            }
        }
    }
}

inline void transpose_z_to_x(
    const std::vector<double>& in,
    size_t nz, size_t nrow, size_t ncol,
    std::vector<double>& out)
{
    out.resize(nrow * ncol * nz);

    for (size_t k = 0; k < nz; ++k) {
        for (size_t i = 0; i < nrow; ++i) {
            for (size_t j = 0; j < ncol; ++j) {
                out[(i * ncol + j) * nz + k] =
                    in[(k * nrow + i) * ncol + j];
            }
        }
    }
}

void f_new_3D_test(
    const std::vector<double>& arr,
    size_t nz, size_t nrow, size_t ncol,
    const std::vector<double>& dx,
    const std::vector<double>& dy,
    const std::vector<double>& dz,
    int axis,
    bool is_arithmetic,
    size_t& out_nz,
    size_t& out_rows,
    size_t& out_cols,
    std::vector<double>& out_arr,
    std::vector<double>& out_dx,
    std::vector<double>& out_dy,
    std::vector<double>& out_dz)
{
    out_nz   = (axis == 2) ? (nz + 1)/2   : nz;
    out_rows = (axis == 1) ? (nrow + 1)/2 : nrow;
    out_cols = (axis == 0) ? (ncol + 1)/2 : ncol;

    if (axis == 0) {
        // already optimal layout
        out_arr.resize(out_nz * out_rows * out_cols);
        out_dx.resize(out_cols);

        reduce_x(arr.data(), nz, nrow, ncol,
                 dx.data(), is_arithmetic,
                 out_arr.data(), out_dx.data(), out_cols);

        out_dy = dy;
        out_dz = dz;
    }

    else if (axis == 1) {
        // transpose Y -> X
        std::vector<double> tmp_in, tmp_out;

        transpose_y_to_x(arr, nz, nrow, ncol, tmp_in);

        tmp_out.resize(nz * ncol * out_rows);
        out_dy.resize(out_rows);

        reduce_x(tmp_in.data(),
                 nz, ncol, nrow,   // swapped dims
                 dy.data(),
                 is_arithmetic,
                 tmp_out.data(),
                 out_dy.data(),
                 out_rows);

        // transpose back
        out_arr.resize(out_nz * out_rows * out_cols);

        for (size_t k = 0; k < nz; ++k) {
            for (size_t j = 0; j < ncol; ++j) {
                for (size_t i = 0; i < out_rows; ++i) {
                    out_arr[(k*out_rows + i)*out_cols + j] =
                        tmp_out[(k*ncol + j)*out_rows + i];
                }
            }
        }

        out_dx = dx;
        out_dz = dz;
    }

    else { // axis == 2
        std::vector<double> tmp_in, tmp_out;

        transpose_z_to_x(arr, nz, nrow, ncol, tmp_in);

        tmp_out.resize(nrow * ncol * out_nz);
        out_dz.resize(out_nz);

        reduce_x(tmp_in.data(),
                 nrow, ncol, nz,
                 dz.data(),
                 is_arithmetic,
                 tmp_out.data(),
                 out_dz.data(),
                 out_nz);

        // transpose back
        out_arr.resize(out_nz * out_rows * out_cols);

        for (size_t i = 0; i < nrow; ++i) {
            for (size_t j = 0; j < ncol; ++j) {
                for (size_t k = 0; k < out_nz; ++k) {
                    out_arr[(k*out_rows + i)*out_cols + j] =
                        tmp_out[(i*ncol + j)*out_nz + k];
                }
            }
        }

        out_dx = dx;
        out_dy = dy;
    }
}

// old functions
inline size_t idx3(size_t k, size_t i, size_t j,
                   size_t nrow, size_t ncol) {
    return (k * nrow + i) * ncol + j;
}


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
    std::vector<double>& out_dz)
{
    // determine output sizes
    out_nz   = (axis == 2) ? (nz + 1)/2   : nz;
    out_rows = (axis == 1) ? (nrow + 1)/2 : nrow;
    out_cols = (axis == 0) ? (ncol + 1)/2 : ncol;

    out_arr.resize(out_nz * out_rows * out_cols);

    // copy unchanged directions
    if (axis != 0) out_dx = dx;
    if (axis != 1) out_dy = dy;
    if (axis != 2) out_dz = dz;

    if (axis == 0) { // X direction

        out_dx.resize(out_cols);

        for (size_t k = 0; k < nz; ++k) { // layers
            for (size_t i = 0; i < nrow; ++i) {  // rows

                size_t base_in  = (k * nrow + i) * ncol;
                size_t base_out = (k * out_rows + i) * out_cols;

                for (size_t j = 0, jj = 0; j < ncol; j += 2, ++jj) {

                    if (j == ncol - 1) {  // last column --> just copy values, only happens when number of col is odd
                        out_arr[base_out + jj] = arr[base_in + j];
                        if (k == 0 && i == 0) out_dx[jj] = dx[j];
                        continue;
                    }
                    
                    // do the average
                    double w1 = dx[j], w2 = dx[j + 1];
                    double v1 = arr[base_in + j];
                    double v2 = arr[base_in + j + 1];

                    out_arr[base_out + jj] =
                        is_arithmetic ?
                        (w1 * v1 + w2 * v2) / (w1 + w2) :
                        (w1 + w2) / ((w1 / v1) + (w2 / v2));

                    if (k == 0 && i == 0)
                        out_dx[jj] = w1 + w2;
                }
            }
        }

    } else if (axis == 1) { // Y direction

        out_dy.resize(out_rows);

        for (size_t k = 0; k < nz; ++k) {
            for (size_t i = 0, ii = 0; i < nrow; i += 2, ++ii) {

                bool last = (i == nrow - 1);

                for (size_t j = 0; j < ncol; ++j) {

                    size_t out_id = idx3(k, ii, j, out_rows, out_cols);

                    if (last) {
                        out_arr[out_id] = arr[idx3(k,i,j,nrow,ncol)];
                        if (k == 0 && j == 0) out_dy[ii] = dy[i];
                        continue;
                    }

                    double w1 = dy[i], w2 = dy[i+1];
                    double v1 = arr[idx3(k,i,j,nrow,ncol)];
                    double v2 = arr[idx3(k,i+1,j,nrow,ncol)];

                    out_arr[out_id] =
                        is_arithmetic ?
                        (w1*v1 + w2*v2)/(w1+w2) :
                        (w1+w2)/((w1/v1)+(w2/v2));

                    if (k == 0 && j == 0)
                        out_dy[ii] = w1 + w2;
                }
            }
        }

    } else { // Z direction

        out_dz.resize(out_nz);

        for (size_t k = 0, kk = 0; k < nz; k += 2, ++kk) {

            bool last = (k == nz - 1);

            for (size_t i = 0; i < nrow; ++i) {
                for (size_t j = 0; j < ncol; ++j) {

                    size_t out_id = idx3(kk, i, j, out_rows, out_cols);

                    if (last) {
                        out_arr[out_id] = arr[idx3(k,i,j,nrow,ncol)];
                        if (i == 0 && j == 0) out_dz[kk] = dz[k];
                        continue;
                    }

                    double w1 = dz[k], w2 = dz[k+1];
                    double v1 = arr[idx3(k,i,j,nrow,ncol)];
                    double v2 = arr[idx3(k+1,i,j,nrow,ncol)];

                    out_arr[out_id] =
                        is_arithmetic ?
                        (w1*v1 + w2*v2)/(w1+w2) :
                        (w1+w2)/((w1/v1)+(w2/v2));

                    if (i == 0 && j == 0)
                        out_dz[kk] = w1 + w2;
                }
            }
        }
    }
}

double find_c3D(
    const std::vector<double>& arr,
    size_t nz,
    size_t nrow, 
    size_t ncol,
    const std::vector<double>& dx,
    const std::vector<double>& dy,
    const std::vector<double>& dz,
    int direction, // direction to obtain c (0, 1, 2) ie (x, y, z)
    bool is_max)
{
    size_t new_nz, new_rows, new_cols;

    bool is_arithmetic;
    int axis, first_axis;
    std::vector<double> current = arr, next;
    std::vector<double> dx_c = dx, dy_c = dy, dz_c = dz;
    std::vector<double> dx_n, dy_n, dz_n;

    // determine first axis
    if (direction == 0 && is_max) {first_axis = 1;}  //cx max
    else if (direction == 0 && !is_max) {first_axis = 0;}  //cx min
    else if (direction == 1 && is_max) {first_axis = 2;}  //cy max
    else if (direction == 1 && !is_max) {first_axis = 1;}  //cy min
    else if (direction == 2 && is_max) {first_axis = 0;}  //cz max
    else if (direction == 2 && !is_max) {first_axis = 2;}  // cz min

    axis = first_axis;

    while (nz > 1 || nrow > 1 || ncol > 1) {

        is_arithmetic = (axis == direction) ? false : true;  // harmonic only when axis is in the direction

        f_new_3D(
            current,
            nz, nrow, ncol,
            dx_c, dy_c, dz_c,
            axis,
            is_arithmetic,
            new_nz, new_rows, new_cols,
            next,
            dx_n, dy_n, dz_n
        );

        current.swap(next);
        dx_c.swap(dx_n);
        dy_c.swap(dy_n);
        dz_c.swap(dz_n);

        nz = new_nz;
        nrow = new_rows;
        ncol = new_cols;

        axis = (axis + 1) % 3;        // rotate x→y→z
        // is_arithmetic = !is_arithmetic;
    }

    return current[0];
}

double get_alpha_3D(double dx, double dy, double dz, int direction){

    // function to obtain alpha parameter to determine how to compute Keq based on cmin and cmax
    double a1, a2, u1, u2;

    if (direction == 0){a1 = dx/dy; a2 = dx/dz;}
    else if (direction == 1){a1 = dy/dx; a2 = dy/dz;}
    else if (direction == 2){a1 = dz/dx; a2 = dz/dy;}

    u1 = f_u(a1);
    u2 = f_u(a2);

    return ((u1 + u2) - 2*u1*u2) / (1 - u1*u2);
}

double simplified_renorm3D(const std::vector<double>& arr,
    size_t nrow,
    size_t ncol,
    size_t nlay,
    const std::vector<double>& dx,
    const std::vector<double>& dy,
    const std::vector<double>& dz,
    int direction){
    
    // variables
    double cmax, cmin, dx_center, dy_center, dz_center;

    // first compute cmin and cmax
    cmax = find_c3D(arr, nlay, nrow, ncol, dx, dy, dz, direction, true);
    cmin = find_c3D(arr, nlay, nrow, ncol, dx, dy, dz, direction, false);

    std::cout << "cmax: " << cmax << std::endl;
    std::cout << "cmin: " << cmin << std::endl;

    // determine alpha --> first determine dx and dy at the center of the field
    dx_center = dx[ncol / 2];
    dy_center = dy[nrow / 2];
    dz_center = dz[nlay / 2];

    double alpha = get_alpha_3D(dx_center, dy_center, dz_center, direction);

    // compute keq
    double keq_val = keq(cmin, cmax, alpha);

    return keq_val;
    }




