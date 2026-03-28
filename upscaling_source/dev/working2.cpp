// collection of function to do upscaling of 2D and 3D arrays in C++. For this we rely on renormalization using mix of arithmetic and harmonic averages
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include "./../simplified_renorm_C/simplified_renorm.cpp"
// #include "./simplified_renorm.hpp"

int main(){

    size_t nrow = 2;
    size_t ncol = 3;
    size_t nlay = 3;
    std::vector<double> arr = {
        1e-2, 1e-3, 3e-2,
        4e-2, 5e-5, 6e-4,
        // 7e-5, 8e-1, 9e-4,

        1e-2, 1e-3, 3e-2,
        4e-2, 5e-5, 6e-4,
        // 7e-5, 8e-1, 9e-4,

        1e-2, 1e-3, 3e-2,
        4e-2, 5e-5, 6e-4,
        // 7e-5, 8e-1, 9e-4,
    };

    // print_arr(arr, nrow, ncol);
    std::vector<double> dx = {1, 1, 1};
    std::vector<double> dy = {1, 1};
    std::vector<double> dz = {1, 1, 1};
    double czmax, czmin, cymax, cymin, cxmax, cxmin, Kx, Ky, Kz;

    // cxmin = find_c3D(arr, nlay, nrow, ncol, dx, dy, dz, 0, false);
    // std::cout << "cx min: " << cxmin << std::endl;
    // cxmax = find_c3D(arr, nlay, nrow, ncol, dx, dy, dz, 0, true);
    // std::cout << "cx max: " << cxmax << std::endl;
    // if we start

    // cymax --> direction x arithmetic
    // cymax = find_c2D(arr, nrow, ncol, dx, dy, true, true);
    // // cymin --> direction y harmonic
    // cymin = find_c2D(arr, nrow, ncol, dx, dy, false, false);

    // // cxmax --> direction y arithmetic
    // cxmax = find_c2D(arr, nrow, ncol, dx, dy, false, true);
    // // cxmin --> direction x harmonic
    // cxmin = find_c2D(arr, nrow, ncol, dx, dy, true, false);

    double dx_center, dy_center, dz_center;

    cxmax = find_c3D(arr, nlay, nrow, ncol, dx, dy, dz, 0, true);
    cxmin = find_c3D(arr, nlay, nrow, ncol, dx, dy, dz, 0, false);
    std::cout << "cmax: " << cxmax << std::endl;
    std::cout << "cmin: " << cxmin << std::endl;

    dx_center = dx[ncol / 2];
    dy_center = dy[nrow / 2];
    dz_center = dz[nlay / 2];

    double alpha = get_alpha_3D(dx_center, dy_center, dz_center, 0);
    double keq_val = keq(cxmin, cxmax, alpha);
    std::cout << keq_val << std::endl;
    
    // Kx = simplified_renorm3D(arr, nlay, nrow, ncol, dx, dy, dz, 0);
    // Ky = simplified_renorm3D(arr, nlay, nrow, ncol, dx, dy, dz, 1);
    // Kz = simplified_renorm3D(arr, nlay, nrow, ncol, dx, dy, dz, 2);

    // std::cout << "Kx: " << Kx << std::endl;
    // std::cout << "Ky: " << Ky << std::endl;
    // std::cout << "Kz: " << Kz << std::endl;

    // auto start2 = std::chrono::high_resolution_clock::now(); // Step 1: Record start time
    // std::cout << "Starting computational task..." << std::endl;
    // for (int b=0; b < 1000; b++){
    //     Kx = simplified_renorm3D(arr, nlay, nrow, ncol, dx, dy, dz, 0);
    //     Ky = simplified_renorm3D(arr, nlay, nrow, ncol, dx, dy, dz, 1);
    //     Kz = simplified_renorm3D(arr, nlay, nrow, ncol, dx, dy, dz, 2);
    // }
    // auto end2 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration2 = end2 - start2;

    // std::cout << "Time taken by task: " << duration2.count() * 1000 / 1000 << " microseconds" << std::endl;

    return 0;
}



// int main(){

//     size_t nrow = 3;
//     size_t ncol = 2;

//     std::vector<double> arr = {
//         4.40648563e-04, 1.67358520e-03,
//        2.41772278e-05, 4.92768708e-03,
//        3.01621040e-03, 1.65061985e-05
//     };

//     // print_arr(arr, nrow, ncol);
//     std::vector<double> dx = {1, 1};
//     std::vector<double> dy = {1, 1, 1};
//     double cymax, cymin, cxmax, cxmin, Kx, Ky;
//     // if we start

//     // cymax --> direction x arithmetic
//     // cymax = find_c2D(arr, nrow, ncol, dx, dy, true, true);
//     // // cymin --> direction y harmonic
//     // cymin = find_c2D(arr, nrow, ncol, dx, dy, false, false);

//     // // cxmax --> direction y arithmetic
//     // cxmax = find_c2D(arr, nrow, ncol, dx, dy, false, true);
//     // // cxmin --> direction x harmonic
//     // cxmin = find_c2D(arr, nrow, ncol, dx, dy, true, false);

//     Kx = simplified_renorm2D(arr, nrow, ncol, dx, dy, true);
//     Ky = simplified_renorm2D(arr, nrow, ncol, dx, dy, false);

//     std::cout << "Kx: " << Kx << std::endl;
//     std::cout << "Ky: " << Ky << std::endl;

//     auto start2 = std::chrono::high_resolution_clock::now(); // Step 1: Record start time
//     std::cout << "Starting computational task..." << std::endl;
//     for (int b=0; b < 1000; b++){
//         Kx = simplified_renorm2D(arr, nrow, ncol, dx, dy, true);
//         Ky = simplified_renorm2D(arr, nrow, ncol, dx, dy, false);
//     }
//     auto end2 = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> duration2 = end2 - start2;

//     std::cout << "Time taken by task: " << duration2.count() * 1000 / 1000 << " microseconds" << std::endl;

//     return 0;
// }