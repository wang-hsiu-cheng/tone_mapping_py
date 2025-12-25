#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <Eigen/Dense>
#include <iomanip>

using namespace std;

const int EXP_LUT_SIZE = 16384;
const int DIVIDE_LUT_SIZE = 8192; 
vector<int> exp_lut, divide_lut;

// --- 硬體模擬：整數飽和函數 ---
long long clamp_int(long long val, long long min_val, long long max_val) {
    if (val > max_val) return max_val;
    if (val < min_val) return min_val;
    return val;
}

std::vector<int> load_lut_from_txt(const std::string& filepath, int SIZE) {
    std::vector<int> lut_array(SIZE);
    std::fill(lut_array.begin(), lut_array.end(), 0);

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "ERROR: Failed to open file: " << filepath << std::endl;
        return {};
    }
    std::string line;
    int line_count = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        int index = 0, value = 0;
        if (ss >> index >> value) {
            if (index >= 0 && index < SIZE) {
                lut_array[index] = value;
                line_count++;
            }
        }
    }
    std::cout << "Successfully loaded LUT: " << filepath << std::endl;
    return lut_array;
}

// --- 修改點 1：回傳類型改為 long long 矩陣 ---
Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic> custom_bilateral_filter_integer(const Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic>& I_int) {
    int h = I_int.rows();
    int w = I_int.cols();
    int r = 2; 
    
    // 初始化為 long long 矩陣
    Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic> B_out(h, w);

    int spatial_kernel[5][5] = {0};
    int dist_weights[9] = {1024, 820, 675, 0, 421, 337, 0, 0, 173}; 

    for (int i = -r; i <= r; ++i) {
        for (int j = -r; j <= r; ++j) {
            int dist_sq = i * i + j * j;
            if (dist_sq == 0) spatial_kernel[i + r][j + r] = 1024;
            else if (dist_sq == 1) spatial_kernel[i + r][j + r] = 820;
            else if (dist_sq == 2) spatial_kernel[i + r][j + r] = 421;
            else if (dist_sq == 4) spatial_kernel[i + r][j + r] = 29;
            else if (dist_sq == 5) spatial_kernel[i + r][j + r] = 4;
            else spatial_kernel[i + r][j + r] = 0;
        }
    }

    std::cout << "Starting Bit-Exact Integer Scan..." << std::endl;

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            long long I_p = I_int(i, j);
            long long numerator_acc = 0;   
            long long denominator_acc = 0; 

            for (int m = -r; m <= r; ++m) {
                for (int n = -r; n <= r; ++n) {
                    int qi = i + m; int qj = j + n;
                    if (qi < 0) qi = 0; if (qi >= h) qi = h - 1;
                    if (qj < 0) qj = 0; if (qj >= w) qj = w - 1;
                    long long I_q = I_int(qi, qj);

                    // Stage 1: Diff
                    long long diff = I_p - I_q; 
                    long long diff_21b = diff & 0x1FFFFF; 
                    long long diff_abs = ((diff >> 21) & 1) ? (-(diff_21b) & 0x1FFFFF) : diff_21b;

                    // Stage 2: Diff Square
                    long long diff_sq = diff_abs * diff_abs; 
                    long long range_idx = (diff_sq >> 17);
                    if (range_idx > 16383) range_idx =  16383;
                    else range_idx =  range_idx & 0x3FFF;
                    // if (i==355 && j==765) {
                    //     std::cout << I_p << " " << I_q << "\n";
                    //     std::cout << m << " " << n << ": " << range_idx << "\n";
                    // }

                    // Stage 3: Weights
                    long long range_w = (long long)exp_lut[range_idx]; 
                    long long spatial_w = (long long)spatial_kernel[m + r][n + r]; 
                    long long total_w = (range_w * spatial_w) >> 10; 
                    // if (i==355 && j==765) {
                    //     std::cout << I_p << " " << I_q << "\n";
                    //     std::cout << m << " " << n << ": " << diff_sq << " " << range_idx << " " << range_w << " " << spatial_w << " " << total_w << "\n";
                    // }

                    // Stage 4: Weighted Q
                    long long weighted_q = I_q * total_w; 

                    numerator_acc += weighted_q;
                    denominator_acc += total_w;
                }
            }

            // Stage 5: Divide LUT
            int den_idx = (int)(denominator_acc & 0x1FFF); 
            long long wp_val = (long long)divide_lut[den_idx]; 
            long long b_val = wp_val * numerator_acc; 
            long long b_quant = b_val >> 20; 

            // Stage 6: Clamp
            // --- 修改點 2：直接儲存整數，不除以 16384.0 ---
            B_out(i, j) = clamp_int(b_quant, -1048576, 1048575);
        }
    }
    return B_out;
}

Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic> read_matrix_from_text(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Error opening file: " + filename);

    int W = 0, H = 0;
    std::string line;
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        ss >> W >> H;
    }

    Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic> matrix(H, W);
    int current_row = 0;
    long long value;

    while (std::getline(file, line) && current_row < H) {
        std::stringstream ss(line);
        int current_col = 0;
        while (ss >> value && current_col < W) {
            matrix(current_row, current_col) = value;
            current_col++;
        }
        current_row++;
    }
    return matrix;
}

// --- 修改點 3：參數改為 long long 矩陣，輸出改為整數格式 ---
void write_matrix_to_text(const Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic>& matrix, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Failed to open file for writing: " << filename << std::endl;
        return;
    }

    int H = (int)matrix.rows();
    int W = (int)matrix.cols();

    // 1. 寫入第一行：W H
    file << W << " " << H << "\n"; 
    
    // 2. 寫入整數矩陣內容
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            file << matrix(i, j); // 直接輸出 long long
            if (j < W - 1) file << " ";
        }
        file << "\n";
    }

    file.close();
    std::cout << "  Successfully saved Integer Output Matrix: " << filename << std::endl;
}

int main() {
    try {
        // 加載輸入 (整數矩陣)
        Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic> I_int_matrix;
        I_int_matrix = read_matrix_from_text("act_data/luminance.txt");
        
        // 加載 LUT
        exp_lut = load_lut_from_txt("lut/My_range_exp_LUT.txt", EXP_LUT_SIZE);
        divide_lut = load_lut_from_txt("lut/total_weight_div.txt", DIVIDE_LUT_SIZE);
        
        // --- 修改點 4：B_matrix 現在是整數矩陣 ---
        Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic> B_matrix;
        B_matrix = custom_bilateral_filter_integer(I_int_matrix);

        // 輸出整數結果
        write_matrix_to_text(B_matrix, "act_data/B_matrix.txt");

        std::cout << "--- C++ Bit-Exact Processing Complete ---" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}