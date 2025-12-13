#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdint> // 用於 int32_t, int64_t
#include <algorithm>
#include <Eigen/Dense> // 引入 Eigen 函式庫
#include <math.h>     // 為了使用 std::exp
#include <iomanip> // 用於控制浮點數精度

// 範例：根據 Python 代碼中隱含的邏輯定義的常量 (N_BITS=16)
const int Q_FRACT = 8;
const int EXP_LUT_SIZE = 16384;
const int DIVIDE_LUT_SIZE = 4096;
std::vector<int> exp_lut, divide_lut;

std::vector<int> load_lut_from_txt(const std::string& filepath, int SIZE) {
    // 1. 初始化 vector，確保它有足夠的容量
    std::vector<int> lut_array(SIZE);
    
    // 為了安全起見，可以先用一個標記值 (例如 -1) 填充，以便追蹤未被設置的值
    std::fill(lut_array.begin(), lut_array.end(), 0);

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "ERROR: Failed to open file: " << filepath << std::endl;
        return {}; // 返回空 vector
    }
    std::string line;
    int line_count = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue; // 跳過空行

        std::stringstream ss(line);
        int index = 0;
        int value = 0;

        // 嘗試從一行中讀取兩個整數
        if (ss >> index >> value) {
            line_count++;
            
            // 2. 邊界檢查
            if (index < 0 || index >= SIZE) {
                std::cerr << "WARNING: Index " << index << " out of bounds [0, " << SIZE - 1 << "] at line " << line_count << ". Skipping." << std::endl;
                continue;
            }

            // 3. 填充矩陣 (LUT)
            lut_array[index] = value;
            
        } else {
            std::cerr << "WARNING: Failed to parse two integers at line " << line_count + 1 << ". Skipping." << std::endl;
        }
    }
    std::cout << "Successfully loaded " << line_count << " entries." << std::endl;
    if (line_count != SIZE) {
        std::cout << "WARNING: Expected " << SIZE << " entries, found " << line_count << "." << std::endl;
    }
    return lut_array;
}

// =========================================================
// 2. enforce_q_precision 函式 (朝向零的無條件捨去與飽和)
// =========================================================
float enforce_q_precision(float f_value, int fract_bits, int total_bits) {
    const long long MAX_FIXED_VALUE = (1LL << (total_bits - 1)) - 1; // 2^15 - 1
    const long long MIN_FIXED_VALUE = -(1LL << (total_bits - 1));    // -2^15
    long long scale = 1LL << fract_bits;
    float scaled_value = f_value * (float)scale;
    int32_t fixed_value_unclipped = static_cast<int32_t>(std::trunc(scaled_value)); 
    long long fixed_value_clipped = std::clamp(
        (long long)fixed_value_unclipped, 
        MIN_FIXED_VALUE, 
        MAX_FIXED_VALUE
    );
    return (float)fixed_value_clipped / (float)scale;
}

const int FILTER_D = 5;        
const float SIGMA_R = 1.0;
const float SIGMA_S = 1.5;    

Eigen::MatrixXf custom_bilateral_filter_with_lut(const Eigen::MatrixXf& I) {
    const double SIGMA_R_2 = enforce_q_precision(1 / (2.0 * std::pow(SIGMA_R, 2)), 6, 16);
    const double SIGMA_S_2 = enforce_q_precision(1 / (2.0 * std::pow(SIGMA_S, 2)), 6, 16);
    std::cout << "start custom bf" << std::endl;

    // 1. 初始化
    int h = I.rows(); // Eigen 獲取行數
    int w = I.cols(); // Eigen 獲取列數
    int r = FILTER_D / 2; // 半徑
    // 初始化輸出矩陣 B (與 I 相同大小和類型)
    Eigen::MatrixXf B = Eigen::MatrixXf::Zero(h, w); 
    // 預先計算空間核 (使用 float 來模擬 Python 中的 spatial_kernel_float)
    // 這裡我們仍然使用 Eigen::MatrixXf 來儲存這個 dxd 矩陣
    Eigen::MatrixXf spatial_kernel_float = Eigen::MatrixXf::Zero(FILTER_D, FILTER_D);

    // 2. 預計算空間核 (Spatial Kernel)
    for (int i = -r; i <= r; ++i) {
        for (int j = -r; j <= r; ++j) {
            float dist_sq = (float)(i * i + j * j); 
            
            // 空間核輸入 (除法結果需要鉗位)
            float exp_input = enforce_q_precision(dist_sq * SIGMA_S_2, 10, 4);
            
            // 計算 exp(-x) 並鉗位
            // float weight = std::exp(-exp_input); 
            float weight = exp_lut[std::trunc(exp_input * 1024)] / 1024.0;
            
            // Eigen 矩陣的元素存取: (行, 列)
            spatial_kernel_float(i + r, j + r) = enforce_q_precision(weight, 10, 11);
        }
    }

    // 3. 滑動窗口掃描
    std::cout << "start scan" << std::endl;
    double max = 0, min = 0;

    for (int i = 0; i < h; ++i) {
        // 顯示進度
        if (i % 100 == 0) {
            std::cout << "  Processing row " << i << "/" << h << std::endl;
        }

        for (int j = 0; j < w; ++j) {
            
            // 1. 初始化
            // 抓 sliding window 中心點
            float I_p = I(i, j);
            float numerator_float = 0.0f; // 分子 (加權和)
            float denominator_float = 0.0f; // 分母 (歸一化因子)
            
            // 2. 周圍的點 計算高斯
            for (int m = -r; m <= r; ++m) {
                for (int n = -r; n <= r; ++n) {
                    int q_i = i + m;
                    int q_j = j + n;
                    // 邊界檢查
                    float I_q;
                    if (q_i >= 0 && q_i < h && q_j >= 0 && q_j < w) {
                        I_q = I(q_i, q_j);
                    } else if (q_i < 0 && q_j < 0) {
                        I_q = I(0, 0);
                    } else if (q_i < 0 && q_j >= w) {
                        I_q = I(0, w-1);
                    } else if (q_i >= h && q_j < 0) {
                        I_q = I(h-1, 0);
                    } else if (q_i >= h && q_j >= w) {
                        I_q = I(h-1, w-1);
                    }
                    // --- 範圍核計算 (Range Kernel) ---
                    // 減法結果和平方結果都需要鉗位
                    float diff = enforce_q_precision(I_p - I_q, 8, 16);
                    float diff_sq = enforce_q_precision(diff * diff, 8, 16);
                    // 範圍核輸入 (除法結果需要鉗位)
                    float range_exp_input = enforce_q_precision(diff_sq * SIGMA_R_2, 6, 12);
                    // float range_weight_float = enforce_q_precision(std::exp(-range_exp_input), 6, 7);
                    float range_weight_float = exp_lut[std::trunc(range_exp_input * 1024)] / 1024.0;
                    // --- 總權重計算 ---
                    float spatial_weight_float = spatial_kernel_float(m + r, n + r); // Eigen 存取
                    // 總權重 (乘法結果需要鉗位)
                    float total_weight = enforce_q_precision(spatial_weight_float * range_weight_float, 10, 16);
                    // 累積
                    // I_q * total_weight (乘法結果需要鉗位)
                    float weighted_I_q = enforce_q_precision(total_weight * I_q, 8, 16);
                    
                    denominator_float = enforce_q_precision(denominator_float+total_weight, 8, 16);
                    numerator_float = enforce_q_precision(numerator_float+weighted_I_q, 8, 16);
                }
            }
            
            // 3. 歸一化 (除法)
            float B_val;
            if (denominator_float > 0.0f) {
                // 紀錄最大值&最小值
                if (denominator_float > max || max == 0)
                    max = denominator_float;
                if (denominator_float < min || min == 0)
                    min = denominator_float;
                // 最終結果的除法需要鉗位
                denominator_float = divide_lut[std::trunc(denominator_float * 64)] / 4096.0;
                B_val = enforce_q_precision(numerator_float * denominator_float, 8, 16);
            } else {
                B_val = I_p; // 避免除以零
            }
            
            B(i, j) = B_val; // Eigen 存取
        }
    }
    std::cout << "denominator range from" << max << " to " << min << std::endl;
    
    return B;
}

Eigen::MatrixXf read_matrix_from_text(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    int W = 0, H = 0;
    std::string line;

    // 1. 讀取第一行：長寬 (W H)
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        if (!(ss >> W >> H)) {
            throw std::runtime_error("Error reading dimensions (W H) from the first line in: " + filename);
        }
    } else {
        throw std::runtime_error("File is empty: " + filename);
    }

    if (W <= 0 || H <= 0) {
        throw std::runtime_error("Invalid dimensions (W or H <= 0) in: " + filename);
    }

    // 初始化 Eigen 矩陣 (注意：Eigen 使用 (row, col) 即 (H, W))
    Eigen::MatrixXf matrix(H, W);
    int current_row = 0;
    int current_col = 0;
    float value;

    // 2. 讀取矩陣內容
    while (std::getline(file, line) && current_row < H) {
        std::stringstream ss(line);
        current_col = 0;
        
        while (ss >> value && current_col < W) {
            matrix(current_row, current_col) = value;
            current_col++;
        }
        
        // 檢查該行是否讀取了正確數量的列 (可選的嚴格檢查)
        if (current_col != W) {
            // 如果該行數字不足 W 個，可能需要報錯或填充
            // 這裡假設每行都應包含 W 個數字
            // throw std::runtime_error("Row " + std::to_string(current_row) + " has incorrect number of columns in: " + filename);
        }

        current_row++;
    }

    // 3. 檢查是否讀取了足夠的行
    if (current_row != H) {
        throw std::runtime_error("The matrix content has fewer rows (" + std::to_string(current_row) + ") than expected (" + std::to_string(H) + ") in: " + filename);
    }

    std::cout << "  Successfully loaded: " << filename << " (" << H << "x" << W << ")" << std::endl;
    return matrix;
}
void write_matrix_to_text(const Eigen::MatrixXf& matrix, const std::string& filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "ERROR: Failed to open file for writing: " << filename << std::endl;
        return;
    }

    int H = matrix.rows(); // 高度 (行數)
    int W = matrix.cols(); // 寬度 (列數)

    // 1. 寫入第一行：寬度 空間 高度 (W H)
    // 雖然 Eigen 內部是 (行, 列)，但傳統影像格式通常以 (寬度, 高度) 開始
    file << W << " " << H << "\n"; 
    
    // 2. 寫入矩陣內容
    // 設定輸出精度，例如小數點後 6 位
    file << std::fixed << std::setprecision(6);

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            // 寫入當前元素
            file << matrix(i, j);
            
            // 如果不是該行的最後一個元素，則添加空格分隔
            if (j < W - 1) {
                file << " ";
            }
        }
        // 每行結束後換行
        file << "\n";
    }

    file.close();
    std::cout << "\n Successfully wrote matrix output to: " << filename << std::endl;
}

int main() {
    const std::string EXP_LUT_FILE = "LUT/exp.txt";
    const std::string DIVIDE_LUT_FILE = "LUT/divide.txt";
    const std::string I_FILE = "data/luminance.txt";
    const std::string OUTPUT_FILE_NAME = "data/B_matrix.txt";
    try {
        // 2. 讀取三個通道的數據
        Eigen::MatrixXf I_matrix = read_matrix_from_text(I_FILE);
        exp_lut = load_lut_from_txt(EXP_LUT_FILE, EXP_LUT_SIZE);
        divide_lut = load_lut_from_txt(DIVIDE_LUT_FILE, DIVIDE_LUT_SIZE);
        
        std::cout << "\n Data Loading Successful. Starting Processing." << std::endl;

        // 4. 輸入矩陣到核心 LTM 函式
        Eigen::MatrixXf B_matrix = custom_bilateral_filter_with_lut(I_matrix);

        write_matrix_to_text(B_matrix, OUTPUT_FILE_NAME);

        std::cout << "\n--- Processing Complete ---" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\nFATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}