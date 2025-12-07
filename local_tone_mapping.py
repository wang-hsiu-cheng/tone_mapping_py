import cv2
import numpy as np
import pandas as pd # 引入 pandas 讀取 Excel
# from matplotlib import pyplot as plt # 移除顯示，只進行儲存

# --- 定點數和 LUT 相關的輔助函數 ---

# 定點數格式設定
Q_IN_FRACT_BITS = 10  # 輸入 exp 的小數部分位數 (Q2.10)
Q_OUT_FRACT_BITS = 14 # 輸出 exp 的小數部分位數 (Q4.14)

def float_to_fixed(f_value, fract_bits):
    """將浮點數轉換為定點數（以整數表示）。"""
    return np.round(f_value * (1 << fract_bits)).astype(np.int64)

def fixed_to_float(fixed_value, fract_bits):
    """將定點數轉換回浮點數。"""
    return fixed_value / (1 << fract_bits)

def load_and_prepare_lut(excel_path, sheet_name='exp'):
    """
    載入 Excel 檔案，構建 LUT 查找表。
    
    返回：一個從 Q2.10 索引（整數）映射到 Q4.14 輸出（整數）的查找數組。
    """
    try:
        # 讀取 Excel 文件
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
        
        # 提取輸入 (Column 0) 和輸出 (Column 3)
        input_float = df.iloc[:, 0].values
        output_float = df.iloc[:, 3].values
        
        # 將輸入的浮點數轉換為 Q2.10 定點數索引
        input_fixed = float_to_fixed(input_float, Q_IN_FRACT_BITS)
        
        # 將輸出的浮點數轉換為 Q4.14 定點數
        output_fixed = float_to_fixed(output_float, Q_OUT_FRACT_BITS)
        
        # 構建查找數組 (假設輸入索引是連續且完整的)
        # 查找數組大小為 2^(2+10) = 4096
        lut_size = 1 << (2 + Q_IN_FRACT_BITS)
        lut_array = np.zeros(lut_size, dtype=np.int64)
        
        # 填充查找表 (假設輸入索引與數據的順序一致)
        # 這裡假設 input_fixed 的值可以直接作為索引
        for idx, val in zip(input_fixed, output_fixed):
            if 0 <= idx < lut_size:
                lut_array[idx] = val
            
        print(f"LUT 載入成功，大小: {lut_size} 點。")
        return lut_array
        
    except Exception as e:
        print(f"載入或處理 LUT 檔案時發生錯誤: {e}")
        # 如果失敗，返回 None 或使用一個簡單的 Numpy exp() 函數作為 fallback
        return None

def fixed_point_exp_lookup(value_float, lut_array, max_input_fixed):
    """
    使用 LUT 執行指數運算，處理定點數轉換和查找。
    
    Args:
        value_float (float): 要計算 exp(-x) 的輸入浮點數 x (必須是非負)。
        lut_array (np.ndarray): 預載入的 LUT 數組 (Q2.10 -> Q4.14)。
        max_input_fixed (int): LUT 數組的最大索引值。
        
    Returns:
        int: exp(-x) 的 Q4.14 定點數結果。
    """
    # 確保輸入非負數 (e.g., d^2/2sigma^2)
    if value_float < 0:
        value_float = 0 

    # 1. 轉換為 Q2.10 定點數
    fixed_index = float_to_fixed(value_float, Q_IN_FRACT_BITS)
    
    # 2. 邊界檢查和鉗位 (如果輸入超出 LUT 定義的範圍，則鉗位到最大或最小索引)
    fixed_index = np.clip(fixed_index, 0, max_input_fixed)
    
    # 3. 查找 (輸出是 Q4.14 定點數)
    return lut_array[fixed_index]

def custom_bilateral_filter_with_lut(I, d, sigma_s, sigma_r, lut_array):
    """
    客製化雙邊濾波器，使用滑動窗口和 LUT 進行指數運算。
    
    Args:
        I (np.ndarray): 對數亮度影像 I(p) (浮點數)。
        d (int): 濾波器窗口直徑 (必須是奇數)。
        sigma_s (float): 空間標準差 σ_s。
        sigma_r (float): 範圍標準差 σ_r。
        lut_array (np.ndarray): 預載入的 LUT 數組。
        
    Returns:
        np.ndarray: 基礎層 B(p) (浮點數)。
    """
    h, w = I.shape
    r = d // 2 # 半徑
    B = np.zeros_like(I, dtype=np.float32)
    
    # LUT 相關參數
    max_lut_index = lut_array.shape[0] - 1
    
    # 預先計算空間核 (由於是滑動窗口，每個像素的空間核都是一樣的)
    spatial_kernel_fixed = np.zeros((d, d), dtype=np.int64)
    sigma_s_sq_2 = 2 * sigma_s**2

    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            dist_sq = float(i**2 + j**2) # 距離平方
            # 空間核輸入: dist^2 / 2*sigma_s^2
            exp_input = dist_sq / sigma_s_sq_2
            # 查找並獲得 Q4.14 定點數權重
            spatial_kernel_fixed[i + r, j + r] = fixed_point_exp_lookup(exp_input, lut_array, max_lut_index)
    
    # 轉換空間核到浮點數 (加速後續運算，雖然權重仍然是 LUT 產生的)
    spatial_kernel_float = fixed_to_float(spatial_kernel_fixed, Q_OUT_FRACT_BITS)


    # 滑動窗口掃描
    for i in range(h):
        for j in range(w):
            
            # 1. 初始化當前像素的計算
            I_p = I[i, j]
            numerator_fixed = 0 # 分子 (加權和) - 使用 Q4.14 模擬
            denominator_fixed = 0 # 分母 (歸一化因子) - 使用 Q4.14 模擬
            
            # 2. 掃描鄰域 (窗口)
            for m in range(-r, r + 1):
                for n in range(-r, r + 1):
                    q_i, q_j = i + m, j + n
                    
                    # 邊界檢查
                    if 0 <= q_i < h and 0 <= q_j < w:
                        I_q = I[q_i, q_j]
                        
                        # --- 範圍核計算 (Range Kernel) ---
                        # 差異平方: |I(p) - I(q)|^2
                        diff_sq = (I_p - I_q)**2
                        sigma_r_sq_2 = 2 * sigma_r**2
                        
                        # 範圍核輸入: diff^2 / 2*sigma_r^2 (非負)
                        range_exp_input = diff_sq / sigma_r_sq_2
                        
                        # 查找並獲得 Q4.14 定點數權重
                        range_weight_fixed = fixed_point_exp_lookup(range_exp_input, lut_array, max_lut_index)
                        
                        # --- 總權重計算 ---
                        # 空間權重已經預先計算為 Q4.14 浮點數
                        spatial_weight_float = spatial_kernel_float[m + r, n + r]

                        # 由於目標是硬體實現，這裡必須對齊數據類型。
                        # 在 Python 浮點數環境下，我們將空間權重轉回 Q4.14 後再相乘，
                        # 但簡化起見，直接使用浮點數乘法 (確保邏輯符合 Bilateral)
                        
                        # 總權重 (Q4.14 * Q4.14，使用浮點數模擬結果)
                        total_weight = spatial_weight_float * fixed_to_float(range_weight_fixed, Q_OUT_FRACT_BITS)
                        
                        # 累積
                        denominator_fixed += total_weight # 歸一化因子的浮點數總和
                        numerator_fixed += total_weight * I_q # 加權平均的浮點數總和
            
            # 3. 歸一化
            if denominator_fixed > 0:
                B[i, j] = numerator_fixed / denominator_fixed
            else:
                B[i, j] = I_p # 避免除以零
                
    return B.astype(np.float32)


# --- 修改核心 LTM 流程 ---

def local_tone_mapping_lut(hdr_image_linear, d, sigma_s, sigma_r, contrast, epsilon, output_gamma, lut_array):
    """
    執行使用客製化雙邊濾波器 (LUT 加速) 的 LTM 流程。
    """
    R_orig, G_orig, B_orig = [hdr_image_linear[..., i] for i in range(3)]

    # --- 1. 計算亮度 (Luminance) ---
    L = 0.2126 * R_orig + 0.7152 * G_orig + 0.0722 * B_orig

    # --- 2. 對數轉換 ---
    I = np.log10(L + epsilon)

    # --- 3. 客製化雙邊濾波 (提取基礎層 B) ---
    if lut_array is None:
        raise ValueError("LUT 載入失敗，無法執行客製化雙邊濾波。")
        
    print(f"3. 執行客製化雙邊濾波 (D={d}, $\sigma_s$={sigma_s}, $\sigma_r$={sigma_r})...")
    B = custom_bilateral_filter_with_lut(I, d, sigma_s, sigma_r, lut_array)
    print(f"   Bilateral Filtered B Max: {B.max():.4f}, Min: {B.min():.4f}")

    # --- 4. 分解為細節層 D ---
    D = I - B

    # --- 5. 基礎層壓縮 ---
    max_B = B.max()
    min_B = B.min()
    B_range = max_B - min_B
    k = np.log10(contrast) / (B_range + epsilon) if B_range >= epsilon else 0.0
    B_compressed = B * k

    # --- 6. 重建與色彩還原 (Reconstruction) ---
    I_prime = B_compressed + D
    L_prime = 10**(I_prime)
    L_safe = np.where(L > epsilon, L, epsilon)
    ratio = L_prime / L_safe

    R_final = R_orig * ratio
    G_final = G_orig * ratio
    B_final = B_orig * ratio
    LDR_final_linear = np.stack([R_final, G_final, B_final], axis=-1)
    
    # --- 7. 輸出編碼與量化 (檔案儲存專用) ---
    white_point = np.percentile(LDR_final_linear, 99.9) 
    LDR_final_normalized = np.clip(LDR_final_linear / white_point, 0, 1)
    LDR_final_gamma = LDR_final_normalized**(1/output_gamma)
    LDR_final_8bit_rgb = (LDR_final_gamma * 255).astype(np.uint8)
    LDR_final_8bit_bgr = cv2.cvtColor(LDR_final_8bit_rgb, cv2.COLOR_RGB2BGR)

    return LDR_final_8bit_bgr

# --- 主程式區塊：請修改此處的檔案路徑 ---
if __name__ == '__main__':
    # 💡 假設 LUT 檔案位於與腳本相同的目錄
    LUT_EXCEL_PATH = "LUT.xlsx" 
    HDR_FILE_PATH = "path/to/your/input_hdr_image.hdr" 
    LDR_OUTPUT_PATH = "path/to/your/output_ldr_image_lut.png" 
    
    # 預載入和處理 LUT
    lut_array_fixed = load_and_prepare_lut(LUT_EXCEL_PATH)

    if lut_array_fixed is not None:
        try:
            # 1. 讀取 HDR 檔案
            hdr_input = read_hdr_image(HDR_FILE_PATH)
            
            print("\n--- 開始局部色調映射 (LUT-Bilateral) 流程 ---")
            
            # 2. 執行色調映射和最終編碼
            final_ldr_8bit_bgr = local_tone_mapping_lut(
                hdr_input, 
                FILTER_D, 
                SIGMA_S, 
                SIGMA_R, 
                CONTRAST, 
                EPSILON,
                OUTPUT_GAMMA,
                lut_array_fixed
            )
            
            # 3. 儲存檔案
            save_ldr_file(final_ldr_8bit_bgr, LDR_OUTPUT_PATH)
            
        except FileNotFoundError as e:
            print(f"錯誤: {e}\n請確認 HDR_FILE_PATH 和 LUT_EXCEL_PATH 替換為有效的檔案路徑。")
        except Exception as e:
            print(f"發生其他錯誤: {e}")
    else:
        print("由於 LUT 載入失敗，程式無法執行客製化雙邊濾波。")