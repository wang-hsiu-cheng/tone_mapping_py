import cv2
import numpy as np
import pandas as pd 

# --- I. 輔助函數與定點數配置 ---

# 定點數格式設定
Q_IN_FRACT_BITS = 10  # 輸入 exp 的小數部分位數 (Q2.10)
Q_OUT_FRACT_BITS = 10 # 輸出 exp 的小數部分位數 (Qx.10, 用於中間浮點數結果)

def fixed_to_float(fixed_value, fract_bits):
    """將定點數轉換回浮點數。"""
    return fixed_value / (1 << fract_bits)

# def enforce_q_precision(f_value, fract_bits):
#     """模擬將浮點數結果四捨五入到 Qx.fract_bits 精度，再轉回浮點數。"""
#     # 1. 轉換為定點數整數 (模擬硬體量化)
#     fixed_value = np.round(f_value * (1 << fract_bits)).astype(np.int32)
#     # 2. 轉換回浮點數 (模擬硬體輸出)
#     return fixed_value / (1 << fract_bits)

# MAX_FIXED_INTEGER_VALUE = (1 << (10 + 2)) - 1 # 4095
scale_factor = 1 << Q_IN_FRACT_BITS
N_BITS = 16
MAX_FIXED_VALUE = (1 << (N_BITS - 1)) - 1
MIN_FIXED_VALUE = -(1 << (N_BITS - 1))

def enforce_q_precision(f_value, fract_bits):
    """
    模擬將浮點數結果朝向零無條件捨去 (Truncation towards Zero) 到 Qx.fract_bits 精度，
    並限制整數部分的位數，同時保留符號。
    
    Args:
        f_value (np.ndarray or float): 輸入浮點數值。
        fract_bits (int): 小數部分位數 (F)。
        I_bits (int): 整數部分位數 (I)。
    
    Returns:
        np.ndarray: 模擬定點數精度和飽和後的浮點數結果。
    """
    
    # 1. 🌟 執行朝向零的無條件捨去 (Truncation towards Zero)
    # Truncate(x) = sign(x) * floor(|x|)
    
    # 縮放：將小數部分移到整數部分
    scaled_value = f_value * scale_factor
    
    # 使用 np.trunc 實現朝向零的無條件捨去 (這是最簡潔且正確的做法)
    fixed_value_unclipped = np.trunc(scaled_value).astype(np.int32) 
    
    # 2. 🌟 飽和/鉗位邏輯 (針對有符號定點數 Q(I+F))
    
    # 鉗位：確保數值在 MIN_FIXED_VALUE 到 MAX_FIXED_VALUE 之間
    fixed_value_clipped = np.clip(fixed_value_unclipped, MIN_FIXED_VALUE, MAX_FIXED_VALUE)
    
    # 3. 轉換回浮點數 (模擬硬體輸出)
    return fixed_value_clipped / scale_factor

def load_and_prepare_lut(excel_path, sheet_name='exp'):
    """
    載入 Excel 檔案，構建 LUT 查找表。
    假設 Column 0 (輸入) 和 Column 3 (輸出) 已經是量化後的整數定點數。
    """
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, skiprows=1, nrows=16384, 
                             usecols=[0, 3], dtype=np.int64)
        
        if len(df) < 16384:
            print(f"警告: LUT 讀取行數少於預期的 16384 行，實際讀取 {len(df)} 行。")
        
        input_fixed_indices = df.iloc[:, 0].values
        output_fixed_weights = df.iloc[:, 1].values 

        if np.isnan(input_fixed_indices).any() or np.isnan(output_fixed_weights).any():
              raise ValueError("LUT 數據中包含非整數或缺失值 (NaN)。")

        # 檢查索引範圍是否正確 (Q2.10 上限為 4095)
        MAX_Q2_10_INDEX = (1 << (4 + Q_IN_FRACT_BITS)) - 1
        
        if input_fixed_indices.min() < 0 or input_fixed_indices.max() > MAX_Q2_10_INDEX:
            print("-" * 50)
            print("🚨 錯誤檢查：LUT 索引超出 Q2.10 範圍。")
            print(f"轉換後的最大索引為 {input_fixed_indices.max()}，超過上限 {MAX_Q2_10_INDEX}。")
            raise ValueError("LUT 索引超出 Q2.10 (0-4095) 範圍，請檢查 Column 0 數值是否小於或等於 4095。")
            
        lut_size = 1 << (2 + Q_IN_FRACT_BITS)
        lut_array = np.zeros(lut_size, dtype=np.int64)
        
        for idx, val in zip(input_fixed_indices, output_fixed_weights):
            if 0 <= idx < lut_size:
                lut_array[idx] = val
            
        print(f"LUT 載入成功，大小: {lut_size} 點。")
        return lut_array
        
    except Exception as e:
        raise RuntimeError(f"載入或處理 LUT 檔案時發生錯誤: {e}") from e

def fixed_point_exp_lookup(value_float, lut_array, max_input_fixed):
    """
    使用 LUT 執行指數運算，輸入為浮點數，輸出為定點數權重。
    """
    if value_float < 0:
        value_float = 0 
    
    # 1. 計算 Q2.10 索引: round(value_float * 2^10)
    fixed_index = np.round(value_float * (1 << Q_IN_FRACT_BITS)).astype(np.int64)
    
    # 2. 邊界檢查和鉗位 
    fixed_index = np.clip(fixed_index, 0, max_input_fixed)
    
    # 3. 查找 (輸出是 Q4.14 定點數)
    return lut_array[fixed_index]

def read_hdr_image(file_path):
    """
    使用 OpenCV 讀取標準 HDR 檔案 (.hdr 或 .exr)，
    並將影像裁剪為左上角 640x480 的區域。
    """
    
    # --- 影像讀取與顏色轉換（保持不變） ---
    hdr_bgr = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    
    if hdr_bgr is None:
        raise FileNotFoundError(f"無法讀取檔案: {file_path}，請檢查檔案路徑或格式是否正確。")
    
    # 確保數據類型為 CV_32F (np.float32) 以避免 cvtColor 錯誤
    if hdr_bgr.dtype == np.float64:
        hdr_bgr = hdr_bgr.astype(np.float32)
        
    hdr_rgb_linear = cv2.cvtColor(hdr_bgr, cv2.COLOR_BGR2RGB)
    
    # --- 影像裁剪 ---
    
    TARGET_HEIGHT = 600  # 目標高度 (H)
    TARGET_WIDTH = 600   # 目標寬度 (W)
    
    # 檢查原始影像是否足夠大
    original_height = hdr_rgb_linear.shape[0]
    original_width = hdr_rgb_linear.shape[1]
    
    if original_height < TARGET_HEIGHT or original_width < TARGET_WIDTH:
        print(f"警告: 原始影像大小 {original_width}x{original_height} 小於目標裁剪尺寸 {TARGET_WIDTH}x{TARGET_HEIGHT}。")
        print("將返回原始影像。")
        return hdr_rgb_linear

    # 使用 NumPy 切片功能：[起始行:結束行, 起始列:結束列, 所有通道]
    # 從左上角 (0, 0) 開始裁剪
    hdr_rgb_cropped = hdr_rgb_linear[
        200:TARGET_HEIGHT, 
        100:TARGET_WIDTH, 
        :
    ]
    
    print(f"影像已成功裁剪。新解析度: {hdr_rgb_cropped.shape[1]}x{hdr_rgb_cropped.shape[0]}")
    
    return hdr_rgb_cropped

def save_ldr_file(image_data, output_path):
    """使用 OpenCV 將 8-bit 影像數據儲存為 LDR 檔案。"""
    success = cv2.imwrite(output_path, image_data)
    if success:
        print(f"成功儲存 LDR 檔案至: {output_path}")
    else:
        print(f"檔案儲存失敗: {output_path}")

# --- II. 客製化雙邊濾波核心 ---

def custom_bilateral_filter_with_lut(I, d, sigma_s, sigma_r, lut_array):
    """
    客製化雙邊濾波器，使用滑動窗口和 LUT 進行指數運算，並模擬 Qx.10 精度。
    """
    print("start custom bf")
    h, w = I.shape
    r = d // 2 # 半徑
    B = np.zeros_like(I, dtype=np.float32)
    
    Q_FRACT = Q_IN_FRACT_BITS # 10 位元小數精度用於中間結果模擬
    
    # max_lut_index = lut_array.shape[0] - 1
    
    # 預先計算空間核
    # spatial_kernel_fixed = np.zeros((d, d), dtype=np.int64)
    spatial_kernel_float = np.zeros((d, d), dtype=np.int64)

    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            dist_sq = float(i**2 + j**2)
            # 空間核輸入 (除法結果需要鉗位)
            # exp_input = enforce_q_precision(dist_sq / sigma_s_sq_2, Q_FRACT)
            # spatial_kernel_fixed[i + r, j + r] = fixed_point_exp_lookup(exp_input, lut_array, max_lut_index)
            spatial_kernel_float[i + r, j + r] = enforce_q_precision(np.exp(-dist_sq * SIGMA_S_2), 8)
    
    # spatial_kernel_float = fixed_to_float(spatial_kernel_fixed, Q_OUT_FRACT_BITS)

    # 滑動窗口掃描
    print("start scan")
    for i in range(h):
        # 顯示進度（僅為了除錯）
        # if i % 10 == 0:
        print(f"  Processing row {i}/{h}")
            
        for j in range(w):
            
            # 1. 初始化當前像素的計算
            I_p = I[i, j]
            numerator_float = 0.0 # 分子 (加權和)
            denominator_float = 0.0 # 分母 (歸一化因子)
            
            # 2. 掃描鄰域 (窗口)
            for m in range(-r, r + 1):
                for n in range(-r, r + 1):
                    q_i, q_j = i + m, j + n
                    
                    # 邊界檢查
                    if 0 <= q_i < h and 0 <= q_j < w:
                        I_q = I[q_i, q_j]
                        
                        # --- 範圍核計算 (Range Kernel) ---
                        diff_sq = enforce_q_precision((I_p - I_q)**2, Q_FRACT)
                        
                        # 範圍核輸入 (除法結果需要鉗位)
                        range_exp_input = enforce_q_precision(-diff_sq * SIGMA_R_2, Q_FRACT)

                        range_weight_float = enforce_q_precision(np.exp(range_exp_input), 6)
                        
                        # --- 總權重計算 ---
                        spatial_weight_float = spatial_kernel_float[m + r, n + r]
                        total_weight = enforce_q_precision(spatial_weight_float * range_weight_float, Q_FRACT)
                        
                        # 累積
                        weighted_I_q = enforce_q_precision(total_weight * I_q, Q_FRACT)
                        
                        denominator_float += total_weight
                        numerator_float += weighted_I_q
            
            # 3. 歸一化 (除法)
            if denominator_float > 0:
                B[i, j] = enforce_q_precision(numerator_float / denominator_float, Q_FRACT)
            else:
                B[i, j] = I_p # 避免除以零
                
    return B.astype(np.float32)

# --- III. LTM 主流程 ---

def local_tone_mapping_lut(hdr_image_linear, d, sigma_s, sigma_r, contrast, epsilon, output_gamma, lut_array):
    """執行使用客製化雙邊濾波器 (LUT 加速) 的 LTM 流程。"""
    R_orig, G_orig, B_orig = [hdr_image_linear[..., i] for i in range(3)]

    # 1. 計算亮度 (Luminance)
    L = 0.2126 * R_orig + 0.7152 * G_orig + 0.0722 * B_orig

    # 2. 對數轉換
    I = np.log10(L + epsilon)

    # 3. 客製化雙邊濾波 (提取基礎層 B)
    if lut_array is None:
        raise ValueError("LUT 載入失敗，無法執行客製化雙邊濾波。")
        
    B = custom_bilateral_filter_with_lut(I, d, sigma_s, sigma_r, lut_array)
    print(f"Bilateral Filtered B Max: {B.max():.4f}, Min: {B.min():.4f}")

    # 4. 分解為細節層 D
    D = I - B

    # 5. 基礎層壓縮
    max_B = B.max()
    min_B = B.min()
    B_range = max_B - min_B
    k = np.log10(contrast) / (B_range + epsilon) if B_range >= epsilon else 0.0
    B_compressed = B * k

    # 6. 重建與色彩還原 (Reconstruction)
    I_prime = B_compressed + D
    L_prime = 10**(I_prime)
    L_safe = np.where(L > epsilon, L, epsilon)
    ratio = L_prime / L_safe

    R_final = R_orig * ratio
    G_final = G_orig * ratio
    B_final = B_orig * ratio
    LDR_final_linear = np.stack([R_final, G_final, B_final], axis=-1)
    
    # 7. 輸出編碼與量化 (檔案儲存專用)
    # white_point = np.percentile(LDR_final_linear, 99.9) 
    LDR_final_normalized = np.clip(LDR_final_linear / 1, 0, 1)
    LDR_final_gamma = LDR_final_normalized**(1/output_gamma)
    LDR_final_8bit_rgb = (LDR_final_gamma * 255).astype(np.uint8)
    LDR_final_8bit_bgr = cv2.cvtColor(LDR_final_8bit_rgb, cv2.COLOR_RGB2BGR)

    return LDR_final_8bit_bgr

# --- IV. 主程式區塊 ---

# --- 參數設定 ---
FILTER_D = 5        # 濾波器直徑 (d)
SIGMA_R = 1.0       # 範圍標準差 (sigmaColor/sigmaRange): 邊緣敏感度閾值
SIGMA_S = 1.5       # 空間標準差 (sigmaSpace): 模糊半徑
CONTRAST = 10.0      # 基礎層壓縮參數：目標對比度 (關鍵可調參數)
EPSILON = 1e-6      # 防止 log(0) 錯誤
OUTPUT_GAMMA = 1  # 輸出 LDR 檔案所使用的 Gamma 值 (例如 sRGB/Rec. 709 接近 2.2)
SIGMA_R_2 = enforce_q_precision(1 / 2 * SIGMA_R**2, 6)
SIGMA_S_2 = enforce_q_precision(1 / 2 * SIGMA_S**2, 6)

if __name__ == '__main__':
    # 💡 請將這裡的路徑替換為您的實際檔案路徑 💡
    LUT_EXCEL_PATH = "LUT/LUT.xlsx" 
    HDR_FILE_PATH = "img/Desk.hdr" 
    LDR_OUTPUT_PATH = "img/Desk.png" 
    
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
            print(f"錯誤: {e}\n請確認檔案路徑是否正確。")
        except Exception as e:
            print(f"發生其他錯誤: {e}")
    else:
        print("由於 LUT 載入失敗，程式無法執行客製化雙邊濾波。")