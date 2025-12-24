import cv2
import numpy as np
import pandas as pd 
import os
import time
import math
import warnings

def generate_output_dat_filename(pattern_number):
  return f"output_{pattern_number:03d}.dat"

def save_output_dat_file(rgb_image, filename):
    """
    將 8-bit RGB 影像數據儲存為 24-bit hex 格式的 .dat 檔 (Golden File)。
    每個 pixel 一行，格式: RRGGBB (Hex)
    """
    # 確保是 uint8
    if rgb_image.dtype != np.uint8:
        rgb_image = rgb_image.astype(np.uint8)
        
    H, W, C = rgb_image.shape
    if C != 3:
        raise ValueError("Image must have 3 channels (RGB).")

    # 確保目錄存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    print(f"Saving output dat file to {filename} ({W}x{H})...")
    with open(filename, "w") as f:
        # 使用向量化操作加速字串格式化 (比雙層迴圈快很多)
        # R << 16 | G << 8 | B
        packed_rgb = (rgb_image[..., 0].astype(np.uint32) << 16) | \
                     (rgb_image[..., 1].astype(np.uint32) << 8) | \
                     rgb_image[..., 2].astype(np.uint32)
        
        # 展平並寫入
        for val in packed_rgb.flatten():
            f.write(f"{val:06x}\n")
            
    print(f"✔ Output DAT file saved: {filename}")

def generate_hdr_filename(pattern_number):
  """
  根據給定的數字產生 PAT_XXX.hdr 格式的檔名。

  Args:
    pattern_number (int): 0 到 999 之間的數字。

  Returns:
    str: 格式化後的檔名字串，例如 "PAT_005.hdr"。
  """
  # 使用 f-string 進行格式化
  # {pattern_number:03d} 的意思是：
  #   - 0: 不足的位數用 0 填充
  #   - 3: 總寬度為 3 位
  #   - d: 以十進位整數形式格式化
  return f"PAT_{pattern_number:03d}.hdr"

def generate_png_filename(pattern_number):
  return f"PAT_{pattern_number:03d}.png"

def generate_spng_filename(pattern_number):
  return f"PAT_{pattern_number:03d}_s.png"

def generate_input_dat_filename(pattern_number):
  return f"input_{pattern_number:03d}.dat"

def generate_lgnum_dat_filename(pattern_number):
  return f"lglum_{pattern_number:03d}.dat"

def generate_basel_dat_filename(pattern_number):
  return f"basel_{pattern_number:03d}.dat"

def write_dat_4bytes(hdr4, filename):
    """
    hdr4: H x W x 4 uint8 RGBE，E 已 clip 到 4-bit signed範圍
    filename: 輸出的 .dat 檔
    每個 pixel 4 bytes (R,G,B,E)，直接 hex 寫入，每行一個 pixel
    """
    H, W, _ = hdr4.shape
    with open(filename, "w") as f:
        for y in range(H):
            for x in range(W):
                pixel = hdr4[y, x]

                # 直接用 4 個 f.write (hex)
                f.write("{:02X}".format(int(pixel[0].item())))  # R
                f.write("{:02X}".format(int(pixel[1].item())))  # G
                f.write("{:02X}".format(int(pixel[2].item())))  # B
                f.write("{:02X}".format(int(pixel[3].item())))  # E
                f.write(f" // {pixel[0]} {pixel[1]} {pixel[2]} {pixel[3]}")
                f.write("\n")  # 每行一個 pixel
    print("✔ DAT file saved:", filename)

def analyze_and_save_dat_fixed_point(I, filename='I_values.dat'):
    """
    Analyzes an INT32 numpy array assuming it contains Q7.14 fixed-point integer values.
    It checks if the values are within the valid 22-bit range, reports any
    out-of-range values, and saves the numbers in 32-bit hexadecimal format to a
    .dat file.

    Each line in the output file includes the original integer value as a comment.
    The output format is: `xxxxxxxx  // original_integer_value`

    The valid integer range for Q7.14 (21-bit signed) is [-1048576, 1048575].
    """
    arr = np.asarray(I)
    if not np.issubdtype(arr.dtype, np.signedinteger):
        raise TypeError("Input array must be a signed integer type.")

    stats = {}
    stats['dtype'] = str(arr.dtype)
    stats['shape'] = list(arr.shape)
    if arr.size > 0:
        stats['min'] = int(arr.min())
        stats['max'] = int(arr.max())
    else:
        stats['min'] = None
        stats['max'] = None

    print("[analyze_and_save_dat] Input dtype:", stats['dtype'], "shape:", stats['shape'])
    print("[analyze_and_save_dat] Input integer min/max:", stats['min'], stats['max'])

    MIN_Q7_14_INT = -1048576  # -2**20
    MAX_Q7_14_INT = 1048575   # 2**20 - 1

    H, W = arr.shape
    out_of_range_found = False

    with open(filename, "w") as f:
        for y in range(H):
            for x in range(W):
                # 儲存原始數值，用於註解
                original_pixel_val = int(arr[y, x])
                
                # 複製一份數值來進行處理 (clamping 和轉換)
                processed_pixel_val = original_pixel_val

                # 1. 檢查原始數值是否在範圍內
                if not (MIN_Q7_14_INT <= original_pixel_val <= MAX_Q7_14_INT):
                    warnings.warn(
                        f"[Out of Range] Value {original_pixel_val} at position ({y}, {x}) is outside "
                        f"the Q7.14 integer range [{MIN_Q7_14_INT}, {MAX_Q7_14_INT}]."
                    )
                    out_of_range_found = True
                    # 將要處理的數值截斷到有效範圍
                    processed_pixel_val = np.clip(original_pixel_val, MIN_Q7_14_INT, MAX_Q7_14_INT)

                # 2. 處理負數的二補數 (基於 32-bit)
                if processed_pixel_val < 0:
                    processed_pixel_val += (1 << 32)

                # 3. 格式化為 8 位十六進制數
                hex_val = f"{processed_pixel_val:08x}"

                # 4. 寫入檔案，包含對齊的註解
                #    {hex_val:<10} 表示將 hex_val 左對齊，佔用 10 個字元寬度
                f.write(f"{hex_val:<10} // {original_pixel_val}\n")

    if not out_of_range_found:
        print("✔ All values were within the Q7.14 range.")

    print(f"✔ DAT file saved: {filename}")

def load_lut_from_excel(file_path, input_col, output_col):
    """
    讀取 Excel 並回傳輸入(X)與輸出(Y)的對照陣列。
    """
    try:
        df = pd.read_excel(file_path)
        if input_col not in df.columns or output_col not in df.columns:
            print(f"錯誤: 檔案 {file_path} 中找不到欄位 '{input_col}' 或 '{output_col}'")
            return None, None
        # 確保數據按輸入值由小到大排序 (np.interp 需要排序過的 X)
        df = df.sort_values(by=input_col)
        
        lut_x = df[input_col].values
        lut_y = df[output_col].values
        
        print(f"LUT 載入成功。範圍: {lut_x.min()} ~ {lut_x.max()}, 點數: {len(lut_x)}")
        return lut_x, lut_y
    except Exception as e:
        print(f"讀取 LUT 失敗: {e}")
        return None, None
    
def load_and_prepare_lut(excel_path, sheet_name, nrows):
    """
    載入 Excel 檔案，構建 LUT 查找表。
    假設 Column 0 (輸入) 和 Column 1 (輸出) 已經是量化後的整數定點數。
    """
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, skiprows=1, nrows=nrows, 
                             usecols=[0, 3], dtype=np.int64)

        if len(df) < nrows:
            print(f"警告: LUT 讀取行數少於預期的 {nrows} 行，實際讀取 {len(df)} 行。")

        input_fixed_indices = df.iloc[:, 0].values
        output_fixed_weights = df.iloc[:, 1].values 

        if np.isnan(input_fixed_indices).any() or np.isnan(output_fixed_weights).any():
              raise ValueError("LUT 數據中包含非整數或缺失值 (NaN)。")

        lut_array = np.zeros(nrows, dtype=np.int64)

        for idx, val in zip(input_fixed_indices, output_fixed_weights):
            if 0 <= idx < nrows:
                lut_array[idx] = val

        print(f"LUT 載入成功，大小: {nrows} 點。")
        return lut_array

    except Exception as e:
        raise RuntimeError(f"載入或處理 LUT 檔案時發生錯誤: {e}") from e

def write_matrix_to_text_file_int(matrix, file_path):
    """
    將二維 NumPy 矩陣 (float) 轉換為 Q7.14 整數後寫入純文字檔案。
    第一行格式: W H
    內容: 整數值，以空格隔開。
    """
    try:
        H, W = matrix.shape
        
        # 1. 放大 2^14 倍並使用 floor (向下取整) 
        # 這樣負數的捨入行為才會與 Verilog 的 [MSB:LSB] 切片一致
        # 例如: -0.0001 * 16384 = -1.6384 -> floor 後變成 -2
        matrix_int = matrix.astype(np.int64)
        
        # 2. 準備寫入內容
        header = f"{W} {H}\n"
        
        with open(file_path, 'w') as f:
            f.write(header)
            # 3. 使用 fmt='%d' 確保儲存為整數格式
            np.savetxt(f, matrix_int, fmt='%d', delimiter=' ')

        print(f"  成功儲存整數矩陣 (Q7.14) 到: {file_path} ({W}x{H})")
    except Exception as e:
        print(f"寫入檔案 {file_path} 失敗: {e}")

def write_matrix_to_text_file(matrix, file_path):
    """
    將二維 NumPy 矩陣寫入純文字檔案。
    第一行格式: W H
    內容: 數字間以空格隔開。
    """
    try:
        H, W = matrix.shape
        
        # 準備要寫入的內容
        header = f"{W} {H}\n"
        
        # 使用 numpy.savetxt 寫入內容，並指定分隔符和格式
        # fmt='%.6f' 確保浮點數精度，delimiter=' ' 確保空格分隔
        with open(file_path, 'w') as f:
            f.write(header)
            np.savetxt(f, matrix, fmt='%.6f', delimiter=' ')

        print(f"  成功儲存矩陣 I (對數亮度) 到: {file_path} ({W}x{H})")
    except Exception as e:
        print(f"寫入檔案 {file_path} 失敗: {e}")

# 輔助函式：從指定格式的純文字檔讀取矩陣
def read_matrix_from_text_file(file_path):
    """
    從純文字檔案讀取矩陣。
    第一行格式: W H
    """
    try:
        with open(file_path, 'r') as f:
            # 讀取第一行以獲取長寬
            header = f.readline().strip()
            W, H = map(int, header.split())
            
            # 使用 numpy.loadtxt 讀取剩餘的數據
            matrix = np.loadtxt(f)
            
            # 檢查讀取到的形狀是否符合預期 (H, W)
            if matrix.shape != (H, W):
                 # numpy.loadtxt可能會將單行矩陣讀取為一維陣列，需要手動reshape
                if matrix.size == H * W:
                    matrix = matrix.reshape(H, W)
                else:
                    raise ValueError(f"讀取到的數據形狀 {matrix.shape} 與標頭 ({H}, {W}) 不匹配。")
            
            print(f"  成功從 {file_path} 讀取矩陣 B (基礎層) ({W}x{H})")
            return matrix

    except Exception as e:
        print(f"讀取檔案 {file_path} 失敗: {e}")
        return None

def enforce_q_precision(f_value, fract_bits, n_bits):    
    # 縮放：將小數部分移到整數部分
    max = (1 << (n_bits - 1)) - 1
    min = -(1 << (n_bits - 1))
    scale_factor = 1 << fract_bits
    scaled_value = f_value * scale_factor
    fixed_value_unclipped = np.trunc(scaled_value).astype(np.int64) 
    fixed_value_clipped = np.clip(fixed_value_unclipped, min, max)
    # 轉換回浮點數 (模擬硬體輸出)
    return fixed_value_clipped / scale_factor

FILTER_D = 5        # 濾波器直徑 (d)
SIGMA_R = 1.0       # 範圍標準差 (sigmaColor/sigmaRange): 邊緣敏感度閾值
SIGMA_S = 1.5       # 空間標準差 (sigmaSpace): 模糊半徑
CONTRAST = 100.0      # 基礎層壓縮參數：目標對比度 (關鍵可調參數)
EPSILON = 1e-6      # 防止 log(0) 錯誤

def local_tone_mapping_lut(Luminance_FILE_PATH, Bmatrix_FILE_PATH, R, G, B, E, lut_data_l=None, lut_data_e=None, pat_number=0):
    """執行使用客製化雙邊濾波器 (LUT 加速) 的 LTM 流程。"""

    # --- 1. 計算亮度 (Luminance) ---
    # 硬體公式: Sum = 256 權重 + 128 Bias (依據你的code)
    Lm = 54 * R + 183 * G + 19 * B
    
    # 計算真實浮點數亮度 (用於後續還原)
    E_float = E.astype(np.float32)
    L = Lm * np.exp2(E_float - 144)

    # --- 2. 硬體 Log10 模擬 (關鍵修正區) ---
    lut_x_l, lut_y_l = lut_data_l
    lut_y_l = np.array(lut_y_l).astype(np.int32) # LUT 轉 int32

    # [FIX 1] 避免 log2(0) 造成 -inf
    Lm_safe = np.maximum(Lm, 1)

    # [FIX 2] 向量化計算 MSB (比 pixel-by-pixel 快1000倍且準確)
    msb = np.floor(np.log2(Lm_safe)).astype(np.int32)
    
    TARGET_MSB = 15
    shift = TARGET_MSB - msb
    
    # [FIX 3 - 解決黑點!] 必須轉成 int32 再移位，否則 16-bit 移位會溢位變成 0
    reg = Lm.astype(np.int32) << shift

    # 取出 Index (Bit 14~3)
    idx = (reg >> 3) & 0xFFF
    base = lut_y_l[idx]

    # 計算 Exponent
    # 假設你的權重對應是 -16 (Sum=256 是 -8, 這裡可能是配合其他縮放)
    exp_val = (E.astype(np.int32) - 128) + msb - 16
    LOG2_CONST = int(math.log10(2) * (1 << 14)) # Q14 format
    exp_log = exp_val * LOG2_CONST

    # 得到 Log 域的亮度 (I)
    I = base + exp_log

    # Save and analyze I values
    
    try:
        lglum_filename = generate_lgnum_dat_filename(pat_number)
        analyze_and_save_dat_fixed_point(I, os.path.join("dat_file/lglum", lglum_filename))
    except Exception as e:
        print(f"Failed to analyze/save I: {e}")

    # 3. 儲存 I 矩陣 (對數亮度)
    
    write_matrix_to_text_file_int(I , Luminance_FILE_PATH)

    # I = I / 16384.0

    # log 函數(輸出有進行定點數處理)
    # I = enforce_q_precision(np.log10(L + EPSILON), 8, 16)

    
    # write_matrix_to_text_file(L, "act_data/Lm.txt")
    print(f"\n==================================================================")
    print(f"等待 C++ 處理：請執行 C++ 雙邊濾波器，將結果寫入 {Bmatrix_FILE_PATH}")
    print(f"==================================================================")
    # 計算完的 B 會在 SRAM

    # 4. 等待 B_matrix.txt 檔案存在
    print(f"檢查檔案 {Bmatrix_FILE_PATH}...")
    while not os.path.exists(Bmatrix_FILE_PATH):
        print("  檔案不存在，等待 1 秒...")
        time.sleep(1)
    
    # 5. 讀取 B 矩陣 (基礎層)
    BB = read_matrix_from_text_file(Bmatrix_FILE_PATH)
    # 只有軟體會有檢查步驟 之後硬體實作不會有
    if BB is None:
        raise RuntimeError("無法從 B_matrix.txt 讀取基礎層矩陣，終止 LTM 流程。")
    # 檢查 B 的尺寸是否與 I 匹配
    if BB.shape != I.shape:
        raise ValueError(f"讀取的 B 矩陣形狀 {B.shape} 與 I 矩陣形狀 {I.shape} 不匹配。")
    # 把 B 存到 dat
    B_dat = BB.astype(np.int32)

    try:
        basel_filename = generate_basel_dat_filename(pat_number)
        analyze_and_save_dat_fixed_point(B_dat, os.path.join("dat_file/basel", basel_filename))
    except Exception as e:
        print(f"Failed to analyze/save Base Layer: {e}")

    # B = B.astype(np.float32) / 16384.0
    BB = BB.astype(np.int64)
    I = I.astype(np.int64)
    print(f"I range from {I.min()} to {I.max()}") # range: Q7.14

    # 4. 分解為細節層 D
    D = I - BB
    print(f"D range from {D.min()} to {D.max()}") # range: Q7.14

    # 5. 基礎層壓縮
    # 搜索整個 B matrix 找到 B_range
    max_B = BB.max()
    min_B = BB.min()
    print(f"BB range from {min_B} to {max_B}") # range: Q6.6
    B_range = max_B - min_B # Q7.14
    B_range.astype(np.int64)
    print(B_range, B_range / 2**8, np.floor(B_range / 2**8))
    divide_lut_index = np.floor(B_range / 2**8).astype(np.int32)
    k = divide_lut[divide_lut_index] * 2 # input Q6.6 output Q6.12, contrast = 100
    # k = 1 / (B_range + EPSILON) if B_range >= EPSILON else 0.0 # 因為 contrast = 10 ，所以分子就是 1
    B_compressed = BB * k # 7.14 * 6.12
    B_compressed = np.floor(B_compressed / 2**12).astype(np.int32) #Q13.14

    # 6. 重建與色彩還原 (Reconstruction)
    I_prime = B_compressed + D
    print(f"I_prime range from {I_prime.min()} to {I_prime.max()}")
    LOG_2_10_FIXED = 108853 # 17-bit Q2.15

    I_safe = I
    I_ratio = I_prime - I_safe # 除法
    print(f"I_ratio range from {I_ratio.min()} to {I_ratio.max()}")
    temp_log2 = np.trunc(I_ratio*LOG_2_10_FIXED / 2**(15)).astype(np.int32)
    print(f"temp_log2 range from {temp_log2.min()} to {temp_log2.max()}")
    I_int = np.floor(temp_log2 / 2**(14)).astype(np.int32)
    print(f"I_int range from {I_int.min()} to {I_int.max()}")
    I_frac = temp_log2 - (I_int.astype(np.int64) * (2**14))
    print(f"I_frac range from {I_frac.min()} to {I_frac.max()}")
    power_lut_index = np.floor(I_frac / 2**2).astype(np.int32)
    print(f"power_lut_index range from {power_lut_index.min()} to {power_lut_index.max()}")
    ratio = power_lut[power_lut_index] # 查表與位移
    # ratio_fix = enforce_q_precision(ratio, 12, 21) # 模擬硬體定點數輸出 // UQ
    # ratio_fix_raw = (ratio_fix * 4096).astype(np.int64)

    print(f"ratio range from {ratio.min()} to {ratio.max()}")
    print(f"E range from {E.min()} to {E.max()}")
    total_shift = E.astype(np.int32) - 140 + I_int.astype(np.int32)
    print(f"total_shift range from {total_shift.min()} to {total_shift.max()}")

    R_temp = R.astype(np.int64) * ratio
    G_temp = G.astype(np.int64) * ratio
    B_temp = B.astype(np.int64) * ratio

    # 模擬桶形移位器 (Barrel Shifter)
    # 當 total_shift 為負時 (如 -23)，執行右移 23 位
    # 當 total_shift 為正時，執行左移
    R_final_int = np.where(total_shift >= 0, 
                        R_temp << np.abs(total_shift), 
                        R_temp >> np.abs(total_shift))

    G_final_int = np.where(total_shift >= 0, 
                        G_temp << np.abs(total_shift), 
                        G_temp >> np.abs(total_shift))

    B_final_int = np.where(total_shift >= 0, 
                        B_temp << np.abs(total_shift), 
                        B_temp >> np.abs(total_shift))
    
    print(f"R_final_int range from {R_final_int.min()} to {R_final_int.max()}")
    print(f"G_final_int range from {G_final_int.min()} to {G_final_int.max()}")
    print(f"B_final_int range from {B_final_int.min()} to {B_final_int.max()}")

    print(f"R_final_int range from {np.clip(R_final_int, 0, 255).min()} to {np.clip(R_final_int, 0, 255).max()}")
    print(f"G_final_int range from {np.clip(G_final_int, 0, 255).min()} to {np.clip(G_final_int, 0, 255).max()}")
    print(f"B_final_int range from {np.clip(B_final_int, 0, 255).min()} to {np.clip(B_final_int, 0, 255).max()}")

    # 最後進行 8-bit 飽和與型別轉換
    R_out = np.clip(R_final_int, 0, 255).astype(np.uint8)
    G_out = np.clip(G_final_int, 0, 255).astype(np.uint8)
    B_out = np.clip(B_final_int, 0, 255).astype(np.uint8)
    

    # LDR_final_linear = np.stack([R_final, G_final, B_final], axis=-1)
    
    # # 7. 輸出編碼與量化 (檔案儲存專用)
    # LDR_final_normalized = np.clip(LDR_final_linear, 0, 1)
    # LDR_final_8bit_rgb = (LDR_final_normalized * 255).astype(np.uint8) # 把 RGB 結果存進 SRAM
    # LDR_final_8bit_bgr = cv2.cvtColor(LDR_final_8bit_rgb, cv2.COLOR_RGB2BGR)
    LDR_final_8bit_bgr = cv2.merge([B_out, G_out, R_out])

    LDR_final_8bit_rgb = np.stack([R_out, G_out, B_out], axis=-1)
    output_dat_name = generate_output_dat_filename(pat_number)
    save_output_dat_file(LDR_final_8bit_rgb, os.path.join("dat_file/output", output_dat_name))

    return LDR_final_8bit_bgr

def local_tone_mapping_opencv(R, G, B, E, lut_data_l=None):
    R_orig = (R / 256.0) * np.power(2, E-128.0)
    G_orig = (G / 256.0) * np.power(2, E-128.0)
    B_orig = (B / 256.0) * np.power(2, E-128.0)

   # --- 1. 計算亮度 (Luminance) ---
    # 硬體公式: Sum = 256 權重 + 128 Bias (依據你的code)
    # 轉型為 uint16 計算，避免乘法溢位
    R_orig = R_orig.astype(np.float32)
    G_orig = G_orig.astype(np.float32)
    B_orig = B_orig.astype(np.float32)
    E = E.astype(np.float32)

    L = R_orig * 0.2126 + G_orig * 0.7152 + B_orig * 0.0722
    # Lm = 54 * R + 183 * G + 19 * B + 128
    
    # 計算真實浮點數亮度 (用於後續還原)
    # E_float = E.astype(np.float32)
    # L = Lm * np.exp2(E_float - 144)

    # # --- 2. 硬體 Log10 模擬 (關鍵修正區) ---
    # lut_x_l, lut_y_l = lut_data_l
    # lut_y_l = np.array(lut_y_l).astype(np.int32) # LUT 轉 int32

    # # [FIX 1] 避免 log2(0) 造成 -inf
    # Lm_safe = np.maximum(Lm, 1)

    # # [FIX 2] 向量化計算 MSB (比 pixel-by-pixel 快1000倍且準確)
    # msb = np.floor(np.log2(Lm_safe)).astype(np.int32)
    
    # TARGET_MSB = 15
    # shift = TARGET_MSB - msb
    
    # # [FIX 3 - 解決黑點!] 必須轉成 int32 再移位，否則 16-bit 移位會溢位變成 0
    # reg = Lm.astype(np.int32) << shift

    # # 取出 Index (Bit 14~3)
    # idx = (reg >> 3) & 0xFFF
    # base = lut_y_l[idx]

    # # 計算 Exponent
    # # 假設你的權重對應是 -16 (Sum=256 是 -8, 這裡可能是配合其他縮放)
    # exp_val = (E.astype(np.int32) - 128) + msb - 16
    # LOG2_CONST = int(math.log10(2) * (1 << 14)) # Q14 format
    # exp_log = exp_val * LOG2_CONST

    # # 得到 Log 域的亮度 (I)
    # I = base + exp_log
    I = np.log10(L + EPSILON)

    # --- 3. 雙邊濾波與 Tone Mapping ---
    # I 是 Q14，需要轉回 float 進行 OpenCV 濾波
    # 注意: 如果 LUT output 是 Q14，這裡除以 16384.0 (2^14) 比較合理
    # 但你的 code 之前是除以 1024，請確認你的 LUT 數值縮放
    # 這裡假設你的 base 和 exp_log 都是 Q14
    I_safe = np.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0)
    # I_float32 = I.astype(np.float32) / 16384.0

    B = cv2.bilateralFilter(I, FILTER_D, SIGMA_R, SIGMA_S)

    # --- 4. 分解為細節層 D ---
    D = I - B

    # --- 5. 基礎層壓縮 ---
    max_B = B.max()
    min_B = B.min()
    B_range = max_B - min_B
    k = np.log10(CONTRAST) / (B_range + EPSILON) if B_range >= EPSILON else 0.0
    B_compressed = B * k

    # --- 6. 重建與色彩還原 (Reconstruction) ---
    I_prime = B_compressed + D
    L_prime = 10**(I_prime)
    
    L_safe = np.where(L > EPSILON, L, EPSILON)
    ratio = L_prime / L_safe

    R_final = R_orig * ratio
    G_final = G_orig * ratio
    B_final = B_orig * ratio
    
    # 線性 LDR 輸出
    LDR_final_linear = np.stack([R_final, G_final, B_final], axis=-1)
    
    # --- 7. 輸出編碼與量化 (檔案儲存專用) ---
    
    # a. 正規化 (到 [0, 1] 範圍)
    LDR_final_normalized = np.clip(LDR_final_linear, 0, 1)
    
    # c. 量化 (轉換為 8-bit 整數 [0, 255])
    LDR_final_8bit_rgb = (LDR_final_normalized * 255).astype(np.uint8)
    
    # d. 轉換回 BGR 順序 (OpenCV 的 imwrite 默認使用 BGR)
    LDR_final_8bit_bgr = cv2.cvtColor(LDR_final_8bit_rgb, cv2.COLOR_RGB2BGR)

    return LDR_final_8bit_bgr

def read_hdr_image(file_path):
    """
    使用 OpenCV 讀取標準 HDR 檔案 (.hdr 或 .exr)
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
    
    TARGET_HEIGHT = 720  # 目標高度 (H)
    TARGET_WIDTH = 1280   # 目標寬度 (W)
    
    # 檢查原始影像是否足夠大
    original_height = hdr_rgb_linear.shape[0]
    original_width = hdr_rgb_linear.shape[1]
    
    if original_height <= TARGET_HEIGHT or original_width <= TARGET_WIDTH:
        print(f"警告: 原始影像大小 {original_width}x{original_height} 小於目標裁剪尺寸 {TARGET_WIDTH}x{TARGET_HEIGHT}。")
        print("將返回原始影像。")
        return hdr_rgb_linear

    # 使用 NumPy 切片功能：[起始行:結束行, 起始列:結束列, 所有通道]
    # 從左上角 (0, 0) 開始裁剪
    hdr_rgb_cropped = hdr_rgb_linear[
        0:TARGET_HEIGHT, 
        0:TARGET_WIDTH, 
        :
    ]
    
    print(f"影像已成功裁剪。新解析度: {hdr_rgb_cropped.shape[1]}x{hdr_rgb_cropped.shape[0]}")
    
    return hdr_rgb_cropped

def read_hdr_rgbe(path):
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode(errors="ignore")
            if line.strip()=="":
                break

        line=f.readline().decode().strip().split()
        H=int(line[1])
        W=int(line[3])

        img=np.zeros((H,W,4),dtype=np.uint8)

        for y in range(H):
            header=f.read(4)
            if header[0]!=2 or header[1]!=2:
                raise ValueError("Not RLE Radiance HDR")

            scan = np.zeros((W,4),dtype=np.uint8)
            for c in range(4):
                x=0
                while x<W:
                    val=ord(f.read(1))
                    if val>128:   # run
                        cnt=val-128
                        b=ord(f.read(1))
                        scan[x:x+cnt,c]=b
                        x+=cnt
                    else:       # literal
                        raw=f.read(val)
                        scan[x:x+val,c]=list(raw)
                        x+=val
            img[y]=scan
    R_m = img[..., 0].astype(np.uint16)
    G_m = img[..., 1].astype(np.uint16)
    B_m = img[..., 2].astype(np.uint16)
    E = img[..., 3].astype(np.uint8)
    return img,W,H,R_m,G_m,B_m,E

def log_lookup(value, lut_array):
    fixed_index = np.clip(value, 0, lut_array.shape[0] - 1)
    I_matrix = lut_array[fixed_index]
    return I_matrix

def save_ldr_file(image_data, output_path):
    """使用 OpenCV 將 8-bit 影像數據儲存為 LDR 檔案。"""
    success = cv2.imwrite(output_path, image_data)
    if success:
        print(f"成功儲存 LDR 檔案至: {output_path}")
    else:
        print(f"檔案儲存失敗: {output_path}")

if __name__ == '__main__':
    # I/O file name
    pat_number = 0
    HDR_FILE_NAME = generate_hdr_filename(pat_number)
    HDR_FILE_PATH = os.path.join("hdr_pat", HDR_FILE_NAME)
    LDR_FILE_NAME = generate_png_filename(pat_number)
    LDR_OUTPUT_PATH  = os.path.join("output_img", LDR_FILE_NAME)
    LDRS_FILE_NAME = generate_spng_filename(pat_number)
    LDRS_OUTPUT_PATH = os.path.join("output_img", LDRS_FILE_NAME)
    
    # Act file
    Luminance_FILE_PATH = "act_data/luminance.txt"
    Bmatrix_FILE_PATH   = "act_data/B_matrix.txt"

    # Lut file
    LUT_PATH = "lut/LUT.xlsx"
    Lm_LUT = "lut/Lm_base_LUT.xlsx"

    # input dat file
    input_dat_file = generate_input_dat_filename(pat_number)
    INPUT_DAT_PATH = os.path.join("dat_file/input", input_dat_file)

    os.makedirs("dat_file/output", exist_ok=True)

    try:
        divide_lut = load_and_prepare_lut(LUT_PATH, 'divide6Q6', 4096)
        divide1_lut = load_and_prepare_lut(LUT_PATH, 'divide0Q13', 8193)
        divide2_lut = load_and_prepare_lut(LUT_PATH, 'divide2_0Q13', 8193)
        power_lut = load_and_prepare_lut(LUT_PATH, 'power2', 8192)
        # 1. 讀取 LUT
        lut_x_l, lut_y_l = load_lut_from_excel(Lm_LUT, input_col="base 12 bit", output_col="1.base base value")
        if lut_x_l is None:
            raise ValueError("LUT 載入失敗，程式終止。")

        # hdr_input = read_hdr_image(HDR_FILE_PATH)
        rgbe_matrix, W, H, R_m, G_m, B_m, E = read_hdr_rgbe(HDR_FILE_PATH)
        write_dat_4bytes(rgbe_matrix, INPUT_DAT_PATH)

        # Software Path
        final_ldr_8bit_bgr1 = local_tone_mapping_opencv(R_m, G_m, B_m, E,
                                                        lut_data_l=(lut_x_l, lut_y_l)
                                                        )
        save_ldr_file(final_ldr_8bit_bgr1, LDRS_OUTPUT_PATH)
        # Hardware Path
        final_ldr_8bit_bgr = local_tone_mapping_lut(Luminance_FILE_PATH, Bmatrix_FILE_PATH, R_m, G_m, B_m, E,
                                                   lut_data_l=(lut_x_l, lut_y_l), pat_number=pat_number)
        save_ldr_file(final_ldr_8bit_bgr, LDR_OUTPUT_PATH)
        # os.remove(Bmatrix_FILE_PATH)  # 圖像處理完成後自動刪除 B_matrix 檔案
        
    except FileNotFoundError as e:
        print(f"錯誤: {e}\n請確認檔案路徑是否正確。")
    except Exception as e:
        print(f"發生其他錯誤: {e}")