import cv2
import numpy as np
import pandas as pd 
import os
import time
import math

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

def local_tone_mapping_lut(Luminance_FILE_PATH, Bmatrix_FILE_PATH, R, G, B, E, lut_data_l=None, lut_data_e=None):
    """執行使用客製化雙邊濾波器 (LUT 加速) 的 LTM 流程。"""
    R_orig = (R / 256.0) * np.power(2, E-128.0)
    G_orig = (G / 256.0) * np.power(2, E-128.0)
    B_orig = (B / 256.0) * np.power(2, E-128.0)

    # --- 1. 計算亮度 (Luminance) ---
    # 硬體公式: Sum = 256 權重 + 128 Bias (依據你的code)
    Lm = 54 * R + 183 * G + 19 * B + 128
    
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
    I = I / 16384

    # log 函數(輸出有進行定點數處理)
    # I = enforce_q_precision(np.log10(L + EPSILON), 8, 16)

    # 3. 儲存 I 矩陣 (對數亮度)
    write_matrix_to_text_file(I, Luminance_FILE_PATH)
    write_matrix_to_text_file(L, "data/Lm.txt")
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
    B = read_matrix_from_text_file(Bmatrix_FILE_PATH)
    # 只有軟體會有檢查步驟 之後硬體實作不會有
    if B is None:
        raise RuntimeError("無法從 B_matrix.txt 讀取基礎層矩陣，終止 LTM 流程。")
    # 檢查 B 的尺寸是否與 I 匹配
    if B.shape != I.shape:
        raise ValueError(f"讀取的 B 矩陣形狀 {B.shape} 與 I 矩陣形狀 {I.shape} 不匹配。")

    # 4. 分解為細節層 D
    D = I - B

    # 5. 基礎層壓縮
    # 搜索整個 B matrix 找到 B_range
    max_B = B.max()
    min_B = B.min()
    print(f"B range from {min_B} to {max_B}") # range: Q6.6
    B_range = max_B - min_B
    k = divide_lut[np.trunc(B_range * 64).astype(np.int32)] / (2**11) # input Q6.6 output Q6.12, contrast = 100
    # k = 1 / (B_range + EPSILON) if B_range >= EPSILON else 0.0 # 因為 contrast = 10 ，所以分子就是 1
    B_compressed = B * k

    # 6. 重建與色彩還原 (Reconstruction)
    I_prime = B_compressed + D
    print(f"I_prime range from {I_prime.min()} to {I_prime.max()}")
    LOG_2_10_FIXED = 108853 # 17-bit Q2.15
    I_int = np.trunc(I_prime*LOG_2_10_FIXED/(2**15))
    I_float = (I_prime*LOG_2_10_FIXED/(2**15)) - I_int # range: signed Q0.12
    L_prime = 2**(I_int) * power_lut[np.trunc(I_float*2048).astype(np.int32)] / 2048.0 # L_prime = 2**(I_prime*3.321928)
    # L_prime = 10**(I_prime)
    L = enforce_q_precision(L, 10, 19)
    L_safe = np.where(L > EPSILON, L, EPSILON) # 把 L=0 的值全部替換成一個極小值
    print(f"L_safe range from {L_safe.min()} to {L_safe.max()}") # range: Q8.5
    # 把 9.10 浮點數 L 轉成 Q9.5 定點數的查表 index
    L_lookup_idx = np.trunc(L * 32).astype(np.int16)+1
    # -- 方法一：泰勒展開(二階) -- divide1_lut 是一階項、divide2_lut 是二階項
    diff = L_safe*1024-L_lookup_idx*32 # fixed-point Q.10
    L_fraction = (divide1_lut[L_lookup_idx] - divide2_lut[L_lookup_idx]*diff / 1024.0) / 1024.0
    # -- 方法二：普通的查表 input Q8.5 output Q5.10
    # L_fraction = divide1_lut[L_lookup_idx] / 1024.0
    # -- 方法三：做除法然後 quantize (input Q9.10 output Q5.10)
    # L_fraction = enforce_q_precision(1 / L_safe, 10, 15)
    print(f"L_safe range from {L_fraction.min()} to {L_fraction.max()}")
    ratio = L_prime * L_fraction
    print(f"ratio range from {ratio.min()} to {ratio.max()}")

    R_final = R_orig * ratio
    G_final = G_orig * ratio
    B_final = B_orig * ratio
    LDR_final_linear = np.stack([R_final, G_final, B_final], axis=-1)
    
    # 7. 輸出編碼與量化 (檔案儲存專用)
    LDR_final_normalized = np.clip(LDR_final_linear, 0, 1)
    LDR_final_8bit_rgb = (LDR_final_normalized * 255).astype(np.uint8) # 把 RGB 結果存進 SRAM
    LDR_final_8bit_bgr = cv2.cvtColor(LDR_final_8bit_rgb, cv2.COLOR_RGB2BGR)

    return LDR_final_8bit_bgr

def local_tone_mapping_opencv(R, G, B, E, lut_data_l=None):
    R_orig = (R / 256.0) * np.power(2, E-128.0)
    G_orig = (G / 256.0) * np.power(2, E-128.0)
    B_orig = (B / 256.0) * np.power(2, E-128.0)

   # --- 1. 計算亮度 (Luminance) ---
    # 硬體公式: Sum = 256 權重 + 128 Bias (依據你的code)
    # 轉型為 uint16 計算，避免乘法溢位
    R = R.astype(np.uint16)
    G = G.astype(np.uint16)
    B = B.astype(np.uint16)
    E = E.astype(np.int16)

    Lm = 54 * R + 183 * G + 19 * B + 128
    
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

    # --- 3. 雙邊濾波與 Tone Mapping ---
    # I 是 Q14，需要轉回 float 進行 OpenCV 濾波
    # 注意: 如果 LUT output 是 Q14，這裡除以 16384.0 (2^14) 比較合理
    # 但你的 code 之前是除以 1024，請確認你的 LUT 數值縮放
    # 這裡假設你的 base 和 exp_log 都是 Q14
    I_float32 = I.astype(np.float32) / 16384.0

    B = cv2.bilateralFilter(I_float32, FILTER_D, SIGMA_R, SIGMA_S)

    # --- 4. 分解為細節層 D ---
    D = I_float32 - B

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
    
    TARGET_HEIGHT = 874  # 目標高度 (H)
    TARGET_WIDTH = 644   # 目標寬度 (W)
    
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

# REC709_R_INT = 54   # 近似 0.2126 * 256
# REC709_G_INT = 183  # 近似 0.7152 * 256
# REC709_B_INT = 18   # 近似 0.0722 * 256

# def rgbe_to_fixed_point_12bit_optimized(rgbe_matrix):
    
#     R_m = rgbe_matrix[..., 0].astype(np.uint16)
#     G_m = rgbe_matrix[..., 1].astype(np.uint16)
#     B_m = rgbe_matrix[..., 2].astype(np.uint16)

#     # 指數保持 8-bit 進行位元操作
#     E = rgbe_matrix[..., 3].astype(np.uint8)

#     # Lm_scaled = R_m*54 + G_m*183 + B_m*18
#     Lm_32bit = (REC709_R_INT * R_m) + (REC709_G_INT * G_m) + (REC709_B_INT * B_m)
#     Lm_8bit_mantissa = Lm_32bit / 512.0
#     E_4bits = ((E >> 4)).astype(np.uint16)
#     Lm_packed = Lm_8bit_mantissa.astype(np.uint16) << 4 
#     final_12bit_fixed = Lm_packed | E_4bits
    
#     return final_12bit_fixed, R_m, G_m, B_m, E

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
    HDR_FILE_PATH = "img/Desk.hdr" 
    LDR_OUTPUT_PATH = "img/Desk.png"
    LDR_OUTPUT_PATH1 = "img/Desk_s.png" 
    
    Luminance_FILE_PATH = "data/luminance.txt"
    Bmatrix_FILE_PATH = "data/B_matrix.txt"

    LUT_PATH = "LUT/LUT.xlsx"
    LUT_EXCEL_PATH = "LUT/log_calculation_int_2.xlsx" 
    Lm_LUT = "LUT/Lm_base_LUT.xlsx"

    try:
        divide_lut = load_and_prepare_lut(LUT_PATH, 'divide6Q6', 4096)
        divide1_lut = load_and_prepare_lut(LUT_PATH, 'divide0Q13', 8192)
        divide2_lut = load_and_prepare_lut(LUT_PATH, 'divide2_0Q13', 8192)
        power_lut = load_and_prepare_lut(LUT_PATH, 'power2', 4096)
        # 1. 讀取 LUT
        lut_x_l, lut_y_l = load_lut_from_excel(Lm_LUT, input_col="base 12 bit", output_col="1.base base value")
        if lut_x_l is None:
            raise ValueError("LUT 載入失敗，程式終止。")

        hdr_input = read_hdr_image(HDR_FILE_PATH)
        rgbe_matrix, W, H, R_m, G_m, B_m, E = read_hdr_rgbe(HDR_FILE_PATH)
        # Software Path
        final_ldr_8bit_bgr1 = local_tone_mapping_opencv(R_m, G_m, B_m, E,
                                                        lut_data_l=(lut_x_l, lut_y_l)
                                                        )
        save_ldr_file(final_ldr_8bit_bgr1, LDR_OUTPUT_PATH1)
        # Hardware Path
        final_ldr_8bit_bgr = local_tone_mapping_lut(Luminance_FILE_PATH, Bmatrix_FILE_PATH, R_m, G_m, B_m, E,
                                                   lut_data_l=(lut_x_l, lut_y_l))
        save_ldr_file(final_ldr_8bit_bgr, LDR_OUTPUT_PATH)
        os.remove(Bmatrix_FILE_PATH)  # 圖像處理完成後自動刪除 B_matrix 檔案
        
    except FileNotFoundError as e:
        print(f"錯誤: {e}\n請確認檔案路徑是否正確。")
    except Exception as e:
        print(f"發生其他錯誤: {e}")