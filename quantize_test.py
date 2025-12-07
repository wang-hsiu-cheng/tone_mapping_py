import cv2
import numpy as np

# --- I. 輔助函數與定點數配置 ---

# 設定小數精度位數
Q_FRACT_BITS = 10 

def fixed_to_float(fixed_value, fract_bits):
    """將定點數轉換回浮點數。"""
    return fixed_value / (1 << fract_bits)

MAX_FIXED_INTEGER_VALUE = (1 << (10 + 2)) - 1 # 4095

def float_to_fixed_sim(f_value, fract_bits):
    """
    將浮點數轉換為定點數整數 (Qx.fract_bits)，並確保輸出不超過 4095。
    """
    scale_factor = 1 << fract_bits
    
    # 1. 進行縮放、四捨五入，並轉換為 32 位元整數
    fixed_value = np.round(f_value * scale_factor).astype(np.int32)
    
    # 2. 🌟 實施總位寬限制 (Max value <= 4095)
    # 這一步模擬了硬體暫存器的飽和邏輯，防止溢位。
    fixed_value = np.clip(fixed_value, 0, MAX_FIXED_INTEGER_VALUE)
    
    return fixed_value

def read_hdr_image(file_path):
    """使用 OpenCV 讀取標準 HDR 檔案 (.hdr 或 .exr)。"""
    hdr_bgr = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    
    if hdr_bgr is None:
        raise FileNotFoundError(f"無法讀取檔案: {file_path}，請檢查檔案路徑或格式是否正確。")
    
    # 🌟 修正步驟 1：在進行顏色轉換前，將輸入類型強制轉換為 float32 (CV_32F)
    # 這確保了 OpenCV 能夠找到對應的 SIMD 轉換路徑。
    if hdr_bgr.dtype == np.float64:
        hdr_bgr = hdr_bgr.astype(np.float32)
        
    hdr_rgb_linear = cv2.cvtColor(hdr_bgr, cv2.COLOR_BGR2RGB)
    
    print(f"檔案讀取成功。影像解析度: {hdr_rgb_linear.shape[1]}x{hdr_rgb_linear.shape[0]}")
    return hdr_rgb_linear

# --- II. 核心轉換函數 ---

def quantize_and_save_hdr(input_hdr_path, output_hdr_path, fract_bits):
    """
    讀取 HDR 檔案，將浮點數轉換為 Qx.10 定點數，再轉回浮點數並儲存。
    """
    print(f"--- 開始處理 HDR 影像 ({input_hdr_path}) ---")
    print(f"定點數精度: {fract_bits} 位小數 (Qx.{fract_bits})")
    
    # 1. 讀取 HDR 影像 (RGB, 線性浮點數)
    hdr_rgb_linear = read_hdr_image(input_hdr_path)
    
    # ----------------------------------------------------
    # 2. 核心量化步驟：浮點數 -> Qx.10 定點數 (模擬硬體量化)
    # ----------------------------------------------------
    print(f"正在執行量化 (float -> Qx.{fract_bits})...")
    
    # 對整個 NumPy 陣列進行批量操作
    hdr_rgb_fixed = float_to_fixed_sim(hdr_rgb_linear, fract_bits)
    
    print(f"量化完成。數據類型：{hdr_rgb_fixed.dtype}")
    print(f"原始最大值: {hdr_rgb_linear.max():.4f}, 定點數最大整數值: {hdr_rgb_fixed.max()}")
    
    # ----------------------------------------------------
    # 3. 反向量化步驟：Qx.10 定點數 -> 浮點數 (模擬轉換回 DAC/顯示訊號)
    # ----------------------------------------------------
    
    hdr_rgb_quantized_linear = fixed_to_float(hdr_rgb_fixed, fract_bits)
    
    print("反向量化完成 (Qx.10 -> float)。")
    print(f"量化後最大值: {hdr_rgb_quantized_linear.max():.4f}")
    
    # 4. 儲存檔案
    
    # 將 RGB 轉回 BGR (OpenCV imwrite 默認使用 BGR)
    hdr_bgr_output = cv2.cvtColor(hdr_rgb_quantized_linear.astype(np.float32), cv2.COLOR_RGB2BGR)
    
    # 儲存為 HDR 格式 (使用浮點數數據類型，例如 .exr 或 .hdr)
    cv2.imwrite(output_hdr_path, hdr_bgr_output)
    
    print(f"--- 成功儲存量化後的 HDR 檔案至: {output_hdr_path} ---")

# --- III. 程式執行 ---

if __name__ == '__main__':
    # 🚨 請將這裡的路徑替換為你的實際檔案路徑 🚨
    INPUT_HDR_PATH = "img/little_paris_eiffel_tower_1k.hdr"  # 範例輸入檔案
    OUTPUT_HDR_PATH = "img/output_quantized_Qx10.hdr" # 輸出檔案
    
    try:
        quantize_and_save_hdr(INPUT_HDR_PATH, OUTPUT_HDR_PATH, Q_FRACT_BITS)
        
    except FileNotFoundError as e:
        print(f"錯誤: {e}\n請確認輸入檔案路徑是否正確。")
    except Exception as e:
        print(f"發生錯誤: {e}")