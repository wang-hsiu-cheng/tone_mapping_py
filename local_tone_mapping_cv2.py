import cv2
import numpy as np
# 移除 matplotlib 導入，因為我們不再需要顯示功能

# --- 參數設定 ---
# 雙邊濾波器的參數 (請根據實際影像調整)
FILTER_D = 5        # 濾波器直徑 (d)
SIGMA_R = 1.0       # 範圍標準差 (sigmaColor/sigmaRange): 邊緣敏感度閾值
SIGMA_S = 1.5       # 空間標準差 (sigmaSpace): 模糊半徑
CONTRAST = 100.0      # 基礎層壓縮參數：目標對比度 (關鍵可調參數)
EPSILON = 1e-6      # 防止 log(0) 錯誤

def read_hdr_image(file_path):
    """
    使用 OpenCV 讀取標準 HDR 檔案 (.hdr 或 .exr)。
    """
    # 設置 cv2.IMREAD_UNCHANGED 確保讀取原始浮點數 HDR 值
    hdr_bgr = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    
    if hdr_bgr is None:
        raise FileNotFoundError(f"無法讀取檔案: {file_path}，請檢查檔案路徑或格式是否正確。")
    
    # OpenCV 默認讀取為 BGR 順序，需要轉換為 RGB 順序
    hdr_rgb_linear = cv2.cvtColor(hdr_bgr, cv2.COLOR_BGR2RGB)
    
    print(f"檔案讀取成功。影像解析度: {hdr_rgb_linear.shape[1]}x{hdr_rgb_linear.shape[0]}")
    return hdr_rgb_linear

def local_tone_mapping_opencv(hdr_image_linear, d, sigma_s, sigma_r, contrast, epsilon):
    """
    執行基於 OpenCV 雙邊濾波器的局部色調映射 (LTM) 流程。
    
    Returns:
        np.ndarray: 經過 LTM、Gamma 編碼和量化後的 8-bit BGR 影像 (準備寫入檔案)。
    """
    R_orig, G_orig, B_orig = [hdr_image_linear[..., i] for i in range(3)]

    # --- 1. 計算亮度 (Luminance) ---
    L = 0.2126 * R_orig + 0.7152 * G_orig + 0.0722 * B_orig

    # --- 2. 對數轉換 ---
    I = np.log10(L + epsilon)

    # --- 3. 雙邊濾波 (提取基礎層 B) ---
    I_float32 = I.astype(np.float32)
    B = cv2.bilateralFilter(I_float32, d, sigma_r, sigma_s)

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

def save_ldr_file(image_data, output_path):
    """
    使用 OpenCV 將 8-bit 影像數據儲存為 LDR 檔案。
    """
    success = cv2.imwrite(output_path, image_data)
    if success:
        print(f"成功儲存 LDR 檔案至: {output_path}")
    else:
        print(f"檔案儲存失敗: {output_path}")


# --- 主程式區塊：請修改此處的檔案路徑 ---
if __name__ == '__main__':
    # 範例輸入 HDR 檔案路徑
    HDR_FILE_PATH = "img/Desk.hdr" 
    # 輸出 LDR 檔案路徑 (請確保副檔名為 .jpg, .png 或 .tif)
    LDR_OUTPUT_PATH = "img/Desk.png" 
    # # 範例輸入 HDR 檔案路徑
    # HDR_FILE_PATH = "img/hay_bales_4k.hdr" 
    # # 輸出 LDR 檔案路徑 (請確保副檔名為 .jpg, .png 或 .tif)
    # LDR_OUTPUT_PATH = "img/hay_bales_4k.png" 
    
    try:
        # 1. 讀取 HDR 檔案
        hdr_input = read_hdr_image(HDR_FILE_PATH)
        
        print("\n--- 開始局部色調映射 (LTM) 流程 ---")
        
        # 2. 執行色調映射和最終編碼
        final_ldr_8bit_bgr = local_tone_mapping_opencv(
            hdr_input, 
            FILTER_D, 
            SIGMA_S, 
            SIGMA_R, 
            CONTRAST, 
            EPSILON
        )
        
        # 3. 儲存檔案
        save_ldr_file(final_ldr_8bit_bgr, LDR_OUTPUT_PATH)
        
    except FileNotFoundError as e:
        print(f"錯誤: {e}\n請確認您已將 HDR_FILE_PATH 替換為有效的 HDR 檔案路徑。")
    except Exception as e:
        print(f"發生其他錯誤: {e}")