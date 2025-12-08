import cv2
import numpy as np
import pandas as pd 
import os
import time

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
    
CONTRAST = 100.0      # 基礎層壓縮參數：目標對比度 (關鍵可調參數)
EPSILON = 1e-6      # 防止 log(0) 錯誤

def local_tone_mapping_lut(hdr_image_linear, Luminance_FILE_PATH, Bmatrix_FILE_PATH):
    """執行使用客製化雙邊濾波器 (LUT 加速) 的 LTM 流程。"""
    R_orig, G_orig, B_orig = [hdr_image_linear[..., i] for i in range(3)]

    # 1. 計算亮度 (Luminance)
    L = 0.2126 * R_orig + 0.7152 * G_orig + 0.0722 * B_orig

    # 2. 對數轉換
    I = np.log10(L + EPSILON)
        
    # 3. 儲存 I 矩陣 (對數亮度)
    write_matrix_to_text_file(I, Luminance_FILE_PATH)
    print(f"\n==================================================================")
    print(f"等待 C++ 處理：請執行 C++ 雙邊濾波器，將結果寫入 {Bmatrix_FILE_PATH}")
    print(f"==================================================================")

    # 4. 等待 B_matrix.txt 檔案存在
    print(f"檢查檔案 {Bmatrix_FILE_PATH}...")
    while not os.path.exists(Bmatrix_FILE_PATH):
        print("  檔案不存在，等待 1 秒...")
        time.sleep(1)
    
    # 5. 讀取 B 矩陣 (基礎層)
    B = read_matrix_from_text_file(Bmatrix_FILE_PATH)
    if B is None:
        raise RuntimeError("無法從 B_matrix.txt 讀取基礎層矩陣，終止 LTM 流程。")
    # 檢查 B 的尺寸是否與 I 匹配
    if B.shape != I.shape:
        raise ValueError(f"讀取的 B 矩陣形狀 {B.shape} 與 I 矩陣形狀 {I.shape} 不匹配。")

    # 4. 分解為細節層 D
    D = I - B

    # 5. 基礎層壓縮
    max_B = B.max()
    min_B = B.min()
    B_range = max_B - min_B
    k = np.log10(CONTRAST) / (B_range + EPSILON) if B_range >= EPSILON else 0.0
    B_compressed = B * k

    # 6. 重建與色彩還原 (Reconstruction)
    I_prime = B_compressed + D
    L_prime = 10**(I_prime)
    L_safe = np.where(L > EPSILON, L, EPSILON)
    ratio = L_prime / L_safe

    R_final = R_orig * ratio
    G_final = G_orig * ratio
    B_final = B_orig * ratio
    LDR_final_linear = np.stack([R_final, G_final, B_final], axis=-1)
    
    # 7. 輸出編碼與量化 (檔案儲存專用)
    # white_point = np.percentile(LDR_final_linear, 99.9) 
    LDR_final_normalized = np.clip(LDR_final_linear / 1, 0, 1)
    LDR_final_8bit_rgb = (LDR_final_normalized * 255).astype(np.uint8)
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
    
    Luminance_FILE_PATH = "data/luminance.txt"
    Bmatrix_FILE_PATH = "data/B_matrix.txt"

    try:
        hdr_input = read_hdr_image(HDR_FILE_PATH)
        final_ldr_8bit_bgr = local_tone_mapping_lut(hdr_input, Luminance_FILE_PATH, Bmatrix_FILE_PATH)
        save_ldr_file(final_ldr_8bit_bgr, LDR_OUTPUT_PATH)
        
    except FileNotFoundError as e:
        print(f"錯誤: {e}\n請確認檔案路徑是否正確。")
    except Exception as e:
        print(f"發生其他錯誤: {e}")