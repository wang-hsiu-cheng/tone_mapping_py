import cv2
import numpy as np
import pandas as pd  # 新增: 用於讀取 Excel

# --- 參數設定 ---
FILTER_D = 5
SIGMA_R = 1.0
SIGMA_S = 1.5
CONTRAST = 100.0
EPSILON = 1e-6
OUTPUT_GAMMA = 1

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

def read_hdr_rgbe(path):
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode(errors="ignore")
            if line.strip()=="":
                break

        line=f.readline().decode().strip().split()
        H=int(line[1])
        W=int(line[3])
        print(f"Width: {W}, Height: {H}")

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
                    else:         # literal
                        raw=f.read(val)
                        scan[x:x+val,c]=list(raw)
                        x+=val
            img[y]=scan
    return img,W,H

def local_tone_mapping_opencv(hdr_image_linear, R_raw, G_raw, B_raw, E_orig, d, sigma_s, sigma_r, contrast, epsilon, output_gamma, lut_data_l=None, lut_data_e=None):
    """
    修改版: 接收 lut_data 參數 (lut_x, lut_y)
    """
    R_orig, G_orig, B_orig = [hdr_image_linear[..., i] for i in range(3)]
    R_u16 = R_raw.astype(np.uint16)
    G_u16 = G_raw.astype(np.uint16)
    B_u16 = B_raw.astype(np.uint16)
    E_u8 = E_orig.astype(np.uint8)

    # print(R_u16)
    # print(G_u16)
    # print(B_u16)
    # print(E_u8)

    # --- 1. 計算亮度 (Luminance) ---
    # L = 0.2126 * R_orig + 0.7152 * G_orig + 0.0722 * B_orig
    Lm = 54 * R_u16 + 183 * G_u16 + 19 * B_u16 + 128
    E_float = E_u8.astype(np.float32)
    power_of_two = np.exp2(E_float - 144)
    L = Lm * power_of_two
    print(Lm)

    # --- 2. 對數轉換 (修改處: 使用 LUT) ---
    # 原始: I = np.log10(L + epsilon)
    
    if lut_data_l is not None:
        lut_x_l, lut_y_l = lut_data_l
        
        # 模擬硬體查表行為
        # 輸入: L + epsilon (確保不為 0，或根據你的 LUT 設計決定是否需要 epsilon)
        # np.interp 會執行線性插值。如果輸入超出 LUT 範圍，會自動 Clamp 到最大/最小值。
        log_Lm = np.interp(Lm, lut_x_l, lut_y_l)
    else:
        print("警告: 未提供 LUT，使用標準 log10 計算")
        log_Lm = np.log10(Lm + epsilon)

    if lut_data_e is not None:
        lut_x_e, lut_y_e = lut_data_e
        
        # 模擬硬體查表行為
        # 輸入: L + epsilon (確保不為 0，或根據你的 LUT 設計決定是否需要 epsilon)
        # np.interp 會執行線性插值。如果輸入超出 LUT 範圍，會自動 Clamp 到最大/最小值。
        E_log2 = np.interp(E_u8, lut_x_e, lut_y_e)
    else:
        print("警告: 未提供 LUT，使用標準 log10 計算")
        E_log2 = np.log10(Lm + epsilon)

    # --- 3. 雙邊濾波 (以下流程不變) ---
    I_fixed = log_Lm + E_log2
    I = I_fixed / 1024.0
    # print(log_Lm)
    # print(E_log2)
    # print(I_fixed)
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

    # --- 6. 重建與色彩還原 ---
    # 注意: 如果你的 HW 設計連這裡的 10^x 也要查表，這裡也需要用另一個 LUT 替換
    I_prime = B_compressed + D
    L_prime = 10**(I_prime) 
    
    L_safe = np.where(L > epsilon, L, epsilon)
    ratio = L_prime / L_safe

    R_final = R_orig * ratio
    G_final = G_orig * ratio
    B_final = B_orig * ratio
    
    LDR_final_linear = np.stack([R_final, G_final, B_final], axis=-1)
    
    # --- 7. 輸出編碼與量化 ---
    white_point = np.percentile(LDR_final_linear, 99.9) 
    LDR_final_normalized = np.clip(LDR_final_linear / white_point, 0, 1)
    LDR_final_gamma = LDR_final_normalized**(1/output_gamma)
    LDR_final_8bit_rgb = (LDR_final_gamma * 255).astype(np.uint8)
    LDR_final_8bit_bgr = cv2.cvtColor(LDR_final_8bit_rgb, cv2.COLOR_RGB2BGR)

    return LDR_final_8bit_bgr

def save_ldr_file(image_data, output_path):
    cv2.imwrite(output_path, image_data)
    print(f"儲存至: {output_path}")

if __name__ == '__main__':
    HDR_FILE_PATH = "img/little_paris_eiffel_tower_4k.hdr" 
    LDR_OUTPUT_PATH = "img/little_paris_eiffel_tower_4k_lut.png" 
    Lm_LUT = "Lm_log_LUT.xlsx"
    E_LUT = "E_log_LUT.xlsx"
    
    # 1. 讀取 LUT
    lut_x_l, lut_y_l = load_lut_from_excel(Lm_LUT, input_col="Lm(int16)", output_col="log(Lm)(q4.10)")
    lut_x_e, lut_y_e = load_lut_from_excel(E_LUT, input_col="E(int8)", output_col="(E-144)log2(q4.10)")
    
    if lut_x_l is None:
        raise ValueError("LUT 載入失敗，程式終止。")
    if lut_x_e is None:
        raise ValueError("LUT 載入失敗，程式終止。")

    # 2. 讀取 HDR
    hdr_input = read_hdr_image(HDR_FILE_PATH)
    # R_ch, G_ch, B_ch = cv2.split(hdr_input)
    hdr,W,H = read_hdr_rgbe(HDR_FILE_PATH)
    R_raw = hdr[:, :, 0]  # 第 0 層: 紅色尾數 (uint8)
    G_raw = hdr[:, :, 1]  # 第 1 層: 綠色尾數 (uint8)
    B_raw = hdr[:, :, 2]  # 第 2 層: 藍色尾數 (uint8)
    E_raw = hdr[:, :, 3]  # 第 3 層: 共同指數 (uint8)
    
    print("\n--- 開始局部色調映射 (使用 LUT) ---")
    
    # 3. 執行處理
    final_ldr_8bit_bgr = local_tone_mapping_opencv(
        hdr_input, R_raw, G_raw, B_raw, E_raw, FILTER_D, SIGMA_S, SIGMA_R, CONTRAST, EPSILON, OUTPUT_GAMMA,
        lut_data_l=(lut_x_l, lut_y_l), lut_data_e=(lut_x_e, lut_y_e) # 傳入 LUT 數據
    )
    
    # 4. 儲存
    save_ldr_file(final_ldr_8bit_bgr, LDR_OUTPUT_PATH)
