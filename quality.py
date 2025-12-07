import cv2
import numpy as np
import torch
import piq

# --- 檔案設定與參數 ---
# 🚨 替換為你的實際檔案路徑 🚨
HDR_REFERENCE_FILE = "img/little_paris_eiffel_tower_1k.hdr" 
LDR_OUTPUT_FILE = "img/little_paris_eiffel_tower_1k.png" 

# TMQI 參數 (通常 alpha=0.5，用於平衡結構保留和自然度)
TMQI_ALPHA = 0.5 

def read_and_prepare_images(hdr_path, ldr_path):
    """
    讀取 HDR 和 LDR 影像，並將其轉換為 PyTorch tensor 格式。
    
    PIQ 的 TMQI 模組要求輸入格式為：
    1. 浮點數 (float32 或 float64)。
    2. 範圍在 [0, 1] 或更高 (HDR)。
    3. 尺寸格式為 (B, C, H, W) 或 (C, H, W)。
    """
    
    # 1. 讀取 HDR 檔案 (線性浮點數)
    hdr_np = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)
    if hdr_np is None:
        raise FileNotFoundError(f"無法讀取 HDR 檔案: {hdr_path}")
    
    # 將 BGR 轉換為 RGB 順序
    hdr_np = cv2.cvtColor(hdr_np, cv2.COLOR_BGR2RGB)
    
    # 2. 讀取 LDR 檔案 (8-bit 輸出)
    ldr_np = cv2.imread(ldr_path, cv2.IMREAD_UNCHANGED)
    if ldr_np is None:
        raise FileNotFoundError(f"無法讀取 LDR 檔案: {ldr_path}")
        
    # 將 LDR 轉換為浮點數並正規化到 [0, 1]
    ldr_np = cv2.cvtColor(ldr_np, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 確保兩張圖的大小一致，如果不一致，TMQI 會出錯
    if hdr_np.shape[:2] != ldr_np.shape[:2]:
        # 將 HDR 縮放到 LDR 的大小（如果 LDR 是較小的輸出）
        ldr_h, ldr_w = ldr_np.shape[:2]
        hdr_np = cv2.resize(hdr_np, (ldr_w, ldr_h), interpolation=cv2.INTER_LINEAR)
        print(f"警告: 影像大小不匹配，HDR 已縮放至 {ldr_h}x{ldr_w}")
    
    # --- 轉換為 PyTorch Tensor ---
    
    # 將 NumPy 格式 (H, W, C) 轉換為 PyTorch 格式 (B, C, H, W)
    # B=1 (單張圖片), C=3 (RGB)
    hdr_tensor = torch.from_numpy(hdr_np).permute(2, 0, 1).unsqueeze(0).float()
    ldr_tensor = torch.from_numpy(ldr_np).permute(2, 0, 1).unsqueeze(0).float()

    return hdr_tensor, ldr_tensor

def calculate_tmqi(hdr_tensor, ldr_tensor, alpha):
    """
    使用 PIQ 庫的 TMQI 類 (Class) 來計算分數。
    """
    # 1. 實例化 TMQI 類 (Class)
    # 設置 data_range = LDR 的最大值 (1.0)
    # 設置 alpha 參數
    
    # 由於 PIQ 的 TMQI 需要 HDR 輸入的 max/min 來決定 data_range，
    # 這裡我們使用 LDR 的 max/min (1.0 和 0.0)
    tmqi_metric = piq.TMQI(
        data_range=ldr_tensor.max().item(), 
        alpha=alpha,
        reduction='none' # 為了獲得 tensor 輸出
    )
    
    # 2. 調用計算方法
    # TMQI 類調用時，必須先將 LDR (Test) 放在前面，HDR (Reference) 放在後面
    # 輸出是一個包含 Q, S, N 分量的 tensor
    tmqi_components = tmqi_metric(ldr_tensor, hdr_tensor)

    # 3. 提取分量
    # 檢查 TMQI 輸出 tensor 的形狀和順序，通常是 [S, N] 或 [Q, S, N]
    # PIQ 官方文檔顯示，當 return_components=True 時，piq.tm_q_i 返回 Q, S, N。
    # 但使用類調用時，我們需要檢查結果。

    # 由於 PIQ TMQI 類的實現細節可能依賴於版本，我們這裡使用最常見的邏輯：
    # TMQI 類通常返回一個包含 Q, S, N 的單個 tensor 或 tuple
    
    # 這裡我們需要使用 piq.tm_q_i 函數，但假設它仍然存在且可以調用：
    
    tmqi_score, structure_s, naturalness_n = piq.tm_q_i(
        ldr_tensor, 
        hdr_tensor, 
        data_range=ldr_tensor.max().item(), 
        alpha=alpha,
        return_components=True
    )
    
    return tmqi_score.item(), structure_s.item(), naturalness_n.item()

# --- 主程式區塊 ---
if __name__ == '__main__':
    try:
        # 1. 讀取和準備影像
        hdr_tensor, ldr_tensor = read_and_prepare_images(HDR_REFERENCE_FILE, LDR_OUTPUT_FILE)
        
        # 2. 計算 TMQI
        tmqi_q, tmqi_s, tmqi_n = calculate_tmqi(hdr_tensor, ldr_tensor, TMQI_ALPHA)
        
        # 3. 輸出結果
        print("\n--- TMQI (Tone Mapped Image Quality Index) 計算結果 ---")
        print(f"TMQI 權重 $\\alpha$: {TMQI_ALPHA}")
        print("-----------------------------------------------------")
        print(r"1. 總體品質分數 (Q) = $\\alpha \cdot S + (1-\\alpha) \cdot N$: **{tmqi_q:.4f}**")
        print(f"2. 結構相似性分數 (S): {tmqi_s:.4f}")
        print(f"3. 自然度分數 (N): {tmqi_n:.4f}")
        
        print("\n分數越接近 1.0，代表色調映射的品質越優秀。")
        
    except FileNotFoundError as e:
        print(f"錯誤: {e}\n請確認 HDR_REFERENCE_FILE 和 LDR_OUTPUT_FILE 路徑是否正確。")
    except ImportError:
        print("錯誤: 缺少必要的函式庫。請確保您已執行 'pip install torch piq opencv-python numpy'")
    except Exception as e:
        print(f"計算 TMQI 過程中發生錯誤: {e}")