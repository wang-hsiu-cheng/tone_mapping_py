import cv2
import os

def generate_filenames(pattern_num):
    input_filename = f"PAT_{pattern_num:03d}.png"
    output_filename = f"outpt_{pattern_num:03d}.dat"
    return input_filename, output_filename

def write_rgb_dat_with_debug(img_bgr, output_path, pattern_id):
    """
    將 OpenCV 讀取的 BGR 影像寫入 .dat 檔，並在處理 PAT_000 時印出除錯資訊。
    """
    H, W, C = img_bgr.shape
    
    # 判斷是否需要印出除錯資訊 (只印第一張圖 PAT_000)
    enable_debug = (pattern_id == 0)

    if enable_debug:
        print(f"\n{'='*20} [DEBUG: PAT_000 前 5 個 Pixel 檢查] {'='*20}")
        print(f"{'座標':<10} | {'OpenCV (BGR)':<15} | {'目標寫入 (RGB)':<15} | {'Hex 結果'}")
        print("-" * 70)

    with open(output_path, "w") as f:
        for y in range(H):
            for x in range(W):
                # OpenCV 讀進來是 BGR [Blue, Green, Red]
                b, g, r = img_bgr[y, x]
                
                # 我們要寫入的是 RGB [Red, Green, Blue]
                # 格式: RR GG BB
                hex_string = f"{r:02X}{g:02X}{b:02X}"
                
                # --- Debug 區塊: 只印第一行(y=0)的前5個點(x<5) ---
                if enable_debug and y == 0 and x < 5:
                    print(f"({x}, {y})     | [{b:>3}, {g:>3}, {r:>3}]   | [{r:>3}, {g:>3}, {b:>3}]   | {hex_string}")
                
                # 實際寫入檔案
                f.write(hex_string)
                f.write(f" // {r} {g} {b}\n")
    
    if enable_debug:
        print(f"{'='*66}\n")

def batch_convert_png_to_dat_debug(start_pat=0, end_pat=0):
    INPUT_DIR = "output_img"
    OUTPUT_DIR = "dat_file/outpt"
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"開始轉換 Pattern {start_pat} 到 {end_pat} ...")

    for i in range(start_pat, end_pat + 1):
        in_name, out_name = generate_filenames(i)
        input_path = os.path.join(INPUT_DIR, in_name)
        output_path = os.path.join(OUTPUT_DIR, out_name)

        if not os.path.exists(input_path):
            print(f"[Skip] {in_name} 不存在")
            continue

        img = cv2.imread(input_path)
        if img is None: continue

        # 呼叫帶有 Debug 功能的寫入函式
        try:
            write_rgb_dat_with_debug(img, output_path, pattern_id=i)
            print(f"[OK] {out_name} 儲存成功")
        except Exception as e:
            print(f"[Fail] {out_name} 失敗: {e}")

if __name__ == "__main__":
    # 執行轉換並檢查
    batch_convert_png_to_dat_debug(start_pat=0, end_pat=9)