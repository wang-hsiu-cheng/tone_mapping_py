import pandas as pd
import os

def extract_columns_to_txt(excel_file_path, sheet_name, column_names, output_txt_path):
    """
    從指定的 Excel 檔案中讀取兩欄數據，並將它們儲存到一個純文字檔案中。

    Args:
        excel_file_path (str): 輸入的 Excel 檔案路徑 (.xlsx 或 .xls)。
        sheet_name (str): 要讀取的工作表名稱。
        column_names (list): 包含要讀取之兩欄名稱的列表 (e.g., ['Column A', 'Column B'])。
        output_txt_path (str): 輸出的純文字檔案路徑 (.txt)。
    """
    
    # 檢查輸入欄位數量是否正確
    if len(column_names) != 2:
        print("錯誤: 必須指定且只指定兩個欄位的名稱。")
        return

    print(f"--- 開始處理檔案: {excel_file_path} ---")

    try:
        # 1. 讀取 Excel 檔案
        # usecols 參數確保只讀取指定的欄位，提高效率
        df = pd.read_excel(
            excel_file_path, 
            sheet_name=sheet_name, 
            usecols=column_names
        )
        
        print(f"  成功讀取工作表 '{sheet_name}'，共 {len(df)} 筆數據。")

        # 2. 將選定的兩欄數據轉換為 NumPy 陣列 (可選，但有助於格式化)
        # 這裡直接使用 DataFrame，並確保兩欄數據被選中
        data_to_save = df[column_names]
        
        # 3. 儲存到純文字檔案
        # header=False: 不寫入欄位名稱
        # index=False: 不寫入 Pandas 索引
        # sep='\t' 或 sep=' ': 選擇分隔符號。這裡使用空格 ' '
        data_to_save.to_csv(
            output_txt_path, 
            sep=' ', 
            header=False, 
            index=False,
            float_format='%.6f' # 確保浮點數輸出有合理的精度
        )
        
        print(f"✅ 數據已成功儲存至: {output_txt_path}")
        
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {excel_file_path}")
    except KeyError as e:
        print(f"錯誤: 找不到指定的欄位 {e} 或工作表 '{sheet_name}'。請檢查名稱是否正確。")
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")


# --- 範例使用區塊 ---
if __name__ == '__main__':
    # 💡 請將這些路徑替換為你的實際檔案路徑和名稱 💡
    
    # 輸入檔案和參數
    INPUT_EXCEL = "LUT/LUT.xlsx" 
    SHEET_NAME = "divide" # 假設工作表名稱
    COLUMNS_TO_EXTRACT = ["input(6Q6)", "output(6Q12)"] # 假設要提取的兩個欄位名稱
    
    # 輸出檔案
    OUTPUT_TEXT_FILE = "LUT/divide.txt"
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(OUTPUT_TEXT_FILE), exist_ok=True)

    # 執行函數
    extract_columns_to_txt(
        excel_file_path=INPUT_EXCEL,
        sheet_name=SHEET_NAME,
        column_names=COLUMNS_TO_EXTRACT,
        output_txt_path=OUTPUT_TEXT_FILE
    )