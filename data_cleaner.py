import pandas as pd
import re
import unicodedata
from config import Config

class DataCleaner:
    """
    負責對已整合的DataFrame進行深度文字內容清理
    """
    def __init__(self, config: Config):
        self.config = config
        print("DataCleaner initialized for deep text cleaning.")

    def _clean_text(self, text: str) -> str:
        """
        私有方法：深度清理單一字串 (最終決定版)
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""

        # ✨ 關鍵修正點：使用一個更強大的正規表示式，連帶移除URL的引導詞
        # (?:...) 是非捕獲分組，用來將多個模式組合在一起
        # \s* 代表可能有0或多個空白
        url_pattern = r'(?:網址是|連結為|網址:|連結:)?\s*(?:https?://\S+|www\.\S+)'
        text = re.sub(url_pattern, ' ', text)
        
        # 替換HTML標籤
        text = re.sub(r'<.*?>', ' ', text)
        
        # 使用NFC正規化
        text = unicodedata.normalize('NFC', text)
        
        # 定義允許的字元模式
        allowed_pattern = (
            r'\u4e00-\u9fa5'      # 中文字元
            r'a-zA-Z0-9'         # 英文字母和數字
            r'\s'                # 空白字元
            r'\'.,!?，。！？'     # 中英文標點
            r'\u00C0-\u017F'      # 拉丁擴展字母
        )
        # 將所有「不在」允許模式中的字元替換成空白
        text = re.sub(f'[^{allowed_pattern}]', ' ', text)
        
        # 將多個連續的空白合併為一個
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 轉換為小寫
        return text.lower()

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        公開方法：對整合後的DataFrame執行清理流程
        """
        print(f"\n🧹 開始深度文字清理，原始筆數: {len(df):,}")
        
        df_clean = df.copy()

        # 步驟 1: 確保關鍵文字欄位存在且為字串
        for col in ['artTitle', 'artContent']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).fillna('')
        
        # 步驟 2: 清理 'artTitle' 和 'artContent'
        print("   - 正在清理文章標題和內容...")
        df_clean['artTitle'] = df_clean['artTitle'].apply(self._clean_text)
        df_clean['artContent'] = df_clean['artContent'].apply(self._clean_text)
        
        # 步驟 3: 移除內容清理後為空的行
        original_count = len(df_clean)
        df_clean = df_clean[df_clean['artContent'].str.len() > self.config.MIN_CONTENT_LENGTH]
        print(f"   - 移除內容過短或空白的文章後，剩下 {len(df_clean):,} 筆 (移除 {original_count - len(df_clean)} 筆)")

        # 步驟 4: 基於清理後的內容去重
        original_count = len(df_clean)
        df_clean.drop_duplicates(subset=['artContent'], keep='first', inplace=True)
        print(f"   - 移除內容重複的文章後，剩下 {len(df_clean):,} 筆 (移除 {original_count - len(df_clean)} 筆)")

        print(f"✅ 深度文字清理完畢，最終筆數: {len(df_clean):,}")
        return df_clean.reset_index(drop=True)

# ================= 模組獨立測試區塊 =================
if __name__ == "__main__":
    print("===== 正在獨立測試 DataCleaner (簡化版) 模組 =====")

    # ✨ 關鍵修正點：先用純 Python list 來準備資料
    # 模擬一個來自DataLoader的整合後DataFrame
    art_titles = ['<p>萊雅好物分享</p>', '  另一個標題  ', '重複內容標題']
    art_contents = ['這是一篇關於L\'Oréal的分享，網址是http://loreal.com', '  這是  一些  內容  ', '這是一篇重複的內容，會被移除。']
    data_sources = ['PTT', 'DCARD', 'PTT']
    art_dates = ['2024-07-21', '2024-07-22', '2024-07-23']

    # 現在用 list.append() 來安全地增加一筆重複內容
    art_titles.append('重複內容標題2')
    art_contents.append('這是一篇重複的內容，會被移除。')
    data_sources.append('DCARD')
    art_dates.append('2024-07-24')

    # 最後，將準備好的 list 組合成字典，並建立 DataFrame
    mock_integrated_data = {
        'artTitle': art_titles,
        'artContent': art_contents,
        'dataSource': data_sources,
        'artDate': pd.to_datetime(art_dates) # 在這裡一次性轉換所有日期
    }

    mock_df = pd.DataFrame(mock_integrated_data)
    
    print("\n--- 測試前，來自DataLoader的整合後資料 ---")
    print(mock_df)
    
    # 後續程式碼不變
    config = Config()
    cleaner = DataCleaner(config)
    
    cleaned_df = cleaner.clean(mock_df)
    
    print("\n--- 測試後，深度清理過的資料 ---")
    print(cleaned_df)
    
    print("\n===== DataCleaner (簡化版) 模組獨立測試完畢 =====")
