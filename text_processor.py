"""
文字處理器模組
Text processor module for L'Oreal project
"""

import pandas as pd
import jieba
import jieba.analyse
import numpy as np
import re
from typing import List, Dict, Set, Tuple
from collections import Counter
from config import Config

class TextProcessor:
    """
    負責將清理後的文字進行結構化處理 - 根據實際需求強化
    """
    def __init__(self, config: Config):
        self.config = config
        self._initialize_resources()
        self._prepare_synonym_dict()
        print("TextProcessor initialized with enhanced processing capabilities.")

    def _initialize_resources(self):
        """
        初始化Jieba和載入所有詞典資源
        """
        # 設置jieba日誌等級，避免過多訊息
        jieba.setLogLevel(60)
        
        # 載入自訂詞典
        try:
            jieba.load_userdict(self.config.USER_DICT_PATH)
            print(f"✅ 載入自訂詞典: {self.config.USER_DICT_PATH}")
        except FileNotFoundError:
            print(f"⚠️ 自訂詞典未找到，使用預設詞典")

        # 載入停用詞集合
        self.stop_words = set(self.config.ENHANCED_STOPWORDS)
        try:
            with open(self.config.STOP_WORDS_PATH, 'r', encoding='utf-8') as f:
                file_stopwords = {line.strip() for line in f if line.strip()}
                self.stop_words.update(file_stopwords)
        except FileNotFoundError:
            print(f"⚠️ 停用詞表檔案未找到，僅使用Config內建停用詞")
        
        print(f"📚 載入停用詞總數: {len(self.stop_words)} 個")

    def _prepare_synonym_dict(self):
        """
        準備同義詞詞典，將多種說法標準化為一個詞
        """
        self.synonym_map = {}
        for standard_word, synonyms in self.config.SYNONYM_GROUPS.items():
            for synonym in synonyms:
                self.synonym_map[synonym.lower()] = standard_word
        print(f"🔄 載入同義詞對: {len(self.synonym_map)} 組")

    def _normalize_synonyms(self, tokens: List[str]) -> List[str]:
        """
        將分詞結果中的同義詞進行標準化
        """
        return [self.synonym_map.get(token.lower(), token) for token in tokens]

    def _tokenize_and_filter(self, text: str) -> List[str]:
        """
        進階分詞與過濾：整合停用詞、長度、純數字等過濾規則
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        # 1. 精確模式分詞
        tokens = jieba.lcut(text, cut_all=False)
        
        # 2. 多層過濾
        filtered_tokens = [
            token.strip() for token in tokens 
            if len(token.strip()) >= self.config.MIN_TOKEN_LENGTH and 
               token.strip().lower() not in self.stop_words and
               not token.strip().isnumeric()
        ]
        
        # 3. 同義詞標準化
        normalized_tokens = self._normalize_synonyms(filtered_tokens)
        
        return normalized_tokens

    def _extract_keywords(self, text: str) -> List[str]:
        """
        增強的關鍵字提取，使用TF-IDF
        """
        if not text:
            return []
        
        # 提取關鍵字時，暫時使用內建停用詞以獲得更廣泛的結果
        keywords = jieba.analyse.extract_tags(
            text, 
            topK=self.config.KEYWORDS_TOP_K,
            withWeight=False
        )
        return keywords

    def process_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        公開方法：對整個DataFrame進行強化文字處理
        """
        print(f"\n📝 開始強化文字處理，總筆數: {len(df):,}")
        df_processed = df.copy()

        # 步驟1: 進階分詞與過濾 (這個結果用於主題模型和情感分析)
        df_processed['tokens'] = df_processed['artContent'].apply(self._tokenize_and_filter)
        print("步驟1: 進階分詞 (Tokenization) 與過濾完成")

        # 步驟2: 關鍵字提取 (這個結果主要用於視覺化和快速概覽)
        df_processed['keywords'] = df_processed['artContent'].apply(self._extract_keywords)
        print("步驟2: 關鍵字提取 (Keyword Extraction) 完成")

        print("✅ 強化文字處理完畢")
        return df_processed

# ================= 模組獨立測試區塊 =================
if __name__ == "__main__":
    print("===== 正在獨立測試 TextProcessor (強化版) 模組 =====")

    # 使用和您一樣的測試資料
    mock_cleaned_data = {
        'artContent': [
            '我超愛巴黎萊雅的青春密碼精華液效果驚人',
            'loreal的新款紫熨斗眼霜撫平細紋很有感',
            '這款產品很不錯值得推薦給大家',
            '效果普普通通沒有廣告說的那麼神奇'
        ]
    }
    mock_df = pd.DataFrame(mock_cleaned_data)
    
    print("\n--- 測試前，來自DataCleaner的乾淨資料 ---")
    print(mock_df)
    
    config = Config()
    processor = TextProcessor(config)
    
    processed_df = processor.process_text_data(mock_df)

    print("\n--- 測試後，處理過的資料 ---")
    print("\n--- 測試後，處理過的資料 (詳細檢視) ---")
    for index, row in processed_df.iterrows():
        print(f"\n[文章 {index+1}]")
        print(f"  原文: {row['artContent']}")
        print(f"  Keywords (快速概覽): {row['keywords']}")
        print(f"  Tokens (精準分析): {row['tokens']}")
    
    print("\n===== TextProcessor (強化版) 模組獨立測試完畢 =====")