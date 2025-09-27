"""
情感分析模組
Sentiment analyzer module for L'Oreal project
"""

import pandas as pd
from snownlp import SnowNLP
from typing import List, Set
from config import Config

class SentimentAnalyzer:
    """
    負責對文本進行情感分析
    採用混合策略：結合通用模型(SnowNLP)和客製化情感詞典
    """
    def __init__(self, config: Config):
        self.config = config
        self.positive_words: Set[str] = self._load_word_set(config.POSITIVE_DICT_PATH)
        self.negative_words: Set[str] = self._load_word_set(config.NEGATIVE_DICT_PATH)
        print("SentimentAnalyzer initialized.")
        print(f"Loaded {len(self.positive_words)} positive words and {len(self.negative_words)} negative words.")

    def _load_word_set(self, path: str) -> Set[str]:
        """私有方法：從指定路徑載入詞典，並存為集合以加速查找"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return {line.strip() for line in f if line.strip()}
        except FileNotFoundError:
            print(f"Warning: Sentiment dictionary not found at {path}. An empty set will be used.")
            return set()

    def _analyze_with_snownlp(self, text: str) -> float:
        """私有方法：使用SnowNLP進行通用情感分析"""
        try:
            return SnowNLP(text).sentiments
        except Exception:
            return 0.5 # 如果出錯，返回中性值

    def _analyze_with_dict(self, tokens: List[str]) -> float:
        """私有方法：使用客製化詞典進行情感分析"""
        if not tokens:
            return 0.5
        
        pos_count = sum(1 for token in tokens if token in self.positive_words)
        neg_count = sum(1 for token in tokens if token in self.negative_words)
        
        # 正規化分數到 0-1 之間
        if pos_count + neg_count == 0:
            return 0.5 # 中性
        
        score = (pos_count - neg_count) / (pos_count + neg_count)
        # 將分數從 [-1, 1] 映射到 [0, 1]
        return (score + 1) / 2

    def _classify_sentiment(self, score: float) -> str:
        """私有方法：根據分數閾值將情感分類為標籤"""
        if score > self.config.SENTIMENT_POSITIVE_THRESHOLD:
            return "正面"
        elif score < self.config.SENTIMENT_NEGATIVE_THRESHOLD:
            return "負面"
        else:
            return "中性"

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        公開方法：對整個DataFrame進行情感分析
        """
        print(f"\n--- 開始情感分析，總筆數: {len(df):,} ---")
        df_sentiment = df.copy()

        # ✨ 關鍵修正點: 確保使用 'artContent' 欄位
        if 'artContent' not in df_sentiment.columns:
            raise ValueError("輸入的DataFrame中缺少 'artContent' 欄位")

        # 步驟1: SnowNLP通用情感分數
        df_sentiment['sentiment_snownlp'] = df_sentiment['artContent'].apply(self._analyze_with_snownlp)
        print("步驟1: SnowNLP通用情感分析完成。")

        # 步驟2: 客製化詞典情感分數
        df_sentiment['sentiment_dict'] = df_sentiment['tokens'].apply(self._analyze_with_dict)
        print("步驟2: 客製化詞典情感分析完成。")

        # 步驟3: 結合兩種分數
        df_sentiment['sentiment_score'] = (df_sentiment['sentiment_snownlp'] + df_sentiment['sentiment_dict']) / 2
        print("步驟3: 混合情感分數計算完成。")
        
        # 步驟4: 根據最終分數產生情感標籤
        df_sentiment['sentiment_label'] = df_sentiment['sentiment_score'].apply(self._classify_sentiment)
        print("步驟4: 情感標籤分類完成。")

        print("--- 情感分析完畢 ---")
        return df_sentiment

# ================= 模組獨立測試區塊 =================
if __name__ == "__main__":
    print("===== 正在獨立測試 SentimentAnalyzer 模組 =====")

    # 模擬來自 TextProcessor 的輸出 DataFrame
    mock_processed_data = {
        'content': [
            '巴黎萊雅這款精華液保濕效果超好，吸收快不黏膩',
            '新款紫熨斗眼霜對撫平細紋很有感，質地清爽',
            '效果普普通通，沒有廣告說的那麼神奇，有點失望',
            '用了會長痘痘，質地太油膩了，不會再回購'
        ],
        'tokens': [
            ['巴黎萊雅', '這款', '精華液', '保濕', '效果', '超好', '吸收', '不黏膩'],
            ['新款', '紫熨斗', '眼霜', '撫平', '細紋', '有感', '質地', '清爽'],
            ['效果', '普普通通', '沒有', '廣告', '神奇', '有點', '失望'],
            ['用', '了', '會', '長', '痘痘', '質地', '太', '油膩', '了', '不會', '再', '回購']
        ]
    }
    mock_df = pd.DataFrame(mock_processed_data)
    
    print("\n--- 測試前，來自TextProcessor的處理後資料 ---")
    print(mock_df[['content', 'tokens']])
    
    # 建立 Config 和 SentimentAnalyzer 物件
    config = Config()
    analyzer = SentimentAnalyzer(config)
    
    # 執行情感分析
    sentiment_df = analyzer.analyze(mock_df)

    # 驗證結果
    print("\n--- 測試後，帶有情感分析的資料 ---")
    display_cols = ['content', 'sentiment_snownlp', 'sentiment_dict', 'sentiment_score', 'sentiment_label']
    print(sentiment_df[display_cols])
    
    print("\n===== SentimentAnalyzer 模組獨立測試完畢 =====")
