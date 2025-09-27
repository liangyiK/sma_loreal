import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from collections import Counter
from config import Config

class Visualizer:
    """
    負責將分析後的數據轉換為各種視覺化圖表
    """
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = config.OUTPUT_PATH
        # 確保輸出目錄存在
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_matplotlib_font()
        print(f"Visualizer initialized. All charts will be saved to '{self.output_dir}'")

    def _setup_matplotlib_font(self):
        """
        私有方法：設定Matplotlib以支援中文字體，防止亂碼
        """
        try:
            # 設定全域字體
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 預設使用微軟正黑體
            plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題
            print(f"Matplotlib font set to 'Microsoft JhengHei'.")
        except Exception:
            print("Warning: 'Microsoft JhengHei' font not found. Please set a valid font in config.py.")
            # 如果找不到，可以嘗試從config讀取備用字體
            if os.path.exists(self.config.CHINESE_FONT_PATH):
                plt.rcParams['font.sans-serif'] = [self.config.CHINESE_FONT_PATH]
                print(f"Using fallback font from config: {self.config.CHINESE_FONT_PATH}")

    def _save_plot(self, fig, filename: str):
        """私有方法：儲存圖表到指定路徑"""
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig) # 關閉圖表以釋放記憶體
        print(f"Chart saved: {path}")

    def plot_word_cloud(self, df: pd.DataFrame):
        """公開方法：根據 'keywords' 欄位產生關鍵字詞雲"""
        print("\n--- Generating Word Cloud ---")
        all_keywords = [keyword for sublist in df['keywords'] for keyword in sublist]
        keyword_counts = Counter(all_keywords)

        wc = WordCloud(
            font_path=self.config.CHINESE_FONT_PATH,
            width=800,
            height=400,
            background_color='white',
            max_words=100
        ).generate_from_frequencies(keyword_counts)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('熱門關鍵字詞雲', fontsize=16)
        self._save_plot(fig, 'word_cloud.png')

    def plot_sentiment_distribution(self, df: pd.DataFrame):
        """公開方法：繪製情感分佈的圓餅圖"""
        print("\n--- Generating Sentiment Distribution Chart ---")
        sentiment_counts = df['sentiment_label'].value_counts()
        colors = {'正面': 'mediumseagreen', '負面': 'tomato', '中性': 'lightgray'}

        fig, ax = plt.subplots(figsize=(8, 6))
        sentiment_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%',
                              colors=[colors.get(x, 'skyblue') for x in sentiment_counts.index],
                              textprops={'fontsize': 12})
        ax.set_title('整體情感分佈', fontsize=16)
        ax.set_ylabel('') # 移除pie圖的ylabel
        self._save_plot(fig, 'sentiment_distribution.png')


    def plot_post_volume_over_time(self, df: pd.DataFrame):
        """公開方法：根據 'artDate' 和 'sentiment_label' 繪製聲量趨勢圖"""
        
        # ✨ 關鍵修正：將 'timestamp' 改為 'artDate'
        date_column = 'artDate'
        
        if date_column not in df.columns:
            print(f"Warning: '{date_column}' column not found. Skipping plot_post_volume_over_time.")
            return

        print("\n--- Generating Post Volume Over Time Chart ---")
        df_time = df.set_index(date_column)
        
        # 按日統計各情感聲量
        time_sentiment = df_time.groupby([pd.Grouper(freq='D'), 'sentiment_label']).size().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        time_sentiment.plot(kind='line', ax=ax, marker='o', linestyle='-')
        
        ax.set_title('聲量時間趨勢圖', fontsize=16)
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('文章數量', fontsize=12)
        ax.legend(title='情感類別')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, 'post_volume_over_time.png')

    def plot_topic_distribution(self, df: pd.DataFrame):
        """公開方法：根據 'topic_keywords' 繪製主題分佈長條圖"""
        print("\n--- Generating Topic Distribution Chart ---")
        topic_counts = df['topic_keywords'].value_counts()

        fig, ax = plt.subplots(figsize=(10, max(6, len(topic_counts) * 0.5)))
        sns.barplot(x=topic_counts.values, y=topic_counts.index, ax=ax, palette='viridis')
        
        ax.set_title('熱門主題分佈', fontsize=16)
        ax.set_xlabel('文章數量', fontsize=12)
        ax.set_ylabel('主題代表關鍵詞', fontsize=12)
        # 在長條上顯示數字
        for i, v in enumerate(topic_counts.values):
            ax.text(v + 0.5, i, str(v), color='black', va='center')
        self._save_plot(fig, 'topic_distribution.png')

    def plot_topic_sentiment_heatmap(self, df: pd.DataFrame):
        """公開方法：繪製主題與情感交叉分析的熱力圖"""
        print("\n--- Generating Topic-Sentiment Heatmap ---")
        topic_sentiment_crosstab = pd.crosstab(df['topic_keywords'], df['sentiment_label'])
        # 正規化，顯示每個主題內的情感百分比
        crosstab_norm = topic_sentiment_crosstab.div(topic_sentiment_crosstab.sum(axis=1), axis=0)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(crosstab_norm, ax=ax, annot=True, fmt='.2%', cmap='coolwarm', linewidths=.5)
        
        ax.set_title('主題情感交叉分析熱力圖', fontsize=16)
        ax.set_xlabel('情感類別', fontsize=12)
        ax.set_ylabel('主題代表關鍵詞', fontsize=12)
        self._save_plot(fig, 'topic_sentiment_heatmap.png')

# ================= 模組獨立測試區塊 =================
if __name__ == "__main__":
    print("===== 正在獨立測試 Visualizer 模組 =====")

    # 模擬來自所有分析步驟的最終DataFrame
    mock_final_data = {
        'keywords': [
            ['保濕', '效果', '水潤'], ['質地', '清爽', '不黏膩'],
            ['細紋', '抗老', '推薦'], ['痘痘', '油膩', '不適合'],
            ['保濕', '補水', '乾燥'], ['質地', '厚重', '太油'],
            ['細紋', '堅持', '變淡'], ['敏感', '泛紅', '不耐受']
        ],
        'sentiment_label': ['正面', '正面', '正面', '負面', '正面', '負面', '正面', '負面'],
        'timestamp': pd.to_datetime([
            '2025-07-21', '2025-07-21', '2025-07-22', '2025-07-22',
            '2025-07-23', '2025-07-23', '2025-07-24', '2025-07-24'
        ]),
        'topic_id': [0, 1, 2, 3, 0, 1, 2, 3],
        'topic_keywords': [
            '保濕, 效果, 水潤', '質地, 清爽, 膚感',
            '細紋, 抗老, 推薦', '痘痘, 油膩, 敏感',
            '保濕, 效果, 水潤', '質地, 清爽, 膚感',
            '細紋, 抗老, 推薦', '痘痘, 油膩, 敏感'
        ]
    }
    mock_df = pd.DataFrame(mock_final_data)
    
    print("\n--- 測試前，模擬的最終分析資料 ---")
    print(mock_df.head())
    
    # 建立 Config 和 Visualizer 物件
    config = Config()
    visualizer = Visualizer(config)
    
    # 逐一執行所有繪圖功能
    visualizer.plot_word_cloud(mock_df)
    visualizer.plot_sentiment_distribution(mock_df)
    visualizer.plot_post_volume_over_time(mock_df)
    visualizer.plot_topic_distribution(mock_df)
    visualizer.plot_topic_sentiment_heatmap(mock_df)

    print("\n===== Visualizer 模組獨立測試完畢 =====")
    print(f"所有圖表已成功儲存至 '{config.OUTPUT_PATH}' 資料夾，請前往查看。")
