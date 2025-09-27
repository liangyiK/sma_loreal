"""
主要分析器模組
Main analyzer module for L'Oreal project
"""

from config import Config
from data_loader import DataLoader
from data_cleaner import DataCleaner
from text_processor import TextProcessor
from sentiment_analyzer import SentimentAnalyzer
from topic_modeler import TopicModeler
from visualizer import Visualizer
import pandas as pd
import os
import warnings


# 忽略不影響執行的警告訊息
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class LorealAnalysisProject:
    """
    整合所有分析模組的總控制器 (重構版)
    """
    def __init__(self):
        self.config = Config()
        self.loader = DataLoader(self.config)
        self.cleaner = DataCleaner(self.config)
        self.processor = TextProcessor(self.config)
        self.sentiment = SentimentAnalyzer(self.config)
        self.topic_modeler = TopicModeler(self.config)
        self.visualizer = Visualizer(self.config)
        print("\n===== 萊雅分析專案已初始化 (新流程) =====")

    def run_complete_analysis(self):
        """
        執行從頭到尾的完整分析流程 (新流程)
        """
        print("\n🚀 開始執行萊雅專案完整分析流程...")
        
        # 步驟 1: 載入並整合所有資料源
        print("\n📂 步驟 1/5: 載入、篩選並整合多平台資料")
        integrated_df = self.loader.load_data()
        
        if integrated_df is None or integrated_df.empty:
            print("❌ 無法載入任何有效資料，分析中止。")
            return None

        # 步驟 2: 深度文字清理
        print("\n🧹 步驟 2/5: 深度文字清理")
        cleaned_df = self.cleaner.clean(integrated_df)

        # 步驟 3: 文字處理與特徵工程
        print("\n📝 步驟 3/5: 文字處理與分詞")
        processed_df = self.processor.process_text_data(cleaned_df)

        # 步驟 4: 情感與主題分析
        print("\n🎯 步驟 4/5: 情感與主題分析")
        sentiment_df = self.sentiment.analyze(processed_df)
        
        # 主題建模
        if len(sentiment_df) > 50:
            optimal_k = self.topic_modeler.find_optimal_topics(sentiment_df['tokens'].tolist())
        else:
            optimal_k = min(5, max(2, len(sentiment_df) // 10))
            print(f"資料量較少，手動設定主題數為: {optimal_k}")
        
        if len(sentiment_df) < 50:
            self.topic_modeler.config.LDA_NO_BELOW = 1

        final_df, lda_model, id2word, corpus = self.topic_modeler.train_model(sentiment_df, num_topics=optimal_k)

        # ✨ 步驟 5: 生成視覺化報告與儲存結果 (補全缺失的程式碼)
        print("\n📊 步驟 5/5: 生成視覺化報告與儲存結果")
        
        # 儲存最終的CSV檔案
        final_csv_path = os.path.join(self.config.OUTPUT_PATH, "final_analysis_results.csv")
        final_df.to_csv(final_csv_path, index=False, encoding='utf-8-sig')
        print(f"📋 詳細分析結果已儲存至: {final_csv_path}")

        # 呼叫Visualizer產生所有圖表
        self.visualizer.plot_word_cloud(final_df)
        self.visualizer.plot_sentiment_distribution(final_df)
        self.visualizer.plot_post_volume_over_time(final_df)
        self.visualizer.plot_topic_distribution(final_df)
        self.visualizer.plot_topic_sentiment_heatmap(final_df)
        
        # 產生LDA互動式視覺化報告
        self.topic_modeler.visualize_topics(lda_model, corpus, id2word)
        
        # ✨ 呼叫摘要函式，顯示最終結果
        self._display_analysis_summary(final_df)
        
        print("\n🎉 ===== 萊雅分析專案完整流程執行完畢！ =====")
        print(f"📁 所有圖表與報告已儲存至 '{self.config.OUTPUT_PATH}' 資料夾。")
        
        return final_df

    def _display_analysis_summary(self, df: pd.DataFrame):
        """
        顯示分析結果摘要
        """
        print("\n" + "="*50)
        print("📊 分析結果摘要")
        print("="*50)
        
        print(f"📝 總文章數: {len(df)}")
        
        sentiment_summary = df['sentiment_label'].value_counts(normalize=True) * 100
        print(f"\n😊 情感分析結果:")
        for sentiment, percentage in sentiment_summary.items():
            count = int(len(df) * percentage / 100)
            print(f"   {sentiment}: {count} 篇 ({percentage:.1f}%)")
        
        topic_summary = df['topic_keywords'].value_counts().head()
        print(f"\n🎯 熱門主題 (前{len(topic_summary)}名):")
        for i, (topic, count) in enumerate(topic_summary.items(), 1):
            percentage = (count / len(df)) * 100
            print(f"   {i}. {topic}: {count} 篇 ({percentage:.1f}%)")
        
        print("="*50)

    def run_quick_analysis(self, skip_topic_modeling: bool = False):
        """
        執行快速分析流程（跳過耗時的主題建模）
        """
        print("\n⚡ 開始執行快速分析流程...")
        
        raw_df = self.loader.load_data()
        cleaned_df = self.cleaner.clean(raw_df)
        processed_df = self.processor.process_text_data(cleaned_df)
        sentiment_df = self.sentiment.analyze(processed_df)
        
        if not skip_topic_modeling:
            # 使用固定主題數，跳過尋找最佳主題數的步驟
            final_df, _, _, _ = self.topic_modeler.train_model(sentiment_df, num_topics=5)
        else:
            final_df = sentiment_df
        
        # 產生部分視覺化圖表
        self.visualizer.plot_word_cloud(final_df)
        self.visualizer.plot_sentiment_distribution(final_df)
        
        print("\n⚡ 快速分析完成！")
        return final_df

# ================= 模組獨立測試區塊 =================
if __name__ == "__main__":
    print("===== 正在獨立測試 Analyzer 模組 =====")
    
    # 建立專案實例
    project = LorealAnalysisProject()
    
    # 執行完整分析
    results = project.run_complete_analysis()
    
    print("\n--- 最終分析結果預覽 ---")
    print(results.head())
    
    print("\n===== Analyzer 模組獨立測試完畢 =====")
