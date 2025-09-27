"""
主題建模模組
Topic modeling module for L'Oreal project
"""

import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import os
from typing import List, Tuple, Dict, Optional
from config import Config

class TopicModeler:
    """
    負責從文本數據中挖掘潛在主題
    """
    def __init__(self, config: Config):
        self.config = config
        print("TopicModeler initialized.")

    def _prepare_corpus(self, tokens_list: List[List[str]]) -> Tuple[corpora.Dictionary, List[List[Tuple[int, int]]]]:
        """
        私有方法：將分詞列表轉換為LDA模型所需的字典和BoW語料庫
        """
        # 建立詞語到ID的映射
        id2word = corpora.Dictionary(tokens_list)
        # 過濾極端詞語
        id2word.filter_extremes(no_below=self.config.LDA_NO_BELOW, no_above=self.config.LDA_NO_ABOVE)
        
        # 將文本轉換為詞袋格式
        corpus = [id2word.doc2bow(tokens) for tokens in tokens_list]
        print(f"Corpus prepared: {len(id2word)} unique tokens, {len(corpus)} documents.")
        return id2word, corpus

    def find_optimal_topics(self, tokens_list: List[List[str]]) -> int:
        """
        公開方法：計算不同主題數下的一致性分數，找到最佳主題數
        """
        print("\n--- 開始尋找最佳主題數 (這可能需要幾分鐘) ---")
        id2word, corpus = self._prepare_corpus(tokens_list)
        
        coherence_values = []
        model_list = []
        
        start = self.config.LDA_TOPIC_RANGE['start']
        limit = self.config.LDA_TOPIC_RANGE['limit']
        step = self.config.LDA_TOPIC_RANGE['step']

        for num_topics in range(start, limit, step):
            print(f"正在測試 {num_topics} 個主題...")
            model = gensim.models.LdaMulticore(
                corpus=corpus,
                id2word=id2word,
                num_topics=num_topics,
                random_state=100,
                chunksize=100,
                passes=10,
                workers=self.config.LDA_WORKERS
            )
            model_list.append(model)
            
            coherencemodel = CoherenceModel(model=model, texts=tokens_list, dictionary=id2word, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        # 找到最高一致性分數對應的主題數
        best_result_index = coherence_values.index(max(coherence_values))
        optimal_topics = range(start, limit, step)[best_result_index]
        print(f"--- 尋找完畢！最佳主題數為: {optimal_topics} (Coherence Score: {max(coherence_values):.4f}) ---")
        
        return optimal_topics

    def train_model(self, df: pd.DataFrame, num_topics: int) -> Tuple[pd.DataFrame, gensim.models.LdaModel, corpora.Dictionary, List]:
        """
        公開方法：使用指定的主題數訓練最終的LDA模型
        """
        print(f"\n--- 使用 {num_topics} 個主題訓練最終模型 ---")
        tokens_list = df['tokens'].tolist()
        id2word, corpus = self._prepare_corpus(tokens_list)
        
        lda_model = gensim.models.LdaMulticore(
            corpus=corpus,
            id2word=id2word,
            num_topics=num_topics,
            random_state=100,
            chunksize=100,
            passes=20,
            workers=self.config.LDA_WORKERS
        )

        # 為每篇文章分配最可能的主題
        doc_topics = [lda_model.get_document_topics(doc) for doc in corpus]
        
        # 找出每篇文章最主要的主題ID和該主題的關鍵詞
        main_topic_id = []
        main_topic_keywords = []
        topic_keywords_map = {i: ', '.join([word for word, _ in lda_model.show_topic(i, topn=5)]) for i in range(num_topics)}

        for doc_topic in doc_topics:
            if not doc_topic:
                main_topic_id.append(-1)
                main_topic_keywords.append("N/A")
                continue
            
            main_topic = sorted(doc_topic, key=lambda x: x[1], reverse=True)[0]
            topic_id = main_topic[0]
            main_topic_id.append(topic_id)
            main_topic_keywords.append(topic_keywords_map[topic_id])

        df_result = df.copy()
        df_result['topic_id'] = main_topic_id
        df_result['topic_keywords'] = main_topic_keywords
        
        print("--- 模型訓練與主題分配完成 ---")
        return df_result, lda_model, id2word, corpus

    def visualize_topics(self, lda_model, corpus, id2word):
        """
        公開方法：使用pyLDAvis產生互動式視覺化報告
        """
        print("\n--- 正在產生視覺化報告 (pyLDAvis)... ---")
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(self.config.LDA_VIS_PATH), exist_ok=True)
        
        vis_data = gensimvis.prepare(lda_model, corpus, id2word, mds='mmds')
        pyLDAvis.save_html(vis_data, self.config.LDA_VIS_PATH)
        print(f"--- 視覺化報告已儲存至: {self.config.LDA_VIS_PATH} ---")
        print("請用瀏覽器開啟此HTML檔案以進行互動式探索。")

# ================= 模組獨立測試區塊 =================
if __name__ == "__main__":
    print("===== 正在獨立測試 TopicModeler 模組 =====")

    # 模擬來自 TextProcessor 的輸出 DataFrame
    mock_processed_data = {
        'tokens': [
            ['精華液', '保濕', '效果', '超好', '皮膚', '水潤'],
            ['質地', '清爽', '不黏膩', '吸收', '很快', '膚感', '不錯'],
            ['這款', '眼霜', '撫平', '細紋', '很有感', '抗老', '推薦'],
            ['用', '了', '會', '長', '痘痘', '質地', '太', '油膩', '不', '適合', '我'],
            ['玻尿酸', '成分', '補水', '效果', '一流', '整天', '不', '乾燥'],
            ['乳霜', '有點', '厚重', '夏天', '用', '可能', '太油'],
            ['青春', '密碼', '精華', '堅持', '使用', '細紋', '變淡', '了'],
            ['皮膚', '敏感', '用', '了', '會', '泛紅', '可能', '不', '耐受']
        ]
    }
    mock_df = pd.DataFrame(mock_processed_data)
    
    print("\n--- 測試前，來自TextProcessor的處理後資料 ---")
    print(mock_df)
    
    # 建立 Config 和 TopicModeler 物件
    config = Config()
    modeler = TopicModeler(config)
    
    # 針對測試，臨時放寬詞彙過濾標準
    print("\n[測試模式] 臨時將詞彙過濾標準 (no_below) 從 5 調整為 1")
    modeler.config.LDA_NO_BELOW = 1

    # 手動設定主題數為4
    optimal_k = 4 
    print(f"\n基於測試數據，我們手動設定最佳主題數為: {optimal_k}")

    # 訓練最終模型
    topic_df, lda_model, id2word, corpus = modeler.train_model(mock_df, num_topics=optimal_k)

    # 驗證結果
    print("\n--- 測試後，帶有主題分析的資料 ---")
    print(topic_df[['tokens', 'topic_id', 'topic_keywords']])
    
    # 產生視覺化報告
    modeler.visualize_topics(lda_model, corpus, id2word)

    print("\n===== TopicModeler 模組獨立測試完畢 =====")
