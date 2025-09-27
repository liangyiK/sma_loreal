# 導入必要套件
import os, re
from datetime import datetime
import warnings
import json
from collections import Counter, namedtuple, defaultdict
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set
from math import log
import time
from functools import reduce
from pprint import pprint

# 文字處理與 NLP
import jieba, jieba.posseg as pseg, jieba.analyse
from snownlp import SnowNLP
from nltk import ngrams, FreqDist
from gensim import corpora, models
from wordcloud import WordCloud

# 機器學習
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict, KFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc,accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# 視覺化
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from itertools import combinations
import community as community_louvain
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Gensim 進階功能
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.matutils import corpus2csc, corpus2dense, Sparse2Corpus
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# 繪圖設定
sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 第二階段：資料載入與初步檢查
# =============================================================================

# 資料載入
try:
    pdata = pd.read_csv('pdata.csv')  # PTT資料（已篩選萊雅相關）
    ddata = pd.read_csv('ddata.csv')  # DCARD資料（美妝版面全資料）
    
    print("✅ 資料載入成功！")
    print(f"PTT資料筆數: {len(pdata)}")
    print(f"DCARD資料筆數: {len(ddata)}")
    
except FileNotFoundError:
    print("❌ 資料檔案未找到，請確認檔案路徑")
    
# 資料基本資訊檢查
def check_data_info(df, name):
    print(f"\n📋 {name} 資料集資訊:")
    print(f"資料形狀: {df.shape}")
    print(f"欄位: {list(df.columns)}")
    print(f"缺失值:\n{df.isnull().sum()}")
    print(f"資料類型:\n{df.dtypes}")
    return df.head()

# 檢查兩個資料集
print("="*50)
pdata_sample = check_data_info(pdata, "PTT")
print("\n" + "="*50)
ddata_sample = check_data_info(ddata, "DCARD")

# 視覺化資料基本統計
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('資料集基本統計', fontsize=16)

# 資料筆數比較
platforms = ['PTT', 'DCARD']
counts = [len(pdata), len(ddata)]
axes[0,0].bar(platforms, counts, color=['#FF6B6B', '#4ECDC4'])
axes[0,0].set_title('平台資料筆數比較')
axes[0,0].set_ylabel('筆數')

# 缺失值比較
ptt_missing = pdata.isnull().sum().sum()
dcard_missing = ddata.isnull().sum().sum()
axes[0,1].bar(platforms, [ptt_missing, dcard_missing], color=['#FF6B6B', '#4ECDC4'])
axes[0,1].set_title('缺失值總數比較')
axes[0,1].set_ylabel('缺失值數量')

# PTT欄位缺失值分布
ptt_missing_by_col = pdata.isnull().sum()
axes[1,0].barh(range(len(ptt_missing_by_col)), ptt_missing_by_col.values)
axes[1,0].set_yticks(range(len(ptt_missing_by_col)))
axes[1,0].set_yticklabels(ptt_missing_by_col.index, rotation=45)
axes[1,0].set_title('PTT各欄位缺失值')

# DCARD欄位缺失值分布
dcard_missing_by_col = ddata.isnull().sum()
axes[1,1].barh(range(len(dcard_missing_by_col)), dcard_missing_by_col.values)
axes[1,1].set_yticks(range(len(dcard_missing_by_col)))
axes[1,1].set_yticklabels(dcard_missing_by_col.index, rotation=45)
axes[1,1].set_title('DCARD各欄位缺失值')

plt.tight_layout()
plt.show()

print("\n✅ 第二階段完成：資料載入與基本檢查")

# =============================================================================
# 第三階段：資料清理與DCARD篩選
# =============================================================================

# 萊雅集團品牌關鍵詞（用於DCARD篩選）
loreal_keywords = {
    # 萊雅主品牌
    '萊雅', '巴黎萊雅', "L'Oréal", 'Loreal', 'loreal', 'LOREAL', 'L\'Oreal',
    
    # 高端品牌
    '蘭蔻', 'Lancôme', 'lancome', 'LANCOME', 'Lancome',
    'YSL', 'ysl', '聖羅蘭', '圣罗兰', '圣洛兰', 'Saint Laurent',
    '亞曼尼', 'Armani', 'armani', 'ARMANI', 'Giorgio Armani',
    '契爾氏', "Kiehl's", 'kiehls', 'KIEHLS', 'Kiehl',
    
    # 藥妝品牌
    '薇姿', 'Vichy', 'vichy', 'VICHY',
    '理膚寶水', 'La Roche-Posay', 'laroche', 'LAROCHE', 'La Roche Posay',
    '修麗可', 'SkinCeuticals', 'skinceuticals', 'SKINCEUTICALS',
    
    # 美髮品牌
    '巴黎卡詩', 'Kérastase', 'kerastase', 'KERASTASE', 'Kerastase',
    '萊雅專業', "L'Oréal Professionnel", 'loreal professionnel',
    
    # 彩妝品牌
    '植村秀', 'shu uemura', 'shuuemura', 'SHUUEMURA', 'Shu Uemura',
    '媚比琳', 'Maybelline', 'maybelline', 'MAYBELLINE', 'Maybelline New York',
    
    # 開架品牌
    '卡尼爾', 'Garnier', 'garnier', 'GARNIER',
    
    # 產品系列（熱門）
    '紫熨斗', '啵啵晶露', '積雪草', '玻色因', '普拉絲鏈', '小黑瓶', '超能眼霜',
    '黑繃帶', '紅腰子', '小棕瓶', '黃金面膜', '紅石榴', '檸檬草', '蜂蜜'
}

def contains_loreal_keywords(text):
    """檢查文本是否包含萊雅關鍵詞"""
    if pd.isna(text):
        return False
    
    text_lower = str(text).lower()
    for keyword in loreal_keywords:
        if keyword.lower() in text_lower:
            return True
    return False

def clean_text(text):
    """文字清理函數"""
    if pd.isna(text):
        return ""
    text = re.sub(r'http\S+|www\S+', '', str(text))  # 移除URL
    text = re.sub(r'[^\u4e00-\u9fa5\w\s]', ' ', text)  # 移除特殊符號
    text = re.sub(r'\s+', ' ', text)  # 移除多餘空白
    return text.strip()

def standardize_date(date_str):
    """日期標準化"""
    if pd.isna(date_str):
        return pd.NaT
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except (ValueError, TypeError):
        return pd.NaT

print("🔧 開始進行資料清理與DCARD篩選...")

# 複製原始資料
pdata_clean = pdata.copy()
ddata_clean = ddata.copy()

# 先篩選DCARD中的萊雅相關文章
print("🔍 篩選DCARD中的萊雅相關文章...")
dcard_loreal_mask = (
    ddata_clean['artTitle'].apply(contains_loreal_keywords) | 
    ddata_clean['artContent'].apply(contains_loreal_keywords)
)
ddata_clean = ddata_clean[dcard_loreal_mask].copy()

print(f"📊 DCARD篩選後剩餘資料: {len(ddata_clean)} 筆")

# 移除內容為空的資料
pdata_clean.dropna(subset=['artContent'], inplace=True)
ddata_clean.dropna(subset=['artContent'], inplace=True)
print(f"🗑️ 移除內容為空的資料後，PTT剩下 {len(pdata_clean)} 筆，DCARD剩下 {len(ddata_clean)} 筆")

# PTT 資料清理
print("✔️ 正在清理 PTT 資料...")
pdata_clean['artPoster'].fillna('未知作者', inplace=True)
pdata_clean['artComment'].fillna('', inplace=True)
pdata_clean['e_ip'].fillna('未知IP', inplace=True)

pdata_clean['artDate'] = pdata_clean['artDate'].apply(standardize_date)
pdata_clean['artTitle'] = pdata_clean['artTitle'].apply(clean_text)
pdata_clean['artContent'] = pdata_clean['artContent'].apply(clean_text)

# DCARD 資料清理
print("✔️ 正在清理 DCARD 資料...")
ddata_clean['department'].fillna('匿名或未知', inplace=True)
ddata_clean['gender'].fillna('未知', inplace=True)
ddata_clean['school'].fillna('未知', inplace=True)

ddata_clean['artDate'] = ddata_clean['artDate'].apply(standardize_date)
ddata_clean['artTitle'] = ddata_clean['artTitle'].apply(clean_text)
ddata_clean['artContent'] = ddata_clean['artContent'].apply(clean_text)

# 刪除日期格式錯誤的資料
pdata_clean.dropna(subset=['artDate'], inplace=True)
ddata_clean.dropna(subset=['artDate'], inplace=True)
print(f"📅 移除無效日期資料後，PTT剩下 {len(pdata_clean)} 筆，DCARD剩下 {len(ddata_clean)} 筆")

# 視覺化清理結果
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('資料清理與篩選結果', fontsize=16)

# PTT 文章時間分布
ptt_dates = pdata_clean['artDate'].dropna()
axes[0,0].hist(ptt_dates, bins=50, color='#FF6B6B', alpha=0.8)
axes[0,0].set_title('PTT 文章時間分布')
axes[0,0].tick_params(axis='x', rotation=45)

# DCARD 文章時間分布
dcard_dates = ddata_clean['artDate'].dropna()
axes[0,1].hist(dcard_dates, bins=50, color='#4ECDC4', alpha=0.8)
axes[0,1].set_title('DCARD 文章時間分布（篩選後）')
axes[0,1].tick_params(axis='x', rotation=45)

# 篩選前後DCARD資料比較
before_after = pd.DataFrame({
    '篩選前': [len(ddata)],
    '篩選後': [len(ddata_clean)]
})
before_after.plot(kind='bar', ax=axes[1,0], color=['#DDD', '#4ECDC4'])
axes[1,0].set_title('DCARD資料篩選前後比較')
axes[1,0].set_ylabel('文章數')
axes[1,0].tick_params(axis='x', rotation=0)

# 文章長度分布比較
axes[1,1].hist(pdata_clean['artContent'].str.len(), bins=50, alpha=0.7, 
               label='PTT', color='#FF6B6B')
axes[1,1].hist(ddata_clean['artContent'].str.len(), bins=50, alpha=0.7, 
               label='DCARD', color='#4ECDC4')
axes[1,1].set_title('文章內容長度分布')
axes[1,1].set_xlabel('字符數')
axes[1,1].legend()

plt.tight_layout()
plt.show()

print("\n✅ 第三階段完成：資料清理與DCARD篩選")

# =============================================================================
# 第四階段：資料整合與統一架構
# =============================================================================

# 設定時間區間
start_date = pd.to_datetime('2024-05-01')
end_date = pd.to_datetime('2025-03-31')

print(f"🕒 設定目標時間區間為: {start_date.date()} 到 {end_date.date()}")

# 根據時間區間篩選資料
pdata_filtered = pdata_clean[
    (pdata_clean['artDate'] >= start_date) & 
    (pdata_clean['artDate'] <= end_date)
].copy()

ddata_filtered = ddata_clean[
    (ddata_clean['artDate'] >= start_date) & 
    (ddata_clean['artDate'] <= end_date)
].copy()

print(f"📊 篩選後 PTT 資料剩下: {len(pdata_filtered)} 筆")
print(f"📊 篩選後 DCARD 資料剩下: {len(ddata_filtered)} 筆")

def create_integrated_dataset(p_df, d_df):
    """將篩選後的PTT與DCARD資料整合為統一格式"""
    
    # 處理 PTT 資料
    ptt_integrated = pd.DataFrame({
        'system_id': 'PTT_' + p_df['system_id'].astype(str),
        'dataSource': 'PTT',
        'artUrl': p_df['artUrl'],
        'artDate': p_df['artDate'],
        'artTitle': p_df['artTitle'],
        'artContent': p_df['artContent'],
        'category': p_df['artCatagory'],
        'user_info': p_df['artPoster'],
        'interaction_data': p_df['artComment'],
        'interaction_count': p_df['artComment'].str.len(),
        'platform_specific': p_df[['e_ip', 'insertedDate']].apply(
            lambda x: json.dumps(x.to_dict(), default=str), axis=1
        )
    })
    
    # 處理 DCARD 資料
    dcard_integrated = pd.DataFrame({
        'system_id': 'DCARD_' + d_df['system_id'].astype(str),
        'dataSource': 'DCARD',
        'artUrl': d_df['artUrl'],
        'artDate': d_df['artDate'],
        'artTitle': d_df['artTitle'],
        'artContent': d_df['artContent'],
        'category': d_df['boardID'],
        'user_info': d_df['department'] + '_' + d_df['gender'] + '_' + d_df['school'],
        'interaction_data': '',
        'interaction_count': d_df['commentCount'],
        'platform_specific': d_df[['gender', 'school', 'department']].apply(
            lambda x: json.dumps(x.to_dict(), default=str), axis=1
        )
    })
    
    # 合併資料
    combined = pd.concat([ptt_integrated, dcard_integrated], ignore_index=True)
    
    # 型別轉換
    combined['artDate'] = pd.to_datetime(combined['artDate'])
    combined['interaction_count'] = pd.to_numeric(combined['interaction_count'], errors='coerce').fillna(0)
    
    return combined

# 執行整合
combined_data = create_integrated_dataset(pdata_filtered, ddata_filtered)

print("\n✅ 資料整合完成！")
print(f"總筆數: {len(combined_data)}")
print(f"時間範圍: {combined_data['artDate'].min().date()} 到 {combined_data['artDate'].max().date()}")
print("\n整合後資料集資訊:")
combined_data.info()

# 視覺化整合結果
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('整合後資料分析', fontsize=16)

# 每日文章數趨勢
daily_posts = combined_data.groupby(combined_data['artDate'].dt.date).size()
axes[0,0].plot(daily_posts.index, daily_posts.values, color='#45B7D1', marker='.', linestyle='-')
axes[0,0].set_title('每日文章數趨勢')
axes[0,0].set_xlabel('日期')
axes[0,0].set_ylabel('文章數')
axes[0,0].tick_params(axis='x', rotation=45)

# 平台分布
platform_counts = combined_data['dataSource'].value_counts()
axes[0,1].pie(platform_counts.values, labels=platform_counts.index, autopct='%1.1f%%',
              colors=['#4ECDC4', '#FF6B6B'], startangle=90)
axes[0,1].set_title('平台資料源分布')

# 互動數分布比較
ptt_interaction = combined_data[combined_data['dataSource'] == 'PTT']['interaction_count']
dcard_interaction = combined_data[combined_data['dataSource'] == 'DCARD']['interaction_count']
axes[1,0].hist(ptt_interaction, bins=50, alpha=0.7, label='PTT', color='#FF6B6B')
axes[1,0].hist(dcard_interaction, bins=50, alpha=0.7, label='DCARD', color='#4ECDC4')
axes[1,0].set_title('互動數分布比較')
axes[1,0].set_xlabel('互動數')
axes[1,0].set_ylabel('頻率')
axes[1,0].set_yscale('log')
axes[1,0].legend()

# 文章內容長度分布
axes[1,1].boxplot([
    combined_data[combined_data['dataSource']=='PTT']['artContent'].str.len(),
    combined_data[combined_data['dataSource']=='DCARD']['artContent'].str.len()
], labels=['PTT', 'DCARD'])
axes[1,1].set_title('文章內容長度分布')
axes[1,1].set_ylabel('字符數')

plt.tight_layout()
plt.show()

print("\n✅ 第四階段完成：資料整合與統一架構")

# =============================================================================
# 第五階段：文字處理與分詞
# =============================================================================

# --- 步驟 1：定義所有需要的函數和詞典 ---

# 載入自訂詞典
user_dict_path = 'user_dict.txt'
if os.path.exists(user_dict_path):
    jieba.load_userdict(user_dict_path)

# 擴充後的停用詞典
enhanced_stopwords = {
    '的', '是', '我', '你', '他', '她', '它', '我們', '你們', '他們', '這', '那', '這個', '那個', '什麼', '怎麼', '為什麼', '因為', '所以', '但是', '然後', '還是', '或者', '如果', '雖然', '可以', '不能', '應該', '必須', '想要', '需要', '覺得', '認為', '知道', '了解', '明白', '說', '講', '問', '答', '回', '推', '噓', '看', '聽', '做', '用', '有', '沒有', '沒', '很', '非常', '特別', '比較', '更', '最', '太', '好', '不好', '對', '錯', '真', '假', '會', '不會', '能', '不能', '要', '不要', '給', '被', '讓', '使', '從', '到', '在', '版主', '樓主', '推文', '回文', '置頂', '精華', '刪除', '修改', '編輯', '原po', 'op', '轉錄', '引用', '回覆', '留言', '評論', '按讚', '分享', '收藏', '匿名', '卡友', '連結', '網址', '圖片', '影片', '檔案', '下載', '上傳', '更新', '登入', '註冊', '真的', '大家', '就是', '感覺', '自己', '而且', '一下', '可能', '不過', '還有', '或是', '只是', '之前', '好像', '看起來', '聽說', '感覺上', '基本上', '老實說', '有點', '其實', '當然', '不然', '年', '月', '日', '小時', '分鐘', '秒', '現在', '以前', '以後', '剛才', '馬上', '一個', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '萬', '起來', '個', '位', '名', '人', '次', '遍', '下', '回', '趟', '番', '場', '件', '條', '張', '啊', '呀', '哦', '哇', '嗯', '嗨', '呢', '吧', '嘛', '啦', '咧', '餒', '欸', '嗯嗯', '呵呵', '哈哈', '對啊', '是說', '話說', '結果', '後來', '剛剛', '剛好', '正好', '差不多', '大概', '左右', '這樣', '那樣', '怎樣', '請問', '想問', '求推', '徵求', '心得', '開箱', '試用', '體驗', '使用', '感謝', '謝謝', '大大', '版友', '同意', '認同', '副本', '本版', '歸團員', '號包', '原原', '集愛會員', '贈品櫃', '中國信', '專櫃售價', '大刷', '投訴', '百貨店', '櫃姐', '門市', '官網', '網站', '北區', '南區', '中區', '東區', '西區', '台北', '台中', '台南', '高雄', '梨果', '碧菲絲特', '高俊熙', '米菲亞', '水寶貝', '美樂家', '肌本', '不好意思', '請教'
}

# 同義詞詞典
synonym_groups = {
    '效果': ['效果', '功效', '作用', '成效'],
    '顏色': ['顏色', '色號', '色彩', '色調'],
    '保濕': ['保濕', '滋潤', '補水', '水潤'],
    '質地': ['質地', '觸感', '手感', '質感', '膚感'],
    '萊雅': ['萊雅', "L'Oréal", 'Loreal', 'loreal', 'LOREAL', 'L\'Oreal', '巴黎萊雅'],
    '蘭蔻': ['蘭蔻', 'Lancôme', 'lancome', 'LANCOME', 'Lancome'],
    '聖羅蘭': ['聖羅蘭', 'YSL', 'ysl', 'Saint Laurent', '圣罗兰'],
    '理膚寶水': ['理膚寶水', 'La Roche-Posay', 'laroche', 'LAROCHE', 'La Roche Posay'],
    '契爾氏': ['契爾氏', "Kiehl's", 'kiehls', 'KIEHLS', 'Kiehl'],
    '玻色因': ['玻色因', '普拉絲鏈', 'Pro-xylane'],
    '粉底液': ['粉底液', '粉底', '底妝'],
    '唇膏': ['唇膏', '口紅', '唇釉', '唇彩'],
    '精華液': ['精華液', '精華']
}

def check_loreal_mentions(tokens):
    """檢查萊雅相關詞彙"""
    mentions = []
    for token in tokens:
        for keyword in loreal_keywords:
            if keyword.lower() in token.lower():
                mentions.append(keyword)
    return list(set(mentions))

print("🔧 正在建立同義詞對照表以加速處理...")
synonym_map = {}
for standard, variants in synonym_groups.items():
    for variant in variants:
        synonym_map[variant] = standard

def advanced_tokenizer_and_normalizer(text):
    """
    整合的進階分詞器：使用優化後的同義詞查詢
    """
    if pd.isna(text) or text == '':
        return []
    
    words_with_pos = pseg.cut(text)
    
    keep_pos = {'n', 'v', 'a', 'ad', 'vn', 'an', 'nr', 'ns', 'nt', 'nz'}
    
    normalized_tokens = []
    for word, pos in words_with_pos:
        word = word.strip()
        if (len(word) >= 2 and 
            word not in enhanced_stopwords and
            pos in keep_pos and
            not word.isdigit() and
            not re.match(r'^[a-zA-Z]+$', word)):
            
            # 【核心優化】從 O(N) 的迴圈查詢變成 O(1) 的字典查詢
            normalized_word = synonym_map.get(word, word)
            normalized_tokens.append(normalized_word)
            
    return normalized_tokens

print("🔤 開始文字處理與分詞 ...")
print(f"📊 處理 {len(combined_data)} 筆資料")

def advanced_tokenizer_and_normalizer(text):
    """
    整合的進階分詞器：包含詞性過濾與同義詞正規化
    """
    if pd.isna(text) or text == '':
        return []
    
    # 使用詞性標註分詞
    words_with_pos = pseg.cut(text)
    
    # 保留的詞性：名詞、動詞、形容詞等
    keep_pos = {'n', 'v', 'a', 'ad', 'vn', 'an', 'nr', 'ns', 'nt', 'nz'}
    
    filtered_tokens = []
    for word, pos in words_with_pos:
        word = word.strip()
        if (len(word) >= 2 and 
            word not in enhanced_stopwords and
            pos in keep_pos and
            not word.isdigit() and
            not re.match(r'^[a-zA-Z]+$', word)):
            filtered_tokens.append(word)
    
    # 同義詞正規化
    normalized_tokens = []
    for token in filtered_tokens:
        found = False
        for standard, variants in synonym_groups.items():
            if token in variants:
                normalized_tokens.append(standard)
                found = True
                break
        if not found:
            normalized_tokens.append(token)
            
    return normalized_tokens

def get_tfidf_top_words(texts, top_n=20):
    """使用 TF-IDF 提取 Top N 關鍵詞"""
    if not texts:
        return []
    
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.7, ngram_range=(1,2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    
    # 計算每個詞的平均 TF-IDF 分數
    avg_tfidf_scores = tfidf_matrix.mean(axis=0).A1
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # 建立 DataFrame 並排序
    df_tfidf = pd.DataFrame({'term': feature_names, 'tfidf': avg_tfidf_scores})
    df_tfidf = df_tfidf.sort_values(by='tfidf', ascending=False)
    
    return list(df_tfidf['term'].head(top_n))

# --- 步驟 2：執行唯一的、優化後的文字處理流程 ---

print("🔤 開始執行優化後的文字處理流程...")

# 合併標題和內容
combined_data['full_text'] = combined_data['artTitle'] + ' ' + combined_data['artContent']

# 執行新的分詞與正規化流程
print("正在進行分詞、詞性過濾與同義詞正規化...")
combined_data['tokens'] = combined_data['full_text'].apply(advanced_tokenizer_and_normalizer)

# 計算文字統計
combined_data['word_count'] = combined_data['tokens'].apply(len)
combined_data['unique_words'] = combined_data['tokens'].apply(lambda x: len(set(x)))
combined_data['lexical_diversity'] = combined_data['unique_words'] / combined_data['word_count'].replace(0, 1)
combined_data['lexical_diversity'].fillna(0, inplace=True)

# 重新檢測萊雅相關度
combined_data['loreal_mentions'] = combined_data['tokens'].apply(check_loreal_mentions)
combined_data['has_loreal'] = combined_data['loreal_mentions'].apply(lambda x: len(x) > 0)


# --- 步驟 3：使用 TF-IDF 提取關鍵詞並輸出 ---

print("\n🔍 正在使用 TF-IDF 提取全域關鍵詞...")
corpus_for_tfidf = [' '.join(tokens) for tokens in combined_data['tokens']]
top_tfidf_words = get_tfidf_top_words(corpus_for_tfidf, top_n=15)

# --- 步驟 4：輸出結果與視覺化 ---

print(f"\n📊 文字處理統計摘要 (優化後):")
print(f"總詞彙數 (過濾後): {sum(combined_data['word_count']):,}")
print(f"獨特詞彙數: {len(set([token for tokens in combined_data['tokens'] for token in tokens])):,}")
print(f"平均文章詞彙數: {combined_data['word_count'].mean():.1f}")

print(f"\n🔥 Top 15 TF-IDF 關鍵詞:")
for i, word in enumerate(top_tfidf_words, 1):
    print(f"{i:2d}. {word}")

# 視覺化（可選，與之前類似）
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('優化後文字處理結果分析', fontsize=16)

# 平台詞彙數比較
sns.boxplot(x='dataSource', y='word_count', data=combined_data, ax=axes[0,0])
axes[0,0].set_title('平台詞彙數分布比較 (優化後)')

# 詞彙豐富度比較
sns.boxplot(x='dataSource', y='lexical_diversity', data=combined_data, ax=axes[0,1])
axes[0,1].set_title('詞彙豐富度分布比較 (優化後)')

# Top 15 TF-IDF 詞彙視覺化
# 為了視覺化，我們需要分數，這裡重新計算一下
tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.7, ngram_range=(1,2))
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_for_tfidf)
avg_tfidf_scores = tfidf_matrix.mean(axis=0).A1
feature_names = tfidf_vectorizer.get_feature_names_out()
df_tfidf = pd.DataFrame({'term': feature_names, 'tfidf': avg_tfidf_scores}).sort_values(by='tfidf', ascending=False).head(15)

sns.barplot(x='tfidf', y='term', data=df_tfidf, ax=axes[1,0], palette='viridis')
axes[1,0].set_title('Top 15 TF-IDF 關鍵詞')

# 萊雅品牌提及頻率
brand_mentions = Counter([m for mentions in combined_data['loreal_mentions'] for m in mentions])
if brand_mentions:
    top_brands = dict(brand_mentions.most_common(10))
    axes[1,1].barh(list(top_brands.keys()), list(top_brands.values()), color='skyblue')
    axes[1,1].set_title('Top 10 萊雅品牌提及頻率')
    axes[1,1].invert_yaxis()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\n✅ 第五階段完成（優化整合版）")

# =============================================================================
# 第六階段：優化情感分析（V3版本）
# =============================================================================

# 擴充美妝領域情感詞典
positive_words = {
    '好用', '喜歡', '推薦', '必買', '回購', '愛用', '神器', '救星', '蜜糖', '服貼', '清爽',
    '滋潤', '保濕', '透亮', '光澤', '穩定', '有效', '改善', '細緻', '滑嫩', '溫和', '舒服',
    '驚艷', '滿意', '超值', '划算', '持久', '不脫妝', '顯色', '好推', '吸收快', '不黏膩',
    '不致痘', '不卡粉', '不浮粉', '有感', '必收'
}

negative_words = {
    '難用', '不推', '反推', '後悔', '失望', '地雷', '毒藥', '致痘', '粉刺', '過敏', '泛紅',
    '刺激', '黏膩', '油膩', '厚重', '悶', '無感', '雞肋', '浪費錢', '暗沉', '脫妝', '卡粉',
    '浮粉', '顯毛孔', '不持久', '難卸', '飛粉', '不顯色', '不好聞', '香精味', '死白', '災難'
}

# 程度副詞詞典
degree_words = {
    '超': 2.0, '超級': 2.0, '非常': 1.8, '很': 1.5, '好': 1.5, '太': 1.5,
    '有點': 0.8, '稍微': 0.7, '不太': -0.5, '不夠': -0.8
}

# 否定詞
negation_words = {'不', '沒', '沒有', '不是', '不像'}

# PMI計算停用詞（過濾雜訊）
pmi_stopwords = {
    '金石', '書店', '誠品', '查價', '稅拔', '特賣會', '特賣價', '樓管', '櫃姐',
    '百貨', '專櫃', '週年慶', '母親節', '會員', '點數', '優惠', '折扣', '免運',
    '蝦皮', '官網', '門市', '取貨', '付款', '宅配', '服務業', '版置', '全家', '7-11',
    '天婦羅', '米菲亞', '水寶貝', '美樂家', '堂書店', '誠品線', '除色', '攜伴',
    '封館', '傳染', '收綠單', '白綠單', '全進化', '大樓', '白單', '綠島'
}

def sentiment_analysis_v3(text, tokens):
    """
    V3版情感分析模型：結合詞典規則、程度副詞、否定詞與SnowNLP
    """
    if not tokens:
        return 0.5, '中性'

    # 計算基於詞典的規則分數
    rule_score = 0
    word_count = len(tokens)
    
    for i, token in enumerate(tokens):
        weight = 1.0
        
        # 檢查前一個詞是否為程度副詞或否定詞
        if i > 0:
            prev_word = tokens[i-1]
            if prev_word in degree_words:
                weight *= degree_words[prev_word]
            elif prev_word in negation_words:
                weight *= -1.0
        
        # 根據情感詞典加權
        if token in positive_words:
            rule_score += weight
        elif token in negative_words:
            rule_score -= weight

    # 標準化規則分數
    if rule_score != 0:
        normalized_rule_score = log(abs(rule_score) + 1) * (rule_score / abs(rule_score)) / log(word_count + 2)
    else:
        normalized_rule_score = 0
    
    # 取得 SnowNLP 基礎分數
    try:
        snownlp_score = SnowNLP(text).sentiments
    except:
        snownlp_score = 0.5

    # 結合兩種分數（規則分數70%，SnowNLP 30%）
    rule_score_0_1 = (normalized_rule_score + 1) / 2
    final_score = 0.7 * rule_score_0_1 + 0.3 * snownlp_score
    final_score = max(0, min(1, final_score))

    # 根據分數定義標籤
    if final_score > 0.6: label = '正面'
    elif final_score < 0.4: label = '負面'
    else: label = '中性'
        
    return final_score, label

def get_pmi_keywords_v3(df, label, top_n=15):
    """
    V3版PMI計算：
    1. 新增商業購物類雜訊詞過濾
    2. 提高詞頻門檻，避免低頻詞干擾
    3. 輸出Top N關鍵詞
    """
    # 原有的PMI停用詞
    pmi_stopwords = {
        '金石堂', '書店', '誠品', '查價', '稅拔', '特賣會', '特賣價', '樓管', '櫃姐',
        '百貨', '專櫃', '週年慶', '母親節', '會員', '點數', '免運',
        '蝦皮', '官網', '門市', '取貨', '宅配', '服務業', '版置', '全家', '7-11',
        '天婦羅', '米菲亞', '水寶貝', '美樂家', '堂書店', '誠品線', '除色', '攜伴',
        '封館', '傳染', '收綠單', '白綠單', '全進化', '大樓', '白單', '綠島',
        # V2版結果中看到的雜訊詞也加入
        '集愛', '特會', '團者', '認購', '樣機', '貨店', '鋪買', '出境', '包兌', '航廈', '請準'
    }

    # 【核心優化】新增商業購物類雜訊詞
    commercial_stopwords = {
        '購物', '價格', '優惠', '團購', '代購', '特價', '貨運', '物流', '包裝', '出貨',
        '折扣', '寄貨', '賣家', '訂購', '出貨', '商品', '賣場', '包裹', '付款', '退款'
    }

    # 合併成完整的停用詞列表
    extended_stopwords = pmi_stopwords.union(commercial_stopwords)

    target_docs = df[df['sentiment_label_v3'] == label]
    other_docs = df[df['sentiment_label_v3'] != label]
    if len(target_docs) == 0 or len(other_docs) == 0:
        return []

    # 使用擴充後的停用詞表進行過濾
    target_tokens = [token for tokens in target_docs['tokens'] for token in tokens
                     if token not in extended_stopwords]
    other_tokens = [token for tokens in other_docs['tokens'] for token in tokens
                    if token not in extended_stopwords]

    target_word_counts = Counter(target_tokens)
    other_word_counts = Counter(other_tokens)
    total_target_words = sum(target_word_counts.values())
    total_other_words = sum(other_word_counts.values())

    if total_target_words == 0 or total_other_words == 0:
        return []

    pmi_scores = {}
    for word, count in target_word_counts.items():
        # 【核心優化】提高詞頻門檻，過濾低頻雜訊
        if count < 15:
            continue

        p_word_target = count / total_target_words
        # 避免分母為0，若在其他文件中沒出現過，給一個極小值(1)
        p_word_other = other_word_counts.get(word, 1) / total_other_words
        pmi = log(p_word_target / p_word_other)
        pmi_scores[word] = pmi

    return sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

# 執行V3情感分析
print("🔄 執行 V3 版情感分析...")
sentiment_results_v3 = combined_data.apply(
    lambda row: sentiment_analysis_v3(row['full_text'], row['tokens']), axis=1
)
combined_data[['sentiment_score_v3', 'sentiment_label_v3']] = pd.DataFrame(
    sentiment_results_v3.tolist(), index=combined_data.index
)

print("\n🔍 正在使用 V3 版 PMI 提取情感特徵詞...")
positive_pmi_keywords_v3 = get_pmi_keywords_v3(combined_data, '正面')
negative_pmi_keywords_v3 = get_pmi_keywords_v3(combined_data, '負面')

# 視覺化分析結果
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('情感分析結果（V3版本）', fontsize=16)

# 情感分布
sentiment_counts = combined_data['sentiment_label_v3'].value_counts()
axes[0,0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
              colors=['#4ECDC4', '#FF6B6B', '#FAD390'], startangle=90)
axes[0,0].set_title('整體情感分布')

# 情感分數分布
sns.histplot(combined_data['sentiment_score_v3'], bins=50, kde=True, ax=axes[0,1], color='skyblue')
axes[0,1].set_title('情感分數分布')
axes[0,1].set_xlabel('情感分數')
axes[0,1].axvline(0.5, color='grey', linestyle='--', alpha=0.5)

# 平台情感比較
platform_sentiment = combined_data.groupby('dataSource')['sentiment_label_v3'].value_counts(normalize=True).unstack()
platform_sentiment.plot(kind='bar', stacked=True, ax=axes[1,0],
                        color=['#4ECDC4', '#FF6B6B', '#FAD390'])
axes[1,0].set_title('平台情感比較')
axes[1,0].set_ylabel('比例')
axes[1,0].tick_params(axis='x', rotation=0)

# 萊雅相關文章情感比較
loreal_sentiment = combined_data.groupby('has_loreal')['sentiment_label_v3'].value_counts(normalize=True).unstack()
loreal_sentiment.index = ['非萊雅相關', '萊雅相關']
loreal_sentiment.plot(kind='bar', stacked=True, ax=axes[1,1],
                       color=['#4ECDC4', '#FF6B6B', '#FAD390'])
axes[1,1].set_title('萊雅相關文章情感比較')
axes[1,1].set_ylabel('比例')
axes[1,1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()

# 輸出關鍵洞察
print("\n📊 V3版情感分析關鍵洞察:")
print(f"整體平均情感分數: {combined_data['sentiment_score_v3'].mean():.3f}")

if combined_data['has_loreal'].sum() > 0:
    loreal_avg_sentiment = combined_data[combined_data['has_loreal']]['sentiment_score_v3'].mean()
    non_loreal_avg_sentiment = combined_data[~combined_data['has_loreal']]['sentiment_score_v3'].mean()
    print(f"萊雅相關文章平均情感: {loreal_avg_sentiment:.3f}")
    print(f"非萊雅相關文章平均情感: {non_loreal_avg_sentiment:.3f}")

print("\n情感分布:")
print(combined_data['sentiment_label_v3'].value_counts(normalize=True).apply("{:.1%}".format))

print("\n平台情感分數:")
for platform in ['PTT', 'DCARD']:
    platform_avg = combined_data[combined_data['dataSource'] == platform]['sentiment_score_v3'].mean()
    print(f"  - {platform}: {platform_avg:.3f}")

print("\n👍 **Top 15 正面評價特徵詞 (PMI - V3):**")
if positive_pmi_keywords_v3:
    print(', '.join([f"{word}({score:.2f})" for word, score in positive_pmi_keywords_v3]))
else:
    print("無法提取正面特徵詞")

print("\n👎 **Top 15 負面評價特徵詞 (PMI - V3):**")
if negative_pmi_keywords_v3:
    print(', '.join([f"{word}({score:.2f})" for word, score in negative_pmi_keywords_v3]))
else:
    print("無法提取負面特徵詞")

print("\n✅ 第六階段完成：優化情感分析")

# =============================================================================
# 第七階段：詞雲生成與視覺化
# =============================================================================

def create_wordcloud(text_data, title, max_words=100):
    """建立詞雲"""
    if not text_data:
        return None
    
    all_text = ' '.join(text_data)
    
    wordcloud = WordCloud(
        font_path='C:/Windows/Fonts/msjh.ttc',
        width=800,
        height=400,
        background_color='white',
        max_words=max_words,
        colormap='viridis',
        relative_scaling=0.5,
        random_state=42
    ).generate(all_text)
    
    return wordcloud

print("☁️ 開始生成詞雲...")

# 準備不同類型的文字資料
all_text_data = [' '.join(tokens) for tokens in combined_data['tokens'] if tokens]
loreal_text_data = [' '.join(tokens) for tokens in combined_data[combined_data['has_loreal']]['tokens'] if tokens]
positive_text_data = [' '.join(tokens) for tokens in combined_data[combined_data['sentiment_label_v3'] == '正面']['tokens'] if tokens]
negative_text_data = [' '.join(tokens) for tokens in combined_data[combined_data['sentiment_label_v3'] == '負面']['tokens'] if tokens]

# 生成詞雲
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('詞雲分析', fontsize=16)

# 整體詞雲
if all_text_data:
    wordcloud_all = create_wordcloud(all_text_data, '整體詞雲')
    if wordcloud_all:
        axes[0,0].imshow(wordcloud_all, interpolation='bilinear')
        axes[0,0].set_title('整體詞雲')
        axes[0,0].axis('off')

# 萊雅相關詞雲
if loreal_text_data:
    wordcloud_loreal = create_wordcloud(loreal_text_data, '萊雅相關詞雲')
    if wordcloud_loreal:
        axes[0,1].imshow(wordcloud_loreal, interpolation='bilinear')
        axes[0,1].set_title('萊雅相關詞雲')
        axes[0,1].axis('off')

# 正面情感詞雲
if positive_text_data:
    wordcloud_positive = create_wordcloud(positive_text_data, '正面情感詞雲')
    if wordcloud_positive:
        axes[1,0].imshow(wordcloud_positive, interpolation='bilinear')
        axes[1,0].set_title('正面情感詞雲')
        axes[1,0].axis('off')

# 負面情感詞雲
if negative_text_data:
    wordcloud_negative = create_wordcloud(negative_text_data, '負面情感詞雲')
    if wordcloud_negative:
        axes[1,1].imshow(wordcloud_negative, interpolation='bilinear')
        axes[1,1].set_title('負面情感詞雲')
        axes[1,1].axis('off')

plt.tight_layout()
plt.show()

# 高頻詞彙分析圖
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('高頻詞彙分析', fontsize=16)

# 整體高頻詞
top_words_all = Counter()
for tokens in combined_data['tokens']:
    top_words_all.update(tokens)

top_30_words = dict(top_words_all.most_common(30))
axes[0,0].barh(range(len(top_30_words)), list(top_30_words.values()))
axes[0,0].set_yticks(range(len(top_30_words)))
axes[0,0].set_yticklabels(list(top_30_words.keys()))
axes[0,0].set_title('Top 30 高頻詞彙')
axes[0,0].set_xlabel('頻率')

# 萊雅相關高頻詞
if combined_data['has_loreal'].sum() > 0:
    loreal_words = Counter()
    for tokens in combined_data[combined_data['has_loreal']]['tokens']:
        loreal_words.update(tokens)
    
    top_loreal_words = dict(loreal_words.most_common(20))
    axes[0,1].barh(range(len(top_loreal_words)), list(top_loreal_words.values()))
    axes[0,1].set_yticks(range(len(top_loreal_words)))
    axes[0,1].set_yticklabels(list(top_loreal_words.keys()))
    axes[0,1].set_title('萊雅相關 Top 20 詞彙')
    axes[0,1].set_xlabel('頻率')

# 正面詞彙
positive_words_counter = Counter()
for tokens in combined_data[combined_data['sentiment_label_v3'] == '正面']['tokens']:
    positive_words_counter.update(tokens)

top_positive_words = dict(positive_words_counter.most_common(20))
axes[1,0].barh(range(len(top_positive_words)), list(top_positive_words.values()))
axes[1,0].set_yticks(range(len(top_positive_words)))
axes[1,0].set_yticklabels(list(top_positive_words.keys()))
axes[1,0].set_title('正面情感 Top 20 詞彙')
axes[1,0].set_xlabel('頻率')

# 負面詞彙
negative_words_counter = Counter()
for tokens in combined_data[combined_data['sentiment_label_v3'] == '負面']['tokens']:
    negative_words_counter.update(tokens)

top_negative_words = dict(negative_words_counter.most_common(20))
axes[1,1].barh(range(len(top_negative_words)), list(top_negative_words.values()))
axes[1,1].set_yticks(range(len(top_negative_words)))
axes[1,1].set_yticklabels(list(top_negative_words.keys()))
axes[1,1].set_title('負面情感 Top 20 詞彙')
axes[1,1].set_xlabel('頻率')

plt.tight_layout()
plt.show()

print("\n✅ 第七階段完成：詞雲生成與視覺化")

# =============================================================================
# 第八階段：結果儲存與報告生成
# =============================================================================

def generate_analysis_report():
    """生成分析報告"""
    
    report = f"""
# 萊雅社群媒體分析報告

## 資料摘要
- 分析期間: {combined_data['artDate'].min()} 到 {combined_data['artDate'].max()}
- 總文章數: {len(combined_data):,}
- PTT文章數: {len(combined_data[combined_data['dataSource'] == 'PTT']):,}
- DCARD文章數: {len(combined_data[combined_data['dataSource'] == 'DCARD']):,}
- 萊雅相關文章: {combined_data['has_loreal'].sum():,}

## 文字分析結果
- 總詞彙數: {sum(combined_data['word_count']):,}
- 平均文章詞彙數: {combined_data['word_count'].mean():.1f}
- 詞彙豐富度: {combined_data['lexical_diversity'].mean():.3f}

## 情感分析結果（V3版本）
- 正面文章: {(combined_data['sentiment_label_v3'] == '正面').sum():,} ({(combined_data['sentiment_label_v3'] == '正面').sum() / len(combined_data) * 100:.1f}%)
- 負面文章: {(combined_data['sentiment_label_v3'] == '負面').sum():,} ({(combined_data['sentiment_label_v3'] == '負面').sum() / len(combined_data) * 100:.1f}%)
- 中性文章: {(combined_data['sentiment_label_v3'] == '中性').sum():,} ({(combined_data['sentiment_label_v3'] == '中性').sum() / len(combined_data) * 100:.1f}%)
- 平均情感分數: {combined_data['sentiment_score_v3'].mean():.3f}

## 萊雅品牌分析
"""
    
    if combined_data['has_loreal'].sum() > 0:
        loreal_sentiment = combined_data[combined_data['has_loreal']]['sentiment_score_v3'].mean()
        non_loreal_sentiment = combined_data[~combined_data['has_loreal']]['sentiment_score_v3'].mean()
        
        report += f"""
- 萊雅相關文章平均情感: {loreal_sentiment:.3f}
- 非萊雅相關文章平均情感: {non_loreal_sentiment:.3f}
- 情感差異: {loreal_sentiment - non_loreal_sentiment:.3f}
"""
    
    # 高頻詞彙
    all_tokens = []
    for tokens in combined_data['tokens']:
        all_tokens.extend(tokens)
    
    word_freq = Counter(all_tokens)
    top_words = word_freq.most_common(15)
    
    report += "\n## Top 15 高頻詞彙\n"
    for word, count in top_words:
        report += f"- {word}: {count:,}\n"
    
    # PMI特徵詞
    if positive_pmi_keywords_v3:
        report += "\n## 正面評價特徵詞\n"
        for word, score in positive_pmi_keywords_v3[:10]:
            report += f"- {word}: {score:.2f}\n"
    
    if negative_pmi_keywords_v3:
        report += "\n## 負面評價特徵詞\n"
        for word, score in negative_pmi_keywords_v3[:10]:
            report += f"- {word}: {score:.2f}\n"
    
    return report

print("📝 生成分析報告...")

# 生成報告
analysis_report = generate_analysis_report()

# 儲存結果
print("💾 儲存分析結果...")

# 儲存處理後的資料
combined_data.to_csv('processed_social_media_data.csv', index=False, encoding='utf-8-sig')

# 儲存報告
with open('analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(analysis_report)

# 儲存關鍵統計資料
key_stats = {
    'total_posts': len(combined_data),
    'ptt_posts': len(combined_data[combined_data['dataSource'] == 'PTT']),
    'dcard_posts': len(combined_data[combined_data['dataSource'] == 'DCARD']),
    'loreal_posts': combined_data['has_loreal'].sum(),
    'positive_posts': (combined_data['sentiment_label_v3'] == '正面').sum(),
    'negative_posts': (combined_data['sentiment_label_v3'] == '負面').sum(),
    'neutral_posts': (combined_data['sentiment_label_v3'] == '中性').sum(),
    'avg_sentiment': combined_data['sentiment_score_v3'].mean(),
    'avg_words_per_post': combined_data['word_count'].mean(),
    'analysis_period': f"{combined_data['artDate'].min()} 到 {combined_data['artDate'].max()}"
}

# 儲存為JSON
with open('key_statistics.json', 'w', encoding='utf-8') as f:
    json.dump(key_stats, f, ensure_ascii=False, indent=2, default=str)

# 生成最終儀表板
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('萊雅社群媒體分析 - 最終儀表板', fontsize=16)

# 平台分布
platform_counts = combined_data['dataSource'].value_counts()
axes[0,0].pie(platform_counts.values, labels=platform_counts.index, autopct='%1.1f%%',
              colors=['#FF6B6B', '#4ECDC4'])
axes[0,0].set_title('平台分布')

# 情感分布
sentiment_counts = combined_data['sentiment_label_v3'].value_counts()
axes[0,1].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
              colors=['#4ECDC4', '#FF6B6B', '#FAD390'])
axes[0,1].set_title('情感分布')

# 萊雅相關度
loreal_counts = combined_data['has_loreal'].value_counts()
axes[0,2].pie(loreal_counts.values, labels=['非萊雅相關', '萊雅相關'], autopct='%1.1f%%',
              colors=['#DDD', '#FF7675'])
axes[0,2].set_title('萊雅相關度')

# 時間趨勢
daily_posts = combined_data.groupby(combined_data['artDate'].dt.date).size()
axes[1,0].plot(daily_posts.index, daily_posts.values, color='#74B9FF')
axes[1,0].set_title('每日文章數趨勢')
axes[1,0].set_xlabel('日期')
axes[1,0].set_ylabel('文章數')
axes[1,0].tick_params(axis='x', rotation=45)

# 詞彙數分布
axes[1,1].hist(combined_data['word_count'], bins=50, alpha=0.7, color='#A29BFE')
axes[1,1].set_title('文章詞彙數分布')
axes[1,1].set_xlabel('詞彙數')
axes[1,1].set_ylabel('頻率')

# 互動數分布
axes[1,2].hist(combined_data['interaction_count'], bins=50, alpha=0.7, color='#FD79A8')
axes[1,2].set_title('互動數分布')
axes[1,2].set_xlabel('互動數')
axes[1,2].set_ylabel('頻率')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("🎉 萊雅社群媒體分析完成！")
print("="*60)
print(analysis_report)
print("="*60)
print("📁 已儲存檔案:")
print("- processed_social_media_data.csv (處理後的完整資料)")
print("- analysis_report.txt (分析報告)")
print("- key_statistics.json (關鍵統計資料)")
print("\n✅ 第八階段完成：結果儲存與報告生成")

import gensim
import pyLDAvis.gensim_models as gensimvis

# =============================================================================
# 第九階段：主題模型分析
# =============================================================================

print("🔍 開始進行主題模型分析...")

# 準備資料：選擇有意義的長文進行分析，效果較好
topic_data = combined_data[combined_data['word_count'] > 30]['tokens'].tolist()
if not topic_data:
    print("❌ 沒有足夠的資料進行主題模型分析。")
else:
    # 1. 建立詞典
    dictionary = corpora.Dictionary(topic_data)
    dictionary.filter_extremes(no_below=15, no_above=0.5) # 過濾低頻和高頻詞

    # 2. 建立語料庫 (Bag-of-Words)
    corpus = [dictionary.doc2bow(text) for text in topic_data]

    # 3. 訓練LDA模型
    # 可調整 num_topics 來決定要找出幾個主題
    lda_model = gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=8,  # 假設我們先找出8個主題
        random_state=100,
        chunksize=100,
        passes=10,
        per_word_topics=True
    )

    # 4. 輸出主題結果
    print("\n📊 主題模型結果：")
    for idx, topic in lda_model.print_topics(-1):
        print(f"主題 #{idx+1}: {topic}")

    # 5. 視覺化主題模型 (這會在瀏覽器中打開一個互動頁面)
    print("\n☁️ 正在生成主題模型視覺化圖表...")
    pyLDAvis.enable_notebook()
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    
    # 如果您在非Jupyter環境，可以儲存成HTML檔案
    pyLDAvis.save_html(vis_data, 'lda_visualization.html')
    print("✅ 主題模型視覺化報告已儲存為 lda_visualization.html")

print("\n✅ 第九階段完成：主題模型分析")


from itertools import combinations

# =============================================================================
# 第十階段：關聯性分析
# =============================================================================

print("🔗 開始進行關聯性分析...")

# 我們可以專注於萊雅相關的文章
loreal_tokens_list = combined_data[combined_data['has_loreal']]['tokens'].tolist()

# 建立一個詞彙共現的計數器
co_occurrence = Counter()

# 計算詞彙對在同一篇文章中出現的次數
for tokens in loreal_tokens_list:
    # 使用 set 確保每個詞在單篇文章中只被計算一次，避免高頻詞的過度影響
    unique_tokens = set(tokens)
    
    # 建立詞彙對
    for w1, w2 in combinations(unique_tokens, 2):
        # 確保順序一致，例如 (A,B) 和 (B,A) 都算成 (A,B)
        key = tuple(sorted((w1, w2)))
        co_occurrence[key] += 1

# 篩選出最常共現的50個詞彙對來建立網絡
top_pairs = co_occurrence.most_common(50)

if not top_pairs:
    print("❌ 沒有足夠的資料進行關聯性分析。")
else:
    # 建立網絡圖
    G = nx.Graph()
    for (w1, w2), weight in top_pairs:
        G.add_edge(w1, w2, weight=weight)

    # 視覺化網絡圖
    plt.figure(figsize=(16, 16))
    
    # 使用 spring_layout 讓節點分佈更美觀
    pos = nx.spring_layout(G, k=0.6, iterations=50, seed=42)
    
    # 繪製節點
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='skyblue', alpha=0.8)
    
    # 繪製邊
    nx.draw_networkx_edges(G, pos, width=[d['weight']/10 for u,v,d in G.edges(data=True)],
                           edge_color='grey', alpha=0.6)
    
    # 繪製標籤
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='Microsoft YaHei')
    
    plt.title('萊雅相關詞彙共現網絡圖', size=20)
    plt.axis('off')
    plt.show()
    
    import community as community_louvain
    partition = community_louvain.best_partition(G)

    # 繪圖時，根據社群來設定節點顏色
    node_colors = [partition.get(node) for node in G.nodes()]

    plt.figure(figsize=(18, 18))
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
    nx.draw(G, pos, node_color=node_colors, with_labels=True, 
        font_family='Microsoft YaHei', cmap=plt.cm.viridis)
    plt.title('詞彙網絡社群偵測結果', size=20)
    plt.show()

print("\n✅ 第十階段完成：關聯性分析")

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack


# =============================================================================
# 第十一階段：機器學習預測模型
# =============================================================================

print("🤖 開始建立高互動文章預測模型...")

# 1. 定義目標變數 (y)
# 將互動數超過75百分位數的文章定義為「高互動」
high_interaction_threshold = combined_data['interaction_count'].quantile(0.75)
combined_data['is_high_interaction'] = (combined_data['interaction_count'] > high_interaction_threshold).astype(int)

# 如果高互動文章太少，則不進行
if combined_data['is_high_interaction'].sum() < 20:
    print("❌ 高互動文章樣本數過少，不適合建立模型。")
else:
    # 2. 準備特徵 (X)
    # 特徵1：文章內容 (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
    text_features = vectorizer.fit_transform([' '.join(tokens) for tokens in combined_data['tokens']])

    # 特徵2：其他數值特徵
    numeric_features_df = combined_data[['sentiment_score_v3', 'word_count']]

    # 【核心優化】對數值特徵進行標準化
    scaler = StandardScaler()
    numeric_features_scaled = scaler.fit_transform(numeric_features_df)

    # 將文字特徵和【標準化後】的數值特徵合併
    X = hstack([text_features, numeric_features_scaled])
    y = combined_data['is_high_interaction']

    # 3. 分割訓練集與測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 4. 訓練模型（使用羅吉斯迴歸作為範例）
    model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000) # 增加 max_iter
    model.fit(X_train, y_train)

    # 5. 評估模型
    y_pred_proba = model.predict_proba(X_test)[:, 1] # 取得預測為「高互動」的機率
    # 【核心優化】設定新的預測門檻，例如 0.65
    custom_threshold = 0.65
    y_pred_adjusted = (y_pred_proba >= custom_threshold).astype(int)
    
    print("\n📈 調整門檻後，模型評估結果:")
    print(f"模型準確率: {accuracy_score(y_test, y_pred_adjusted):.2%}")
    print("\n分類報告:")
    print(classification_report(y_test, y_pred_adjusted, target_names=['一般文章', '高互動文章']))

    # (可選) 查看哪些詞彙對預測高互動最重要
    feature_names = vectorizer.get_feature_names_out()
    all_feature_names = np.append(feature_names, ['sentiment_score', 'word_count'])
    
    # 取得模型係數
    coefs = model.coef_[0]
    top_coef_indices = np.argsort(coefs)[-15:] # 找出最重要的15個特徵
    
    print("\n🔥 **預測高互動的Top 15關鍵特徵**:")
    for i in top_coef_indices:
        print(f"- {all_feature_names[i]}: {coefs[i]:.4f}")

print("\n✅ 第十一階段完成：機器學習預測")

