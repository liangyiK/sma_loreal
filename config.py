"""
萊雅專案配置檔案
Configuration file for L'Oreal project
"""

import multiprocessing
import pandas as pd
import os  # ✨ 1. 唯一的額外匯入

# ✨ 2. 最小且最關鍵的新增：動態定義專案的根目錄
#    __file__ 代表 config.py 這個檔案本身
#    os.path.abspath(__file__) 取得它的絕對路徑
#    os.path.dirname(...) 取得該檔案所在的目錄路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    """
    萊雅專案的統一配置管理類別 - 根據實際數據需求優化
    """
    
    # ======== 資料來源設定 ========
    # ✨ 3. 將所有相對路徑，用 os.path.join() 與 BASE_DIR 結合
    DATA_SOURCES = {
        'ptt': os.path.join(BASE_DIR, 'data', 'pdata.csv'),
        'dcard': os.path.join(BASE_DIR, 'data', 'ddata.csv')
    }
    
    # 時間區間設定
    START_DATE = '2024-05-01'
    END_DATE = '2025-03-31'
    
    # ... (LOREAL_KEYWORDS, 記憶體優化, 資料清理設定維持不變) ...
    LOREAL_KEYWORDS = {
        '萊雅', '巴黎萊雅', "L'Oréal", 'Loreal', 'loreal', 'LOREAL', "L'Oreal",
        '蘭蔻', 'Lancôme', 'lancome', 'LANCOME', 'Lancome', 'YSL', 'ysl', '聖羅蘭', '圣罗兰', '圣洛兰', 'Saint Laurent',
        '亞曼尼', 'Armani', 'armani', 'ARMANI', 'Giorgio Armani', '契爾氏', "Kiehl's", 'kiehls', 'KIEHLS', 'Kiehl',
        '薇姿', 'Vichy', 'vichy', 'VICHY', '理膚寶水', 'La Roche-Posay', 'laroche', 'LAROCHE', 'La Roche Posay',
        '修麗可', 'SkinCeuticals', 'skinceuticals', 'SKINCEUTICALS', '巴黎卡詩', 'Kérastase', 'kerastase', 'KERASTASE', 'Kerastase',
        '萊雅專業', "L'Oréal Professionnel", 'loreal professionnel', '植村秀', 'shu uemura', 'shuuemura', 'SHUUEMURA', 'Shu Uemura',
        '媚比琳', 'Maybelline', 'maybelline', 'MAYBELLINE', 'Maybelline New York', '卡尼爾', 'Garnier', 'garnier', 'GARNIER',
        '紫熨斗', '啵啵晶露', '積雪草', '玻色因', '普拉絲鏈', '小黑瓶', '超能眼霜', '黑繃帶', '紅腰子', '小棕瓶', '黃金面膜', '紅石榴', '檸檬草', '蜂蜜'
    }
    CHUNK_SIZE = 10000
    DTYPE_OPTIMIZATION = {'int64': 'int32', 'float64': 'float32'}
    MIN_CONTENT_LENGTH = 10
    SIMILARITY_THRESHOLD = 0.9

    # ======== 文字處理器設定（強化版）========
    USER_DICT_PATH = os.path.join(BASE_DIR, 'resources', 'user_dict.txt')
    STOP_WORDS_PATH = os.path.join(BASE_DIR, 'resources', 'stop_words.txt')
    
    # ... (ENHANCED_STOPWORDS, SYNONYM_GROUPS 等設定維持不變) ...
    ENHANCED_STOPWORDS = {
        '的', '是', '我', '你', '他', '她', '它', '我們', '你們', '他們', '這', '那', '這個', '那個', '什麼', '怎麼', '為什麼', '因為', '所以', '但是', '然後', '還是', '或者', '如果', '雖然', '可以', '不能', '應該', '必須', '想要', '需要', '覺得', '認為', '知道', '了解', '明白', '看', '聽', '做', '用', '有', '沒有', '沒', '很', '非常', '特別', '比較', '更', '最', '太', '好', '不好', '對', '錯', '真', '假', '會', '不會', '能', '不能', '要', '不要', '真的', '大家', '就是', '感覺', '自己', '而且', '一下', '可能', '不過', '還有', '或是', '只是', '之前', '好像', '看起來', '聽說', '感覺上', '基本上', '老實說', '有點', '其實', '當然', '不然', '使用', '這樣', '版主', '樓主', '推文', '回文', '置頂', '精華', '刪除', '修改', '編輯', '原po', 'op', '轉錄', '引用', '回覆', '留言', '評論', '按讚', '分享', '收藏', '匿名', '卡友', '連結', '網址', '圖片', '影片', '檔案'
    }
    SYNONYM_GROUPS = {
        '效果': ['效果', '功效', '作用', '成效'], '顏色': ['颜色', '色號', '色彩', '色調'], '保濕': ['保濕', '滋潤', '補水', '水潤'], '質地': ['質地', '觸感', '手感', '質感', '膚感'],
        '萊雅': ['萊雅', "L'Oréal", 'Loreal', 'loreal', 'LOREAL', "L'Oreal", '巴黎萊雅'], '蘭蔻': ['蘭蔻', 'Lancôme', 'lancome', 'LANCOME', 'Lancome'],
        '聖羅蘭': ['聖羅蘭', 'YSL', 'ysl', 'Saint Laurent', '圣罗兰'], '理膚寶水': ['理膚寶水', 'La Roche-Posay', 'laroche', 'LAROCHE', 'La Roche Posay'],
        '契爾氏': ['契爾氏', "Kiehl's", 'kiehls', 'KIEHLS', 'Kiehl'], '玻色因': ['玻色因', '普拉絲鏈', 'Pro-xylane'],
        '粉底液': ['粉底液', '粉底', '底妝'], '唇膏': ['唇膏', '口紅', '唇釉', '唇彩'], '精華液': ['精華液', '精華']
    }
    MIN_TOKEN_LENGTH = 2
    KEYWORDS_TOP_K = 8

    # ======== 情感分析器設定 ========
    POSITIVE_DICT_PATH = os.path.join(BASE_DIR, 'resources', 'positive_sentiment.txt')
    NEGATIVE_DICT_PATH = os.path.join(BASE_DIR, 'resources', 'negative_sentiment.txt')

    # ... (情感閾值設定維持不變) ...
    SENTIMENT_POSITIVE_THRESHOLD = 0.6
    SENTIMENT_NEGATIVE_THRESHOLD = 0.35

    # ======== 主題模型設定 ========
    # ... (LDA設定維持不變) ...
    LDA_NO_BELOW = 3
    LDA_NO_ABOVE = 0.6
    LDA_TOPIC_RANGE = {'start': 3, 'limit': 12, 'step': 1}
    LDA_WORKERS = max(1, multiprocessing.cpu_count() - 1)
    
    LDA_VIS_PATH = os.path.join(BASE_DIR, 'output', 'lda_visualization.html')
    
    # ======== 視覺化設定 ========
    OUTPUT_PATH = os.path.join(BASE_DIR, 'output/')
    
    # ✨ 中文字體路徑仍可使用您系統的絕對路徑，無需修改
    CHINESE_FONT_PATH = 'C:/Windows/Fonts/msyh.ttc'

