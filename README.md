# 萊雅品牌社群媒體分析專案 (L'Oréal Social Media Analysis)

## 專案概述

本專案是一個針對萊雅品牌在PTT和DCARD社群媒體平台的綜合分析工具，提供完整的文字分析、情感分析、主題建模和視覺化功能。

## 專案結構

```
loreal_project/
├── config.py              # 配置檔案
├── dataloader.py          # 資料載入器
├── datacleaner.py         # 資料清理器
├── textprocessor.py       # 文字處理器
├── sentimentanalyzer.py   # 情感分析器
├── topicmodeler.py        # 主題建模器
├── visualizer.py          # 視覺化器
├── analyzer.py            # 主要分析器
├── main.py                # 主程式
├── usage_example.py       # 使用範例
├── requirements.txt       # 套件需求
└── README.md             # 專案說明
```

## 安裝與設定

### 1. 安裝相依套件

```bash
pip install -r requirements.txt
```

### 2. 準備資料

請確保以下資料檔案存在於專案根目錄：
- `pdata.csv` (PTT資料)
- `ddata.csv` (DCARD資料)

### 3. 配置設定

可在 `config.py` 中調整以下參數：
- 分析時間範圍
- 萊雅品牌關鍵字
- 主題建模參數
- 輸出路徑等

## 使用方式

### 基本使用

```python
from analyzer import LorealAnalysisProject

# 初始化專案
project = LorealAnalysisProject()

# 執行完整分析
success = project.run_complete_analysis(
    generate_visualizations=True,
    save_results=True,
    output_dir="./results"
)
```

### 客製化分析

```python
# 只執行特定分析類型
project.run_custom_analysis(
    analysis_types=['text', 'sentiment', 'topic'],
    output_dir="./custom_results"
)
```

### 逐步執行

```python
project.load_data()           # 載入資料
project.clean_data()          # 清理資料
project.process_text()        # 文字處理
project.analyze_sentiment()   # 情感分析
project.analyze_topics()      # 主題分析
```

## 主要功能

### 1. 資料處理
- 自動載入PTT和DCARD資料
- 資料清理和標準化
- 萊雅品牌關鍵字過濾
- 整合多平台資料

### 2. 文字分析
- jieba中文分詞
- TF-IDF關鍵字提取
- 詞頻統計和分析
- 文字特徵提取

### 3. 情感分析
- SnowNLP基本情感分析
- 詞典式情感分析
- 多層次情感評分
- 品牌情感對比分析

### 4. 主題建模
- LDA主題建模
- 自動主題命名
- 文檔-主題分布
- 主題一致性評估

### 5. 視覺化
- 資料概覽圖表
- 情感分析圖表
- 主題分布圖表
- 時間序列分析
- 詞雲生成
- 綜合儀表板

## 輸出檔案

執行完成後會在輸出目錄生成：

### 資料檔案
- `integrated_social_media_data.csv` - 整合處理後的資料
- `analysis_report.txt` - 文字分析報告
- `key_statistics.json` - 統計數據JSON

### 視覺化檔案
- `data_overview.png` - 資料概覽
- `sentiment_analysis.png` - 情感分析
- `topic_analysis.png` - 主題分析
- `time_series.png` - 時間序列
- `wordcloud_all.png` - 整體詞雲
- `wordcloud_loreal.png` - 萊雅詞雲
- `comprehensive_dashboard.png` - 綜合儀表板

## 模組說明

### Config (config.py)
- 集中管理所有配置參數
- 萊雅品牌關鍵字定義
- 分析參數設定

### DataLoader (dataloader.py)  
- 載入原始PTT和DCARD資料
- 資料品質檢查
- 基本統計資訊

### DataCleaner (datacleaner.py)
- 清理和標準化資料
- 萊雅關鍵字過濾
- 整合多平台資料格式

### TextProcessor (textprocessor.py)
- 中文分詞處理
- 關鍵字提取
- 文字特徵計算
- 詞彙表建立

### SentimentAnalyzer (sentimentanalyzer.py)
- 多種情感分析方法
- 品牌情感對比
- 情感關鍵詞分析

### TopicModeler (topicmodeler.py)
- LDA主題建模
- 主題詞提取
- 文檔主題分配
- 模型儲存/載入

### Visualizer (visualizer.py)
- 完整視覺化功能
- 多種圖表類型
- 自動儲存圖表

### LorealAnalysisProject (analyzer.py)
- 主控制器類
- 整合所有模組
- 完整分析流程

## 進階使用

### 自訂配置

```python
from config import Config

config = Config()
config.TOPIC_NUM = 10  # 調整主題數
config.MIN_WORD_COUNT = 50  # 調整最小字數

project = LorealAnalysisProject(config)
```

### 單獨使用模組

```python
from sentimentanalyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze_sentiment_advanced("測試文字")
```

### 模型儲存與載入

```python
# 儲存主題模型
project.topic_modeler.save_model("./models/topic_model")

# 載入主題模型  
project.topic_modeler.load_model("./models/topic_model")
```

## 擴展性設計

本專案採用模組化設計，具有良好的擴展性：

1. **新增分析模組**：可輕鬆加入新的分析功能
2. **客製化視覺化**：可自訂圖表樣式和類型  
3. **多資料源支援**：可擴展支援更多社群媒體平台
4. **彈性配置**：所有參數都可透過配置檔案調整

## 疑難排解

### 常見問題

1. **編碼問題**：確保資料檔案使用UTF-8編碼
2. **記憶體不足**：可調整批次大小參數
3. **中文字體**：視覺化可能需要安裝中文字體

### 效能最佳化

1. 使用較小的資料集進行測試
2. 調整主題建模參數
3. 考慮使用多核心處理

## 版本資訊

- **版本**: 2.0.0
- **最後更新**: 2025-07-21
- **Python需求**: >= 3.8

## 貢獻

歡迎提交Issue和Pull Request來改善這個專案。

## 授權

本專案僅供學術和研究用途。
