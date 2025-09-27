"""
資料載入器模組
Data loader module for L'Oreal project
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Optional, Union, Tuple
from config import Config
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """
    負責載入和整合多資料源，支援記憶體優化和資料品質檢查
    """
    def __init__(self, config: Config):
        self.config = config
        print("DataLoader initialized with multi-source support.")
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        優化DataFrame的資料類型以減少記憶體使用
        """
        original_memory = df.memory_usage().sum()
        
        # 數值型欄位優化
        for col in df.columns:
            if df[col].dtype == 'int64':
                if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype('int32')
            elif df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        # 分類型欄位優化
        for col in df.columns:
            if df[col].dtype == 'object':
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                if num_unique_values / num_total_values < 0.5:
                    df[col] = df[col].astype('category')
        
        optimized_memory = df.memory_usage().sum()
        # 避免除以零，並確保 reduction 不為負數
        if original_memory > 0:
            reduction = max(0, (original_memory - optimized_memory) / original_memory * 100)
        else:
            reduction = 0
        
        print(f"記憶體使用優化: 減少了 {reduction:.1f}% ({original_memory:,} -> {optimized_memory:,} bytes)")
        
        return df
    
    def _contains_loreal_keywords(self, text: str) -> bool:
        """
        檢查文本是否包含萊雅關鍵詞
        """
        if pd.isna(text):
            return False
        
        text_lower = str(text).lower()
        for keyword in self.config.LOREAL_KEYWORDS:
            if keyword.lower() in text_lower:
                return True
        return False
    
    def _load_single_source(self, file_path: str, source_type: str) -> pd.DataFrame:
        """
        載入單一資料源並進行初步處理 (最終修正版)
        """
        if not os.path.exists(file_path):
            print(f"⚠️ 檔案不存在: {file_path}")
            return pd.DataFrame()
        
        try:
            chunks = []
            print(f"   - 開始解析檔案: {os.path.basename(file_path)}")
            # 加入 on_bad_lines='skip' 來跳過格式錯誤的行
            for chunk in pd.read_csv(file_path, chunksize=self.config.CHUNK_SIZE, 
                                   low_memory=True, encoding='utf-8',
                                   parse_dates=['artDate'],
                                   infer_datetime_format=True,
                                   on_bad_lines='skip'):
                
                # 即時篩選萊雅相關內容（僅對DCARD）
                if source_type == 'dcard':
                    if 'artTitle' in chunk.columns and 'artContent' in chunk.columns:
                        title_mask = chunk['artTitle'].fillna(False).apply(self._contains_loreal_keywords)
                        content_mask = chunk['artContent'].fillna(False).apply(self._contains_loreal_keywords)
                        chunk = chunk[title_mask | content_mask]
                
                if not chunk.empty:
                    chunks.append(chunk)
            
            if not chunks:
                print(f"⚠️ 在 {source_type.upper()} 中沒有找到有效內容")
                return pd.DataFrame()
            
            df = pd.concat(chunks, ignore_index=True)
            df = self._optimize_dtypes(df)
            
            print(f"✅ {source_type.upper()} 資料載入成功: {len(df):,} 筆")
            return df
            
        except Exception as e:
            if "did not match allocated fields" in str(e):
                 print(f"❌ 檔案格式錯誤: {file_path} 的欄位數量與預期不符。")
                 print("   請檢查CSV檔案的標頭和內容是否一致。")
            else:
                print(f"❌ 載入 {source_type} 資料時發生未預期錯誤: {e}")
            return pd.DataFrame()

    # ✨✨✨ 這是被意外刪除的方法，現在將它加回來 ✨✨✨
    def load_multi_source_data(self) -> Dict[str, pd.DataFrame]:
        """
        載入並初步處理多個資料源
        """
        data_sources = {}
        print("🔄 開始載入多資料源...")
        
        for source_name, file_path in self.config.DATA_SOURCES.items():
            print(f"\n📂 正在載入 {source_name.upper()} 資料...")
            df = self._load_single_source(file_path, source_name)
            
            if not df.empty:
                data_sources[source_name] = df
                print(f"   資料形狀: {df.shape}")
                print(f"   欄位: {list(df.columns)}")
                if 'artDate' in df.columns and pd.api.types.is_datetime64_any_dtype(df['artDate']):
                    # 只有在 artDate 是日期類型且不為空時才計算範圍
                    if not df['artDate'].dropna().empty:
                        date_range = f"{df['artDate'].min().date()} 到 {df['artDate'].max().date()}"
                        print(f"   時間範圍: {date_range}")
        
        print(f"\n✅ 成功載入 {len(data_sources)} 個資料源")
        return data_sources

    def create_integrated_dataset(self, data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        將多個資料源整合為統一格式
        """
        print("\n🔄 開始整合資料源...")
        integrated_datasets = []
        
        for source_name, df in data_sources.items():
            print(f"正在處理 {source_name.upper()} 資料...")
            
            if source_name == 'ptt':
                integrated_df = self._process_ptt_data(df)
            elif source_name == 'dcard':
                integrated_df = self._process_dcard_data(df)
            else:
                print(f"⚠️ 未知的資料源類型: {source_name}")
                continue
            
            if not integrated_df.empty:
                integrated_datasets.append(integrated_df)
        
        if not integrated_datasets:
            print("❌ 沒有有效的資料可以整合")
            return pd.DataFrame()
        
        combined_df = pd.concat(integrated_datasets, ignore_index=True)
        combined_df = self._filter_by_date_range(combined_df)
        combined_df = self._optimize_dtypes(combined_df)
        
        print(f"\n✅ 資料整合完成!")
        if not combined_df.empty and 'artDate' in combined_df.columns and not combined_df['artDate'].dropna().empty:
            print(f"   總筆數: {len(combined_df):,}")
            print(f"   時間範圍: {combined_df['artDate'].min().date()} 到 {combined_df['artDate'].max().date()}")
            print(f"   資料源分布: {combined_df['dataSource'].value_counts().to_dict()}")
        
        return combined_df
    
    def _process_ptt_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        處理PTT資料為統一格式
        """
        try:
            integrated_df = pd.DataFrame({
                'system_id': 'PTT_' + df['system_id'].astype(str),
                'dataSource': 'PTT',
                'artUrl': df.get('artUrl', ''),
                'artDate': pd.to_datetime(df['artDate'], errors='coerce'),
                'artTitle': df['artTitle'].fillna(''),
                'artContent': df['artContent'].fillna(''),
                'category': df.get('artCatagory', df.get('artCategory', '')),
                'user_info': df.get('artPoster', ''),
                'interaction_data': df.get('artComment', ''),
                'interaction_count': df.get('artComment', '').astype(str).str.len(),
                'platform_specific': df.apply(
                    lambda x: json.dumps({
                        'e_ip': x.get('e_ip', ''),
                        'insertedDate': str(x.get('insertedDate', ''))
                    }, ensure_ascii=False), axis=1
                )
            })
            return integrated_df
        except Exception as e:
            print(f"❌ 處理PTT資料時發生錯誤: {e}")
            return pd.DataFrame()
    

    def _process_dcard_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        處理DCARD資料為統一格式
        """
        try:
            # ✨ 關鍵修正點: 在進行字串操作前，先將 category 型別轉回 object (string) 型別
            # 這樣就可以安全地使用 .fillna('') 和進行字串拼接
            user_info_str = (
                df.get('department', pd.Series(dtype=str)).astype(str).fillna('') + '_' + 
                df.get('gender', '').astype(str).fillna('') + '_' + 
                df.get('school', '').astype(str).fillna('')
            )

            integrated_df = pd.DataFrame({
                'system_id': 'DCARD_' + df['system_id'].astype(str),
                'dataSource': 'DCARD',
                'artUrl': df.get('artUrl', ''),
                'artDate': pd.to_datetime(df['artDate'], errors='coerce'),
                'artTitle': df['artTitle'].fillna(''),
                'artContent': df['artContent'].fillna(''),
                'category': df.get('boardID', ''),
                'user_info': user_info_str, # 使用上面處理好的字串
                'interaction_data': '',
                'interaction_count': pd.to_numeric(df.get('commentCount', 0), errors='coerce').fillna(0),
                'platform_specific': df.apply(
                    lambda x: json.dumps({
                        'gender': x.get('gender', ''),
                        'school': x.get('school', ''),
                        'department': x.get('department', '')
                    }, ensure_ascii=False), axis=1
                )
            })
            return integrated_df
        except Exception as e:
            print(f"❌ 處理DCARD資料時發生錯誤: {e}")
            return pd.DataFrame()
                

    def _create_standardized_df(self, df: pd.DataFrame, source_name: str, schema_mapping: Dict) -> pd.DataFrame:
        """
        通用的標準化DataFrame建立輔助方法 - (最終修正版)
        """
        integrated_data = {}
        for target_col, source_expression in schema_mapping.items():
            if callable(source_expression):
                # 如果是函式，應用它
                integrated_data[target_col] = source_expression(df)
            
            elif isinstance(source_expression, str):
                # ✨ 關鍵修正點: 檢查字串是「欄位名」還是「靜態值」
                if source_expression in df.columns:
                    # 如果字串是來源 df 的一個欄位名，就獲取該欄位的資料
                    integrated_data[target_col] = df.get(source_expression, '')
                else:
                    # 否則，將這個字串本身當作靜態值賦予所有行
                    integrated_data[target_col] = source_expression
            
            else:
                # 處理其他靜態值，例如數字或布林值
                integrated_data[target_col] = source_expression
        
        # 為了確保所有欄位長度一致，將靜態值廣播到整個 DataFrame
        df_len = len(df)
        for col, val in integrated_data.items():
            if not isinstance(val, pd.Series) and not isinstance(val, list):
                integrated_data[col] = [val] * df_len

        return pd.DataFrame(integrated_data)

    def _process_ptt_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        處理PTT資料為統一格式 (重構版)
        """
        schema = {
            'system_id': lambda d: 'PTT_' + d['system_id'].astype(str),
            'dataSource': 'PTT',
            'artUrl': 'artUrl',
            'artDate': lambda d: pd.to_datetime(d['artDate'], errors='coerce'),
            'artTitle': lambda d: d['artTitle'].fillna(''),
            'artContent': lambda d: d['artContent'].fillna(''),
            'category': 'artCatagory',
            'user_info': 'artPoster',
            'interaction_data': 'artComment',
            'interaction_count': lambda d: d.get('artComment', '').astype(str).str.len(),
            'platform_specific': lambda d: d.apply(
                lambda x: json.dumps({'e_ip': x.get('e_ip', ''), 'insertedDate': str(x.get('insertedDate', ''))}, ensure_ascii=False), axis=1
            )
        }
        return self._create_standardized_df(df, 'PTT', schema)

    def _process_dcard_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        處理DCARD資料為統一格式 (重構版)
        """
        schema = {
            'system_id': lambda d: 'DCARD_' + d['system_id'].astype(str),
            'dataSource': 'DCARD',
            'artUrl': 'artUrl',
            'artDate': lambda d: pd.to_datetime(d['artDate'], errors='coerce'),
            'artTitle': lambda d: d['artTitle'].fillna(''),
            'artContent': lambda d: d['artContent'].fillna(''),
            'category': 'boardID',
            'user_info': lambda d: d.get('department', '').astype(str).fillna('') + '_' + 
                                   d.get('gender', '').astype(str).fillna('') + '_' + 
                                   d.get('school', '').astype(str).fillna(''),
            'interaction_data': '',
            'interaction_count': lambda d: pd.to_numeric(d.get('commentCount', 0), errors='coerce').fillna(0),
            'platform_specific': lambda d: d.apply(
                lambda x: json.dumps({'gender': x.get('gender', ''), 'school': x.get('school', ''), 'department': x.get('department', '')}, ensure_ascii=False), axis=1
            )
        }
        return self._create_standardized_df(df, 'DCARD', schema)
        
    def _filter_by_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根據設定的時間區間篩選資料
        """
        df_filtered = df.dropna(subset=['artDate']).copy()
        start_date = pd.to_datetime(self.config.START_DATE)
        end_date = pd.to_datetime(self.config.END_DATE)
        
        original_count = len(df_filtered)
        df_filtered = df_filtered[
            (df_filtered['artDate'] >= start_date) & 
            (df_filtered['artDate'] <= end_date)
        ]
        
        filtered_count = len(df_filtered)
        print(f"📅 時間篩選: {original_count:,} -> {filtered_count:,} 筆 "
              f"({start_date.date()} 到 {end_date.date()})")
        
        return df_filtered
    
    def load_data(self) -> pd.DataFrame:
        """
        主要載入方法，整合所有資料處理流程
        """
        data_sources = self.load_multi_source_data()
        
        if not data_sources:
            print("⚠️ 沒有找到有效的資料源，建立示範資料...")
            return self._create_sample_data()
        
        integrated_data = self.create_integrated_dataset(data_sources)
        
        if integrated_data.empty:
            print("⚠️ 資料整合後為空，建立示範資料...")
            return self._create_sample_data()
        
        return integrated_data
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        建立示範資料供測試使用（保留原有功能）
        """
        # (此方法內容不變，故省略以節省篇幅)
        # ... 原有的 _create_sample_data 程式碼 ...
        sample_data = {
            'system_id': [f'SAMPLE_{i:03d}' for i in range(1, 21)],
            'dataSource': ['PTT'] * 10 + ['DCARD'] * 10,
            'artUrl': [f'https://example.com/post_{i}' for i in range(1, 21)],
            'artDate': pd.date_range('2024-07-01', periods=20, freq='D'),
            'artTitle': [
                '巴黎萊雅這款精華液保濕效果超好，吸收快不黏膩！',
                '新款紫熨斗眼霜對撫平細紋很有感，質地清爽',
                '效果普普通通，沒有廣告說的那麼神奇，有點失望',
                '用了會長痘痘，質地太油膩了，不會再回購',
                '玻尿酸成分補水效果一流，整天都不乾燥',
                '乳霜有點厚重，夏天用可能太油',
                '青春密碼精華堅持使用後細紋變淡了',
                '皮膚敏感用了會泛紅，可能不耐受',
                "L'Oréal這個品牌值得信賴，產品品質很好",
                '價格稍微偏高，但效果確實不錯',
                '包裝很精美，送人也很有面子',
                '香味很淡，不會過於濃郁，很舒服',
                '質地輕薄，適合油性皮膚使用',
                '使用一個月後膚質有明顯改善',
                '客服態度很好，解答問題很詳細',
                '快遞包裝很仔細，沒有任何損壞',
                '這個系列的產品都很不錯，會繼續回購',
                '朋友推薦的，用了之後確實很滿意',
                '網路評價很高，實際使用也沒讓人失望',
                '總體來說是一款值得推薦的好產品'
            ],
            'artContent': [
                '巴黎萊雅這款精華液保濕效果超好，吸收快不黏膩！質地很棒，推薦給大家。',
                '新款紫熨斗眼霜對撫平細紋很有感，質地清爽，早晚使用效果更佳。',
                '效果普普通通，沒有廣告說的那麼神奇，有點失望，可能不適合我的膚質。',
                '用了會長痘痘，質地太油膩了，不會再回購，建議油性肌膚慎用。',
                '玻尿酸成分補水效果一流，整天都不乾燥，非常適合乾性肌膚使用。',
                '乳霜有點厚重，夏天用可能太油，但冬天使用感覺剛好。',
                '青春密碼精華堅持使用後細紋變淡了，需要長期使用才能看到效果。',
                '皮膚敏感用了會泛紅，可能不耐受，建議敏感肌先試用。',
                "L'Oréal這個品牌值得信賴，產品品質很好，一直都在使用他們家的產品。",
                '價格稍微偏高，但效果確實不錯，物有所值。',
                '包裝很精美，送人也很有面子，品牌形象很好。',
                '香味很淡，不會過於濃郁，很舒服，適合不喜歡濃香的人。',
                '質地輕薄，適合油性皮膚使用，不會造成負擔。',
                '使用一個月後膚質有明顯改善，會繼續使用下去。',
                '客服態度很好，解答問題很詳細，購物體驗很好。',
                '快遞包裝很仔細，沒有任何損壞，物流很快。',
                '這個系列的產品都很不錯，會繼續回購，值得信賴。',
                '朋友推薦的，用了之後確實很滿意，推薦給其他朋友。',
                '網路評價很高，實際使用也沒讓人失望，名不虛傳。',
                '總體來說是一款值得推薦的好產品，各方面都很滿意。'
            ],
            'category': ['Beauty'] * 20,
            'user_info': [f'User_{i}' for i in range(1, 21)],
            'interaction_data': [''] * 20,
            'interaction_count': np.random.randint(0, 100, 20),
            'platform_specific': ['{}'] * 20
        }
        df = pd.DataFrame(sample_data)
        df = self._optimize_dtypes(df)
        print(f"📊 建立示範資料: {len(df)} 筆")
        return df

# ================= 模組獨立測試區塊 =================
if __name__ == "__main__":
    print("===== 正在獨立測試 DataLoader 模組 =====")
    
    config = Config()
    loader = DataLoader(config)
    
    # 測試載入資料
    df = loader.load_data()
    
    if not df.empty:
        print("\n--- 載入的資料預覽 ---")
        print(df.head())
        print(f"\n資料形狀: {df.shape}")
        print(f"欄位名稱: {list(df.columns)}")
        print(f"資料源分布: {df['dataSource'].value_counts().to_dict()}")
        print(f"記憶體使用: {df.memory_usage().sum():,} bytes")
    else:
        print("\n--- 無法載入任何資料 ---")
    
    print("\n===== DataLoader 模組獨立測試完畢 =====")

