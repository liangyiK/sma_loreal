"""
萊雅專案主程式
Main program for L'Oreal project
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from analyzer import LorealAnalysisProject

def main():
    """
    萊雅專案主程式入口點
    """
    print("🎨 歡迎使用萊雅品牌分析系統")
    print("="*50)
    
    try:
        # 建立專案物件
        project = LorealAnalysisProject()
        
        # 執行完整分析流程
        results = project.run_complete_analysis()
        
        print("\n✅ 分析完成！請查看 output 資料夾中的結果。")
        
    except Exception as e:
        print(f"\n❌ 執行過程中發生錯誤: {e}")
        print("請檢查設定檔和資源檔案是否正確配置。")

if __name__ == "__main__":
    main()
