"""Streamlit main entry point for the Weakly-Supervised Dynamic Metric-Learning System."""

from __future__ import annotations

import streamlit as st

from api_client import BackendApiClient

st.set_page_config(
    page_title="New Metric-Learning System",
    page_icon="🎯",
    layout="wide",
)

client = BackendApiClient()

st.title("弱教師あり異常検知：前処理自動分類 ＆ 動的Metric-Learning 統合システム")
st.markdown(
    """
本アプリケーションは、製造業外観検査における **100:2 の極端不均衡データ** を対象に、
**フェーズ1（2段階密度分離クラスタリング＋疑似多クラスアノテーション）** から
**フェーズ2（128次元超球面 非対称MS-Loss ＋ Cross-Batch Memory Bank 双方向Top-K境界反発学習 ＆ 3軸空間評価）**
までを一貫実行する開発・学習基盤です。
"""
)

col1, col2, col3, col4 = st.columns(4)
try:
    health = client.get_health()
    ds = client.get_datasets()
    evals = client.list_evaluations()

    col1.metric("Backend API Status", health.get("status", "unknown").upper())
    col2.metric("Raw Inspection Images", ds.get("total_images", 0))
    col3.metric(
        "Phase 1 Annotations",
        "Confirmed" if ds.get("has_annotations") else ("Clustered" if ds.get("has_clustering_result") else "Pending"),
    )
    col4.metric("Saved Training Epochs", evals.get("total_epochs", 0))
except Exception as exc:
    st.warning(f"Backend API (`{client.base_url}`) への接続待機中: {exc}")

st.divider()
st.subheader("ワークフロー・ナビゲーション（左サイドバーから各ステップを選択）")
st.markdown(
    """
1. **1_clustering_annotation（フェーズ1：前処理自動分類 ＆ ラベル確定）**
   - 検証用100:2合成検査画像のワンクリック生成（または実画像の読み込み）
   - `timm` バックボーン（GeM / GAP）特徴抽出 ＋ **Stage 1 高密度良品コア分離** ＋ **Stage 2 低密度テールPCA & シルエット自動選定K-means**
   - クラスタ確認・良品コアの抜き取りスポットチェック・疑似多クラスラベル確定（`annotations.json` 出力）
2. **2_training_monitor（フェーズ2：非対称MS-Loss ＋ Cross-Batch Memory Bank 学習）**
   - 128次元単位超球面上での非対称マルチクラスMS-Loss学習
   - **Cross-Batch Memory Bank (XBM)** による境界良品 Top-K（絶対件数 / 上位割合 % 指定対応）の動的押し戻しと **エポック毎のメモリ解放（OOM防止）**
   - 学習時デフォルト判定閾値 $\\tau_{\\text{sim}} = 0.0$ での検証 Recall / F1 と高次元最小コサイン距離のリアルタイム監視
3. **3_space_evaluation（Step 6：3軸空間評価 ＆ 動的閾値スライダー）**
   - **コサイン類似度閾値 $\\tau_{\\text{sim}} \\in [-1.0, 1.0]$ スライダー** による Recall / Precision / F1 / 混同行列のリアルタイム再計算
   - **高次元最小コサイン距離ログ** ＋ **Plotly 128次元生ベクトルヒートマップ** ＋ **3D UMAP インタラクティブ散布図**
"""
)
