"""Streamlit main entry point for the Weakly-Supervised Dynamic Metric-Learning System."""

from __future__ import annotations

import pandas as pd
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
本アプリケーションは、製造業外観検査における **極端不均衡データ** を対象に、
**複数データセットの独立保存・切り替え検証** と **学習セッションNo管理（過去履歴・モデルの呼び出し）** を備えた
**フェーズ1（良品固定＋異常クラス自動細分化クラスタリング）** から
**フェーズ2（128次元 Multi-Proxy Anchor Loss ＋ NAB-Sampler ＋ 境界ハードサンプル合成学習 ＆ 3軸空間評価）**
までを一貫実行する開発・学習基盤です。
"""
)

col1, col2, col3, col4, col5 = st.columns(5)
try:
    health = client.get_health()
    catalog = client.get_datasets_catalog()
    active_ds = catalog.get("active_dataset", "default")
    ds = client.get_datasets(dataset_name=active_ds)
    sessions_info = client.list_sessions()
    evals = client.list_evaluations()

    col1.metric("Backend API Status", health.get("status", "unknown").upper())
    col2.metric("アクティブデータセット", f"{active_ds} (全{catalog.get('total_datasets', 1)}種)")
    col3.metric("現在のデータセット画像数", f"{ds.get('total_images', 0)} 枚")
    col4.metric(
        "Phase 1 アノテーション",
        "Confirmed" if ds.get("has_annotations") else ("Clustered" if ds.get("has_clustering_result") else "Pending"),
    )
    col5.metric(
        "保存済み学習セッション数",
        f"{sessions_info.get('total_sessions', 0)} セッション",
    )

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("##### 📂 保存済みデータセット一覧")
        ds_rows = [
            {
                "状態": "✅ Active" if d.get("is_active") else "",
                "データセット名": d.get("dataset_name"),
                "登録画像数": d.get("total_images", 0),
                "サブフォルダ内訳": str(d.get("subfolder_classes", {})),
                "クラスタリング": "済" if d.get("has_clustering_result") else "未",
                "アノテーション": "済" if d.get("has_annotations") else "未",
            }
            for d in catalog.get("datasets", [])
        ]
        if ds_rows:
            st.dataframe(pd.DataFrame(ds_rows), use_container_width=True, hide_index=True, key="main_ds_catalog_df")

    with c_right:
        st.markdown("##### 🗂️ 学習セッション履歴（最新順）")
        sess_rows = [
            {
                "Session No": f"#{s['session_number']:04d} ({s['session_id']})",
                "データセット": s.get("dataset_name", "default"),
                "Loss": s.get("loss_type", ""),
                "Epochs": f"{s.get('epochs_completed', 0)}/{s.get('total_epochs', 0)}",
                "Best Recall": f"{s.get('best_val_recall', 0.0):.4f}",
                "Best F1": f"{s.get('best_val_f1', 0.0):.4f}",
                "状態": s.get("status", ""),
            }
            for s in sessions_info.get("sessions", [])
        ]
        if sess_rows:
            st.dataframe(pd.DataFrame(sess_rows), use_container_width=True, hide_index=True, key="main_sessions_df")
        else:
            st.caption("まだ学習セッション履歴がありません。")

except Exception as exc:
    st.warning(f"Backend API (`{client.base_url}`) への接続待機中: {exc}")

st.divider()
st.subheader("ワークフロー・ナビゲーション（左サイドバーから各ステップを選択）")
st.markdown(
    """
1. **1_clustering_annotation（フェーズ1：データセット管理・フォルダアップロード ＆ 異常クラス自動細分化クラスタリング）**
   - **複数データセットの分けて保存・ワンクリック切り替え・別名複製**
   - フォルダ一括アップロード（サブフォルダ名＝初期クラス名）または合成検査データの生成
   - 良品クラスは `Cluster 0` に固定保持し、異常クラスのみを PCA ＋ シルエット自動選定 K-means で複数サブクラスへ分割（`annotations.json` 出力）
2. **2_training_monitor（フェーズ2：Multi-Proxy Anchor Loss 学習 ＆ セッションNo履歴呼び出し）**
   - 学習ごとに **`Session #0001`, `Session #0002`, ...** を自動採番してモデル（`best_model.pt`）と全エポック履歴を永続保管
   - 過去の任意のセッションNoをUI上から呼び出して、学習曲線・分散推移・設定パラメータをいつでも振り返り比較可能
3. **3_space_evaluation（Step 6：3D UMAP 分散推移・ポップアップ画像インスペクタ・128次元生ヒートマップ）**
   - **1. 高次元定量的マージン ＆ 混同行列（クリック＆矢印送りポップアップ画像確認付き）**
   - **2. 128次元 → 3D UMAP 立体散布図（学習回数 Epoch に応じたクラスタ分散・分離度進化グラフ ＆ 点選択ポップアップ画像確認付き）**
   - **3. Plotly 128次元生ベクトル活性化ヒートマップ**
"""
)
