"""Step 6 UI: 3-Axis Space Evaluation (Quantitative Margin, 128-D Raw Heatmap, 3D UMAP) & Interactive Threshold Slider."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api_client import BackendApiClient

st.set_page_config(page_title="Step 6: 3-Axis Space Evaluation", layout="wide")
client = BackendApiClient()

st.title("Step 6：128次元空間 3軸多角評価 ＆ コサイン類似度閾値 (τ_sim) 動的シミュレーション")

try:
    eval_list = client.list_evaluations()
    epochs_meta = eval_list.get("epochs", [])
except Exception as exc:
    epochs_meta = []
    st.error(f"評価データ一覧の取得に失敗しました: {exc}")

if not epochs_meta:
    st.info("保存済みのエポックスナップショット (`.pkl`) がありません。先に `2_training_monitor` で学習を実行してください。")
else:
    epoch_numbers = [int(ep["epoch"]) for ep in epochs_meta]
    ctrl_c1, ctrl_c2 = st.columns([1, 2])
    selected_epoch = ctrl_c1.selectbox(
        "評価対象エポックを選択 (.pkl スナップショット)",
        options=epoch_numbers,
        index=len(epoch_numbers) - 1,
    )
    tau_sim = ctrl_c2.slider(
        "判定閾値: 良品中心とのコサイン類似度 τ_sim (s(x) < τ_sim を不良 NG と判定)",
        min_value=-1.0,
        max_value=1.0,
        value=0.0,
        step=0.02,
        help="学習時のデフォルト閾値は 0.0（直交）です。スライダーを動かすと即座に Recall / F1 と混同行列が再計算されます。",
    )

    eval_data = client.get_epoch_evaluation(epoch=int(selected_epoch), tau_sim=float(tau_sim))
    val_m = eval_data["val_metrics"]
    all_m = eval_data["all_metrics"]

    # Axis 1: Quantitative High-Dimensional Cosine Margin & Recall/F1 Metrics
    st.subheader("1. 【最重要基準①】高次元定量的マージン ＆ 閾値連動 Recall / F1 スコア")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("検証 Recall (再現率)", f"{val_m['recall']:.4f}", help="不良を見落とさなかった割合")
    k2.metric("検証 F1 スコア", f"{val_m['f1']:.4f}")
    k3.metric("検証 Precision", f"{val_m['precision']:.4f}")
    k4.metric("全データ Recall", f"{all_m['recall']:.4f}")
    k5.metric("境界良品〜不良 最小Cos距離", f"{eval_data['d_min_boundary']:.4f}")
    k6.metric("良品中心〜不良 最小Cos距離", f"{eval_data['d_min_centroid']:.4f}")

    cm_col, curve_col, hist_col = st.columns(3)
    with cm_col:
        st.markdown(f"**混同行列（全データ @ $\\tau_{{\\text{{sim}}}} = {tau_sim:.2f}$）**")
        cm_df = pd.DataFrame(
            [
                {"Actual": "Actual Defect (NG)", "Pred Defect (NG)": all_m["tp"], "Pred Normal (OK)": all_m["fn"]},
                {"Actual": "Actual Normal (OK)", "Pred Defect (NG)": all_m["fp"], "Pred Normal (OK)": all_m["tn"]},
            ]
        ).set_index("Actual")
        st.table(cm_df)
        st.caption(
            f"TP(不良検出)={all_m['tp']} / FN(不良見落とし)={all_m['fn']} / FP(過検出)={all_m['fp']} / TN(正常合格)={all_m['tn']}"
        )

    with curve_col:
        df_curve = pd.DataFrame(eval_data["threshold_curve"])
        fig_curve = px.line(
            df_curve,
            x="tau_sim",
            y=["recall", "precision", "f1"],
            title="コサイン類似度閾値 (τ_sim) 対 Recall / Precision / F1 曲線",
        )
        fig_curve.add_vline(x=float(tau_sim), line_dash="dash", line_color="red", annotation_text=f"τ={tau_sim:.2f}")
        st.plotly_chart(fig_curve, use_container_width=True)

    with hist_col:
        df_samples = pd.DataFrame(eval_data["samples"])
        fig_hist = px.histogram(
            df_samples,
            x="cosine_similarity",
            color="label_name",
            nbins=30,
            barmode="overlay",
            title="良品中心とのコサイン類似度分布 s(x) ∈ [-1, 1]",
        )
        fig_hist.add_vline(x=float(tau_sim), line_dash="dash", line_color="red", annotation_text=f"τ={tau_sim:.2f}")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # Axis 2: Plotly 128-D Raw Vector Heatmap
    st.subheader("2. 【最重要基準②】Plotly 128次元生ベクトル活性化ヒートマップ（次元圧縮なし）")
    hm_matrix = eval_data["heatmap_matrix"]
    hm_labels = eval_data["heatmap_labels"]
    fig_hm = go.Figure(
        data=go.Heatmap(
            z=hm_matrix,
            x=[f"D{d}" for d in range(len(hm_matrix[0]))] if hm_matrix else [],
            y=hm_labels,
            colorscale="RdBu_r",
            zmin=-0.35,
            zmax=0.35,
            colorbar=dict(title="Activation"),
        )
    )
    fig_hm.update_layout(
        title="良品群 vs 各不良モードの128次元単位超球面生ベクトル比較",
        xaxis_title="Embedding Dimension (0 .. 127)",
        yaxis_title="Sample (Label & ID)",
        height=520,
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    st.divider()

    # Axis 3: Interactive 3D UMAP Scatter Plot
    st.subheader("3. 【補助基準】128次元 → 3D UMAP インタラクティブ立体散布図")
    fig_umap = px.scatter_3d(
        df_samples,
        x="umap_x",
        y="umap_y",
        z="umap_z",
        color="label_name",
        symbol="predicted_defect",
        hover_data=["file_name", "cosine_similarity", "cosine_distance", "split"],
        title=f"Epoch {selected_epoch}: 128次元超球面空間の 3D UMAP 投影（ドラッグで360度回転可能）",
        height=640,
    )
    st.plotly_chart(fig_umap, use_container_width=True)

    st.divider()
    st.subheader("4. 単発画像推論テスト (`POST /api/v1/inference`)")
    sample_paths = [s["image_path"] for s in eval_data["samples"]]
    selected_img = st.selectbox("テスト対象画像を選択", options=sample_paths)
    if st.button("選択した画像で単発推論を実行"):
        inf_res = client.run_single_inference(image_path=selected_img, tau_sim=float(tau_sim))
        st.json(
            {
                "image_path": inf_res["image_path"],
                "prediction": inf_res["prediction"],
                "cosine_similarity": inf_res["cosine_similarity"],
                "cosine_distance": inf_res["cosine_distance"],
                "tau_sim": inf_res["tau_sim"],
            }
        )
