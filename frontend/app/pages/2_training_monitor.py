"""Phase 2 UI: Asymmetric Multi-Similarity Loss + Cross-Batch Memory Bank Training Monitor."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api_client import BackendApiClient

st.set_page_config(page_title="Phase 2: Dynamic Metric-Learning", layout="wide")
client = BackendApiClient()

st.title("フェーズ2：128次元 非対称MS-Loss ＋ Cross-Batch Memory Bank 動的反発学習")

try:
    ann = client.get_annotations()
    st.info(
        f"確定済みアノテーション (`annotations.json`): 総数 **{ann['total_samples']} 枚** ｜ "
        f"良品クラス: `{ann.get('normal_class_name', 'normal')}` (`label_id=0`) ｜ "
        f"学習クラス内訳（サブフォルダ名）: `{ann['class_counts']}` ｜ "
        f"クラスID割り当て: `{ann.get('class_map', {})}`"
    )
except Exception:
    st.warning("まだフェーズ1のアノテーションが確定していません。先に `1_clustering_annotation` でラベルを確定してください。")

with st.form("training_config_form"):
    st.subheader("学習・カリキュラム・Cross-Batch Memory Bank パラメータ設定 (Step 4〜5)")
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    backbone_name = r1c1.selectbox(
        "バックボーン",
        options=["efficientnet_b3", "vit_base_patch16_224", "resnet18", "lightweight_cnn"],
        index=0,
    )
    pooling_type = r1c2.selectbox("プーリング層", options=["gem", "avg_max", "gap"], index=0)
    epochs = r1c3.number_input("学習エポック数", min_value=2, max_value=100, value=8, step=1)
    batch_size = r1c4.number_input("ミニバッチサイズ", min_value=8, max_value=128, value=24, step=4)

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    learning_rate = r2c1.number_input(
        "学習率 (Learning Rate)", min_value=1e-5, max_value=5e-2, value=2e-3, format="%.4f"
    )
    tau_sim = r2c2.slider(
        "学習時基準コサイン類似度閾値 (τ_sim)",
        min_value=-0.80,
        max_value=0.80,
        value=0.0,
        step=0.05,
        help="デフォルト 0.0（良品中心と直交）。学習後の評価画面で任意に変更可能です。",
    )
    w_defect = r2c3.number_input(
        "不良クラス反発重み (w_defect)", min_value=1.0, max_value=20.0, value=2.5, step=0.5
    )
    w_aux = r2c4.number_input(
        "異種不良モード間分離重み (w_aux)", min_value=0.1, max_value=5.0, value=0.5, step=0.1
    )

    st.markdown("#### Cross-Batch Memory Bank (XBM) ＆ Top-K 境界良品押し戻し設定")
    r3c1, r3c2, r3c3, r3c4, r3c5 = st.columns(5)
    top_k_mode = r3c1.selectbox(
        "Top-K 指定モード",
        options=["count", "ratio"],
        index=0,
        help="count: 絶対件数指定（例: 30件→20件） / ratio: 良品母数に対する上位割合指定（例: 0.25=25% → 0.10=上位10%）",
    )
    default_init = 30.0 if top_k_mode == "count" else 0.25
    default_final = 20.0 if top_k_mode == "count" else 0.10
    top_k_initial = r3c2.number_input(
        "初期 Top-K 値 (学習初期: 広く)",
        min_value=0.01,
        max_value=500.0,
        value=default_init,
    )
    top_k_final = r3c3.number_input(
        "後期 Top-K 値 (学習後期: 絞る)",
        min_value=0.01,
        max_value=500.0,
        value=default_final,
    )
    gamma_initial = r3c4.number_input("初期反発ペナルティ係数 (γ_init)", min_value=0.0, max_value=5.0, value=0.4)
    gamma_final = r3c5.number_input("後期反発ペナルティ係数 (γ_final)", min_value=0.1, max_value=10.0, value=1.5)

    submitted = st.form_submit_button(
        "非対称MS-Loss ＋ XBM 動的反発学習を開始", type="primary", use_container_width=True
    )

if submitted:
    job = client.trigger_training(
        {
            "backbone_name": backbone_name,
            "pooling_type": pooling_type,
            "pretrained": False,
            "freeze_backbone": True,
            "unfreeze_last_block": False,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "tau_sim": float(tau_sim),
            "top_k_mode": top_k_mode,
            "top_k_initial": float(top_k_initial),
            "top_k_final": float(top_k_final),
            "gamma_initial": float(gamma_initial),
            "gamma_final": float(gamma_final),
            "w_defect": float(w_defect),
            "w_aux": float(w_aux),
            "full_snapshot_interval": max(1, int(epochs) // 2),
        }
    )
    job_id = job["job_id"]
    prog = st.progress(0.0, text="学習ジョブを開始しました...")
    metrics_placeholder = st.empty()

    while True:
        status = client.get_job_status(job_id)
        prog.progress(float(status.get("progress", 0.0)), text=status.get("message", ""))
        history = status.get("history", [])
        if history:
            df_hist = pd.DataFrame(history)
            with metrics_placeholder.container():
                c_left, c_mid, c_right = st.columns(3)
                fig_loss = px.line(
                    df_hist,
                    x="epoch",
                    y=["train_loss", "curriculum_progress", "gamma_repel"],
                    markers=True,
                    title="Loss & カリキュラム進捗度 p(t) / 反発係数 γ(t)",
                )
                c_left.plotly_chart(fig_loss, use_container_width=True)

                fig_rec = px.line(
                    df_hist,
                    x="epoch",
                    y=["val_recall", "val_f1", "val_precision"],
                    markers=True,
                    title=f"検証 Recall / F1 / Precision (@ τ_sim={tau_sim:.2f})",
                )
                c_mid.plotly_chart(fig_rec, use_container_width=True)

                fig_margin = px.line(
                    df_hist,
                    x="epoch",
                    y=["d_min_boundary", "d_min_centroid"],
                    markers=True,
                    title="高次元最小コサイン距離 (境界良品対不良 & 良品中心対不良)",
                )
                c_right.plotly_chart(fig_margin, use_container_width=True)

                st.markdown("##### エポック別ログ ＆ Cross-Batch Memory Bank メモリ解放監査")
                st.dataframe(
                    df_hist[
                        [
                            "epoch",
                            "train_loss",
                            "val_recall",
                            "val_f1",
                            "d_min_boundary",
                            "d_min_centroid",
                            "curriculum_progress",
                            "gamma_repel",
                            "effective_top_k",
                            "xbm_released_entries",
                            "xbm_remaining_entries",
                        ]
                    ],
                    use_container_width=True,
                )

        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.6)

    if status["status"] == "completed":
        st.success("学習完了！左サイドバーの `3_space_evaluation` から3軸空間評価と閾値スライダー調整を行えます。")
    else:
        st.error(f"学習エラー: {status.get('error')}")
