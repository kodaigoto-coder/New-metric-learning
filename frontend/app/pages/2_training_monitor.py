"""Phase 2 UI: 50:50 Power-Balanced Multi-Proxy Loss Training Monitor & Numbered Session History Recall."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api_client import BackendApiClient

st.set_page_config(page_title="Phase 2: Dynamic Metric-Learning", layout="wide")
client = BackendApiClient()


def _render_training_history(
    container: Any,
    history: list[dict[str, Any]],
    tau_sim: float,
    render_seq: int,
    prefix: str = "live",
) -> None:
    """Render training curves and XBM audit table with unique element keys to prevent duplicate ID errors."""
    if not history:
        return
    df_hist = pd.DataFrame(history)
    with container.container():
        c_left, c_mid, c_right = st.columns(3)
        loss_cols = [
            col
            for col in ["train_loss", "loss_centripetal", "loss_repel", "curriculum_progress"]
            if col in df_hist.columns
        ]
        fig_loss = px.line(
            df_hist,
            x="epoch",
            y=loss_cols,
            markers=True,
            title="Loss推移 (Total / ①良品求心力 loss_centripetal / ②不良斥力 loss_repel)",
        )
        c_left.plotly_chart(
            fig_loss,
            use_container_width=True,
            key=f"{prefix}_train_loss_chart_{render_seq}",
        )

        fig_rec = px.line(
            df_hist,
            x="epoch",
            y=[col for col in ["val_recall", "val_f1", "val_precision"] if col in df_hist.columns],
            markers=True,
            title=f"検証 Recall / F1 / Precision (@ τ_sim={tau_sim:.2f})",
        )
        c_mid.plotly_chart(
            fig_rec,
            use_container_width=True,
            key=f"{prefix}_train_recall_chart_{render_seq}",
        )

        margin_cols = [
            col
            for col in ["d_min_boundary", "d_min_centroid", "normal_intra_disp", "inter_class_dist"]
            if col in df_hist.columns
        ]
        fig_margin = px.line(
            df_hist,
            x="epoch",
            y=margin_cols,
            markers=True,
            title="高次元最小コサイン距離 & 良品クラスタ内分散 (normal_intra_disp) 推移",
        )
        c_right.plotly_chart(
            fig_margin,
            use_container_width=True,
            key=f"{prefix}_train_margin_chart_{render_seq}",
        )

        st.markdown("##### エポック別ログ（50:50 求心力・斥力内訳 ＆ クラス内分散・XBMメモリ解放監査）")
        display_cols = [
            col
            for col in [
                "epoch",
                "train_loss",
                "loss_centripetal",
                "loss_repel",
                "normal_loss_ratio",
                "val_recall",
                "val_f1",
                "val_precision",
                "normal_intra_disp",
                "defect_intra_disp",
                "inter_class_dist",
                "fisher_separation_ratio",
                "d_min_boundary",
                "d_min_centroid",
                "effective_top_k",
                "xbm_released_entries",
            ]
            if col in df_hist.columns
        ]
        st.dataframe(
            df_hist[display_cols],
            use_container_width=True,
            key=f"{prefix}_train_hist_df_{render_seq}",
        )


st.title("フェーズ2：50:50 パワーバランス深層距離学習 ＆ セッションNo履歴管理")

# Fetch dataset catalog & active dataset
try:
    catalog_data = client.get_datasets_catalog()
except Exception:
    catalog_data = {"active_dataset": "default", "datasets": []}

active_dataset_name: str = catalog_data.get("active_dataset", "default")
dataset_names_list: list[str] = [d["dataset_name"] for d in catalog_data.get("datasets", [])] or ["default"]

if (
    st.session_state.get("_synced_active_ds") != active_dataset_name
    or st.session_state.get("phase2_sidebar_ds_select") not in dataset_names_list
):
    st.session_state["_synced_active_ds"] = active_dataset_name
    st.session_state["phase2_sidebar_ds_select"] = active_dataset_name


def _on_phase2_ds_change() -> None:
    chosen = st.session_state.get("phase2_sidebar_ds_select")
    if chosen and chosen != st.session_state.get("_synced_active_ds"):
        client.select_or_create_dataset(dataset_name=str(chosen))
        st.session_state["_synced_active_ds"] = ""


with st.sidebar:
    st.markdown("### 📂 アクティブデータセット切替")
    st.selectbox(
        "学習対象データセット",
        options=dataset_names_list,
        key="phase2_sidebar_ds_select",
        on_change=_on_phase2_ds_change,
    )
    st.caption(f"現在のアクティブDB: **`{active_dataset_name}`**")

try:
    ann = client.get_annotations(dataset_name=active_dataset_name)
    st.info(
        f"📂 **学習対象データセット: `{active_dataset_name}`** ｜ "
        f"確定済みアノテーション (`annotations.json`): 総数 **{ann['total_samples']} 枚** ｜ "
        f"良品クラス: `{ann.get('normal_class_name', 'normal')}` (`label_id=0`) ｜ "
        f"学習クラス内訳: `{ann['class_counts']}` ｜ "
        f"クラスID割り当て: `{ann.get('class_map', {})}`"
    )
except Exception:
    st.warning(
        f"データセット `{active_dataset_name}` にはまだ確定済みアノテーションがありません。"
        "先に `1_clustering_annotation` でラベルを確定してください。"
    )

# Section A: New Training Run Form
with st.form("training_config_form"):
    st.subheader(f"1. 50:50 パワーバランス Loss 設定 ＆ 新規学習セッション実行（対象: `{active_dataset_name}`）")
    st.latex(
        r"L_{\text{total}} = \underbrace{\lambda_{\text{norm}} \frac{1}{|\mathcal{B}_{\text{norm}}|} \sum_{x \in \mathcal{B}_{\text{norm}}} \left(1 - S(x, p_{\text{norm}})\right)}_{\text{① 良品の求心力（基準 50\%: }\lambda_{\text{norm}}=1.0\text{）}} + \underbrace{\lambda_{\text{anom}} \frac{1}{|\mathcal{B}_{\text{anom}}|} \sum_{x \in \mathcal{B}_{\text{anom}}} \tilde{w}(x) \max\left(0, S(x, p_{\text{norm}}) - \delta\right)}_{\text{② 不良の斥力（基準 50\%: }\lambda_{\text{anom}}=1.0\text{）}}"
    )

    p_col1, p_col2, p_col3 = st.columns([1.8, 1.3, 1.3])
    normal_loss_ratio = p_col1.slider(
        "⚖️ 良品求心力 : 不良斥力 のパワーバランス比率 (Normal Ratio)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="0.50（デフォルト）で 良品求心力 50% (λ_norm=1.0) : 不良斥力 50% (λ_anom=1.0) の対称スケールになります。良品の分散をより強く絞りたい場合は 0.55〜0.70 へ上げてください。",
    )
    repel_delta = p_col2.slider(
        "🎯 不良斥力コサインマージン δ (Target Margin)",
        min_value=-0.50,
        max_value=0.50,
        value=0.00,
        step=0.05,
        help="max(0, S(x, p_norm) - δ) の目標境界 δ です。デフォルト 0.00（良品プロキシと直交・90度）で [0.0, 1.0] の対称物差しになります。",
    )
    nab_normal_ratio = p_col3.slider(
        "📦 ミニバッチ内 良品サンプル比率 (NAB-Sampler)",
        min_value=0.30,
        max_value=0.70,
        value=0.50,
        step=0.05,
        help="ミニバッチ内の良品比率（デフォルト 0.50 = 良品50% : 全不良モード均等50%）。",
    )

    r_memo_c1, r0c1, r0c2, r0c3 = st.columns([1.6, 1.4, 1.0, 1.0])
    session_label = r_memo_c1.text_input(
        "セッションメモ / 実験名（任意）",
        value="",
        placeholder="例: 50:50バランス δ=0.0 M=1",
        help="学習開始時に自動で Session #0001, #0002 ... が採番されます。",
    )
    loss_type = r0c1.selectbox(
        "距離学習アルゴリズム (Loss Type)",
        options=["multi_proxy_anchor", "ms_loss"],
        index=0,
        help="multi_proxy_anchor (推奨): 50:50 パワーバランスのコサイン求心力・斥力 Loss を使用します。",
    )
    num_normal_proxies = r0c2.number_input(
        "良品プロキシ数 (M)",
        min_value=1,
        max_value=12,
        value=1,
        step=1,
        help="単一中心に強く凝集させる場合は M=1（推奨）、複数ロットの広がりを許容する場合は M=2〜4 に設定します。",
    )
    hard_anomaly_boost = r0c3.number_input(
        "Top-K 危険不良 相対優先度 (λ_hard)",
        min_value=1.0,
        max_value=20.0,
        value=3.0,
        step=0.5,
        help="不良バッチ内部での相対的な優先度です（平均1.0に自動正規化されるため 50:50 の全体比率は崩れません）。",
    )

    st.markdown("#### 2. バックボーン・学習ハイパーパラメータ")
    r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns([1.4, 1.1, 1.0, 1.0, 1.2])
    backbone_name = r1c1.selectbox(
        "バックボーン",
        options=["efficientnet_b3", "vit_base_patch16_224", "resnet18", "lightweight_cnn"],
        index=0,
    )
    pooling_type = r1c2.selectbox("プーリング層", options=["gem", "avg_max", "gap"], index=0)
    epochs = r1c3.number_input("学習エポック数", min_value=2, max_value=100, value=20, step=1)
    batch_size = r1c4.number_input("ミニバッチサイズ", min_value=8, max_value=128, value=32, step=4)
    use_pretrained = r1c5.checkbox(
        "ImageNet事前学習重みを使用",
        value=True,
    )

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    learning_rate = r2c1.number_input(
        "学習率 (Learning Rate)", min_value=1e-5, max_value=5e-2, value=2e-3, format="%.4f"
    )
    tau_sim = r2c2.slider(
        "評価基準コサイン類似度閾値 (τ_sim)",
        min_value=-0.80,
        max_value=0.80,
        value=0.0,
        step=0.05,
        help="デフォルト 0.0（良品プロキシと直交）。学習後の評価画面でもスライダーで自在に変更可能です。",
    )
    w_defect = r2c3.number_input(
        "不良クラス基準重み (w_defect)", min_value=0.1, max_value=10.0, value=1.0, step=0.1
    )
    w_aux = r2c4.number_input(
        "プロキシ直交補助重み (w_ortho)", min_value=0.0, max_value=2.0, value=0.2, step=0.05
    )

    st.markdown("#### 3. 境界ハードサンプル合成（Defect Transfer ＆ 片方向 Anomalous Mixup / Phase 3）")
    b_c1, b_c2, b_c3, b_c4 = st.columns([1.5, 1.1, 1.1, 1.1])
    enable_boundary_aug = b_c1.checkbox(
        "Defect Transfer ＆ 片方向 Anomalous Mixup を有効化",
        value=True,
        help="残差サリエンシーによる欠陥転写（Defect Transfer）と境界補間（Anomalous Mixup）を生成し、緩やかな重み（0.25x）で境界をガードします。",
    )
    mixup_alpha_min = b_c2.number_input(
        "Anomalous Mixup 最小良品比率 (α_min)", min_value=0.30, max_value=0.85, value=0.55, step=0.05
    )
    mixup_alpha_max = b_c3.number_input(
        "Anomalous Mixup 最大良品比率 (α_max)", min_value=0.35, max_value=0.90, value=0.75, step=0.05
    )
    class_balance_beta = b_c4.number_input(
        "Class-Balanced Weight β", min_value=0.80, max_value=0.9999, value=0.99, format="%.4f"
    )

    st.markdown("#### 4. Top-K ハードマイニング ＆ Cross-Batch Memory Bank (XBM) カリキュラム設定")
    r3c1, r3c2, r3c3, r3c4, r3c5 = st.columns(5)
    top_k_mode = r3c1.selectbox(
        "Top-K 指定モード",
        options=["count", "ratio"],
        index=1,
    )
    default_init = 30.0 if top_k_mode == "count" else 0.30
    default_final = 20.0 if top_k_mode == "count" else 0.15
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
    gamma_initial = r3c4.number_input("初期反発係数 (γ_init)", min_value=0.1, max_value=5.0, value=0.6)
    gamma_final = r3c5.number_input("後期反発係数 (γ_final)", min_value=0.1, max_value=10.0, value=1.5)

    submitted = st.form_submit_button(
        f"🚀 新規セッションNoを採番して学習を開始 (対象: `{active_dataset_name}` ｜ 求心力 {int(round(normal_loss_ratio*100))}% : 斥力 {int(round((1-normal_loss_ratio)*100))}%)",
        type="primary",
        use_container_width=True,
    )

if submitted:
    job = client.trigger_training(
        {
            "dataset_name": active_dataset_name,
            "session_label": session_label.strip() or None,
            "loss_type": loss_type,
            "backbone_name": backbone_name,
            "pooling_type": pooling_type,
            "pretrained": bool(use_pretrained),
            "freeze_backbone": True,
            "unfreeze_last_block": False,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "tau_sim": float(tau_sim),
            "normal_loss_ratio": float(normal_loss_ratio),
            "repel_delta": float(repel_delta),
            "num_normal_proxies": int(num_normal_proxies),
            "hard_anomaly_boost": float(hard_anomaly_boost),
            "class_balance_beta": float(class_balance_beta),
            "enable_boundary_aug": bool(enable_boundary_aug),
            "mixup_alpha_min": float(mixup_alpha_min),
            "mixup_alpha_max": float(mixup_alpha_max),
            "nab_normal_ratio": float(nab_normal_ratio),
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
    assigned_session_id = job.get("session_id", "")
    assigned_session_num = job.get("session_number")
    st.info(
        f"🗂️ **学習セッション採番完了: `Session #{assigned_session_num:04d}` (`{assigned_session_id}`)** ｜ "
        f"データセット: `{active_dataset_name}` ｜ パワーバランス: 良品求心力 {int(round(normal_loss_ratio*100))}% : 不良斥力 {int(round((1-normal_loss_ratio)*100))}%"
        if assigned_session_num
        else f"学習ジョブ `{job_id}` を開始しました..."
    )
    prog = st.progress(0.0, text="学習ジョブを開始しました...")
    metrics_placeholder = st.empty()
    last_rendered_len = 0
    render_seq = 0

    while True:
        status = client.get_job_status(job_id)
        prog.progress(float(status.get("progress", 0.0)), text=status.get("message", ""))
        history = status.get("history", [])
        is_done = status["status"] in ("completed", "failed")

        if history and (len(history) > last_rendered_len or is_done):
            last_rendered_len = len(history)
            render_seq += 1
            _render_training_history(
                container=metrics_placeholder,
                history=history,
                tau_sim=float(tau_sim),
                render_seq=render_seq,
                prefix="live",
            )

        if is_done:
            break
        time.sleep(0.6)

    if status["status"] == "completed":
        st.success(
            f"✅ 学習完了！ `{assigned_session_id}` にモデル (`best_model.pt`) と全エポック履歴を保存しました。"
            "左サイドバーの `3_space_evaluation` または下の「過去の学習セッション履歴の呼び出し」からいつでも振り返れます。"
        )
    else:
        st.error(f"学習エラー: {status.get('error')}")

# Section B: Past Training Sessions Recall & Comparison
st.divider()
st.subheader("🗂️ 保存済み学習セッション履歴の呼び出し・振り返り (Session History Recall)")
st.caption(
    "これまでに実行した全ての学習セッション（`Session #0001`, `Session #0002`, ...）がモデル・ハイパーパラメータ・エポック履歴とともに保管されています。"
    "UI上から任意のセッションNoを選択して当時の学習曲線と最高到達精度を振り返ることができます。"
)

try:
    sessions_resp = client.list_sessions()
    all_sessions: list[dict[str, Any]] = sessions_resp.get("sessions", [])
except Exception as exc:
    all_sessions = []
    st.warning(f"セッション一覧の取得に失敗しました: {exc}")

if not all_sessions:
    st.info("まだ保管された学習セッションがありません。上のフォームから学習を実行すると `Session #0001` から順に自動保管されます。")
else:
    df_sessions = pd.DataFrame(
        [
            {
                "Session No": f"Session #{s['session_number']:04d}",
                "Session ID": s["session_id"],
                "メモ / ラベル": s.get("session_label", ""),
                "データセット": s.get("dataset_name", "default"),
                "Loss": s.get("loss_type", ""),
                "バックボーン": s.get("backbone_name", ""),
                "完了エポック": f"{s.get('epochs_completed', 0)}/{s.get('total_epochs', 0)}",
                "Best Epoch": s.get("best_epoch", 0),
                "Best Recall": f"{s.get('best_val_recall', 0.0):.4f}",
                "Best F1": f"{s.get('best_val_f1', 0.0):.4f}",
                "Best Precision": f"{s.get('best_val_precision', 0.0):.4f}",
                "Best d_min (境界)": f"{s.get('best_d_min_boundary', 0.0):.4f}",
                "状態": s.get("status", ""),
                "実行日時": str(s.get("created_at", ""))[:19].replace("T", " "),
            }
            for s in all_sessions
        ]
    )
    st.dataframe(df_sessions, use_container_width=True, hide_index=True, key="all_sessions_summary_df")

    session_options = [s["session_id"] for s in all_sessions]
    session_labels_map = {
        s["session_id"]: (
            f"Session #{s['session_number']:04d} ({s['session_id']}) ｜ "
            f"Dataset: {s.get('dataset_name', 'default')} ｜ "
            f"Loss: {s.get('loss_type', '')} ｜ "
            f"Best Recall={s.get('best_val_recall', 0.0):.3f}, F1={s.get('best_val_f1', 0.0):.3f} "
            f"({s.get('session_label', '')})"
        )
        for s in all_sessions
    }

    selected_recall_session = st.selectbox(
        "🔍 履歴・学習曲線を呼び出すセッションNoを選択",
        options=session_options,
        format_func=lambda sid: session_labels_map.get(sid, sid),
        index=0,
        key="recall_session_selector",
    )

    if selected_recall_session:
        try:
            s_detail = client.get_session_detail(selected_recall_session)
            s_hist = s_detail.get("history", [])
            s_cfg = s_detail.get("config", {})

            sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
            sc1.metric("呼び出しセッション", f"#{s_detail.get('session_number', 1):04d}")
            sc2.metric("対象データセット", s_detail.get("dataset_name", "default"))
            sc3.metric("最高到達 Recall", f"{s_detail.get('best_val_recall', 0.0):.4f}", f"Best Epoch {s_detail.get('best_epoch', 1)}")
            sc4.metric("最高到達 F1", f"{s_detail.get('best_val_f1', 0.0):.4f}")
            sc5.metric("最高 境界最小距離 d_min", f"{s_detail.get('best_d_min_boundary', 0.0):.4f}")
            sc6.metric("完了エポック数", f"{s_detail.get('epochs_completed', len(s_hist))} ep")

            with st.expander(f"📋 `{selected_recall_session}` の学習ハイパーパラメータ設定を確認", expanded=False):
                st.json(s_cfg if s_cfg else {"loss_type": s_detail.get("loss_type"), "backbone_name": s_detail.get("backbone_name")})

            recall_container = st.empty()
            _render_training_history(
                container=recall_container,
                history=s_hist,
                tau_sim=float(s_cfg.get("tau_sim", 0.0)) if isinstance(s_cfg, dict) else 0.0,
                render_seq=int(s_detail.get("session_number", 1)),
                prefix=f"recall_{selected_recall_session}",
            )
        except Exception as exc:
            st.error(f"セッション `{selected_recall_session}` の詳細読み込みに失敗しました: {exc}")
