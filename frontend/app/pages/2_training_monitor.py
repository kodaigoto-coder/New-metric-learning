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


_METRIC_TOOLTIP_MAP: dict[str, str] = {
    "train_loss": "学習データ(Train 60%)の総合Loss (①良品求心力 + ②不良斥力 + 直交補助項)。0に近いほど学習データの凝集・分離が進んでいます。",
    "val_loss": "評価データ(Val 20%)の総合Loss。学習に使っていない評価データでのLossであり、次エポックのLoss方向性(求心力vs斥力バランス)へフィードバックされます。",
    "loss_centripetal": "① 良品の求心力Loss (Train): 1 - S(x, p_norm)。0に近いほど良品がプロキシ中心へ凝集しています。",
    "loss_repel": "② 不良の斥力Loss (Train): max(0, S(x, p_norm) - δ)。0に近いほど不良品がマージンδの外側へ押し出されています。",
    "val_loss_centripetal": "① 評価データ(Val 20%)の良品求心力Loss。高い場合は次エポックで求心力比率を引き上げます。",
    "val_loss_repel": "② 評価データ(Val 20%)の不良斥力Loss。高い場合は次エポックで斥力比率を引き上げます。",
    "effective_normal_ratio": "次エポックへ返す動的求心力比率 (Val評価フィードバック反映後)。0.50が50:50均衡、>0.50で求心力強化、<0.50で斥力強化。",
    "val_accuracy": "評価データ(Val 20%)の正解率 Accuracy = (TP+TN)/(TP+TN+FP+FN)。全体のうち良品・不良品を正しく判定できた割合。",
    "val_recall": "評価データ(Val 20%)の不良検出率 Recall = TP/(TP+FN)。不良品を見逃さず検知できた割合（1.0で見逃しゼロ）。",
    "val_precision": "評価データ(Val 20%)の適合率 Precision = TP/(TP+FP)。不良と判定した中で実際に不良だった割合（1.0で過検出ゼロ）。",
    "val_f1": "評価データ(Val 20%)のF1スコア = 2·Precision·Recall/(Precision+Recall)。見逃し防止と過検出抑制の総合バランス指標。",
    "test_accuracy": "テストデータ(Test 20%: 完全未知)の正解率 Accuracy = (TP+TN)/N。カンニングなしの最終総合正解率。",
    "test_recall": "テストデータ(Test 20%: 完全未知)の不良検出率 Recall = TP/(TP+FN)。カンニングなしの最終見逃し防止性能。",
    "test_f1": "テストデータ(Test 20%: 完全未知)のF1スコア。カンニングなしの最終総合識別性能。",
    "d_min_boundary": "境界最小コサイン余裕度 = (良品の最小類似度) - (不良の最大類似度)。0より大きければ良品と不良の分布が完全に分離しています。",
    "d_min_centroid": "中心間最小コサイン距離 = 1 - (不良の平均類似度)。良品プロキシから不良群がどれだけ離れたかを表します（目標 ≥ 1.0）。",
    "normal_intra_disp": "良品クラスタ内分散 σ²_norm = Mean(1 - S(x_norm, p_norm))。小さいほど良品が一点に鋭く凝集しています。",
    "inter_class_dist": "良品プロキシ〜不良群の平均コサイン距離 D_inter。大きいほど不良クラスタが良品から遠くへ離れています。",
}


def _apply_series_hover_tooltips(fig: Any) -> None:
    """Attach explanatory Japanese tooltips to each trace in a Plotly line chart."""
    for trace in fig.data:
        series_name = str(getattr(trace, "name", ""))
        desc = _METRIC_TOOLTIP_MAP.get(series_name, series_name)
        trace.hovertemplate = (
            f"<b>{series_name}</b><br>"
            "Epoch: %{x}<br>"
            "値 (Value): <b>%{y:.4f}</b><br>"
            f"💡 <i>{desc}</i><extra></extra>"
        )


def _render_training_history(
    container: Any,
    history: list[dict[str, Any]],
    tau_sim: float,
    render_seq: int,
    prefix: str = "live",
) -> None:
    """Render training curves and XBM audit table with unique element keys and explanatory tooltips."""
    if not history:
        return
    df_hist = pd.DataFrame(history)
    with container.container():
        c_left, c_mid, c_right = st.columns(3)

        with c_left:
            with st.popover("ℹ️ Loss・次エポック方向性グラフの項目説明 (Tooltip)"):
                st.markdown(
                    """
                    - **`train_loss`**: 学習データ (`train` 60%) に対する総合Loss。エポックとともに低下するほど学習が進んでいます。
                    - **`val_loss`**: 評価データ (`val` 20%) に対する総合Loss。**次エポックのLoss方向性（求心力 vs 斥力バランス）を決める基準**です。
                    - **`loss_centripetal`**: ① 良品の求心力項 $1 - S(x, p_{\\text{norm}})$。`0.0` に近づくほど良品クラスタの分散が小さくなります。
                    - **`loss_repel`**: ② 不良の斥力項 $\\max(0, S(x, p_{\\text{norm}}) - \\delta)$。`0.0` に近づくほど不良品が境界 $\\delta$ の外へ押し出されています。
                    - **`effective_normal_ratio`**: 評価データ (`val`) の結果を受けて次エポックに適用される求心力比率（基準 `0.50`）。
                    """
                )
            loss_cols = [
                col
                for col in [
                    "train_loss",
                    "val_loss",
                    "loss_centripetal",
                    "loss_repel",
                    "effective_normal_ratio",
                ]
                if col in df_hist.columns
            ]
            fig_loss = px.line(
                df_hist,
                x="epoch",
                y=loss_cols,
                markers=True,
                title="① Loss推移 (Train / Val / 求心力 / 斥力 / 次Epoch方向性比率)",
            )
            _apply_series_hover_tooltips(fig_loss)
            st.plotly_chart(
                fig_loss,
                use_container_width=True,
                key=f"{prefix}_train_loss_chart_{render_seq}",
            )

        with c_mid:
            with st.popover("ℹ️ Accuracy / Recall / F1 / Precision グラフの項目説明 (Tooltip)"):
                st.markdown(
                    """
                    - **`val_accuracy` (評価データ正解率)**: 評価データ (`val` 20%) 全体のうち良品・不良品を正しく分類できた割合 $\\frac{TP+TN}{TP+TN+FP+FN}$。
                    - **`test_accuracy` (テストデータ正解率)**: 学習・制御に一切使っていない未知テストデータ (`test` 20%) の最終正解率。
                    - **`val_recall` / `test_recall` (不良検出率)**: 不良品を見逃さず検出できた割合 $\\frac{TP}{TP+FN}$（`1.0` で見逃しゼロ）。
                    - **`val_f1` / `test_f1` (F1スコア)**: Recall（見逃し防止）と Precision（過検出防止）の調和平均。
                    - **`val_precision` (適合率)**: NG判定した中で本当に不良品だった割合 $\\frac{TP}{TP+FP}$。
                    """
                )
            acc_rec_cols = [
                col
                for col in [
                    "val_accuracy",
                    "val_recall",
                    "val_f1",
                    "val_precision",
                    "test_accuracy",
                    "test_recall",
                    "test_f1",
                ]
                if col in df_hist.columns
            ]
            fig_rec = px.line(
                df_hist,
                x="epoch",
                y=acc_rec_cols,
                markers=True,
                title=f"② Accuracy / Recall / F1 / Precision 推移 (@ τ_sim={tau_sim:.2f})",
            )
            _apply_series_hover_tooltips(fig_rec)
            st.plotly_chart(
                fig_rec,
                use_container_width=True,
                key=f"{prefix}_train_recall_chart_{render_seq}",
            )

        with c_right:
            with st.popover("ℹ️ 最小距離マージン & 良品クラスタ内分散グラフの項目説明 (Tooltip)"):
                st.markdown(
                    """
                    - **`d_min_boundary` (境界最小余裕度)**: $(\\text{良品の最小類似度}) - (\\text{不良の最大類似度})$。**`0.0` を超えると良品と不良品の分布に重なりがゼロ（完全分離）**であることを意味します。
                    - **`d_min_centroid` (中心間コサイン距離)**: $1 - \\text{Mean}(S(x_{\\text{anom}}, p_{\\text{norm}}))$。良品中心から不良群がどれだけ離れたかを示します（目標 $\\ge 1.0$：直交以上）。
                    - **`normal_intra_disp` (良品クラスタ内分散 $\\sigma^2_{\\text{norm}}$)**: 良品データのプロキシ周囲でのばらつき。**エポックが進むにつれて `0.0` に向かって下がり続けるのが理想**です。
                    - **`inter_class_dist` (クラス間距離 $D_{\\text{inter}}$)**: 良品中心と不良クラスタ間の平均コサイン距離。上がるほど分離が進んでいます。
                    """
                )
            margin_cols = [
                col
                for col in [
                    "d_min_boundary",
                    "d_min_centroid",
                    "normal_intra_disp",
                    "inter_class_dist",
                ]
                if col in df_hist.columns
            ]
            fig_margin = px.line(
                df_hist,
                x="epoch",
                y=margin_cols,
                markers=True,
                title="③ 高次元最小コサイン距離 & 良品クラスタ内分散 (normal_intra_disp) 推移",
            )
            _apply_series_hover_tooltips(fig_margin)
            st.plotly_chart(
                fig_margin,
                use_container_width=True,
                key=f"{prefix}_train_margin_chart_{render_seq}",
            )

        st.markdown(
            "##### エポック別ログ（6:2:2分割 Val評価フィードバック ＆ Test最終指標・Accuracy・分散監査）"
        )
        display_cols = [
            col
            for col in [
                "epoch",
                "train_loss",
                "val_loss",
                "loss_centripetal",
                "loss_repel",
                "effective_normal_ratio",
                "loss_direction_feedback",
                "val_accuracy",
                "val_recall",
                "val_f1",
                "val_precision",
                "test_accuracy",
                "test_recall",
                "test_f1",
                "test_precision",
                "train_accuracy",
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
    sp_cnts = ann.get("split_counts", {})
    st.info(
        f"📂 **学習対象データセット: `{active_dataset_name}`** ｜ "
        f"総数 **{ann['total_samples']} 枚** ｜ "
        f"**6:2:2 分割**: 🏋️学習 (`train` 60%)=**{sp_cnts.get('train', '-')}枚**, "
        f"📋評価 (`val` 20%: Loss方向性フィードバック)=**{sp_cnts.get('val', '-')}枚**, "
        f"🧪テスト (`test` 20%: 最終性能評価)=**{sp_cnts.get('test', '-')}枚** ｜ "
        f"良品クラス: `{ann.get('normal_class_name', 'normal')}` (`label_id=0`) ｜ "
        f"クラス内訳: `{ann['class_counts']}`"
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
        placeholder="例: 50:50バランス δ=0.0 M=Auto",
        help="学習開始時に自動で Session #0001, #0002 ... が採番されます。",
    )
    loss_type = r0c1.selectbox(
        "距離学習アルゴリズム (Loss Type)",
        options=["multi_proxy_anchor", "ms_loss"],
        index=0,
        help="multi_proxy_anchor (推奨): 50:50 パワーバランスのコサイン求心力・斥力 Loss を使用します。",
    )
    num_normal_proxies = r0c2.number_input(
        "良品プロキシ数 (M: 0=自動判定)",
        min_value=0,
        max_value=12,
        value=0,
        step=1,
        help="0（推奨デフォルト）: 学習用良品データの許容半径 ε とコサイン・シルエット係数から最適な M* を自動決定します（単峰なら M*=1、複数サブクラスタ時は M*=2〜6 に自動分割・Möbius半径較正）。1〜12 で手動固定も可能です。",
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
        "学習率 (Learning Rate)", min_value=1e-5, max_value=5e-2, value=2e-3, format="%.5f"
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

    st.markdown("#### 3. 境界ハードサンプル合成（同座標 Zスコア欠陥転写 ＆ 同一サブクラスタ異常残差注入 / Phase 3）")
    b_c1, b_c2, b_c3, b_c4 = st.columns([1.5, 1.1, 1.1, 1.1])
    enable_boundary_aug = b_c1.checkbox(
        "同座標 Zスコア欠陥転写 ＆ 異常残差注入 を有効化",
        value=True,
        help="非対称ワークの位置情報を保持したまま、同一サブクラスタ内の良品へ同座標（±2px）欠陥パッチ転写と純粋異常残差 Δ_def 注入を行い、正半球の浅い安全網（重み 0.15x）で境界をガードします（異位相間の線形Mixupは行いません）。",
    )
    mixup_alpha_min = b_c2.number_input(
        "境界合成 最小強度係数 (α_min)", min_value=0.30, max_value=0.85, value=0.55, step=0.05
    )
    mixup_alpha_max = b_c3.number_input(
        "境界合成 最大強度係数 (α_max)", min_value=0.35, max_value=0.90, value=0.75, step=0.05
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
    alpha_min_clean = round(min(0.90, max(0.30, float(mixup_alpha_min))), 4)
    alpha_max_clean = round(
        min(0.95, max(0.35, alpha_min_clean + 0.05, float(mixup_alpha_max))), 4
    )
    epochs_clean = max(1, min(200, int(epochs)))
    tk_init_clean = round(max(0.01, float(top_k_initial)), 4)
    tk_final_clean = round(max(0.01, float(top_k_final)), 4)
    if top_k_mode == "count":
        if tk_init_clean < 1.0:
            tk_init_clean = float(max(1, int(round(tk_init_clean * 100.0))))
        if tk_final_clean < 1.0:
            tk_final_clean = float(max(1, int(round(tk_final_clean * 100.0))))
    else:
        if tk_init_clean > 1.0:
            tk_init_clean = round(min(1.0, max(0.01, tk_init_clean / 100.0)), 4)
        if tk_final_clean > 1.0:
            tk_final_clean = round(min(1.0, max(0.01, tk_final_clean / 100.0)), 4)
    try:
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
                "epochs": epochs_clean,
                "batch_size": max(4, min(256, int(batch_size))),
                "learning_rate": round(min(1.0, max(1e-6, float(learning_rate))), 6),
                "tau_sim": round(min(1.0, max(-1.0, float(tau_sim))), 4),
                "normal_loss_ratio": round(min(0.95, max(0.05, float(normal_loss_ratio))), 4),
                "repel_delta": round(min(0.80, max(-0.80, float(repel_delta))), 4),
                "num_normal_proxies": max(0, min(12, int(num_normal_proxies))),
                "hard_anomaly_boost": round(min(20.0, max(1.0, float(hard_anomaly_boost))), 4),
                "class_balance_beta": round(min(0.9999, max(0.50, float(class_balance_beta))), 6),
                "enable_boundary_aug": bool(enable_boundary_aug),
                "mixup_alpha_min": alpha_min_clean,
                "mixup_alpha_max": alpha_max_clean,
                "nab_normal_ratio": round(min(0.75, max(0.25, float(nab_normal_ratio))), 4),
                "top_k_mode": top_k_mode,
                "top_k_initial": tk_init_clean,
                "top_k_final": tk_final_clean,
                "gamma_initial": round(max(0.0, float(gamma_initial)), 4),
                "gamma_final": round(max(0.0, float(gamma_final)), 4),
                "w_defect": round(max(0.1, float(w_defect)), 4),
                "w_aux": round(max(0.0, float(w_aux)), 4),
                "full_snapshot_interval": max(1, epochs_clean // 2),
            }
        )
    except Exception as exc:
        st.error(f"学習ジョブの起動に失敗しました: {exc}")
        st.stop()
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

    poll_failures = 0
    status: dict[str, Any] = {"status": "running"}
    while True:
        try:
            status = client.get_job_status(job_id)
            poll_failures = 0
        except Exception as exc:
            poll_failures += 1
            if poll_failures >= 6:
                st.error(f"ジョブ状態の取得中に通信エラーが継続しました: {exc}")
                st.stop()
            time.sleep(1.0)
            continue
        prog.progress(
            min(1.0, max(0.0, float(status.get("progress", 0.0)))),
            text=status.get("message", ""),
        )
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
                "完了エポック": f"{s.get('epochs_completed', 0)}/{s.get('total_epochs', 0)}",
                "Best Epoch": s.get("best_epoch", 0),
                "Test Accuracy": f"{s.get('best_test_accuracy', s.get('best_val_accuracy', 0.0)):.4f}",
                "Test Recall": f"{s.get('best_test_recall', s.get('best_val_recall', 0.0)):.4f}",
                "Test F1": f"{s.get('best_test_f1', s.get('best_val_f1', 0.0)):.4f}",
                "Val Accuracy": f"{s.get('best_val_accuracy', 0.0):.4f}",
                "Val Recall": f"{s.get('best_val_recall', 0.0):.4f}",
                "Val F1": f"{s.get('best_val_f1', 0.0):.4f}",
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
            f"Test Acc={s.get('best_test_accuracy', s.get('best_val_accuracy', 0.0)):.3f}, "
            f"Test Recall={s.get('best_test_recall', s.get('best_val_recall', 0.0)):.3f}, "
            f"Val F1={s.get('best_val_f1', 0.0):.3f} "
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

            sc1, sc2, sc3, sc4, sc5, sc6, sc7 = st.columns(7)
            sc1.metric("呼び出しセッション", f"#{s_detail.get('session_number', 1):04d}")
            sc2.metric("対象データセット", s_detail.get("dataset_name", "default"))
            sc3.metric(
                "Test Accuracy (テスト正解率)",
                f"{s_detail.get('best_test_accuracy', s_detail.get('best_val_accuracy', 0.0)):.4f}",
                f"Best Epoch {s_detail.get('best_epoch', 1)}",
                help="学習・制御に未使用の未知テストデータ(20%)における正解率 (TP+TN)/N。",
            )
            sc4.metric(
                "Test Recall / F1",
                f"{s_detail.get('best_test_recall', s_detail.get('best_val_recall', 0.0)):.4f}",
                f"F1={s_detail.get('best_test_f1', s_detail.get('best_val_f1', 0.0)):.4f}",
                help="未知テストデータ(20%)における不良検出率 Recall と F1 スコア。",
            )
            sc5.metric(
                "Val Accuracy / Recall",
                f"{s_detail.get('best_val_accuracy', 0.0):.4f}",
                f"Rec={s_detail.get('best_val_recall', 0.0):.4f}",
                help="評価データ(20%)における正解率と不良検出率（次エポックのLoss方向性制御・Bestモデル選定基準）。",
            )
            sc6.metric(
                "最高 境界最小距離 d_min",
                f"{s_detail.get('best_d_min_boundary', 0.0):.4f}",
                help="良品の最小類似度と不良の最大類似度の差（>0で完全分離）。",
            )
            sc7.metric("完了エポック数", f"{s_detail.get('epochs_completed', len(s_hist))} ep")

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
