"""Step 6 UI: 3-Axis Space Evaluation with Session Recall, Swapped 3D UMAP + Dispersion Evolution, and Arrow-Navigable Image Popup."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api_client import BackendApiClient

st.set_page_config(page_title="Step 6: 3-Axis Space Evaluation", layout="wide")
client = BackendApiClient()


def _resolve_image_file(image_path_str: str) -> Path | None:
    """Resolve an image path stored in `.pkl` across container (`/app/data/...`) or local workspace."""
    if not image_path_str:
        return None
    p = Path(image_path_str)
    if p.exists() and p.is_file():
        return p
    # Fallback: remap /app/data/... relative to local workspace ./data/...
    if "/app/data/" in image_path_str:
        rel = image_path_str.split("/app/data/", 1)[1]
        for candidate_root in (Path("/app/data"), Path("data"), Path("../data"), Path("../../data")):
            cand = candidate_root / rel
            if cand.exists() and cand.is_file():
                return cand
    return None


def _render_inspector_body(key_prefix: str = "dialog") -> None:
    """Render the arrow-navigable (`◀ 前へ` / `次へ ▶`) image & tooltip metadata inspector."""
    samples: list[dict[str, Any]] = st.session_state.get("inspector_samples", [])
    title: str = st.session_state.get("inspector_title", "選択サンプル")
    if not samples:
        st.info("該当する画像サンプルがありません。")
        return

    n_total = len(samples)
    cur_idx = int(st.session_state.get("inspector_idx", 0)) % n_total
    st.session_state["inspector_idx"] = cur_idx

    st.markdown(f"#### 🖼️ {title} — 該当データ 全 **{n_total} 枚** 中 **{cur_idx + 1} 枚目** を表示中")

    # Navigation bar with Prev / Direct Selector / Next arrows
    nav_c1, nav_c2, nav_c3 = st.columns([1.1, 2.8, 1.1])
    with nav_c1:
        if st.button(
            "◀ 前の画像 (Prev)",
            key=f"{key_prefix}_prev_btn",
            use_container_width=True,
            disabled=(n_total <= 1),
        ):
            st.session_state["inspector_idx"] = (cur_idx - 1) % n_total
            st.rerun()

    with nav_c2:
        option_labels = [
            f"[{i + 1}/{n_total}] {s['file_name']} ({s['label_name']} | {s.get('cm_category', '')} | cos={s['cosine_similarity']:.3f})"
            for i, s in enumerate(samples)
        ]
        chosen_pos = st.selectbox(
            "表示する画像を直接選択（または左右の矢印ボタンで送り）",
            options=list(range(n_total)),
            index=cur_idx,
            format_func=lambda idx: option_labels[idx],
            key=f"{key_prefix}_select_{cur_idx}_{n_total}",
            label_visibility="collapsed",
        )
        if chosen_pos != cur_idx:
            st.session_state["inspector_idx"] = int(chosen_pos)
            st.rerun()

    with nav_c3:
        if st.button(
            "次の画像 (Next) ▶",
            key=f"{key_prefix}_next_btn",
            use_container_width=True,
            disabled=(n_total <= 1),
        ):
            st.session_state["inspector_idx"] = (cur_idx + 1) % n_total
            st.rerun()

    cur_sample = samples[st.session_state["inspector_idx"]]
    img_col, meta_col = st.columns([1.3, 1.7])

    with img_col:
        resolved_p = _resolve_image_file(str(cur_sample.get("image_path", "")))
        if resolved_p is not None:
            with Image.open(resolved_p) as pil_img:
                st.image(
                    pil_img.copy(),
                    caption=f"{cur_sample['file_name']} ({cur_sample['label_name']})",
                    use_container_width=True,
                )
        else:
            st.warning(f"画像ファイルが見つかりません: `{cur_sample.get('image_path')}`")

    with meta_col:
        cm_cat = cur_sample.get("cm_category", "")
        badge_map = {
            "TP": "🟢 **TP (不良を正しくNG検出)**",
            "FN": "🔴 **FN (不良の見落とし — 良品と誤判定)**",
            "FP": "🟠 **FP (良品の過検出 — 不良と誤判定)**",
            "TN": "🔵 **TN (良品を正しくOK合格)**",
        }
        st.markdown(f"##### 📋 Tooltip 詳細メタデータ ({badge_map.get(cm_cat, cm_cat)})")
        m_a, m_b, m_c = st.columns(3)
        m_a.metric("コサイン類似度 s(x)", f"{cur_sample['cosine_similarity']:.4f}")
        m_b.metric("コサイン距離 1 - s(x)", f"{cur_sample['cosine_distance']:.4f}")
        m_c.metric(
            "AI判定結果",
            "NG (Defect)" if cur_sample.get("predicted_defect") else "OK (Normal)",
        )

        st.markdown(
            f"""
            | 項目 | 値 |
            | :--- | :--- |
            | **ファイル名 (`file_name`)** | `{cur_sample['file_name']}` |
            | **正解クラス (`label_name`)** | `{cur_sample['label_name']}` (`{'不良 (NG)' if cur_sample.get('is_defect') else '良品 (OK)'}`) |
            | **混同行列セル (`cm_category`)** | `{cm_cat}` |
            | **データ分割 (`split`)** | `{cur_sample.get('split', 'train')}` |
            | **3D UMAP 座標 `(x, y, z)`** | `({cur_sample['umap_x']:.4f}, {cur_sample['umap_y']:.4f}, {cur_sample['umap_z']:.4f})` |
            | **画像パス (`image_path`)** | `{cur_sample.get('image_path', '')}` |
            """
        )

    # Thumbnail strip when multiple images exist
    if n_total > 1:
        st.markdown("###### 🎞️ 該当グループ内のサムネイル一覧（最大8枚プレビュー）")
        start_t = max(0, min(cur_idx - 3, n_total - 8))
        strip_samples = samples[start_t : start_t + 8]
        t_cols = st.columns(len(strip_samples))
        for offset, s_item in enumerate(strip_samples):
            abs_i = start_t + offset
            is_cur = abs_i == cur_idx
            tp_path = _resolve_image_file(str(s_item.get("image_path", "")))
            prefix_mark = "👉 " if is_cur else ""
            if tp_path is not None:
                with Image.open(tp_path) as t_img:
                    t_cols[offset].image(
                        t_img.copy(),
                        caption=f"{prefix_mark}#{abs_i + 1} {s_item['file_name']}\n(cos={s_item['cosine_similarity']:.2f})",
                        use_container_width=True,
                    )
            else:
                t_cols[offset].caption(f"{prefix_mark}#{abs_i + 1} {s_item['file_name']}")


@st.dialog("🖼️ 検査画像ポップアップ・インスペクタ（矢印送り対応）", width="large")
def _open_image_inspector_dialog() -> None:
    """Modal dialog displaying the selected Confusion Matrix cell or 3D UMAP points with arrow buttons."""
    _render_inspector_body(key_prefix="modal_dialog")


def _trigger_popup_with_samples(samples: list[dict[str, Any]], title: str) -> None:
    """Store target samples in session state and open the modal popup dialog."""
    st.session_state["inspector_samples"] = samples
    st.session_state["inspector_title"] = title
    st.session_state["inspector_idx"] = 0
    _open_image_inspector_dialog()


_STEP6_SERIES_TOOLTIPS: dict[str, str] = {
    "accuracy": "正解率 Accuracy = (TP+TN)/(TP+TN+FP+FN)。良品・不良品の両方を正しく判定できた総合割合（1.0が満点）。",
    "recall": "不良検出率（再現率）Recall = TP/(TP+FN)。実際の不良品のうち見逃さずにNG検出できた割合（1.0で見逃しゼロ）。",
    "precision": "適合率 Precision = TP/(TP+FP)。NGと判定した画像のうち実際に不良品だった割合（1.0で良品過検出ゼロ）。",
    "f1": "F1スコア = 2·Precision·Recall/(Precision+Recall)。見逃し防止(Recall)と過検出抑制(Precision)の調和平均。",
    "良品クラスタ内分散 (σ²_norm)": "選択Split内の良品サンプルと良品プロキシ間の平均コサイン距離 1 - S(x_norm, p_norm)。0に近いほど良品が一点に凝集。",
    "学習データ良品内分散 (Train σ²_norm)": "学習データ(Train 60%)の良品クラスタ内分散。学習したクラスタ分布の確からしさ・凝集度を確認する基準指標。",
    "不良クラス内分散 (σ²_def)": "各不良サブクラス内の重心からの平均コサイン距離。小さいほど不良モードごとにまとまっています。",
    "クラス間平均Cos距離 (D_inter)": "良品プロキシから不良サンプル群までの平均コサイン距離 1 - S(x_anom, p_norm)。大きいほど良品と不良が分離。",
    "境界最小Cos距離 (d_min)": "(良品の最小類似度) - (不良の最大類似度)。0.0を超えると良品と不良の分布に重なりがゼロ（完全分離）。",
    "空間分離比スコア J(t)": "フィッシャー空間分離比 D_inter / (σ²_norm + σ²_def)。大きいほど『クラス内は密・クラス間は遠い』理想的な特徴空間。",
    "学習データ空間分離比 (Train J(t))": "学習データ(Train 60%)における空間分離比 J(t)。テストデータの分離比と比較して過学習の有無を確認できます。",
    "3D UMAP 良品半径": "3D UMAP空間上での良品クラスタ重心からの二乗平均平方根(RMS)半径。",
    "3D UMAP 不良半径": "3D UMAP空間上での不良クラスタ重心からの二乗平均平方根(RMS)半径。",
}


def _apply_step6_hover_tooltips(fig: Any, x_label: str = "Epoch") -> None:
    """Attach explanatory Japanese tooltips to each trace in a Plotly line chart."""
    for trace in fig.data:
        series_name = str(getattr(trace, "name", ""))
        desc = _STEP6_SERIES_TOOLTIPS.get(series_name, series_name)
        trace.hovertemplate = (
            f"<b>{series_name}</b><br>"
            f"{x_label}: %{{x}}<br>"
            "値 (Value): <b>%{y:.4f}</b><br>"
            f"💡 <i>{desc}</i><extra></extra>"
        )


st.title("Step 6：128次元空間 3軸多角評価（6:2:2分割 テストデータ厳密評価・Accuracy・学習データの分散確認）")

# Sidebar & Top Bar: Dataset + Training Session Selector (Enhancement 1 & 2)
try:
    sessions_resp = client.list_sessions()
    all_sessions: list[dict[str, Any]] = sessions_resp.get("sessions", [])
except Exception:
    all_sessions = []

selected_session_id: str | None = None
selected_session_meta: dict[str, Any] = {}

if all_sessions:
    session_ids = [s["session_id"] for s in all_sessions]
    session_fmt = {
        s["session_id"]: (
            f"Session #{s['session_number']:04d} ({s['session_id']}) ｜ "
            f"Dataset: {s.get('dataset_name', 'default')} ｜ "
            f"Test Acc={s.get('best_test_accuracy', s.get('best_val_accuracy', 0.0)):.3f}, "
            f"Test Recall={s.get('best_test_recall', s.get('best_val_recall', 0.0)):.3f}, "
            f"Val F1={s.get('best_val_f1', 0.0):.3f}"
        )
        for s in all_sessions
    }
    if (
        "step6_active_session" not in st.session_state
        or st.session_state["step6_active_session"] not in session_ids
    ):
        st.session_state["step6_active_session"] = session_ids[0]
        st.session_state["step6_sidebar_session_select"] = session_ids[0]
        st.session_state["step6_main_session_select"] = session_ids[0]

    def _on_sb_session_change() -> None:
        val = st.session_state.get("step6_sidebar_session_select")
        if val in session_ids:
            st.session_state["step6_active_session"] = val
            st.session_state["step6_main_session_select"] = val

    def _on_main_session_change() -> None:
        val = st.session_state.get("step6_main_session_select")
        if val in session_ids:
            st.session_state["step6_active_session"] = val
            st.session_state["step6_sidebar_session_select"] = val

    with st.sidebar:
        st.markdown("### 🗂️ 評価対象セッションNo切替")
        st.selectbox(
            "呼び出す学習セッションNo",
            options=session_ids,
            format_func=lambda sid: session_fmt.get(sid, sid),
            key="step6_sidebar_session_select",
            on_change=_on_sb_session_change,
        )

    top_s_col1, top_s_col2 = st.columns([2.2, 1.8])
    with top_s_col1:
        st.selectbox(
            "🗂️ 評価・振り返り対象の学習セッションNoを選択 (過去の学習モデル・履歴を即座に呼び出し)",
            options=session_ids,
            format_func=lambda sid: session_fmt.get(sid, sid),
            key="step6_main_session_select",
            on_change=_on_main_session_change,
        )
    selected_session_id = str(st.session_state["step6_active_session"])
    selected_session_meta = next(
        (s for s in all_sessions if s["session_id"] == selected_session_id), {}
    )
    with top_s_col2:
        if selected_session_meta:
            st.info(
                f"**呼び出し中**: `Session #{selected_session_meta.get('session_number', 1):04d}` ｜ "
                f"データセット: `{selected_session_meta.get('dataset_name', 'default')}` ｜ "
                f"Best Epoch: `{selected_session_meta.get('best_epoch', 1)}` "
                f"(Test Acc `{selected_session_meta.get('best_test_accuracy', selected_session_meta.get('best_val_accuracy', 0.0)):.3f}`, "
                f"Test Recall `{selected_session_meta.get('best_test_recall', selected_session_meta.get('best_val_recall', 0.0)):.3f}`)"
            )

# Split Selector (6:2:2 -> Test 20% default, Val 20%, Train 60%, All 100%)
SPLIT_LABELS_MAP: dict[str, str] = {
    "test": "🧪 テストデータのみ (test: 20%) — 学習・制御に未使用の最終性能評価【デフォルト・カンニングなし】",
    "val": "📋 評価データのみ (val: 20%) — エポック毎のLoss方向性フィードバック・Bestモデル選定用",
    "train": "🏋️ 学習データのみ (train: 60%) — クラスタ分布の確からしさ・凝集分散の確認用",
    "all": "🌐 全データ統合 (all: 100%) — Train(60%) + Val(20%) + Test(20%) 全体の空間俯瞰",
}

sp_bar_c1, sp_bar_c2 = st.columns([2.6, 1.4])
with sp_bar_c1:
    selected_split = st.selectbox(
        "🎯 評価対象データ分割 (6:2:2 Split 切替 — 学習の良し悪しはテストデータ `test` で評価)",
        options=["test", "val", "train", "all"],
        index=0,
        format_func=lambda sp: SPLIT_LABELS_MAP.get(sp, sp),
        key="step6_split_selector",
        help=(
            "ディープラーニングの厳密評価のため、データは 学習(train 60%) : 評価(val 20%) : テスト(test 20%) に分割されています。"
            "最終的な学習の良し悪しは『テストデータ (test: 20%)』で評価し、学習したクラスタ分布の確からしさ（分散の収縮）を確認したい場合は『学習データ (train: 60%)』に切り替えて確認できます。"
        ),
    )
with sp_bar_c2:
    with st.popover("ℹ️ 6:2:2 データ分割（学習・評価・テスト）の役割と評価ルール説明"):
        st.markdown(
            """
            #### 📏 6 : 2 : 2 データ分割によるカンニングなし厳密評価
            1. **🏋️ 学習データ (`train`: 60%)**
               - モデルの重み更新・プロキシ更新・特徴量正規化統計量（平均・分散）の算出・初期K-Meansプロキシ初期化に**唯一使用されるデータ**です。
               - Step 6 では**「学習によって形成されたクラスタ分布の確からしさ（良品クラスタ内分散 $\\sigma^2_{\\text{norm}}$ の収縮度合い）」**を確認する用途で使用します。
            2. **📋 評価データ (`val`: 20%)**
               - 各エポック終了時に `val_loss`（求心力Loss vs 斥力Loss）および `val_recall` / `val_precision` を算出し、**次のエポックのLoss方向性（求心力比率 $\\lambda_{\\text{norm}}$ の増減）とカリキュラムへフィードバック**します。また `best_model.pt` の選定基準となります。
            3. **🧪 テストデータ (`test`: 20%)【デフォルト表示】**
               - 学習にもエポック間のLoss方向性制御にも**一切使用していない完全未知データ**です。**最終的な学習の良し悪し（Accuracy / Recall / Precision / F1）はこのテストデータで評価**します。
            """
        )

try:
    eval_list = client.list_evaluations(session_id=selected_session_id, split=selected_split)
    epochs_meta: list[dict[str, Any]] = eval_list.get("epochs", [])
except Exception as exc:
    epochs_meta = []
    st.error(f"評価データ一覧の取得に失敗しました: {exc}")

if not epochs_meta:
    st.info("保存済みのエポックスナップショット (`.pkl`) がありません。先に `2_training_monitor` で学習を実行してください。")
else:
    epoch_numbers = [int(ep["epoch"]) for ep in epochs_meta]
    best_ep_candidate = int(selected_session_meta.get("best_epoch") or epoch_numbers[-1])
    default_ep_val = best_ep_candidate if best_ep_candidate in epoch_numbers else epoch_numbers[-1]

    if "step6_selected_epoch" not in st.session_state or st.session_state["step6_selected_epoch"] not in epoch_numbers:
        st.session_state["step6_selected_epoch"] = default_ep_val
    st.session_state["step6_epoch_dropdown"] = int(st.session_state["step6_selected_epoch"])

    def _on_epoch_dropdown_change() -> None:
        val = st.session_state.get("step6_epoch_dropdown")
        if val in epoch_numbers:
            st.session_state["step6_selected_epoch"] = int(val)

    # Top Epoch & Threshold Controls
    ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4, ctrl_c5 = st.columns([0.7, 1.3, 0.7, 0.9, 2.4])
    cur_ep_idx = epoch_numbers.index(st.session_state["step6_selected_epoch"])

    with ctrl_c1:
        st.write("")
        if st.button("◀ Epoch -1", use_container_width=True, disabled=(cur_ep_idx <= 0)):
            new_ep = epoch_numbers[cur_ep_idx - 1]
            st.session_state["step6_selected_epoch"] = new_ep
            st.rerun()

    with ctrl_c2:
        st.selectbox(
            "評価対象エポック (学習回数)",
            options=epoch_numbers,
            key="step6_epoch_dropdown",
            on_change=_on_epoch_dropdown_change,
        )

    with ctrl_c3:
        st.write("")
        if st.button("Epoch +1 ▶", use_container_width=True, disabled=(cur_ep_idx >= len(epoch_numbers) - 1)):
            new_ep = epoch_numbers[cur_ep_idx + 1]
            st.session_state["step6_selected_epoch"] = new_ep
            st.rerun()

    with ctrl_c4:
        st.write("")
        if st.button(f"★ Best (Ep {default_ep_val})", use_container_width=True):
            st.session_state["step6_selected_epoch"] = default_ep_val
            st.rerun()

    selected_epoch = int(st.session_state["step6_selected_epoch"])
    tau_sim = ctrl_c5.slider(
        "判定閾値: 良品プロキシとの最大コサイン類似度 τ_sim (s(x) < τ_sim を不良 NG と判定)",
        min_value=-1.0,
        max_value=1.0,
        value=0.0,
        step=0.02,
        help="学習時のデフォルト閾値は 0.0（直交）です。スライダーを動かすと即座に Recall / F1 と混同行列が再計算されます。",
    )

    tau_sim_clean = round(min(1.0, max(-1.0, float(tau_sim))), 4)
    try:
        eval_data = client.get_epoch_evaluation(
            epoch=int(selected_epoch),
            tau_sim=tau_sim_clean,
            session_id=selected_session_id,
            split=selected_split,
        )
    except Exception as exc:
        st.error(f"Epoch {selected_epoch} の評価データ取得に失敗しました: {exc}")
        st.stop()
    test_m = eval_data.get("test_metrics") or eval_data["val_metrics"]
    val_m = eval_data["val_metrics"]
    train_m = eval_data.get("train_metrics") or eval_data["all_metrics"]
    all_m = eval_data["all_metrics"]
    active_m = eval_data.get("active_metrics") or (
        test_m
        if selected_split == "test"
        else (val_m if selected_split == "val" else (train_m if selected_split == "train" else all_m))
    )
    split_counts_map: dict[str, int] = eval_data.get("split_counts") or {}
    disp_by_split_map: dict[str, dict[str, Any]] = eval_data.get("dispersion_by_split") or {}

    all_samples_list: list[dict[str, Any]] = eval_data["samples"]
    if selected_split == "all":
        samples_list = all_samples_list
    else:
        filtered_sp = [s for s in all_samples_list if s.get("split") == selected_split]
        samples_list = filtered_sp if filtered_sp else all_samples_list
    df_samples = pd.DataFrame(samples_list)

    split_short_badge = {
        "test": f"🧪 テストデータ (test: {len(samples_list)}枚 / 20%)",
        "val": f"📋 評価データ (val: {len(samples_list)}枚 / 20%)",
        "train": f"🏋️ 学習データ (train: {len(samples_list)}枚 / 60%)",
        "all": f"🌐 全データ統合 (all: {len(samples_list)}枚 / 100%)",
    }.get(selected_split, selected_split)

    # Group samples in the active split by Confusion Matrix category (TP, FN, FP, TN)
    samples_by_cm: dict[str, list[dict[str, Any]]] = {
        "TP": [s for s in samples_list if s.get("cm_category") == "TP"],
        "FN": [s for s in samples_list if s.get("cm_category") == "FN"],
        "FP": [s for s in samples_list if s.get("cm_category") == "FP"],
        "TN": [s for s in samples_list if s.get("cm_category") == "TN"],
    }

    # =========================================================================
    # SECTION 1: Quantitative High-Dimensional Cosine Margin & Interactive Confusion Matrix with Image Tooltips / Popup
    # =========================================================================
    st.subheader(
        f"1. 【最重要基準①】高次元定量的マージン ＆ 閾値連動 Accuracy / Recall / F1 スコア（対象: {split_short_badge}）"
    )
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric(
        f"[{selected_split.upper()}] Accuracy (正解率)",
        f"{float(active_m.get('accuracy', 0.0)):.4f}",
        help="選択中のデータ分割(Split)における総合正解率 (TP+TN)/(TP+TN+FP+FN)。良品・不良品の両方を正しく判定できた割合です。",
    )
    k2.metric(
        f"[{selected_split.upper()}] Recall (不良検出率)",
        f"{float(active_m.get('recall', 0.0)):.4f}",
        help="選択中のデータ分割(Split)における不良検出率 TP/(TP+FN)。実際の不良品を見逃さずNG検出できた割合です（1.0で見逃しゼロ）。",
    )
    k3.metric(
        f"[{selected_split.upper()}] F1 スコア",
        f"{float(active_m.get('f1', 0.0)):.4f}",
        help="Recall(見逃し防止)とPrecision(過検出防止)の調和平均 2·P·R/(P+R)。",
    )
    k4.metric(
        f"[{selected_split.upper()}] Precision (適合率)",
        f"{float(active_m.get('precision', 0.0)):.4f}",
        help="NGと判定したサンプルのうち実際に不良品だった割合 TP/(TP+FP)（1.0で過検出ゼロ）。",
    )
    k5.metric(
        f"[{selected_split.upper()}] 境界最小距離 d_min",
        f"{float(active_m.get('d_min_boundary', eval_data['d_min_boundary'])):.4f}",
        help="(選択Split内の良品の最小類似度) - (不良の最大類似度)。0.0より大きければ良品と不良の分布に重なりがありません。",
    )
    k6.metric(
        f"[{selected_split.upper()}] 中心間距離 d_cent",
        f"{float(active_m.get('d_min_centroid', eval_data['d_min_centroid'])):.4f}",
        help="1 - Mean(S(x_anom, p_norm))。良品プロキシから不良クラスタ重心までの平均コサイン距離（目標 ≥ 1.0）。",
    )
    k7.metric(
        "Val→次Epoch 方向性制御",
        str(eval_data.get("loss_direction_feedback", "balanced_hold")),
        f"λ_norm比率={float(eval_data.get('effective_normal_ratio', 0.5)):.2f}",
        help="評価データ(val 20%)のLoss・Recall/Precisionバランスから次のエポックへ返されたLoss方向性フィードバック（centripetal_focus=求心力強化 / repulsion_focus=斥力強化 / balanced_hold=均衡維持）。",
    )

    # 6:2:2 Split-by-Split Comparison Table (Test 20% vs Val 20% vs Train 60% vs All 100%)
    with st.expander(
        f"📊 6:2:2 分割別（Test 20% / Val 20% / Train 60% / All 100%）精度・クラスタ分散 比較サマリー (@ τ_sim={tau_sim:.2f})",
        expanded=True,
    ):
        st.caption(
            "💡 **学習の良し悪し（汎化性能）は『🧪 Test (20%)』の Accuracy / Recall / F1 で評価**し、"
            "**形成されたクラスタ分布の確からしさ（良品が中心に鋭く凝集しているか）は『🏋️ Train (60%)』の良品内分散 $\\sigma^2_{\\text{norm}}$ と空間分離比 $J(t)$** で確認できます。"
        )
        split_summary_rows = []
        for sp_key, sp_title, sp_m in [
            ("test", "🧪 Test (20%: 最終性能評価・未知データ)", test_m),
            ("val", "📋 Val (20%: 次Epoch方向性制御・モデル選定)", val_m),
            ("train", "🏋️ Train (60%: 学習・クラスタ分布確からしさ確認)", train_m),
            ("all", "🌐 All (100%: 全データ合計)", all_m),
        ]:
            sp_disp = disp_by_split_map.get(sp_key) or eval_data.get("dispersion_metrics", {})
            split_summary_rows.append(
                {
                    "選択中": "👉 表示中" if sp_key == selected_split else "",
                    "データ分割 (Split)": sp_title,
                    "枚数": int(sp_m.get("sample_count", split_counts_map.get(sp_key, 0))),
                    "Accuracy (正解率)": f"{float(sp_m.get('accuracy', 0.0)):.4f}",
                    "Recall (不良検出率)": f"{float(sp_m.get('recall', 0.0)):.4f}",
                    "Precision (適合率)": f"{float(sp_m.get('precision', 0.0)):.4f}",
                    "F1 スコア": f"{float(sp_m.get('f1', 0.0)):.4f}",
                    "TP / FN / FP / TN": f"{sp_m.get('tp', 0)} / {sp_m.get('fn', 0)} / {sp_m.get('fp', 0)} / {sp_m.get('tn', 0)}",
                    "境界最小距離 d_min": f"{float(sp_m.get('d_min_boundary', 0.0)):.4f}",
                    "良品内分散 σ²_norm": f"{float(sp_disp.get('normal_intra_disp', 0.0)):.4f}",
                    "空間分離比 J(t)": f"{float(sp_disp.get('fisher_separation_ratio', 0.0)):.3f}",
                }
            )
        st.dataframe(
            pd.DataFrame(split_summary_rows),
            use_container_width=True,
            hide_index=True,
            key="step6_split_comparison_df",
        )

    cm_col, curve_col, hist_col = st.columns([1.25, 1.0, 1.0])
    with cm_col:
        cm_hdr_c1, cm_hdr_c2 = st.columns([3.2, 0.8])
        cm_hdr_c1.markdown(
            f"**混同行列（{split_short_badge} @ $\\tau_{{\\text{{sim}}}} = {tau_sim:.2f}$）**"
        )
        with cm_hdr_c2.popover("ℹ️ 説明"):
            st.markdown(
                """
                - **TP (左上)**: 実際の不良品を正しく `NG` と検出した枚数（多いほど良い）。
                - **FN (右上)**: 実際の不良品を `OK` と見落とした枚数（**0枚が目標**）。
                - **FP (左下)**: 実際の良品を `NG` と過検出した枚数（少ないほど良い）。
                - **TN (右下)**: 実際の良品を正しく `OK` と合格判定した枚数（多いほど良い）。
                """
            )

        def _cm_tooltip_files(cat: str) -> str:
            arr = samples_by_cm.get(cat, [])
            if not arr:
                return "該当画像なし"
            names = [f"{s['file_name']}({s['label_name']}, cos={s['cosine_similarity']:.2f})" for s in arr[:6]]
            more = f" ...他 {len(arr) - 6}枚" if len(arr) > 6 else ""
            return "<br>".join(names) + more

        cm_z = [
            [active_m["tp"], active_m["fn"]],
            [active_m["fp"], active_m["tn"]],
        ]
        cm_text = [
            [f"TP (不良検出)<br>{active_m['tp']} 枚", f"FN (不良見落とし)<br>{active_m['fn']} 枚"],
            [f"FP (過検出)<br>{active_m['fp']} 枚", f"TN (正常合格)<br>{active_m['tn']} 枚"],
        ]
        cm_hover = [
            [
                f"<b>[{selected_split.upper()}] TP (不良を正しくNG検出): {active_m['tp']}枚</b><br>💡 実不良で s(x) &lt; τ_sim のサンプル<br>代表ファイル:<br>{_cm_tooltip_files('TP')}",
                f"<b>[{selected_split.upper()}] FN (不良の見落とし): {active_m['fn']}枚</b><br>💡 実不良なのに s(x) ≥ τ_sim となった要注意サンプル<br>代表ファイル:<br>{_cm_tooltip_files('FN')}",
            ],
            [
                f"<b>[{selected_split.upper()}] FP (良品の過検出): {active_m['fp']}枚</b><br>💡 実良品なのに s(x) &lt; τ_sim となったサンプル<br>代表ファイル:<br>{_cm_tooltip_files('FP')}",
                f"<b>[{selected_split.upper()}] TN (良品を正しくOK判定): {active_m['tn']}枚</b><br>💡 実良品で s(x) ≥ τ_sim のサンプル<br>代表ファイル:<br>{_cm_tooltip_files('TN')}",
            ],
        ]
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm_z,
                x=["Pred Defect (NG判定)", "Pred Normal (OK判定)"],
                y=["Actual Defect (実不良)", "Actual Normal (実良品)"],
                text=cm_text,
                texttemplate="%{text}",
                hovertext=cm_hover,
                hoverinfo="text",
                colorscale="Blues",
                showscale=False,
            )
        )
        fig_cm.update_layout(
            height=240,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_cm, use_container_width=True, key="eval_cm_heatmap_chart")

        st.caption(f"👇 `{selected_split.upper()}` 混同行列の各セルに該当する実画像をポップアップ（矢印送り対応）で確認:")
        btn_tp, btn_fn, btn_fp, btn_tn = st.columns(4)
        if btn_tp.button(
            f"🟢 TP ({active_m['tp']}枚)",
            use_container_width=True,
            help="不良を正しくNG検出した画像をポップアップで確認します",
        ):
            _trigger_popup_with_samples(
                samples_by_cm["TP"], f"[{selected_split.upper()}] 混同行列 [TP: 不良検出] (Epoch {selected_epoch})"
            )
        if btn_fn.button(
            f"🔴 FN ({active_m['fn']}枚)",
            use_container_width=True,
            help="不良を見落とした（OKと誤判定した）画像をポップアップで確認します",
        ):
            _trigger_popup_with_samples(
                samples_by_cm["FN"], f"[{selected_split.upper()}] 混同行列 [FN: 不良見落とし] (Epoch {selected_epoch})"
            )
        if btn_fp.button(
            f"🟠 FP ({active_m['fp']}枚)",
            use_container_width=True,
            help="良品を過検出した（NGと誤判定した）画像をポップアップで確認します",
        ):
            _trigger_popup_with_samples(
                samples_by_cm["FP"], f"[{selected_split.upper()}] 混同行列 [FP: 良品過検出] (Epoch {selected_epoch})"
            )
        if btn_tn.button(
            f"🔵 TN ({active_m['tn']}枚)",
            use_container_width=True,
            help="良品を正しくOK合格とした画像をポップアップで確認します",
        ):
            _trigger_popup_with_samples(
                samples_by_cm["TN"], f"[{selected_split.upper()}] 混同行列 [TN: 正常合格] (Epoch {selected_epoch})"
            )

    with curve_col:
        with st.popover("ℹ️ 閾値 (τ_sim) 対 Accuracy/Recall/Precision/F1 曲線の説明 (Tooltip)"):
            st.markdown(
                """
                - **横軸 (`tau_sim`)**: 判定閾値 $\\tau_{\\text{sim}}$。コサイン類似度 $s(x) < \\tau_{\\text{sim}}$ の画像を不良 (NG) と判定します。
                - **`accuracy` (正解率)**: $(TP+TN)/N$。良品・不良品全体の正解率。
                - **`recall` (不良検出率)**: $TP/(TP+FN)$。閾値 $\\tau_{\\text{sim}}$ を右（高め）に動かすほど見逃しが減り `1.0` に近づきます。
                - **`precision` (適合率)**: $TP/(TP+FP)$。閾値 $\\tau_{\\text{sim}}$ を左（低め）に動かすほど良品の過検出が減り `1.0` に近づきます。
                - **`f1` (F1スコア)**: Recall と Precision の両方が高いスイートスポット（山が最も高い位置）が最適閾値です。
                """
            )
        df_curve = pd.DataFrame(eval_data["threshold_curve"])
        curve_y_cols = [
            c for c in ["accuracy", "recall", "precision", "f1"] if c in df_curve.columns
        ]
        fig_curve = px.line(
            df_curve,
            x="tau_sim",
            y=curve_y_cols,
            title=f"[{selected_split.upper()}] 閾値 (τ_sim) 対 Accuracy / Recall / Precision / F1",
        )
        _apply_step6_hover_tooltips(fig_curve, x_label="閾値 τ_sim")
        fig_curve.add_vline(
            x=float(tau_sim), line_dash="dash", line_color="red", annotation_text=f"τ={tau_sim:.2f}"
        )
        st.plotly_chart(fig_curve, use_container_width=True, key="eval_threshold_curve_chart")

    with hist_col:
        with st.popover("ℹ️ コサイン類似度分布ヒストグラムの説明 (Tooltip)"):
            st.markdown(
                """
                - **横軸 (`cosine_similarity`)**: 良品プロキシ $p_{\\text{norm}}$ とのコサイン類似度 $s(x) \\in [-1, 1]$。
                - **右側の山 ($s(x) \\to +1.0$)**: 良品クラスタ。右端 `+1.0` 付近に狭く固まっているほど良品の分散が小さく優秀です。
                - **左側の山 ($s(x) < \\tau_{\\text{sim}}$)**: 不良クラスタ。赤破線（判定閾値 $\\tau_{\\text{sim}}$）より左側に全ての不良サンプルが分離していれば見逃しゼロ・過検出ゼロを達成できます。
                """
            )
        fig_hist = px.histogram(
            df_samples,
            x="cosine_similarity",
            color="label_name",
            nbins=30,
            barmode="overlay",
            hover_data=["file_name", "cm_category", "split"],
            title=f"[{selected_split.upper()}] 良品プロキシとのコサイン類似度分布 s(x) ∈ [-1, 1]",
        )
        fig_hist.add_vline(
            x=float(tau_sim), line_dash="dash", line_color="red", annotation_text=f"τ={tau_sim:.2f}"
        )
        st.plotly_chart(fig_hist, use_container_width=True, key="eval_cos_sim_hist_chart")

    st.divider()

    # =========================================================================
    # SECTION 2 (SWAPPED UP!): 128-D -> 3D UMAP Interactive Scatter Plot + Dispersion Evolution Across Epochs
    # =========================================================================
    st.subheader(
        f"2. 【最重要基準②】128次元 → 3D UMAP 立体散布図 ＆ 学習回数（Epoch）に応じた分散推移（選択中: {split_short_badge} ＋ 学習データ分散比較）"
    )
    st.caption(
        "学習回数（Epoch）が進むにつれて、**選択中の評価分割（デフォルト `test` 20%）および学習データ（`train` 60%: クラスタ分布の確からしさ確認用）の"
        "良品クラスタ内分散（$\\sigma_{\\text{norm}}^2$）がどのように収縮し、クラス間距離（$D_{\\text{inter}}$）と空間分離比（$J(t)$）がどう拡大していくか**を可視化します。"
    )

    # Build Epoch-by-Epoch Dispersion History DataFrame from epochs_meta (including Train 60% reference for cluster plausibility)
    disp_rows: list[dict[str, Any]] = []
    for ep_item in epochs_meta:
        dm = ep_item.get("dispersion_metrics", {})
        dm_by_sp = ep_item.get("dispersion_by_split") or {}
        dm_train = dm_by_sp.get("train", dm)
        disp_rows.append(
            {
                "epoch": int(ep_item["epoch"]),
                "良品クラスタ内分散 (σ²_norm)": float(dm.get("normal_intra_disp", 0.0)),
                "学習データ良品内分散 (Train σ²_norm)": float(dm_train.get("normal_intra_disp", 0.0)),
                "不良クラス内分散 (σ²_def)": float(dm.get("defect_intra_disp", 0.0)),
                "クラス間平均Cos距離 (D_inter)": float(dm.get("inter_class_dist", 0.0)),
                "境界最小Cos距離 (d_min)": float(ep_item.get("d_min_boundary", 0.0)),
                "空間分離比スコア J(t)": float(dm.get("fisher_separation_ratio", 0.0)),
                "学習データ空間分離比 (Train J(t))": float(dm_train.get("fisher_separation_ratio", 0.0)),
                "3D UMAP 良品半径": float(dm.get("umap_normal_radius", 0.0)),
                "3D UMAP 不良半径": float(dm.get("umap_defect_radius", 0.0)),
            }
        )
    df_disp_hist = pd.DataFrame(disp_rows)

    cur_disp: dict[str, Any] = eval_data.get("dispersion_metrics", {})
    train_disp_now: dict[str, Any] = disp_by_split_map.get("train", cur_disp)
    ep1_disp: dict[str, Any] = epochs_meta[0].get("dispersion_metrics", {}) if epochs_meta else {}

    # Dispersion KPI Cards with Delta vs Epoch 1 + Train Cluster Dispersion Plausibility Card
    d_k1, d_k2, d_k3, d_k4, d_k5, d_k6 = st.columns(6)
    cur_norm_d = float(cur_disp.get("normal_intra_disp", 0.0))
    ep1_norm_d = float(ep1_disp.get("normal_intra_disp", cur_norm_d))
    train_norm_d = float(train_disp_now.get("normal_intra_disp", 0.0))
    cur_def_d = float(cur_disp.get("defect_intra_disp", 0.0))
    ep1_def_d = float(ep1_disp.get("defect_intra_disp", cur_def_d))
    cur_inter_d = float(cur_disp.get("inter_class_dist", 0.0))
    ep1_inter_d = float(ep1_disp.get("inter_class_dist", cur_inter_d))
    cur_fisher = float(cur_disp.get("fisher_separation_ratio", 0.0))
    ep1_fisher = float(ep1_disp.get("fisher_separation_ratio", cur_fisher))

    d_k1.metric(
        f"[{selected_split.upper()}] 良品内分散 (σ²_norm)",
        f"{cur_norm_d:.4f}",
        delta=f"{cur_norm_d - ep1_norm_d:+.4f} (vs Ep1)",
        delta_color="inverse",
        help="選択Split内の良品サンプルと良品プロキシ間の平均コサイン距離（小さいほど良品が密に凝集）",
    )
    d_k2.metric(
        "🏋️ 学習データ良品分散 (Train σ²_norm)",
        f"{train_norm_d:.4f}",
        delta=f"差 {cur_norm_d - train_norm_d:+.4f} (Test-Train)",
        delta_color="off",
        help="学習データ(Train 60%)の良品クラスタ内分散。学習で形成されたクラスタ分布の確からしさ・凝集度を確認できます。",
    )
    d_k3.metric(
        f"[{selected_split.upper()}] 不良内分散 (σ²_def)",
        f"{cur_def_d:.4f}",
        delta=f"{cur_def_d - ep1_def_d:+.4f} (vs Ep1)",
        delta_color="inverse",
        help="各不良モード内の重心からの平均コサイン距離（小さいほど各不良モードが凝集）",
    )
    d_k4.metric(
        f"[{selected_split.upper()}] クラス間距離 (D_inter)",
        f"{cur_inter_d:.4f}",
        delta=f"{cur_inter_d - ep1_inter_d:+.4f} (vs Ep1)",
        help="良品プロキシから不良サンプルまでの平均コサイン距離（大きいほど良品と不良が分離）",
    )
    d_k5.metric(
        f"[{selected_split.upper()}] 空間分離比 J(t)",
        f"{cur_fisher:.3f}",
        delta=f"{cur_fisher - ep1_fisher:+.3f} (vs Ep1)",
        help="J(t) = D_inter / (σ²_norm + σ²_def) — 大きいほどクラス内分散が小さくクラス間が離れている理想的な状態",
    )
    d_k6.metric(
        "3D UMAP 良品/不良 半径",
        f"{float(cur_disp.get('umap_normal_radius', 0.0)):.2f} / {float(cur_disp.get('umap_defect_radius', 0.0)):.2f}",
        help="3D UMAP空間上での良品群および不良群のRMS半径",
    )

    # Epoch-by-Epoch Dispersion Evolution Charts with Explanatory Tooltips
    if len(df_disp_hist) >= 1:
        ev_col1, ev_col2 = st.columns(2)
        with ev_col1:
            with st.popover("ℹ️ クラスタ内分散・クラス間距離 推移グラフの項目説明 (Tooltip)"):
                st.markdown(
                    """
                    - **`良品クラスタ内分散 (σ²_norm)`**: 選択中Split（デフォルト `test` 20%）の良品データのプロキシ周囲でのばらつき（**下がるほど良い**）。
                    - **`学習データ良品内分散 (Train σ²_norm)`**: 学習データ (`train` 60%) の良品クラスタ内分散。**学習したクラスタ分布の確からしさ**を確認でき、`test` の分散と近い値で一緒に下がっていれば過学習なく健全に収縮しています。
                    - **`不良クラス内分散 (σ²_def)`**: 各不良サブクラス内のばらつき（**下がるほど良い**）。
                    - **`クラス間平均Cos距離 (D_inter)`**: 良品プロキシから不良群までの平均コサイン距離（**上がるほど良い、目標 $\\ge 1.0$**）。
                    - **`境界最小Cos距離 (d_min)`**: 最も外側の良品と最も内側の不良のコサイン類似度差（**`0.0` を超えると完全分離**）。
                    """
                )
            fig_disp_curve = px.line(
                df_disp_hist,
                x="epoch",
                y=[
                    "良品クラスタ内分散 (σ²_norm)",
                    "学習データ良品内分散 (Train σ²_norm)",
                    "不良クラス内分散 (σ²_def)",
                    "クラス間平均Cos距離 (D_inter)",
                    "境界最小Cos距離 (d_min)",
                ],
                markers=True,
                title=f"📈 学習回数 (Epoch) に伴う分散収縮 ({selected_split.upper()} vs Train 60% 分布比較)",
            )
            _apply_step6_hover_tooltips(fig_disp_curve, x_label="Epoch")
            fig_disp_curve.add_vline(
                x=int(selected_epoch),
                line_dash="dash",
                line_color="red",
                annotation_text=f"選択中: Ep {selected_epoch}",
            )
            fig_disp_curve.update_layout(height=320)
            st.plotly_chart(fig_disp_curve, use_container_width=True, key="eval_disp_evolution_chart")

        with ev_col2:
            with st.popover("ℹ️ 空間分離比 J(t) ＆ 3D UMAP 半径推移グラフの項目説明 (Tooltip)"):
                st.markdown(
                    """
                    - **`空間分離比スコア J(t)`**: フィッシャー判別比に基づく指標 $J(t) = \\frac{D_{\\text{inter}}}{\\sigma^2_{\\text{norm}} + \\sigma^2_{\\text{def}}}$。**値が高く右肩上がりであるほど、「良品は一点に凝集し、不良は遠くへ離れている」理想的な特徴空間**です。
                    - **`学習データ空間分離比 (Train J(t))`**: 学習データ (`train` 60%) での空間分離比。テストデータの $J(t)$ と比較することで、クラスタ構造の汎化性を確認できます。
                    - **`3D UMAP 良品半径` / `3D UMAP 不良半径`**: 3次元UMAP空間上での各クラスタの広がり（RMS半径）。
                    """
                )
            fig_sep_curve = px.line(
                df_disp_hist,
                x="epoch",
                y=[
                    "空間分離比スコア J(t)",
                    "学習データ空間分離比 (Train J(t))",
                    "3D UMAP 良品半径",
                    "3D UMAP 不良半径",
                ],
                markers=True,
                title=f"📈 学習回数 (Epoch) に伴う空間分離比 J(t) ({selected_split.upper()} vs Train 60%)",
            )
            _apply_step6_hover_tooltips(fig_sep_curve, x_label="Epoch")
            fig_sep_curve.add_vline(
                x=int(selected_epoch),
                line_dash="dash",
                line_color="red",
                annotation_text=f"選択中: Ep {selected_epoch}",
            )
            fig_sep_curve.update_layout(height=320)
            st.plotly_chart(fig_sep_curve, use_container_width=True, key="eval_sep_evolution_chart")

    # Option to overlay Train cluster distribution on 3D UMAP alongside Test samples
    umap_overlay_c1, umap_overlay_c2 = st.columns([2.6, 1.4])
    with umap_overlay_c1:
        overlay_all_splits_on_umap = st.checkbox(
            "🔬 3D/2D UMAP 上に学習データ (train 60%) も重ねて表示し、学習クラスタ分布に対するテストデータの位置を確認する",
            value=False,
            key="step6_umap_overlay_all",
            help="ONにすると、Section 1 の定量評価（Accuracy/Recall/混同行列）は選択Split（test 20%）に厳密に限定したまま、3D UMAP 上で学習データ(train)のクラスタ分布とテストデータ(test)の位置関係を重ね合わせて確認できます。",
        )
    with umap_overlay_c2:
        with st.popover("ℹ️ 3D UMAP 立体散布図の見方・記号説明 (Tooltip)"):
            st.markdown(
                """
                - **色 (`color`)**: 各サンプルのクラス名（良品クラスおよび各不良サブクラス）。
                - **マーカー形状 (`symbol`)**: 混同行列の判定ステータス（`TP`, `FN`, `FP`, `TN`）。
                - **★ 金色ダイヤ (`Normal Multi-Proxy`)**: 学習された良品プロキシの中心座標。良品サンプルがこの周囲に密に集まっているかを確認します。
                - **◆ マゼンタ十字 (`Defect Mode Proxy`)**: 各不良サブクラスのプロキシ座標。
                """
            )

    umap_source_samples = all_samples_list if overlay_all_splits_on_umap else samples_list
    df_umap_samples = pd.DataFrame(umap_source_samples)

    # 3D UMAP Scatter Plot + Interactive Point Selection / Tooltip Popup
    umap_col_main, umap_col_side = st.columns([1.85, 1.15])

    with umap_col_main:
        df_umap_samples["判定ステータス"] = df_umap_samples["cm_category"].map(
            {
                "TP": "TP: 不良検出 (NG)",
                "FN": "FN: 不良見落とし (要注意!)",
                "FP": "FP: 良品過検出",
                "TN": "TN: 正常合格 (OK)",
            }
        )
        df_umap_samples["クラス (Split)"] = (
            df_umap_samples["label_name"].astype(str)
            + " ["
            + df_umap_samples["split"].astype(str)
            + "]"
            if overlay_all_splits_on_umap
            else df_umap_samples["label_name"].astype(str)
        )
        fig_umap = px.scatter_3d(
            df_umap_samples,
            x="umap_x",
            y="umap_y",
            z="umap_z",
            color="クラス (Split)" if overlay_all_splits_on_umap else "label_name",
            symbol="判定ステータス",
            custom_data=[
                "index",
                "file_name",
                "label_name",
                "cm_category",
                "cosine_similarity",
                "cosine_distance",
                "split",
            ],
            title=(
                f"Epoch {selected_epoch}: 128次元超球面空間の 3D UMAP 立体投影 "
                f"（表示: {'全Split重ね合わせ' if overlay_all_splits_on_umap else split_short_badge} ｜ 点ホバーで詳細Tooltip表示）"
            ),
            height=620,
        )
        fig_umap.update_traces(
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "クラス: %{customdata[2]}<br>"
                "データ分割 (Split): <b>%{customdata[6]}</b><br>"
                "混同行列セル: <b>%{customdata[3]}</b><br>"
                "Cos類似度 s(x): %{customdata[4]:.4f} (1.0に近いほど良品中心に近い)<br>"
                "Cos距離 1-s(x): %{customdata[5]:.4f}<br>"
                "3D UMAP座標: (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>"
            )
        )

        proxy_points = eval_data.get("proxy_points", [])
        if proxy_points:
            norm_p = [p for p in proxy_points if p.get("proxy_type") == "normal_proxy"]
            def_p = [p for p in proxy_points if p.get("proxy_type") == "defect_proxy"]
            if norm_p:
                fig_umap.add_trace(
                    go.Scatter3d(
                        x=[p["umap_x"] for p in norm_p],
                        y=[p["umap_y"] for p in norm_p],
                        z=[p["umap_z"] for p in norm_p],
                        mode="markers+text",
                        name="★ Normal Multi-Proxy",
                        text=[p["proxy_name"] for p in norm_p],
                        textposition="top center",
                        marker=dict(
                            size=10,
                            symbol="diamond",
                            color="gold",
                            line=dict(width=2, color="black"),
                        ),
                    )
                )
            if def_p:
                fig_umap.add_trace(
                    go.Scatter3d(
                        x=[p["umap_x"] for p in def_p],
                        y=[p["umap_y"] for p in def_p],
                        z=[p["umap_z"] for p in def_p],
                        mode="markers+text",
                        name="◆ Defect Mode Proxy",
                        text=[p["proxy_name"] for p in def_p],
                        textposition="top center",
                        marker=dict(
                            size=10,
                            symbol="cross",
                            color="magenta",
                            line=dict(width=2, color="black"),
                        ),
                    )
                )

        umap_3d_event = st.plotly_chart(
            fig_umap,
            use_container_width=True,
            key=f"eval_3d_umap_chart_ep_{selected_epoch}_{selected_split}",
            on_select="rerun",
        )

    with umap_col_side:
        st.markdown("##### 🎯 UMAP マッピング上の点クリック ／ 領域選択で画像ポップアップ確認")
        st.caption(
            "下の 2D UMAP ナビゲータ上で**点をクリック**または**ドラッグ（Box/Lasso）で複数点を囲む**と、"
            "該当する全画像が即座に矢印送りポップアップで開きます。"
        )
        fig_umap_2d = px.scatter(
            df_umap_samples,
            x="umap_x",
            y="umap_y",
            color="label_name",
            symbol="cm_category",
            custom_data=[
                "index",
                "file_name",
                "label_name",
                "cm_category",
                "cosine_similarity",
                "cosine_distance",
                "split",
            ],
            title=f"クリック/矩形選択用 UMAP (X-Y平面投影 | Ep {selected_epoch})",
            height=340,
        )
        fig_umap_2d.update_traces(
            marker=dict(size=9, line=dict(width=1, color="DarkSlateGrey")),
            hovertemplate=(
                "<b>%{customdata[1]}</b> (%{customdata[2]} | Split: %{customdata[6]})<br>"
                "判定: %{customdata[3]} | cos=%{customdata[4]:.3f}<extra></extra>"
            ),
        )
        fig_umap_2d.update_layout(
            dragmode="select",
            margin=dict(l=10, r=10, t=35, b=10),
        )
        umap_2d_event = st.plotly_chart(
            fig_umap_2d,
            use_container_width=True,
            key=f"eval_2d_umap_selector_ep_{selected_epoch}_{selected_split}",
            on_select="rerun",
            selection_mode=("points", "box", "lasso"),
        )

        # Extract selected sample indices from 2D or 3D UMAP selection event
        selected_indices: list[int] = []
        for ev in (umap_2d_event, umap_3d_event):
            if ev and hasattr(ev, "selection") and ev.selection and ev.selection.get("points"):
                for pt in ev.selection["points"]:
                    cd = pt.get("customdata")
                    if (
                        isinstance(cd, list)
                        and len(cd) >= 1
                        and isinstance(cd[0], (int, float))
                        and not isinstance(cd[0], bool)
                    ):
                        idx_val = int(cd[0])
                        if idx_val not in selected_indices:
                            selected_indices.append(idx_val)

        if selected_indices:
            matched_umap_samples = [
                s for s in all_samples_list if int(s["index"]) in selected_indices
            ]
            st.success(f"UMAP上で **{len(matched_umap_samples)} 点** が選択されています。")
            if st.button(
                f"🔍 UMAP選択中の {len(matched_umap_samples)} 枚をポップアップ画像ビューアで開く (矢印送り)",
                type="primary",
                use_container_width=True,
                key="open_umap_sel_modal_btn",
            ):
                _trigger_popup_with_samples(
                    matched_umap_samples,
                    f"UMAP 空間選択サンプル ({len(matched_umap_samples)}枚 | Epoch {selected_epoch})",
                )

            # Auto-pop modal the first time a new UMAP selection signature appears
            sel_sig = f"{selected_epoch}_{selected_split}_{','.join(str(i) for i in selected_indices)}"
            if st.session_state.get("last_umap_sel_sig") != sel_sig:
                st.session_state["last_umap_sel_sig"] = sel_sig
                _trigger_popup_with_samples(
                    matched_umap_samples,
                    f"UMAP 空間選択サンプル ({len(matched_umap_samples)}枚 | Epoch {selected_epoch})",
                )

        # Quick class / cluster popup launcher for 3D UMAP classes
        st.markdown("###### 📂 3D UMAP クラス別・境界近傍サンプルのポップアップ確認")
        unique_cls_in_eval = sorted(df_umap_samples["label_name"].unique().tolist())
        q_c1, q_c2 = st.columns([1.6, 1.4])
        chosen_umap_cls = q_c1.selectbox(
            "確認したいクラスを選択",
            options=["(境界に最も近い要注意サンプル Top-15)"] + unique_cls_in_eval,
            index=0,
            key="umap_class_quick_select",
        )
        if q_c2.button("🖼️ ポップアップで画像送り表示", use_container_width=True):
            if chosen_umap_cls.startswith("("):
                boundary_sorted = sorted(
                    umap_source_samples,
                    key=lambda s: abs(float(s["cosine_similarity"]) - float(tau_sim)),
                )[:15]
                _trigger_popup_with_samples(
                    boundary_sorted,
                    f"境界 τ_sim={tau_sim:.2f} 近傍の要注意サンプル Top-15 (Epoch {selected_epoch})",
                )
            else:
                cls_samples = [
                    s for s in umap_source_samples if s["label_name"] == chosen_umap_cls
                ]
                _trigger_popup_with_samples(
                    cls_samples,
                    f"3D UMAP クラス [{chosen_umap_cls}] (全{len(cls_samples)}枚 | Epoch {selected_epoch})",
                )

        # Per-class dispersion table
        per_cls_disp = cur_disp.get("per_class_dispersion", [])
        if per_cls_disp:
            st.markdown(f"###### 📊 Epoch {selected_epoch} [{selected_split.upper()}] クラス別空間分散・中心距離内訳")
            df_pcd = pd.DataFrame(
                [
                    {
                        "クラス": r["label_name"],
                        "枚数": r["count"],
                        "128D内分散": f"{r['intra_cosine_disp']:.4f}",
                        "良品からの距離": f"{r['mean_dist_from_normal']:.4f}",
                        "3D UMAP半径": f"{r['umap_radius_3d']:.3f}",
                    }
                    for r in per_cls_disp
                ]
            )
            st.dataframe(df_pcd, use_container_width=True, hide_index=True, height=150, key="per_cls_disp_df")

    # Inline synchronized Image Inspector (always accessible on page too)
    if st.session_state.get("inspector_samples"):
        with st.expander(
            f"🖼️ インライン画像インスペクタ（現在選択中: {st.session_state.get('inspector_title', '')} — 全 {len(st.session_state['inspector_samples'])} 枚）",
            expanded=True,
        ):
            _render_inspector_body(key_prefix="inline_panel")

    st.divider()

    # =========================================================================
    # SECTION 3 (MOVED DOWN! Was Section 2): Plotly 128-D Raw Vector Heatmap
    # =========================================================================
    hm_hdr_c1, hm_hdr_c2 = st.columns([3.0, 1.0])
    hm_hdr_c1.subheader(
        f"3. 【補助・詳細基準】Plotly 128次元生ベクトル活性化ヒートマップ（次元圧縮なし ｜ 対象: {split_short_badge}）"
    )
    with hm_hdr_c2.popover("ℹ️ 128次元ヒートマップの見方・項目説明 (Tooltip)"):
        st.markdown(
            """
            - **横軸 (`D0 .. D127`)**: L2正規化された128次元特徴ベクトル $z \\in \\mathbb{R}^{128}$ の各次元インデックス。
            - **縦軸 (`[split] クラス名 (ファイルID)`)**: 選択中Splitの良品サンプル群（上部）と各不良サンプル群（下部）。
            - **色 (`Activation`: 赤＝正の活性化、青＝負の活性化)**:
              - **良品行同士で縦の縞模様（赤・青のパターン）が揃っているほど、良品クラスタ内分散が小さく一様に凝集**しています。
              - **不良行で良品とは異なる次元が強く発火（または反転）しているほど、128次元超球面上で直交・分離**できています。
            """
        )
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
            hovertemplate=(
                "サンプル: <b>%{y}</b><br>"
                "次元 (Dimension): <b>%{x}</b><br>"
                "正規化活性値 (z_d): <b>%{z:.4f}</b><br>"
                "💡 <i>良品行と縦の活性パターンが揃うほど類似、反転するほどコサイン距離が遠い</i><extra></extra>"
            ),
        )
    )
    fig_hm.update_layout(
        title=f"Epoch {selected_epoch} [{selected_split.upper()}]: 良品群 vs 各不良モードの128次元単位超球面生ベクトル比較",
        xaxis_title="Embedding Dimension (0 .. 127)",
        yaxis_title="Sample ([Split] Label & ID)",
        height=520,
    )
    st.plotly_chart(fig_hm, use_container_width=True, key="eval_128d_heatmap_chart")

    st.divider()
    st.subheader("4. 単発画像推論テスト (`POST /api/v1/inference`)")
    sample_paths = [s["image_path"] for s in samples_list if s.get("image_path")]
    selected_img = st.selectbox("テスト対象画像を選択", options=sample_paths)
    if st.button("選択した画像で単発推論を実行", disabled=not bool(selected_img)):
        try:
            inf_res = client.run_single_inference(
                image_path=str(selected_img),
                tau_sim=tau_sim_clean,
                session_id=selected_session_id,
            )
            inf_col1, inf_col2 = st.columns([1, 2])
            with inf_col1:
                rp = _resolve_image_file(str(selected_img))
                if rp is not None:
                    with Image.open(rp) as im:
                        st.image(im.copy(), caption=Path(str(selected_img)).name, use_container_width=True)
            with inf_col2:
                st.json(
                    {
                        "session_id": inf_res.get("session_id", selected_session_id),
                        "image_path": inf_res["image_path"],
                        "prediction": inf_res["prediction"],
                        "cosine_similarity": inf_res["cosine_similarity"],
                        "cosine_distance": inf_res["cosine_distance"],
                        "tau_sim": inf_res["tau_sim"],
                    }
                )
        except Exception as exc:
            st.error(f"単発画像推論に失敗しました: {exc}")
