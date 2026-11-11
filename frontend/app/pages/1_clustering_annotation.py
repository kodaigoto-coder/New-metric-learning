"""Phase 1 UI: Multi-Dataset Manager, Folder Upload, Fixed-Normal + Anomaly-Only Clustering & Pseudo-Multiclass Annotation."""

from __future__ import annotations

import base64
import io
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api_client import BackendApiClient

st.set_page_config(page_title="Phase 1: Clustering & Annotation", layout="wide")
client = BackendApiClient()

_VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
_NORMAL_CANDIDATE_NAMES = {"normal", "nomal", "good", "ok", "良品", "正常", "pass"}


def _detect_default_normal_class(class_names: list[str]) -> str:
    """Select the most likely Normal (OK) class name from a list of subfolder names."""
    for name in class_names:
        if name.lower() in _NORMAL_CANDIDATE_NAMES:
            return name
    for name in class_names:
        if any(cand in name.lower() for cand in ("normal", "nomal", "good", "ok", "良品", "正常")):
            return name
    return class_names[0] if class_names else "normal"


def _resolve_subfolder_and_filename(
    rel_path: str,
    shared_root: str | None,
    fallback_class: str,
) -> tuple[str, str] | None:
    """Extract `(subfolder_class_name, safe_file_name)` from a relative path inside an uploaded folder."""
    norm_path = rel_path.replace("\\", "/").strip("/")
    parts = [p for p in norm_path.split("/") if p and p != "."]
    if not parts:
        return None
    if any(p.startswith(".") or p == "__MACOSX" for p in parts):
        return None

    fname = parts[-1]
    if Path(fname).suffix.lower() not in _VALID_IMAGE_EXTS:
        return None

    if shared_root and len(parts) >= 3 and parts[0] == shared_root:
        parts = parts[1:]

    if len(parts) >= 2:
        class_name = parts[0]
        rel_fname = "_".join(parts[1:])
        return class_name, rel_fname

    return (fallback_class.strip() or "normal"), fname


def _parse_uploaded_objects(
    uploaded_files: list[Any],
    fallback_class: str = "normal",
) -> list[dict[str, str]]:
    """Parse Streamlit UploadedFile objects (from directory picker, multi-file, or .zip) into upload items."""
    raw_entries: list[tuple[str, bytes]] = []

    for uf in uploaded_files:
        uf_name = str(getattr(uf, "name", "") or "")
        uf_bytes = uf.getvalue()
        if uf_name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(uf_bytes)) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        raw_entries.append((info.filename, zf.read(info.filename)))
            except zipfile.BadZipFile:
                st.warning(f"ZIPファイルの展開に失敗しました: {uf_name}")
        else:
            raw_entries.append((uf_name, uf_bytes))

    deep_roots: set[str] = set()
    has_shallow_two_part = False
    for rpath, _ in raw_entries:
        parts = [p for p in rpath.replace("\\", "/").strip("/").split("/") if p and p != "."]
        if any(p.startswith(".") or p == "__MACOSX" for p in parts):
            continue
        if len(parts) >= 3:
            deep_roots.add(parts[0])
        elif len(parts) == 2:
            has_shallow_two_part = True

    shared_root = next(iter(deep_roots)) if (len(deep_roots) == 1 and not has_shallow_two_part) else None

    parsed_items: list[dict[str, str]] = []
    for rpath, data_bytes in raw_entries:
        resolved = _resolve_subfolder_and_filename(rpath, shared_root, fallback_class)
        if resolved is None:
            continue
        cls_name, safe_fname = resolved
        parsed_items.append(
            {
                "class_name": cls_name,
                "file_name": safe_fname,
                "content_base64": base64.b64encode(data_bytes).decode("ascii"),
            }
        )

    return parsed_items


st.title("フェーズ1：複数データセット管理・フォルダアップロード ＆ 異常クラス自動細分化クラスタリング")

# Fetch dataset catalog & active dataset
try:
    catalog_data = client.get_datasets_catalog()
except Exception:
    catalog_data = {"active_dataset": "default", "datasets": []}

active_dataset_name: str = catalog_data.get("active_dataset", "default")
all_datasets_meta: list[dict[str, Any]] = catalog_data.get("datasets", [])
dataset_names_list: list[str] = [d["dataset_name"] for d in all_datasets_meta] or ["default"]

# Sidebar quick dataset switcher
with st.sidebar:
    st.markdown("### 📂 アクティブデータセット切替")
    sb_idx = (
        dataset_names_list.index(active_dataset_name)
        if active_dataset_name in dataset_names_list
        else 0
    )
    sb_selected_ds = st.selectbox(
        "検証対象データセット",
        options=dataset_names_list,
        index=sb_idx,
        key="sidebar_ds_select",
        help="切り替えると Phase 1 / Phase 2 / Step 6 の対象データセットが即座に切り替わります。",
    )
    if sb_selected_ds != active_dataset_name:
        client.select_or_create_dataset(dataset_name=sb_selected_ds)
        st.rerun()

# Section 0: Multi-Dataset Storage & Switching Panel
with st.expander("0. 📂 複数データセットの管理（新規作成・別名複製保存・切り替え検証）", expanded=True):
    st.caption(
        "複数の検証データセット（製品品種別・ロット別・テスト条件別など）を独立したディレクトリに分けて保存し、"
        "いつでもワンクリックで切り替えながらクラスタリング・学習・評価を行えます。"
    )
    ds_col1, ds_col2, ds_col3 = st.columns([1.4, 1.8, 1.3])

    with ds_col1:
        st.markdown("##### ① 保存済みデータセットの切り替え")
        cur_idx = (
            dataset_names_list.index(active_dataset_name)
            if active_dataset_name in dataset_names_list
            else 0
        )
        chosen_ds = st.selectbox(
            "現在選択中のデータセット (Active Dataset)",
            options=dataset_names_list,
            index=cur_idx,
            key="main_ds_selector",
        )
        if st.button("🔄 このデータセットに切り替え", use_container_width=True):
            client.select_or_create_dataset(dataset_name=chosen_ds)
            st.success(f"アクティブデータセットを `{chosen_ds}` に切り替えました。")
            st.rerun()

    with ds_col2:
        st.markdown("##### ② 新規データセット作成 ／ 既存データセットの別名保存")
        new_ds_name = st.text_input(
            "新しいデータセット名",
            value="",
            placeholder="例: bottle_dataset_v2, metal_nut, screw_lotB",
            key="new_ds_name_input",
        )
        clone_options = ["(空の新規データセットとして作成)"] + dataset_names_list
        clone_choice = st.selectbox(
            "初期データの引き継ぎ元（任意）",
            options=clone_options,
            index=0,
            key="clone_ds_select",
            help="既存のデータセットを選択すると、その画像とアノテーションを新しいデータセット名へ丸ごと複製して保存します。",
        )
        if st.button("➕ 新規データセットを作成して切り替え", type="primary", use_container_width=True):
            if not new_ds_name.strip():
                st.warning("新しいデータセット名を入力してください。")
            else:
                clone_src = None if clone_choice.startswith("(") else clone_choice
                res_sel = client.select_or_create_dataset(
                    dataset_name=new_ds_name.strip(),
                    clone_from=clone_src,
                )
                st.success(
                    f"データセット `{res_sel['active_dataset']}` を作成し、アクティブに設定しました！"
                )
                st.rerun()

    with ds_col3:
        st.markdown("##### ③ データセット一覧・削除")
        if all_datasets_meta:
            df_cat = pd.DataFrame(
                [
                    {
                        "Active": "✅" if d.get("is_active") else "",
                        "名前": d["dataset_name"],
                        "枚数": d.get("total_images", 0),
                        "クラス": len(d.get("subfolder_classes", {})),
                        "確定済": "済" if d.get("has_annotations") else "未",
                    }
                    for d in all_datasets_meta
                ]
            )
            st.dataframe(df_cat, use_container_width=True, hide_index=True, height=140, key="phase1_cat_df")

        if active_dataset_name != "default":
            if st.button(f"🗑️ `{active_dataset_name}` を削除", use_container_width=True):
                client.delete_dataset(active_dataset_name)
                st.info(f"データセット `{active_dataset_name}` を削除し、`default` に戻しました。")
                st.rerun()

# Fetch active dataset details
try:
    ds_info = client.get_datasets(dataset_name=active_dataset_name)
except Exception:
    ds_info = {
        "dataset_name": active_dataset_name,
        "total_images": 0,
        "subfolder_classes": {},
        "detected_class_names": [],
    }

detected_classes_now: list[str] = ds_info.get("detected_class_names", [])

# Section 1: Dataset Preparation (Folder Upload with Subfolder=Class Name OR Synthetic Generator)
with st.expander(
    f"1. 検査画像データの準備（対象データセット: `{active_dataset_name}` ｜ フォルダ一括アップロード / 合成データ生成）",
    expanded=True,
):
    sub_counts: dict[str, int] = ds_info.get("subfolder_classes", {})
    st.info(
        f"📂 **現在のアクティブデータセット: `{active_dataset_name}`** ｜ "
        f"登録枚数: **合計 {ds_info.get('total_images', 0)} 枚** ｜ "
        f"検出クラス（サブフォルダ内訳）: `{sub_counts if sub_counts else '未登録 / フラット配置'}`"
    )

    tab_upload, tab_synth = st.tabs(
        [
            "📁 フォルダから画像をアップロード（サブフォルダ名 ＝ 初期クラス名）",
            "🧪 100:2 不均衡合成検査データの自動生成（動作確認用）",
        ]
    )

    with tab_upload:
        st.markdown(
            f"""
            **アップロード先データセット**: `{active_dataset_name}`
            * 親フォルダ内にサブフォルダ（例: `nomal/` と `anomaly/`、または `normal/`, `scratch/`, `stain/` など）を入れてアップロードしてください。
            * 他のデータセット（`{[n for n in dataset_names_list if n != active_dataset_name] or ['別名保存データセット']}`）の画像やマニフェストには一切影響しません。
            """
        )
        up_col1, up_col2 = st.columns([2, 1])

        with up_col1:
            uploaded_dir_files: list[Any] = []
            try:
                res_dir = st.file_uploader(
                    "① 親フォルダを選択して一括アップロード（サブフォルダ構造を自動認識）",
                    type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"],
                    accept_multiple_files="directory",
                    key="folder_dir_uploader",
                )
                if res_dir:
                    uploaded_dir_files.extend(res_dir if isinstance(res_dir, list) else [res_dir])
            except Exception:
                pass

            res_zip_or_multi = st.file_uploader(
                "② または ZIP圧縮フォルダ (.zip) / サブフォルダ内の複数画像を選択",
                type=["zip", "png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"],
                accept_multiple_files=True,
                key="folder_zip_or_multi_uploader",
                help="親フォルダを圧縮した .zip ファイル、または単一サブフォルダ内の複数画像をドラッグ＆ドロップできます。",
            )
            if res_zip_or_multi:
                uploaded_dir_files.extend(
                    res_zip_or_multi if isinstance(res_zip_or_multi, list) else [res_zip_or_multi]
                )

        with up_col2:
            fallback_cls_name = st.text_input(
                "サブフォルダ階層を含まない単発画像選択時のクラス名",
                value="normal",
                help="フォルダ選択やZIPではなく、画像ファイル単体を直接選択した場合に割り当てるサブフォルダ（クラス）名です。",
            )
            clear_before_upload = st.checkbox(
                f"アップロード時に `{active_dataset_name}` 内の既存画像をクリアする",
                value=False,
                help="デフォルトOFF（サブフォルダ単位でマージ保存されます）。このデータセット内の既存フォルダを消去して入れ直したい場合のみチェックを入れてください。",
            )

        parsed_upload_items = (
            _parse_uploaded_objects(uploaded_dir_files, fallback_class=fallback_cls_name)
            if uploaded_dir_files
            else []
        )

        if parsed_upload_items:
            preview_counts: dict[str, int] = {}
            for item in parsed_upload_items:
                c_name = item["class_name"]
                preview_counts[c_name] = preview_counts.get(c_name, 0) + 1

            st.markdown(
                f"**アップロード準備完了 (`{active_dataset_name}`): 全 {len(parsed_upload_items)} 枚（検出クラス数: {len(preview_counts)}）**"
            )
            df_preview = pd.DataFrame(
                [{"サブフォルダ名（クラス名）": k, "画像枚数": v} for k, v in sorted(preview_counts.items())]
            )
            st.dataframe(df_preview, use_container_width=True, hide_index=True, key="upload_preview_df")

            if st.button(
                f"📁 データセット `{active_dataset_name}` へ合計 {len(parsed_upload_items)} 枚をアップロード",
                type="primary",
                use_container_width=True,
            ):
                prog = st.progress(0.0, text="画像をサーバーへアップロード中...")

                def _on_prog(done: int, total: int) -> None:
                    prog.progress(done / max(1, total), text=f"アップロード中... ({done}/{total} 枚)")

                up_res = client.upload_folder_images(
                    items=parsed_upload_items,
                    clear_existing=clear_before_upload,
                    dataset_name=active_dataset_name,
                    batch_size=25,
                    progress_callback=_on_prog,
                )
                st.success(
                    f"データセット `{active_dataset_name}` へのアップロード完了！ 合計 {up_res['total_images']} 枚 ｜ "
                    f"クラス内訳: {up_res['subfolder_classes']}"
                )
                st.rerun()

    with tab_synth:
        col_a, col_b, col_c, col_d, col_btn = st.columns([1, 1, 1, 1, 1.5])
        num_normal = col_a.number_input("良品枚数 (Normal)", min_value=20, max_value=1000, value=100, step=10)
        num_scratch = col_b.number_input("不良A: 傷 (Scratch)", min_value=1, max_value=100, value=3, step=1)
        num_stain = col_c.number_input("不良B: 汚れ (Stain)", min_value=1, max_value=100, value=3, step=1)
        num_dent = col_d.number_input("不良C: 打痕 (Dent)", min_value=0, max_value=100, value=2, step=1)

        if col_btn.button(f"`{active_dataset_name}` に合成データを生成", type="primary", use_container_width=True):
            with st.spinner(f"データセット `{active_dataset_name}` に 100:2 合成検査画像を生成中..."):
                res = client.generate_synthetic_dataset(
                    {
                        "dataset_name": active_dataset_name,
                        "num_normal": int(num_normal),
                        "num_scratch": int(num_scratch),
                        "num_stain": int(num_stain),
                        "num_dent": int(num_dent),
                        "image_size": 128,
                        "seed": 42,
                        "clear_existing": True,
                    }
                )
                st.success(f"`{active_dataset_name}` に合計 {res['total_generated']} 枚を生成しました: {res['counts']}")
                st.rerun()

# Section 2: Trigger Anomaly-Only Clustering (Normal Fixed as Cluster 0)
st.subheader(f"2. 既製モデル特徴抽出 ＆ 異常クラス自動細分化クラスタリング (対象: `{active_dataset_name}`)")
st.caption(
    "良品クラス（例: `nomal` / `normal`）は一切クラスタリングせずそのまま `Cluster 0` として固定し、"
    "異常クラス（例: `anomaly`）の画像のみを PCA ＋ シルエットスコア最大化 K-means で複数の不良サブクラス（`Cluster 1 .. k*`）へ自動分割します。"
)

c1, c2, c3, c4, c5 = st.columns([1.3, 1.0, 1.3, 1.0, 1.1])
backbone_name = c1.selectbox(
    "特徴抽出バックボーン (timm)",
    options=["efficientnet_b3", "vit_base_patch16_224", "resnet18", "lightweight_cnn"],
    index=0,
)
pooling_type = c2.selectbox(
    "空間プーリング層",
    options=["gem", "avg_max", "gap"],
    index=0,
    help="GeM または Avg+Max プーリングにより微小な局所傷・汚れの希釈を防ぎます。",
)

norm_candidates_for_cluster = detected_classes_now if detected_classes_now else ["normal", "nomal", "good"]
default_norm_cluster = _detect_default_normal_class(norm_candidates_for_cluster)
norm_cluster_idx = (
    norm_candidates_for_cluster.index(default_norm_cluster)
    if default_norm_cluster in norm_candidates_for_cluster
    else 0
)
normal_class_for_cluster = c3.selectbox(
    "良品クラス名（クラスタリングせず固定保持）",
    options=norm_candidates_for_cluster,
    index=norm_cluster_idx,
    help="このサブフォルダの画像はクラスタリングせず、全件そのまま Cluster 0（良品）として固定します。",
)
max_k = c4.number_input(
    "異常クラスの最大分割数 (K_max)",
    min_value=2,
    max_value=15,
    value=6,
    help="異常フォルダ内の画像のみを対象に k = 2 〜 K_max の範囲でシルエットスコアが最大となるクラスタ数 k* を自動選定します。",
)
use_pretrained_cluster = c5.checkbox("ImageNet事前学習重み", value=True)

if st.button(
    f"データセット `{active_dataset_name}` の異常クラス自動細分化クラスタリングを実行",
    type="primary",
):
    job = client.trigger_clustering(
        {
            "dataset_name": active_dataset_name,
            "backbone_name": backbone_name,
            "pooling_type": pooling_type,
            "normal_class_name": normal_class_for_cluster,
            "core_quantile": 0.85,
            "k_nn": 15,
            "max_k": int(max_k),
            "max_pca_dim": 64,
            "pca_variance": 0.95,
            "pretrained": bool(use_pretrained_cluster),
        }
    )
    job_id = job["job_id"]
    prog_bar = st.progress(0.0, text="クラスタリングジョブ起動中...")
    while True:
        status = client.get_job_status(job_id)
        prog_bar.progress(float(status.get("progress", 0.0)), text=status.get("message", ""))
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)
    if status["status"] == "completed":
        st.success(status["message"])
    else:
        st.error(f"クラスタリング失敗: {status.get('error')}")

# Section 3: Inspect Clusters & Confirm Pseudo-Multiclass Labels (Step 3)
st.divider()
st.subheader(f"3. クラスタリング結果の確認 ＆ 疑似多クラスラベル確定 (対象: `{active_dataset_name}`)")

try:
    cluster_data = client.get_clusters(dataset_name=active_dataset_name)
except Exception:
    cluster_data = None

if not cluster_data:
    st.info(
        f"データセット `{active_dataset_name}` にはまだクラスタリング結果がありません。"
        "上の「異常クラス自動細分化クラスタリングを実行」ボタンを押してください。"
    )
else:
    norm_cls_in_res = cluster_data.get("normal_class_name", normal_class_for_cluster)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("総サンプル数", cluster_data["total_samples"])
    m2.metric(f"良品クラス固定数 (Cluster 0: `{norm_cls_in_res}`)", cluster_data["normal_core_count"])
    m3.metric("異常クラスタリング対象数", cluster_data["tail_count"])
    m4.metric("自動選定された異常分割クラス数 k*", cluster_data["best_tail_k"])

    items = cluster_data["items"]
    df_items = pd.DataFrame(items)
    df_items["x"] = df_items["preview_2d"].apply(lambda v: v[0])
    df_items["y"] = df_items["preview_2d"].apply(lambda v: v[1])
    df_items["cluster_str"] = df_items["cluster_id"].apply(
        lambda cid: f"Cluster 0 (良品固定: {norm_cls_in_res})"
        if cid == 0
        else f"Cluster {cid} (異常サブクラス {cid})"
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        sil_scores = cluster_data.get("silhouette_scores", {})
        if sil_scores:
            df_sil = pd.DataFrame(
                [{"k": f"k={k}", "Silhouette Score": v} for k, v in sil_scores.items()]
            )
            fig_sil = px.bar(
                df_sil,
                x="k",
                y="Silhouette Score",
                title="異常データ領域 K-means シルエットスコア推移（最大値 k* を自動選定）",
                text_auto=".4f",
            )
            st.plotly_chart(fig_sil, use_container_width=True, key="phase1_sil_chart")

    with chart_col2:
        fig_2d = px.scatter(
            df_items,
            x="x",
            y="y",
            color="cluster_str",
            symbol="ground_truth_hint",
            hover_data=["file_name", "density_score", "ground_truth_hint"],
            title="2D PCA 特徴空間プレビュー（Cluster 0: 良品固定 vs Cluster 1..k*: 異常サブクラス）",
        )
        st.plotly_chart(fig_2d, use_container_width=True, key="phase1_pca_2d_chart")

    st.markdown("#### クラスタ単位のラベル割当（良品は固定 / 異常は分割サブクラス化）")

    suggested_map: dict[str, str] = cluster_data.get("suggested_cluster_labels", {})
    detected_from_clusters = [str(x) for x in cluster_data.get("detected_classes", []) if x]
    hint_values = [str(x) for x in df_items["ground_truth_hint"].dropna().unique().tolist() if x]
    suggested_values = [str(v) for v in suggested_map.values() if v]

    unique_clusters = sorted(df_items["cluster_id"].unique())
    auto_subcluster_names = [f"anomaly_cluster_{cid}" for cid in unique_clusters if cid > 0]

    cfg_col1, cfg_col2, cfg_col3 = st.columns([1.4, 1.6, 2.0])
    custom_labels_raw = cfg_col3.text_input(
        "追加のカスタム不良クラス名（カンマ区切りで任意追加）",
        value="",
        placeholder="例: broken_large, broken_small, contamination",
    )
    custom_labels = [x.strip() for x in custom_labels_raw.split(",") if x.strip()]

    dynamic_classes: list[str] = []
    for candidate in (
        [norm_cls_in_res]
        + suggested_values
        + auto_subcluster_names
        + custom_labels
        + detected_classes_now
        + detected_from_clusters
        + hint_values
        + ["normal", "defect_scratch", "defect_stain", "defect_dent", "defect_other"]
    ):
        if candidate and candidate not in dynamic_classes:
            dynamic_classes.append(candidate)

    normal_cls_idx = (
        dynamic_classes.index(norm_cls_in_res) if norm_cls_in_res in dynamic_classes else 0
    )
    normal_class_name = cfg_col1.selectbox(
        "良品（正常: label_id=0）として扱うクラス名",
        options=dynamic_classes,
        index=normal_cls_idx,
        help="Cluster 0（良品）に割り当てるクラス名です（label_id=0, is_defect=False として固定されます）。",
    )
    use_subfolder_labels = cfg_col2.checkbox(
        "異常クラスタ分割を使わず、元のサブフォルダ名をそのまま適用する",
        value=False,
        help="通常はOFF（推奨）のままにしてください。OFFにすると、異常データが Cluster 1..k* の複数サブクラスに分かれた状態でフェーズ2の学習へ渡されます。",
    )

    cluster_labels: dict[str, str] = {}

    for cid in unique_clusters:
        sub_df = df_items[df_items["cluster_id"] == cid].sort_values("density_score")
        hints = sub_df["ground_truth_hint"].dropna().value_counts().to_dict()

        if cid == 0:
            default_lbl = normal_class_name
        else:
            default_lbl = suggested_map.get(str(cid), f"anomaly_cluster_{cid}")

        default_idx = dynamic_classes.index(default_lbl) if default_lbl in dynamic_classes else 0

        with st.container(border=True):
            top_c1, top_c2 = st.columns([2, 3])
            role_title = (
                f"Cluster 0 — 良品クラス固定 ({len(sub_df)}枚 / クラスタリング対象外)"
                if cid == 0
                else f"Cluster {cid} — 異常データの自動分割サブクラス {cid} ({len(sub_df)}枚)"
            )
            top_c1.markdown(f"**{role_title}**")
            top_c1.caption(f"元サブフォルダ内訳: {hints}")
            chosen_label = top_c2.selectbox(
                f"Cluster {cid} の確定クラス名",
                options=dynamic_classes,
                index=default_idx,
                key=f"cluster_lbl_{active_dataset_name}_{cid}",
            )
            cluster_labels[str(cid)] = chosen_label

            sample_rows = sub_df.head(6).to_dict("records")
            img_cols = st.columns(min(6, max(1, len(sample_rows))))
            for idx, row in enumerate(sample_rows):
                img_p = Path(row["image_path"])
                sub_cls = row.get("ground_truth_hint") or "-"
                if img_p.exists():
                    img_cols[idx].image(
                        Image.open(img_p),
                        caption=f"[{sub_cls}] {row['file_name']}\n(sim={row['density_score']:.3f})",
                        use_container_width=True,
                    )
                else:
                    img_cols[idx].caption(f"[{sub_cls}] {row['file_name']}")

    with st.expander("個別サンプルのラベル手動上書き（任意）"):
        override_text = st.text_area(
            "1行に1件 `ファイル名=クラス名` の形式で入力（例: `0001.png=anomaly_cluster_2`）",
            value="",
            height=80,
        )
        sample_overrides: dict[str, str] = {}
        for line in override_text.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                sample_overrides[k.strip()] = v.strip()

    if st.button(
        f"ラベルを確定してデータセット `{active_dataset_name}` のマニフェスト (annotations.json) を保存",
        type="primary",
    ):
        saved = client.save_annotations(
            {
                "dataset_name": active_dataset_name,
                "cluster_labels": cluster_labels,
                "sample_overrides": sample_overrides,
                "normal_class_name": normal_class_name,
                "use_subfolder_labels": bool(use_subfolder_labels),
                "val_ratio": 0.25,
                "seed": 42,
            }
        )
        st.success(
            f"データセット `{active_dataset_name}` のアノテーション確定完了（全 {saved['total_samples']} 件, 良品クラス=`{saved.get('normal_class_name', normal_class_name)}`）: "
            f"学習クラス内訳 = {saved['class_counts']} ｜ クラスIDマップ = {saved['class_map']}"
        )
