"""Phase 1 UI: Folder Upload (Subfolder = Class Name), 2-Stage Clustering & Pseudo-Multiclass Annotation."""

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
_NORMAL_CANDIDATE_NAMES = {"normal", "good", "ok", "良品", "正常", "pass"}


def _detect_default_normal_class(class_names: list[str]) -> str:
    """Select the most likely Normal (OK) class name from a list of subfolder names."""
    for name in class_names:
        if name.lower() in _NORMAL_CANDIDATE_NAMES:
            return name
    for name in class_names:
        if any(cand in name.lower() for cand in ("normal", "good", "ok", "良品", "正常")):
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

    # Strip shared root directory if user selected a top-level parent directory (e.g. `dataset/normal/001.png`)
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

    # Determine if all entries with depth >= 3 share a common parent root folder name
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


st.title("フェーズ1：フォルダ画像アップロード・2段階クラスタリング ＆ クラスアノテーション")

# Fetch current dataset info
try:
    ds_info = client.get_datasets()
except Exception:
    ds_info = {"total_images": 0, "subfolder_classes": {}, "detected_class_names": []}

# Section 1: Dataset Preparation (Folder Upload with Subfolder=Class Name OR Synthetic Generator)
with st.expander("1. 検査画像データの準備（フォルダ一括アップロード［サブフォルダ名＝クラス名］ / 合成データ生成）", expanded=True):
    sub_counts: dict[str, int] = ds_info.get("subfolder_classes", {})
    if ds_info.get("total_images", 0) > 0:
        st.info(
            f"現在のデータセット登録枚数: **合計 {ds_info['total_images']} 枚** ｜ "
            f"検出クラス（サブフォルダ内訳）: `{sub_counts if sub_counts else 'フラット配置'}`"
        )

    tab_upload, tab_synth = st.tabs(
        [
            "📁 フォルダから画像をアップロード（サブフォルダ名 ＝ 学習クラス名）",
            "🧪 100:2 不均衡合成検査データの自動生成（動作確認用）",
        ]
    )

    with tab_upload:
        st.markdown(
            """
            **フォルダ構成のルール**:
            親フォルダ内にクラスごとのサブフォルダ（例: `normal/`, `scratch/`, `stain/` や `OK/`, `打痕/`, `傷/` など）を入れてアップロードしてください。
            **サブフォルダ名が自動的にクラス名として抽出**され、クラスタリング時の初期ラベルおよびフェーズ2（MS-Loss学習）のクラス名として使用できます。
            """
        )
        up_col1, up_col2 = st.columns([2, 1])

        with up_col1:
            uploaded_dir_files: list[Any] = []
            try:
                # Streamlit >= 1.49 supports native directory picker via accept_multiple_files="directory"
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
                "アップロード時に既存の登録画像をクリアする",
                value=True,
                help="サブフォルダを1つずつ複数回に分けて追加アップロードする場合はチェックを外してください。",
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
                f"**アップロード準備完了: 全 {len(parsed_upload_items)} 枚（検出されたサブフォルダクラス数: {len(preview_counts)}）**"
            )
            df_preview = pd.DataFrame(
                [{"サブフォルダ名（クラス名）": k, "画像枚数": v} for k, v in sorted(preview_counts.items())]
            )
            st.dataframe(df_preview, use_container_width=True, hide_index=True)

            if st.button(
                f"📁 合計 {len(parsed_upload_items)} 枚の画像をサブフォルダクラス構造でアップロード",
                type="primary",
                use_container_width=True,
            ):
                prog = st.progress(0.0, text="画像をサーバーへアップロード中...")

                def _on_prog(done: int, total: int) -> None:
                    prog.progress(done / max(1, total), text=f"アップロード中... ({done}/{total} 枚)")

                up_res = client.upload_folder_images(
                    items=parsed_upload_items,
                    clear_existing=clear_before_upload,
                    batch_size=25,
                    progress_callback=_on_prog,
                )
                st.success(
                    f"アップロード完了！ 合計 {up_res['total_images']} 枚 ｜ "
                    f"クラス内訳: {up_res['subfolder_classes']}"
                )
                st.rerun()

        # Quick direct annotation confirmation from subfolder names
        detected_classes_now: list[str] = ds_info.get("detected_class_names", [])
        if detected_classes_now:
            st.divider()
            st.markdown("##### サブフォルダ名をそのまま学習クラス名として設定")
            q_col1, q_col2 = st.columns([2, 3])
            default_norm = _detect_default_normal_class(detected_classes_now)
            norm_idx = (
                detected_classes_now.index(default_norm)
                if default_norm in detected_classes_now
                else 0
            )
            selected_normal_cls = q_col1.selectbox(
                "良品（正常: label_id=0）として扱うサブフォルダ名",
                options=detected_classes_now,
                index=norm_idx,
                key="direct_normal_cls_select",
                help="ここで選択したサブフォルダが『良品（正常）』となり、それ以外の全サブフォルダが『不良モード（各サブフォルダ名）』として学習されます。",
            )
            if q_col2.button(
                "サブフォルダ名をそのまま学習用クラス名として確定 (annotations.json 生成)",
                use_container_width=True,
                help="下の2段階クラスタリングで特徴空間を確認する前でも、サブフォルダ名をそのまま正解クラス名としてPhase 2学習用マニフェストに保存できます。",
            ):
                saved_direct = client.save_annotations_from_subfolders(
                    {
                        "normal_class_name": selected_normal_cls,
                        "val_ratio": 0.25,
                        "seed": 42,
                    }
                )
                st.success(
                    f"サブフォルダ名から学習用クラスを確定しました（全 {saved_direct['total_samples']} 件, "
                    f"良品クラス=`{saved_direct['normal_class_name']}`）: クラス内訳 = {saved_direct['class_counts']}"
                )

    with tab_synth:
        col_a, col_b, col_c, col_d, col_btn = st.columns([1, 1, 1, 1, 1.5])
        num_normal = col_a.number_input("良品枚数 (Normal)", min_value=20, max_value=1000, value=100, step=10)
        num_scratch = col_b.number_input("不良A: 傷 (Scratch)", min_value=1, max_value=100, value=3, step=1)
        num_stain = col_c.number_input("不良B: 汚れ (Stain)", min_value=1, max_value=100, value=3, step=1)
        num_dent = col_d.number_input("不良C: 打痕 (Dent)", min_value=0, max_value=100, value=2, step=1)

        if col_btn.button("合成検査データセットを生成", type="primary", use_container_width=True):
            with st.spinner("100:2 不均衡合成検査画像を生成中..."):
                res = client.generate_synthetic_dataset(
                    {
                        "num_normal": int(num_normal),
                        "num_scratch": int(num_scratch),
                        "num_stain": int(num_stain),
                        "num_dent": int(num_dent),
                        "image_size": 128,
                        "seed": 42,
                        "clear_existing": True,
                    }
                )
                st.success(f"合計 {res['total_generated']} 枚の検査画像を生成しました: {res['counts']}")
                st.rerun()

# Section 2: Trigger 2-Stage Clustering
st.subheader("2. 既製モデル特徴抽出 ＆ 2段階密度分離クラスタリング (Step 1〜2)")
c1, c2, c3, c4, c5 = st.columns(5)
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
core_quantile = c3.slider(
    "Stage 1 良品コア抽出分位 (τ_core)",
    min_value=0.60,
    max_value=0.95,
    value=0.85,
    step=0.05,
)
k_nn = c4.number_input("局所密度 k-NN 近傍数", min_value=3, max_value=50, value=15)
max_k = c5.number_input("Stage 2 最大探索クラスタ数 (K_max)", min_value=3, max_value=15, value=6)

if st.button("2段階クラスタリングを実行 (Stage 1: 良品コア分離 → Stage 2: テールPCA + シルエットK-means)", type="primary"):
    job = client.trigger_clustering(
        {
            "backbone_name": backbone_name,
            "pooling_type": pooling_type,
            "core_quantile": float(core_quantile),
            "k_nn": int(k_nn),
            "max_k": int(max_k),
            "max_pca_dim": 64,
            "pca_variance": 0.95,
            "pretrained": False,
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
st.subheader("3. クラスタリング結果の確認 ＆ 疑似多クラスラベル確定 (Step 3)")

try:
    cluster_data = client.get_clusters()
except Exception:
    cluster_data = None

if not cluster_data:
    st.info("まだクラスタリング結果がありません。上の「2段階クラスタリングを実行」ボタンを押してください。")
else:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("総サンプル数", cluster_data["total_samples"])
    m2.metric("Stage 1: 良品コア抽出数 (Cluster 0)", cluster_data["normal_core_count"])
    m3.metric("Stage 2: 低密度テール対象数", cluster_data["tail_count"])
    m4.metric("自動選定された最適テール k*", cluster_data["best_tail_k"])

    items = cluster_data["items"]
    df_items = pd.DataFrame(items)
    df_items["x"] = df_items["preview_2d"].apply(lambda v: v[0])
    df_items["y"] = df_items["preview_2d"].apply(lambda v: v[1])
    df_items["cluster_str"] = df_items["cluster_id"].apply(
        lambda cid: "Cluster 0 (Stage 1: Normal Core)" if cid == 0 else f"Cluster {cid} (Stage 2: Tail)"
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
                title="Stage 2: テール領域 K-means シルエットスコア推移（最大値を自動選定）",
                text_auto=".4f",
            )
            st.plotly_chart(fig_sil, use_container_width=True)

    with chart_col2:
        fig_2d = px.scatter(
            df_items,
            x="x",
            y="y",
            color="cluster_str",
            symbol="ground_truth_hint",
            hover_data=["file_name", "density_score", "ground_truth_hint"],
            title="2D PCA 特徴空間プレビュー（Stage 1 良品コア vs Stage 2 テールクラスタ / 記号＝サブフォルダ名）",
        )
        st.plotly_chart(fig_2d, use_container_width=True)

    st.markdown("#### クラスタ単位のラベル割当（サブフォルダ名を自動反映）＆ スポットチェック")

    # Dynamically assemble available class label options from uploaded subfolders + cluster hints + custom input
    detected_from_clusters = [
        str(x) for x in cluster_data.get("detected_classes", []) if x
    ]
    detected_from_dataset = [
        str(x) for x in ds_info.get("detected_class_names", []) if x
    ]
    hint_values = [
        str(x) for x in df_items["ground_truth_hint"].dropna().unique().tolist() if x
    ]

    cfg_col1, cfg_col2, cfg_col3 = st.columns([1.5, 1.5, 2])
    custom_labels_raw = cfg_col3.text_input(
        "追加のカスタムクラス名（カンマ区切りで任意追加）",
        value="",
        placeholder="例: defect_burr, defect_crack",
    )
    custom_labels = [x.strip() for x in custom_labels_raw.split(",") if x.strip()]

    dynamic_classes: list[str] = []
    for candidate in (
        detected_from_dataset
        + detected_from_clusters
        + hint_values
        + custom_labels
        + ["normal", "defect_scratch", "defect_stain", "defect_dent", "defect_other"]
    ):
        if candidate and candidate not in dynamic_classes:
            dynamic_classes.append(candidate)

    default_normal_cls = _detect_default_normal_class(
        detected_from_dataset or detected_from_clusters or dynamic_classes
    )
    normal_cls_idx = (
        dynamic_classes.index(default_normal_cls)
        if default_normal_cls in dynamic_classes
        else 0
    )
    normal_class_name = cfg_col1.selectbox(
        "良品（正常: label_id=0）として扱うクラス名",
        options=dynamic_classes,
        index=normal_cls_idx,
        help="選択したクラス名が良品（label_id=0）となり、それ以外のクラス名が不良モード（label_id=1..C）としてPhase 2で学習されます。",
    )
    use_subfolder_labels = cfg_col2.checkbox(
        "各画像の元サブフォルダ名を優先してクラス名に適用する",
        value=bool(detected_from_dataset or detected_from_clusters),
        help="ONにすると、クラスタ内に他クラスの画像が混入していても、アップロード時のサブフォルダ名を各画像の学習クラス名として正確に維持します。",
    )

    cluster_labels: dict[str, str] = {}
    unique_clusters = sorted(df_items["cluster_id"].unique())

    for cid in unique_clusters:
        sub_df = df_items[df_items["cluster_id"] == cid].sort_values("density_score")
        hints = sub_df["ground_truth_hint"].dropna().value_counts().to_dict()
        dominant_hint = next(iter(hints.keys()), normal_class_name) if hints else (
            normal_class_name if cid == 0 else dynamic_classes[min(1, len(dynamic_classes) - 1)]
        )
        default_idx = dynamic_classes.index(dominant_hint) if dominant_hint in dynamic_classes else 0

        with st.container(border=True):
            top_c1, top_c2 = st.columns([2, 3])
            role_title = (
                f"Cluster 0 — Stage 1 高密度良品コア ({len(sub_df)}枚) ※境界付近サンプルの抜き取りチェック"
                if cid == 0
                else f"Cluster {cid} — Stage 2 低密度テールクラスタ ({len(sub_df)}枚)"
            )
            top_c1.markdown(f"**{role_title}**")
            top_c1.caption(f"サブフォルダ（クラス）内訳: {hints}")
            chosen_label = top_c2.selectbox(
                f"Cluster {cid} の確定ラベル",
                options=dynamic_classes,
                index=default_idx,
                key=f"cluster_lbl_{cid}",
            )
            cluster_labels[str(cid)] = chosen_label

            # Show up to 6 representative images (lowest density first for spot checking)
            sample_rows = sub_df.head(6).to_dict("records")
            img_cols = st.columns(min(6, max(1, len(sample_rows))))
            for idx, row in enumerate(sample_rows):
                img_p = Path(row["image_path"])
                sub_cls = row.get("ground_truth_hint") or "-"
                if img_p.exists():
                    img_cols[idx].image(
                        Image.open(img_p),
                        caption=f"[{sub_cls}] {row['file_name']}\n(density={row['density_score']:.3f})",
                        use_container_width=True,
                    )
                else:
                    img_cols[idx].caption(f"[{sub_cls}] {row['file_name']}")

    with st.expander("混入外れ値サンプルの個別ラベル上書き（任意）"):
        override_text = st.text_area(
            "1行に1件 `ファイル名=クラス名` の形式で入力（例: `0001.png=scratch`）",
            value="",
            height=80,
        )
        sample_overrides: dict[str, str] = {}
        for line in override_text.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                sample_overrides[k.strip()] = v.strip()

    if st.button("ラベルを確定してフェーズ2用マニフェスト (annotations.json) を保存", type="primary"):
        saved = client.save_annotations(
            {
                "cluster_labels": cluster_labels,
                "sample_overrides": sample_overrides,
                "normal_class_name": normal_class_name,
                "use_subfolder_labels": bool(use_subfolder_labels),
                "val_ratio": 0.25,
                "seed": 42,
            }
        )
        st.success(
            f"アノテーション確定完了（全 {saved['total_samples']} 件, 良品クラス=`{saved.get('normal_class_name', normal_class_name)}`）: "
            f"学習クラス内訳 = {saved['class_counts']} ｜ クラスIDマップ = {saved['class_map']}"
        )
