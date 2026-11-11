"""Phase 1 UI: Multi-Dataset Manager, Folder Upload, Subfolder Class Manager, Anomaly-Only Clustering & Interactive Label Editor."""

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
    """Extract `(subfolder_class_name, safe_file_name)` from a relative path inside an uploaded folder.

    Uses the immediate parent directory (`parts[-2]`) as the class name so nested structures
    like `transistor/test/bent_lead/000.png` or `transistor/train/good/000.png` resolve to
    class `bent_lead` and `good`, while prefixing intermediate folders (`test_000.png`,
    `train_000.png`) so identical filenames in `train/good` and `test/good` never collide.
    """
    norm_path = rel_path.replace("\\", "/").strip("/")
    parts = [p for p in norm_path.split("/") if p and p != "."]
    if not parts:
        return None
    if any(p.startswith(".") or p == "__MACOSX" or p == "ground_truth" for p in parts):
        return None

    fname = parts[-1]
    if Path(fname).suffix.lower() not in _VALID_IMAGE_EXTS:
        return None

    if shared_root and len(parts) >= 3 and parts[0] == shared_root:
        parts = parts[1:]

    if len(parts) >= 2:
        class_name = parts[-2]
        prefix_parts = parts[:-2]
        rel_fname = "_".join(prefix_parts + [fname]) if prefix_parts else fname
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
        if not parts:
            continue
        if any(p.startswith(".") or p == "__MACOSX" or p == "ground_truth" for p in parts):
            continue
        if Path(parts[-1]).suffix.lower() not in _VALID_IMAGE_EXTS:
            continue
        if len(parts) >= 3:
            deep_roots.add(parts[0])
        elif len(parts) == 2:
            has_shallow_two_part = True

    shared_root = (
        next(iter(deep_roots)) if (len(deep_roots) == 1 and not has_shallow_two_part) else None
    )

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


def _activate_dataset_and_sync_state(target_ds: str, clone_from: str | None = None) -> dict[str, Any]:
    """Switch the backend active dataset and mark session_state selectors for pre-widget sync on next run."""
    res = client.select_or_create_dataset(dataset_name=target_ds, clone_from=clone_from)
    # Do NOT assign to already-instantiated widget keys (`sidebar_ds_select`, `main_ds_selector`) here;
    # resetting `_synced_active_ds` triggers pre-widget synchronization at the top of the rerun.
    st.session_state["_synced_active_ds"] = ""
    return res


st.title("フェーズ1：複数データセット管理・フォルダアップロード ＆ クラス／ラベル管理")

# Fetch dataset catalog & active dataset
try:
    catalog_data = client.get_datasets_catalog()
except Exception:
    catalog_data = {"active_dataset": "default", "datasets": []}

active_dataset_name: str = catalog_data.get("active_dataset", "default")
all_datasets_meta: list[dict[str, Any]] = catalog_data.get("datasets", [])
dataset_names_list: list[str] = [d["dataset_name"] for d in all_datasets_meta] or ["default"]

# Synchronize session_state selectors whenever backend active_dataset_name changes
if (
    st.session_state.get("_synced_active_ds") != active_dataset_name
    or st.session_state.get("sidebar_ds_select") not in dataset_names_list
):
    st.session_state["_synced_active_ds"] = active_dataset_name
    st.session_state["sidebar_ds_select"] = active_dataset_name
    st.session_state["main_ds_selector"] = active_dataset_name


def _on_sidebar_ds_change() -> None:
    chosen = st.session_state.get("sidebar_ds_select")
    if chosen and chosen != st.session_state.get("_synced_active_ds"):
        _activate_dataset_and_sync_state(str(chosen))


# Sidebar quick dataset switcher (uses on_change to prevent state snap-back)
with st.sidebar:
    st.markdown("### 📂 アクティブデータセット切替")
    st.selectbox(
        "検証対象データセット",
        options=dataset_names_list,
        key="sidebar_ds_select",
        on_change=_on_sidebar_ds_change,
        help="切り替えると Phase 1 / Phase 2 / Step 6 の対象データセットが即座に切り替わります。",
    )
    st.caption(f"現在のアクティブDB: **`{active_dataset_name}`**")

# Section 0: Multi-Dataset Storage & Switching Panel
with st.expander("0. 📂 複数データセットの管理（新規作成・別名複製保存・切り替え検証）", expanded=True):
    st.caption(
        "複数の検証データセット（製品品種別・ロット別・テスト条件別など）を独立したディレクトリに分けて保存し、"
        "いつでもワンクリックで切り替えながらクラスタリング・学習・評価を行えます。"
    )
    ds_col1, ds_col2, ds_col3 = st.columns([1.4, 1.8, 1.3])

    with ds_col1:
        st.markdown("##### ① 保存済みデータセットの切り替え")
        chosen_ds = st.selectbox(
            "切り替え先データセットを選択",
            options=dataset_names_list,
            key="main_ds_selector",
        )
        if st.button("🔄 このデータセットに切り替え", use_container_width=True):
            _activate_dataset_and_sync_state(chosen_ds)
            st.toast(f"データセットを `{chosen_ds}` に切り替えました。")
            st.rerun()

    with ds_col2:
        st.markdown("##### ② 新規データセット作成 ／ 既存データセットの別名保存")
        new_ds_name = st.text_input(
            "新しいデータセット名",
            value="",
            placeholder="例: transistor, bottle_v2, metal_nut",
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
                res_sel = _activate_dataset_and_sync_state(
                    target_ds=new_ds_name.strip(),
                    clone_from=clone_src,
                )
                st.toast(f"データセット `{res_sel['active_dataset']}` を作成・アクティブ化しました！")
                st.rerun()

    with ds_col3:
        st.markdown("##### ③ データセット一覧・初期化/削除")
        if all_datasets_meta:
            df_cat = pd.DataFrame(
                [
                    {
                        "Active": "✅" if d.get("is_active") else "",
                        "名前": d["dataset_name"],
                        "枚数": d.get("total_images", 0),
                        "フォルダ数": len(d.get("subfolder_classes", {})),
                        "確定済": "済" if d.get("has_annotations") else "未",
                    }
                    for d in all_datasets_meta
                ]
            )
            st.dataframe(df_cat, use_container_width=True, hide_index=True, height=140, key="phase1_cat_df")

        del_btn_label = (
            f"🗑️ `{active_dataset_name}` の全データをクリア"
            if active_dataset_name == "default"
            else f"🗑️ `{active_dataset_name}` を削除"
        )
        if st.button(del_btn_label, use_container_width=True):
            client.delete_dataset(active_dataset_name)
            _activate_dataset_and_sync_state("default")
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
sub_counts: dict[str, int] = ds_info.get("subfolder_classes", {})

# Check if annotations.json already exists for this dataset
try:
    existing_ann = client.get_annotations(dataset_name=active_dataset_name)
except Exception:
    existing_ann = None

# Section 1: Dataset Preparation (Folder Upload + Subfolder Class Manager + Direct Registration)
with st.expander(
    f"1. 検査画像データの登録 ＆ サブフォルダ（クラス）管理（対象データセット: `{active_dataset_name}`）",
    expanded=True,
):
    st.info(
        f"📂 **現在のアクティブデータセット: `{active_dataset_name}`** ｜ "
        f"登録画像数: **合計 {ds_info.get('total_images', 0)} 枚** ｜ "
        f"サブフォルダ内訳: `{sub_counts if sub_counts else '未登録'}` ｜ "
        f"確定済みアノテーション: "
        + (
            f"**登録済 ({existing_ann['total_samples']}枚, 良品=`{existing_ann.get('normal_class_name')}`, 内訳={existing_ann.get('class_counts')})**"
            if existing_ann
            else "**未確定**"
        )
    )

    tab_upload, tab_class_mgr, tab_synth = st.tabs(
        [
            "📁 ① フォルダ／ZIPから画像をアップロード（サブフォルダ名＝クラス名）",
            "🛠️ ② 登録済みサブフォルダ（クラス）の名称変更・統合・個別削除 ＆ 即時ラベル確定",
            "🧪 ③ 100:2 不均衡合成検査データの自動生成（動作確認用）",
        ]
    )

    with tab_upload:
        st.markdown(
            f"""
            **アップロード先データセット**: **`{active_dataset_name}`**
            * 親フォルダ（例: `transistor/test/` や `dataset/`）を選択すると、各画像の**直上の親サブフォルダ名**（例: `good`, `bent_lead`, `cut_lead`, `nomal`, `anomaly` 等）が自動的にクラス名として認識されます（`ground_truth` マスクフォルダは自動除外されます）。
            * サブフォルダ単位で複数回に分けてアップロードしても、既存サブフォルダと安全にマージ保存されます。
            """
        )
        up_col1, up_col2 = st.columns([2, 1])

        with up_col1:
            uploaded_dir_files: list[Any] = []
            # Note: `type=None` is critical for `accept_multiple_files="directory"` in Streamlit 1.67+.
            # When `type=[...]` is passed, Streamlit's `FileUploaderSerde.deserialize()` runs
            # `enforce_filename_restriction()` on every file in the directory tree and throws
            # `StreamlitInvalidFileExtensionError` if `.DS_Store` (macOS) or `readme.txt` / `license.txt`
            # (MVTec AD) exists anywhere in the selected folder. We filter valid images in Python below.
            res_dir = st.file_uploader(
                "① フォルダを選択して一括アップロード（直上の親サブフォルダ名を自動認識・選択と同時に自動保存）",
                type=None,
                accept_multiple_files="directory",
                max_upload_size=2000,
                key=f"folder_dir_uploader_{active_dataset_name}",
                help="親フォルダ（例: bottle, transistor/test）やクラス別フォルダ（例: nomal, anomaly）を選択すると、.DS_Store や ground_truth を自動除外してデータセットへ即時保存します。",
            )
            if res_dir:
                uploaded_dir_files.extend(res_dir if isinstance(res_dir, list) else [res_dir])

            res_zip_or_multi = st.file_uploader(
                "② または ZIP圧縮フォルダ (.zip) / 単一クラスの複数画像を選択（選択と同時に自動保存）",
                type=None,
                accept_multiple_files=True,
                max_upload_size=2000,
                key=f"folder_zip_or_multi_uploader_{active_dataset_name}",
                help="親フォルダを圧縮した .zip ファイル、または右の『単発画像選択時のクラス名』へ登録したい複数画像を選択できます。",
            )
            if res_zip_or_multi:
                uploaded_dir_files.extend(
                    res_zip_or_multi if isinstance(res_zip_or_multi, list) else [res_zip_or_multi]
                )

        with up_col2:
            fallback_cls_name = st.text_input(
                "サブフォルダ階層を含まない画像選択時の保存先クラス名",
                value="normal",
                key=f"fallback_cls_input_{active_dataset_name}",
                help="②でZIPではなく画像ファイル単体を複数選択した場合に割り当てるサブフォルダ（クラス）名です（例: normal, anomaly, bent_lead）。",
            )
            clear_before_upload = st.checkbox(
                f"アップロード前に `{active_dataset_name}` の既存画像をすべて消去する",
                value=False,
                key=f"clear_before_up_{active_dataset_name}",
                help="通常はOFF（既存サブフォルダにマージ追記）のままにしてください。",
            )
            auto_save_on_select = st.checkbox(
                "フォルダ／ファイル選択時に自動でデータセットへ即時保存する",
                value=True,
                key=f"auto_save_up_{active_dataset_name}",
                help="ON（推奨）にすると、左のアップローダーでフォルダや画像を選択した瞬間にサーバーのデータセットへ自動保存されます。",
            )

        if st.session_state.get("_last_upload_success_msg"):
            st.success(st.session_state.pop("_last_upload_success_msg"))

        parsed_upload_items = (
            _parse_uploaded_objects(uploaded_dir_files, fallback_class=fallback_cls_name)
            if uploaded_dir_files
            else []
        )

        if uploaded_dir_files and not parsed_upload_items:
            st.warning(
                "選択されたフォルダ／ファイル内に有効な画像ファイル (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, `.tif`, `.tiff`) "
                "が見つかりませんでした（`ground_truth` や隠しファイルは自動除外されます）。"
            )

        if parsed_upload_items:
            preview_counts: dict[str, int] = {}
            for item in parsed_upload_items:
                c_name = item["class_name"]
                preview_counts[c_name] = preview_counts.get(c_name, 0) + 1

            upload_batch_sig = (
                active_dataset_name,
                fallback_cls_name.strip() or "normal",
                bool(clear_before_upload),
                tuple(
                    (
                        str(getattr(uf, "file_id", getattr(uf, "name", ""))),
                        str(getattr(uf, "name", "")),
                        int(getattr(uf, "size", 0)),
                    )
                    for uf in uploaded_dir_files
                ),
            )
            already_auto_saved = st.session_state.get("_last_auto_uploaded_sig") == upload_batch_sig

            # Automatically persist newly selected folder/files immediately to prevent the 2-step upload trap
            if auto_save_on_select and not already_auto_saved:
                prog = st.progress(0.0, text=f"データセット `{active_dataset_name}` へ画像を自動保存中...")

                def _on_auto_prog(done: int, total: int) -> None:
                    prog.progress(done / max(1, total), text=f"自動保存中... ({done}/{total} 枚)")

                up_res = client.upload_folder_images(
                    items=parsed_upload_items,
                    clear_existing=clear_before_upload,
                    dataset_name=active_dataset_name,
                    batch_size=25,
                    progress_callback=_on_auto_prog,
                )
                st.session_state["_last_auto_uploaded_sig"] = upload_batch_sig
                st.session_state["_last_upload_success_msg"] = (
                    f"✅ データセット `{active_dataset_name}` へのアップロード保存が完了しました！ "
                    f"（今回保存: {len(parsed_upload_items)} 枚 ｜ データセット合計: {up_res['total_images']} 枚 ｜ "
                    f"サブフォルダ内訳: {up_res['subfolder_classes']}）"
                )
                st.rerun()

            st.markdown(
                f"**選択中のアップロードデータ (`{active_dataset_name}`): 全 {len(parsed_upload_items)} 枚（検出サブフォルダクラス数: {len(preview_counts)}）**"
            )
            df_preview = pd.DataFrame(
                [{"サブフォルダ名（クラス名）": k, "画像枚数": v} for k, v in sorted(preview_counts.items())]
            )
            st.dataframe(df_preview, use_container_width=True, hide_index=True, key="upload_preview_df")

            btn_label = (
                f"🔄 選択中の {len(parsed_upload_items)} 枚をデータセット `{active_dataset_name}` へ再保存する（保存済み）"
                if already_auto_saved
                else f"📁 データセット `{active_dataset_name}` へ合計 {len(parsed_upload_items)} 枚をアップロード保存"
            )
            if st.button(
                btn_label,
                type="secondary" if already_auto_saved else "primary",
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
                st.session_state["_last_auto_uploaded_sig"] = upload_batch_sig
                st.session_state["_last_upload_success_msg"] = (
                    f"✅ データセット `{active_dataset_name}` へのアップロード完了！ 合計 {up_res['total_images']} 枚 ｜ "
                    f"サブフォルダ内訳: {up_res['subfolder_classes']}"
                )
                st.rerun()

    with tab_class_mgr:
        if not sub_counts:
            st.info(f"データセット `{active_dataset_name}` にはまだサブフォルダが登録されていません。")
        else:
            st.markdown("##### A. クラスタリングを行わず、現在のサブフォルダ名をそのまま学習クラスとして確定 (`annotations.json` 作成)")
            st.caption(
                "すでにサブフォルダごとに正解クラス（例: `good`, `bent_lead`, `cut_lead` や `nomal`, `anomaly`）が分かれている場合、"
                "クラスタリングを実行しなくてもここで即座に学習用マニフェスト (`annotations.json`) を登録・上書きできます。"
            )
            dir_c1, dir_c2 = st.columns([2, 2])
            default_norm_sub = _detect_default_normal_class(detected_classes_now)
            norm_sub_idx = (
                detected_classes_now.index(default_norm_sub)
                if default_norm_sub in detected_classes_now
                else 0
            )
            direct_norm_cls = dir_c1.selectbox(
                "良品（正常: label_id=0）として扱うサブフォルダ名",
                options=detected_classes_now,
                index=norm_sub_idx,
                key=f"direct_sub_norm_{active_dataset_name}",
            )
            if dir_c2.button(
                f"✅ サブフォルダ名をそのまま `{active_dataset_name}` の正解ラベルとして登録 (`annotations.json` 保存)",
                type="primary",
                use_container_width=True,
            ):
                saved_sub = client.save_annotations_from_subfolders(
                    {
                        "dataset_name": active_dataset_name,
                        "normal_class_name": direct_norm_cls,
                        "val_ratio": 0.25,
                        "seed": 42,
                    }
                )
                st.success(
                    f"サブフォルダ名から `{active_dataset_name}` のアノテーションを確定しました！"
                    f"（全 {saved_sub['total_samples']} 枚 ｜ 良品=`{saved_sub['normal_class_name']}` ｜ 内訳={saved_sub['class_counts']}）"
                )
                st.rerun()

            st.divider()
            st.markdown("##### B. 登録済みサブフォルダ（クラス）の名称変更・統合・個別削除")
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown("**サブフォルダ名の変更 ／ 別クラスへの統合**")
                old_cls_sel = st.selectbox(
                    "変更元のサブフォルダ名",
                    options=detected_classes_now,
                    key=f"rename_old_{active_dataset_name}",
                )
                new_cls_inp = st.text_input(
                    "新しいクラス名（既存のクラス名を入力すると自動統合されます）",
                    value=old_cls_sel,
                    key=f"rename_new_{active_dataset_name}",
                )
                if st.button("✏️ クラス名を変更／統合する", use_container_width=True):
                    if new_cls_inp.strip() and new_cls_inp.strip() != old_cls_sel:
                        client.rename_or_merge_subfolder_class(
                            dataset_name=active_dataset_name,
                            old_class_name=old_cls_sel,
                            new_class_name=new_cls_inp.strip(),
                        )
                        st.success(f"`{old_cls_sel}` を `{new_cls_inp.strip()}` に変更・統合しました。")
                        st.rerun()

            with m_col2:
                st.markdown("**誤って登録したサブフォルダ（クラス）の個別削除**")
                del_cls_sel = st.selectbox(
                    "削除するサブフォルダ名",
                    options=detected_classes_now,
                    key=f"del_cls_{active_dataset_name}",
                )
                if st.button(f"🗑️ サブフォルダ `{del_cls_sel}` ({sub_counts.get(del_cls_sel, 0)}枚) を削除", use_container_width=True):
                    client.delete_subfolder_class(
                        dataset_name=active_dataset_name,
                        class_name=del_cls_sel,
                    )
                    st.info(f"サブフォルダ `{del_cls_sel}` を削除しました。")
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
    "良品クラス（例: `nomal` / `normal` / `good`）は一切クラスタリングせずそのまま `Cluster 0` として固定し、"
    "それ以外の異常クラスの画像のみを PCA ＋ シルエットスコア最大化 K-means で複数の不良サブクラス（`Cluster 1 .. k*`）へ自動分割します。"
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
    key=f"cluster_norm_cls_sel_{active_dataset_name}",
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

total_imgs_now = int(ds_info.get("total_images", 0))
if total_imgs_now == 0:
    st.info(
        f"💡 現在データセット `{active_dataset_name}` には画像が登録されていません（0枚）。"
        "上の「1. 検査画像データの登録」からフォルダまたはZIP／画像を選択すると自動で保存されます。"
    )
elif len(detected_classes_now) == 1:
    st.warning(
        f"⚠️ 現在 `{active_dataset_name}` にはサブフォルダ `{detected_classes_now[0]}` の1種類しか登録されていません。"
        "この状態でクラスタリングを実行すると、単一フォルダ用の密度フォールバック（下位15%を仮の異常クラスタに分割）が作動します。"
        "不良品フォルダ（`anomaly` 等）がある場合は、先に上のセクション1からアップロードしてください。"
    )

if st.button(
    f"データセット `{active_dataset_name}` の異常クラス自動細分化クラスタリングを実行（現在 {total_imgs_now} 枚）",
    type="primary",
    disabled=(total_imgs_now < 3),
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
        # Clear cached cluster label widget keys so fresh clustering suggestions are loaded
        for k in list(st.session_state.keys()):
            if str(k).startswith(f"cluster_lbl_{active_dataset_name}_"):
                del st.session_state[k]
        st.success(status["message"])
        st.rerun()
    else:
        st.error(f"クラスタリング失敗: {status.get('error')}")

# Section 3: Inspect Clusters & Confirm Pseudo-Multiclass Labels (Step 3)
st.divider()
st.subheader(f"3. クラスタリング結果の確認 ＆ クラスタ単位のラベル割当 (対象: `{active_dataset_name}`)")

try:
    cluster_data = client.get_clusters(dataset_name=active_dataset_name)
except Exception:
    cluster_data = None

if not cluster_data:
    st.info(
        f"データセット `{active_dataset_name}` にはまだクラスタリング結果がありません。"
        "上の「異常クラス自動細分化クラスタリングを実行」ボタンを押すか、セクション1のタブ②から「サブフォルダ名をそのまま正解ラベルとして登録」してください。"
    )
else:
    norm_cls_in_res = cluster_data.get("normal_class_name", normal_class_for_cluster)
    c_mode = cluster_data.get("clustering_mode", "anomaly_only")

    if c_mode == "density_fallback":
        st.warning(
            f"⚠️ このクラスタリング結果は単一フォルダに対する **`density_fallback`（密度分離モード）** で作成されたため、"
            f"`{norm_cls_in_res}` のうち外縁部 {cluster_data['tail_count']} 枚が `Cluster 1..{cluster_data['best_tail_k']}` に分割されています。"
            "元のサブフォルダ名をそのまま使う場合は、下の「元のサブフォルダ名をそのまま適用して確定」ボタンを押すか、セクション4のテーブルで一括修正してください。"
        )

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

    st.markdown("#### クラスタ単位のラベル割当（いつでも変更・再保存可能）")

    suggested_map: dict[str, str] = cluster_data.get("suggested_cluster_labels", {})
    detected_from_clusters = [str(x) for x in cluster_data.get("detected_classes", []) if x]
    hint_values = [str(x) for x in df_items["ground_truth_hint"].dropna().unique().tolist() if x]
    suggested_values = [str(v) for v in suggested_map.values() if v]
    existing_ann_classes = list(existing_ann.get("class_counts", {}).keys()) if existing_ann else []

    unique_clusters = sorted(df_items["cluster_id"].unique())
    auto_subcluster_names = [f"anomaly_cluster_{cid}" for cid in unique_clusters if cid > 0]

    cfg_col1, cfg_col2, cfg_col3 = st.columns([1.4, 1.6, 2.0])
    custom_labels_raw = cfg_col3.text_input(
        "追加のカスタムクラス名（カンマ区切りで任意追加）",
        value="",
        placeholder="例: bent_lead, cut_lead, damaged_case, misplaced",
        key=f"custom_lbl_input_{active_dataset_name}",
    )
    custom_labels = [x.strip() for x in custom_labels_raw.split(",") if x.strip()]

    dynamic_classes: list[str] = []
    for candidate in (
        [norm_cls_in_res]
        + detected_classes_now
        + existing_ann_classes
        + suggested_values
        + auto_subcluster_names
        + custom_labels
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
        key=f"sec3_norm_cls_{active_dataset_name}",
        help="良品クラス（label_id=0, is_defect=False）として扱うクラス名です。",
    )
    use_subfolder_labels = cfg_col2.checkbox(
        "異常クラスタ分割を使わず、元のサブフォルダ名をそのまま適用する",
        value=False,
        key=f"sec3_use_subfolder_{active_dataset_name}",
        help="ONにすると、K-meansのクラスタ番号ではなく各画像の親サブフォルダ名をそのままラベルとして適用します。",
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
        widget_key = f"cluster_lbl_{active_dataset_name}_{cid}"
        if st.session_state.get(widget_key) not in dynamic_classes:
            st.session_state[widget_key] = dynamic_classes[default_idx]

        with st.container(border=True):
            top_c1, top_c2 = st.columns([2, 3])
            role_title = (
                f"Cluster 0 — 良品クラス固定 ({len(sub_df)}枚)"
                if cid == 0
                else f"Cluster {cid} — 異常サブクラス {cid} ({len(sub_df)}枚)"
            )
            top_c1.markdown(f"**{role_title}**")
            top_c1.caption(f"元サブフォルダ内訳: {hints}")
            chosen_label = top_c2.selectbox(
                f"Cluster {cid} の確定クラス名",
                options=dynamic_classes,
                key=widget_key,
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

    if st.button(
        f"💾 クラスタ割当ラベルを確定して `{active_dataset_name}` のマニフェスト (annotations.json) を保存",
        type="primary",
        use_container_width=True,
    ):
        saved = client.save_annotations(
            {
                "dataset_name": active_dataset_name,
                "cluster_labels": cluster_labels,
                "sample_overrides": {},
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
        st.rerun()

# Section 4: Interactive Per-Image / Bulk Label Correction Table (Direct Database Editor)
st.divider()
st.subheader(f"4. 📝 登録済みデータのラベル一覧確認・個別／一括修正エディタ (対象: `{active_dataset_name}`)")
st.caption(
    "すでに登録された `annotations.json`（またはクラスタリング結果）のラベル付けミスを、"
    "「特定ラベルの一括変更」または「テーブル上での直接プルダウン編集」でいつでも自由に修正・再保存できます。"
)

# Refresh existing_ann in case it was just saved
try:
    existing_ann = client.get_annotations(dataset_name=active_dataset_name)
except Exception:
    existing_ann = None

editor_source_items: list[dict[str, Any]] = []
editor_norm_default = _detect_default_normal_class(detected_classes_now)

if existing_ann and existing_ann.get("items"):
    editor_source_items = existing_ann["items"]
    editor_norm_default = existing_ann.get("normal_class_name", editor_norm_default)
elif cluster_data and cluster_data.get("items"):
    editor_norm_default = cluster_data.get("normal_class_name", editor_norm_default)
    for it in cluster_data["items"]:
        lbl = it.get("suggested_label") or it.get("subfolder_class") or editor_norm_default
        editor_source_items.append(
            {
                "image_path": it["image_path"],
                "file_name": it["file_name"],
                "cluster_id": int(it.get("cluster_id", 0)),
                "label_name": lbl,
                "label_id": 0 if lbl == editor_norm_default else 1,
                "is_defect": lbl != editor_norm_default,
                "split": "train",
            }
        )

if not editor_source_items:
    st.info(
        f"データセット `{active_dataset_name}` にはまだ編集可能なアイテムがありません。"
        "先に画像をアップロードし、「サブフォルダ名をそのまま正解ラベルとして登録」または「クラスタリング」を実行してください。"
    )
else:
    df_edit_src = pd.DataFrame(editor_source_items)
    df_edit_src["current_subfolder"] = df_edit_src["image_path"].apply(
        lambda p: Path(str(p)).parent.name
    )

    all_label_options: list[str] = []
    for c_cand in (
        [editor_norm_default]
        + detected_classes_now
        + sorted(df_edit_src["label_name"].dropna().unique().tolist())
        + sorted(df_edit_src["current_subfolder"].dropna().unique().tolist())
        + ["normal", "nomal", "good", "anomaly", "defect_scratch", "defect_stain", "defect_dent"]
    ):
        if c_cand and c_cand not in all_label_options:
            all_label_options.append(str(c_cand))

    ed_top1, ed_top2, ed_top3 = st.columns([1.4, 1.8, 1.4])
    with ed_top1:
        editor_norm_cls = st.selectbox(
            "良品（正常: label_id=0）クラス名",
            options=all_label_options,
            index=all_label_options.index(editor_norm_default)
            if editor_norm_default in all_label_options
            else 0,
            key=f"editor_norm_cls_{active_dataset_name}",
        )
        new_custom_cls_ed = st.text_input(
            "新しいクラス選択肢を追加（任意）",
            value="",
            placeholder="例: defect_crack",
            key=f"editor_new_cls_{active_dataset_name}",
        )
        if new_custom_cls_ed.strip() and new_custom_cls_ed.strip() not in all_label_options:
            all_label_options.append(new_custom_cls_ed.strip())

    with ed_top2:
        st.markdown("**⚡ ラベルの一括置換（誤ラベルの一括修正）**")
        current_unique_labels = sorted(df_edit_src["label_name"].unique().tolist())
        b_col_from, b_col_to, b_col_btn = st.columns([1.2, 1.2, 1.0])
        bulk_from = b_col_from.selectbox(
            "変更前のラベル",
            options=current_unique_labels,
            key=f"bulk_from_{active_dataset_name}",
        )
        bulk_to = b_col_to.selectbox(
            "変更後のラベル",
            options=all_label_options,
            index=all_label_options.index(editor_norm_cls)
            if editor_norm_cls in all_label_options
            else 0,
            key=f"bulk_to_{active_dataset_name}",
        )
        sync_sub_bulk = st.checkbox(
            "実ファイルのサブフォルダ (`raw/<クラス名>/`) も同時に移動・同期する",
            value=True,
            key=f"sync_sub_bulk_{active_dataset_name}",
        )
        if b_col_btn.button("一括置換して保存", use_container_width=True):
            updated_rows = []
            for row in editor_source_items:
                old_l = str(row["label_name"])
                new_l = bulk_to if old_l == bulk_from else old_l
                updated_rows.append(
                    {
                        "image_path": str(row["image_path"]),
                        "file_name": str(row["file_name"]),
                        "label_name": new_l,
                        "split": str(row.get("split", "train")),
                        "cluster_id": int(row.get("cluster_id", 0)),
                    }
                )
            res_upd = client.direct_update_annotations(
                {
                    "dataset_name": active_dataset_name,
                    "normal_class_name": editor_norm_cls,
                    "items": updated_rows,
                    "sync_raw_subfolders": bool(sync_sub_bulk),
                    "rebalance_splits": True,
                    "val_ratio": 0.25,
                    "seed": 42,
                }
            )
            st.success(
                f"ラベル `{bulk_from}` → `{bulk_to}` の一括修正を保存しました！ 現在の内訳: {res_upd['class_counts']}"
            )
            st.rerun()

    with ed_top3:
        st.markdown("**📊 現在のラベル内訳**")
        counts_series = df_edit_src["label_name"].value_counts()
        df_cnt_summary = pd.DataFrame(
            [
                {
                    "クラス名 (label_name)": k,
                    "種別": "🟢 良品 (0)" if k == editor_norm_cls else "🔴 不良 (>0)",
                    "枚数": int(v),
                }
                for k, v in counts_series.items()
            ]
        )
        st.dataframe(df_cnt_summary, use_container_width=True, hide_index=True, height=135)

    # Interactive Data Editor Table
    st.markdown("##### 📋 個別画像のラベル・Train/Val分割 直接編集テーブル（セルをクリックして変更可能）")
    edited_df = st.data_editor(
        df_edit_src[
            ["file_name", "current_subfolder", "cluster_id", "label_name", "split", "image_path"]
        ],
        column_config={
            "file_name": st.column_config.TextColumn("ファイル名", disabled=True),
            "current_subfolder": st.column_config.TextColumn("現在のサブフォルダ", disabled=True),
            "cluster_id": st.column_config.NumberColumn("Cluster ID", disabled=True),
            "label_name": st.column_config.SelectboxColumn(
                "確定クラス名 (label_name) ✏️",
                options=all_label_options,
                required=True,
                help="クリックして任意のクラス名に変更できます。",
            ),
            "split": st.column_config.SelectboxColumn(
                "データ分割 (split) ✏️",
                options=["train", "val"],
                required=True,
            ),
            "image_path": st.column_config.TextColumn("画像パス", disabled=True),
        },
        use_container_width=True,
        hide_index=True,
        height=340,
        key=f"direct_ann_editor_{active_dataset_name}",
    )

    save_c1, save_c2, save_c3 = st.columns([1.5, 1.5, 2.0])
    rebalance_chk = save_c1.checkbox(
        "保存時に Train/Val 分割 (75%:25%) をクラス別に自動再計算する",
        value=False,
        key=f"rebalance_chk_{active_dataset_name}",
    )
    sync_folders_chk = save_c2.checkbox(
        "変更したクラス名に合わせて `raw/<クラス名>/` フォルダ内の実ファイルも移動する",
        value=True,
        key=f"sync_folders_chk_{active_dataset_name}",
    )
    if save_c3.button(
        f"💾 テーブルの編集内容を `{active_dataset_name}` のデータベース (`annotations.json`) に保存",
        type="primary",
        use_container_width=True,
    ):
        payload_items = [
            {
                "image_path": str(r["image_path"]),
                "file_name": str(r["file_name"]),
                "label_name": str(r["label_name"]),
                "split": str(r["split"]) if str(r["split"]) in ("train", "val") else "train",
                "cluster_id": int(r["cluster_id"]),
            }
            for r in edited_df.to_dict("records")
        ]
        res_direct = client.direct_update_annotations(
            {
                "dataset_name": active_dataset_name,
                "normal_class_name": editor_norm_cls,
                "items": payload_items,
                "sync_raw_subfolders": bool(sync_folders_chk),
                "rebalance_splits": bool(rebalance_chk),
                "val_ratio": 0.25,
                "seed": 42,
            }
        )
        st.success(
            f"✅ 編集内容を保存しました！（全 {res_direct['total_samples']} 枚 ｜ 良品=`{res_direct['normal_class_name']}` ｜ クラス内訳={res_direct['class_counts']}）"
        )
        st.rerun()
