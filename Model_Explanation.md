# Model_Explanation.md — 外観検査向け深層距離学習（Metric Learning）モデル・損失関数・ハイパーパラメータ技術解説書

本ドキュメントは、本リポジトリで開発した **外観検査自動化向け深層距離学習（Deep Metric Learning）システム** のモデル構造、学習アルゴリズム、損失関数（**50:50 Power-Balanced Multi-Proxy Anchor Loss** および **Asymmetric Multi-Similarity Loss**）、ならびに類似手法（**標準 MS-Loss / 標準 Proxy-Anchor Loss / ArcFace**）との理論的・実装的差異と全ハイパーパラメータの意味を、実際の実装コードに基づいて体系的に解説した技術仕様書である。

---

## 目次

1. [開発背景と産業外観検査における深層距離学習の課題](#1-開発背景と産業外観検査における深層距離学習の課題)
2. [全体学習パイプラインと 6 : 2 : 2 データ分割による厳密評価設計](#2-全体学習パイプラインと-6--2--2-データ分割による厳密評価設計)
3. [特徴抽出バックボーンと 128次元単位超球面射影モデル (`MetricProjectionModel`)](#3-特徴抽出バックボーンと-128次元単位超球面射影モデル-metricprojectionmodel)
4. [損失関数①（推奨メイン）：50:50 パワーバランス Multi-Proxy Anchor Loss](#4-損失関数推奨メイン5050-パワーバランス-multi-proxy-anchor-loss)
5. [損失関数②（比較検証用）：非対称マルチクラス MS-Loss ＋ XBM 双方向反発](#5-損失関数比較検証用非対称マルチクラス-ms-loss--xbm-双方向反発)
6. [類似手法（MS-Loss / Proxy-Anchor Loss / ArcFace）との徹底比較と違い](#6-類似手法ms-loss--proxy-anchor-loss--arcfaceとの徹底比較と違い)
7. [不均衡対策サンプラー（NAB-Sampler）と境界ハードサンプル合成（Phase 3）](#7-不均衡対策サンプラーnab-samplerと境界ハードサンプル合成phase-3)
8. [全ハイパーパラメータ完全リファレンス（コード対応・意味・調整指針）](#8-全ハイパーパラメータ完全リファレンスコード対応意味調整指針)
9. [空間評価指標（Step 6）の数理定義と解釈ガイド](#9-空間評価指標step-6の数理定義と解釈ガイド)

---

## 1. 開発背景と産業外観検査における深層距離学習の課題

製造ラインにおける自動外観検査（Visual Inspection）には、一般的な画像分類タスク（ImageNet 等の均衡マルチクラス分類）とは根本的に異なる **4つの構造的課題** が存在する。

1. **極端なクラス不均衡（100 : 2 問題）**:
   正常な製品（良品）は毎日大量に得られる一方、不良（異常）は発生頻度が極めて低く、特定の不良モード（微小な打痕や異物混入など）は数枚しか存在しないケースが常態である。
2. **不良モードの多峰性（Multi-Modality）とオープンセット性**:
   「良品」は設計仕様という単一（または少数のロット条件）の分布に従うのに対し、「不良」は傷（Scratch）・汚れ（Stain）・欠け（Broken）・打痕（Dent）・リード曲がり（Bent Lead）など多種多様であり、さらに運用開始後に未知の不良モードが発生する。識別空間を単純な「OK / NG の1次元直線」に射影すると、互いに異なる特徴を持つ不良モードを無理やり同じ1点へ押し込めることになり、表現崩壊（Representation Collapse）を引き起こす。
3. **良品内の多峰性（キャビティ差・ロット差・反射モード差）および「位置で不良の意味が変わる非対称ワーク」での境界競合**:
   インライン検査では、金型キャビティ差や表面仕上げモード差によって良品データ自体が複数の離散サブクラスタ（多峰性）を持つケースが少なくない。また非対称ワークでは空間座標 $(x, y)$ によって突起・刻印（正常構造）と打痕・異物（異常）の意味が変わるため、安易な反転・回転拡張や異モード間の線形 Mixup を行うと良品多様体の内部に偽の不良斥力源が発生し、良品の過検出（FP）を引き起こす。
4. **ペアベース距離学習における「不良同士の斥力による勾配の浪費」**:
   従来のペアベース距離学習を多クラス化した不良データにそのまま適用すると、「不良モードA vs 不良モードB」を遠ざける力にモデルの勾配パワーが消費され、本来最も重要な **「良品クラスタの凝集（求心力）」と「良品 vs 良品に酷似した不良の分離（斥力）」** が弱まる。

本システムはこれらの課題を解決するため、**128次元単位超球面（$\mathbb{S}^{127}$）** 上で良品の求心力と不良の斥力を **50 : 50（1.0 : 1.0 の同一コサインスケール）** で精密にバランスさせた独自のプロキシ型距離学習アーキテクチャを採用している。

---

## 2. 全体学習パイプラインと 6 : 2 : 2 データ分割による厳密評価設計

本システムの処理フローは、フェーズ1（前処理クラスタリング＆アノテーション）、フェーズ2（動的 Metric Learning 学習）、および Step 6（3軸空間評価）の3段階で構成される。

```mermaid
flowchart TD
    subgraph Phase1["フェーズ1：前処理クラスタリング ＆ 6:2:2 層化アノテーション"]
        A["検査画像データ登録<br/>(サブフォルダ名 = クラス名)"] --> B["特徴抽出 (timm + 1x1 GeM/GAP + 4x4空間グリッド + 表面統計量)"]
        B --> C["良品クラス固定保持 (Cluster 0) ＆ 良品潜在サブモード検出 (M*)<br/>＋ サブモード別残差標準化 & 異常限定 PCA・シルエット K-means (Cluster 1..k*)"]
        C --> D["ラベル確定 & (クラス × 良品サブモード) 6:2:2 層化分割<br/>train (60%) / val (20%) / test (20%)"]
    end

    subgraph Phase2["フェーズ2：50:50 パワーバランス深層距離学習 (train 60% のみで勾配更新)"]
        D --> E["train 良品のみでサブクラスタ内特徴標準化 (calibrate_feature_stats)<br/>＆ 最適良品プロキシ数 M* 自動決定 (0=Auto) ＆ 希少不良の微小シフト拡張バンク生成"]
        E --> F["NAB-Sampler ミニバッチ編成<br/>(良品 50% [サブモード均等層化] : 全不良モード均等 50%)"]
        F --> G["Phase 3 非対称ワーク対応 境界ハードサンプル合成<br/>(同座標 Zスコア欠陥転写 & 同一サブクラスタ異常残差 Δ_def 注入)"]
        G --> H["50:50 Multi-Proxy Anchor Loss<br/>①サブクラスタ求心力 + ②実不良斥力 & 正半球浅層合成ガード + ③北半球コーン直交正則化"]
        H --> I["エポック末に Möbius 許容半径較正 (α_m) ＆ val (20%) 評価<br/>→ 次エポックの求心力比率 (effective_normal_ratio) へフィードバック"]
    end

    subgraph Step6["Step 6：カンニングなし厳密空間評価"]
        I --> J["🧪 test (20% 未知データ) で最終性能評価<br/>(Accuracy / Recall / Precision / F1 / 混同行列)"]
        I --> K["🏋️ train (60% 学習データ) でクラスタ分布の確からしさ確認<br/>(良品内分散 σ²_norm / 空間分離比 J(t) / 3D UMAP)"]
    end
```

### 2.1. フェーズ1：良品固定保持・潜在サブモード検出 ＋ 異常クラス限定サブクラスタリング
実装ファイル: [`trainer/app/core/two_stage_cluster.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/two_stage_cluster.py)

1. **良品クラスの固定保持 (`Cluster 0`, `label_id = 0`) と潜在サブモード検出 (`normal_subcluster_id`)**:
   良品サブフォルダ（`normal`, `nomal`, `good`, `ok` 等）に属する画像は不良と混ぜてクラスタリングせず、全件そのまま単一の良品クラス（`Cluster 0`）として100%固定保持する。同時に、良品内部に金型キャビティ差やロット差などの離散サブモードが存在するかを [`estimate_optimal_normal_proxies`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L16-L121) によって自動診断し、各良品サンプルに潜在サブモードID `normal_subcluster_id`（単峰時はすべて `0`、多峰時は `0 .. M*-1`）を記録する（[`two_stage_cluster.py` L180-L191](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/two_stage_cluster.py#L180-L191)）。
2. **良品サブモード基準の欠陥残差標準化と異常クラス限定 K-means (`Cluster 1 .. k*`)**:
   異常クラスの画像群のみを対象とし、最近傍の良品サブモード重心 $\mu_{\text{norm}, m^*(i)}$ とプールされた良品サブモード内標準偏差 $\sigma_{\text{within}}$（ゼロ分散チャネル保護 $\hat{\sigma}_{\text{within}} = \text{where}(\sigma_{\text{within}} > 10^{-5}, \sigma_{\text{within}}, 1.0)$）を用いて
   $$\tilde{h}_i = \text{L2Normalize}\left(\text{clip}\left(\frac{h_i - \mu_{\text{norm}, m^*(i)}}{\hat{\sigma}_{\text{within}} + 10^{-5}}, -10, 10\right)\right)$$
   と標準化・L2正規化することで「製品およびサブモード固有の正常背景成分」を相殺し、純粋な欠陥残差ベクトルを抽出する（[`two_stage_cluster.py` L192-L225](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/two_stage_cluster.py#L192-L225)）。その後、PCA（上限64次元・累積寄与率95%）で次元圧縮し、コサイン距離に基づくシルエットスコア（Silhouette Score）が最大となる最適クラスタ数 $k^* \in [2, \min(K_{\max}, N_{\text{anom}}-1)]$ を自動選定して複数の不良サブクラスへと分割する。
   *(※単一の未整理フォルダのみが与えられた場合に限り、大域中央値類似度 $0.65$ ＋ 局所 $k$-NN 密度 $0.35$ の上位 `core_quantile=0.85` を仮の良品コアとする `density_fallback` モードが動作する。)*

### 2.2. 6 : 2 : 2 データ分割（`train : val : test = 60% : 20% : 20%`）とゼロ・データリーク原則
実装ファイル: [`backend/app/routers/datasets.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/backend/app/routers/datasets.py)（関数 [`_build_and_save_annotations_manifest`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/backend/app/routers/datasets.py#L144-L249)）, [`trainer/app/routers/train_jobs.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py)

ディープラーニングの厳密な評価のため、すべての不良サブクラスに加えて良品内の潜在サブモード `(label_name, normal_subcluster_id)` ごとに **6 : 2 : 2（学習 `train` 60% : 評価 `val` 20% : テスト `test` 20%）** に層化分割する（少数派の良品サブモードが `train` や `val` から欠落するのを防止）。各層内のサンプル数 $n_{\text{cls}}$ に対する決定論的分割規則（シード固定シャッフル後）は以下の通りである（[`datasets.py` L169-L202](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/backend/app/routers/datasets.py#L169-L202)）：
* **$n_{\text{cls}} \ge 3$ の場合**: $n_{\text{val}} = \max(1, \text{round}(0.20 \cdot n_{\text{cls}}))$, $n_{\text{test}} = \max(1, \text{round}(0.20 \cdot n_{\text{cls}}))$（ただし $n_{\text{val}} + n_{\text{test}} \ge n_{\text{cls}}$ となる $n_{\text{cls}} \in \{3, 4\}$ では $n_{\text{val}} = 1, n_{\text{test}} = 1$ に補正し、3分割すべてに最低1枚を保証）、残り $n_{\text{train}} = n_{\text{cls}} - n_{\text{val}} - n_{\text{test}}$。
* **$n_{\text{cls}} = 2$ の場合**: `train` に1枚、`test` に1枚（`val` は0枚）。
* **$n_{\text{cls}} = 1$ の場合**: `train` に1枚。

各分割の役割は以下の通り厳格に分離されている：

| データ分割 | 比率 | 役割と使用範囲（データリーク防止契約） |
| :--- | :---: | :--- |
| **🏋️ 学習データ (`train`)** | **60%** | 1. モデルの重み更新およびプロキシパラメータ更新（`NAB-Sampler` によるミニバッチ学習）<br/>2. **学習用良品データのみ（不良サンプル完全除外）** を用いたサブクラスタ内特徴標準化統計量（`feat_mean`, `feat_std`）のキャリブレーション（[`model.calibrate_feature_stats`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L428-L487)）<br/>3. 最適良品プロキシ数 $M^*$ の自動決定（[`estimate_optimal_normal_proxies`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L440-L445)）、初期プロキシ重心初期化（[`initialize_proxies_from_embeddings`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L570-L574)）、および Möbius 許容半径較正（[`calibrate_proxy_radii`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L808-L812)）<br/>4. Effective-Number クラス不均衡重み $W_c$ の算出（[`compute_class_balanced_weights`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L394-L398)）<br/>5. 希少実不良の微小シフト特徴バンク（[`extract_micro_augmented_gap_bank`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L587-L594)）および同座標 Zスコア欠陥転写バンク（[`_extract_pixel_boundary_augmentation_bank`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L608-L619)）の生成<br/>6. Step 6 における**学習クラスタ分布の確からしさ（良品内分散 $\sigma^2_{\text{norm}}$ の収縮度）の確認** |
| **📋 評価データ (`val`)** | **20%** | 1. 各エポック終了時の検証 Loss（`val_loss`, `val_loss_centripetal`, `val_loss_repel`）および `val_recall` / `val_precision` の算出<br/>2. **次のエポックの Loss 方向性（動的求心力比率 `effective_normal_ratio`）とカリキュラムへのフィードバック**（[`CurriculumPenaltyScheduler.step`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/scheduler.py#L61-L112)）<br/>3. 最良モデルチェックポイント（`best_model.pt`）の保存判定（[`composite_quality`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L920)） |
| **🧪 テストデータ (`test`)** | **20%** | 学習にもエポック間のハイパーパラメータ／Loss方向性制御にも**一切使用しない完全未知データ**。<br/>Step 6（[`3_space_evaluation.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/3_space_evaluation.py)）のデフォルト評価対象として、最終的な学習の良し悪し（`Accuracy`, `Recall`, `Precision`, `F1`, 混同行列）をカンニングなしで厳密評価する。 |

---

## 3. 特徴抽出バックボーンと 128次元単位超球面射影モデル (`MetricProjectionModel`)

実装ファイル: [`trainer/app/core/feature_extractor.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py)

### 3.1. アーキテクチャ構成と非対称ワーク向け空間座標保持設計
[`MetricProjectionModel`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L211-L417) は、「位置 $(x, y)$ で不良の意味が変わる非対称ワーク」に対応するため、入力前処理から特徴プーリング・標準化・射影ヘッドに至る以下の5ステージを経て 128次元単位超球面 $\mathbb{S}^{127} = \{z \in \mathbb{R}^{128} \mid \|z\|_2 = 1\}$ 上の埋め込みベクトル $z$ を出力する。

0. **非対称ワーク向け位置保持型画像変換 (`build_image_transforms` / `build_asymmetric_micro_augment_transforms`)**:
   * 左右・上下反転（`RandomHorizontalFlip` 等）や回転変換は空間座標の意味を破壊するため**完全に禁止**し、治具公差に相当する微小平行移動 `RandomAffine(degrees=0.0, translate=(0.02, 0.02))` と照明揺らぎに相当する微小輝度・コントラスト変動 `ColorJitter(brightness=0.04, contrast=0.04)` のみを適用する（[`feature_extractor.py` L420-L475](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L420-L475)）。
   * バックボーン凍結キャッシュ学習時（`freeze_backbone=True`）にも、希少な `train` 不良サンプルに対して3通りの決定論的微小シフト＋輝度・コントラスト摂動ビューを事前抽出した特徴バンク（[`extract_micro_augmented_gap_bank`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L479-L506)）を用意し、学習ステップごとに確率的に切り替えることで位置意味を保ったまま少数不良の暗記（過学習）を防ぐ。
1. **特徴抽出バックボーン (`self.backbone`)**:
   * `timm` ライブラリ経由で `efficientnet_b3`（デフォルト）、`vit_base_patch16_224`、`resnet18`、または決定論的3層畳み込みステム `lightweight_cnn`（[`LightweightInspectionStem`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L117-L140)）をロードする。
   * 少数データでの過学習と BatchNorm 統計量の崩壊を防ぐため、デフォルトでは `freeze_backbone=True` かつ学習中もバックボーンを `.eval()` モードに固定する（[`MetricProjectionModel.train()`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L277-L282)）。
2. **大域プーリング ＋ `4x4` CNN 空間座標保持プーリング (`SpatialPoolingHead`)**:
   * **1×1 大域意味プーリング (`forward_global`)**: 数ピクセルの微小傷の活性化が背景平均で希釈されるのを防ぐため、デフォルトで **Generalized Mean (GeM) Pooling**（[`GeMPooling`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L25-L46)、べき乗パラメータ $p = 3.0$）を採用する（`avg_max` / `gap` も選択可能）：
     $$h_{\text{pool}, c} = \left( \frac{1}{H \cdot W} \sum_{u=1}^{H} \sum_{v=1}^{W} \max(f_{c,u,v}, \epsilon)^p \right)^{\frac{1}{p}} \quad (p = 3.0, \; \epsilon = 10^{-6})$$
   * **4×4 空間グリッド意味プーリング (`forward_spatial_grid`, `enable_spatial_grid=True`)**: `1x1` プーリング単体では CNN 高次意味特徴の $(x, y)$ 座標情報が消失するため、特徴マップを $4 \times 4$ の空間グリッドへ集約した上でチャネル方向を $32$ グループへ適応平均プーリングし、**$4 \times 4 \times 32 = 512$ 次元の空間座標付き意味記述子** $h_{\text{spatial}}(x) \in \mathbb{R}^{512}$ を抽出する（[`feature_extractor.py` L85-L107](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L85-L107)）。これにより「同じ形状でも所定位置にあれば正常突起、別位置にあれば異常打痕」という非対称ワークの位置依存異常を高次特徴レベルで識別できる（過去セッションのチェックポイント推論時は `enable_spatial_grid=False` により後方互換性を維持）。
3. **マルチスケール空間グリッド表面統計量記述子との結合 (`_extract_raw_concat_features`)**:
   大域プーリング特徴 $h_{\text{pool}}$ および `4x4` 空間グリッド特徴 $h_{\text{spatial}}$ に加え、[`extract_handcrafted_surface_statistics(x)`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L143-L208) によって抽出された **678次元のマルチスケール表面統計ベクトル** $h_{\text{surf}}$（全体RGB平均[3]・標準偏差[3]、8×8パッチの95%/5%分位点[3+3]、4×4最大水平・垂直エッジ勾配[3+3]、8×8平均グリッド[192]、4×4最大/最小グリッド[48+48]、6×6局所標準偏差グリッド[108]、6×6水平・垂直エッジ勾配グリッド[108+108]、4×4色差グリッド[48]：計 $18 + 288 + 108 + 216 + 48 = 678$ 次元）をそれぞれブロック別に L2 正規化して連結する（[`feature_extractor.py` L299-L321](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L299-L321)）：
   $$h_{\text{raw}}(x) = \left[ \frac{h_{\text{pool}}(x)}{\|h_{\text{pool}}(x)\|_2}, \; 0.75 \cdot \frac{h_{\text{spatial}}(x)}{\|h_{\text{spatial}}(x)\|_2}, \; \frac{h_{\text{surf}}(x)}{\|h_{\text{surf}}(x)\|_2} \right] \in \mathbb{R}^{D_{\text{gap}}}$$
   （例: `efficientnet_b3` + `gem` では $1536 + 512 + 678 = 2726$ 次元）。
4. **学習用良品データ限定・サブクラスタ内特徴量標準化 (`calibrate_feature_stats` & `standardize_features`)**:
   同じ製品を撮影した画像群では共通背景バイアスが支配的となる一方、標準化の母集団に不良サンプルを混ぜると「特定位置の欠陥が発火させたチャネル」の標準偏差 $\sigma$ が膨張して欠陥信号自体が減衰する。さらに良品が多峰性（$M^* \ge 2$）を持つ場合、全体の標準偏差で割るとモード間距離によって $\sigma$ が過大評価される。
   そこで学習開始前に **`train` (60%) 分割の良品サンプルのみ（不良サンプル 0 枚）** を抽出し、単峰時（$M^*=1$）は良品の平均・標準偏差、多峰時（$M^* \ge 2$）は各良品サブモード重心 $\mu_m$ の均等平均 $\mu_{\text{bal}} = \frac{1}{M^*}\sum_{m=1}^{M^*}\mu_m$ と**プールされたサブクラスタ内標準偏差**
   $$\sigma_{\text{within}} = \sqrt{\frac{1}{\max(1, N_{\text{norm}} - M^*)} \sum_{m=1}^{M^*} \sum_{i \in \mathcal{I}_m} \big(h_{\text{raw}}(x_i) - \mu_m\big)^2}$$
   （ゼロ分散保護 $\hat{\sigma}_{\text{norm}} = \text{where}(\sigma_{\text{within}} > 10^{-5}, \sigma_{\text{within}}, 1.0)$）を用いて Zスコア標準化＋L2正規化を行う（[`feature_extractor.py` L324-L394](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L324-L394)、呼び出し部 [`train_jobs.py` L428-L486](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L428-L486)）：
   $$h(x) = \text{L2Normalize}\left( \text{clip}\left( \frac{h_{\text{raw}}(x) - \mu_{\text{bal}}}{\hat{\sigma}_{\text{norm}} + 10^{-5}}, \; -10.0, \; +10.0 \right) \right)$$
5. **入力特徴 Dropout ＋ 128次元 MLP Projection Head (`self.feat_dropout` & `self.projection_head`)**:
   少数不良サンプルの特定ノイズチャネルへの過剰適合（ショートカット暗記）を抑制するため入力特徴 `Dropout(0.20)` を適用した上で、非線形射影ヘッドと最終 L2 正規化により 128次元単位超球面 $\mathbb{S}^{127}$ 上へ写像する（[`feature_extractor.py` L266-L275, L401-L411](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L266-L275)）：
   $$\text{Dropout}(0.20) \to \text{Linear}(D_{\text{gap}} \to 384) \to \text{BatchNorm1d}(384) \to \text{GELU} \to \text{Dropout}(0.10) \to \text{Linear}(384 \to 128) \to \text{L2Normalize}$$

---

## 4. 損失関数①（推奨メイン）：50:50 パワーバランス Multi-Proxy Anchor Loss

実装ファイル: [`trainer/app/core/proxy_anchor_loss.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py)（クラス名: `CustomizedMultiProxyAnchorLoss`）

### 4.1. 設計思想・自動プロキシ数決定 ($M^*$)・Möbius 許容半径較正と全体数式
本モデルのメイン損失関数は、**「① 良品データを対応する良品プロキシ中心へ引き寄せる求心力（Centripetal Force）」** と **「② 不良データを良品プロキシ群から境界マージン $\delta$ の外側へ押し出す斥力（Centrifugal Repulsion Force）」** を、どちらも同一のコサイン類似度スケール（物差し）に揃えた上で、基準比率 **50 : 50（$\lambda_{\text{norm}} = 1.0 : \lambda_{\text{anom}} = 1.0$、スライダーで可変）** で結合した関数である。

1. **最適良品プロキシ数 $M^*$ の自動決定 (`estimate_optimal_normal_proxies`, デフォルト `num_normal_proxies = 0`)**:
   128次元単位超球面上に、学習可能な **$M$ 個の良品サブセンタープロキシ** $\hat{p}_{\text{norm}}^{(1)}, \dots, \hat{p}_{\text{norm}}^{(M)} \in \mathbb{S}^{127}$ と、**$C$ 個の不良モードプロキシ** $\hat{p}_{\text{def}}^{(1)}, \dots, \hat{p}_{\text{def}}^{(C)} \in \mathbb{S}^{127}$ を定義する。`num_normal_proxies = 0`（推奨デフォルト `Auto`）のとき、`train` 良品特徴に対して平均中心化 PCA（累積寄与率90%・上限32次元）＋球面 K-Means を $k \in \{2, \dots, 6\}$ で走査し、
   * 最小クラスタサイズ条件（$\ge \max(5, \lfloor 0.08 N_{\text{norm}} \rfloor)$）、
   * コサイン・シルエット係数 $\text{Sil}(k) \ge 0.25$、
   * 単一重心比での90%分位ユークリッド半径の低減率 $\ge 18\%$
   のすべてを満たす離散サブモードが存在する場合に限り、各サブクラスタの90%分位コサイン被覆半径が許容上限 $\epsilon = 0.22$ 以下となる最小の $M^* \in \{2, \dots, 6\}$ を自動採択する（単峰性の良品データでは厳格に **$M^* = 1$** を維持する。また `1〜12` の手動固定も可能：[`proxy_anchor_loss.py` L16-L121](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L16-L121)）。
2. **Möbius 変換によるサブプロキシ別許容半径較正 (`calibrate_proxy_radii`)**:
   $M > 1$ のとき、キャビティAはばらつきが小さく（半径 $r_1$ 小）、キャビティBは表面反射のばらつきがやや大きい（半径 $r_2$ 大）といった異方性が存在すると、生の最大類似度では単一閾値 $\tau_{\text{sim}}$ の意味がサブクラスタごとにずれる。そこで各サブプロキシ $m$ に属する `train` 良品の生コサイン距離 $d_m(z) = 1 - \langle z, \hat{p}_{\text{norm}}^{(m)} \rangle \in [0, 2]$ の90%分位半径 $r_m = Q_{0.90}(\{d_m(z_i)\}_{i \in \mathcal{I}_m})$ とその平均 $\bar{r} = \frac{1}{M}\sum_{m=1}^{M} r_m$ から、$[0, 2] \to [0, 2]$ の全単射・単調増加 **Möbius 距離較正関数** を構成する（[`proxy_anchor_loss.py` L355-L450](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L355-L450)）：
   $$\alpha_m = \text{clip}\left(\frac{\bar{r}(2 - r_m)}{r_m(2 - \bar{r})}, \; 0.40, \; 4.00\right), \qquad \tilde{d}_m(z) = \frac{2 \alpha_m d_m(z)}{(\alpha_m - 1) d_m(z) + 2} \in [0, 2]$$
   $$S_{\text{cal}}(x, p_{\text{norm}}) = 1.0 - \min_{m \in \{1, \dots, M\}} \tilde{d}_m\big(z(x)\big) \in [-1.0, +1.0]$$
   この変換は $d_m = r_m$ のとき常に $\tilde{d}_m(r_m) = \bar{r}$ となって全サブクラスタの外縁等高線を揃え、かつ **$M = 1$ のときは $\alpha_1 = 1.0, \; \tilde{d}_1(z) = d_1(z)$ となって従来のコサイン類似度 $\langle z(x), \hat{p}_{\text{norm}}^{(1)} \rangle$ に厳密一致する**。

全体の損失関数 $L_{\text{total}}$ は次式で与えられる（[`proxy_anchor_loss.py` L479-L694](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L479-L694)）：

$$L_{\text{total}} = \underbrace{\lambda_{\text{norm}} \cdot L_{\text{centripetal}}}_{\text{① 良品の求心力（基準 50\%: }\lambda_{\text{norm}}=1.0\text{）}} + \underbrace{\lambda_{\text{anom}} \cdot L_{\text{repel}}}_{\text{② 不良の斥力（基準 50\%: }\lambda_{\text{anom}}=1.0\text{）}} + \underbrace{L_{\text{aux}}}_{\text{③ 補助正則化項}}$$

ここで、パワーバランス比率パラメータ $r_{\text{norm}} \in [0.05, 0.95]$（コード変数名: `normal_loss_ratio`、デフォルト `0.50`）に対して、重み係数は
$$\lambda_{\text{norm}} = 2.0 \cdot r_{\text{norm}}, \qquad \lambda_{\text{anom}} = 2.0 \cdot (1.0 - r_{\text{norm}})$$
と定義される。デフォルトの $r_{\text{norm}} = 0.50$ のとき $\lambda_{\text{norm}} = 1.0, \; \lambda_{\text{anom}} = 1.0$ となり、文字通り **50 : 50 の力関係** となる。

---

### 4.2. 第1項：良品の求心力 (`loss_centripetal`) — サブクラスタ・ルーティング付き線形コサイン凝集
ミニバッチ内の良品サンプル集合を $\mathcal{B}_{\text{norm}}$、$M > 1$ のときにサブモード $m \in \{1, \dots, M\}$ に属する良品部分集合を $\mathcal{B}_{\text{norm}, m}$（バッチ内に1枚以上存在する活性サブモード集合を $\mathcal{M}_{\text{active}}$）とすると、第1項は次式で計算される（[`proxy_anchor_loss.py` L531-L560](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L531-L560)）：

$$L_{\text{centripetal}} = \begin{cases} \displaystyle\frac{1}{|\mathcal{B}_{\text{norm}}|} \sum_{x \in \mathcal{B}_{\text{norm}}} \Big( 1.0 - S_{\text{cal}}(x, p_{\text{norm}}) \Big) & (M = 1 \text{ またはサブクラスタ未指定時}) \\[10pt] \displaystyle\frac{1}{|\mathcal{M}_{\text{active}}|} \sum_{m \in \mathcal{M}_{\text{active}}} \frac{1}{|\mathcal{B}_{\text{norm}, m}|} \sum_{x \in \mathcal{B}_{\text{norm}, m}} \tilde{d}_m\big(z(x)\big) & (M > 1 \text{ サブクラスタ・ルーティング時}) \end{cases}$$

#### なぜ `Softplus` や二乗誤差 $(1-S)^2$ ではなくサブクラスタ・ルーティング付き線形コサイン距離なのか？
* **Softplus / LogSumExp の問題点**: $\log(1 + \sum e^{-\alpha(S - \delta_{\text{pos}})})$ は $S > \delta_{\text{pos}}$（例: $0.65$）を超えると勾配がゼロへ飽和する「不感帯」を持ち、さらに指数重み $e^{-\alpha S}$ によってバッチ内で最も遠い外れ値良品1枚だけに勾配の99%が集中するため、良品クラスタ全体の分散 $\sigma^2_{\text{norm}}$ が収縮しない。
* **二乗誤差 $(1 - S)^2$ の問題点**: $S$ が $0.90$ に近づくと微分値 $-2(1-S)$ が $-0.20 \to 0$ へと減衰し、中心付近での凝集力が失われる。
* **線形コサイン距離 $1 - S_{\text{cal}}$ とサブクラスタ独立ルーティングの利点**:
  * $M = 1$（単峰良品）では $\frac{\partial (1 - S)}{\partial S} = -1.0$ の減衰しない一定の求心勾配が全良品サンプルに均等に働き続け、エポック進行とともに良品クラスタ内分散が単調収縮する。
  * $M > 1$（多峰良品）では、各良品サンプルを自身が属するサブプロキシ $\hat{p}_{\text{norm}}^{(m)}$ のみへ引き寄せ、かつ活性サブモード間で均等平均をとることで、**異なる良品サブクラスタ同士を無理に1点へ引きずり合って表現が歪む現象や、少数派良品モードの勾配埋没を完全に防ぐ**。
* **良品サンプルへの遠心力の完全排除**: 良品サンプル $z_{\text{norm}}$ には不良プロキシ $p_{\text{def}}$ からの斥力を一切作用させない（良品が外向きに弾き飛ばされて分散が拡大する現象を根絶）。

---

### 4.3. 第2項：不良の斥力 (`loss_repel`) — 実不良と浅層合成ガードの完全分離・アンカー固定 `p_norm.detach()`
ミニバッチ内の実不良サンプル集合 $\mathcal{B}_{\text{def}}$（枚数 $N_{\text{real}}$）と、Phase 3 で生成された境界ハード合成サンプル集合 $\mathcal{B}_{\text{synth}}$（枚数 $N_{\text{synth}}$）は、**実不良が収束した後に合成サンプルが斥力勾配の100%を乗っ取って良品を押し出す現象を防ぐため、正規化母集団と目標マージンを完全に分離して計算する**（[`proxy_anchor_loss.py` L569-L633](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L569-L633)）。

1. **アンカー固定 (`p_norm.detach()`) による良品プロキシのドリフト遮断**:
   不良サンプルと良品プロキシの較正類似度を計算する際、**良品プロキシ側の勾配を完全に切り離す（`p_norm_anchor = p_norm.detach()`）**：
   $$S_{\text{detach}}(x, p_{\text{norm}}) = S_{\text{cal}}\big(z(x), \text{stop\_grad}(p_{\text{norm}})\big)$$
   `p_norm.detach()` により、反作用で良品プロキシ $p_{\text{norm}}$ が良品重心から引きずり出される現象を完全に防ぎ、斥力を **100% 特徴抽出器（不良サンプル $z(x)$ を遠ざける方向）のみ** に作用させる。
2. **実不良サンプル集合 ($\mathcal{B}_{\text{def}}$) の平均 $1.0$ 正規化重み $\tilde{w}_{\text{real}}(x)$ と2段ヒンジ斥力 ($L_{\text{repel\_real}}$)**:
   * 学習データのクラス別サンプル数 $n_c$ から算出される Effective-Number クラス不均衡補正重み $W_{y_i}$（[`compute_class_balanced_weights` (`nab_sampler.py` L8-L38)](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/nab_sampler.py#L8-L38)）
     $$W_c^{\text{raw}} = \frac{1 - \beta}{1 - \beta^{n_c}}, \qquad W_c = \frac{W_c^{\text{raw}}}{\frac{1}{|\mathcal{C}_{\text{train}}|} \sum_{j \in \mathcal{C}_{\text{train}}} W_j^{\text{raw}}}$$
     をベースとし、良品プロキシへの侵入度 $S_{\text{detach}}(x_i, p_{\text{norm}})$ が最も高い上位 $K_{\text{eff}}$ 件（**Top-$K$ 危険実不良**）にのみブースト係数 $\lambda_{\text{hard}}$（`hard_anomaly_boost`、デフォルト `3.0`）を乗じる（[`proxy_anchor_loss.py` L584-L593](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L584-L593)）：
     $$K_{\text{eff}} = \begin{cases} \text{clip}\big(\lceil K(t) \cdot N_{\text{real}} \rceil, \; 1, \; N_{\text{real}}\big) & (\text{top\_k\_mode} = \text{"ratio"}) \\ \text{clip}\big(\text{round}(K(t)), \; 1, \; N_{\text{real}}\big) & (\text{top\_k\_mode} = \text{"count"}) \end{cases}$$
     $$w_{\text{raw}}(x_i) = \begin{cases} W_{y_i} \cdot \lambda_{\text{hard}} & (x_i \in \text{Top-}K_{\text{eff}} \text{ 危険実不良}) \\ W_{y_i} & (\text{それ以外の実不良}) \end{cases}$$
   * **実不良内部での平均 $1.0$ 正規化**:
     $$\tilde{w}_{\text{real}}(x_i) = \frac{w_{\text{raw}}(x_i)}{\frac{1}{N_{\text{real}}} \sum_{j=1}^{N_{\text{real}}} \text{stop\_grad}(w_{\text{raw}}(x_j)) + 10^{-8}} \qquad \left(\text{常に } \frac{1}{N_{\text{real}}} \sum_{i=1}^{N_{\text{real}}} \tilde{w}_{\text{real}}(x_i) = 1.0\right)$$
   * **2段コサインマージン・ヒンジ斥力と XBM 境界反発ブレンド**:
     目標コサインマージン $\delta$（`repel_delta`、デフォルト `0.0` ＝ 良品プロキシと直交 90°）と深部マージン $\delta_{\text{deep}} = \min(\delta - 0.25, -0.25)$ を組み合わせ、さらに XBM に良品埋め込みが蓄積されている場合は境界良品上位 $K_{\text{xbm}}$ 件とのヒンジ損失 $L_{\text{xbm\_hinge}}$ を $15\%$ ブレンドする（[`proxy_anchor_loss.py` L594-L620](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L594-L620)）：
     $$\ell_{\text{repel}}(x_i) = 0.85 \cdot \max\Big(0, \; S_{\text{detach}}(x_i, p_{\text{norm}}) - \delta\Big) + 0.15 \cdot \max\Big(0, \; S_{\text{detach}}(x_i, p_{\text{norm}}) - \delta_{\text{deep}}\Big)$$
     $$L_{\text{repel\_real}}^{\text{batch}} = \frac{1}{N_{\text{real}}} \sum_{i=1}^{N_{\text{real}}} \tilde{w}_{\text{real}}(x_i) \cdot \ell_{\text{repel}}(x_i), \qquad L_{\text{repel\_real}} = 0.85 \cdot L_{\text{repel\_real}}^{\text{batch}} + 0.15 \cdot L_{\text{xbm\_hinge}}$$
3. **合成境界サンプル集合 ($\mathcal{B}_{\text{synth}}$) の正半球浅層ガードレール斥力 ($L_{\text{repel\_synth}}$)**:
   正常背景を多く含む合成境界サンプルを負半球（$< 0.0$）まで無理に押し込むと正常特徴まで歪むため、判定閾値 $\tau_{\text{sim}} = 0.0$ より手前の正半球浅層マージン $\delta_{\text{synth}} = \max(\delta + 0.05, +0.05)$ で勾配を停止させ、固定小重み $w_{\text{synth}} = 0.15$（`synth_repel_weight`）で加算する（[`proxy_anchor_loss.py` L621-L633](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L621-L633)）：
   $$L_{\text{repel\_synth}} = \frac{1}{N_{\text{synth}}} \sum_{x \in \mathcal{B}_{\text{synth}}} \max\Big(0, \; S_{\text{detach}}(x, p_{\text{norm}}) - \delta_{\text{synth}}\Big), \qquad L_{\text{repel}} = L_{\text{repel\_real}} + 0.15 \cdot L_{\text{repel\_synth}}$$

---

### 4.4. 第3項：補助正則化項 (`loss_aux`) — 不良モード別凝集と北半球共有コーン・プロキシ直交化
異なる不良モードが128次元空間上で別々の方向へ整然と分かれ、かつ $M > 1$ のときに良品サブプロキシ群が「融合せず、かつ反対半球へも散逸しない北半球共有コーン」を形成するよう、良品サンプル $z_{\text{norm}}$ には一切直接の反発力を及ぼさない以下の3つの補助項を加える（[`proxy_anchor_loss.py` L637-L678](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L637-L678)）：

$$L_{\text{aux}} = 0.10 \cdot L_{\text{def\_sub}} + 0.10 \cdot L_{\text{def\_hemisphere}} + w_{\text{ortho}} \cdot L_{\text{ortho}}$$

1. **実不良サブクラス凝集項 ($L_{\text{def\_sub}}$)**:
   ミニバッチ内の不良クラス $c \in \{1, \dots, C\}$ のサンプル部分集合を $\mathcal{B}_c = \{i \in \mathcal{B}_{\text{def}} \mid y_i = c\}$、活性不良クラス集合を $C_{\text{active}} = \{c \mid |\mathcal{B}_c| > 0\}$ とするとき、各実不良サンプル $z_i$ を対応する不良モードプロキシ $\hat{p}_{\text{def}}^{(c)}$ へ緩やかに引き寄せる：
   $$L_{\text{def\_sub}} = \frac{1}{|C_{\text{active}}|} \sum_{c \in C_{\text{active}}} \frac{1}{|\mathcal{B}_c|} \sum_{i \in \mathcal{B}_c} \left( 1.0 - \langle z_i, \hat{p}_{\text{def}}^{(c)} \rangle \right)$$
2. **不良プロキシ負半球維持項 ($L_{\text{def\_hemisphere}}$)**:
   不良プロキシ $\hat{p}_{\text{def}}^{(c)}$ 自体が良品プロキシ群と反対側の負の半球（類似度 $\le -0.20$）に留まるよう制約する：
   $$L_{\text{def\_hemisphere}} = \frac{1}{C \cdot M} \sum_{c=1}^{C} \sum_{m=1}^{M} \max\left(0, \; \langle \hat{p}_{\text{def}}^{(c)}, \text{stop\_grad}(\hat{p}_{\text{norm}}^{(m)}) \rangle - (-0.20)\right)$$
3. **プロキシ間直交 ＆ 良品サブプロキシ北半球共有コーン正則化項 ($L_{\text{ortho}}$)**:
   異なる不良プロキシ同士（$c \ne c'$）のコサイン類似度が `ortho_margin_defect = 0.20` を超えないよう直交方向へ分散させ、さらに $M > 1$ の場合は良品サブプロキシ同士の内積を **$[+0.20, \; +0.75]$（なす角約 $41^\circ \sim 78^\circ$ の北半球共有コーン内）** に保つ両側2乗ヒンジ制約を課す（[`proxy_anchor_loss.py` L653-L672](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L653-L672)）：
   $$L_{\text{ortho}} = \frac{1}{C(C-1)} \sum_{c \ne c'} \max\left(0, \; \langle \hat{p}_{\text{def}}^{(c)}, \hat{p}_{\text{def}}^{(c')} \rangle - 0.20\right)^2 + \frac{1}{M(M-1)} \sum_{m \ne m'} \left[ \max\left(0, \; \langle \hat{p}_{\text{norm}}^{(m)}, \hat{p}_{\text{norm}}^{(m')} \rangle - 0.75\right)^2 + 0.50 \cdot \max\left(0, \; 0.20 - \langle \hat{p}_{\text{norm}}^{(m)}, \hat{p}_{\text{norm}}^{(m')} \rangle\right)^2 \right]$$

---

### 4.5. エポック毎の `val` 評価フィードバックによるカリキュラム進捗 $p(t)$ と次エポック Loss 方向性自動制御
実装ファイル: [`trainer/app/core/scheduler.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/scheduler.py)（クラス名: `CurriculumPenaltyScheduler`）

各エポック $t \in \{1, \dots, T_{\text{total}}\}$ の終了時、学習に使っていない **評価データ (`val`: 20%)** に対して不良検出率 $\text{Recall}_{\text{val}}$、適合率 $\text{Precision}_{\text{val}}$、境界最小コサイン距離 $d_{\min}^{\text{boundary}}$、求心力損失 $\mathcal{L}_{\text{cent}}^{\text{val}}$、斥力損失 $\mathcal{L}_{\text{repel}}^{\text{val}}$ を計算し、以下の2つの制御ループを同時に駆動する（[`scheduler.py` L61-L112](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/scheduler.py#L61-L112)）：

1. **カリキュラム進捗度 $p(t)$ と Top-$K$・反発スケジュール更新**:
   目標境界距離 $d_{\text{target}}$（`target_d_min`、デフォルト `1.0`）に対する達成度と $\text{Recall}_{\text{val}}$、エポック進行率から複合進捗度 $p(t) \in [0, 1]$ を算出し、次エポックの反発係数 $\gamma(t)$、反発マージン $m(t)$、Top-$K$ 抽出パラメータ $K(t)$ を線形補間する：
   $$p(t) = \text{clip}\left(0.45 \cdot \text{Recall}_{\text{val}} + 0.35 \cdot \text{clip}\left(\frac{d_{\min}^{\text{boundary}}}{d_{\text{target}}}, 0, 1\right) + 0.20 \cdot \text{clip}\left(\frac{t}{T_{\text{total}}}, 0, 1\right), \; 0.0, \; 1.0\right)$$
   $$\gamma(t) = \gamma_{\text{init}} + p(t)(\gamma_{\text{final}} - \gamma_{\text{init}}), \quad m(t) = m_{\text{init}} + p(t)(m_{\text{final}} - m_{\text{init}}), \quad K(t) = K_{\text{init}} + p(t)(K_{\text{final}} - K_{\text{init}})$$
2. **次エポックの求心力 vs. 斥力方向性フィードバック (`effective_normal_ratio`)**:
   $$\Delta_{\text{loss}} = \mathcal{L}_{\text{cent}}^{\text{val}} - \mathcal{L}_{\text{repel}}^{\text{val}}, \qquad \Delta_{\text{metric}} = \text{Recall}_{\text{val}} - \text{Precision}_{\text{val}}$$
   $$\Delta r = \text{clip}\Big( 0.25 \cdot \Delta_{\text{loss}} + 0.10 \cdot \Delta_{\text{metric}}, \; -0.15, \; +0.15 \Big)$$
   $$r_{\text{eff}}(t+1) = \text{clip}\Big( r_{\text{base}} + \Delta r, \; 0.10, \; 0.90 \Big)$$
   * **`centripetal_focus`（$\Delta r > +0.012$：良品凝集強化）**:
     `val` データで良品の求心力Lossが不良斥力Lossより大きい、または `Recall > Precision`（良品の分散が大きく過検出 FP が発生している）場合、次エポックの求心力比率 $r_{\text{eff}}$ を自動的に引き上げて良品クラスタを引き締める。
   * **`repulsion_focus`（$\Delta r < -0.012$：不良斥力強化）**:
     `val` データで不良の斥力Lossが残っている、または `Recall < Precision`（不良の見逃し FN が発生している）場合、次エポックの求心力比率 $r_{\text{eff}}$ を引き下げ（＝不良斥力比率 $1 - r_{\text{eff}}$ を引き上げ）、不良の押し出しを強化する。
   * **`balanced_hold`（$|\Delta r| \le 0.012$：均衡維持）**:
     両者がバランスしている場合は基準比率 $r_{\text{base}}$（デフォルト `0.50`）付近を維持する。

---

## 5. 損失関数②（比較検証用）：非対称マルチクラス MS-Loss ＋ XBM 双方向反発

実装ファイル: [`trainer/app/core/ms_loss.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/ms_loss.py)（クラス名: `AsymmetricMultiSimilarityLoss`）

UI 上で `loss_type = "ms_loss"` を選択した場合に実行される比較用アルゴリズムであり、Wang et al. (CVPR 2019) の Multi-Similarity Loss を外観検査向けに非対称拡張したものである（デフォルト定数: $\alpha = 2.0, \; \beta = 16.0, \; \lambda_{\text{pos}} = 0.75, \; \lambda_{\text{primary}} = -0.15, \; \lambda_{\text{aux}} = 0.15, \; \epsilon = 0.1, \; \beta_{\text{repel}} = 12.0, \; \text{local\_pos\_k} = 3$）：

$$L_{\text{MS}} = \frac{1}{B} \sum_{i=1}^{B} w_i \left\{ \frac{1}{\alpha} \log \left[ 1 + \sum_{k \in \mathcal{P}_i^*} e^{-\alpha (S_{ik} - \lambda_{\text{pos}})} \right] + \frac{1}{\beta} \log \left[ 1 + \sum_{k \in \mathcal{N}_i^{\text{primary}}} w_{\text{defect}} e^{\beta (S_{ik} - \lambda_{\text{primary}})} + \sum_{k \in \mathcal{N}_i^{\text{aux}}} w_{\text{aux}} e^{\beta (S_{ik} - \lambda_{\text{aux}})} \right] \right\} + L_{\text{proto}} + \gamma(t) L_{\text{repel}}^{\text{XBM}}$$

* **非対称アンカー重み ($w_i$)**: アンカー $i$ が不良（$y_i > 0$）のとき $w_i = w_{\text{defect}}$、良品（$y_i = 0$）のとき $w_i = 1.0$（[`ms_loss.py` L86](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/ms_loss.py#L86)）。
* **局所近傍ポジティブマイニング (`local_pos_k = 3`) と相対類似度マイニング ($\epsilon = 0.1$)**:
  不良アンカー $y_i > 0$ に対しては、同一クラス内の全サンプルではなくコサイン類似度上位 $k=3$ 件の局所近傍候補を抽出し、さらに相対類似度条件 $S_{ik} < \max_{j \in \mathcal{N}_i} S_{ij} + \epsilon$ を満たすハード正例集合 $\mathcal{P}_i^*$（条件を満たすペアが無い場合はフォールバックとして局所近傍候補全体）のみを引き寄せる（[`ms_loss.py` L91-L119](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/ms_loss.py#L91-L119)）。
* **Primary vs. Auxiliary ネガティブ分割とマイニング**:
  最小正例類似度 $\min_{j \in \mathcal{P}_i^*} S_{ij} - \epsilon$ より大きいハード負例のうち、「良品 vs 不良（$\mathcal{N}_i^{\text{primary}}$、マージン $\lambda_{\text{primary}} = -0.15$、重み $w_{\text{defect}}$）」と「不良モードA vs 不良モードB（$\mathcal{N}_i^{\text{aux}}$、マージン $\lambda_{\text{aux}} = +0.15$、重み $w_{\text{aux}}$）」に異なるマージンと重みを設定する（[`ms_loss.py` L121-L152](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/ms_loss.py#L121-L152)）。
* **バッチ内良品プロトタイプ半球整列項 ($L_{\text{proto}}$)**:
  ミニバッチ内の良品平均を L2 正規化したプロトタイプ $\hat{\mu}_{\text{norm}}^{\text{batch}}$ に対し、良品を正のキャップ（$\ge 0.70$）へ凝集させ、不良を負の半球（$\le -0.25$）へ押し出す（[`ms_loss.py` L161-L174](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/ms_loss.py#L161-L174)）：
  $$L_{\text{proto}} = \frac{1}{|\mathcal{B}_{\text{norm}}|} \sum_{x \in \mathcal{B}_{\text{norm}}} \max\big(0, \; 0.70 - \langle z(x), \hat{\mu}_{\text{norm}}^{\text{batch}} \rangle\big) + w_{\text{defect}} \cdot \frac{1}{|\mathcal{B}_{\text{def}}|} \sum_{x \in \mathcal{B}_{\text{def}}} \max\big(0, \; \langle z(x), \hat{\mu}_{\text{norm}}^{\text{batch}} \rangle - (-0.25)\big)$$
* **3方向 XBM 双方向境界反発項 ($L_{\text{repel}}^{\text{XBM}}$)**:
  実効マージン $m^* = \min(m(t), -0.10)$ と温度 $\beta_{\text{repel}} = 12.0$ を用い、①ライブ不良 vs XBM 内 Top-$K_{\text{eff}}$ 境界良品、②ライブ良品 vs XBM 内全不良、③大域 XBM 良品プロトタイプ $\hat{\mu}_{\text{norm}}^{\text{XBM}}$ に対する正負半球 Softplus 整列の3項の和として計算する（[`ms_loss.py` L175-L232](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/ms_loss.py#L175-L232)）。

---

## 6. 類似手法（MS-Loss / Proxy-Anchor Loss / ArcFace）との徹底比較と違い

本システムで開発した **`CustomizedMultiProxyAnchorLoss`（50:50 パワーバランス Multi-Proxy Loss）** と、深層距離学習・顔認識で広く使われる代表的な3手法（**標準 MS-Loss**、**標準 Proxy-Anchor Loss**、**ArcFace**）の理論的・構造的な違いを以下に整理する。

### 6.1. 4手法の横断比較マトリクス

| 比較軸 | ① 標準 MS-Loss<br/>*(Wang et al., CVPR 2019)* | ② 標準 Proxy-Anchor Loss<br/>*(Kim et al., CVPR 2020)* | ③ ArcFace (Additive Angular Margin)<br/>*(Deng et al., CVPR 2019)* | ④ 本システム：`CustomizedMultiProxyAnchorLoss`<br/>*(50:50 Power-Balanced Multi-Proxy)* |
| :--- | :--- | :--- | :--- | :--- |
| **基本原理** | **Sample-to-Sample**<br/>バッチ内のサンプル対サンプルの類似度行列 $S_{ij}$ からペア重み付け | **Sample-to-Proxy (対称)**<br/>各クラス1個のプロキシをアンカーとし、`LogSumExp` で全クラス対称に学習 | **Classification Logit**<br/>クラス重みベクトル $W_y$ との角度 $\theta_y$ に加算マージン $m$ を加えた Softmax 交差エントロピー | **Asymmetric Sample-to-Multi-Proxy**<br/>良品 $M$ プロキシへの線形求心力と、固定アンカー `p_norm.detach()` からの不良ヒンジ斥力を **50:50** で結合 |
| **計算量（バッチサイズ $B$、クラス数 $C$）** | $O(B^2)$<br/>（ペア数爆発・バッチ内サンプル構成に強く依存） | $O(B \cdot C)$<br/>（プロキシ経由のため高速） | $O(B \cdot C)$<br/>（線形分類層と同等で高速） | **$O(B \cdot (M + C))$**<br/>（高速・ミニバッチのペア欠損に影響されない） |
| **1クラス1〜2枚の極少不良モードへの耐性** | ❌ **破綻しやすい**<br/>ミニバッチ内に同一不良が1枚しか無いとポジティブペア $\mathcal{P}_i = \emptyset$ となり学習シグナルが消失 | ⚪ **動作する**<br/>プロキシが常駐するためバッチ内1枚でも勾配は計算可能 | ⚠️ **過学習しやすい**<br/>1〜2枚の不良サンプルに向かって重みベクトル $W_c$ が過剰適合する | ✅ **極めて高い**<br/>常駐プロキシ ＋ `NAB-Sampler` 層化抽出 ＋ Effective-Number $W_c$ で1枚の不良モードでも安定学習 |
| **「不良A vs 不良B」の不要な斥力（勾配の分散）** | ❌ **発生する**<br/>不良クラス同士のネガティブペアが大量に作られ、良品vs不良の分離勾配が奪われる | ❌ **発生する**<br/>各不良プロキシが他の全不良サンプルを強く反発するため不良同士の押し合いに勾配が浪費される | ❌ **発生する**<br/>Softmax の分母で不良クラス同士が競合し、全クラスを等しくボロノイ分割しようとする | ✅ **発生しない（完全解消）**<br/>メイン斥力項 $L_{\text{repel}}$ は **「全不良 vs 良品プロキシ」のみ** に集中。不良間の分離はプロキシ同士の直交補助項 $L_{\text{ortho}}$ で穏やかに処理 |
| **良品クラスタの求心力特性（分散の収縮性）** | ⚠️ **不感帯あり**<br/>$S_{ik} > \lambda_{\text{pos}}$ または $\max \mathcal{N}_i + \epsilon$ を超えるとペアが破棄され収縮が止まる | ❌ **外れ値偏重・分散膨張**<br/>`Softplus` の指数重みにより最も遠い良品1枚に勾配の99%が集中し、さらに不良プロキシからの遠心力で良品が散乱する | ⚠️ **角度マージン内に留まる**<br/>他クラスとの識別境界さえ超えればそれ以上中心へ1点凝集する力は弱まる | ✅ **全域一定勾配（$-1.0$）で単調収縮**<br/>線形コサイン距離 $1 - S(x, p_{\text{norm}})$ により全良品に均等な求心力が働き続け、良品への遠心力もゼロ |
| **プロキシ（クラス中心）のドリフト耐性** | プロキシなし（バッチ内サンプルに依存） | ❌ **ドリフトする**<br/>不良サンプルからの強い反発勾配が良品プロキシにも直接流入し、良品中心が押し動かされる | ⚠️ **不均衡で歪む**<br/>多数派（良品）と少数派（不良）のサンプル数差で決定境界が少数派側へ押し込まれる | ✅ **`p_norm.detach()` で完全固定**<br/>不良斥力計算時に良品プロキシへの勾配を遮断し、良品プロキシは純粋に良品データの中心のみを追跡 |
| **未知の不良モード（Open-Set Anomaly）への汎化** | ⚪ 距離閾値判定が可能 | ⚪ 距離閾値判定が可能 | ❌ **閉集合（Closed-Set）前提**<br/>既知の不良クラス $1..C$ のいずれかに分類しようとするため、未知パターンの不良が良品側へ漏れやすい | ✅ **Open-Set 異常検知に最適**<br/>「良品プロキシとの類似度 $S(x, p_{\text{norm}}) < \tau_{\text{sim}}$ なら全て不良」という全方位包囲型の識別境界を形成 |

---

### 6.2. 各類似手法との詳細な数理的比較

#### (1) 標準 MS-Loss（Multi-Similarity Loss）との違い
標準 MS-Loss（Wang et al., CVPR 2019）は、ミニバッチ内のサンプル間類似度行列 $S_{ij} = \langle z_i, z_j \rangle$ のみを用いて次式を最小化する：
$$L_{\text{MS}}^{\text{std}} = \frac{1}{B} \sum_{i=1}^{B} \left\{ \frac{1}{\alpha} \log \left( 1 + \sum_{k \in \mathcal{P}_i} e^{-\alpha(S_{ik} - \lambda)} \right) + \frac{1}{\beta} \log \left( 1 + \sum_{k \in \mathcal{N}_i} e^{\beta(S_{ik} - \lambda)} \right) \right\}$$
* **外観検査での限界**:
  1. 異常クラスを $C$ 個のサブクラス（傷・汚れ・欠け等）に分けたとき、ミニバッチ内の不良サンプル同士（傷 vs 汚れ、傷 vs 欠け）がすべてネガティブ集合 $\mathcal{N}_i$ に入る。その結果、**「不良Aと不良Bを引き離すこと」に勾配の大半が費やされ、「良品と微小傷を引き離す」ための勾配が相殺・希釈される**。
  2. 特定の不良モードの枚数が少ない場合、バッチ内にその不良モードが1枚しか入らないと $\mathcal{P}_i = \emptyset$ となり、ポジティブ項が完全にゼロになる。
* **本システムの違い**:
  サンプル対サンプルの計算を廃止し、学習可能なプロキシ（`normal_proxies`, `defect_proxies`）を媒介させることで、バッチ内に1枚しか存在しない希少不良モードでも常に安定した勾配を生成し、かつ斥力を「良品プロキシ vs 不良サンプル」の1点に集中させている。

#### (2) 標準 Proxy-Anchor Loss（Kim et al., CVPR 2020）との違い
標準 Proxy-Anchor Loss は、全クラスのプロキシ集合 $P$（ポジティブプロキシ集合 $P^+$）に対して以下の対称な Softplus / LogSumExp 損失を定義する：
$$L_{\text{PA}}^{\text{std}} = \frac{1}{|P^+|} \sum_{p \in P^+} \log \left( 1 + \sum_{x \in \mathcal{X}_p^+} e^{-\alpha(s(x, p) - \delta)} \right) + \frac{1}{|P|} \sum_{p \in P} \log \left( 1 + \sum_{x \in \mathcal{X}_p^-} e^{\alpha(s(x, p) + \delta)} \right)$$
* **外観検査での3つの破綻メカニズム（本開発の実験で実証された課題）**:
  1. **LogSumExp による外れ値1点集中と不感帯**: 第1項において $\alpha = 16 \sim 32$ の指数重み $\exp(-\alpha(s - \delta))$ をとると、たとえばバッチ内に類似度 $s = 0.20$ の外れ値良品が1枚あるだけで、残りの良品（$s = 0.70 \sim 0.85$）に対する相対勾配比は $e^{-16(0.80 - 0.20)} = e^{-9.6} \approx 0.00006$ となり、**99.9%の良品サンプルに求心力が働かなくなる**。
  2. **良品サンプルへの遠心力混入（第2項の `p in P_defect`）**: 第2項を全プロキシ $p \in P$ で回すと、「不良プロキシ $p_{\text{def}}$ から良品サンプル $x_{\text{norm}}$ を遠ざける力」が働き、各不良プロキシが異なる方向から良品クラスタを突き飛ばすため、**エポックが進むほど良品クラスタの分散が膨張する**。
  3. **良品プロキシのドリフト**: 第2項で $p = p_{\text{norm}}$ に対する不良サンプル $x_{\text{anom}}$ の反発を計算する際、$p_{\text{norm}}$ のパラメータにも勾配が流れるため、不良群に押されて良品プロキシ自体が良品の重心から逃げてしまう。
* **本システムの違い**:
  1. 第1項を不感帯のない線形コサイン距離 $\frac{1}{|\mathcal{B}_{\text{norm}}|} \sum (1 - S(x, p_{\text{norm}}))$ に置き換え、全良品に均等な単位求心力（勾配 $-1.0$）を保証した。
  2. 良品サンプルに対する不良プロキシからの遠心力を完全に削除した。
  3. 第2項の不良斥力では `p_norm.detach()` によって良品プロキシを固定アンカー化し、かつ重み $\tilde{w}(x)$ を平均 $1.0$ に正規化したヒンジ関数 $\max(0, S - \delta)$ とすることで、第1項と第2項が常に **`[0.0, 1.0]` の同一スケール（50 : 50）** で釣り合うように設計した（実測で良品クラスタ内分散 $\sigma^2_{\text{norm}}$ が `0.5621` $\to$ `0.0481` へと **91.4% 収縮** することを確認済み）。

#### (3) ArcFace（Additive Angular Margin Loss, Deng et al., CVPR 2019）との違い
ArcFace は顔認識等で標準的に用いられる分類ベースの距離学習手法であり、クラス重みベクトル $W_j \in \mathbb{S}^{D-1}$ と特徴ベクトル $z_i \in \mathbb{S}^{D-1}$ のなす角 $\theta_{j} = \arccos(\langle z_i, W_j \rangle)$ に対して角度マージン $m$ を加算した Softmax 交差エントロピー損失である：
$$L_{\text{ArcFace}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \ne y_i} e^{s \cdot \cos \theta_j}}$$
* **外観検査での限界**:
  1. **閉集合（Closed-Set）ボロノイ分割の仮定**: ArcFace は「すべてのクラス $j \in \{0, 1, \dots, C\}$ が同等の面積・凝集度を持つ独立カテゴリである」と仮定して超球面を分割する。そのため、学習時に存在しなかった**未知の不良モード（Open-Set Anomaly）** が入力された際、既存のどの不良重みベクトル $W_1, \dots, W_C$ とも角度が一致せず、最大面積を占める良品クラス $W_0$ 側へ誤分類されやすい。
  2. **少数不良クラスの過学習と不良間競合**: Softmax の分母 $\sum_{j \ne y_i} e^{s \cos \theta_j}$ により、不良クラス $y_i = 1$ のサンプルは不良クラス $j = 2, 3, \dots, C$ とも強く反発し続ける。また 1クラス数枚の不良データを狭い角度コーン $\theta_{y_i} + m$ に無理やり押し込めると、背景ノイズまで暗記する過学習を起こす。
* **本システムの違い**:
  本システムは Softmax 分類器を使わず、**「良品プロキシ群（自動決定 $M^*$ 本・Möbius 許容半径較正済み）を中心とする超球面キャップ集合の内側（$S_{\text{cal}}(x, p_{\text{norm}}) \ge \tau_{\text{sim}}$）だけが正常であり、その外側の全空間（$S_{\text{cal}}(x, p_{\text{norm}}) < \tau_{\text{sim}}$）は既知・未知を問わずすべて異常である」** という **非対称オープンセット包囲設計** を採っている。これにより、良品内の多峰性（キャビティ差等）を吸収しつつ、既知・未知の不良モードに対して高い検出力（Recall）と低過検出（高 Precision）を両立できる。

---

## 7. 不均衡対策サンプラー（NAB-Sampler）と境界ハードサンプル合成（Phase 3）

### 7.1. Normal-Anomaly Balanced Sampler (`NAB-Sampler`) と少数不良・少数良品モードの枯渇回避アルゴリズム
実装ファイル: [`trainer/app/core/nab_sampler.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/nab_sampler.py)

100 : 2 のような不均衡データセットで通常のランダムミニバッチ抽出を行うと、ミニバッチの大半が良品で埋め尽くされ、不良データの勾配が埋没する。逆に単純に良品と不良を 50% : 50% で非復元抽出しようとすると、枚数の少ない不良データが数ステップで先に枯渇してしまう。さらに良品内部に複数のサブクラスタ（$M^* \ge 2$）がある場合、無作為抽出では少数派キャビティの良品がミニバッチから欠落して該当プロキシの更新が滞る。

[`NormalAnomalyBalancedSampler`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/nab_sampler.py#L41-L241) は、以下の **非対称2プール巡回・サブモード層化オーバーサンプリング方式** によってこの問題を解決している：

1. **ミニバッチスロットの分割（デフォルト `batch_size = 32`, `nab_normal_ratio = 0.50`）**:
   * **正常（良品）スロット（50% ＝ 16枚）**:
     * 前ステップまでに Cross-Batch Memory Bank (XBM) が特定した「不良に最も近い境界ハード良品インデックス (`hard_normal_indices`)」が存在する場合、良品スロットの最大 **35%（最大5枚）** にその境界ハード良品を優先注入する（[`nab_sampler.py` L164-L169](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/nab_sampler.py#L164-L169)）。
     * 残りの良品スロットは、$M^* \ge 2$ のときは各良品サブクラスタ `normal_by_subcluster[sc_id]` に均等割り当てして巡回ポインタ `sub_cursors[sc_id]` で層化抽出し（[`nab_sampler.py` L172-L198](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/nab_sampler.py#L172-L198)）、$M^* = 1$ のときは全良品リスト `norm_pool` を巡回ポインタ `norm_cursor` で順番に取り出す。1エポックあたりの総バッチ数を `max(min_steps, ceil(|norm_pool| / n_norm_slots) * 2)`（`min_steps` はメソッドデフォルト `12`、[`train_jobs.py` L638](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L638) での実行時は `14`）に設定しているため、**すべての良品データおよびすべての良品サブモードが1エポックにつき必ず2回以上漏れなく学習される**。
   * **異常（不良）スロット（50% ＝ 16枚）**:
     * 存在する不良クラス $c \in \{1, \dots, C\}$ を選択し、16枚のスロットを各不良クラスへ均等に割り当てる（例: 4クラスなら各4枚）。
     * クラス $c$ の実サンプル数 $|D_c|$ が割り当て枚数 $k_{\text{mode}}$ より少ない場合（例: 1〜2枚しか無い希少不良モード）、そのクラス内部で `replace=True`（復元抽出）を自動的に有効化する（[`nab_sampler.py` L233-L236](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/nab_sampler.py#L233-L236)）。
     * 抽出されたサンプルには、**非対称ワークの位置情報を一切壊さない微小平行移動（$\pm 2\%$）＋微小輝度・コントラスト摂動（$0.04$）の3ビュー特徴バンク**（[`defect_micro_aug_bank` (`train_jobs.py` L587-L594, L665-L670)](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L587-L594)：左右・上下反転や回転は完全禁止）、特徴空間上の微小ガウシアン摂動 `noise = randn * 0.012`、および入力特徴 `Dropout(0.20)` が適用されるため、同じ元画像が複数回サンプリングされても背景暗記（過学習）やポジティブペア欠損を起こさず安定してプロキシ更新に寄与する。

### 7.2. Phase 3：非対称ワーク対応・同座標 Zスコア残差サリエンシー欠陥転写 ＆ 同一サブクラスタ純粋異常残差注入
実装ファイル: [`trainer/app/core/boundary_augmentation.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/boundary_augmentation.py)

従来の大域的な線形補間（異位相・異キャビティ間の Anomalous Mixup）は、良品サブクラスタ同士を結ぶ弦（良品多様体の内部）に偽の不良斥力源を配置して良品の過検出（FP）を誘発し、またランダム座標への CutPaste は非対称ワークの空間座標意味を破壊する。そこで本システムでは大域的線形 Mixup を完全に廃止し、`enable_boundary_aug = True` のとき **「空間座標の保持」と「同一良品サブクラスタ内限定」** を厳守した以下の2種類の境界ハード異常サンプルを生成する（生成された合成サンプルはすべて第2項の実不良とは分離された正半球浅層ガードレール斥力 $\delta_{\text{synth}} = \max(\delta + 0.05, +0.05)$・固定小重み **$0.15\times$** として作用させ、良品クラスタを侵食せずに境界の安全網として機能させる）：

1. **同座標 Zスコア残差サリエンシー欠陥転写 ([`synthesize_defect_transfer_images` (`boundary_augmentation.py` L80-L193)](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/boundary_augmentation.py#L80-L193))**:
   フォルダアップロードされた検査画像にはピクセル単位の欠陥マスクが存在せず、また非対称ワークの外形エッジには微小な位置決め公差が存在する。そこで不良画像 $I_{\text{bad}}$ に対して最も低解像度 L1 距離が近い参照良品 $I_{\text{ref}}$ とその**同一良品サブクラスタ候補群**（[`_select_same_subcluster_normal_pool`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/boundary_augmentation.py#L31-L76)）を検索し、ピクセルごとの良品標準偏差 $\sigma_{\text{norm}}(u, v)$ で割った **Zスコア残差サリエンシーマップ**
   $$Z(u, v) = \text{AvgPool}_{7\times 7}\left( \frac{|I_{\text{bad}}(u, v) - I_{\text{ref}}(u, v)|}{\sigma_{\text{norm}}(u, v) + 0.03} \right)$$
   を計算して外形エッジの位置ズレ高分散を抑制し、真の欠陥ピーク座標 $(u^*, v^*) = \arg\max Z(u, v)$ を特定する。その同座標（治具公差内の微小シフト $|\Delta u|, |\Delta v| \le 2\text{px}$ のみ許容）を中心とする局所ガウシアン窓 $G(u, v)$（$\sigma_{\text{ratio}} = \text{clip}(0.035 + 0.05(\alpha_{\min} + \alpha_{\max}), 0.07, 0.14)$［デフォルト `0.10`］、最大不透明度 $0.88$）を用い、**同一サブクラスタ内の別良品 $I_{\text{good}}$ の同じ座標** へパッチブレンドと残差加算のハイブリッド転写を行う：
   $$I_{\text{transfer}} = 0.5 \cdot \Big((1 - G) I_{\text{good}} + G \cdot \text{shift}(I_{\text{bad}})\Big) + 0.5 \cdot \text{clip}\Big(I_{\text{good}} + G \cdot \text{shift}(I_{\text{bad}} - I_{\text{ref}}), \; I_{\min}, \; I_{\max}\Big)$$
2. **同一サブクラスタ純粋異常残差抽出・注入 ([`synthesize_boundary_gap_features` (`boundary_augmentation.py` L221-L311)](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/boundary_augmentation.py#L221-L311))**:
   標準化特徴空間上では、各不良特徴 $h_{\text{bad}}$ から最近傍の参照良品 $h_{\text{ref}}$ を引いて正常背景モード成分を打ち消した **純粋異常残差ベクトル $\Delta_{\text{def}} = h_{\text{bad}} - h_{\text{ref}}$** を抽出する。次に $h_{\text{ref}}$ と**同一の良品サブクラスタかつ局所コサイン近傍（$\cos \ge 0.45$）** に属するベース良品 $h_{\text{base}}$ を選び、`|\Delta_{\text{def}}|` の上位異変チャネルマスク $M_{\text{top}}$（Variant 1: 上位 $30\%$・維持率 $80\%$ Dropout、Variant 2: 上位 $45\%$・維持率 $75\%$ Dropout）と強度係数 $\beta_1 \sim \mathcal{U}(\alpha_{\min}+0.20, \alpha_{\max}+0.50)$（デフォルト `[0.75, 1.25]`）、$\beta_2 \sim \mathcal{U}(\alpha_{\min}+0.05, \alpha_{\max}+0.35)$（デフォルト `[0.60, 1.10]`）を用いて同一モード外縁へ注入・L2正規化する：
   $$h_{\text{synth}} = \text{L2Normalize}\Big( h_{\text{base}} + \beta \cdot \big(\text{Dropout}(M_{\text{top}}) \odot \Delta_{\text{def}}\big) \Big)$$
   これにより、異なる良品サブクラスタ間の空白地帯を一切汚染せず、各良品モードの外縁にのみ多様な疑似微細欠陥バリアを形成して見逃し（FN）と過検出（FP）の双方を抑制する。

---

## 8. 全ハイパーパラメータ完全リファレンス（コード対応・意味・調整指針）

実装スキーマ: [`trainer/app/schemas/job_models.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/schemas/job_models.py#L21-L84), [`backend/app/schemas/api_models.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/backend/app/schemas/api_models.py#L109-L198), [`frontend/app/pages/2_training_monitor.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/2_training_monitor.py#L282-L449)

### 8.1. フェーズ1：前処理クラスタリング ＆ データ分割パラメータ

| コード変数名 (`ClusteringJobRequest` 等) | UI 表示名 | デフォルト値 | 設定可能範囲 (UI / API) | 数理的意味とチューニング指針 |
| :--- | :--- | :---: | :---: | :--- |
| `dataset_name` | 対象データセット名 | `None` (現在選択中) | 登録済みデータセット名 | クラスタリング対象のデータセット識別子（省略時はアクティブデータセットを使用）。 |
| `backbone_name` | 特徴抽出バックボーン (timm) | `"efficientnet_b3"` | `"efficientnet_b3"`, `"vit_base_patch16_224"`, `"resnet18"`, `"lightweight_cnn"` | 画像から高次元特徴マップを抽出するネットワーク。表面の微細テクスチャ・傷には `efficientnet_b3`、大域的な構造欠損には `vit_base_patch16_224` が適する。 |
| `pooling_type` | 空間プーリング層 | `"gem"` | `"gem"`, `"avg_max"`, `"gap"` | 大域特徴マップの空間集約方式（これに加えて非対称ワーク向けに `4x4` CNN 空間座標グリッド特徴 `512D` が自動併用される）。`"gem"`（$p=3.0$）または `"avg_max"` は局所的な微小傷の活性化が背景平均で薄まるのを防ぐ。 |
| `pretrained` | ImageNet 事前学習重みの使用 | `True` | `True` / `False` | `timm` バックボーンの事前学習重みをロードするかどうかのフラグ（フォールバック時は決定論的ステムを使用）。 |
| `normal_class_name` | 良品クラス名（固定保持） | `"normal"` (自動検出) | 任意のサブフォルダ名 | この名前と一致するサブフォルダの画像は不良と混ぜてクラスタリングせず、`Cluster 0`（良品: `label_id=0`）として100%固定保持する（内部の潜在サブモード `normal_subcluster_id` のみ自動診断・記録）。 |
| `max_k` | 異常クラスの最大分割数 ($K_{\max}$) | `6` (UI) / `8` (API) | `2` 〜 `15` (UI)<br/>`2` 〜 `20` (API) | 異常クラスの画像を K-means で自動細分化する際の探索上限。$k \in [2, K_{\max}]$ の中でコサイン・シルエットスコアが最大となる $k^*$ が自動選択される。 |
| `max_pca_dim` | PCA 最大次元数 | `64` | `2` 〜 `256` (API) | 異常クラスタリング前段の PCA 次元上限。高次元空間での「次元の呪い」を抑えシルエット計算を安定化させる。 |
| `pca_variance` | PCA 累積寄与率閾値 | `0.95` | `0.50` 〜 `0.999` (API) | 累積分散説明率がこの値（95%）に達する主成分数までを保持する。 |
| `core_quantile` | 密度フォールバック時の良品割合 | `0.85` | `0.50` 〜 `0.95` (UI)<br/>`0.50` 〜 `0.97` (API) | サブフォルダ分けが無い単一フォルダ投入時（`density_fallback`）のみ使用。密度スコア上位何割を仮の良品コアとするかを表す。 |
| `k_nn` | 局所密度計算の近傍数 | `15` | `2` 〜 `100` (API) | `density_fallback` 時に局所コサイン密度を計算する際の近傍サンプル数。 |
| `train_ratio` / `val_ratio` / `test_ratio` | 学習 : 評価 : テスト 分割比率 | `0.60` / `0.20` / `0.20` | `train`: `0.20〜0.80`<br/>`val`: `0.05〜0.45`<br/>`test`: `0.05〜0.45`<br/>(合計で正規化) | `(label_name, normal_subcluster_id)` 別層化分割の比率（**6 : 2 : 2**）。`train` (60%) で学習、`val` (20%) で次エポックのLoss方向性制御とBestモデル選定、`test` (20%) で最終性能を評価する。 |

---

### 8.2. フェーズ2：50:50 パワーバランス Multi-Proxy Anchor Loss パラメータ

| コード変数名 (`TrainingJobRequest`) | UI 表示名 | デフォルト値 | 設定可能範囲 (UI / API) | 数理的意味とチューニング指針 |
| :--- | :--- | :---: | :---: | :--- |
| `loss_type` | 距離学習アルゴリズム | `"multi_proxy_anchor"` | `"multi_proxy_anchor"`, `"ms_loss"` | `"multi_proxy_anchor"`（推奨）で 50:50 パワーバランス Multi-Proxy Loss を使用。`"ms_loss"` で従来のペアベース非対称 MS-Loss を使用。 |
| `normal_loss_ratio` | ⚖️ 良品求心力 : 不良斥力 のパワーバランス比率 ($r_{\text{norm}}$) | **`0.50`** | `0.10` 〜 `0.90` (UI)<br/>`0.05` 〜 `0.95` (API) | $\lambda_{\text{norm}} = 2 r_{\text{norm}}, \; \lambda_{\text{anom}} = 2(1 - r_{\text{norm}})$ を決定する最重要パラメータ。<br/>・**`0.50`（デフォルト）**: 良品求心力 50% ($\lambda_{\text{norm}}=1.0$) : 不良斥力 50% ($\lambda_{\text{anom}}=1.0$)。<br/>・**`0.60 〜 0.70`**: 良品クラスタの分散 $\sigma^2_{\text{norm}}$ をさらに小さく絞り込み、過検出（FP）を減らしたい場合に上げる。<br/>・**`0.35 〜 0.45`**: 微細不良の見逃し（FN）があり、不良をより強く遠ざけたい場合に下げる。 |
| `repel_delta` | 🎯 不良斥力コサインマージン ($\delta$) | **`0.00`** | `-0.50` 〜 `+0.50` (UI)<br/>`-0.80` 〜 `+0.80` (API) | 実不良斥力項 $\max(0, S_{\text{detach}}(x, p_{\text{norm}}) - \delta)$ の目標境界コサイン類似度（合成サンプルのガードレールは $\delta_{\text{synth}} = \max(\delta + 0.05, +0.05)$ に自動設定）。<br/>・**`0.00`（デフォルト）**: 良品プロキシと直交（90°）まで実不良を押し出す。このとき第1項と第2項がともに `[0.0, 1.0]` の完全対称スケールになる。<br/>・**負の値（例: `-0.15`）**: 実不良を反対側の負の半球（鈍角 $> 90^\circ$）までより深く押し込みたい場合に設定。 |
| `num_normal_proxies` | 良品プロキシ数 ($M$: `0`=自動判定) | **`0`** (UI / API 共通) | `0` 〜 `12` | 良品クラスタを表現するサブセンタープロキシの本数。<br/>・**`0`（推奨デフォルト `Auto`）**: `train` 良品の許容被覆半径 $\epsilon = 0.22$・コサインシルエット係数（$\ge 0.25$）・90%分位分散低減率（$\ge 18\%$）から最適数 $M^*$ を自動決定する（単峰なら $M^* = 1$、キャビティ差等の多峰性があれば $M^* \in \{2 \dots 6\}$ に自動分割し Möbius 許容半径較正を適用）。<br/>・**`1 〜 12`（手動固定）**: 既知の金型キャビティ数・品種モード数に合わせてプロキシ本数を明示指定したい場合に設定。 |
| `hard_anomaly_boost` | Top-K 危険不良 相対優先度 ($\lambda_{\text{hard}}$) | **`3.0`** | `1.0` 〜 `20.0` | 実不良バッチの中で良品プロキシに最も近い上位 $K_{\text{eff}}$ 件（見逃し危険度が高いハード実不良）に掛ける相対重み倍率。実不良内で平均 $1.0$ に自動正規化されるため、値を上げても 50:50 の全体バランスは崩れず、実不良内部でのリソース集中度だけが高まる。 |
| `class_balance_beta` | Class-Balanced Weight $\beta$ | **`0.99`** | `0.80` 〜 `0.9999` (UI)<br/>`0.50` 〜 `0.9999` (API) | Cui et al. (CVPR 2019) の有効サンプル数 $(1-\beta)/(1-\beta^{n_c})$ の減衰率。`0.99` に設定すると、サンプル数が数枚しか無い希少不良サブクラスの重み $W_c$ が自動的に引き上げられる。 |
| `w_defect` | 不良クラス基準重み | `1.0` (API/UI)<br/>(`2.5` in `ms_loss.py`) | `0.1` 〜 `10.0` (UI)<br/>`>= 0.1` (API) | `ms_loss` 選択時の Primary Negative 重み倍率および不良アンカー重み（`multi_proxy_anchor` では `normal_loss_ratio` と平均正規化重み $\tilde{w}_{\text{real}}(x)$ によって制御される）。 |
| `w_aux` | プロキシ直交補助重み ($w_{\text{ortho}}$) / 異種不良間重み | `0.20` (API/UI)<br/>(`0.5` in `ms_loss.py`) | `0.0` 〜 `2.0` (UI)<br/>`>= 0.0` (API) | `multi_proxy_anchor` では不良プロキシ間の直交化および良品サブプロキシ間の北半球共有コーン制約（$[+0.20, +0.75]$）を担う $L_{\text{ortho}}$ の重み係数（[`train_jobs.py` L502](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L502) にて下限 `max(0.1, req.w_aux)` で保護）。`ms_loss` では Auxiliary Negative の重み。 |

---

### 8.3. フェーズ2：学習制御・NAB-Sampler・境界ハード合成・XBM カリキュラムパラメータ

| コード変数名 (`TrainingJobRequest`) | UI 表示名 | デフォルト値 | 設定可能範囲 (UI / API) | 数理的意味とチューニング指針 |
| :--- | :--- | :---: | :---: | :--- |
| `session_label` | セッション識別ラベル | `None` (自動生成) | 任意の文字列 | 学習セッション履歴（`session_XXXX`）に付与する任意のメモ・ラベル。 |
| `freeze_backbone` | バックボーン凍結フラグ | `True` | `True` / `False` | `True`（デフォルト）で事前学習バックボーンを凍結し特徴キャッシュによる高速・省メモリ学習を実行。 |
| `unfreeze_last_block` | 最終ブロックのみ Unfreeze | `False` | `True` / `False` | `freeze_backbone=True` 時にバックボーンの最終ステージ（`blocks.5/6`, `layer4` 等）のみ勾配更新を許可する微調整フラグ。 |
| `epochs` | 学習エポック数 | `20` (UI) / `10` (API) | `2` 〜 `100` (UI)<br/>`1` 〜 `200` (API) | 総学習エポック数。特徴キャッシュが有効（`freeze_backbone=True`）な場合、100エポックでも数秒〜十数秒で高速に完了する。 |
| `batch_size` | ミニバッチサイズ ($B$) | `32` | `8` 〜 `128` (UI)<br/>`4` 〜 `256` (API) | 1ステップあたりのサンプル数。`NAB-Sampler` により `batch_size * nab_normal_ratio` 枚が良品（$M^* \ge 2$ 時は各サブモード均等）、残りが不良に割り当てられる。 |
| `learning_rate` | 学習率 (Learning Rate) | `0.002` (`2e-3`) | `1e-5` 〜 `5e-2` (UI)<br/>`(0.0, 1.0]` (API) | `AdamW` オプティマイザ（`weight_decay=1e-4`）の Projection Head 学習率。プロキシパラメータには `0.5 * learning_rate` が適用され、プロキシの急激な振動を防ぐ。 |
| `nab_normal_ratio` | 📦 ミニバッチ内 良品サンプル比率 | **`0.50`** | `0.30` 〜 `0.70` (UI)<br/>`0.25` 〜 `0.75` (API) | `NAB-Sampler` が構成するミニバッチ内の良品スロット比率（デフォルト `0.50` ＝ 良品50%［サブクラスタ均等層化］ : 全不良モード均等50%）。 |
| `enable_boundary_aug` | 同座標 Zスコア欠陥転写 ＆ 異常残差注入 を有効化 | **`True`** | `True` / `False` | `True` のとき、同一良品サブクラスタ内限定の同座標（$\pm 2\text{px}$）Zスコア欠陥転写と純粋異常残差 $\Delta_{\text{def}}$ 注入を行い、正半球浅層ガードレール（重み $0.15\times$）として境界を補強する（異位相間の線形 Mixup は実行しない）。 |
| `mixup_alpha_min` | 境界合成 最小強度係数 ($\alpha_{\min}$) | **`0.55`** | `0.30` 〜 `0.85` (UI)<br/>`0.30` 〜 `0.90` (API) | 異常残差注入スケール $\beta_1 \sim \mathcal{U}(\alpha_{\min}+0.20, \alpha_{\max}+0.50)$・$\beta_2 \sim \mathcal{U}(\alpha_{\min}+0.05, \alpha_{\max}+0.35)$ および欠陥転写ガウシアン窓幅の下限を決める強度係数。 |
| `mixup_alpha_max` | 境界合成 最大強度係数 ($\alpha_{\max}$) | **`0.75`** | `0.35` 〜 `0.90` (UI)<br/>`0.35` 〜 `0.95` (API) | 境界合成強度の上限係数。デフォルト `0.55 〜 0.75` で $\beta_1 \in [0.75, 1.25], \; \beta_2 \in [0.60, 1.10]$・転写窓 $\sigma_{\text{ratio}} = 0.10$ の推奨スイートスポットとなる。 |
| `top_k_mode` | Top-K 指定モード | `"ratio"` | `"ratio"`, `"count"` | ハード実不良および XBM 境界良品の抽出数を「母数に対する割合 (`"ratio"`)」で指定するか「絶対件数 (`"count"`)」で指定するかの切替。 |
| `top_k_initial` | 初期 Top-K 値（学習初期） | `0.30` (`ratio` API/UI)<br/>`30` (`count` UI) / `50.0` (`scheduler.py`) | `0.01` 〜 `500.0` (UI)<br/>`> 0.0` (API) | 学習初期（進捗度 $p(t) \approx 0$）における Top-$K$ 対象範囲。初期は上位30%と広く取って大局的な境界を形成する。 |
| `top_k_final` | 後期 Top-K 値（学習後期） | `0.15` (`ratio` API/UI)<br/>`20` (`count` UI / `scheduler.py`) | `0.01` 〜 `500.0` (UI)<br/>`> 0.0` (API) | 学習後期（進捗度 $p(t) \to 1$）における Top-$K$ 対象範囲。後期は上位15%の最も危険な境界サンプルへフォーカスを絞る。 |
| `gamma_initial` / `gamma_final` | 初期 / 後期 反発係数 ($\gamma_{\text{init}}, \gamma_{\text{final}}$) | `0.6` / `1.5` (API/UI)<br/>(`0.3` / `1.5` in `scheduler.py`) | `0.1〜5.0` / `0.1〜10.0` (UI)<br/>`>= 0.0` (API) | カリキュラム進捗度 $p(t)$ に連動する境界反発強度の初期値と最終値（`scheduler.py` 内のマージン定数は `margin_initial=0.15, margin_final=-0.10, target_d_min=1.0`）。 |
| `tau_sim` | 評価基準コサイン類似度閾値 ($\tau_{\text{sim}}$) | **`0.00`** | `-0.80` 〜 `+0.80` (UI)<br/>`-1.00` 〜 `+1.00` (API) | 較正後コサイン類似度 $S_{\text{cal}}(x, p_{\text{norm}}) < \tau_{\text{sim}}$ を不良（NG）、$\ge \tau_{\text{sim}}$ を良品（OK）と判定する閾値。学習後も Step 6 のスライダーでリアルタイムに再計算可能。 |
| `full_snapshot_interval` | フルスナップショット保存間隔 | `4` | `>= 1` (API) | 高次元 `gap_features` を含む `epoch_XXXX_full.pkl` を定期保存するエポック周期（エポック1・ベスト更新該当時・定期周期・最終エポックで保存）。 |

---

## 9. 空間評価指標（Step 6）の数理定義と解釈ガイド

実装ファイル: [`trainer/app/core/evaluator.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/evaluator.py), [`backend/app/routers/evaluation.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/backend/app/routers/evaluation.py), [`frontend/app/pages/3_space_evaluation.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/3_space_evaluation.py)

Step 6 では、対象データ分割（デフォルトは未知テストデータ `test` 20%、切替で `val` 20% / `train` 60% / `all` 100%）に対して以下の定量指標を算出・可視化する。

### 9.1. 識別性能指標（閾値 $\tau_{\text{sim}}$ 連動）
各サンプル $x_i$ の良品プロキシ群に対する **Möbius 許容半径較正済みコサイン類似度**（[`compute_calibrated_normal_similarities_np` (`evaluator.py` L19-L40)](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/evaluator.py#L19-L40)）
$$s(x_i) = S_{\text{cal}}(x_i, p_{\text{norm}}) = 1.0 - \min_{m \in \{1, \dots, M\}} \frac{2 \alpha_m \big(1 - \langle z(x_i), \hat{p}_{\text{norm}}^{(m)} \rangle\big)}{(\alpha_m - 1)\big(1 - \langle z(x_i), \hat{p}_{\text{norm}}^{(m)} \rangle\big) + 2}$$
（※ $M = 1$ または未較正時は $\alpha_1 = 1.0$ となり、通常のコサイン類似度 $\langle z(x_i), \hat{p}_{\text{norm}}^{(1)} \rangle$ に厳密一致する）に対し、$\hat{y}_i = \mathbb{I}[s(x_i) < \tau_{\text{sim}}]$（$1$: NG判定、$0$: OK判定）としたときの混同行列要素（`TP`: 実不良をNG検出、`FN`: 実不良をOK見逃し、`FP`: 実良品をNG過検出、`TN`: 実良品をOK合格）から以下を算出する：

* **Accuracy（総合正解率）**:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
* **Recall（不良検出率 / 再現率）** — *製造外観検査における最重要指標（`1.0000` で見逃しゼロ）*:
  $$\text{Recall} = \frac{TP}{TP + FN}$$
* **Precision（適合率）** — *（`1.0000` で良品の過検出ゼロ）*:
  $$\text{Precision} = \frac{TP}{TP + FP}$$
* **F1 スコア（Recall と Precision の調和平均）**:
  $$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 9.2. 高次元空間マージン・クラスタ分散指標（閾値非依存の空間品質指標）
評価対象分割内の良品サンプル添字集合を $\mathcal{D}_{\text{norm}}$、不良サンプル添字集合を $\mathcal{D}_{\text{def}}$、不良サブクラス $c \in \{1, \dots, C\}$ のサンプル添字集合を $\mathcal{D}_c$ とする：

* **境界最小コサイン余裕度 (`d_min_boundary`)**:
  $$d_{\min}^{\text{boundary}} = \min_{i \in \mathcal{D}_{\text{norm}}} s(x_i) - \max_{j \in \mathcal{D}_{\text{def}}} s(x_j)$$
  最も外側に位置する良品の類似度と、最も良品側に食い込んでいる不良の類似度の差。**この値が $> 0.0$ であれば、良品分布と不良分布に重なりが一切なく、適切な閾値 $\tau_{\text{sim}}$ によって `Accuracy = 1.0, Recall = 1.0, Precision = 1.0`（完全分離）が達成可能であることを保証する**。
* **中心間最小コサイン距離 (`d_min_centroid` / `inter_class_dist` $D_{\text{inter}}$)**:
  $$D_{\text{inter}} = 1.0 - \frac{1}{|\mathcal{D}_{\text{def}}|} \sum_{j \in \mathcal{D}_{\text{def}}} s(x_j)$$
  良品プロキシ中心から不良群までの平均コサイン距離（直交時 $1.0$、正反対時 $2.0$）。大きいほど不良群が良品から遠くへ押し出されている。
* **良品クラスタ内分散 ($\sigma^2_{\text{norm}}$ / `normal_intra_disp`)**:
  $$\sigma^2_{\text{norm}} = \frac{1}{|\mathcal{D}_{\text{norm}}|} \sum_{i \in \mathcal{D}_{\text{norm}}} \big(1.0 - s(x_i)\big)$$
  良品サンプル群が良品プロキシの周りにどれだけ密に集まっているかを表す平均コサイン距離。**エポックが進むにつれて `0.0` に向かって単調減少していること**（特に `train` 60% と `test` 20% の双方で低く収まっていること）が健全な学習の証明となる。
* **不良モード内平均分散 ($\sigma^2_{\text{def}}$ / `defect_intra_disp`)**:
  $$\sigma^2_{\text{def}} = \frac{1}{C} \sum_{c=1}^{C} \frac{1}{|\mathcal{D}_c|} \sum_{i \in \mathcal{D}_c} \big(1.0 - \langle z_i, \mu_c \rangle\big)$$
  各不良サブクラス内の重心 $\mu_c$ まわりの平均コサイン距離。
* **フィッシャー空間分離比スコア ($J(t)$ / `fisher_separation_ratio`)**:
  $$J(t) = \frac{D_{\text{inter}}(t)}{\sigma^2_{\text{norm}}(t) + \sigma^2_{\text{def}}(t) + 10^{-4}}$$
  「クラス内分散の和」に対する「クラス間距離」の比。エポック進行に伴って $J(t)$ が右肩上がりに増大するほど、**「同クラス内は鋭く凝集し、良品と不良の間は大きく離れている」** 理想的な超球面特徴空間が形成されていることを意味する。
