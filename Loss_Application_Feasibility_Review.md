# 開発済み損失関数（Multi-Proxy Anchor / 非対称MS-Loss）の他領域・基盤モデル学習への適用可能性に関する技術検討書

> [!IMPORTANT]
> 本ドキュメントは、リポジトリ内の実装コード（[proxy_anchor_loss.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py)、[ms_loss.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/ms_loss.py)、[two_stage_cluster.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/two_stage_cluster.py)、[feature_extractor.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py)、[boundary_augmentation.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/boundary_augmentation.py)、[nab_sampler.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/nab_sampler.py)、[scheduler.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/scheduler.py)、[train_jobs.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py)、[evaluator.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/evaluator.py)）および仕様書（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md)、[Model_Explanation.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/Model_Explanation.md)）を直接監査し、さらに3つの独立した EGM Goldfish レビュー（コード整合性・数理最適化/スケーラビリティ・YAGNI/代替手法比較）を経て確定した技術検討報告書である。

---

## 1. 前提条件の検証：現行コードの数理的実態と「多クラス対多クラス分類」との構造的乖離

各領域の可否判定に先立ち、「1クラスに内包される微小なグループを分けることで多クラス対多クラスの分類を実施する」という前提認識と、**現行実装の実際の数理挙動との間にある4つの構造的乖離**を整理する。この前提差を無視して他領域へ移植すると学習または推論が破綻する。

### 1.1. 乖離①：現行実装は「対称な多クラス対多クラス」ではなく「非対称な 1基準クラス（良品）対 残余群（不良）」専用の2値包囲設計である

主力損失関数 [`CustomizedMultiProxyAnchorLoss`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L14-L387) の入力分割（[proxy_anchor_loss.py L213-L216](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L213-L216)）は `is_defect = labels > 0` によって2群へハードコード分離されている（なお `z_norm = z[~is_defect]` は **`labels <= 0`** 全件を良品として扱うため、物体検知・セグメンテーション・NLPで標準的な無視ラベル `ignore_index = -1` / `-100` を入力すると無視領域がすべて良品中心へ引き寄せられる実装上の罠がある）。

* **良品群（`labels <= 0`）**: $M$ 個の良品サブプロキシ $\hat{p}_{\text{norm}}^{(1..M)} \in \mathbb{S}^{127}$ に対し、最大類似度 $S(x, p_{\text{norm}}) = \max_m \langle z(x), \hat{p}_{\text{norm}}^{(m)} \rangle$ を用いて線形コサイン距離 $1.0 - S(x, p_{\text{norm}})$ による求心力のみを作用させる（$M=1$ 時は全域一定勾配 $-1.0$、$M>1$ 時は `0.90 * primary + 0.10 * subcenter_anchor`：[proxy_anchor_loss.py L223-L236](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L223-L236)）。良品サンプルには不良プロキシからの遠心力を一切与えない。
* **不良サブクラス群（`labels > 0`, $1..C$）**: 勾配を遮断した固定良品アンカー `p_norm.detach()` に対して、主マージン $\delta$（85%）＋深部マージン $\delta_{\text{deep}} = \min(\delta - 0.25, -0.25)$（15%）の2段ヒンジ斥力、および [`CrossBatchMemoryBank`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/memory_bank.py#L14-L167) 稼働時の境界良品ヒンジ（15%ブレンド）を作用させる（[proxy_anchor_loss.py L264-L328](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L264-L328)）。
* **補助正則化項 $L_{\text{aux}}$（[proxy_anchor_loss.py L330-L372](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L330-L372)）**: ①実不良サンプルの対応モードプロキシへの緩やかな凝集 `0.10 * loss_def_sub`、②全不良プロキシを良品プロキシに対し負半球（$\le -0.20$）へ押し込む `0.10 * loss_def_proxy_hemisphere`、③プロキシ間重なり抑制 `ortho_weight * loss_ortho`（不良プロキシ間上限 `ortho_margin_defect = 0.20` ＋ $M>1$ 時の良品サブプロキシ間上限 `ortho_margin_normal = 0.85`［係数 `0.25`］）の3項で構成される。

**多クラス分類へ流用する際の致命的制約**:
1. **クラス $c_1 \in \{1..C\}$ のサンプルとクラス $c_2 \neq c_1 \in \{1..C\}$ のプロキシ間には、サンプルレベルの斥力（Negative Push）が一切存在しない**（[README.md L110](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L110)）。（※比較用の [`AsymmetricMultiSimilarityLoss`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/ms_loss.py#L127-L152) には不良サブクラス間の補助反発 `w_aux = 0.5, lambda_aux = 0.15` があるが、そちらも `labels <= 0` を正半球 `>= 0.70`、`labels > 0` を負半球 `<= -0.25` に縛る非対称設計［[ms_loss.py L161-L174](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/ms_loss.py#L161-L174)］である。）
2. **評価器・推論API・保存モデルすべてが2値判定専用である**: 空間評価器（[evaluator.py L297-L303](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/evaluator.py#L297-L303)）も単発推論API [`run_single_image_inference`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L1037-L1104) も $S(x, p_{\text{norm}}) < \tau_{\text{sim}}$ の **OK/NG 2値判定のみ** を行う。さらに最良チェックポイント `best_model.pt`（[train_jobs.py L769-L783](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L769-L783)）は `normal_proxies` のみを保存し **`defect_proxies` を保存していない**。

したがって、「不良Aを不良Bと混同してもNG判定としては正解」である異常検知ではこの非対称設計が有効に機能するが、**クラス $1..C$ 間の相互誤分類がすべてエラーとなる一般の多クラス分類に現行コードをそのまま適用すると、クラス間の識別境界が形成されず精度が劣化する。**

### 1.2. 乖離②：「微小グループの分割」は損失関数単体ではなくオフライン前処理に依存している

* **異常側（`labels > 0`）**: [`CustomizedMultiProxyAnchorLoss`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L336-L337) は分割済みの整数ID `y_def in {1..C}` を外部から受け取る前提であり、損失関数単体では1クラス内の微小グループを発見できない。グループ分割を担っているのはフェーズ1のオフライン処理 [`run_two_stage_clustering`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/two_stage_cluster.py#L119-L316)（良品基準残差標準化＋PCA＋コサイン距離シルエット最大化K-Means）である。
  * **前処理コードの隠れた制約**: ①シルエット探索ループ `range(2, max(3, upper_k + 1))`（[two_stage_cluster.py L229-L242](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/two_stage_cluster.py#L229-L242)）は $k \ge 2$ のみを探索するため、**本来単峰のクラスであっても強制的に2クラスタ以上へ分割し $M_k = 1$ を選べない**。②現行フロー（[README.md L38-L41](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L38-L41)）は **6:2:2 分割の前** に全データ100%でクラスタリングを実行しており、厳密な多クラス評価へ転用する場合は `train`（60%）のみでサブクラスタを学習しないと `val`/`test` の分布情報がリークする。
* **良品側（`labels <= 0`）**: `num_normal_proxies`（UIデフォルト `1`［[2_training_monitor.py L322](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/2_training_monitor.py#L322)］、Python/APIデフォルト `4`［[proxy_anchor_loss.py L47](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L47)］）により $M$ サブセンターを持てるが、初期K-Means分割と `0.10 * subcenter_anchor` のみであり、ミニバッチ内で選ばれなかったサブプロキシを再配置する機構はない。

### 1.3. 乖離③：個体別の不確実性（確率キャリブレーション）を消失させる幾何学的設計である

不感帯のない線形求心力 $1 - S$ は、境界すれすれの曖昧な正常個体（真の事後確率 $0.65$ 相当）も典型的な正常個体（$0.99$ 相当）も区別せず同じ極点 $S \to 1.0$ へ単調収縮させる。この多対1圧縮により特徴空間上で個体レベルの不確実性（Aleatoric Uncertainty）が失われるため、**後段で温度スケーリングや Dirichlet Calibration を当てても個体別の較正済み確率（Brier Score / RLCD水準）は原理的に回復しない**。
また実装上のデッドパラメータとして、[proxy_anchor_loss.py L51-L55](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L51-L55) の `alpha_pos`, `gamma_neg`, `delta_pos`, `delta_neg`, `w_defect` および `forward()` の `gamma_repel`（[L180](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L180)）は内部で一切参照されておらず、マージンも [`CurriculumPenaltyScheduler`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/scheduler.py#L46-L47) の動的値ではなく固定値 `req.repel_delta` が学習・検証の双方（[train_jobs.py L606, L689, L857](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L606)）に渡されている。

### 1.4. 乖離④：周辺パイプラインが「固定構図の2D RGB画像」に強く結合している

他モダリティへ移植する際、以下のモジュールは2D画像テンソル `[B, 3, H, W]` や画像ファイル拡張子をハードコードしているためそのままでは動作しない：
1. [`extract_handcrafted_surface_statistics`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L101-L166)（678次元RGB空間グリッド統計量）および [`MetricProjectionModel.__init__`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L194) の `dummy = torch.zeros(2, 3, 128, 128)` プローブ。
2. [`synthesize_defect_transfer_images`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/boundary_augmentation.py#L32-L103) および 4階テンソル `alphas` を持つ [`synthesize_anomalous_mixup_images`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/boundary_augmentation.py#L135)。
3. [`GeMPooling.forward`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L38-L41)：3階テンソル分岐 `x.ndim == 3` は存在するが、**`x.clamp(min=1e-6)` によりゼロ中心のTransformer隠れ状態や3D点群特徴の負値成分（情報の約半分）をすべて切り捨てる**上、`attention_mask`（パディング除外）に非対応である。
4. [`standardize_features`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L291-L297)：未キャリブレーション時に `batch_size >= 4` ならバッチ統計標準化、`batch_size < 4` なら標準化スキップと挙動が不連続に切り替わる。

---

## 2. 対称「多クラス対多クラス」拡張（Mode B）の数理要件と既存手法に対する優劣

本損失関数の思想を「各クラスが微小サブグループを内包する対称な $K$ クラス分類」へ拡張する場合、現行の非対称型（**Mode A**）とは別に、以下の数理的欠陥を解消した **Mode B（一般化階層サブプロキシ損失）** が必要となる。

### 2.1. Mode B を設計する場合に回避すべき4つの数理的破綻点
1. **Top-$K$ 最大値縮退の回避**: 異クラス侵入度 $S_{i,j}^- = \max_{m \in \{1..M_j\}} \langle z_i, \text{stop\_grad}(\hat{p}_{j,m}) \rangle$ に対し $\max_{j \in \text{Top-}K_{\text{neg}}}$ をとると単調性により Top-1 と恒等になり、複数クラス境界の三重点で勾配がジグザグ振動する。必ず上位 $K_{\text{neg}}$ クラスの **平均** $\frac{1}{K_{\text{neg}}} \sum_{j \in \text{Top-}K_{\text{neg}}(i)}$ または `LogSumExp` を用いる必要がある。
2. **絶対マージン単独による誤分類不感帯の解消**: クラス数 $K$ が増えると球面パッキング制約により $\delta_{\text{abs}} > 0$ となる。このとき $S_{i, y_i}^+ = 0.20 < S_{i, j}^- = 0.28 \le \delta_{\text{abs}} = 0.30$ のサンプルはクラス $j$ に誤分類されているにもかかわらず絶対ヒンジ $\max(0, S_{i,j}^- - \delta_{\text{abs}}) = 0$ となり斥力勾配が消失する。したがって **相対マージンヒンジ** $\max(0, S_{i,j}^- - \text{stop\_grad}(S_{i,y_i}^+) + m_{\text{rel}})$ の併用が不可欠である。
3. **Rankin–Welch 球面パッキング限界と Farkas の補題（空集合の罠）**:
   * 現行の深部マージン $\delta_{\text{deep}} = -0.25$（[proxy_anchor_loss.py L301](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L301)）を残すと、Rankin の単体境界 $N \le \lfloor 1 - 1/\delta \rfloor = 5$ により **任意の次元数 $D$ でも $K \ge 6$ クラスで損失ゼロの配置が数学的に存在しなくなる**。総プロキシ数 $P = \sum_{k=1}^K M_k > D$ では Welch 境界 $\delta_{\text{abs}} > \sqrt{(P - D)/(D(P - 1))}$ が下限となる。
   * さらに $P \ge D + 1$ 個のプロキシが $\mathbb{S}^{D-1}$ 全体に分散して原点を凸包内部に含むと、**Farkas の補題により任意の単位ベクトル $z \in \mathbb{S}^{D-1}$ に対して $\max_{k,m} \langle z, \hat{p}_{k,m} \rangle \ge \eta > 0$ が成立する**。結果として、背景クラスや未知異常（OOD）に対して $\max_{k,m} \langle z, \hat{p}_{k,m} \rangle < \delta_{\text{ood}} \le 0$ を要求すると**実行可能領域が空集合 $\emptyset$ となる**（OOD/背景棄却を行うには、全既知プロキシを半空間コーン $\langle \hat{p}_{k,m}, e_0 \rangle \ge \gamma_0 > 0$ に拘束するか埋め込み次元 $D > P$ に拡大する必要がある）。
4. **`stop_grad` によるプロキシ分離勾配の $O(1/P^2)$ 希釈と AdamW 重み減衰崩壊**:
   異クラスプロキシに `stop_grad` をかけるとプロキシ間の分離は `L_ortho` のみに依存するが、全 $P(P-1)$ ペア単純平均では $P=200$ 時に勾配が約 $1/39{,}800$ に希釈され求心力に負けてプロキシが癒着する（違反ペアのみの平均化が必要）。また `AdamW(weight_decay=1e-4)`（[train_jobs.py L405](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L405)）をプロキシに適用すると、$K > B$ でバッチに現れないクラスのノルムがゼロへ減衰し次回出現時に勾配が爆発するため、プロキシの `weight_decay` は `0.0` 固定とし、推論時はクラスごとのサブプロキシ数 $M_k$ の違いによる最大値期待値バイアス（$\propto \sqrt{2 \ln M_k / D}$）を `val` 上のオフセット $b_k$ で補正する必要がある。

### 2.2. 既存手法（Sub-Center ArcFace / SoftTriple）との YAGNI 比較原則
閉集合の対称多クラス分類において「1クラス内の微小サブグループを分離する」だけであれば、**Sub-Center ArcFace**（Deng et al., ECCV 2020：各クラスに $M$ 個のサブ重みを持ち $\max_m \langle z_i, W_{k,m} \rangle$ に角度マージンを課す約15行のステートレス実装）および **SoftTriple Loss**（Qian et al., ICCV 2019）がすでに確立されている。
したがって、オフラインK-MeansやXBM・プロキシ管理を伴う **Mode B の新規実装はデフォルトでは推奨しない**。Mode B の構築が正当化されるのは、Sub-Center ArcFace をベースラインとして実測した上で **「閉集合の識別精度だけでなく、`detach()` 固定アンカーによる Open-Set 未知異常棄却と極端不均衡耐性が同時に必須である」** と確認された場合に限定すべきである。

---

## 3. 全8領域に対する適用可否判定一覧（損失統合の可否と定量 Go/No-Go 基準）

> [!WARNING]
> 「対象モデルの学習に本損失関数を組み込むこと（損失統合）」と、「対象モデルとは独立した前段フィルタとして現行分類器を並置すること（外部システム連携）」を明確に分離して判定する。不向きな対象への複雑なハイブリッド延命策は **❌ 適用非推奨（No-Go）** と判定する。

| 対象領域 | 損失統合の総合判定 | 現行非対称 (Mode A) | 多クラス拡張 (Mode B) | 技術的結論と必須ベースライン比較（Go/No-Go 条件） |
| :--- | :---: | :---: | :---: | :--- |
| **1. `jev`**<br/>*(TypeSafe AI 判断モデル & OSS `Kev`)* | ❌ **適用非推奨**<br/>*(No-Go)* | ❌ 不可 | ❌ 非推奨 | 本家 `Jev` は重み・学習非公開のため不可。OSS版（`Kev` 等）でも動的選択肢評価と統計的確率較正（RLCD / Brier Score）が中核要件であり、個体別不確実性を $S \to 1.0$ へ潰す本損失は不適。標準の **Pointer CE + Brier Loss** を採用すべき（※固定カテゴリの外部前段ルーター並置は別システム構成）。 |
| **2. `djev`**<br/>*(DiffusionGemma基盤 Jev互換実装)* | ❌ **適用非推奨**<br/>*(No-Go)* | ❌ 不可 | ❌ 非推奨 | `djev` はvLLM上でDiffusionGemmaのピン留めCanvasスロットの語彙対数確率を読む推論機構であり、距離損失への置換で動作不能になる。26B LoRAへの補助損失付加も、スロット直接の **Class-Balanced Focal / Brier Loss** に対しVRAM・実装負荷が見合わない。 |
| **3. `Gemma4`**<br/>*(E2B/E4B/12B/26B/31B)* | ⚠️ **凍結特徴抽出のみ条件付き可**<br/>*(生成学習は ❌ 不可)* | ⚠️ 凍結抽出器上のみ可 | ❌ 非推奨 | 次トークン生成学習への適用はピーク $>1.1\text{ TB}$ のVRAM爆発と生成崩壊を招き不可。**Gemma 4（E2B / 12B Unified）を凍結特徴抽出器として使い、テキスト仕様＋画像のマルチモーダル外観検査（Mode A）を行う用途に限り有効**（Go条件：正常:異常 $\ge 20:1$ かつ `SigLIP` 特徴＋Cosine k-NN に対し Recall@FP1% で $+3\text{pt}$ 以上改善）。 |
| **4. `DiffusionGemma`**<br/>*(26B A4B 離散一様拡散モデル)* | ❌ **適用非推奨**<br/>*(No-Go)* | ❌ 不可 | ❌ 非推奨 | 離散一様拡散の復元交差エントロピー $\mathcal{L}_{\text{diff-SFT}}$ は置換不可。52GB VRAM（BF16）の26B拡散モデルを単ステップ識別エンコーダにする案は $t=0$ での自明コピー縮退・256トークン制限・専用軽量エンコーダ比での極端なコスト劣位により不合理。SD-RL連続コサイン報酬も報酬ハッキングに脆弱。 |
| **5. `自然言語`**<br/>*(テキスト異常検知 / 分類 / 検索)* | ⚠️ **異常・監査検知のみ可**<br/>*(検索・一般分類は ❌)* | ✅ 監査・異常検知で可 | ⚠️ `Sub-Center ArcFace` 敗北時のみ | オープンドメインRAG検索は **InfoNCE + MRL** が優位で不可。画像専用処理と座標スプライスを除去した上で、**コンプライアンス違反・異常ログ検知（Mode A）** に適合（Go条件：正常:異常 $\ge 20:1$ かつ未知違反のOpen-Set検出 AUROC で `ModernBERT + Focal CE` を上回ること）。一般多峰性分類はまず **Sub-Center ArcFace** を優先。 |
| **6. `3D分類`**<br/>*(3D外観検査 / 点群・CAD分類)* | ✅ **3D外観検査で有力**<br/>*(一般3D分類は ⚠️)* | ✅ 3D欠陥検査で有力 | ⚠️ `Sub-Center ArcFace` 敗北時のみ | `GeMPooling` の負値切断バグ修正と局所3Dパッチ特徴化を条件に、**位置決め済み工業部品の3D点群・深度異常検知（Mode A）** に適合（Go条件：正常のみ学習の `PatchCore-3D` / `M3DM` に対し微細欠陥 Recall と推論速度の両面で優位が出ること）。任意 $SO(3)$ 回転下の一般3D分類は既存手法が優位。 |
| **7. `物体検知`**<br/>*(1-Stage / DETR / 2-Stage RoI)* | ❌ **E2E検知は非推奨**<br/>*(切り出しRoI分類のみ ⚠️)* | ❌ 背景凝集で崩壊 | ❌ QFL/VFLに劣位 | 座標回帰は代替不可。一般物体検知の背景（Class 0）凝集による空間崩壊、IoU順位崩壊（$\text{AP}_{75}$ 劣化）、Farkas補題による背景棄却領域の消失により、エンドツーエンド検知器の分類ヘッド置換は **Quality Focal Loss / Varifocal Loss** に劣る。**固定基板等の切り出し済み候補矩形（RoI）に対する後段欠陥判定（実質2Dパッチ検査）のみ可**。 |
| **8. `セグメンテーション`**<br/>*(汎用領域分割 / パッチ異常検知)* | ✅ **工業パッチ異常検知で最有力**<br/>*(汎用閉集合分割は ❌)* | ✅ パッチ展開で最有力 | ❌ Mask2Former標準に劣位 | 全画素計算はOOM、汎用閉集合分割（Mask2Former等）もハンガリアン確率整合とDice/CE直接最適化が優位なため非推奨。一方、**本システムのバックボーン中間特徴マップ上で正常部位を $M$ プロキシ化し異常ヒートマップを出す「パッチ異常セグメンテーション（Mode A 空間展開）」は最小改修で実現可能な最有力応用先**（Go条件：`PaDiM` / `PatchCore-1%` に対し Pixel-AUROC / PRO 同等以上かつ推論レイテンシ $1/5$ 以下）。 |

---

## 4. 領域別の詳細技術監査（破綻メカニズム・改修要件・反証比較）

### 4.1. `jev`（TypeSafe AI System One 判断モデル ＆ OSS `Kev` 等）— 判定：❌ 適用非推奨（No-Go）

1. **対象構造**: 本家 `Jev`（TypeSafe AI）は単一フォワードパスで型付き選択肢（Choice / Boolean / Score）に対する較正済み確率を返す商用クローズドモデル（RLCD学習）であり、OSS実装（`Kev` 等）はデコーダ隠れ状態 $h_{\text{dec}}$ とプロンプト内の動的選択肢トークン隠れ状態 $h_k$ の内積ロジット $z_k = (W_K h_k)^\top (W_Q h_{\text{dec}}) / \sqrt{d_p}$ を温度付き Softmax＋Brier Loss で学習する。
2. **破綻理由（Critique）**:
   * **商用 `Jev` は重み非公開・微調整API非提供**であり、損失関数の差し替え自体が不可能である。
   * **動的スキーマとの不整合**: `CustomizedMultiProxyAnchorLoss` は学習時に固定されたクラスの `nn.Parameter` プロキシしか持てず、リクエストごとに選択肢集合が動的に変化するゼロショット判断に対応できない。
   * **確率キャリブレーションの不可逆な劣化**: 1.3節の通り、線形求心力 $1 - S^+$ は正常クラス内の境界個体も中心個体も $S \to 1.0$ へ圧縮するため、第2段階で温度較正や Dirichlet Calibration を追加しても個体ごとの真の不確実性を復元できない（さらに本リポジトリの20% `val` 分割は希少クラスで1枚しかなく［[README.md L75](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L75)］、すでに [`CurriculumPenaltyScheduler`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/scheduler.py#L61) が使用しているため、同一 `val` 上での較正層学習は過学習とデータリークを引き起こす）。
3. **代替結論**: OSS版 Jev 系モデルの学習には標準の **Multi-Head Pointer Cross-Entropy ＋ Brier Loss** を用いるべきである。もし固定カテゴリのトランザクション振り分けで未知異常（OOD）を事前遮断したい場合は、`jev` の学習を変えるのではなく、独立した軽量埋め込みゲートを `jev` の前段に直列配置するシステム構成をとる。

---

### 4.2. `djev`（DiffusionGemma基盤 マルチモーダルJev互換スタック）— 判定：❌ 適用非推奨（No-Go）

1. **対象構造**: `djev`（`mmastrac/djev`, `Davipar/djev-dev`）は、DiffusionGemma の 256トークン双方向 Canvas 上に回答テンプレートを固定し、vLLM ランタイム経由で候補ラベルスロット（複数BPEトークンに跨る場合を含む）のデノイジング対数確率を読み出して正規化するサービング実装である。
2. **破綻理由（Critique）**:
   * **出力ヘッドの非互換性**: `djev` は語彙空間上の `lm_head` ロジットを直接参照するため、出力を128次元超球面プロキシへ置換すると `djev` の推論ランタイムが成立しなくなる。
   * **26B LoRA への補助損失追加の低効率性**: スロット隠れ状態に分岐ヘッドを付けて本損失を補助加算する案は、スロットロジットに対して直接 **Class-Balanced Focal Loss ＋ Brier Loss** を課す標準 LoRA SFT と比べて、XBM管理やプロキシ初期化の複雑さが増す一方で確率較正に寄与しない。
   * **前段カスケード並置時の注意（外部連携の罠）**: 現行 [`MetricProjectionModel`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L322-L340) を `djev` の前段フィルタに置いて明白なサンプルを短絡処理しようとすると、現行モデルは画像を `128x128` へ縮小しテキスト入力も無視するため、高解像度でのみ視認できる微細欠陥やテキスト条件依存の異常が $S \approx +0.98$（高確信度OK）と誤判定されて `djev` に到達せずすり抜ける（短絡できるのは明白なNG側 $S \ll \tau_{\text{sim}}$ のみである）。

---

### 4.3. `Gemma4`（E2B / E4B / 12B Unified / 26B A4B MoE / 31B）— 判定：❌ 生成学習は不可 ／ ⚠️ 凍結特徴抽出器利用のみ条件付き可

1. **対象構造**: Hybrid Attention（Sliding-Window ＋ 最終層Global Attention）、Dual RoPE、Shared KV Cache を備えたマルチモーダル自己回帰モデル群（エッジ向け `E2B` / `E4B` は Per-Layer Embeddings［PLE］を搭載し、`12B Unified` は外部エンコーダなしで画像パッチ・音声を直接統合）。
2. **生成学習が完全不可である理由**:
   * **ピーク $>1.1\text{ TB}$ のVRAM爆発**: 語彙数 $V = 256{,}000$ に対して [`proxy_anchor_loss.py` L350-L355](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L350-L355) の直交項を計算すると、FP32グラム行列（262 GB）＋対角マスク（65.5 GB）＋連続メモリコピー（262 GB）＋2乗用 `F.relu` 保存テンソル（262 GB）＋逆伝播勾配（262 GB）で **単一ステップあたり 1.1 TB 超のVRAM** を要求し即座に破綻する。またヒンジ損失は次トークン分布のエントロピーを破壊し、文章生成および Thinking Mode を崩壊させる。
3. **凍結特徴抽出器として利用する場合の設計制約とリスク**:
   * **LoRA微調整時のキャッシュ無効化とエポック毎全件推論ボトルネック**: 現行トレーナーが高速に動作するのは `use_cached_backbone=True`（[train_jobs.py L475](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L475)）による事前特徴キャッシュ時のみである。Gemma 4 に LoRA を入れてバックボーンを更新するとキャッシュが無効化され、毎ステップ実バッチ32枚＋Phase 3合成画像16〜32枚（[train_jobs.py L579-L596](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L579-L596)）のフォワード・バックワードが走り、さらに**毎エポック末に全データ（`train+val+test`）を再推論してUMAPを保存する処理**（[train_jobs.py L655](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L655)）で学習時間が桁違いに増大する。したがって **Gemma 4 E2B / 12B Unified は完全凍結（特徴事前抽出）で使うことを原則** とし、Projection Head 内の `BatchNorm1d`（[feature_extractor.py L219](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L219)）はLLM隠れ状態の外れ値チャネル対策として `LayerNorm` へ変更する必要がある。
   * **安全性ガードレール用途での長文希釈リスク**: 長文RAG文脈（32K〜256Kトークン）の末尾 `<eos>` 1トークンのみをプーリングすると、文書中盤に埋め込まれた間接プロンプトインジェクションが正常文脈に希釈されて正常サブプロキシ側へ吸収される（また敵対的脱獄プロンプトは正常分布を模倣するため、未知のゼロデイ攻撃をコサイン閾値だけで完全遮断できるわけではない）。長文入力ではチャンク単位のスライディング窓プーリングに対する最小類似度 $\min_w S(z_w, p_{\text{norm}}) < \tau_{\text{sim}}$ 判定が必要となる。
4. **Go/No-Go 基準**: 「画像単体では良否が決まらず、製品仕様テキストとの組み合わせで良否が決まる外観検査（正常:異常 $\ge 20:1$）」において、凍結 `SigLIP-2` / `Gemma 4 E2B` 特徴上の単純な **Cosine k-NN / Mahalanobis 距離** と比較し、Recall@FP1% が **+3.0 pt 以上向上する場合のみ採用** とする。

---

### 4.4. `DiffusionGemma`（26B A4B 離散一様拡散モデル）— 判定：❌ 適用非推奨（No-Go）

1. **対象構造**: Gemma 4 26B A4B MoE 重みを共有し、プロンプトを処理する **Causal Encoder モード** と、語彙一様ノイズ $q_t(x_i^t \mid x_i^0) = \kappa_t \delta(x_i^t, x_i^0) + (1-\kappa_t)/V$ から256トークン **Canvas** を反復復元する **Bidirectional Denoiser モード** を切り替える離散拡散言語モデル（主損失は平坦な復元交差エントロピー $\mathcal{L}_{\text{diff-SFT}}$、第2段階は SD-RL）。
2. **破綻理由（Critique）**:
   * **拡散デノイジング主損失への置換不可**: 反復サンプリング時のトークン再ランダム化・早期停止制御は語彙確率分布 $p_\theta(x_i^0 \mid x_t)$ のエントロピーに依存しており、連続空間のコサインヒンジ損失では離散トークン復元が成立しない。
   * **単ステップ双方向エンコーダ転用案の欠陥**: Denoiser モードは破壊入力（$t > 0$）からの復元用に学習されているため無ノイズ $t=0$ 入力では自明な恒等コピーへ縮退しやすく、256トークンを超える入力は結局 Causal KV キャッシュ側へ溢れる。さらに [`GeMPooling`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L40-L41) の `clamp(min=1e-6)` とマスク非対応によりそのままでは表現が破壊される上、BF16で約52GBのVRAMを要する26Bモデルを純粋な分類エンコーダとして使うことは **ModernBERT（0.15B〜0.4B）** 等の専用双方向エンコーダに対して計算効率で大幅に劣る。
   * **SD-RL 報酬モデル転用案の脆弱性**: 連続コサイン類似度 $S(z(y), p_{\text{norm}})$ を強化学習（SD-RL）の報酬にすると、RL方策が「構文的には破綻しているが埋め込み空間上で正常プロキシの正コーンに落ちる敵対的トークン列」を探索する **報酬ハッキング（Goodhart's Law）** を起こすため、構文解析器・単体テスト等の決定論的検証器（RLVR）に劣る。

---

### 4.5. `自然言語`（異常・コンプライアンス検知 ／ 多峰性インテント分類 ／ 検索）— 判定：⚠️ 異常・監査検知（Mode A）のみ可

1. **不適領域の切り捨て**:
   * **オープンドメイン検索・RAG埋め込み（❌ 不可）**: 数百万パッセージの意味空間は固定数のプロキシ `nn.Parameter` では表現できず、**In-Batch InfoNCE ＋ Hard Negatives ＋ MRL** が標準である。
   * **一般の均衡多クラス分類（❌ 原則不要）**: `ModernBERT` / `DeBERTa-v3` の **Cross-Entropy** または **Sub-Center ArcFace** で十分であり、複雑な2段階クラスタリング＋XBMを導入する理由がない。
2. **適合領域（Mode A）：社内コンプライアンス違反・機密漏洩・システム異常ログ検知**
   * 定型業務テキストや正常システムログ（部署・テンプレート別に $M$ 個のサブセンターを形成）に対し、少数かつ多様な違反・障害ログ（$C$ クラスタ＋未知パターン）を検出する問題設定では Mode A が機能する。ただし工業製品画像とは異なり自然言語は「正常トピック自体の経時ドリフト（新製品名・新業務用語の出現）」が激しいため、固定構図画像と同等の境界安定性を仮定してはならず、定期的な正常プロキシの再キャリブレーション運用が前提となる。
3. **NLP移植時に必須となる6つのコード修正**:
   1. 2D画像専用処理（[`extract_handcrafted_surface_statistics`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L101-L166)、[`synthesize_defect_transfer_images`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/boundary_augmentation.py#L32-L103)、[`synthesize_anomalous_mixup_images`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/boundary_augmentation.py#L107-L143)、`2x3x128x128` プローブ、画像拡張子チェック）の完全除去。
   2. 特徴空間合成 [`synthesize_boundary_gap_features`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/boundary_augmentation.py#L187-L196) のうち、**上位30%次元の座標スプライス `h_transfer` の無効化**（678次元空間グリッドと異なり、密な言語埋め込みの個別座標軸に局所意味はないためスプライスは多様体外ノイズを生む。測地線 Mixup `h_mixup` のみ使用する）。
   3. 異方性補正の明確化：現行 [`calibrate_feature_stats`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L261-L284) は対角Zスコア標準化であり、言語埋め込みの非対角共分散異方性を除去するには必要に応じて PCA/SVD Whitening 層へ置き換える（併せて [`standardize_features` L291](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L291) の未較正ミニバッチ分岐を廃止する）。
   4. **6:2:2 分割後の `train`（60%）データ限定でのサブクラスタリング徹底**、および少数クラス（$n_{\text{train}, c} < 8 M_c$）や低シルエットスコア時に分割を行わず $M_c = 1$ に留める下限閾値ガードの追加。
   5. `labels < 0`（パディング／対象外サンプル）を損失計算前に除外するマスクの追加。
   6. [`NormalAnomalyBalancedSampler`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/nab_sampler.py#L41-L190) と [`CustomizedMultiProxyAnchorLoss`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L14-L387) の接続。
4. **Go/No-Go 基準**: 正常:異常比率 $\ge 20:1$ かつ未知違反タイプの検出が必要なテキスト監査データセットにおいて、`ModernBERT + Class-Balanced Focal Loss` および `One-Class Mahalanobis` に対し、未知違反に対する Open-Set Recall@FP1% が **+3.0 pt 以上改善する場合のみ採用** とする。

---

### 4.6. `3D分類`（3D点群・深度マップ外観検査 ／ 3D形状・CAD分類）— 判定：✅ 3D外観検査で有力 ／ ⚠️ 一般3D分類は既存手法優先

1. **コード流用時の重大な技術的注意点（3点）**:
   * **① `GeMPooling` の負値切断バグ**: 前述の通り [`GeMPooling.forward`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L40-L41) の `x.clamp(min=1e-6)` は、`PointTransformerV3` や `PointNet++`（LayerNorm系）が出力する符号付き特徴の負値をすべて消去し、さらに可変点数点群の無効点マスクにも対応していない。**マスク付き Attention Pooling または符号保持型プーリングへの差し替えが必須**である。
   * **② 大規模点群における微小欠陥の空間希釈**: レーザープロファイラ等が得る $N = 10^5$ 点の点群全体をいきなり1ベクトルへグローバルプーリングすると、50点程度の微小打痕（全体の $0.05\%$）は $p=3.0$ でも背景点群に希釈される。局所3Dパッチ／ボクセル特徴を先に集約し、上位異常度パッチ群を射影する設計（PatchCore-3D / M3DM 相当の局所粒度）が必要となる。
   * **③ $SO(3)$ 回転自由度の被覆限界**: 3次元回転群 $SO(3)$ は3自由度多様体であり、角度許容誤差 $30^\circ$ で球面全体を覆うには $60 \sim 550$ 個の代表姿勢が必要となる。したがって $M = 6..12$ 個のサブプロキシで吸収できるのは **1軸ターンテーブル回転 $SO(2)$（または治具による位置決め＋数パターンの表裏反転）まで** であり、任意の3次元姿勢をとる点群に対しては **ICP/PCA による事前姿勢正規化（Canonicalization）** または **$SO(3)$ 同変バックボーン（Vector Neurons 等）** が必須である。
2. **有望な適用先（Mode A）：3Dスキャナによる工業部品形状検査（MVTec 3D-AD / Real3D-AD 領域）**
   2D照明では判別困難な「打痕深さ・反り・鋳造バリ・はんだ体積異常」を扱う3D外観検査は、治具で概ね位置決めされており（$SO(2)$ 以下）、かつ極端な良品過多（100:2問題）と多様な不良モードを持つため、2D表面統計量を **3D局所曲率・法線分散・基準CAD面からのZ残差統計量** に置換することで本システムの設計資産を直接活かせる。
3. **Go/No-Go 基準**:
   * **3D外観検査（Mode A）**: 正常点群のみで動作する学習不要ベースライン（`PatchCore-3D` / `FPFH + Mahalanobis`）と比較し、少数不良サンプル投入時に微細欠陥 Recall@FP1% が **+3.0 pt 以上向上**、かつプロキシ圧縮により推論レイテンシが近傍探索比で **1/5 以下** になること。
   * **対称3D形状・CAD分類**: まず `PointTransformerV3 + Sub-Center ArcFace (M=4)` を評価し、Open-Set未知部品棄却要件で不足が生じた場合のみ Mode B を検討する。

---

### 4.7. `物体検知`（1-Stage密検知 / DETR / 2-Stage RoI検知）— 判定：❌ E2E検知は非推奨（No-Go） ／ ⚠️ 切り出しRoI後段検査のみ可

1. **エンドツーエンド物体検知器（YOLO / Faster R-CNN / DETR）への組込みを非推奨とする4つの理由**:
   1. **座標回帰の非互換性**: L2正規化超球面埋め込みはノルムを持たず、ボックス座標回帰（GIoU / Smooth L1 / DFL）を一切代替できない。
   2. **背景（Class 0）凝集による空間特徴崩壊**: 一般物体検知の `Background` は空・道路・壁・部分領域など最大エントロピーの雑多集合であり、`loss_centripetal` で1点へ凝集させると特徴マップ全体が崩壊する。
   3. **Farkas の補題による背景棄却領域の消失**: 仮に極性を反転して「前景 $K$ クラスをそれぞれ $M_k$ サブプロキシへ凝集、背景を全前景プロキシから反発」させても、総前景プロキシ数 $P = \sum_k M_k \ge D + 1$ が超球面全体に広がるとすべての単位ベクトルがいずれかの前景プロキシと正の類似度を持つため、背景を全前景から $\delta_{\text{bg}} \le 0$ へ押し出す解が存在しなくなる（全前景プロキシを特定の半空間コーン内に閉じ込める追加制約が別途必要になる）。
   4. **IoUランキング崩壊とDETR重複クエリ問題**: 物体検知の分類スコアは NMS や DETR のハンガリアンマッチングにおいて位置精度（IoU）の順序を反映する必要がある。線形求心力 $1 - S^+$ は $\text{IoU}=0.55$ の不正確なボックスも $\text{IoU}=0.95$ のボックスも同じ $S^+ \to 1.0$ へ引き寄せるため NMS で低品質ボックスが生き残り $\text{AP}_{75}$ が低下する（別個に IoUness / Objectness ヘッドを増設するなら、最初から **Quality Focal Loss［QFL］/ Varifocal Loss［VFL］** を使う方が単一損失で完結し優れている）。
2. **唯一合理的な活用形態（外部2段構成）**:
   検知器内部の損失は書き換えず、基板検査等で位置決めテンプレートやクラス非依存提案器が切り出した **個別部品パッチ画像（RoI）に対する後段の良否・不良モード判定器** として現行システム（Mode A）をそのまま適用する形態のみが低リスクで成立する。

---

### 4.8. `セグメンテーション`（汎用領域分割 ／ 工業パッチ異常セグメンテーション）— 判定：❌ 汎用閉集合分割は非推奨 ／ ✅ 工業パッチ異常検知は最有力拡張

1. **汎用セマンティック／パノプティックセグメンテーション（Mask2Former等）を非推奨とする理由**:
   * 全画素直接計算は $512 \times 512 \times B=16$ で出力テンソル単体約 $2.15\text{ GB}$＋XBM辞書破綻によりOOMを起こす。
   * Mask2Former の 100〜200 クエリへ適用する案も、二部マッチング（Hungarian Matching）と最終画素推論 $\sum_i P(c \mid q_i) M_i(u,v)$ が較正済みクラス確率と高エントロピーな `no-object`（$\varnothing$）クラスの扱いを前提としており、標準の **Cross-Entropy ＋ Binary Mask BCE ＋ Dice Loss** に対して mIoU / PQ 上の優位性がない（さらに既知クラス境界ではCNN特徴の線形補間により常に低類似度の「偽の未知異常ハロー」が発生する）。
2. **最有力拡張（Mode A 空間展開）：工業外観検査における「多部品サブプロキシ・パッチ異常セグメンテーション」**
   [README.md L29](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L29) では現行システムの非目標とされているが、1枚の製品画像内部には「金属面」「樹脂コネクタ」「銅ピン」「刻印」「背景治具」といった**正常クラス内の明確な構造サブグループ**が存在するため、`num_normal_proxies`（$M$）によるサブグループ表現と `p_norm.detach()` ヒンジ斥力をパッチ特徴マップへ展開する価値が高い。
   **実コードに基づく必須改修仕様（4点）**:
   1. **マルチスケール中間特徴マップの抽出**: 現行 [`_load_backbone`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py#L236-L243) の `timm.create_model(..., num_classes=0, global_pool="")` は最終ステージ（stride 32：入力 `128x128` 時はわずか $4 \times 4$、`256x256` でも $8 \times 8$）しか返さない。`features_only=True` で Stage 2（stride 8）＋ Stage 3（stride 16）の特徴マップをアップサンプル結合し、非線形2層 `1x1 Conv -> GroupNorm -> GELU -> 1x1 Conv(128) -> L2Normalize` ヘッドを通す必要がある。
   2. **微小正常部位を取りこぼさない Greedy Coreset（k-Center）初期化**: パッチ数に対して単純な K-Means（[proxy_anchor_loss.py L119-L129](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L119-L129)）を行うと、面積の90%を占める背景や平滑面にプロキシが集中し、面積 $0.5\%$ 未満の正常な端子や刻印が外れ値（偽陽性）になる。正常パッチ群から最大最小距離基準（Greedy Coreset）で $M = 16 \sim 32$ 個の正常部位プロキシを初期化する必要がある。
   3. **合成欠陥マスクの返却化と遷移リングの Ignore 処理**: 現行 [`synthesize_defect_transfer_images`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/boundary_augmentation.py#L85-L103) はガウシアン窓 $G(u,v)$ を内部で破棄して返却しておらず、かつデフォルト `sigma_ratio = 0.16` は画像の約40%の面積に裾野が広がる。低不透明度の裾野パッチ（受容野の大半が正常画素）に `loss_repel` を課すと正常境界が破壊されるため、①マスク $G(u,v)$ を返却するよう変更し、②高不透明度コア（$G(u,v) \ge 0.60$）のみを不良反発対象とし、③境界遷移帯（$0.05 < G(u,v) < 0.60$）は損失計算から除外（Ignore）する設計が必須である。
   4. **XBM のパッチ非適用**: パッチ埋め込みを [`CrossBatchMemoryBank`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/memory_bank.py#L14-L167) に入れるとメモリ肥大化を招くためオフとし、常駐する $M$ 個の正常プロキシとの内積のみで学習・推論を行う。
3. **Go/No-Go 基準**: 同一バックボーン特徴を用いた学習不要の正常のみ手法（**`PaDiM`** および **`PatchCore-1%`**）をベースラインとし、少数の実不良画像を加えた際に **Pixel-AUROC / AUPRO で同等以上（または微細欠陥検出率 +2.0 pt 以上）、かつプロキシ集約により推論速度が PatchCore 近傍探索比で 5倍以上高速** であることを確認できた場合のみ本採用とする。

---

## 5. 設計上の主要トレードオフと運用コスト一覧

| 設計論点 | 発生する技術的トレードオフ | 採るべき設計選択 | 受け入れる運用コスト・制約 |
| :--- | :--- | :--- | :--- |
| **1. クラス内多峰性 vs. 異クラス分離（対称多クラス課題）** | 現行 Mode A はクラス $1..C$ 間のサンプル斥力がゼロであり対称分類では誤分類が増える一方、独自に Mode B を作ると球面パッキング制約・プロキシ管理コストが急増する。 | 閉集合の多峰性分類にはまずステートレスな **Sub-Center ArcFace** を適用し、Open-Set未知異常棄却が不可欠な場合のみ相対マージン＋半空間コーン制約付き Mode B を実装する。 | Mode B 採用時は、埋め込み次元 $D > \sum_k M_k$ の確保、`weight_decay=0` 分離、および `train` 分割限定クラスタリング＋$M_k$ バイアス較正のパイプライン保守が必要となる。 |
| **2. 幾何学的マージン凝集 vs. 確率較正（`jev` / `djev` 等）** | 線形求心力 $1 - S^+$ は正常クラスタ分散を最小化する引き換えに、個体ごとの事後確率の濃淡（不確実性）を消失させ、事後較正でも個体別 Brier 精度が戻らない。 | 較正済み確率出力を主目的とするモデル（`jev` / `djev` / Mask2Former 等）への本損失の組込みを見送り（**No-Go**）、固有スコアリング規則（CE + Brier Loss）と使い分ける。 | Open-Set棄却と確率出力の両方が必要なシステムでは、「OOD遮断用の距離学習ゲート」と「確率出力モデル」を疎結合な2コンポーネントとして運用する（前段は高解像度またはNG短絡限定）。 |
| **3. 空間局在化解像度 vs. メモリ・検索計算量（パッチ異常検知）** | 画素/パッチ単位でメモリバンク（PatchCore / XBM）を持つとデータ量に比例してメモリと推論時間が増大する。 | XBMを廃止し、正常製品の多様な部位を **Greedy Coreset 初期化した $M=16..32$ 個の学習可能正常プロキシ** に圧縮して $O(M)$ 内積のみで異常マップを算出する。 | 合成欠陥の遷移帯マスク除外（Ignore処理）とマルチスケール特徴結合のハイパーパラメータ調整が必要となり、また実不良データが0枚の完全教師なし環境では `PaDiM`/`PatchCore` に譲る。 |

---

## 6. 結論とゲート付き実行ロードマップ

### 6.1. 最終結論
* **採用推奨（Go to PoC）**:
  1. **工業外観検査のパッチ異常セグメンテーション（4.8節・経路2）** — 本リポジトリからの構造的連続性が最も高く、画像単位判定から欠陥位置ヒートマップへの拡張を最小工数で達成できる。
  2. **3D点群・深度マップを用いた位置決め部品の3D外観検査（4.6節）** — `GeMPooling` の負値切断修正と局所3D記述子への差し替えにより、100:2不均衡・Open-Set異常検知の強みがそのまま活きる。
  3. **自然言語監査・異常ログ検知、および凍結 Gemma 4（E2B / 12B Unified）特徴を用いた仕様条件付き外観検査（4.5節・4.3節）** — 正常多様体（複数テンプレート）対 多様な希少違反の非対称構造（Mode A）に合致する。
* **条件付き保留（既存ベースライン敗北時のみ検討）**:
  * 自然言語・3Dの対称多クラス分類への拡張（Mode B）は、まず **Sub-Center ArcFace** を評価し、Open-Set未知クラス棄却性能で要件を満たせない場合のみ着手する。
* **明確な不採用（No-Go）**:
  * **`jev`、`djev`、`DiffusionGemma`、`Gemma4` の生成学習、エンドツーエンド `物体検知`、および汎用閉集合 `セグメンテーション`** への損失関数組込みは、アーキテクチャ非互換・確率較正破壊・計算量爆発・既存損失（RLCD/Brier, QFL/VFL, Mask2Former Dice+CE）に対する明確な劣位のため実施しない。

### 6.2. 段階的検証ロードマップ（定量ゲート付き）

1. **Gate 0：現行コードベースの負債解消と不良サブクラス分離度の現状計測（着手前必須・工数小）**
   * **改修内容**:
     1. [`proxy_anchor_loss.py` L213](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L213) / [`ms_loss.py` L79](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/ms_loss.py#L79) に `labels < 0`（Ignore）除外マスクを追加し、[`proxy_anchor_loss.py` L51-L55, L180](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L51-L55) の未使用引数を整理する。
     2. [`CurriculumPenaltyScheduler`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/scheduler.py#L46-L47) のマージン初期値・最終値を `req.repel_delta` 基準（例：`margin_initial = min(0.80, req.repel_delta + 0.15)`, `margin_final = req.repel_delta`）に連動させた上で、学習・検証の双方（[train_jobs.py L606, L689](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L606)）へ一貫して渡すか、固定マージン仕様として一本化する。
     3. [`train_jobs.py` L769-L783](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L769-L783) の `best_model.pt` に `defect_proxies` を保存し、[`evaluator.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/evaluator.py#L297-L303) に不良サブクラス間の `Macro-F1` 計測を追加する。また [`two_stage_cluster.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/two_stage_cluster.py#L119) を 6:2:2 分割後の `train` のみでフィットするモードとシルエット下限閾値（単峰時 $k^*=1$ 維持）を追加する。
   * **完了判定**: 既存2D検査データセットの `test`（20%）分割において、2値 Recall/Precision を劣化させずに不良サブクラス間の分離指標を定量確認できること。

2. **Gate 1：パッチ異常セグメンテーション PoC（最優先応用・工数中）**
   * **改修内容**: `timm` の `features_only=True` Stage 2+3 結合特徴マップ＋Greedy Coreset 正常部位プロキシ（$M=16..32$）＋合成マスク高不透明度コア反発（$G \ge 0.60$、遷移帯 Ignore）を実装する。
   * **Go/No-Go 判定**: 同一データセット上の `PaDiM` / `PatchCore-1%` に対し、Pixel-AUROC / AUPRO 同等以上かつ推論レイテンシ $1/5$ 以下を達成できなければ中止する。

3. **Gate 2：非画像1モダリティ（3D外観検査 または NLP異常監査のどちらか1つに絞って実施・工数中）**
   * **改修内容**: 複数モダリティへの同時展開を禁じ、実データセットが存在する1領域のみを選択。画像専用前処理を分離し、マスク対応の符号保持プーリング＋`LayerNorm` 射影ヘッド上で Mode A を検証する。
   * **Go/No-Go 判定**: 対象領域の単純ベースライン（3Dなら `PatchCore-3D`、NLPなら `ModernBERT + Focal CE`）に対し、`test` 分割での Recall@FP1% が **+3.0 pt 以上** 上回る場合のみ本番統合へ進む（対称多クラス分類が必要になった場合は、Mode B を書く前に必ず `Sub-Center ArcFace` を先に計測する）。
