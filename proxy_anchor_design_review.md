# 設計レビュー・数理検証レポート：Multi-Proxy Anchor Loss ＆ NAB-Sampler ＆ 境界生成拡張

対象仕様書：ユーザー提案「外観検査自動化のための深層距離学習（Metric Learning）設計仕様書」（[note.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/note.md) セクション15参照）

---

## 1. 総評（理論的妥当性と先行研究との整合性）

提案された設計仕様は、従来のペアベース手法（MS-Loss / Triplet Loss）が「極端不均衡（100:2）かつ多モード不良」の外観検査データにおいて直面する2つの構造的欠陥——**①不良サンプル間の不要な斥力による勾配分散**、および **②バッチ内の少数不良枯渇による正例ペア不成立**——を根本から解消するアーキテクチャである。

具体的には、以下の先行研究の数理的知見と合致している：
1. **Proxy-Anchor Loss** (Kim et al., *CVPR 2020*): サンプル対サンプル（$O(B^2)$）ではなく「プロキシ対サンプル（$O(B \cdot |P|)$）」で勾配を計算するため、バッチ内に特定不良モードが少数しかなくても学習が破綻せず、かつサンプル間の相対硬さ（Hardness）に応じた動的勾配重み付けを維持できる。
2. **SoftTriple / Sub-Center 表現** (Qian et al., *ICCV 2019*): 単一クラスに複数の代表プロキシ（Multi-Proxy）を持たせることで、照明ムラやロット変動による良品クラス内部の多峰性・広がりを無理なく吸収できる。
3. **Class-Balanced Loss** (Cui et al., *CVPR 2019*): 有効サンプル数（Effective Number of Samples, $n_{\text{eff}} = \max(1, n_c)$）に基づく重みを全不良クラス $C$ の平均が $1.0$ になるよう正規化した
   $$\tilde{W}_c = \frac{1-\beta}{1-\beta^{\max(1, n_c)}}, \quad W_c = \frac{\tilde{W}_c}{\frac{1}{C}\sum_{k=1}^{C} \tilde{W}_k} \quad (\beta = 0.99)$$
   により、極少数の不良モードの勾配消失を防ぐ（実装：[`compute_class_balanced_weights`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/nab_sampler.py#L8-L38)）。

一方で、**提案仕様をそのまま素朴に実装すると学習崩壊（Representation Collapse）や過検出（False Positive 急増）を引き起こす数理的トラップが3点存在する**。以下にその批判的検証（Stress-Test）と、実装コードに組み込まれた確定版の数理定式化を示す。

---

## 2. 重大な数理的トラップ（Critique & Stress-Test）と解決策

### 🔴 トラップ1：「不良同士の斥力を完全にゼロ」にすると全不良プロキシが1点に縮退する（Proxy Collapse）

* **問題点（反証）**:
  ユーザー提案仕様の損失設計（4.1節）では「不良同士の不要な斥力計算を完全に排除する」と定義されている。確かに**サンプル（画像特徴）間**の不良A vs 不良Bの斥力は排除すべきである。しかし、各不良クラスの代表ベクトルである**不良プロキシ $\{p_{\text{def}}^{(1)}, \dots, p_{\text{def}}^{(C)}\}$ 同士**の分離制約まで完全にゼロにすると、すべての不良プロキシにとっての損失最小化点は「良品プロキシ群の重心と正反対の単一極（Antipodal Point：$p^* = -\mu_{\text{norm}}$）」となり、**全不良クラスのプロキシが再び1点に重なって潰れる（1次元2値分類への退化）**。
* **解決策（プロキシパラメータ間のみの軽量直交ヒンジ正則化 $L_{\text{proxy-ortho}}$）**:
  * **画像サンプルに対する「不良A vs 不良B」の斥力は提案通り完全に排除**し、サンプルの勾配は 100%「良品と不良の分離」と「自クラスプロキシへの凝集」に集中させる。
  * その上で、学習可能なパラメータである**不良プロキシベクトル同士（$p_{\text{def}}^{(a)}, p_{\text{def}}^{(b)}$）および良品サブセンタープロキシ同士（$p_{\text{norm}}^{(m_1)}, p_{\text{norm}}^{(m_2)}$）にのみ**、それぞれコサイン類似度が閾値 $\tau_{\text{ortho}}^{\text{def}} = 0.20$、$\tau_{\text{ortho}}^{\text{norm}} = 0.85$ を超えないようにする直交ヒンジ正則化項を課す（実装：[`CustomizedMultiProxyAnchorLoss.forward`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L348-L370)）：
    $$L_{\text{proxy-ortho}} = \frac{1}{\max(1, C(C-1))} \sum_{a \neq b} \max\!\left(0, \, \langle p_{\text{def}}^{(a)}, p_{\text{def}}^{(b)} \rangle - 0.20\right)^2 + \frac{0.25}{\max(1, M(M-1))} \sum_{m_1 \neq m_2} \max\!\left(0, \, \langle p_{\text{norm}}^{(m_1)}, p_{\text{norm}}^{(m_2)} \rangle - 0.85\right)^2$$
  * これにより、画像特徴抽出器の勾配パワーを一切浪費することなく、128次元球面上で各不良モードのプロキシおよび良品サブセンターを異なる方向へ展開させた状態を維持できる。

---

### 🔴 トラップ2：Anomalous Mixup ($\alpha \in [0.80, 0.90]$) とマスクなし Defect Transfer の副作用

* **問題点（反証）**:
  1. **不良プロキシの良品領域への引きずり込み**:
     良品80〜90%＋不良10〜20%で合成したサンプル $x_{\text{mix}}$ を通常の不良サンプルと同等に扱って「不良プロキシ $p_{\text{def}}^{(c)}$ への引き寄せ（Positive項）」に入れてしまうと、**不良プロキシ自体が良品プロキシのすぐ隣（80〜90%良品側）へと引きずり込まれ、空間全体が崩壊する**。
  2. **ピクセルマスクが存在しない環境での Defect Transfer**:
     弱教師あり（フォルダ単位アップロード）環境では「どこに傷があるか」のピクセルマスクが存在しない。ランダムな矩形を切り抜いて良品に貼ると、不良画像の「正常な背景部分」を切り抜いて良品に貼り、「これは不良である」という誤ラベルを生成してしまう。
* **解決策（片方向境界反発ルール ＆ 残差サリエンシー転写・特徴スプライシング）**:
  1. **合成サンプルの片方向反発ルール（One-Way Boundary Repulsion）**:
     Anomalous Mixup（$\alpha \in [0.80, 0.90]$）および Defect Transfer で生成された合成ハードサンプル集合 $\mathcal{B}_{\text{synth}}$ は、**「良品プロキシ $\{p_{\text{norm}}^{(m)}\}$ および XBM 境界良品から押し出す Negative 反発項（$L_{\text{repel}}$ の Direction A / C）」にのみ使用し、「不良プロキシ $p_{\text{def}}^{(c)}$ を引き寄せる Positive 項（$L_{\text{def-pos}}$）」からは厳密に除外する**（実装：[`proxy_anchor_loss.py` L222-L268](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L222-L268)）。これにより、不良プロキシの位置を負の半球の遠方に保ったまま、良品プロキシ周囲の識別境界だけを内側に鋭く削り込むことができる。
  2. **ピクセル空間＋特徴空間のハイブリッド境界合成（[`boundary_augmentation.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/boundary_augmentation.py)）**:
     * **ピクセル空間 Defect Transfer（[`synthesize_defect_transfer_images`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/boundary_augmentation.py#L30-L95)）**: 不良画像 $I_{\text{bad}}$ に対し、$16\times 16$ 平均プーリング $L_1$ 距離で最近傍の参照良品画像 $I_{\text{ref}}$ を検索し、平滑化残差マップ $R = \text{AvgPool}_{11\times 11}(\|I_{\text{bad}} - I_{\text{ref}}\|_1)$ が最大となる欠陥ピーク座標 $(u^*, v^*)$ を特定する。その座標を中心に 2D ガウシアンフェザリング窓 $G(u,v)$（`sigma_ratio=0.16, peak_opacity=0.82`）を生成し、別の新鮮な良品画像 $I_{\text{good}}$ へ $(1-G)\odot I_{\text{good}} + G\odot I_{\text{bad}}$ としてインプレース転写する。
     * **ピクセル空間 Anomalous Mixup（[`synthesize_anomalous_mixup_images`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/boundary_augmentation.py#L99-L133)）**: $\alpha \sim \text{Uniform}(0.80, 0.90)$ で $I_{\text{mix}} = \alpha I_{\text{good}} + (1-\alpha) I_{\text{bad}}$ を合成する。
     * **特徴空間での動的残差スプライシング＋Mixup（[`synthesize_boundary_gap_features`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/boundary_augmentation.py#L137-L195)）**: バックボーン凍結高速学習時（`use_cached_backbone=True`）には、学習開始前に最大48枚の良品・48枚の不良から事前生成したピクセル合成特徴バンク（[`_extract_pixel_boundary_augmentation_bank`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L232-L284)、毎ステップ最大8件抽出）に加え、毎ステップのミニバッチ内で「不良特徴と最近傍良品特徴の差 $|h_{\text{bad}} - h_{\text{ref}}|$ が大きい上位30%次元（`transfer_ratio=0.30`）を重み `0.85` で別良品特徴へ移植する特徴空間 Defect Transfer」と「$\alpha \in [0.80, 0.90]$ の特徴空間 Anomalous Mixup」をオンザフライ生成する。

---

### 🟡 トラップ3：良品 Multi-Proxy（$M=4$）と不良プロキシの初期化・推論スコアの一貫性

* **問題点**:
  良品に $M$ 個のプロキシ $\{p_{\text{norm}}^{(1)}, \dots, p_{\text{norm}}^{(M)}\}$ を割り当てても、全良品サンプルを全 $M$ 個のプロキシへ均等に引き寄せると、$M$ 個のプロキシがすべて同じ1点に収束してしまう。
* **解決策（Sub-Center 最大類似度割当・負半球初期化・推論ルールの統一）**:
  1. **初期プロキシ配置（[`initialize_proxies_from_embeddings`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L93-L146)）**:
     * 学習開始時（キャリブレーション直後）に良品特徴群に対して K-Means（$K=M$）を実行し、そのクラスタ重心を初期良品プロキシ $\{p_{\text{norm}}^{(1)}, \dots, p_{\text{norm}}^{(M)}\}$ として初期化する。
     * 各不良クラス $c \in \{1..C\}$ のプロキシ $p_{\text{def}}^{(c)}$ はそのクラスの初期平均ベクトルから初期化し、良品プロキシ平均方向 $\bar{p}_{\text{norm}}$ との内積が $> -0.10$ の場合は $\text{Normalize}(c_t - (\langle c_t, \bar{p}_{\text{norm}}\rangle + 0.35)\bar{p}_{\text{norm}})$ によって最初から負の半球（良品と反対側の半球）へ射影初期化する。
     * また、`AdamW` オプティマイザではプロキシパラメータの学習率をバックボーン／Projection Head の **`2.5倍`**（`lr * 2.5`、[`train_jobs.py` L354-L357](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L354-L357)）に設定し、空間形成を高速化する。
  2. **良品サンプルの Sub-Center 引力計算**: 各良品サンプル $z_j$ は、$M$ 個の良品プロキシのうち**最もコサイン類似度が高い担当良品プロキシ $m^*(j) = \arg\max_{m \in \{1..M\}} \langle z_j, p_{\text{norm}}^{(m)} \rangle$ のみ**へ引き寄せる。
  3. **推論・Step 6 空間評価判定ルール（[`evaluator.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/evaluator.py#L168-L173)）**:
     推論および Step 6 空間評価において、サンプル $x$ の良品類似度スコアを
     $$s(x) = \max_{m \in \{1..M\}} \langle z(x), p_{\text{norm}}^{(m)} \rangle \in [-1, 1]$$
     と定義し、$s(x) < \tau_{\text{sim}}$ を不良（NG）と判定する。

---

## 3. 確定版の数理定式化（コード完全準拠仕様）

### 3.1. Normal-Anomaly Balanced Sampler (`NAB-Sampler`：[`nab_sampler.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/nab_sampler.py))

極端なクラス不均衡環境（例：良品225枚 vs 不良数枚）においてミニバッチサイズ $B$ を良品50%・不良50%（`normal_ratio = 0.50`）で構成する際、**少数派である不良品データの枯渇と重複過学習を防ぐため**、以下のアルゴリズムで1エポックのバッチ列を生成する：

1. **良品データ2周網羅に基づくエポックステップ数の決定（[`nab_sampler.py` L126-L128](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/nab_sampler.py#L126-L128)）**:
   エポックあたりのステップ数 `num_batches` は少数不良ではなく**全学習良品枚数 $|D_{\text{norm}}|$ を基準**に決定する：
   $$B_{\text{norm}} = \max(2, \lfloor B \cdot r_{\text{norm}} \rfloor), \quad B_{\text{anom}} = \max(2, B - B_{\text{norm}})$$
   $$\text{num\_batches} = \max\!\left(\text{min\_steps},\; 2 \times \left\lceil \frac{|D_{\text{norm}}|}{B_{\text{norm}}} \right\rceil\right)$$
   （デフォルト `min_steps=12`、[`train_jobs.py` L452](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L452) では `min_steps=14` を指定）。
2. **良品スロット（$B_{\text{norm}}$ 枚）の巡回抽出＋XBM ハード良品35%優先注入（[L135-L154](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/nab_sampler.py#L135-L154)）**:
   * 前ステップまでに XBM が特定した境界ハード良品インデックス `valid_hard_normals` が存在する場合、良品枠の最大 **35%**（$n_{\text{hard}} = \min(|\text{valid\_hard\_normals}|, \max(1, \lfloor 0.35 B_{\text{norm}} \rfloor))$ 枚）を非復元抽出（`replace=False`）で優先配置する。
   * 残りの $B_{\text{norm}} - n_{\text{hard}}$ 枚（65%以上）は、シャッフルされた全良品リスト `norm_pool` から巡回カーソル `norm_cursor` で順次取り出し（同一バッチ内で既に選ばれたハード良品との重複を回避）、`norm_pool` を1周するたびに再シャッフルする。
3. **異常スロット（$B_{\text{anom}}$ 枚）のステップ独立・モード別条件付き復元抽出（[L156-L178](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/nab_sampler.py#L156-L178)）**:
   * 不良モード総数 $C$ が $C \le \lfloor B_{\text{anom}} / 2 \rfloor$ または $C \le 4$ の場合は全 $C$ モードを選択し、それを超える多モード環境ではステップ毎に $M_{\text{active}} = \max(1, \lfloor B_{\text{anom}} / 2 \rfloor)$ 個のモードを非復元ランダム選択して各モードに最低2枚以上を割り当てる。
   * 選択されたモード集合 $C_{\text{active}}$ をステップ毎にシャッフルし、各モード $c$ の割当枚数を $k_{\text{mode}} = \lfloor B_{\text{anom}} / |C_{\text{active}}| \rfloor + \mathbb{I}(m < B_{\text{anom}} \bmod |C_{\text{active}}|)$ とする。
   * **枯渇回避ルール**: 各ステップは独立に不良プール `defect_by_class[c]` から $k_{\text{mode}}$ 枚を抽出する。このとき、クラス $c$ の実データ枚数 $|D_{\text{def}}^{(c)}|$ が割当枠未満（$|D_{\text{def}}^{(c)}| < k_{\text{mode}}$）であれば **`replace = True`（ステップ内復元抽出）**、十分にあれば **`replace = False`（ステップ内非復元抽出）** を自動選択する（なお、`train` 分割に該当クラスのサンプルが0件の極小クラスは自動的に全分割のサンプルへフォールバックする）。
4. **重複サンプリングされた少数不良の過学習（丸暗記）を防ぐ4層機構**:
   * ① Sample-to-Proxy 損失により同一サンプル同士の自明なペア計算が発生しない。
   * ② 毎ステップの標準化特徴に対しガウシアン摂動 $\epsilon \sim \mathcal{N}(0, 0.015^2 I)$ を加えて球面再正規化（`F.normalize(batch_gap + noise, p=2, dim=-1)`）する。
   * ③ 毎ステップ入れ替わる新鮮な良品特徴と組み合わせて Defect Transfer ＆ Anomalous Mixup（$\mathcal{B}_{\text{synth}}$）を動的合成する。
   * ④ 平均 $1.0$ に正規化された有効サンプル数重み $W_c$ によりクラス間の勾配スケールを均衡化する。

---

### 3.2. カスタマイズド Multi-Proxy Anchor Loss ([`proxy_anchor_loss.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py))

* **学習可能な正規化プロキシ群**:
  * 良品マルチプロキシ：$\hat{p}_{\text{norm}}^{(m)} = \frac{p_{\text{norm}}^{(m)}}{\|p_{\text{norm}}^{(m)}\|_2} \in \mathbb{S}^{127}$ （$m \in \{1, \dots, M\}$、デフォルト $M = 4$）
  * 不良クラス別プロキシ：$\hat{p}_{\text{def}}^{(c)} = \frac{p_{\text{def}}^{(c)}}{\|p_{\text{def}}^{(c)}\|_2} \in \mathbb{S}^{127}$ （$c \in \{1, \dots, C\}$）
* **全体の目的関数（[`proxy_anchor_loss.py` L372-L373](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L372-L373)）**:
  $$L_{\text{total}} = L_{\text{pos}} + \max(0.5, \gamma_{\text{repel}}(t)) \cdot L_{\text{repel}} + \lambda_{\text{ortho}} L_{\text{proxy-ortho}}$$
  （ここで $\lambda_{\text{ortho}} = \max(0.2, w_{\text{aux}})$、デフォルト $0.5$。戻り値の診断辞書 `stats` では後方互換性のため $L_{\text{pos}}$ を `"loss_ms"` キーに格納する）。

#### (1) 正例凝集項 $L_{\text{pos}} = L_{\text{norm-pos}} + L_{\text{def-pos}}$（$\alpha_{\text{pos}} = 16.0, \; \delta_{\text{pos}} = 0.65$）
1. **良品サブセンター凝集 $L_{\text{norm-pos}}$（[L199-L220](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L199-L220)）**:
   各良品サンプル $z_j \in \mathcal{B}_{\text{norm}}$ を最寄りの良品プロキシ $m^*(j) = \arg\max_{m} \langle z_j, \hat{p}_{\text{norm}}^{(m)} \rangle$ に割り当て、サンプルが割り当てられたアクティブな良品プロキシ集合 $\mathcal{M}_{\text{active}}$ 上で平均をとる：
   $$L_{\text{norm-pos}} = \frac{1}{|\mathcal{M}_{\text{active}}|} \sum_{m \in \mathcal{M}_{\text{active}}} \frac{1}{\alpha_{\text{pos}}} \log\!\left(1 + \sum_{j \in \mathcal{B}_{\text{norm}} : m^*(j)=m} \exp\!\left(-\alpha_{\text{pos}} \left(\langle z_j, \hat{p}_{\text{norm}}^{(m)} \rangle - \delta_{\text{pos}}\right)\right)\right)$$
2. **実不良モード別凝集 $L_{\text{def-pos}}$（[L222-L248](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L222-L248)）**:
   **実不良サンプル $z_i \in \mathcal{B}_{\text{def}}$ のみ**を対象とし（合成サンプル $\mathcal{B}_{\text{synth}}$ は除外）、バッチ内に存在するアクティブな不良クラス集合 $\mathcal{C}_{\text{active}}$ 上でクラスバランス重み $W_c$ を乗じて平均をとる：
   $$L_{\text{def-pos}} = \frac{1}{|\mathcal{C}_{\text{active}}|} \sum_{c \in \mathcal{C}_{\text{active}}} \frac{W_c}{\alpha_{\text{pos}}} \log\!\left(1 + \sum_{i \in \mathcal{B}_{\text{def}} : y_i = c} \exp\!\left(-\alpha_{\text{pos}} \left(\langle z_i, \hat{p}_{\text{def}}^{(c)} \rangle - \delta_{\text{pos}}\right)\right)\right)$$

#### (2) 非対称境界反発項 $L_{\text{repel}}$（3方向平均・不良同士のサンプル間斥力ゼロ：$\gamma_{\text{neg}} = 20.0, \; \delta_{\text{neg}}^{\text{eff}} = \min(\text{repel\_margin}(t), -0.15)$）
実不良と片方向合成ハードサンプルを結合した負例集合を $\mathcal{B}_{\text{all\_neg}} = \mathcal{B}_{\text{def}} \cup \mathcal{B}_{\text{synth}}$ とおく。`loss_repel` は以下の **Direction A, B, C の算術平均** として計算される（[L270-L345](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L270-L345)）：
* **Direction A（全実不良＋合成ハードサンプル vs 全 $M$ 個の良品プロキシ ＋ Top-$K$ 危険不良インプレースブースト、[L272-L301](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L272-L301)）**:
  各 $z_i \in \mathcal{B}_{\text{all\_neg}}$ について良品プロキシ群への最大侵入類似度 $s_{\text{norm}}(z_i) = \max_{m \in \{1..M\}} \langle z_i, \hat{p}_{\text{norm}}^{(m)} \rangle$ を計算し、上位 $K_{\text{eff}}$ 件（`count` モードでは $\text{round}(\text{top\_k\_value})$、`ratio` モードでは $\lceil |\mathcal{B}_{\text{all\_neg}}| \cdot \text{clamp}(q, 0.05, 1.0) \rceil$）のインデックス集合 $\mathcal{H}_K$ に対してブースト倍率 $\lambda_{\text{hard}} = 3.0$（`hard_anomaly_boost`）を付与する：
  $$\omega_i = \max\!\left(10^{-4},\; \text{boost}_i \cdot W_{y_i} \cdot w_{\text{defect}}\right), \quad \text{boost}_i = \begin{cases} \lambda_{\text{hard}} & (i \in \mathcal{H}_K) \\ 1.0 & (i \notin \mathcal{H}_K) \end{cases}$$
  $$L_{\text{repel}}^{(A)} = \frac{1}{M} \sum_{m=1}^{M} \frac{1}{\gamma_{\text{neg}}} \log\!\left(1 + \sum_{i \in \mathcal{B}_{\text{all\_neg}}} \omega_i \exp\!\left(\gamma_{\text{neg}} \left(\langle z_i, \hat{p}_{\text{norm}}^{(m)} \rangle - \delta_{\text{neg}}^{\text{eff}}\right)\right)\right)$$
* **Direction B（バッチ内良品サンプル vs 全 $C$ 個の不良クラスプロキシ、[L303-L317](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L303-L317)）**:
  $$L_{\text{repel}}^{(B)} = \frac{1}{C} \sum_{c=1}^{C} \frac{W_c}{\gamma_{\text{neg}}} \log\!\left(1 + \sum_{j \in \mathcal{B}_{\text{norm}}} \exp\!\left(\gamma_{\text{neg}} \left(\langle z_j, \hat{p}_{\text{def}}^{(c)} \rangle - \delta_{\text{neg}}^{\text{eff}}\right)\right)\right)$$
* **Direction C（全実不良＋合成ハードサンプル vs XBM 内のキャッシュ済み Top-$K$ 境界良品サンプル、[L319-L344](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L319-L344)）**:
  Cross-Batch Memory Bank (`memory_bank.get_top_k_boundary_normals`) から各 $z_i \in \mathcal{B}_{\text{all\_neg}}$ に最も近い上位 $K_{\text{xbm}}$ 件の**キャッシュ済み境界良品ベクトル** $\{b_{i,k}\}_{k=1}^{K_{\text{xbm}}}$ を取得し、境界良品サンプルからも直接押し戻す：
  $$L_{\text{repel}}^{(C)} = \frac{1}{|\mathcal{B}_{\text{all\_neg}}|} \sum_{i \in \mathcal{B}_{\text{all\_neg}}} \frac{1}{\gamma_{\text{neg}}} \log\!\left(1 + \sum_{k=1}^{K_{\text{xbm}}} \exp\!\left(\gamma_{\text{neg}} \left(\langle z_i, b_{i,k} \rangle - \delta_{\text{neg}}^{\text{eff}}\right)\right)\right)$$

---

## 4. 直近の実行ログ（`cluster-63b296a1.json` / `train-96f0aa6f.json`）から判明したデータ運用上の注意点と修正

直近の実行ログと `data/raw/` ディレクトリを調査したところ、以下の事実が判明し修正を完了した：
* `nomal` フォルダ（225枚）をアップロードされた際、UI（[`1_clustering_annotation.py`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/1_clustering_annotation.py)）のチェックボックス「アップロード時に既存の登録画像をクリアする (`clear_before_upload`)」がデフォルト `True` になっていたため、先に存在していた `data/raw/anomaly` フォルダが削除され、**`data/raw/` 内に `nomal`（225枚）しか存在しない状態でクラスタリングと学習が実行されていた**。
* そのため、クラスタリングが未ラベルフォールバック（`density_fallback`）に入り、良品225枚のうち34枚が仮の `anomaly_cluster_1, 2` に割り当てられて学習されていた。
* **UIおよびクライアントの修正（[`1_clustering_annotation.py` L188-L192](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/1_clustering_annotation.py#L188-L192), [`api_client.py` L39-L45](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/api_client.py#L39-L45), [`datasets.py` L196-L238](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/backend/app/routers/datasets.py#L196-L238)）**:
  サブフォルダを個別に複数回アップロードしても既存のサブフォルダや別名ファイルが消えないよう、「アップロード時に既存の登録画像をクリアする (`clear_before_upload`)」のデフォルトを **`False`（`/app/data/raw/<subfolder>/<file_name>` へのファイル単位マージ保存・同一ファイル名のみ上書き）** に変更した（既存データセット全体を初期化したい場合のみ、同チェックボックスを `True` にチェックしてアップロードを実行する）。
