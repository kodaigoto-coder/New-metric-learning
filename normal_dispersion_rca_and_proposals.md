# 良品クラスタ分散拡大の根本原因調査報告書 ＆ 対策案

## 1. 発生事象と実測データの定量分析（Session #0001: Epoch 1〜100）

直近の100エポック学習ログ（[train-430498da.json](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/data/artifacts/jobs/train-430498da.json) および `/app/data/artifacts/sessions/session_0001/epoch_0001_light.pkl` 〜 `epoch_0100_light.pkl`）から、エポック進行に伴う **良品サンプル群（`train_norm`, `val_norm`）と不良サンプル群（`train_def`, `val_def`）のコサイン類似度 $s(x) = \max_m \langle z(x), p_{\text{norm}}^{(m)} \rangle$ の分布推移** を抽出した結果を以下に示す。

| Epoch | 学習Loss | 良品〜不良 中心距離 `d_min_centroid` | 学習良品 `train_norm` 平均類似度 | 学習良品 標準偏差 `std` (分散指標) | 検証良品 `val_norm` 平均類似度 | 検証良品 標準偏差 `std` (分散指標) | 検証良品のうち $s(x)<0$ に溢れた過検出数 (FP) | 学習不良 `train_def` 平均類似度 | 検証 Precision / F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ep 1** | 1.6016 | 0.2562 | **+0.858** | **0.079** | **+0.842** | **0.091** | **0 / 48枚 (0%)** | -0.514 | **1.0000** / 0.7692 |
| **Ep 2** | 0.7537 | 0.2462 | **+0.856** | **0.081** | **+0.795** | **0.170** | **0 / 48枚 (0%)** | -0.768 | **1.0000 / 0.9333** |
| **Ep 4** | 0.5024 | 0.3014 | **+0.827** | **0.072** | **+0.776** | **0.184** | **0 / 48枚 (0%)** | -0.779 | **1.0000 / 0.9333** |
| **Ep 6** | 0.4543 | 0.1227 | +0.819 | 0.072 | +0.667 | 0.351 | 5 / 48枚 (10%) | -0.819 | 0.5833 / 0.7000 |
| **Ep 11** | 0.3658 | **1.4181** | +0.735 | 0.200 | +0.591 | 0.390 | 4 / 48枚 (8%) | -0.893 | 0.6667 / 0.8000 |
| **Ep 20** | 0.3322 | 0.2451 | +0.647 | 0.280 | +0.496 | 0.507 | 8 / 48枚 (17%) | -0.949 | 0.4667 / 0.6087 |
| **Ep 40** | 0.1140 | 0.4120 | +0.591 | 0.483 | +0.311 | 0.723 | 17 / 48枚 (35%) | -0.961 | 0.2917 / 0.4375 |
| **Ep 80** | 0.0410 | 0.2916 | +0.189 | **0.680** | +0.162 | **0.834** | 19 / 48枚 (40%) | -0.975 | 0.2333 / 0.3684 |
| **Ep 100** | 0.0313 | 0.0372 | +0.444 | **0.552** | **+0.076** | **0.860** | **24 / 48枚 (50%)** | **-0.974** | **0.2258 / 0.3590** |

### 実測データから読み取れる事実
1. **不良クラスタと良品中心の距離は想定通り大きく離れた**:
   * 学習不良サンプル（`train_def`）の良品プロキシに対する平均コサイン類似度は、Epoch 1 の `-0.514` から Epoch 100 には **`-0.974`（ほぼ完全に反対側の極 $-1.0$）** まで押し出されている。
2. **しかしエポックが進むにつれて「良品クラスタの分散」が約8倍〜9倍に膨張して崩壊した**:
   * Epoch 1〜4 では、良品クラスタは平均類似度 `+0.83〜+0.86`、標準偏差 `0.07〜0.18` の狭い領域に密集しており、過検出（FP）は **0枚**（`Precision = 1.000, F1 = 0.9333`）だった。
   * ところが Epoch 11 以降、エポックが進むごとに良品クラスタの標準偏差（分散）が `0.079` → `0.200` → `0.483` → **`0.680`（検証データでは `0.860`）** へと単調に拡大し、最小類似度が **`-0.979`** まで垂れ下がった。結果として Epoch 100 では検証用良品の半数（`24 / 48枚`）が $\tau_{\text{sim}} = 0.0$ を下回って不良領域側へ飛び散り、分離境界が引けなくなっている。

---

## 2. 良品クラスタの分散が大きくなった5つの根本原因

コードベース（[proxy_anchor_loss.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py)、[boundary_augmentation.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/boundary_augmentation.py)、[train_jobs.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py)）と上記スナップショットの勾配挙動を解析した結果、以下の5つの要因が複合して良品クラスタを膨張させていることが判明した。

### 原因①：【引力設計の欠陥】`L_norm_pos` のマージン不感帯（$\delta_{\text{pos}} = 0.65$）と `LogSumExp` による凝集力の消失
[proxy_anchor_loss.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L203-L220) の良品引力項 `L_norm_pos` は以下の式で実装されている：
$$L_{\text{norm-pos}} = \frac{1}{|M_{\text{active}}|} \sum_{m} \frac{1}{\alpha_{\text{pos}}} \log \left( 1 + \sum_{j \in \mathcal{B}_{\text{norm}}^{(m)}} e^{-\alpha_{\text{pos}} (s_{j,m} - \delta_{\text{pos}})} \right), \quad \alpha_{\text{pos}} = 16.0, \; \delta_{\text{pos}} = 0.65$$
この定式化には外観検査の良品クラスタリングにおいて2つの構造的欠陥がある：
1. **$\delta_{\text{pos}} = 0.65$（角度約 $49.5^\circ$）を超えると引力がほぼゼロになる（不感帯）**:
   一般物体認識（鳥や車の分類）ではコサイン類似度 $0.65$ で十分だが、良品・不良品の境界を争う外観検査において半径 $1 - 0.65 = 0.35$（角度 $\pm 49.5^\circ$）の不感帯は広すぎる。類似度が $0.65$ を超えた良品サンプルにはそれ以上中心へ集まろうとする勾配が働かず、クラスタ内部で自由に散らばってしまう。
2. **`LogSumExp` がバッチ内で最も外れた1枚の良品だけに勾配の99%を集中させる**:
   $\alpha_{\text{pos}} = 16.0$ の指数和をとるため、バッチ内の16枚の良品のうち1枚でも類似度が低い個体（例: $s = -0.2$）があると、`LogSumExp` の微分（Softmax重み）はその1枚に $99.9\%$ 集中し、**残りの15枚の良品（例: $s = 0.55$）を引き締める勾配が事実上ゼロ** になる。

---

### 原因②：【逆方向斥力の副作用】`L_repel` の Direction B（不良プロキシから良品サンプルへの斥力）が良品クラスタに遠心力を与えている
[proxy_anchor_loss.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L303-L317) の **Direction B** では、ミニバッチ内の良品サンプル $z_{\text{norm}}$ に対して、$C$ 個の不良プロキシ $p_{\text{def}}^{(1)}, \dots, p_{\text{def}}^{(C)}$ からの反発損失を計算している：
```python
# Direction B: Repel Normal Samples from ALL C Defect Mode Proxies
sim_norm_def = z_norm @ p_def.T  # [B_norm, C]
```
* $C$ 個の不良プロキシは直交正則化 `L_proxy_ortho` によって超球面上の別々の方向を向いている。
* そこへ Direction B によって「良品サンプル $z_{\text{norm}}$ を各不良プロキシ $p_{\text{def}}^{(c)}$ の反対方向 $-p_{\text{def}}^{(c)}$ へ押し出す勾配」が毎ステップ加わると、**バッチ内の良品サンプルごとに異なる方向（$-p_{\text{def}}^{(1)}, -p_{\text{def}}^{(2)}, \dots$）へ弾き飛ばされる遠心力** が働く。
* 本来、良品サンプル $z_{\text{norm}}$ は「良品プロキシ $p_{\text{norm}}$ へ一極集中する力」だけを受ければ自動的に不良領域から離れるにもかかわらず、複数の不良プロキシから四方八方へ押し出されたことで良品クラスタの形状が歪み、分散が増大した。

---

### 原因③：【プロキシのドリフト】`L_repel` Direction A で良品プロキシ $p_{\text{norm}}$ が不良からの反作用を直接受けて移動している
[proxy_anchor_loss.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L272-L301) の **Direction A**（`sim_neg_norm = z_all_neg @ p_norm.T`）では、`p_norm` を `detach()` せずに内積をとっており、かつ [train_jobs.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L430-L433) でプロキシの学習率をモデルの **2.5倍（`lr * 2.5`）** に設定している。
* Epoch 1〜4 で良品サンプルが $\delta_{\text{pos}} = 0.65$ を超えて `L_norm_pos`（良品プロキシを良品重心に繋ぎ止める力）が弱まった後、Direction A において **最大 $7.5$ 倍（$\lambda_{\text{hard}} 3.0 \times w_{\text{defect}} 2.5$）にブーストされた不良＆合成サンプル群が `p_norm` を強く押し返す**。
* その結果、「固定された錨（アンカー）」であるべき良品プロキシ $p_{\text{norm}}$ 自身が、不良サンプルからの反作用で本来の良品重心（`normal_prototype`）から押し流されてドリフトしている（実測値でも `dot(p_norm, normal_prototype)` が Epoch 1 の `0.957` から Epoch 80 には **`0.539`** まで乖離していた）。
* 毎ステップ `p_norm` が逃げ回るため、そのステップのバッチに含まれた一部の良品だけが移動後の `p_norm` に引っ張られ、バッチ外の良品や検証用良品が取り残されて良品クラスタが引き裂かれた。

---

### 原因④：【境界合成の過剰ブレンド】$\alpha \in [0.80, 0.90]$ の Anomalous Mixup が良品特徴そのものを反対半球へ弾き飛ばしている
[boundary_augmentation.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/boundary_augmentation.py#L147-L207) および [train_jobs.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py#L477-L504) では、毎ステップ以下の合成特徴を生成して `z_all_neg` に追加している：
$$\tilde{h}_{\text{mix}} = \alpha h_{\text{norm}} + (1 - \alpha) h_{\text{def}}, \quad \alpha \in [0.80, 0.90]$$
* $\alpha = 0.85$ のとき、合成ベクトル $\tilde{h}_{\text{mix}}$ は **元の良品特徴 $h_{\text{norm}}$ とコサイン類似度で $0.98$ 以上一致する（ほぼ良品そのもの）**。
* この「85%良品の合成サンプル」を、毎ステップ Top-$K$ ハード不良ブースト（重み $7.5$ 倍）の対象に含めて「コサイン類似度 $-0.15$ 以下（反対半球）へ飛ばせ」と100エポック学習させると、射影ヘッド `projection_head` は **「良品特徴 $h_{\text{norm}}$ のほんのわずかな揺らぎを極端に拡大して $-1.0$ 側へ吹き飛ばす」** ように重みを先鋭化させてしまう（過学習）。
* これが、エポックが進むほど検証用良品（`val_norm`）の半数が `-0.979` 付近（不良クラスタのど真ん中）へ吹き飛んだ直接のメカニズムである。

---

### 原因⑤：【データセット状態】直近セッション（`session_0001`）が `nomal/` 225枚のみの `density_fallback` アノテーションで学習されていた点
[data/manifests/clustering_result.json](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/data/manifests/clustering_result.json#L1-L18) と [data/manifests/annotations.json](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/data/manifests/annotations.json#L1-L15) を確認したところ、`/app/data/raw/` に `nomal/`（`test_good_*`, `train_good_*` 計225枚）のみが存在する状態で Phase 1 クラスタリングが実行されたため、`clustering_mode: "density_fallback"` が作動し、**良品225枚のうち周辺34枚（`test_good_006` 等）が `anomaly_cluster_1` (16枚)・`anomaly_cluster_2` (18枚) として疑似不良ラベル化** されていた。
* 実質的に同じ良品分布（`good`）から切り出された34枚を「不良」として扱い、原因①〜④の強い斥力を100エポック適用したため、同じ特徴を持つ検証用良品（`val_norm`）が次々と不良側へ引きずり込まれた。

---

## 3. 50:50 パワーバランス Loss 設計の検証（ストレステスト）と可変比率化仕様

ユーザー提案の **コサイン類似度ベース 50:50 パワーバランス設計**：
$$L_{\text{total}} = \underbrace{\lambda_{\text{norm}} \cdot \frac{1}{|\mathcal{B}_{\text{norm}}|} \sum_{x \in \mathcal{B}_{\text{norm}}} \left(1 - S(x, p_{\text{norm}})\right)}_{\text{① 良品の求心力（基準重み：50%, }\lambda_{\text{norm}}=1.0\text{）}} + \underbrace{\lambda_{\text{anom}} \cdot \frac{1}{|\mathcal{B}_{\text{anom}}|} \sum_{x \in \mathcal{B}_{\text{anom}}} \max\left(0, S(x, p_{\text{norm}}) - \delta\right)}_{\text{② 不良の斥力（基準重み：50%, }\lambda_{\text{anom}}=1.0\text{）}}$$

### 3.1 数学的ストレステスト（強みと潜在的な死角の洗い出し）

1. **強み：二乗項 $(1-S)^2$ や `LogSumExp` に対する線形コサイン距離 $1 - S(x, p_{\text{norm}})$ の優位性**
   * Deep SVDD 型の二乗項 $(1 - S)^2$ は、類似度 $S$ が $1.0$ に近づくほど勾配 $\frac{\partial}{\partial S}(1-S)^2 = -2(1-S) \to 0$ が減衰し、逆に外れ値（$S \approx -1$）では勾配が $-4$ に跳ね上がるため、学習初期に求心力が過剰になりやすい。
   * これに対し、提案された一次形式 $1 - S(x, p_{\text{norm}})$ は **$S \in (-1, 1)$ の全域で微係数が常に一定（$\frac{\partial L_{\text{norm}}}{\partial S} = -1.0$）** であり、不良斥力項 $\max(0, S - \delta)$ のアクティブ領域（$S > \delta$）における微係数（$\frac{\partial L_{\text{anom}}}{\partial S} = +1.0$）と **勾配の絶対値スケールが厳密に $1.0 : 1.0$ で一致** する。また、`LogSumExp` のようにハズレ値1枚が勾配を独占せず、バッチ内の全良品サンプル $|\mathcal{B}_{\text{norm}}|$ に均等な求心力が働く。

2. **死角①（要対策）：第2項（不良斥力）における `p_norm` への逆向き勾配の流入**
   * 第2項 $\max(0, S(x, p_{\text{norm}}) - \delta)$ を計算する際、良品プロキシ $p_{\text{norm}}$ に勾配を流したままにすると、第1項が $p_{\text{norm}}$ を良品重心へ引き寄せる力（$+1.0$）と、第2項が $p_{\text{norm}}$ を不良と反対側へ押し出す力（$-1.0$）が正面衝突し、再びプロキシのドリフトを引き起こす。
   * **対策**: 第2項の類似度は **`S(x, p_norm.detach())`** として計算し、不良斥力の勾配（50%）は **100% 不良サンプルの特徴抽出器（`projection_head`）の移動のみ** に使わせる。これにより良品プロキシ $p_{\text{norm}}$ は常に純粋な良品重心（アンカー）に固定される。

3. **死角②（要対策）：Top-$K$ ブーストやクラス重み $W_c$ による 50:50 バランスの崩壊**
   * 従来コードのように第2項の内部で `hard_anomaly_boost`（$\lambda_{\text{hard}} = 3.0$）や `w_defect`（$2.5$）をそのまま乗算すると、第2項の実効スケールが $1.0 \times 7.5 = 7.5$ 倍に膨れ上がり、実質的に「良品求心力 12% : 不良斥力 88%」へと偏ってしまう。
   * **対策**: 不良バッチ $\mathcal{B}_{\text{anom}}$ 内で Top-$K$ 危険不良や希少不良クラスに相対的な優先度重み $w_i$ を付与する際は、**バッチ内平均が必ず $1.0$ になるよう正規化した重み $\tilde{w}_i = \frac{w_i}{\frac{1}{|\mathcal{B}_{\text{anom}}|} \sum_{k} w_k}$** を用いる：
     $$L_{\text{anom}} = \frac{1}{|\mathcal{B}_{\text{anom}}|} \sum_{x_i \in \mathcal{B}_{\text{anom}}} \tilde{w}_i \max\left(0, S(x_i, p_{\text{norm}}\text{.detach()}) - \delta\right), \quad \left(\frac{1}{|\mathcal{B}_{\text{anom}}|} \sum_i \tilde{w}_i = 1.0\right)$$
     これにより、「不良サンプル間の優先順位付け（良品に近い危険不良を狙い撃ちする効果）」を維持しながら、**良品求心力 vs 不良斥力の全体パワーバランス（50 : 50）が絶対に崩れない** 構造が成立する。

4. **死角③（要対策）：ハードヒンジ $\max(0, S - \delta)$ における $\delta = 0.0$ 境界停止とマルチ不良モードの分離**
   * 純粋なハードヒンジ $\max(0, S - \delta)$ は $S \le \delta$ に到達した瞬間に勾配が完全に $0$ になる。$\delta = 0.0$ が推論閾値 $\tau_{\text{sim}} = 0.0$ と同一だと、学習不良サンプルが $S = 0.0$（識別境界線上ギリギリ）で停止し、未知の検証不良に対する安全マージンが不足する場合がある。
   * **対策**:
     1. マージン $\delta$（`repel_margin`）を UI 上からスライダーで調整可能とし（デフォルト $\delta = 0.0$、カリキュラムまたは手動で $-0.15 \sim +0.20$ まで可変）、平滑化ヒンジ（微小温度付き Soft-Hinge または Straight-Through Hard Hinge）で境界付近の振動を抑える。
     2. 複数種類の不良サブクラス（`Cluster 1..C`）同士が1点に潰れないよう、実不良サンプルに対する不良モードプロキシ $p_{\text{def}}^{(c)}$ への補助凝集項 $L_{\text{def-sub}}$ は全体のパワーバランスを邪魔しない微小係数（`0.10`）かつ **良品サンプルには一切遠心力（Direction B）を及ぼさない形式** に限定する。

### 3.2 可変パワーバランス比率（`normal_loss_ratio`）の定式化

UI（`2_training_monitor.py`）のスライダーで **「良品求心力 : 不良斥力 のパワー比率」** $r_{\text{norm}} \in [0.10, 0.90]$（デフォルト **`0.50` ＝ 50% : 50%**）を指定可能とする。
$$\lambda_{\text{norm}} = 2.0 \cdot r_{\text{norm}}, \qquad \lambda_{\text{anom}} = 2.0 \cdot (1.0 - r_{\text{norm}})$$
* $r_{\text{norm}} = 0.50$（デフォルト 50:50）のとき：$\lambda_{\text{norm}} = 1.0, \; \lambda_{\text{anom}} = 1.0$ となり、ユーザー提案式と完全一致する。
* 良品をさらに強く1点に凝集させたい場合は $r_{\text{norm}} = 0.65$（65:35 $\implies \lambda_{\text{norm}}=1.3, \lambda_{\text{anom}}=0.7$）、逆に不良の押し出しを強めたい場合は $r_{\text{norm}} = 0.35$（35:65 $\implies \lambda_{\text{norm}}=0.7, \lambda_{\text{anom}}=1.3$）へと直感的に調整できる。

---

## 4. データセット登録・切り替え・ラベル修正に関するアプリ仕様の問題点と修正設計

実機ログ（`data/datasets/transistor`、`data/active_dataset.json`、`cluster-fd28762a.json`、`train-e47c37a4.json`）およびフロントエンド／バックエンド実装を監査した結果、**「新しいデータセットへの切り替えが勝手に `default` に戻ってしまうバグ」** と **「一度付いたラベルやサブフォルダ構成をUIから修正できない仕様欠陥」** の計4点が特定された。

### 問題①：データセット切り替えが直後に `default` に巻き戻るバグ（`st.session_state` 競合）
* **原因**: [1_clustering_annotation.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/1_clustering_annotation.py#L131-L175) において、サイドバーの `st.selectbox("検証対象データセット", key="sidebar_ds_select")` とメイン画面セクション0の「新規データセット作成 / 切り替え」ボタンが共存している。セクション0で新規データセット（例: `transistor`）を作成して `client.select_or_create_dataset("transistor")` → `st.rerun()` が実行されると、再描画時の先頭（139行目）でサイドバーの `selectbox` が評価される。このとき Streamlit の仕様上、`key="sidebar_ds_select"` がすでに `st.session_state` に古い値 `"default"` を保持していると引数 `index=sb_idx` が無視されて `sb_selected_ds = "default"` が返り、146行目の `if sb_selected_ds != active_dataset_name:` が発火して **即座にバックエンドを `"default"` に書き戻していた**（16:20 に作成された `transistor` が空のまま 16:21 のクラスタリングと 16:25 の Session #0002 学習が `default` で回っていた直接原因）。
* **修正**:
  * 全ページ（`1_clustering_annotation.py`, `2_training_monitor.py`, `3_space_evaluation.py`）において、`active_dataset_name` が変わった際やボタン操作時に `st.session_state["sidebar_ds_select"]` と `st.session_state["main_ds_selector"]` を強制同期し、`selectbox` の切り替えは `on_change` コールバックでのみ発火させる。

### 問題②：多階層フォルダ（例: MVTec AD `transistor/test/bent_lead/000.png`）アップロード時のクラス名誤認識とファイル上書き
* **原因**: [1_clustering_annotation.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/1_clustering_annotation.py#L39-L65) の `_resolve_subfolder_and_filename` は、共通ルートを1階層だけ剥がした後の先頭要素 `parts[0]` をクラス名としていた。そのため `transistor/train/good/000.png` や `transistor/test/bent_lead/000.png` のような階層構造をアップロードすると、クラス名が `"good"` や `"bent_lead"` ではなく **`"train"` や `"test"`** になってしまい、さらに `train/good/000.png` と `test/good/000.png` のファイル名 `000.png` が衝突して上書きされる。
* **修正**:
  * 画像ファイルの **直上の親フォルダ名 `parts[-2]`**（例: `good`, `bent_lead`, `nomal`, `anomaly`）をクラス名として抽出する（標準的な検査データセット構造すべてに対応）。
  * 中間フォルダ階層（`train`, `test` 等）が存在する場合はファイル名にプレフィックス付与（`train_000.png`, `test_000.png`）して同名ファイルの上書き消失を防止する。

### 問題③：単一フォルダ状態でクラスタリングを実行した際の `density_fallback` による疑似不良誤ラベル化
* **原因**: [two_stage_cluster.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/two_stage_cluster.py#L191-L203) において、良品フォルダ1つしか存在しない状態でクラスタリングを実行すると、無条件で `density_fallback` が作動して良品の下位15%（34枚）を `anomaly_cluster_1, 2`（不良）に強制変換していた。
* **修正**:
  * サブフォルダ構造が存在し、全画像が指定された良品クラス（`normal_class_name`）に属している場合は、勝手に良品を不良へ分割しないよう UI に「単一フォルダ時の密度フォールバック分割を許可する（ラベルなし混合データ用）」オプションを分離し、誤って良品だけの状態で実行した際に警告を表示する。
  * さらに、クラスタリングを実行しなくても **現在のサブフォルダ構成からワンクリックで直接 `annotations.json` を生成・更新できるボタン**（「サブフォルダ名をそのまま正解ラベルとして登録」）をセクション1・セクション3の両方に設ける。

### 問題④：登録済みデータセットのクラス名変更・削除・個別画像のラベル修正ができない問題
* **原因**: 一度アップロードまたはアノテーション確定（`annotations.json` 保存）を行うと、UI 上に「特定サブフォルダ（クラス）の削除・リネーム」「誤ラベル画像の別クラスへの移動」「確定済み `annotations.json` の一覧確認とインタラクティブ編集（`st.data_editor`）」が存在しなかった。
* **修正**:
  1. **サブフォルダ（クラス）管理 API ＆ UI の追加**:
     * `POST /api/v1/datasets/{dataset_name}/classes/rename`: 指定したサブフォルダ（クラス）を別名に変更または既存クラスへ統合（`raw/` 内のディレクトリ移動＋マニフェスト自動更新）。
     * `DELETE /api/v1/datasets/{dataset_name}/classes/{class_name}`: 誤って登録した特定のクラスフォルダのみを個別削除。
  2. **インタラクティブ・アノテーション編集テーブル (`st.data_editor`) ＆ 一括ラベル修正機能の追加**:
     * `1_clustering_annotation.py` のセクション3末尾（およびセクション4）に、現在の `annotations.json`（未作成時はクラスタリング結果または `raw/` スキャン結果）の全画像リストをテーブル表示し、**UIのプルダウンや一括変更で任意の画像の `label_name`（良品 / 各不良クラス）や `split`（`train` / `val`）を直接書き換えて即座にデータベース（`annotations.json` および必要に応じて `raw/` サブフォルダ）へ反映できる機能** を実装する。

---

## Detailed Implementation Plan

### Step 1: 50:50 パワーバランス Loss ＆ 可変比率スライダーの実装
* **対象ファイル**:
  * [trainer/app/core/proxy_anchor_loss.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py) (`CustomizedMultiProxyAnchorLoss`)
  * [trainer/app/core/boundary_augmentation.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/boundary_augmentation.py) (`synthesize_anomalous_mixup_images`, `synthesize_boundary_gap_features`)
  * [trainer/app/schemas/job_models.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/schemas/job_models.py) (`TrainingJobRequest`)
  * [backend/app/schemas/api_models.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/backend/app/schemas/api_models.py) (`TrainingTriggerRequest`)
  * [trainer/app/routers/train_jobs.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py) (`_execute_training_task`)
  * [frontend/app/pages/2_training_monitor.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/2_training_monitor.py)
* **実装詳細**:
  1. `CustomizedMultiProxyAnchorLoss` に `normal_loss_ratio: float = 0.50`（デフォルト 50:50 $\implies \lambda_{\text{norm}} = 1.0, \lambda_{\text{anom}} = 1.0$）を導入。
  2. **① 良品の求心力項 (`loss_norm_centripetal`)**:
     $$S_{\text{norm}}(x, p_{\text{norm}}) = \max_{m \in \{1..M\}} \langle z(x), p_{\text{norm}}^{(m)} \rangle, \qquad L_{\text{norm}} = \frac{1}{|\mathcal{B}_{\text{norm}}|} \sum_{x \in \mathcal{B}_{\text{norm}}} \left(1 - S_{\text{norm}}(x, p_{\text{norm}})\right)$$
     全良品サンプルに対して減衰のない一定の求心勾配（スケール $1.0$）を与え、エポック進行に伴う良品分散拡大を防止する。
  3. **② 不良の斥力項 (`loss_anom_repel`)**:
     * 良品プロキシを `p_norm.detach()` でアンカー固定した上で、各不良サンプルの最大侵入類似度 $S_{\text{anom}}(x_i, p_{\text{norm}}) = \max_m \langle z(x_i), p_{\text{norm}}^{(m)}\text{.detach()} \rangle$ に対し、平均 $1.0$ に正規化した優先度重み $\tilde{w}_i$（実不良の Top-$K$ 危険サンプルを優先し、合成サンプルは $0.25$ 倍に抑制）を掛けたマージン付きコサイン斥力損失を計算する：
       $$L_{\text{anom}} = \frac{1}{|\mathcal{B}_{\text{anom}}|} \sum_{x_i \in \mathcal{B}_{\text{anom}}} \tilde{w}_i \max\left(0, S_{\text{anom}}(x_i, p_{\text{norm}}\text{.detach()}) - \delta\right)$$
     * 良品サンプルに遠心力を与えていた旧 Direction B（`z_norm @ p_def.T`）から良品サンプルへの勾配を完全排除（`z_norm.detach()`）する。
     * プロキシのオプティマイザ学習率を `req.learning_rate * 0.5` に落ち着かせ、アンカーの安定性を保証する。
  4. `boundary_augmentation.py` およびスキーマの Anomalous Mixup デフォルトブレンド範囲を `alpha_min = 0.55, alpha_max = 0.75`（下限許容 `0.30`）に改め、良品特徴そのものへの過学習を防止する。
  5. `2_training_monitor.py` に **「⚖️ 良品求心力 vs 不良斥力 パワーバランス比率 (`normal_loss_ratio`)」スライダー（デフォルト `0.50` = 50% : 50%）** を追加し、エポック履歴にも `loss_centripetal`（良品求心項）と `loss_repel`（不良斥力項）を記録・可視化する。

### Step 2: データセット切り替えバグ修正・サブフォルダ管理・ラベル直接編集機能の実装
* **対象ファイル**:
  * [backend/app/schemas/api_models.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/backend/app/schemas/api_models.py)
  * [backend/app/routers/datasets.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/backend/app/routers/datasets.py)
  * [frontend/app/api_client.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/api_client.py)
  * [frontend/app/pages/1_clustering_annotation.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/1_clustering_annotation.py)
  * [frontend/app/pages/2_training_monitor.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/2_training_monitor.py)
  * [frontend/app/pages/3_space_evaluation.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/3_space_evaluation.py)
* **実装詳細**:
  1. **アクティブデータセット切替の `st.session_state` 競合解消**:
     * 全ページのサイドバーおよびデータセット管理パネルで `_sync_active_dataset_state(active_dataset_name)` を実行し、`on_change` コールバック方式に統一することで、新規データセット作成や切り替え後に `default` へ巻き戻るバグを根絶する。
  2. **多階層フォルダアップロード時の直上親フォルダ抽出**:
     * `_resolve_subfolder_and_filename` を改修し、`transistor/test/bent_lead/000.png` のような階層でも直上の親ディレクトリ `parts[-2]` (`bent_lead`) をクラス名とし、中間パスをファイル名プレフィックス（`test_000.png`）として保持する。
  3. **クラス（サブフォルダ）のリネーム・統合・個別削除 API ＆ UI**:
     * `POST /api/v1/datasets/{dataset_name}/classes/rename`: `old_class_name` から `new_class_name` へ画像を移動・統合し、既存マニフェストも同期更新する。
     * `DELETE /api/v1/datasets/{dataset_name}/classes/{class_name}`: 指定したサブフォルダクラスのみを削除する。
     * `POST /api/v1/annotations/direct-update`: UI のテーブルエディタ（`st.data_editor`）や一括ラベル変更から送信された各画像の `label_name`（およびオプションで `sync_subfolders=True` による実ファイルのサブフォルダ移動）を受け取り、即座に `annotations.json` を再構築・保存する。
  4. **Phase 1 UI (`1_clustering_annotation.py`) のラベル修正ワークフロー強化**:
     * セクション1に「登録済みクラス（サブフォルダ）の名称変更・統合・個別削除」および「クラスタリングなしでサブフォルダ名をそのまま学習ラベル (`annotations.json`) として即時登録」ボタンを追加。
     * セクション3に「特定クラスタの全画像を一括で良品/指定クラスへ変更」および **「📝 登録済み全画像のラベル一覧確認・個別/一括修正テーブル (`st.data_editor`)」** を追加し、ラベル付けミスをいつでも何度でもUI上から自在に修正・再保存できるようにする。
