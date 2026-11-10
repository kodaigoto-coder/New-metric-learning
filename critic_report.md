# Goldfish Review Report (Stage 7: Critic Review)

**Target:** `README.md` (`Lines 1-110`)

## Reviewer Status

| Reviewer | Status |
| --- | --- |
| `egm-design` | completed |
| `egm-structural` | completed |
| `egm-style` | completed |

## Prioritized Findings (Blockers -> Major -> Nits)

### 1. 旧ドラフト記述（セクション1〜4）と確定版仕様（セクション5）の併存による仕様の自己矛盾 - Severity: Blocker
* **Reviewer Lens:** `egm-design`, `egm-structural`, `egm-style`
* **Affected Scope:** `Lines 10-68` (Sections 1–4) vs `Lines 71-108` (Section 5)
* **Discrepancy Description**:
  Section 5（`Lines 71-108`）として確定版仕様が末尾に追記された一方、冒頭の Section 1〜3（`Lines 10-59`）に初期ドラフトの記述がそのまま残存しているため、同一ドキュメント内で以下の5箇所が直接矛盾している：
  1. **フェーズ1のクラスタリング構造**: Section 2.2（`Line 35`）では「生データの分布のままK-means法等で自動分類」と単段階クラスタリングを記述しているのに対し、Section 5.1（`Line 76`）では「直接全体PCA＋K-meansを適用すると少数不良が埋没するため2段階クラスタリングを採用する」と否定している。
  2. **データ拡張の適用対象**: Section 3.1（`Line 43`）では「不良画像に対してのみデータ拡張を適用」としているのに対し、Section 5.2（`Line 90`）ではショートカット学習防止のため「良品・不良の双方に同条件で適用」と規定している。
  3. **3D UMAP の前処理次元**: Section 3.3（`Line 58`）では「`PCA(300次元) -> UMAP(3D)`」としているのに対し、Section 5.3（`Line 106`）では「128次元射影ベクトルにはPCAを介さず直接UMAPを適用」としている。
  4. **`.pkl` 永続化の粒度**: Section 3.3（`Line 53`）では「毎エポック全データのGAP特徴量＋射影ベクトルを保存」としているのに対し、Section 5.3（`Line 107`）では2階層保存（毎エポックは128次元ベクトルのみ、GAP特徴量はベスト更新時等のみ）としている。
  5. **最小コサイン距離の定義**: Section 3.3（`Line 55`）では「良品中心から最も近い不良までの距離」としているのに対し、Section 5.2（`Line 98`）では「不良〜最近傍境界良品間の距離」としている。
* **Rationale & Recommendation**:
  ドキュメント内の単一情報源（Single Source of Truth）を確立するため、Section 1〜4 の古い記述を Section 5 の確定仕様に統合・一本化し、矛盾する旧記述を排除する。最小コサイン距離については「中心対不良（Prototype Margin）」と「境界良品対不良（Boundary Instance Margin）」の双方を評価ログに記録し、カリキュラム制御には後者（または両者の組み合わせ）を用いることを一貫して記述する。

---

### 2. Feature Spec Standard 必須セクション（Goals & Non-Goals、代替案の検討、リスクと緩和策）の欠落 - Severity: Blocker
* **Reviewer Lens:** `egm-structural`, `egm-design`
* **Affected Scope:** `Lines 1-110` (Feature Specification 全体)
* **Discrepancy Description**:
  `feature-spec-standard.md` で要求される以下の構造的セクションが独立して定義されていない：
  1. **Goals & Non-Goals（目標と非目標）**: 本システムが達成する目標と、明確にスコープ外とする非目標（例：ピクセル単位の異常セグメンテーションは対象外、製造ラインPLCとのミリ秒単位のハードリアルタイム制御は対象外等）。
  2. **Alternatives Considered（代替案の検討と棄却理由）**: 単一設計の提示にとどまり、他の手法（例：クラスタリングにおける単段階K-meansやHDBSCAN、損失関数における1次元2値分類・Triplet Loss・標準対称MS-Loss、バッチ内マイニング単独）と比較したトレードオフと棄却理由が体系化されていない。
  3. **Risks, Limitations & Mitigations（リスク・制約事項と緩和策）**: Stage 1での微細不良混入リスク、Cross-Batch Memory Bankの特徴ドリフト（Staleness）、極少数サンプル時の統計的制約とそれらの緩和策が独立セクションとして整理されていない。
* **Rationale & Recommendation**:
  FeatureRange 内に「目標と非目標（Goals & Non-Goals）」「代替案の検討（Alternatives Considered）」「リスク・制約事項と緩和策（Risks & Mitigations）」の各セクションを明示的に設け、設計判断の根拠を自己完結させる。

---

### 3. 主観的・宣伝的な修辞表現および口語的比喩の残存 - Severity: Blocker
* **Reviewer Lens:** `egm-style`
* **Affected Scope:** `Lines 18, 30, 34, 44, 48, 53-58, 63, 65-68`
* **Discrepancy Description**:
  * `Lines 65-68`（「💡 統合システムとしての『圧倒的な新規性・進歩性』…極めて強固なオリジナリティを確立しています」）は定量的根拠を伴わない宣伝的修辞（Marketing Fluff）であり、文体も敬体が混在している。
  * `Lines 18, 44, 48, 55, 58, 63` に「幾何学的狙い撃ち」「動的に炙り出し」「くるくくる回転（誤字＋擬態語）」「過誤負例（非標準用語：正しくは偽陰性／False Negative）」等の口語・比喩表現が残っている。
  * `Line 30`（「クラスタリング精度およびシルエットスコアの計算負荷を軽減するため」）に日本語の係り受け誤り（精度を軽減する＝下げるという意味になる）がある。
* **Rationale & Recommendation**:
  `Lines 65-68` の宣伝的段落を削除（または先行研究4論文との関係を客観的に記述するセクションへ置換）し、口語的比喩や誤字をすべて客観的な工学・数理用語へ統一する。

---

### 4. Global Average Pooling (GAP) による局所微小欠陥の希釈リスクと Stage 1 密度定義の精緻化 - Severity: Major
* **Reviewer Lens:** `egm-design`
* **Affected Scope:** `Lines 77-83` (Sections 5.1–5.2)
* **Discrepancy Description**:
  1. 外観検査における微小な傷・打痕は画像面積の数%未満であることが多く、単純な空間平均（GAP）＋バックボーン完全Freezeのみでは、局所的な異常活性化が98%の正常背景に平均化・希釈されるリスクがある。
  2. Section 5.1 Stage 1（`Line 77`）において、小さな近傍数 $k$ による局所 $k$-NN 密度だけを用いると、同一モードの不良が数枚密集した微小クラスタも「局所密度が高い」と判定されて良品コア（$\tau_{\text{core}}$）へ混入する恐れがある。
* **Rationale & Recommendation**:
  1. プーリング層として標準の GAP に加え、局所的な強い反応を保持できる **GeM (Generalized Mean) Pooling** または **Avg+Max 連結プーリング** を設定選択可能とし、バックボーンの最終ステージ（Stage 4 / 最終Transformerブロック）の微調整（Partial Unfreeze）をサポートする仕様を明記する。
  2. Stage 1 の高密度良品コア抽出では、微小不良クラスタを良品コアと誤認しないよう、データセット全体の空間メディアン（またはロバスト重心）からのコサイン距離と、想定不良総数を上回る近傍数 $k$ を用いた大域的密度スコアで良品コアを抽出する仕様とする。

---

### 5. Cross-Batch Memory Bank (XBM) のエポック毎解放に伴う「エポック冒頭 Top-K 枯渇」と「特徴ドリフト」へのガード条件 - Severity: Major
* **Reviewer Lens:** `egm-design`
* **Affected Scope:** `Lines 91-98` (Section 5.2.3)
* **Discrepancy Description**:
  1. OOM Kill 防止のため各エポック終了時に `memory_bank.clear()` でメモリバンクを解放する仕様（`Line 93`）において、次エポック開始直後の最初の数ミニバッチではメモリバンクに蓄積された良品ベクトル数 $|\mathcal{M}_{\text{normal}}|$ が少なく、設定された `top_k_count`（例：100件）を下回るため、そのまま `torch.topk` を呼ぶと実行時エラーとなる。
  2. 学習初期エポックで射影ヘッドの重みが急激に変化する際、メモリバンク内の古いステップのベクトルとの間に特徴ドリフト（Representation Staleness）が生じる。
* **Rationale & Recommendation**:
  1. エポック毎のメモリ解放（`clear()` ＋ `gc.collect()` ＋ `empty_cache()`）を維持しつつ、各ステップでの実効 $K$ 値を **$K_{\text{eff}}(t) = \min\big(K(t), |\mathcal{M}_{\text{normal}}|\big)$**（割合指定時は $\max(1, \lfloor r(t) \cdot |\mathcal{M}_{\text{normal}}| \rfloor)$）として安全にクリップするガード条件を明記する。
  2. 特徴ドリフト対策として、初期ウォームアップエポック（例：$E_{\text{warmup}}$ エポック）はバッチ内基本 MS-Loss のみで学習し、射影空間が安定した後に XBM 反発項 $L_{\text{repel}}$ を有効化する仕様を追加する。

---

### 6. 非対称MS-Lossの基本項 $L_{\text{MS}}$ の数式明記と、閾値 $\tau_{\text{sim}}=0.0$ での初期エポック進捗度 $p(t)$ デッドロック防止 - Severity: Major
* **Reviewer Lens:** `egm-design`
* **Affected Scope:** `Lines 84-104` (Sections 5.2.2–5.3.1)
* **Discrepancy Description**:
  1. `Line 96` に反発項 $L_{\text{repel}}$ の数式はあるが、3種のペア（$\mathcal{P}_i$, $\mathcal{N}_i^{\text{primary}}$, $\mathcal{N}_i^{\text{aux}}$）を非対称に重み付けする基本項 $L_{\text{MS}}$ の数式およびカリキュラム進捗関数 $p(t)$ の式が省略されている。
  2. 学習時のデフォルト閾値 $\tau_{\text{sim}} = 0.0$（`Line 103`）において、学習初期にすべてのサンプルが正の類似度コーン（$s(x) > 0$）内にある間は検証 Recall（@$\tau_{\text{sim}}=0.0$）が $0.0$ のまま推移するため、もし進捗度 $p(t)$ が Recall のみに比例するとスケジューラが初期状態から動かないリスクがある。
* **Rationale & Recommendation**:
  1. 非対称 $L_{\text{MS}}$ の明示的な数式（Primary Negative と Auxiliary Negative の重み分割）を記載する。
  2. カリキュラム進捗度 $p(t) \in [0, 1]$ を、検証データの Recall（@$\tau_{\text{sim}}=0.0$、ユーザー設定可）と、閾値に依存しない連続量である「境界最小コサイン距離 $d_{\min}(t)$（またはエポック進行度）」の重み付き和として定義し、学習初期から滑らかにカリキュラムが進行することを保証する。

---

### 7. 少数サンプル時のシルエットスコア境界条件ガードと Train/Val 分割フォールバック - Severity: Major
* **Reviewer Lens:** `egm-design`
* **Affected Scope:** `Lines 76-79` (Section 5.1)
* **Discrepancy Description**:
  Stage 2 の対象となる低密度テール領域のサンプル数 $N_{\text{tail}}$ が少ない場合、探索上限 $K_{\max}$ が $N_{\text{tail}} - 1$ 以上になるとシルエットスコアが定義不能となる。また、特定の不良モードのサンプル数が極端に少ない（例：1〜2枚）場合、通常の層化 Train/Val 分割が失敗する。
* **Rationale & Recommendation**:
  Stage 2 のクラスタ数探索範囲に $K_{\max} = \min(K_{\max\_\text{config}}, N_{\text{tail}} - 1)$ のガード条件を明記し、各不良クラスのサンプル数が最小分割閾値（例：2枚）未満の場合の Train/Val 割り当て規則（最低1枚をTrain・1枚をValへ確保、1枚のみの場合はTrainへ配置し拡張サンプルまたはLeave-One-Outで検証）を定義する。

---

### 8. Markdown 見出し階層・箇条書き記法の統一とローカル絶対パスリンクの相対パス化 - Severity: Nit
* **Reviewer Lens:** `egm-structural`, `egm-style`
* **Affected Scope:** `Lines 7-73`
* **Discrepancy Description**:
  * Section 1〜4 の見出しに `##` / `###` がなく、箇条書きに Unicode の `• ` とタブ文字が使われている。
  * `Line 73` のリンクが `file:///Users/tmcqdtj1089/...` というローカル絶対パスになっており、GitHub リポジトリ上の `README.md` としてポータビリティを欠く。
* **Rationale & Recommendation**:
  ドキュメント全体を標準の GitHub Flavored Markdown（`##` / `###` 見出し、`-` / `1.` リスト、コードフェンス付きフロー図、相対パスリンク `./note.md`、参考文献リンク）に統一する。

