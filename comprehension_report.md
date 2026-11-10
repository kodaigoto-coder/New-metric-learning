**What this proposes**
本書（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L1-L68)）は、製造業の外観検査における未ラベル・不均衡データ（100:2）を対象に、「フェーズ1：前処理自動分類」と「フェーズ2：弱教師あり動的Metric-Learning」を統合した異常検知AI開発基盤を提案しています。フェーズ1（Step 1〜3）では、事前学習済みEfficientNet/ViTによる特徴量抽出とPCA次元削減の後、シルエットスコアで最適クラスター数 $k$ を自動選定してK-meansクラスタリングを行い、人間がUI上で「正常」および複数の「不良モード」のラベルを確定（疑似多クラス化）します。フェーズ2（Step 4〜6）では、不良画像のデータ拡張とClass Weightsを適用しつつ128次元球体空間へ射影する独自MS-Lossモデルを学習し、検証データの再現率（Recall）や最小コサイン距離に連動して各不良データに近接する境界線上の良品（Top K）へ動的に斥力を課すペナルティ・スケジューラーを適用します。さらに各エポックで特徴量・射影ベクトル・ラベルを `.pkl` 保存し、良品中心からの最小コサイン距離、128次元生ベクトルのPlotlyヒートマップ、およびPCA(300次元)→UMAP(3D)散布図の3軸で空間を評価します。

**How the existing system works, per this document**
本書（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L1-L68)）には、既存のシステム・既存のコードベース・現行の運用パイプラインがどのように動作しているかについての記述が存在しません。比較対象としては、画像ピクセルを直接クラスタリングすることの回避（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L29)）や、識別軸をOK/NGの1次元直線にすると特定の不良に引っ張られるという一般的な課題（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L44)）、3D可視化の見た目だけに頼る評価への懸念（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L55)）が断片的に言及されているのみで、本プロジェクトの現状実装やベースライン仕様は記載されていません。

---

### 監査結果（Comprehension Findings）

#### 1. [Blocker] Step 4「独自MS-Loss」と Step 5「ペナルティ・スケジューラー」の数理的定義および接続関係の欠落
* **Severity**: 🔴 Blocker
* **Affected Scope**: `1. 統合システム全体像・処理フロー`（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L18-L20)）、`3.1. 多次元空間（128次元）を維持した「疑似多クラス」MS-Loss学習（Step 4）`（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L41-L45)）、`3.2. 進捗駆動型（正解率・マージン連動）ペナルティ・スケジューラー（Step 5）`（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L46-L49)）
* **Discrepancy Description & Rationale**:
  プロジェクト固有のコア概念である「独自MS-Loss」および「動的押し戻し（反発力／斥力）」の具体的な定義がなく、Step 4 と Step 5 がどのように接続されるのかを初見の読者が一貫してパラフレーズ・実装できません。
  1. **「独自MS-Loss」の未定義**: 一般的な Multi-Similarity Loss (MS-Loss) に対して何が「独自」なのか（数式、ハイパーパラメータ、ペア重み付けの変更点）が一切定義されていません。
  2. **疑似多クラスとClass Weightsの作用方法の不明**: フェーズ1で作られた「正常」「不良モードA」「不良モードB」という疑似多クラスラベルに対し、不良モード同士（AとB）も互いに反発させるのか、あるいは「良品 vs 各不良モード」のみを反発させるのかが不明です。また、ペアベースの損失関数である MS-Loss に対して「不良クラスの損失（Loss）の重みを動的に増大させる（Class Weights）」（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L43)）をアンカー単位・ペア単位のどちらにどう乗じるのかが説明されていません。
  3. **Step 4 と Step 5 の関係性の矛盾・曖昧さ**: フロー図（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L18-L20)）では Step 4 の後に Step 5 が直列に実行されるように描かれていますが、本文（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L48-L49)）では「毎ステップ/ミニバッチ内で」「学習初期から学習後期へと」と記述されており、同一学習ループ内の仕組みに見えます。この「強力な反発力（斥力）」が MS-Loss 内のネガティブペア重み/マージン項を直接変更するものなのか、別個のペナルティ損失項 $L_{\text{penalty}}$ を加算するものなのかが不明です。

---

#### 2. [Blocker] 128次元空間から「正常／不良（OK/NG）」を判定する推論・識別ルール（閾値決定法）の未定義
* **Severity**: 🔴 Blocker
* **Affected Scope**: `1. 統合システム全体像・処理フロー`（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L9)）、`3.1`（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L44)）、`3.2`（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L49)）、`4. 非機能要件・評価基準`（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L63)）
* **Discrepancy Description & Rationale**:
  モデルの最終出力は2値分類ロジットではなく「128次元などの多次元球体空間」上のベクトル（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L44)）とされていますが、その128次元ベクトルからどのようにして各サンプルを「正常（良品）」または「異常（不良）」と2値判定するのかが文書内のどこにも定義されていません。
  * Step 5 では「検証データの再現率（Recall）」（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L49)）や「正解率」（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L20), [L46](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L46)）に連動してスケジューラーを制御するとし、Section 4 では「再現率（Recall）およびF1スコアを最重要視する」（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L63)）と規定していますが、判定ルール（例：「良品中心」からのコサイン距離に対する閾値判定なのか、k-NNなのか、別途分類ヘッドを持つのか、またその閾値や「良品中心」を学習データの何からどう算出するのか）が定義されていないため、Recall や F1 スコアの算出自体が不可能な状態になっています。

---

#### 3. [Major] モデルのネットワーク構造（GAP特徴量・射影ベクトル・学習対象範囲）の未定義と次元数の不整合（128次元 vs PCA 300次元）
* **Severity**: 🟠 Major
* **Affected Scope**: `2.1`（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L29-L30)）、`3.1`（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L41-L44)）、`3.3. 空間の永続保存 ＆ 3軸多角評価システム（Step 6）`（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L53-L58)）、`4. 非機能要件・評価基準`（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L62)）
* **Discrepancy Description & Rationale**:
  1. **「GAP特徴量」と「射影ベクトル」の唐突な出現と学習範囲の曖昧さ**: Step 6（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L53)）で初めて「GAP特徴量」「射影ベクトル」という用語が未定義のまま登場します。Step 4 では画像に対する「回転・反転・輝度調整」のデータ拡張（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L43)）を行うため画像から特徴抽出器を通す必要がありますが、フェーズ2においてバックボーン（EfficientNet-B3 / ViT）の重みまでファインチューニングするのか、バックボーンは固定（Frozen）して Global Average Pooling (GAP) 後の特徴量から128次元への射影ヘッド（Projection Head）のみを学習するのかが説明されていません。
  2. **128次元空間と `PCA(300次元) -> UMAP(3D)` の次元数矛盾**: Step 4 および Step 6（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L41), [L44](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L44), [L56](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L56)）では学習される空間（生ベクトル／射影ベクトル）を **128次元** としていますが、[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L58) では「良品空間から不良品が全方位に引き離されていく様子」を可視化するために `PCA(300次元) -> UMAP(3D) パイプライン` を使うと記述しています。128次元の射影ベクトルに対して300次元へのPCAは数学的に成立せず、もしバックボーンのGAP特徴量（例：1536次元）に対してPCA(300次元)をかけるのであれば、なぜ128次元のMetric-Learning射影空間ではなくGAP特徴量空間を可視化するのか（特にバックボーンがFrozenの場合はGAP特徴量はエポックが進んでも変化しない）という説明が欠落しています。

---

#### 4. [Major] Step 5 カリキュラム学習・動的サンプリングの制御仕様の曖昧さ（用語の揺れ・バッチ制約・遷移条件の未定義）
* **Severity**: 🟠 Major
* **Affected Scope**: `1. 統合システム全体像・処理フロー`（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L20)）、`3.2. 進捗駆動型（正解率・マージン連動）ペナルティ・スケジューラー（Step 5）`（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L46-L49)）
* **Discrepancy Description & Rationale**:
  * **「正解率」と「再現率（Recall）」の用語の不一致**: 見出し（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L20), [L46](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L46)）では「正解率/マージン連動」と書かれている一方、本文（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L49)）では「検証データの再現率（Recall）や高次元空間の最小コサイン距離に連動」と書かれており、さらに [README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L63) では「全体のAccuracyではなく、再現率（Recall）およびF1スコアを最重要視する」と述べられています。スケジューラーのトリガー指標が Accuracy なのか Recall なのかが矛盾しています。
  * **自動遷移ロジックのブラックボックス化**: 「学習初期（ペナルティ弱、Kは広く）から学習後期（ペナルティ強、KをTop 20に絞り込む）へと、自動でマージン制約の厳しさを変化させる」（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L49)）とありますが、初期の $K$ の値、「ペナルティ弱／強」が具体的にどのパラメータ（マージン値、損失重み等）をどう変えるのか、また Recall と最小コサイン距離がどの条件を満たしたときに初期から後期へ遷移するのか（連続関数か段階的閾値か）が一切定義されていません。
  * **ミニバッチ内 Top K 抽出の前提条件**: 「毎ステップ/ミニバッチ内で、各不良データに対して...良品データ（Top K件）を動的に炙り出し」（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L48)）とあり、後期でさえ $K=20$（初期は「Kは広く」＝20超）としていますが、ミニバッチサイズやバッチ内の良品・不良品サンプリング戦略（各ミニバッチに必ず不良データと $K$ 件以上の良品データが含まれる保証）が定義されていません。

---

#### 5. [Major] フェーズ1（教師なしクラスタリング）からフェーズ2（弱教師あり学習）へのデータ受け渡しとラベリング粒度の未定義
* **Severity**: 🟠 Major
* **Affected Scope**: `1. 統合システム全体像・処理フロー`（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L10-L16)）、`2.1`〜`2.2`（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L27-L37)）、`3.1`（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L43)）
* **Discrepancy Description & Rationale**:
  * **アノテーションUIの操作粒度と混在クラスターへの対処**: [README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L36) では「クラスタリング結果をユーザーが画面上で確認し、『正常』『不良モードA（傷）』『不良モードB（汚れ）』と、最小限の手間で正解ラベルを確定（疑似多クラス化）できる出力インタフェースを構築する」と記載されています。しかし、K-means の1つのクラスター内に正常品と不良品が混在した場合に、クラスター単位でしか一括ラベル付けできないのか、個別画像を除外・修正できるのか、また確定したラベルがどのようなデータ形式（CSV、ディレクトリ構造、`.pkl` 等）でフェーズ2へ引き渡されるのかが定義されていません。
  * **「100:2」という数値の意味の曖昧さ**: [README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L10), [L35](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L35), [L43](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L43) で繰り返される「100:2の不均衡」が、単なる比率の例示（正常:不良 = 50:1）なのか、実データ枚数（正常100枚・不良2枚）なのかが明記されていません。もし実データ枚数が不良2枚であれば、不良モードA/Bへの分割かつ学習データ・検証データ（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L49) の検証データ Recall 算出用）への分割が成立しないため、想定データ規模・データ分割（Train/Validation split）の前提条件を明記する必要があります。
  * **Step 1 の「最適な次元削減」の基準未定義**: [README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L30) の「PCA（主成分分析）等による最適な次元削減」について、何次元（あるいは累積寄与率何%）をもって「最適」とするのかが定義されていません。

---

#### 6. [Nit] 表記ミスおよび未レンダリングのLaTeX記法
* **Severity**: ⚪ Nit
* **Affected Scope**: `3.3. 空間の永続保存 ＆ 3軸多角評価システム（Step 6）`（[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L58)）
* **Discrepancy Description & Rationale**:
  * [README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md#L58) に `PCA(300次元) \(\rightarrow \) UMAP(3D)`（標準的なGitHub Flavored Markdownではレンダリングされない `\(\rightarrow \)` 記法）および「くるくくる回転」（「くるくる回転」のタイポ）が含まれています。
