# 開発・設計思考ノート (note.md)

本ドキュメントは、[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md) に定義された「弱教師あり異常検知に向けた『前処理自動分類 ＆ 動的Metric-Learning』統合システム」の要件精査、数理設計、アーキテクチャ決定、および実装に至るまでの思考プロセスを段階的に記録する。

---

## 1. プロジェクトの目的と背景整理

製造業の外観検査において、初期収集データは圧倒的多数の「良品（正常）」と極少数の「不良品（異常）」（例：`100:2` の不均衡、かつ不良モードが未分類）で構成される。

これを単一の2値分類（OK/NGの1軸）として学習させると、以下の構造的破綻が生じる：
1. **多数派クラスへの崩壊**: 損失関数が良品側に支配され、不良の見落とし（False Negative）が発生する。
2. **多峰性不良の潰れ（モード崩壊）**: 「傷」「汚れ」「打痕」「異物」など空間上で異なる方向に分布する多様な不良を無理やり単一の「NGクラス」に押し込めると、特徴空間が歪み、良品との境界が曖昧になる。

本システムは、この問題を**2段階のパイプライン**で解決する：
* **フェーズ1（前処理自動分類）**: 生データの分布（100:2）を歪めずに既製バックボーン（EfficientNet / ViT）＋ PCA ＋ シルエットスコア自動選定K-meansでクラスタリングし、人間が最小限の確認で「良品」「不良モードA」「不良モードB」…へと**疑似多クラス化**する。
* **フェーズ2（本番異常検知モデル学習）**: 128次元の単位超球面（L2正規化空間）上で疑似多クラス MS-Loss（Multi-Similarity Loss）を適用し、さらに各不良サンプルに肉薄する境界線上の良品（Top-K ハードネガティブ）に対してカリキュラム制御付きの斥力ペナルティを動的に課すことで、良品クラスタを多方向から立体的に削り出す。

---

## 2. 理論的支柱（参考論文とシステム要件の対応関係）

| # | 論文 | コア概念 | 本システムへの接続・設計上の示唆 |
| :--- | :--- | :--- | :--- |
| 1 | **Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning** (CVPR 2019) | General Pair Weighting (GPW): 自己類似度・正例相対類似度・負例相対類似度の3種を同時に重み付けするペアマイニング＆損失関数 | **Step 4 & Step 5 の数理コア**。ただし原著MS-Lossは「クラス間が対称」な一般的なMetric Learningを前提としているため、本システムの「良品 vs 多モード少数不良（100:2）」という非対称構造に合わせた重み付け・Top-K境界良品反発項の定式化が必要。 |
| 2 | **Weakly Supervised Anomaly Detection: A Survey** (arXiv:2302.04549) | 不完全・少数異常ラベル環境下での異常検知（WSAD）の体系化 | **問題設定の根拠**。少数のラベル付き異常データからいかに汎化境界を引くか、また未知の微細異常への感度をどう維持するかの評価設計指針。 |
| 3 | **From Anomaly Detection to Defect Classification** (U2S-CNN) | 教師なし特徴抽出 → クラスタリング → 最小限の人間介入 → 教師あり分類への段階的移行 | **フェーズ1（Step 1〜3）からフェーズ2（Step 4〜6）への接続根拠**。初期のクラスタリング品質が後段の教師ありMetric Learningの上限を決めるため、クラスタリング時の特徴量正規化と外れ値（少数不良）の埋没防止が重要論点となる。 |
| 4 | **Few-Shot Anomaly-Driven Generation for Anomaly Classification and Segmentation** (AnoGen, ECCV / arXiv:2505.09263) | Few-shot実異常からの疑似異常生成と弱教師あり境界制御 | **Step 4 のデータ拡張設計および将来拡張**。単純な幾何・光学変換（回転・反転・輝度調整）が「良品を不良に見せる／不良の特徴を壊す」リスクへの配慮と、将来的な生成ベース拡張を差し替え可能にするインターフェース設計。 |

---

## 3. 要件定義書（README.md）に対する初期の技術的論点・ストレスチェック

実装に入る前に、EGM Goldfish Review プロトコルを通じて [README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md) の仕様を多角的に検証する。初期読解の段階で浮上している重要論点は以下の通り：

1. **フェーズ1（Step 2-3）における「100:2不均衡下でのK-meansとシルエットスコア」の数理的罠**
   * 全データ（良品100：不良2）に対してそのままK-meansを実行し、全体の平均シルエットスコアで $k$ を選ぶと、サンプル数の98%を占める「良品クラスタの分割・凝集」だけでシルエットスコアが決定され、わずか2%の不良データが良品クラスタの端に吸収されて独立クラスタを形成できないリスクがある。
   * 「全データを一括クラスタリングするのか」、それとも「良品中心からの距離等で粗く異常候補を分離・あるいは不良候補群に対して不良モード分割（Sub-clustering）を行うのか、もしくはクラスタ数を大きめに取って人間がマージするのか」を明確化する必要がある。
2. **フェーズ2（Step 4-5）における「ミニバッチ内の不良サンプル枯渇問題」**
   * 100:2 の比率のまま通常のランダムミニバッチサンプリングを行うと、バッチサイズ32や64の中に不良サンプルが0〜1個しか入らないステップが多発し、MS-Lossの正例ペア（同一不良モード内のペア）が成立しない。
   * データ拡張（Up-sampling）と Class-Balanced Batch Sampler（P-Kサンプリング：各バッチに必ず良品と複数の不良モードを含める制御）の組み合わせをどう設計するか。
3. **Step 5「Top-K境界良品の動的押し戻し」と MS-Loss の関係性**
   * MS-Loss自体が $\beta$ パラメータと閾値 $\epsilon$ によるハードネガティブ重み付けを内包している。Step 5の「境界線上の良品 Top-K の動的押し戻し」を、(A) MS-Loss内部の負例マイニング範囲 $K$ とマージン $\lambda$ / スケール $\beta$ の動的スケジューリングとして実装するのか、(B) MS-Lossとは別の明示的な境界斥力正則化項（Boundary Repulsion Loss）として加算するのか、数式レベルでの定義が必要。
4. **Step 6「推論・判定ロジック（異常スコアの定義）」と「PCA(300次元)の次元整合性」**
   * 学習後の128次元空間において、検証・本番データに対する「OK/NG判定（およびRecall/F1算出）」を何の距離指標（良品プロトタイプ中心とのコサイン距離、k-NN距離、あるいは各不良モード中心との相対距離）と閾値で決定するかが未定義。
   * また、Step 6の補助基準に「PCA(300次元) → UMAP(3D)」とあるが、射影ベクトルが128次元である場合、300次元へのPCAは数学的に不可能（入力次元 $\ge$ PCA次元である必要がある）であり、GAP特徴量（例：EfficientNet-B3の1536次元）に対してPCA(300次元)をかけるのか、128次元射影ベクトルを直接UMAPにかけるのかを整理する必要がある。

---

## 4. EGM デザインレビュー進行ログ

* **Step 1 (Intake & Context)**: 対象ドキュメント [README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md)（全68行）、追加コンテキストとして上記4論文を登録。
* **Step 2 (Preset Selection)**: `Interactive review with all review rubrics`（段階的インタラクティブレビュー）を選択。
* **Step 3 (Semantic Split Discovery)**: 現在の [README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md) には高レベルの機能・非機能要件（1〜68行）のみが記載されており、`Detailed Implementation Plan`（詳細実装計画）セクションはまだ存在しないため、`FeatureRange = 1..68`、`DetailedRange = N/A` として設定。Stage 5〜8 で要件・基本設計を徹底検証した上で、詳細実装設計（Stage 9対象）へと展開する。
* **Stage 5 (Initial Zoom Out Review)**: 完了。監査レポートを [zoom_out_report.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/zoom_out_report.md) に出力。
  * **Stage 5 で抽出された核心的インサイト（設計ブラッシュアップの要点）**:
    1. **128次元超球面における異常判定（推論・閾値）の定式化（Blocker）**: 128次元空間を維持したまま検証Recall/F1や本番OK/NG判定を出すための「距離スコア関数（良品プロトタイプ中心または良品 $k$-NN からのコサイン距離）」と「閾値 $\tau$ の校正方法」を仕様に明記する必要がある。
    2. **100:2 不均衡下でのフェーズ1クラスタリング構造（Blocker）**: 98%が良品の生データ全体にそのまま PCA + K-means + シルエットスコア最大化をかけると、主成分もシルエットスコアも良品内の変動に支配され、2%の不良が良品クラスタに吸収される。
       * *解決策*: ①高密度な「良品コア」を局所密度（$k$-NN距離等）で先に分離抽出（Extraction）し、②残りの低密度テール（境界良品＋少数不良）に対してPCA＋シルエット最適化クラスタリングを行う2段階方式（またはNG候補群に対するサブクラスタリング）とすることで、「生データの分布を歪めない」という思想を守りつつ少数不良モードを確実に分離できる。
    3. **ミニバッチ内 Top-K の限界と不良限定データ拡張のショートカット学習防止（Major）**:
       * ミニバッチ内だけで Top-K の境界良品を探すと、バッチ内の良品母数が少なすぎて真のグローバル境界良品を捉えられないため、全データの128次元ベクトルを軽量キャッシュする **Cross-Batch Memory Bank（またはグローバル境界マイニング）** を導入する。
       * 不良画像だけに回転・輝度変換をかけると「変換アーティファクト＝不良」と誤学習するため、拡張オペレータ自体は良品・不良で対称にし、サンプリング頻度（Batch Sampler）とLoss重みで不均衡を制御する。
    4. **「良品中心距離」センサーと「境界良品押し戻し」アクチュエータの整合（Major）**: 単一の良品重心からの距離だけでなく、Step 5が直接押し戻す「不良〜最近傍境界良品間の最小コサイン距離（Boundary-to-Defect Margin）」をカリキュラム制御・評価指標に組み込む。
    5. **128次元射影ベクトルに対する `PCA(300次元) -> UMAP(3D)` の次元矛盾解消（Major）**: 128次元射影空間の可視化は直接 `UMAP(3D)` を適用する（PCAは高次元GAP特徴量を扱う場合のみ適用）。

---

## 5. Stage 5 指摘事項に対する設計決定と技術的ストレスチェック

ユーザー回答に基づき、基本方針として [zoom_out_report.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/zoom_out_report.md) の推奨対策を採用しつつ、以下の3点の具体仕様を確定・検証した。

### 5.1. コサイン類似度に基づく2値判定閾値（学習時デフォルト $\tau_{\text{sim}} = 0.0$ ＋ ユーザー可変UI）
* **確定仕様**:
  * 128次元L2正規化空間におけるコサイン類似度は $[-1, 1]$ の範囲をとるため、学習時のデフォルト判定閾値を **$\tau_{\text{sim}} = 0.0$（直交＝コサイン距離 $1.0$）** とし、学習結果（分布・PR曲線・ヒートマップ等）を確認しながらユーザーがUI/設定で閾値を任意に変更できる機構とする。
* **技術的ストレスチェック（反証・エッジケース対策）**:
  * **学習初期の正領域集中リスク**: 事前学習済みバックボーンの特徴空間では、同一製品カテゴリの画像は良品・不良品ともに狭い正のコーン領域（例：類似度 $+0.4 \sim +0.9$）に密集しやすい。もしMS-Lossのアンカー・マージン設定（$\lambda$）が $+0.5$ 付近のままだと、学習が進んでも不良品が類似度 $0.0$ 未満（負の半球）まで押し出されず、閾値 $0.0$ 固定では学習中ずっと `Recall = 0.0` と判定され、Step 5のカリキュラムスケジューラーが発火しないリスクがある。
  * **実装上の解決策**:
    1. MS-LossおよびStep 5の反発マージン設計において、良品中心と不良間の目標類似度マージンを $0.0$ 以下へ押し込む設定を可能にする（あるいは類似度スコアを中心化・スケーリング可能にする）。
    2. カリキュラムスケジューラーのトリガーが初期エポックでデッドロックしないよう、学習時の閾値パラメータ（デフォルト `0.0`）を設定変更可能にしつつ、連続値である「境界良品〜不良間の最小コサイン類似度（距離）」との複合条件でスケジューラーを駆動させる。

### 5.2. フェーズ1における2段階クラスタリングの採用
* **確定仕様**:
  * 100:2 の不均衡生データに対し、直接全体PCA＋K-meansをかけるのではなく、推奨された **2段階クラスタリングパターン** を採用する。
    1. **Stage 1（高密度良品コア抽出）**: 事前学習バックボーン特徴空間上の局所密度（$k$-NNコサイン距離等）に基づき、大勢を占める高密度な「良品コア」を分離抽出する。
    2. **Stage 2（低密度テール領域のPCA＋シルエット自動選定K-means）**: 残りの低密度領域（境界良品＋少数不良群）に対してPCAとシルエットスコア最大化K-meansを実行し、少数不良モード（傷・汚れ等）が主成分とクラスタ分割に明確に現れるようにする。

### 5.3. Cross-Batch Memory Bank（エポック毎のメモリ解放設計）＋ 可変 Top-K（件数 / 割合指定）
* **確定仕様**:
  * ミニバッチサイズの制約を超えてグローバルな境界良品を特定するため **Cross-Batch Memory Bank (XBM)** を実装する。
  * **OOM Kill 防止設計**:
    1. メモリバンクに格納するベクトルは必ず計算グラフから切り離したテンソル（`tensor.detach().cpu()` または `.detach()`）とし、PyTorch Autograd グラフの残留によるメモリリークを厳格に遮断する。
    2. **各エポック終了時にメモリバンクのバッファを明示的に破棄・解放**（`memory_bank.clear()`, `del`, `gc.collect()`, `torch.cuda.empty_cache()` / `torch.mps.empty_cache()`）し、エポックを跨いだインプロセスメモリの肥大化を防止する。
  * **Top-K の柔軟な指定（件数 / パーセンテージ両対応）**:
    * `Top 20` は固定値ではなく、バックエンド側で **絶対件数指定（例：`top_k_count = 20`）** と **母数に対する割合指定（例：`top_k_ratio = 0.10` ＝ 上位10%）** の双方を切り替え・スケジュール変更できる設計とする。

---

## 6. Stage 6 (Comprehension Test) 進行ログと抽出された設計課題

* **Stage 6 (Comprehension Test)**: 完了。監査レポートを [comprehension_report.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/comprehension_report.md) に出力。
* **Stage 6 で新たに明確化が必要となった設計論点**:
  1. **疑似多クラス MS-Loss における「不良モード間（A vs B）」と「良品 vs 不良モード」の非対称設計**:
     * 通常の多クラス MS-Loss は全クラス間を対称に引き離すが、本システムの主目的は「良品領域から全不良モードを全方位に押し出しつつ、各不良モードが1点に潰れないようにすること」である。
     * したがって、①各不良モード内（Positive pairs）の凝集、②良品 vs 各不良モード間（Primary Negative pairs）の強い反発（＋Step 5のTop-K境界良品反発）、③異なる不良モード間（A vs B: Auxiliary Negative pairs）の適度な分離（球面上で異なる方向へ展開させる）という3種のペア関係を明示的に定式化する。
  2. **ネットワーク構造（GAP特徴量 → Projection Head → L2正規化128次元空間）とバックボーン学習範囲**:
     * `timm` バックボーン（EfficientNet-B3 / ViT）の Global Average Pooling (GAP) 出力（$\mathbb{R}^{D_{\text{backbone}}}$）を入力とし、MLP Projection Head（例：`Linear -> BatchNorm/LayerNorm -> GELU/ReLU -> Linear(128) -> L2Normalize`）を経て単位超球面 $\mathbb{S}^{127}$ 上の128次元射影ベクトルを出力する構造を明記する。
     * 少数の不良データでバックボーン全体を最初からフル学習すると過学習（Catastrophic Forgetting）を起こしやすいため、**「バックボーン Freeze（または後半ブロックのみ微調整／段階的 Unfreeze）＋ Projection Head 学習」** を選択可能にする設計が有効。
  3. **フェーズ1 → フェーズ2 のデータ引き渡しフォーマットとアノテーション粒度**:
     * クラスタ単位の一括ラベル付与（「正常」「不良モードA」「不良モードB」…）に加え、クラスタ内に混入した外れ値サンプルを個別にチェック・ラベル上書きできるUIとし、確定結果を構造化マニフェスト（CSV / JSON：画像パス、クラスタID、確定クラス名、OK/NGフラグ、Train/Val分割フラグ）としてフェーズ2へ渡す。
  4. **用語の統一**:
     * 見出しの「正解率/マージン連動」は、本文およびSection 4の評価基準に合わせて「再現率（Recall）/マージン連動」に統一する。

---

## 7. アプリケーション構成・インフラ設計（ユーザー要件の統合とREADME反映）

ユーザーからの追加アーキテクチャ要件に基づき、[README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md) 下部（Section 5〜7）に詳細数理仕様・3コンテナ構成・デプロイ手順を追記した。

1. **3コンテナ分離アーキテクチャ（ローカル Docker Compose 開発）**:
   * **`frontend` (Streamlit)**: UI描画とユーザー操作に特化。自身ではモデル推論やディスクI/Oを行わず、`api-backend` をRESTコールする。
   * **`api-backend` (FastAPI)**: データセット・アノテーション管理、評価指標（閾値 $\tau_{\text{sim}}$ 変更時のRecall/F1）の即時算出、および学習ワーカーへのジョブプロキシを担う軽量ゲートウェイ。
   * **`trainer-service` (FastAPI + PyTorch / `timm`)**: リソースを大量消費する Step 1〜3（特徴抽出・2段階クラスタリング）および Step 4〜6（MS-Loss学習・Cross-Batch Memory Bank・UMAP算出）を専用コンテナへ隔離。Dockerメモリ上限とエポック毎の明示的メモリ解放により、学習負荷がUIやAPIゲートウェイを巻き込まない設計とする。
2. **GCE + Terraform + GitHub Actions による CI/CD**:
   * インフラ（GCE VM、Persistent Disk、Google Artifact Registry、Firewall、Workload Identity IAM）を `infra/terraform/` でコード管理。
   * GitHub Actions にて PR時のCI（Ruff/Mypy/Pytest/Docker Build）と `main` マージ時のCD（GARへのPush → Terraform Apply → GCE上での `docker compose pull & up -d`）を一貫自動化する手順を Section 7 に定義。
3. **EGM Review セマンティック境界の再計算**:
   * [README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md) に `## Detailed Implementation Plan` が追加されたため、以下の通り範囲を更新して **Stage 7 (Critic Review)** を実行する：
     * `FeatureRange`: `Lines 1-110`（Section 1〜5：機能・数理要件仕様）
     * `DetailedRange`: `Lines 111-332`（Section 6〜7：アプリケーションアーキテクチャ＆デプロイ実装計画）

---

## 8. Stage 7 (Critic Review) 監査結果と仕様統合のポイント

* **Stage 7 (Critic Review)**: 完了。監査レポートを [critic_report.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/critic_report.md) に出力。
* **抽出された重要課題と解決アプローチ**:
  1. **旧仕様（Section 1〜4）と確定版仕様（Section 5）の一本化**:
     * 追記形式によって残っていた旧記述（単段階K-means、不良のみのデータ拡張、`PCA(300D)->UMAP(3D)`、毎エポック全量`.pkl`保存）を確定仕様に完全統合し、ドキュメント内の矛盾をゼロにする。
  2. **Feature Spec Standard 準拠（Goals & Non-Goals / Alternatives Considered / Risks & Mitigations）**:
     * なぜ単段階K-meansや1次元2値分類、標準MS-Loss単独ではなく「2段階密度分離クラスタリング＋非対称マルチクラスMS-Loss＋XBM」を採用したのかの代替案比較表と、リスク・緩和策セクションを明文化する。
  3. **微小欠陥（局所異常）の特徴希釈対策（GAP / GeM Pooling ＆ Stage 1 大域密度基準）**:
     * 面積の小さい傷・汚れが単純な空間平均（GAP）で背景に埋もれないよう、特徴抽出層で **GAP / GeM (Generalized Mean) Pooling / Avg+Max Pool** を選択可能にする。
     * Stage 1 の良品コア抽出では、微小不良の局所クラスタを高密度と誤認しないよう、全体空間メディアン（ロバスト中心）からの距離＋大域的近傍数 $k$ を用いる。
  4. **XBM のエポック冒頭ガード条件 $K_{\text{eff}}(t)$ とウォームアップ**:
     * エポック終了時のメモリ解放（`clear()`）に伴い、次エポック開始直後の数バッチでは蓄積良品数 $|\mathcal{M}_{\text{normal}}|$ が少ないため、$K_{\text{eff}}(t) = \min(K(t), |\mathcal{M}_{\text{normal}}|)$ で安全にクリップし、初期ウォームアップ期間を設けて特徴ドリフトを防ぐ。
  5. **非対称 $L_{\text{MS}}$ の完全数式化とカリキュラム進捗度 $p(t)$ のデッドロック防止**:
     * 学習時デフォルト閾値 $\tau_{\text{sim}}=0.0$ において、学習初期に Recall@0.0 が 0 でもカリキュラムが停止しないよう、連続指標である境界最小コサイン距離 $d_{\min}(t)$ とエポック進行度を組み合わせた進捗関数 $p(t)$ を定義する。

---

## 9. Stage 8 (Intermediate Zoom Out Review) 監査結果：実装アーキテクチャの重要改善点

* **Stage 8 (Intermediate Zoom Out Review)**: 完了。監査レポートを [intermediate_zoom_out_report.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/intermediate_zoom_out_report.md) に出力。
* **3コンテナ構成およびXBM学習に関する極めて重要な実装設計ポイント**:
  1. **Cross-Batch Memory Bank における「境界良品そのものへの勾配付与（双方向反発＋Hard-Negative優先サンプリング）」**:
     * メモリバンク内のベクトルは `detach()` されているため、不良アンカーからの片方向損失だけでは「境界良品側のベクトルを良品中心へ押し戻す勾配」が流れない。
     * **解決策**: ①メモリバンクに良品・不良双方の `detach` 済みベクトルを保持し、バッチ内の不良アンカーだけでなくバッチ内の境界良品アンカーからも斥力勾配を流す**双方向反発損失**とするとともに、②前ステップで特定された Top-$K$ 境界良品インデックスを Batch Sampler が優先的に次バッチへ組み込む**Hard-Negative 優先バッチ編成**を実装する。
  2. **`api-backend` と `trainer-service` 間の `.pkl` データ契約（PyTorch非依存の保証）**:
     * `api-backend` は軽量化のため `torch` をインストールしない。したがって、`trainer-service` が `.pkl` を保存する際、`torch.Tensor` のまま保存すると `api-backend` 側の `pickle.load()` が `ModuleNotFoundError: No module named 'torch'` でクラッシュする。
     * **解決策**: `.pkl` に保存するすべての配列（128次元射影ベクトル、GAP特徴量、類似度、UMAP 3D座標）は必ず `tensor.detach().cpu().numpy().astype(np.float32)` で **Pure NumPy `ndarray` および Python プリミティブ型** に変換し、かつ書き込み中の競合を防ぐため一時ファイル（`.tmp`）出力後に `os.replace()` でアトミック保存する。
  3. **`trainer-service` の非同期ジョブ実行とイベントループ保護**:
     * 重い学習処理中に FastAPI の `GET /internal/jobs/{job_id}` や `GET /health` がハングアップしないよう、ジョブ実行ワーカーをイベントループから分離し、進捗ステータスをアトミック更新して常時即時応答させる。

---

## 10. Stage 9 (Implementation Readiness Test) 監査完了：実装直前の全契約確定

* **Stage 9 (Implementation Readiness Test)**: 完了。監査レポートを [implementation_readiness_report.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/implementation_readiness_report.md) に出力。
* **実装時に一切の迷い（Zero-Decision）をなくすために確定した技術契約**:
  1. **全 REST API エンドポイントと Pydantic スキーマの確定**:
     * `api-backend` (Port 18000): `/health`, `/api/v1/datasets`, `/api/v1/datasets/generate-synthetic`（サンプル合成画像生成プロキシ）, `/api/v1/clusters`, `/api/v1/annotations`, `/api/v1/jobs/clustering`, `/api/v1/jobs/training`, `/api/v1/jobs/{job_id}`, `/api/v1/evaluations`, `/api/v1/evaluations/{epoch}?tau_sim=0.0`, `/api/v1/inference`
     * `trainer-service` (Port 18001): `/health`, `/internal/datasets/generate-synthetic`, `/internal/jobs/clustering`, `/internal/jobs/training`, `/internal/jobs/{job_id}`, `/internal/inference`
  2. **共有ボリューム `/app/data` のデータスキーマ確定**:
     * `clustering_result.json`（Stage 1/2 のクラスタリング結果、シルエットスコア推移、良品コア判定フラグ、2D/3Dプレビュー座標）
     * `annotations.json`（確定クラスID/ラベル名、OK/NG 2値フラグ、層化Train/Val分割フラグ）
     * `epoch_{epoch:04d}_light.pkl` & `epoch_{epoch:04d}_full.pkl`（すべて Pure NumPy `float32` 配列＋Python標準型でアトミック保存し、PyTorch非依存の `api-backend` から即座にロード可能にする）
  3. **コアMLモジュールのガード条件とデフォルトハイパーパラメータ**:
     * PCA: `n_components = min(64, n_samples - 1, n_features)` 後に累積寄与率95%でスライス。
     * K-means シルエット探索: $k \in [2, \min(K_{\max}, N_{\text{tail}} - 1)]$。
     * UMAP(3D): `n_neighbors = min(15, max(2, N - 1))` で少数サンプル時のスペクトル初期化エラーを回避。
     * 非対称 MS-Loss + XBM: $\alpha=2.0, \beta=50.0, \lambda=0.5, \epsilon=0.1, w_{\text{defect}}=2.0, w_{\text{aux}}=0.5, \beta_{\text{rep}}=20.0$、各エポック終了時に `memory_bank.clear()` + `gc.collect()` + デバイスキャッシュ解放を実行し、エポック内の実効 $K$ を $K_{\text{eff}}(t) = \min(K(t), |\mathcal{M}_{\text{normal}}|)$ でクリップ。

---

## 11. アプリケーション実装完了記録（3コンテナ＋Terraform＋GitHub Actions）

EGM Goldfish Review（Stage 5〜9）およびユーザー設計決定をすべて反映し、以下のモジュール群を実装・検証した。

### 11.1. コアMLアルゴリズム実装 (`trainer/app/core/`)
1. **[synthetic_data.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/synthetic_data.py)**:
   * 100:2の不均衡環境をローカルですぐに再現・検証できるよう、金属ヘアライン／ウェハ表面テクスチャの良品画像（`normal`）と、3種の異なる不良モード（`defect_scratch`：線状傷、`defect_stain`：しみ・汚れ、`defect_dent`：打痕・ピット）を合成生成するジェネレータを実装。
2. **[feature_extractor.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py)**:
   * `timm` バックボーン（`efficientnet_b3`, `vit_base_patch16_224`, `resnet18`、およびオフライン環境用 `lightweight_cnn` フォールバック）と、微小欠陥の希釈を防ぐ **GeM (Generalized Mean) Pooling / Avg+Max Pooling / GAP** を実装。
   * 128次元単位超球面への射影ヘッド（`MetricProjectionModel`）および良品・不良の双方に同条件で適用する対称データ拡張（`build_image_transforms`）を実装。
3. **[two_stage_cluster.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/two_stage_cluster.py)**:
   * **Stage 1**: 大域ロバストメディアン中心とのコサイン類似度＋$k$-NN局所密度により、高密度良品コア（デフォルト上位85%：`cluster_id = 0`）を分離抽出。
   * **Stage 2**: 低密度テール領域に対して PCA（上限64次元・累積寄与率95%）とシルエットスコア自動選定 K-means を実行し、少数不良モードを独立クラスタ（`cluster_id = 1..k*`）として分離。
4. **[memory_bank.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/memory_bank.py)**:
   * `CrossBatchMemoryBank`: 計算グラフから完全に切り離した（`detach().cpu()`）128次元ベクトルを保持し、**絶対件数指定（`top_k_mode="count"`：例 Top 20）** と **母数割合指定（`top_k_mode="ratio"`：例 上位10%）** の両方で不良に肉薄する境界良品を抽出。
   * **OOM Kill 防止**: 各エポック終了時に `release_epoch_memory()`（`clear()`, `gc.collect()`, `empty_cache()`）を実行し、エポック毎の解放件数・残存件数（0件）をジョブ履歴に記録。
5. **[ms_loss.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/ms_loss.py) & [scheduler.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/scheduler.py)**:
   * `AsymmetricMultiSimilarityLoss`: 同一クラス凝集（$\mathcal{P}_i$）、良品 vs 不良の強反発（$\mathcal{N}_i^{\text{primary}}$、重み $w_{\text{defect}}$）、異種不良モード間の分離（$\mathcal{N}_i^{\text{aux}}$、重み $w_{\text{aux}}$）に加え、XBM を用いた双方向 Top-K 境界反発損失 $L_{\text{repel}}$ を実装。
   * `CurriculumPenaltyScheduler`: 検証 Recall と境界最小コサイン距離 $d_{\min}^{\text{boundary}}$ に連動して反発係数 $\gamma(t)$、マージン $m(t)$、Top-K 対象数 $K(t)$ を初期から後期へ自動制御。
6. **[evaluator.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/evaluator.py)**:
   * 学習データの良品群から算出した良品プロトタイプ中心 $\mu_{\text{normal}}$ とのコサイン類似度 $s(x) \in [-1, 1]$ に対し、学習時デフォルト閾値 $\tau_{\text{sim}} = 0.0$ で Recall / Precision / F1 / 混同行列を算出。
   * 128次元射影ベクトルに対する直接 3D UMAP 投影と、PyTorch 非依存の 2階層 Pure NumPy `.pkl` アトミック保存を実装。

### 11.2. 3コンテナ・インフラ・CI/CD 実装
* **Trainer Service (`trainer/`)**: [train_jobs.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py), [main.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/main.py)
* **Backend API Gateway (`backend/`)**: [datasets.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/backend/app/routers/datasets.py), [jobs_proxy.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/backend/app/routers/jobs_proxy.py), [evaluation.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/backend/app/routers/evaluation.py), [main.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/backend/app/main.py)
* **Streamlit Frontend (`frontend/`)**: [1_clustering_annotation.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/1_clustering_annotation.py), [2_training_monitor.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/2_training_monitor.py), [3_space_evaluation.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/3_space_evaluation.py)
* **Docker Compose & IaC / CI-CD**: [docker-compose.yml](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/docker-compose.yml), [docker-compose.prod.yml](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/docker-compose.prod.yml), [main.tf](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/infra/terraform/main.tf), [ci.yml](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/.github/workflows/ci.yml), [cd-gce.yml](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/.github/workflows/cd-gce.yml)

---

## 12. 運用環境調整ログ（ポート番号 18000番台化 ＆ GCPリージョン `us-central1` 統一）

1. **Docker ポート番号の 18000番台への変更（8000番台競合回避）**:
   * ローカルホスト上の他 Docker コンテナとのポート競合（`cannot bind tcp port :8001: address already in use`）を回避するため、全3コンテナの待受・公開ポートを **18000番台** に統一した：
     * `api-backend`: `18000`（旧 `8000`）
     * `trainer-service`: `18001`（旧 `8001`）
     * `frontend`: `18501`（旧 `8501`）
   * ビルド済みイメージでも再ビルド不要で即座に新ポートで起動できるよう、[docker-compose.yml](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/docker-compose.yml) に `command:` オーバーライドを明記するとともに、各 `Dockerfile`・設定ファイル（[config.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/backend/app/config.py), [api_client.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/api_client.py), [main.tf](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/infra/terraform/main.tf)）を同期更新した。
2. **Google Cloud リージョンの `us-central1` 統一**:
   * [.env.example](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/.env.example)、[docker-compose.prod.yml](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/docker-compose.prod.yml)、[variables.tf](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/infra/terraform/variables.tf)、および [README.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/README.md) におけるデフォルトリージョンを `us-central1`（ゾーン `us-central1-a`）に統一した。
3. **フォルダ画像アップロード ＆ サブフォルダ名の学習クラス名利用機能**:
   * **UI ([1_clustering_annotation.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/1_clustering_annotation.py))**:
     * 親フォルダ直接選択（`accept_multiple_files="directory"`）、ZIP圧縮フォルダ（`.zip` 自動展開）、およびサブフォルダ単位の複数画像選択に対応。
     * アップロードされた相対パス（例: `dataset/<subfolder>/001.png` または `<subfolder>/001.png`）から `<subfolder>` 名をクラス名として自動抽出し、枚数内訳プレビューを表示した上でチャンク分割アップロードを実行。
   * **Backend & Trainer ([datasets.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/backend/app/routers/datasets.py), [train_jobs.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py), [two_stage_cluster.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/two_stage_cluster.py))**:
     * `/app/data/raw/<subfolder_class_name>/<file_name>` として保存し、2段階クラスタリング実行時にサブフォルダ名を `subfolder_class` / `ground_truth_hint` および `detected_classes` として `clustering_result.json` に記録。
     * 任意のサブフォルダ名（`normal`, `good`, `OK`, `良品` 等）を良品クラス（`label_id=0, is_defect=False`）に指定でき、残りの全サブフォルダ名を不良モード（`label_id=1..C, is_defect=True`）として `annotations.json` に保存（クラスタ確認後の保存と、サブフォルダからの直接保存の両方に対応）。
   * **ホットリロード用バインドマウント ([docker-compose.yml](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/docker-compose.yml))**:
     * `./trainer/app:/app/app`、`./backend/app:/app/app`、`./frontend/app:/app/app`、および `./data:/app/data` を全コンテナにマウントし、Dockerイメージの再ビルドなしでコード変更・画像プレビューが即時反映される構成とした。


