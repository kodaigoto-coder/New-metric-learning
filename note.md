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

---

## 13. 実データ（MVTec Bottle 292枚：`nomal` 229枚 / `anomaly` 63枚）学習時の不具合 RCA（根本原因分析）と解決策

実データ投入後の初回10エポック学習時に発生した ① `StreamlitDuplicateElementId` エラー、および ② 学習が進まない現象（`Recall=0.0`, `d_min=0.0`, 全サンプルのコサイン類似度が `0.99999` に縮退）について、`epoch_0001_full.pkl` 〜 `epoch_0010_full.pkl` のバイナリ解析により以下の根本原因を特定し修正した。

### 13.1. 根本原因①：Streamlit ポーリングループ内の `StreamlitDuplicateElementId`
* **原因**: [2_training_monitor.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/2_training_monitor.py) の `while True:` ポーリングループ（0.6秒間隔）において、次エポック完了前の同一 `history` に対して毎回同一パラメータの `px.line` 図を生成し、`key` 引数なしで `plotly_chart` を再描画していたため、Streamlit 内部の自動生成 Element ID が衝突した。
* **対策**:
  1. `len(history)` が更新されたタイミング（またはジョブ完了時）のみグラフコンテナを再描画するように変更。
  2. 全画面（Page 1〜3）のすべての `plotly_chart` および `dataframe` に一意の `key`（例: `key=f"train_loss_chart_{render_seq}"`）を明示付与。

### 13.2. 根本原因②：高次元特徴空間の「共通背景バイアスによる超球面鞍点縮退（Cosine Similarity $\approx 0.99999$）」
* **原因**:
  1. **未中心化特徴量による単位超球面上の勾配消失（Saddle-Point Collapse）**:
     同一製品（ボトル）の検査画像群では、背景や製品輪郭に由来する共通平均ベクトル $\mu_h$ のノルムが個体差・欠陥差分 $\delta_i = h_i - \mu_h$ より圧倒的に大きい（$\|\mu_h\| \gg \|\delta_i\|$）。サンプル間での中心化・標準化を行わずに L2 正規化すると、全サンプルの射影ベクトル $z_i, z_k \in \mathbb{S}^{127}$ が同一方向（$S_{ik} = z_i^\top z_k \approx 0.999999$）に集中する。
     単位超球面上のコサイン類似度の勾配は
     $$\frac{\partial (z_i^\top z_k)}{\partial u_i} = \frac{1}{\|u_i\|}\left(z_k - (z_i^\top z_k) z_i\right)$$
     であるため、$z_i \approx z_k$（$S_{ik} \to 1$）では接空間射影 $z_k - S_{ik} z_i \to 0$ となり、**勾配が約 $10^{-4}$ に消失して学習が完全にロック**していた。
  2. **単一 `anomaly` フォルダ内の多モード不良に対する全ペア正例引力（Mode Pulling across Normal Core）**:
     `anomaly/` フォルダ内に複数種類の不良（`broken_large`, `broken_small`, `contamination`）が一括格納されて単一ラベル `label_id=1` となった場合、全 `label_id=1` ペアに正例引力をかけると、超球面上で反対方向に位置する異種不良同士が引き合い、その中点にある良品コア（$\mu_{\text{normal}}$）へと引き戻される。
  3. **`pretrained=False` 初期設定と空間グリッド特徴の不足**:
     未初期化の40層CNNは定数バイアスに近い出力となり、かつ統計特徴が空間グリッド（8×8パッチ構造）を保持していなかったため局所欠陥の信号が希釈されていた。
* **対策**:
  1. **データセット特徴量キャリブレーション（`calibrate_feature_stats`）と `BatchNorm1d` の導入 ([feature_extractor.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/feature_extractor.py))**:
     学習開始時に全画像の特徴平均 $\mu_h$・標準偏差 $\sigma_h$ をバッファ（`feat_mean`, `feat_std`）に記録して共通背景成分を完全に除去（$(h - \mu_h)/(\sigma_h + \epsilon)$）し、さらに空間グリッド記述子（8×8パッチRGB・6×6局所分散・6×6エッジ勾配）を追加。
  2. **2段階クラスタリングにおけるロバスト・メディアン中心化 ([two_stage_cluster.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/two_stage_cluster.py))**:
     特徴量が狭いコーン領域に密集している場合、自動的にロバスト中央値（メディアン）を減算・標準化してから Stage 1 密度スコアと Stage 2 PCA + K-means を実行。
  3. **局所 $k$-近傍正例マイニング（Local Positive Mining）と数値安定化 `logsumexp` ([ms_loss.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/ms_loss.py))**:
     不良アンカーの正例引力をバッチ内の最近傍 $k_{\text{pos}}=3$ 件に限定することで、単一 `anomaly` フォルダに複数不良モードが混在していても球面上の異なる方向へサブクラスタを形成させつつ、良品プロトタイプ中心から $\tau_{\text{sim}} < 0.0$（負の半球）へ強力に押し出すよう改善。

---

## 14. フェーズ1 クラスタリング仕様の刷新（良品クラス固定保持 ＋ 異常クラス限定サブクラスタリング）

* **背景・課題**:
  当初の Stage 1（密度分位 $\tau_{\text{core}}=0.85$ による良品コア推測）は「OK/NG の区別すら存在しない完全未ラベル単一フォルダ」を前提としていたため、`nomal`（229枚）と `anomaly`（63枚）のサブフォルダに分かれたデータを投入した際にも全データを85%:15%で切ってしまい、良品と異常が混ざって分割される問題があった。
* **確定仕様と実装 ([two_stage_cluster.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/two_stage_cluster.py), [train_jobs.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/routers/train_jobs.py), [1_clustering_annotation.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/1_clustering_annotation.py))**:
  1. **良品クラスの100%固定保持 (`Cluster 0`)**:
     良品サブフォルダ（`nomal` / `normal` / `good` 等）の画像は一切クラスタリングせず、全件そのまま `Cluster 0`（`is_normal_core = True`, `label_id = 0`）として固定する。
  2. **異常クラス限定の欠陥残差標準化 ＆ PCA + シルエット自動選定 K-means (`Cluster 1 .. k*`)**:
     異常サブフォルダ（`anomaly` 等）の画像のみを対象に、良品特徴分布（平均 $\mu_{\text{norm}}$・標準偏差 $\sigma_{\text{norm}}$）に対する標準化残差 $\tilde{h}_i = \text{L2Normalize}((h_i - \mu_{\text{norm}})/\sigma_{\text{norm}})$ を算出し、PCA ＋ シルエットスコア最大化 K-means によって `Cluster 1 .. k*`（デフォルト提案ラベル：`anomaly_cluster_1 .. k*`）へ自動細分化する。
  3. **フェーズ2 への疑似多クラス引き渡し (`annotations.json`)**:
     確定時には良品が `nomal` (`label_id=0`)、分割された各異常クラスタが `anomaly_cluster_1` (`label_id=1`), `anomaly_cluster_2` (`label_id=2`)... として保存され、フェーズ2の非対称マルチクラス MS-Loss で各不良サブクラスを128次元超球面上の異なる方向へ分離学習できる。


---

## 15. 次世代距離学習設計仕様（NAB-Sampler ＋ Multi-Proxy Anchor Loss ＋ 境界データ拡張）の数理検証・実装ログ

ペアベース MS-Loss において「不良クラス同士のサンプル間斥力に勾配が浪費され、最重要である『良品 vs 境界近傍の微細不良』の分離にパワーが集中しない」という課題に対し、新たに策定された **Customized Multi-Proxy Anchor Loss ＋ NAB-Sampler ＋ 境界データ拡張仕様** の理論的妥当性を検証し、コード完全準拠の数式仕様を [proxy_anchor_design_review.md](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/proxy_anchor_design_review.md) に記録した。

### 15.1. 提案仕様の理論的妥当性と実装定式化（5つの柱）
1. **Normal-Anomaly Balanced Sampler (`NAB-Sampler`：枯渇回避 ＆ 4層過学習防止機構)**:
   * ミニバッチサイズ $B$ のうち **50% を良品スロット（$B_{\text{norm}} = \lfloor B/2 \rfloor$）**、**残り 50% を不良スロット（$B_{\text{anom}} = B - B_{\text{norm}}$）** に固定する非対称層化サンプリングを行う。
   * **少数不良データの枯渇回避ロジック（[nab_sampler.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/nab_sampler.py#L103-L182)）**:
     1. **良品2周網羅に基づくエポックステップ数決定**: 1エポックのステップ数は少数不良ではなく全学習良品枚数 $|D_{\text{norm}}|$ を基準に $\text{num\_batches} = \max(\text{min\_steps}, 2 \times \lceil |D_{\text{norm}}| / B_{\text{norm}} \rceil)$（`train_jobs.py` では `min_steps=14`）と定義し、良品枠の最大35%（$\lfloor 0.35 B_{\text{norm}} \rfloor$）に XBM 境界ハード良品を優先注入しつつ、残り65%以上を巡回カーソル `norm_cursor` で走査（1周ごとに再シャッフル）する。
     2. **ステップ独立・モード別条件付き復元抽出**: 各ステップで不良モード（$C \le \lfloor B_{\text{anom}}/2 \rfloor$ または $C \le 4$ なら全 $C$ モード、それ以上の場合はランダムな $\lfloor B_{\text{anom}}/2 \rfloor$ モード）へ均等枠 $k_{\text{mode}}$ を割り当て、保有枚数が枠未満（$|D_{\text{def}}^{(c)}| < k_{\text{mode}}$）の希少モードにはステップ内復元抽出（`replace=True`）、十分なモードには非復元抽出（`replace=False`）を適用し、かつ全ステップ間で独立に再抽出することで不良データの途中枯渇を完全に防ぐ。
     3. **4層の丸暗記（過学習）防止**: ① Sample-to-Proxy 損失による自己ペア縮退排除、② 毎ステップの標準化特徴ガウシアン摂動 $\epsilon \sim \mathcal{N}(0, 0.015^2 I)$ ＋ $L_2$ 再正規化、③ 新鮮な良品と組み合わせる動的 Defect Transfer / Anomalous Mixup 合成、④ 平均 $1.0$ 正規化済み Class-Balanced Weight $W_c$ により少数不良の過学習を防ぐ。
2. **Sample-to-Proxy 型の Customized Multi-Proxy Anchor Loss への転換（[proxy_anchor_loss.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L199-L374)）**:
   * サンプル間の $O(B^2)$ ペア計算を廃止し、学習可能なクラス代表プロキシ（良品マルチプロキシ $\hat{p}_{\text{norm}}^{(m)} \in \mathbb{S}^{127}$ と不良モード別プロキシ $\hat{p}_{\text{def}}^{(c)} \in \mathbb{S}^{127}$、`AdamW` にて学習率 `2.5x` 倍）をアンカーとする `LogSumExp` 形式に刷新した：
     $$L_{\text{total}} = L_{\text{pos}} + \max(0.5, \gamma_{\text{repel}}(t)) \cdot L_{\text{repel}} + \lambda_{\text{ortho}} L_{\text{proxy-ortho}} \quad (\lambda_{\text{ortho}} = \max(0.2, w_{\text{aux}}))$$
   * $L_{\text{pos}} = L_{\text{norm-pos}} + L_{\text{def-pos}}$（$\alpha_{\text{pos}} = 16.0, \delta_{\text{pos}} = 0.65$、戻り値 `stats` では後方互換性のため `"loss_ms"` キーに格納）とし、バッチ内の「不良Aサンプル vs 不良Bサンプル」および「不良Aサンプル vs 不良Bプロキシ」の斥力を完全にゼロにした。
3. **良品マルチプロキシ化（Multi-Proxy Normal, デフォルト $M = 4$）**:
   * ロット差・照明ムラを含む良品の多峰性分布を $M$ 個のサブセンター $\{p_{\text{norm}}^{(m)}\}_{m=1}^M$ で吸収し、単一プロトタイプへの無理な凝集による過学習を防ぐ。
4. **3方向境界反発 $L_{\text{repel}}$ と Top-$K$ Hard Anomaly Penalty Boost（$\gamma_{\text{neg}} = 20.0, \delta_{\text{neg}}^{\text{eff}} = \min(\text{repel\_margin}(t), -0.15)$）**:
   * `loss_repel` は以下の3方向の算術平均として計算される：
     * **Direction A**: 全実不良＋合成ハードサンプル $\mathcal{B}_{\text{all\_neg}} = \mathcal{B}_{\text{def}} \cup \mathcal{B}_{\text{synth}}$ を全 $M$ 個の良品プロキシから反発させる。この際、良品プロキシへの最大類似度 $s_{\text{norm}}(z_i) = \max_m \langle z_i, \hat{p}_{\text{norm}}^{(m)} \rangle$ の上位 $K_{\text{eff}}$ 件（または上位割合 $q\%$）の危険不良に対し、`LogSumExp` 内の重み $\omega_i = \max(10^{-4}, \text{boost}_i \cdot W_{y_i} \cdot w_{\text{defect}})$ にて $\text{boost}_i = \lambda_{\text{hard}}$（デフォルト $3.0$）を乗算ブーストする。
     * **Direction B**: バッチ内の良品サンプル $z_j \in \mathcal{B}_{\text{norm}}$ を全 $C$ 個の不良クラスプロキシ $\hat{p}_{\text{def}}^{(c)}$ から重み $W_c$ 付きで反発させる。
     * **Direction C**: $\mathcal{B}_{\text{all\_neg}}$ の各サンプルを、Cross-Batch Memory Bank (`get_top_k_boundary_normals`) から取得したキャッシュ済み Top-$K$ 境界良品ベクトル群から直接反発させる。
5. **Defect Transfer / Anomalous Mixup / 平均正規化 Class-Balanced Weighting ($W_c$)**:
   * 良品画像への欠陥ブレンド（Defect Transfer）および良品比率 $\alpha \in [0.80, 0.90]$ の高難度補間（Anomalous Mixup）により境界近傍のハードネガティブ密度を高め、さらに有効サンプル数に基づく重みを全不良クラス平均が $1.0$ になるよう正規化した $W_c \propto (1-\beta)/(1-\beta^{\max(1, n_c)})$（$\beta = 0.99$）で少数不良モードの勾配を補正する。

### 15.2. 実装に組み込まれた3つの数理的セーフガード（崩壊防止条件）
1. **プロキシ間直交ヒンジ正則化 $L_{\text{proxy-ortho}}$（不良プロキシ1次元縮退 ＆ 良品サブセンター重なり防止）**:
   * サンプル間の「不良A vs 不良B」斥力は完全にゼロにしつつ、不良プロキシ同士（余弦上限 `0.20`）および良品サブセンター同士（余弦上限 `0.85`、重み `0.25`）にのみ以下の直交ヒンジ正則化を付加する：
     $$L_{\text{proxy-ortho}} = \frac{1}{\max(1, C(C-1))}\sum_{a \neq b}\max\!\left(0, \langle \hat{p}_{\text{def}}^{(a)}, \hat{p}_{\text{def}}^{(b)} \rangle - 0.20\right)^2 + \frac{0.25}{\max(1, M(M-1))}\sum_{m_1 \neq m_2}\max\!\left(0, \langle \hat{p}_{\text{norm}}^{(m_1)}, \hat{p}_{\text{norm}}^{(m_2)} \rangle - 0.85\right)^2$$
2. **合成ハードサンプルの「片方向反発（One-Way Boundary Repulsion）」とハイブリッド Defect Transfer**:
   * Mixup / Transfer 合成サンプル $\mathcal{B}_{\text{synth}}$ は **「$L_{\text{repel}}$（Direction A / C）の反発項のみに投入し、不良プロキシの正例引力項 $L_{\text{def-pos}}$ からは除外する」** 片方向反発として定式化し、不良プロキシが良品領域へ引きずり込まれる現象を防ぐ。
   * ピクセル空間では $16\times 16$ 平均プーリング $L_1$ 最近傍良品との平滑化残差ピーク座標 $(u^*, v^*)$ を中心とする 2D ガウシアン窓（`sigma_ratio=0.16, peak_opacity=0.82`）でインプレース転写し（[`synthesize_defect_transfer_images`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/boundary_augmentation.py#L30-L95)）、バックボーン凍結高速学習時には事前抽出ピクセル合成バンク（最大48枚×2から毎ステップ最大8件抽出）と特徴空間での上位30%残差次元スプライシング（`transfer_ratio=0.30, weight=0.85`）＋ $\alpha \in [0.80, 0.90]$ 特徴 Mixup（[`synthesize_boundary_gap_features`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/boundary_augmentation.py#L137-L195)）を併用する。
3. **Multi-Proxy 良品の K-means 初期化・不良プロキシの負半球初期化と推論スコア定義**:
   * 学習開始時に良品プロキシ $\{p_{\text{norm}}^{(m)}\}_{m=1}^M$ を良品特徴の K-means で初期化し、不良プロキシ $\{p_{\text{def}}^{(c)}\}_{c=1}^C$ は各不良クラス平均から初期化した上で良品平均方向との内積が $> -0.10$ なら負の半球へ射影初期化する（[`initialize_proxies_from_embeddings`](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py#L93-L146)）。推論・空間評価時の正常度スコアは $s(x) = \max_{m \in \{1..M\}} \langle z(x), \hat{p}_{\text{norm}}^{(m)} \rangle$ と定義する。

### 15.3. フェーズ2 学習エンジン刷新（実装ステップ1〜3）のファイル一覧とフォルダマージ修正ログ
1. **新規作成モジュール (`trainer/app/core/`)**:
   * [nab_sampler.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/nab_sampler.py): `NormalAnomalyBalancedSampler`（良品50% : 全不良モード均等50%の非対称層化バッチ構成、良品2周基準のエポック長、XBM ハード良品35%優先注入、ステップ独立・条件付き復元抽出）および `compute_class_balanced_weights`（平均1.0正規化済み有効サンプル数重み $W_c$）を実装。
   * [proxy_anchor_loss.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/proxy_anchor_loss.py): `CustomizedMultiProxyAnchorLoss`（良品マルチプロキシ $M$ 個の Sub-Center 凝集、実不良モード別凝集、Direction A/B/C の3方向反発と Top-$K$ 危険不良インプレースブースト $\lambda_{\text{hard}}$、合成サンプルの片方向反発、2項プロキシ直交正則化 $L_{\text{proxy-ortho}}$）を実装。
   * [boundary_augmentation.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/boundary_augmentation.py): ピクセル空間ガウシアン残差欠陥転写（`synthesize_defect_transfer_images`）、良品80〜90%比率の `synthesize_anomalous_mixup_images`、および特徴空間上の `synthesize_boundary_gap_features` を実装。
2. **評価・推論・UI 統合およびフォルダ個別アップロード時のマージ保護**:
   * [evaluator.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/evaluator.py) & [evaluation.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/backend/app/routers/evaluation.py) & [3_space_evaluation.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/3_space_evaluation.py): 推論時の正常度スコアをマルチプロキシ最大余弦類似度 $s(x) = \max_{m \in \{1..M\}} \langle z(x), \hat{p}_{\text{norm}}^{(m)} \rangle$ に刷新し、学習済み良品マルチプロキシ（`★ Normal Multi-Proxy`）と不良モードプロキシ（`◆ Defect Mode Proxy`）を 3D UMAP 空間上に同時投影・可視化。
   * [1_clustering_annotation.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/1_clustering_annotation.py) & [api_client.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/api_client.py): `clear_before_upload`（「アップロード時に既存の登録画像をクリアする」）のデフォルトを `False` に変更し、`anomaly/` と `nomal/` を別々に順番にアップロードしても既存のサブフォルダや別名ファイルが消去されず `/app/data/raw/<subfolder>/<file_name>` へファイル単位でマージ保存されるように修正（全クリアしたい場合のみ同チェックボックスを `True` に設定）。

---

## 16. 複数データセット管理・学習セッションNo履歴呼び出し・画像ポップアップ＆分散推移可視化の拡張 (2026-11-11)

実運用および複数条件での比較検証を効率化するため、下記4点のアプリケーション機能拡張を実施した。

### 16.1 複数データセットの独立保存とワンクリック切り替え（機能①）
* **ディレクトリ分離構造**:
  * デフォルトデータセット（`default`）は既存互換性を100%維持するため `/app/data/raw/` および `/app/data/manifests/` にマッピング。
  * 名前付きデータセット（例: `bottle_lotB`, `metal_nut` 等）は `/app/data/datasets/<dataset_name>/raw/` および `/app/data/datasets/<dataset_name>/manifests/` に完全分離して保存。
  * 現在選択中のアクティブデータセット名は `/app/data/active_dataset.json` にアトミック永続化され、[TrainerConfig](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/config.py) と [BackendConfig](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/backend/app/config.py) の双方が共有参照する。
* **API ＆ UI**:
  * `GET /api/v1/datasets/catalog`: 保存済み全データセットの画像枚数・サブフォルダ内訳・クラスタリング/アノテーション有無を一括取得。
  * `POST /api/v1/datasets/select`: アクティブデータセットの切り替え、空の新規データセット作成、および既存データセットからの丸ごと複製保存（`clone_from` 指定時は `clustering_result.json` / `annotations.json` 内の画像パスも新ディレクトリへ自動書き換え）に対応。
  * `DELETE /api/v1/datasets/{dataset_name}`: 指定データセットの削除・初期化。

### 16.2 学習セッションNoの自動採番と過去履歴・モデルの呼び出し（機能②）
* **セッション単位の永続化構造**:
  * 学習ジョブ開始時に [TrainerConfig.allocate_next_session()](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/config.py#L225-L239) が連番 `session_0001`, `session_0002`, ...（表示名: `Session #0001`, `Session #0002`, ...）を自動採番。
  * `/app/data/artifacts/sessions/session_XXXX/` 配下に、そのセッション専用の最高精度チェックポイント `best_model.pt`、全エポックの空間スナップショット `epoch_XXXX_light.pkl` / `epoch_XXXX_full.pkl`、および設定パラメータ・学習履歴を格納した `session_meta.json` を永続保管。
  * 既存の `/app/data/artifacts/` 直下のレガシースナップショット群も起動時に自動的に `session_0001` へマイグレーション保護される。
* **UIからの履歴呼び出し**:
  * [2_training_monitor.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/2_training_monitor.py) 下部の「保存済み学習セッション履歴の呼び出し・振り返り」および [3_space_evaluation.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/3_space_evaluation.py) 上部のセッション選択ドロップダウンから、任意の過去セッションNoを選択して当時の学習曲線・ハイパーパラメータ・3D UMAP空間・単発推論モデルを即座に呼び出し可能。

### 16.3 混同行列・3D UMAP上のTooltips表示と矢印送りポップアップ画像ビューア（機能③）
* **混同行列 (Confusion Matrix) 連動**:
  * `GET /api/v1/evaluations/{epoch}` にて各サンプルに現在の閾値 $\tau_{\text{sim}}$ における混同行列カテゴリ `cm_category` (`TP`, `FN`, `FP`, `TN`) を付与。
  * [3_space_evaluation.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/3_space_evaluation.py) の混同行列を Plotly インタラクティブヒートマップ化し、各セルへのマウスホバーで該当件数と代表ファイル名を Tooltip 表示。
  * 各セル直下のボタン（`🟢 TP`, `🔴 FN`, `🟠 FP`, `🔵 TN`）をクリックすると、該当する全画像を **`@st.dialog` モーダルポップアップ画面** で表示し、複数枚ある場合は **`◀ 前の画像 (Prev)` / `次の画像 (Next) ▶` の矢印ボタン** とサムネイルストリップで1枚ずつ確認可能。
* **3D UMAP / 2D UMAP ナビゲータ連動**:
  * 3D UMAP 散布図の各点に詳細 Tooltip（ファイル名・クラス・`TP/FN/FP/TN`・コサイン類似度・コサイン距離・Split・3D座標）を表示。
  * 点クリックやドラッグ矩形選択（Box/Lasso）で1点または複数点を選択すると、選択されたサンプル群が自動的に矢印送り対応ポップアップ画像ビューアで開く。さらにクラス別および「境界 $\tau_{\text{sim}}$ 近傍の要注意サンプル Top-15」のワンクリックポップアップ起動も搭載。

### 16.4 Space Evaluationの表示順序入れ替えと学習回数（Epoch）に応じた空間分散推移の可視化（機能④）
* **表示順序の入れ替え**:
  * [3_space_evaluation.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/frontend/app/pages/3_space_evaluation.py) において、**セクション2** に「**128次元 → 3D UMAP 立体散布図（学習済みプロキシ座標付き）＆ 学習回数（Epoch）に応じた分散推移**」を配置し、**セクション3** に「**Plotly 128次元生ベクトル活性化ヒートマップ（次元圧縮なし）**」を移動。
* **空間分散・分離度進化指標の定式化 ([evaluator.py](file:///Users/tmcqdtj1089/Desktop/New-metric-learning/trainer/app/core/evaluator.py#L61-L157))**:
  学習回数（Epoch $t = 1 \dots E$）が進むにつれて各クラスの分散がどう収縮し、良品と不良がどう分離していくかを定量化するため、各エポックのスナップショットから以下の指標を算出・可視化した：
  1. **良品クラスタ内分散（平均コサイン距離）**:
     $$\sigma_{\text{norm}}^2(t) = \frac{1}{|N_{\text{norm}}|} \sum_{i \in N_{\text{norm}}} \left(1 - s(x_i)\right)$$
  2. **不良モード内平均分散**:
     $$\sigma_{\text{def}}^2(t) = \frac{1}{C} \sum_{c=1}^{C} \frac{1}{|N_c|} \sum_{i \in N_c} \left(1 - \langle z_i, \mu_c \rangle\right)$$
  3. **良品〜不良 クラス間平均コサイン距離**:
     $$D_{\text{inter}}(t) = \frac{1}{|N_{\text{def}}|} \sum_{i \in N_{\text{def}}} \left(1 - s(x_i)\right)$$
  4. **空間分離比スコア（Fisher風分離度）**:
     $$J(t) = \frac{D_{\text{inter}}(t)}{\sigma_{\text{norm}}^2(t) + \sigma_{\text{def}}^2(t) + 10^{-4}}$$
  5. **3D UMAP クラスタ半径**: 3D UMAP 投影空間上における良品群・不良群それぞれの重心からの二乗平均平方根（RMS）半径。
  * セクション2の上部に **`◀ Epoch -1` / `Epoch +1 ▶` ステップ送りボタン**、**Epoch 1 からの変化量（Delta）付き分散KPIカード**、および **全エポックにわたる分散収縮・空間分離比 $J(t)$ の推移折れ線グラフ** を配置し、学習回数に応じた空間構造の進化を一目で把握できるようにした。

