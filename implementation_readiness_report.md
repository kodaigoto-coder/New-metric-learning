# Goldfish Review Report (Stage 9: Implementation Readiness Test)

**Target:** `README.md` (`Lines 111-332`, Detailed Implementation Plan)

## Reviewer Status

| Reviewer | Status |
| --- | --- |
| `egm-implementation-readiness` | completed |
| `egm-codebase-details` | completed |
| `egm-style` | completed |

## Prioritized Findings (Blockers -> Major -> Nits)

### 1. コンテナ間 REST API エンドポイント契約一覧および Pydantic リクエスト/レスポンススキーマの未確定 - Severity: Blocker
* **Reviewer Lens:** `egm-implementation-readiness`, `egm-codebase-details`
* **Affected Scope:** `Lines 155-160`, `Lines 179-200`, `Line 237`, `Line 331`
* **Discrepancy Description**:
  Section 6.2 の表では「主な責務・エンドポイント例」として一部のパスのみが記載されており、実装者が推測なしに `frontend/app/api_client.py`、`backend/app/routers/*.py`、`trainer/app/routers/train_jobs.py` を実装するための完全な API 契約が定義されていない：
  1. フロントエンドがジョブ進捗をポーリングする `GET /api/v1/jobs/{job_id}`、クラスタリング結果を取得する `GET /api/v1/clusters`、手順7.1（`Line 237`）で言及されている「サンプル合成データ生成API（`POST /api/v1/datasets/generate-synthetic`）」、手順7.3（`Line 331`）のヘルスチェック `GET /health`、および単発推論 API のルーティング契約が未記載。
  2. `POST /api/v1/jobs/training` のリクエストパラメータ（`top_k_mode: "count" | "ratio"`, `top_k_initial`, `top_k_final`, `tau_sim`, `backbone_name`, `freeze_backbone` 等）や、`GET /api/v1/evaluations/{epoch}?tau_sim=0.0` のレスポンスJSON構造（Recall/Precision/F1/混同行列、128次元ヒートマップ配列、3D UMAP座標配列）のフィールド定義が未記載。
* **Rationale & Recommendation**:
  `api-backend`（Port 8000）および `trainer-service`（Port 8001）の全エンドポイント一覧表と、`backend/app/schemas/`・`trainer/app/schemas/` の Pydantic モデル定義を明記する。また、モデル重み（PyTorch / `timm`）を必要とする単発推論および合成データ生成は `trainer-service` 側で実行し、`api-backend` がプロキシする構成に統一する。

---

### 2. 共有ボリューム（`/app/data`）上の永続化データスキーマ（`clustering_result.json`, `annotations.json`, 2階層 `.pkl`）と PyTorch 非依存シリアライズ契約の未定義 - Severity: Blocker
* **Reviewer Lens:** `egm-implementation-readiness`, `egm-codebase-details`
* **Affected Scope:** `Line 79`, `Line 107`, `Line 130`, `Lines 158-160`, `Lines 206-211`
* **Discrepancy Description**:
  1. **PyTorch非依存 `.pkl` 契約**: `api-backend` は `torch` を持たない軽量コンテナであるため、`trainer-service` が出力する `.pkl` 内に `torch.Tensor` が残っていると `api-backend` の `pickle.load()` 時に `ModuleNotFoundError: No module named 'torch'` が発生する。すべての配列を `numpy.ndarray`（`float32`）および Python 標準型へ変換してアトミック保存（一時ファイル書き込み後に `os.replace`）する仕様を明文化する必要がある。
  2. **ファイル命名規則と Dict / JSON スキーマ**:
     * クラスタリング中間結果 `data/manifests/clustering_result.json`
     * 確定アノテーション `data/manifests/annotations.json`
     * 毎エポック軽量スナップショット `data/artifacts/epoch_{epoch:04d}_light.pkl` とフルスナップショット `data/artifacts/epoch_{epoch:04d}_full.pkl`、およびベスト重み `data/artifacts/best_model.pt` の厳密なキー名とデータ型が未定義。
* **Rationale & Recommendation**:
  上記ファイルの命名規則、JSONスキーマ、および `.pkl` 辞書キー構造（`embeddings_128d: float32[N, 128]`, `cosine_similarities: float32[N]`, `umap_3d: float32[N, 3]`, `normal_prototype: float32[128]`, `gap_features: float32[N, D] | None` 等）を確定仕様として明記する。

---

### 3. 全作成ファイル境界リスト（`[Create]` / `[Modify]`）、段階的実装フェーズ（Phasing & Checkpoints）、および具体的テスト計画（Test Plan）の欠落 - Severity: Blocker
* **Reviewer Lens:** `egm-implementation-readiness`, `egm-codebase-details`
* **Affected Scope:** `Lines 161-223`, `Lines 310-313`
* **Discrepancy Description**:
  `implementation-readiness-standard.md` で必須とされる以下の3要素が不足している：
  1. Section 6.3 のディレクトリツリーで `backend/app/schemas/`、`backend/tests/`、`trainer/tests/` の配下ファイル名が省略されており、各ファイルの操作区分（`[Create]` / `[Modify]`）が明記されていない。
  2. どの順序でモジュールを実装し、各段階の完了をどの検証コマンドで確認するかという **Phasing & Checkpoints（段階的実装計画）** がない。
  3. 具体的なテストファイル名と、正常系・境界異常系（バッチ内不良0件時、エポック冒頭のMemory Bank要素数不足時、少数サンプル時のPCA/シルエットスコア/UMAPガード、`.pkl` 内の `torch.Tensor` 非混入検証、閾値 $\tau_{\text{sim}}$ 変更時のRecall/F1再計算等）のテストケース一覧がない。
* **Rationale & Recommendation**:
  全対象ファイルの完全なパスリスト（`[Create]` / `[Modify]` 付き）、Phase 1〜Phase 5 の段階的実装ステップと検証コマンド、および単体・結合テストケース一覧を追記する。

---

### 4. コアMLモジュールの初期ハイパーパラメータ値および `scikit-learn` / `umap-learn` の境界ガード条件の確定 - Severity: Major
* **Reviewer Lens:** `egm-implementation-readiness`, `egm-codebase-details`
* **Affected Scope:** `Lines 75-108`, `Lines 200-206`
* **Discrepancy Description**:
  1. `sklearn.decomposition.PCA` では `n_components=0.95`（累積寄与率）と整数上限 `64` を同時に単一引数へ渡せないため、`PCA(n_components=min(64, n_samples - 1, n_features))` 実行後に累積寄与率95%到達次元でスライスする実装規則が必要。
  2. `umap.UMAP(n_components=3, metric="cosine")` においてサンプル数 $N$ が小さい場合に `n_neighbors >= N` となるとエラーが発生するため、`n_neighbors = min(15, max(2, N - 1))` のガードが必要。
  3. 非対称MS-Lossおよびカリキュラムスケジューラのデフォルト定数（$\alpha=2.0, \beta=50.0, \lambda=0.5, \epsilon=0.1, w_{\text{defect}}=2.0, w_{\text{aux}}=0.5, \beta_{\text{rep}}=20.0$、初期/後期マージン $m(t)$、進捗関数 $p(t)$ の計算式）を明記する必要がある。
* **Rationale & Recommendation**:
  `trainer/app/core/` の各モジュールが参照する数式・デフォルトハイパーパラメータ値・境界ガード条件を確定仕様として明記する。

---

### 5. `trainer-service` の非同期ジョブ実行スレッド/プロセス分離、Docker/Terraform/CD パイプラインの具体化 - Severity: Major
* **Reviewer Lens:** `egm-implementation-readiness`, `egm-codebase-details`
* **Affected Scope:** `Lines 159-170`, `Lines 212-222`, `Lines 233-332`
* **Discrepancy Description**:
  1. **イベントループ保護と排他制御**: `trainer-service` で重い学習ジョブを実行中に `/internal/jobs/{job_id}` や `/health` がブロックされないよう、`ThreadPoolExecutor(max_workers=1)`（または別プロセス）で実行し、実行中の重複起票には `409 Conflict` を返す仕様が必要。
  2. **`.env.example` と Docker Compose リソース制限**: 環境変数一覧と、`trainer-service` の `mem_limit`（例：`4g`〜`8g`）・`shm_size` 設定の明記。
  3. **GCE CD パイプラインの `docker-compose*.yml` 転送と IAP フラグ**: `cd-gce.yml` において `gcloud compute ssh` の前に `gcloud compute scp --tunnel-through-iap` で最新の `docker-compose.yml` / `docker-compose.prod.yml` を GCE インスタンスへ転送するステップ、および `--tunnel-through-iap` と `gcloud auth configure-docker` の明記が必要。
* **Rationale & Recommendation**:
  上記の設定値およびデプロイコマンドを Section 6〜7 に明記する。

---

### 6. 見出し番号・コンテナ名称の統一と用語の精緻化 - Severity: Nit
* **Reviewer Lens:** `egm-style`
* **Affected Scope:** `Lines 111-113`, `Lines 127-207`, `Lines 319-332`
* **Discrepancy Description**:
  * `Line 111` の `## Detailed Implementation Plan` と `## 6.` の見出し階層の整理。
  * Mermaid図・表・ディレクトリツリーにおけるコンテナ識別子（`frontend`, `api-backend`, `trainer-service`）の統一。
  * `Line 319` の「GCE ローリングデプロイ」を実態に即した「GCE コンテナ更新デプロイ（Docker Compose 再起動）」へ修正。
