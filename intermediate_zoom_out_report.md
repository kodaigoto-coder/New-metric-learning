# Goldfish Review Report (Stage 8: Intermediate Zoom Out Review)

**Target:** `README.md` (`Lines 1-332`, Full Document)

## Reviewer Status

| Reviewer | Status |
| --- | --- |
| `egm-zoom-out` | completed |

## Prioritized Findings (Blockers -> Major -> Nits)

### 1. 旧仕様（Sections 1〜4）と確定版仕様（Section 5）の並存による自己矛盾（Split-Brain Specification） - Severity: Blocker
* **Reviewer Lens:** `egm-zoom-out`
* **Affected Scope:** `Lines 10-68` (Sections 1–4) vs. `Lines 71-108` (Section 5)
* **Discrepancy Description**:
  Stage 5 で指摘された設計課題への対策が `Section 5` として末尾に追記された一方、前半の `Sections 1–4`（`Lines 10-68`）に初期ドラフトの記述が未修正のまま残っており、同一ドキュメント内で正反対の仕様（単段階クラスタリング vs 2段階クラスタリング、不良限定データ拡張 vs 良品・不良対称データ拡張、ミニバッチ内限定Top-20 vs Cross-Batch Memory Bank可変Top-K、`PCA(300D)->UMAP(3D)` vs 直接`UMAP(3D)`、毎エポック全量`.pkl`保存 vs 2階層保存）が並存している。
* **Rationale & Recommendation**:
  誤った旧仕様を残したままパッチ節を継ぎ足す構造を解消し、`Sections 1–4` と `Section 5` を統合して単一の確定仕様（Single Source of Truth）に一本化する。

---

### 2. Cross-Batch Memory Bank (XBM) における境界良品側への勾配付与とエポック冒頭クリップ設計 - Severity: Blocker
* **Reviewer Lens:** `egm-zoom-out`
* **Affected Scope:** `Lines 91-98` (Section 5.2.3)
* **Discrepancy Description**:
  1. **境界良品サンプル自体への勾配伝播の欠落**: メモリバンク内の良品ベクトル $z_k \in \mathcal{M}_{\text{normal}}$ は `z.detach()` により計算グラフから切り離されているため、不良アンカー $i \in \mathcal{D}_{\text{defect}}$ との内積 $S_{ik} = z_i^\top z_k^{\text{detached}}$ から計算される $L_{\text{repel}}$ の勾配は不良アンカー $z_i$ 側にのみ流れ、境界良品サンプル $z_k$ 自体を良品コア側へ押し戻す勾配が発生しない。
  2. **エポック毎 `clear()` 直後の初期バッチにおけるサンプル数不足**: 各エポック終了時にメモリバンクを解放（`clear()`）した場合、次エポック開始直後の初期ミニバッチではメモリバンク内の良品数 $|\mathcal{M}_{\text{normal}}|$ が `top_k_count` 未満となる。
* **Rationale & Recommendation**:
  1. **双方向反発＋ハードネガティブ優先バッチサンプリング**: メモリバンク上で特定された「不良に肉薄する Top-$K$ 境界良品インデックス $\mathcal{B}^{(K(t))}$」を次ステップ以降のミニバッチ編成（Batch Sampler）へ優先供給してライブ順伝播・逆伝播に乗せるとともに、ミニバッチ内の良品アンカーからもメモリバンク内の不良ベクトルに対する斥力項を計算する双方向設計とする。これにより、境界良品そのものにも良品中心へ引き戻す勾配が直接作用する。
  2. **エポック毎メモリ解放と実効 $K_{\text{eff}}(t)$ クリップ**: OOM Kill 防止のためのエポック毎のメモリ解放（`clear()`, `gc.collect()`, `empty_cache()`）を実行しつつ、エポック冒頭では $K_{\text{eff}}(t) = \min(K(t), |\mathcal{M}_{\text{normal}}|)$ で安全にクリップする。

---

### 3. 3コンテナ構成における責務境界の整合（PyTorch非依存 `.pkl` シリアライズと `trainer-service` の非同期プロセス分離） - Severity: Major
* **Reviewer Lens:** `egm-zoom-out`
* **Affected Scope:** `Lines 115-211` (Sections 6.1–6.3)
* **Discrepancy Description**:
  1. **コンテナ間 `.pkl` 共有における `torch.Tensor` 脱直列化エラーのリスク**: `trainer-service`（PyTorch環境）が出力した `.pkl` ファイルを、PyTorchを持たない軽量な `api-backend` コンテナが `pickle.load()` で読み込む際、もし `.pkl` 内部に `torch.Tensor` オブジェクトが含まれていると `ModuleNotFoundError: No module named 'torch'` でクラッシュする。また `api-backend` の `evaluation.py`（`Line 189`）にモデルが必要な「単発推論」を含めると `api-backend` にも PyTorch / `timm` が必要になりコンテナ分離の意味が失われる。
  2. **`trainer-service` における FastAPI `BackgroundTasks` のイベントループ阻害**: `trainer-service` 内で CPU/GPU を占有するクラスタリングや学習処理を FastAPI 標準の `BackgroundTasks`（同一プロセス・同一イベントループ）で実行すると、GIL や CPU 飽和により `GET /internal/jobs/{job_id}`（進捗取得）や `GET /health`（ヘルスチェック）がブロックされタイムアウトする。
* **Rationale & Recommendation**:
  1. **Pure NumPy / 標準型での `.pkl` アトミック保存と推論エンドポイントの一本化**: `.pkl` に保存する特徴量・射影ベクトル・類似度配列はすべて `numpy.ndarray`（`float32`）および Python 標準型に変換してから一時ファイル経由でアトミック保存（`os.replace`）する。これにより `api-backend` は PyTorch 非依存のまま高速に `.pkl` を読み込んで閾値 $\tau_{\text{sim}}$ のリアルタイム再計算を行える。また、画像からの特徴抽出を伴う推論処理はすべて `trainer-service` 側に一本化する。
  2. **別プロセスワーカー（`ProcessPoolExecutor` / `multiprocessing`）による非同期ジョブ実行**: `trainer-service` の学習・クラスタリングジョブはサブプロセスで実行し、エポック毎の進捗状態を共有ステータスファイル（またはプロセス間共有状態）へ書き出すことで、学習中も FastAPI の HTTP エンドポイントがミリ秒単位で応答できるようにする。

---

### 4. 学習時デフォルト判定閾値 $\tau_{\text{sim}} = 0.0$ と損失内反発マージン $m(t)$ の記号分離およびスケジューラ進捗保証 - Severity: Major
* **Reviewer Lens:** `egm-zoom-out`
* **Affected Scope:** `Lines 95-104` (Sections 5.2.3–5.3.1)
* **Discrepancy Description**:
  `Line 95` で損失内の押し戻しマージンと推論時の判定閾値の双方に $\tau_{\text{sim}}$ が使われており数式内の $m(t)$ と混同されている。また、学習初期に全ベクトルが正の類似度領域にある間は固定閾値 $\tau_{\text{sim}} = 0.0$ での Recall が $0$ となるため、スケジューラがデッドロックしない進捗関数が必要である。
* **Rationale & Recommendation**:
  損失関数内の反発マージンは $m(t)$、推論・評価時の2値判定閾値は $\tau_{\text{sim}}$（デフォルト $0.0$、UIスライダーで $[-1.0, 1.0]$ に可変）として記号を明確に分離し、カリキュラム進捗度 $p(t)$ は検証 Recall と連続値である境界最小コサイン距離 $d_{\min}(t)$（およびエポック進捗）の複合指標で駆動する。

---

### 5. CI/CD パイプラインにおける Terraform 実行条件の分離と GCE セキュアアクセス（IAP Tunnel） - Severity: Nit
* **Reviewer Lens:** `egm-zoom-out`
* **Affected Scope:** `Lines 291-325` (Sections 7.2–7.3)
* **Discrepancy Description**:
  `cd-gce.yml` でアプリコードのみの変更時も毎回 `terraform apply` を実行する構成になっている点、および GCE 上の Streamlit（ポート8501）をパブリック Firewall で直接開放するリスクがある点。
* **Rationale & Recommendation**:
  Terraform apply は `infra/terraform/**` の変更時（またはインフラ更新ジョブ）に限定し、GCE 上の Streamlit UI へのアクセスは IAP TCP Forwarding（`gcloud compute start-iap-tunnel` または SSH ポートフォワーディング）を推奨手順として併記する。

