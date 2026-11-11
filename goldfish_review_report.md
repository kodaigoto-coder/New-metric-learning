# Goldfish Annotated Review Report

**Status:** `resolved` (Round 1: 7/7 `PROPOSE FIX` items applied; Round 2: 2/2 `PROPOSE FIX` items applied, 0 Blockers)
**Target:** `proxy_anchor_design_review.md` (`Lines 1-94`) & `note.md` (`Lines 283-376`)

## Prioritized Findings & AI Suggestions

### [Issue 1] [`egm-implementation-readiness` / `egm-comprehension` / `egm-codebase-details`] - Severity: Blocker
*   **Affected Scope:** [Lines 67-71](proxy_anchor_design_review.md#L67-L71) (`3.1. Normal-Anomaly Balanced Sampler (NAB-Sampler)`) & [Lines 289-290, 312](note.md#L289-L312) (`15.1.1 & 15.3.1`)
*   **Discrepancy Description:** Both documents state only that `NAB-Sampler` splits each mini-batch of size $B$ into 50% normal slots ($B_{\text{norm}} = \lfloor B/2 \rfloor$) and 50% anomaly slots ($B_{\text{anom}} = B - B_{\text{norm}}$) drawn evenly across defect classes $c \in \{1..C\}$. They omit how [`NormalAnomalyBalancedSampler.build_epoch_batches`](trainer/app/core/nab_sampler.py#L103-L182) and [`_execute_training_task`](trainer/app/routers/train_jobs.py#L446-L535) prevent minority defect exhaustion and representation memorization when $n_c \ll B_{\text{anom}}$:
    1. **Normal-Driven Epoch Length & Cyclic Cursor**: `steps_for_normals = ceil(len(norm_pool) / n_norm_slots) * 2` and `num_batches = max(min_steps, steps_for_normals)` (`min_steps=12` default in `nab_sampler.py`, called with `min_steps=14` in `train_jobs.py`), traversing `norm_pool` via a cyclic cursor `norm_cursor` that reshuffles upon completing each full pass (`norm_cursor % len(norm_pool) == 0`).
    2. **35% Cap on XBM Hard-Normal Injection**: Caps XBM boundary hard normals at `n_hard = min(len(valid_hard_normals), max(1, int(n_norm_slots * 0.35)))` (`replace=False`), leaving $\ge 65\%$ of normal slots for cyclic global normal coverage.
    3. **Per-Step Independent Defect Sampling with Conditional Replacement & Mode Subsampling**: At every batch step, each active defect mode $c$ receives $k_{\text{mode}} = \lfloor B_{\text{anom}} / |C_{\text{active}}| \rfloor + \mathbb{I}(m < B_{\text{anom}} \bmod |C_{\text{active}}|)$ slots (randomly subsampling `max_active_modes = max(1, n_def_slots // 2)` modes when $C > B_{\text{anom}} // 2$ and $C > 4$). When $n_c < k_{\text{mode}}$ (`replace_flag = len(mode_pool) < k_mode`), it samples **with replacement within the step** (`replace=True`) and **re-draws independently across every batch step** so scarce defect modes never exhaust.
    4. **Anti-Memorization via On-the-Fly Jitter & Dual-Space Boundary Synthesis**: Injects Gaussian feature jitter $\epsilon \sim \mathcal{N}(0, 0.015^2 I)$ + $L_2$ re-normalization (`batch_gap_aug`) and combines on-the-fly feature-space Defect Transfer + Anomalous Mixup (`synthesize_boundary_gap_features`) with up to 8 samples/step from the pre-extracted pixel-level synthesis bank (`_extract_pixel_boundary_augmentation_bank`).
*   **Rationale:** Under a 100:2 class imbalance, a naive 50:50 sampler without replacement exhausts defect samples on Step 1. Documenting the exact normal-anchored epoch sizing, conditional replacement rule, and anti-memorization pipeline is required for theoretical and implementation parity.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** The critique is valid and directly addresses the user's question regarding how minority defect exhaustion is prevented under 50:50 batch sampling.
*   ***Proposed Edit:***
    *   Location: `proxy_anchor_design_review.md` Lines 67–71 (`### 3.1. Normal-Anomaly Balanced Sampler`) and `note.md` Lines 289–290 (`### 15.1.1`)
    *   Change: Expand Section 3.1 in `proxy_anchor_design_review.md` and Section 15 in `note.md` to explicitly document (1) Normal-driven epoch length (`max(min_steps, ceil(N_norm / B_norm) * 2)`) with cyclic `norm_cursor` reshuffling and 35% XBM hard-normal cap, (2) Per-step stratified defect sampling with conditional replacement (`replace = (n_c < k_mode)`) and mode subsampling when $C > B_{\text{anom}} // 2 \land C > 4$, and (3) the 4-layer anti-memorization safeguards (Proxy anchor, $\mathcal{N}(0, 0.015^2)$ feature jitter, dual-space Defect Transfer / Mixup, and mean-normalized $W_c$).

---

### [Issue 2] [`egm-codebase-details` / `egm-implementation-readiness` / `egm-comprehension`] - Severity: Blocker
*   **Affected Scope:** [Lines 76-86](proxy_anchor_design_review.md#L76-L86) (`3.2. カスタマイズド Multi-Proxy Anchor Loss`) & [Lines 291-299](note.md#L291-L299) (`15.1.2 - 15.1.4`)
*   **Discrepancy Description:**
    1. [proxy_anchor_design_review.md](proxy_anchor_design_review.md#L77) writes $L_{\text{total}} = L_{\text{proxy-pos}} + L_{\text{proxy-neg}} + \gamma(t) L_{\text{topK-hard}} + \lambda_{\text{ortho}} L_{\text{proxy-ortho}}$ with a single scale $\alpha_{\text{pa}}$. In [`CustomizedMultiProxyAnchorLoss.forward`](trainer/app/core/proxy_anchor_loss.py#L199-L374), there is no separate additive $L_{\text{topK-hard}}$ term; instead, `loss_total = loss_pos + (repel_scale * loss_repel) + (self.ortho_weight * loss_ortho)` where `repel_scale = max(0.5, float(gamma_repel))` scales the entire `loss_repel`, positive attraction uses $\alpha_{\text{pos}} = 16.0$, and negative repulsion uses $\gamma_{\text{neg}} = 20.0$.
    2. Top-$K$ hard-anomaly boosting is applied **multiplicatively inside Direction A of `loss_repel`** ([Lines 272-301](trainer/app/core/proxy_anchor_loss.py#L272-L301)) via $\omega_i = \operatorname{clamp}(\text{boost}_i \cdot W_{y_i} \cdot w_{\text{defect}}, 10^{-4})$ with $\text{boost}_i = \lambda_{\text{hard}} (=3.0)$ for $i \in \operatorname{TopK}(\max_m z_i^\top p_{\text{norm}}^{(m)})$ and $w_{\text{defect}} = 2.5$.
    3. `loss_repel` averages three components (`torch.stack(repel_terms).mean()`): **Direction A** (all real + synthetic anomalies repelled from all $M$ normal proxies), **Direction B** ([Lines 303-317](trainer/app/core/proxy_anchor_loss.py#L303-L317): batch normal samples repelled from all $C$ defect proxies weighted by $W_c$), and **Direction C** ([Lines 319-344](trainer/app/core/proxy_anchor_loss.py#L319-L344): batch anomalies repelled from XBM cached Top-$K$ **boundary normal embeddings**, not cached XBM defects).
*   **Rationale:** The equations in Section 3.2 of `proxy_anchor_design_review.md` must match the exact 3-direction repulsion structure and in-place LogSumExp weighting of `CustomizedMultiProxyAnchorLoss.forward`.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Aligning Section 3.2 with the exact mathematical structure in `proxy_anchor_loss.py` eliminates all divergence between specification and code.
*   ***Proposed Edit:***
    *   Location: `proxy_anchor_design_review.md` Lines 76–86 (`### 3.2. カスタマイズド Multi-Proxy Anchor Loss`) and `note.md` Lines 291–299
    *   Change: Replace the additive $L_{\text{topK-hard}}$ equation with the exact implemented decomposition $L_{\text{total}} = L_{\text{pos}} + \max(0.5, \gamma_{\text{repel}}(t)) L_{\text{repel}} + \lambda_{\text{ortho}} L_{\text{proxy-ortho}}$, explicitly writing out $L_{\text{norm-pos}}$, $L_{\text{def-pos}}$ ($\alpha_{\text{pos}}=16.0, \delta_{\text{pos}}=0.65$), and the 3 directions of $L_{\text{repel}}$ (Direction A with $\omega_i = \operatorname{clamp}(\text{boost}_i \cdot W_{y_i} \cdot w_{\text{defect}}, 10^{-4})$, Direction B repelling normals from defect proxies, and Direction C repelling anomalies from XBM Top-$K$ boundary normals with $\gamma_{\text{neg}}=20.0$).

---

### [Issue 3] [`egm-codebase-details` / `egm-implementation-readiness` / `egm-comprehension`] - Severity: Blocker
*   **Affected Scope:** [Lines 34-48, 79](proxy_anchor_design_review.md#L34-L79) (`2. トラップ2` & `3.2`) & [Lines 298, 304-306, 314](note.md#L298-L314)
*   **Discrepancy Description:**
    1. [proxy_anchor_design_review.md](proxy_anchor_design_review.md#L79) states $\delta_{\text{pos}} = 0.70$, whereas [`CustomizedMultiProxyAnchorLoss.__init__`](trainer/app/core/proxy_anchor_loss.py#L47) sets `delta_pos: float = 0.65`.
    2. [proxy_anchor_design_review.md](proxy_anchor_design_review.md#L45) states $\alpha \in [0.65, 0.85]$ in "パッチ整合ピクセル空間" (patch-aligned pixel space), contradicting its own heading at Line 34 ($\alpha = 0.8 \sim 0.9$), [note.md](note.md#L298) ($\alpha \in [0.80, 0.90]$), and [`boundary_augmentation.py`](trainer/app/core/boundary_augmentation.py#L111-L112) (`alpha_min=0.80, alpha_max=0.90`).
    3. [`synthesize_defect_transfer_images`](trainer/app/core/boundary_augmentation.py#L32-L103) locates `(peak_y, peak_x) = argmax AvgPool_{11x11}(|I_bad - I_ref|)` (where `I_ref` is matched via $16\times 16$ pooled $L_1$ distance) and blends full images in-place using a 2D Gaussian window (`sigma_ratio=0.16, peak_opacity=0.82`) rather than cropping/translating to a random coordinate, while [`synthesize_boundary_gap_features`](trainer/app/core/boundary_augmentation.py#L147-L207) splices the top 30% residual feature dimensions (`weight=0.85`) and combines them with up to 8 samples/step from `_extract_pixel_boundary_augmentation_bank` (up to 48 normal / 48 defect images).
*   **Rationale:** Exact parameter values ($\delta_{\text{pos}} = 0.65$, $\alpha \in [0.80, 0.90]$) and the dual pixel-bank + feature-splicing synthesis mechanism must be consistently stated across both documents.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Updating `proxy_anchor_design_review.md` and `note.md` to reflect $\delta_{\text{pos}} = 0.65$, $\alpha \in [0.80, 0.90]$, in-place Gaussian residual blending, and top-30% feature residual splicing resolves all parameter and algorithmic discrepancies.
*   ***Proposed Edit:***
    *   Location: `proxy_anchor_design_review.md` Lines 44–48, 79 and `note.md` Lines 304–306, 314
    *   Change: Standardize $\delta_{\text{pos}} = 0.65$ and $\alpha \in [0.80, 0.90]$, and document both image-space residual Gaussian feathering (`sigma_ratio=0.16, peak_opacity=0.82`) and feature-space top-30% residual dimension splicing (`transfer_ratio=0.30, weight=0.85`).

---

### [Issue 4] [`egm-codebase-details` / `egm-implementation-readiness`] - Severity: Blocker
*   **Affected Scope:** [Line 94](proxy_anchor_design_review.md#L94) (`4. 直近の実行ログから判明したデータ運用上の注意点`) & [Line 317](note.md#L317) (`15.3.2`)
*   **Discrepancy Description:** [proxy_anchor_design_review.md](proxy_anchor_design_review.md#L94) states that `clear_before_upload` OFF performs `"同名のサブフォルダのみ上書き・別名サブフォルダは追加統合"` and claims an explicit `"全データセット初期化"` button is separated. In [`1_clustering_annotation.py`](frontend/app/pages/1_clustering_annotation.py#L188-L192) and [`datasets.py`](backend/app/routers/datasets.py#L196-L238), `clear_before_upload` defaults to `False` and performs file-level merge inside `/app/data/raw/<subfolder>/<file_name>` (overwriting files with the same filename while preserving other files and subfolders), and full dataset reset is performed by checking the `clear_before_upload` checkbox rather than a separate button.
*   **Rationale:** Documentation of UI controls and file-storage behavior must match `1_clustering_annotation.py` and `datasets.py`.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Aligning Line 94 of `proxy_anchor_design_review.md` with the implemented `clear_before_upload=False` checkbox behavior ensures accurate operational guidance.
*   ***Proposed Edit:***
    *   Location: `proxy_anchor_design_review.md` Line 94
    *   Change: Replace the reference to a separate reset button and subfolder-wipe with the exact implemented behavior: `clear_before_upload` defaults to `False` (merging uploaded files into `/app/data/raw/<subfolder>/<file_name>` while preserving existing subfolders/files; checking `clear_before_upload=True` clears all existing raw images and manifests before upload).

---

### [Issue 5] [`egm-codebase-details` / `egm-implementation-readiness` / `egm-comprehension`] - Severity: Major
*   **Affected Scope:** [Lines 14, 28-30, 56-61, 80](proxy_anchor_design_review.md#L14-L80) & [Lines 298, 302-308](note.md#L298-L308)
*   **Discrepancy Description:**
    1. **Normal Sub-Center Orthogonality Term**: [`CustomizedMultiProxyAnchorLoss.forward`](trainer/app/core/proxy_anchor_loss.py#L348-L370) includes a second hinge penalty in $L_{\text{proxy-ortho}}$ over the $M$ normal sub-centers (`ortho_margin_normal = 0.85`, weight `0.25`) and scales $L_{\text{proxy-ortho}}$ by `ortho_weight = max(0.2, req.w_aux)` (default `0.5`).
    2. **Mean Normalization of $W_c$**: [`compute_class_balanced_weights`](trainer/app/core/nab_sampler.py#L8-L38) normalizes $W_c = \frac{1-\beta}{1-\beta^{\max(1, n_c)}}$ by its mean across all $C$ defect classes so $\frac{1}{C}\sum_{c=1}^C W_c = 1.0$.
    3. **Negative-Hemisphere Defect Proxy Init & `2.5x` Proxy LR**: [`initialize_proxies_from_embeddings`](trainer/app/core/proxy_anchor_loss.py#L135-L146) projects initial defect proxy centroids into the negative hemisphere (`if proj > -0.10: F.normalize(c_t - (proj + 0.35) * p_norm_mean)`), and [`train_jobs.py`](trainer/app/routers/train_jobs.py#L354-L357) optimizes proxy parameters with `lr = req.learning_rate * 2.5`.
*   **Rationale:** Omitting mean-normalization of $W_c$ or the normal sub-center orthogonality term leaves mathematical gaps between the specification and `proxy_anchor_loss.py`.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Adding these four mathematical details to `proxy_anchor_design_review.md` and `note.md` makes the specification self-contained and exact.
*   ***Proposed Edit:***
    *   Location: `proxy_anchor_design_review.md` Lines 14, 28–30, 56–61, 80 and `note.md` Lines 298, 302–308
    *   Change: Document the 2-part $L_{\text{proxy-ortho}}$ equation (defect margin `0.20` + normal sub-center margin `0.85` weighted by `0.25`), mean-normalized $W_c$ ($\frac{1}{C}\sum_c W_c = 1.0$), negative-hemisphere defect proxy initialization, and the `2.5x` proxy learning rate multiplier.

---

### [Issue 6] [`egm-comprehension`] - Severity: Major
*   **Affected Scope:** [Lines 3, 25](proxy_anchor_design_review.md#L3-L25) & [Line 310](note.md#L310)
*   **Discrepancy Description:**
    1. `note.md` Line 310 uses `"Phase 1〜3"` to refer to implementation stages of the new loss/sampler/augmentation inside the Phase 2 trainer, colliding with the system's top-level `"フェーズ1（前処理クラスタリング）"` and `"フェーズ2（距離学習）"` terminology.
    2. `proxy_anchor_design_review.md` Line 25 references `"提案仕様 4.1"` without inline context.
*   **Rationale:** Disambiguating "Phase 2 学習アルゴリズム刷新（Step A〜C / 実装フェーズ 1〜3）" prevents confusion between application workflow phases and implementation stages.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Clarifying the heading in `note.md` and adding inline context in `proxy_anchor_design_review.md` improves readability.
*   ***Proposed Edit:***
    *   Location: `proxy_anchor_design_review.md` Line 25 and `note.md` Line 310
    *   Change: Clarify that the 3 implementation stages belong to the Phase 2 Metric-Learning engine, and state the context of Section 4.1 inline.

---

### [Issue 7] [`egm-comprehension` / `egm-codebase-details`] - Severity: Nit
*   **Affected Scope:** [Line 313](note.md#L313) (`15.3.1`)
*   **Discrepancy Description:** [`CustomizedMultiProxyAnchorLoss.forward`](trainer/app/core/proxy_anchor_loss.py#L375-L381) returns `loss_pos` under the dictionary key `"loss_ms"` for backward compatibility with the job history schema, which is not noted in `note.md`.
*   **Rationale:** Noting the `"loss_ms"` key mapping helps maintainers inspecting `stats` dictionaries.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** A 1-line note in `note.md` Section 15.3 documents this schema compatibility key.
*   ***Proposed Edit:***
    *   Location: `note.md` Line 313
    *   Change: Append a note that `loss_pos` is returned under `stats["loss_ms"]` for backward compatibility.

---

## Round 2: 4 Application Enhancements & `note.md` Section 16 (`Status: resolved`)

### [Issue 8] [`egm-codebase-details`] - Severity: Nit
*   **Affected Scope:** [3_space_evaluation.py](frontend/app/pages/3_space_evaluation.py#L185-L235)
*   **Discrepancy Description:** Both the sidebar (`step6_sidebar_session_select`) and top bar (`step6_main_session_select`) rendered independent session selectboxes without synchronizing widget state via `on_change` callbacks.
*   ***AI Suggestion:*** PROPOSE FIX (Applied)
*   ***Resolution:*** Synchronized both selectboxes via `st.session_state["step6_active_session"]` and `on_change` callbacks (`_on_sidebar_session_change` and `_on_main_session_change`).

### [Issue 9] [`egm-codebase-details`] - Severity: Nit
*   **Affected Scope:** [note.md](note.md#L346)
*   **Discrepancy Description:** Line anchor for `TrainerConfig.allocate_next_session()` pointed to `#L221-L236` instead of `#L225-L239`.
*   ***AI Suggestion:*** PROPOSE FIX (Applied)
*   ***Resolution:*** Updated line anchor in `note.md` Section 16.2 to `#L225-L239`.

