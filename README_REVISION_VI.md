# Code mô phỏng (Robust Covariance UAV-ISAC)

Gói này là bản đã làm sạch của `code_paper_robustcov`. Nó gồm phần lõi mô phỏng,
các tính năng mới cho phần lý thuyết bổ sung (Prop. 2–6, Lemma 1–4, Algorithm 2),
và các thí nghiệm R1–R11 trả lời phản biện. Các nghiên cứu của bản nộp (S2, S3,
S4, S5A, S5B, S9, S9R) được giữ **nguyên cấu hình**, nên kết quả cũ dùng lại
được mà không phải chạy lại.

## 1. Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate      # Python >= 3.10
pip install -r requirements.txt
# tùy chọn, cho R3 (so sánh solver): pip install mosek   (cần license)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

## 2. Smoke test (chạy trước tiên, khoảng 2–3 phút)

```bash
python -m pytest -q tests
```

Có 39 test, chia làm ba nhóm.

* `tests/test_theory.py` (khoảng 1 s) kiểm tra:
  * Prop. 1: công thức vô hướng (25) khớp `pcrb()`; rank(G) ≤ 3; điều kiện giảm ngặt.
  * Prop. 2: ngưỡng P_min chính xác, tức PCRB tại P_min bằng ε, còn ở 0.999·P_min thì vượt ε.
  * Lemma 4: bán kính phủ của tập 9 điểm bằng κ.
  * Tái dựng gia tốc ở điểm cuối, Eq. (42).
  * Prop. 3: N_s = 29.142 khi d = 1284.
  * Lemma 2, Lemma 3.
  * Prop. 4 và tính hợp lệ của dual certificate trong Prop. 5.
* `tests/test_pipeline_smoke.py` (khoảng 2 phút, nhiệm vụ nhỏ 2×2 UPA, 6 slot) chạy đầu–cuối từng tính năng:
  * Dạng tuyến tính cho đúng cùng năng lượng với dạng Schur.
  * Scenario certificate đạt ngưỡng ρ.
  * Algorithm 2 xác nhận được thiết kế.
  * Tái sử dụng lời giải của candidate và screening theo Prop. 4.
  * Baseline propulsion-only.
  * Random search và Gaussian randomization.
  * Đánh giá lệch mô hình (Rician, coordinated-turn).
  * Chứng nhận bất khả thi Prop. 5.
  * Post-hoc và xuất bảng.
* `tests/test_plan.py`: mọi kế hoạch thí nghiệm có fingerprint duy nhất, và fingerprint của run Table II không đổi so với bản cũ (`2137019914ce`).

## 3. Dùng lại kết quả của bản nộp (khuyến nghị)

Fingerprint của cấu hình cũ được giữ nguyên, vì các trường mới ở giá trị mặc định
không đi vào fingerprint. Chép thư mục kết quả cũ vào cấu trúc mới:

```bash
mkdir -p results/legacy
cp -r <code_paper_robustcov>/results_reference/outputs_cp6_lock      results/legacy/
cp -r <code_paper_robustcov>/results_reference/outputs_cp7_postlock  results/legacy/
cp -r <code_paper_robustcov>/results_reference/outputs_cp7_s9r       results/legacy/
python run_revision.py legacy --which S2,S3,S4,S5A,S9,S5B,S9R
```

Đã kiểm tra: 82/90 run cũ được nạp lại trong khoảng 3 s, không giải lại, và các
bảng sinh ra khớp với số trong bản thảo (Table II, Table III, Fig. 5).

Có 8 run S9 với topology T1/T2 có fingerprint khác bản lưu trữ: chúng được tạo
trước khi topology bị đóng băng. Lệnh `legacy --which S9` sẽ chạy lại 8 run này
(mỗi run khoảng 1–5 h). Chúng không xuất hiện trong bản thảo hiện tại, nên có thể
bỏ qua bằng `--which S2,S3,S4,S5A,S5B,S9R`.

## 4. Thí nghiệm mới

```bash
python run_revision.py plan                     # danh sách và số run
python run_revision.py run --exp R11            # vài giây
python run_revision.py run --exp R2 --shard 0/3 # chia shard, chạy song song nhiều process
python run_revision.py run --exp R1 --opt n_local=30 n_unrestricted=10 'r1_methods=["nine_state","refined","scenario"]'
python run_revision.py run --exp R6 --only reference_scenario   # chạy một run cụ thể
```

Mọi run đều resume được: chạy lại đúng lệnh cũ thì run đã xong được nạp lại.
Mỗi shard ghi vào thư mục run riêng. Sau khi chạy xong:
`python run_revision.py collect --exp R1`.

| Exp | Số run | Nội dung | Phản biện | Chi phí ước tính (CPU, 1 luồng) |
|---|---|---|---|---|
| R11 | 5 | Prop. 5: chứng nhận bất khả thi ở ε = 4 m² (mọi rate) + báo cáo cả 48 slot | Ed.3, R1.2, R3.1 | vài giây (4 run) + khoảng 1 h (1 run) |
| **R12** | 11 | **Thiết kế propulsion-first** (Cor. 2): AO khởi tạo từ quỹ đạo tối ưu propulsion (Powell 30 start). Chạy lại Table II (3 phương pháp) và nghiên cứu displaced (offset 180–260, ε 5–9) | Ed.3, R1.2, R2.5, R3.1 | 3 × khoảng 1 h + 8 × khoảng 15 phút |
| R2 | 6 | Baseline trajectory: propulsion-only, random search, Powell (tham chiếu và displaced) | R1.7, R2.5, R3.3, Ed.3 | 3 run × 1–3 h + 3 run × khoảng 10 phút |
| R5 | 3 | Fixed sensing power **có** xử lý bất định robust (10/20/30 %) | R3.3 | 3 × 1–3 h |
| R6 | 8 | Scenario (Prop. 3) và refinement (Alg. 2) trên reference, Local-1, Local-2, góc S3 (8 m/1 %) | Ed.4, R2.3, R3.2 | refinement: 2–6 h mỗi run; scenario: **chưa biết**, chạy thử 1 run trước |
| R7 | 15 | Quét mức siết 0–20 % × {T0, Local-1, Local-2}, fixed trajectory, dạng tuyến tính | R1.3 | 15 × khoảng 0.5–1 h |
| R3 | 6 | SCS / CLARABEL / MOSEK × Schur / tuyến tính, fixed trajectory | R3.3, R2.6 | 6 × 0.5–2 h |
| R4 | 4 | Rank recovery: bank vs Gaussian randomization | R3.3, R3.4 | 4 × 0.5–2 h |
| R8 | 9 | Độ nhạy theo số start, số đoạn, số vòng lặp SLSQP (geometry displaced) | R2.4, R1.10 | 9 × 5–15 phút |
| R1 | 60 | 20 topology cục bộ ±25 m + 10 topology ngẫu nhiên tự do × {nine-state, refined}; bật Phase I | R1.6, R1.9, Ed.5 | 60 × 1–4 h, **nặng nhất** |

Thứ tự khuyến nghị: R11 → R2 → R5 → R8 → R7 → R3 → R4 → R6 → R1.

R6 với scenario có 29.142 mẫu cho mỗi target mỗi slot. Giải một lần với đủ 58k hàng
tuyến tính bằng SCS mất hơn 8 giờ cho một slot, vì vậy `solve_covariance_slot` giải
chương trình scenario bằng **sinh ràng buộc dần** (`SCENARIO_CG_*` trong `covariance.py`):
- Giải trên một working set (ban đầu là 32 mẫu có P_min lớn nhất mỗi target).
- Kiểm tra toàn bộ mẫu bằng a^H R a ≥ P_min.
- Thêm các mẫu vi phạm rồi giải lại.

Mỗi bài toán con là một nới lỏng của chương trình đầy đủ, nên nghiệm cuối cùng chính là
nghiệm của chương trình scenario đầy đủ, và chứng nhận của Prop. 3 không đổi.
Residual PCRB được tính lại trên toàn bộ mẫu.

Kiểm tra ở slot 24 với N = 150, giữa CG và giải đủ hàng:
- Công suất chênh nhau tương đối 1.2e-6.
- CG nhanh hơn 4.7 lần.

Với N_s = 29.142, CG cần 2 vòng và working set 64 + 44 mẫu, mất 172 s cộng 43 s tính
P_min. Kiểm tra lại bằng `python tools/check_scenario_cg.py 24 150 compare`.

## 5. Phân tích post-hoc (không tối ưu lại, chạy nhanh)

```bash
python run_revision.py posthoc lipschitz    # Lemma 4: L_w cho R7, T0, S9R       (R1.3)
python run_revision.py posthoc mismatch     # Rician K = 15/10/5 dB, CT 2/5 deg/s trên S9 T0 (R1.4)
python run_revision.py posthoc seeds        # 3 seed Monte Carlo độc lập          (R1.9)
python run_revision.py posthoc jointbound   # Prop. 5 trên mọi slot của T0, S9R
```

## 6. Xuất dữ liệu cho bài báo

```bash
python run_revision.py outputs --out paper_outputs
```

`paper_outputs/INDEX.csv` liệt kê từng file, ý phản biện tương ứng và vị trí trong bản thảo:

| File | Dùng cho |
|---|---|
| `T2_matched_baselines.csv` | Table II, kèm baseline fixed sensing robust (R5) và các baseline trajectory |
| `F2_csi_robustness.csv`, `F3_prior_risk.csv` | Fig. 2, Fig. 3 |
| `F4_rate_pcrb_and_certificates.csv` | Fig. 4 + chứng nhận Prop. 5 (cột `joint_bound_slot1_certified_w`) |
| `F5_mobility_decomposition.csv` | Fig. 5 + phân tách propulsion/RF, cận Cor. 1 |
| `T3_local_topologies.csv` | Table III |
| `R1_random_topologies_summary.csv`, `fig_R1_ucb_cdf.pdf` | Bảng mới: tỷ lệ đạt ρ kèm khoảng Clopper–Pearson (R1.6) |
| `R2_trajectory_baselines.csv`, `fig_trajectories_*.pdf` | Hình quỹ đạo + baseline (R1.7, R2.5) |
| `R3_solver_formulation.csv` | So sánh solver và kiểm chứng Prop. 2 trên cả nhiệm vụ (R3.3, R2.6) |
| `R4_rank_recovery.csv` | R3.3, R3.4 |
| `R6_reliability_remedies.csv` | Nine-state vs scenario vs refinement (Ed.4, R2.3) |
| `R7_tightening_lipschitz.csv`, `fig_R7_tightening.pdf` | R1.3, Lemma 4 |
| `R8_trajectory_sensitivity.csv` | R2.4 |
| `R9_model_mismatch.csv`, `R1_9_monte_carlo_seeds.csv` | R1.4, R1.9 |
| `convergence_iterations.csv` | Đường hội tụ (Ed.2, R1.10) |
| `runtime_breakdown.csv` | Thời gian theo từng pha: restoration, covariance, trajectory, audit, refinement (R2.6) |
| `lemma_checks.csv` | Kiểm tra số Lemma 2 và Lemma 3 cho từng run |

## 7. Những thay đổi hành vi cần biết

* **Bản sửa ngày 31/10:** baseline propulsion-only giờ dùng Powell nhiều điểm khởi đầu
  (`solver.propulsion_only_method="powell"`, 30 start). Lý do: hover là điểm dừng của P_h
  (dP_h/dv = 0 tại v = 0), nên SLSQP xuất phát từ hover không dịch chuyển, và R2 cũ báo
  propulsion-only = hover. Thêm `solver.seed_mode="propulsion_optimal"` (R12) để AO khởi
  tạo từ quỹ đạo tối ưu propulsion.
  - Các trường mới chỉ được đưa vào fingerprint khi khác giá trị mặc định. Vì vậy run R2
    `*_propulsion_only` giữ nguyên thư mục cũ và sẽ được **nạp lại kết quả cũ (hover)**.
    Phải chuyển hai thư mục đó ra chỗ khác trước khi chạy lại R2 (xem RESUME_UBUNTU.md).
  - `paper_outputs` có thêm `R12_propulsion_first.csv` (R12 cạnh S9 T0, S5A, S5B) và cột
    `propulsion_only_method` trong `R2_trajectory_baselines.csv` để nhận ra kết quả cũ.

* **Tiêu chí dừng** (`solver.stopping_rule="accepted_gain"`, mặc định): dừng khi update được chấp nhận gần nhất giảm năng lượng không quá 10⁻³ (tương đối). Lời giải covariance của candidate được dùng lại thay vì giải lại cùng trajectory. Candidate trùng với trajectory hiện tại thì không giải lại. Kết quả của mọi run cũ không đổi, vì không có gain nào vượt 10⁻³; số lượt giải covariance giảm từ 3 xuống còn 1–2 cho mỗi run. Muốn tái tạo đúng code cũ thì đặt `"stopping_rule": "legacy"`.
* **Run "hoàn tất nhưng không đạt audit"** (ví dụ Local-1) giờ được nạp lại thay vì bị tính lại.
* Các tính năng mới đều **tắt** ở cấu hình gốc và chỉ được bật trong R1–R11: `pcrb_formulation`, `pcrb_enforcement="scenario_certified"`, `refinement_enabled`, `joint_bound_check`, `candidate_screening`, `phase1_enabled`, `rank_recovery_mode`, `trajectory_method="random_search"`, `scheme.trajectory_objective="propulsion_only"`, `evaluation.rician_k_db`, `evaluation.target_turn_rate_dps`.
* Audit ghi thêm `lemma2_bcrb_check.csv`, `lemma3_closed_vs_open.csv` và các cột `time_*`, `wall_total_s`, `covariance_passes`.

Xem `docs/LOGIC_MAP.md` để biết hàm nào hiện thực mệnh đề nào. `docs/REMOVED_LEGACY.md`
liệt kê những gì đã bỏ khỏi gói cũ.
