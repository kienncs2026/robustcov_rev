# Revision 2: baseline công bằng và seed độc lập

Bốn tệp mới nằm trong `tools/`, không sửa mã nguồn `src/`:

| Tệp | Việc làm |
|---|---|
| `fair_baselines.py` | Cận dưới năng lượng đẩy có chứng nhận (LP) và hai baseline không dùng thông tin ISAC |
| `run_fair_baselines.py` | Dựng baseline, giải covariance robust và audit trên chính đường bay đó |
| `run_seed_study.py` | Chạy lại thiết kế level propulsion-first với nhiều seed độc lập; tùy chọn chạy lại thiết kế AO tham chiếu |
| `summarize_rev2.py` | Tổng hợp thành `results_rev2/summary_rev2.md`, `.csv` và `table_rev2.tex` |

## Cách chạy (từ thư mục `robustcov_rev/`)

```bash
pip install -r requirements.txt          # cần cvxpy, clarabel, scs như các lần chạy trước
bash tools/run_rev2.sh 4                  # 4 = số tiến trình song song
```

Có thể chạy riêng từng phần như sau:

```bash
python tools/run_fair_baselines.py --headings 0 90 180 270     # khoảng 15-30 phút
python tools/run_seed_study.py --seeds 0 1 2 3 4 --jobs 4       # phần nặng nhất
python tools/run_seed_study.py --seeds 0 1 2 --missions reference --ao-reference --jobs 3
python tools/summarize_rev2.py
```

Các task đã xong sẽ được bỏ qua khi chạy lại, nên nếu bị dừng giữa chừng thì chỉ cần chạy lại cùng lệnh. Muốn thử nhanh xem môi trường có chạy được không, dùng `--multistart 1 --seeds 0`.

## Thời gian ước tính

Ước tính dựa trên log trên máy của tác giả:
- **Một task level:** 25 phút (tham chiếu) đến 60 phút (đi rồi về) cho phần tìm propulsion, cộng 2–11 phút giải covariance.
- **Năm seed cho hai nhiệm vụ:** khoảng 2 giờ với 4 tiến trình.
- **`--ao-reference`:** mỗi seed khoảng 70 phút (SCS, dạng Schur, như Table II).

## Kết quả đã có, kiểm tra trên máy khác (không cần solver conic)

| Nhiệm vụ | Cận dưới LB | Baseline công bằng | Thiết kế level trong bài |
|---|---:|---:|---:|
| Tham chiếu (50→550 m) | 24.310,4 J | Đường thẳng tối ưu tốc độ: 24.483,9 J | 24.485,6–24.486,6 J |
| Đi rồi về | 23.919,6 J | Vòng tròn R = 42,5 m, tối đa 13,4 m/s: 25.985,3 J | 25.816,4–25.961,8 J |

- **LB** là một cận đúng cho mọi quỹ đạo thỏa điều kiện biên và giới hạn động học. Vì vậy, với bất kỳ đường bay θ nào, `E_tot(θ) − E* ≤ E_tot(θ) − LB` (Corollary 2).
- **Với nhiệm vụ tham chiếu**, cận này chặt: khoảng cách dưới 1%.
- **Với nhiệm vụ đi rồi về**, cận lỏng (khoảng 8%), vì LP bỏ qua ràng buộc phải quay đầu để trở về. Ở nhiệm vụ này, baseline vòng tròn là đối chứng độc lập.

## Cần gửi lại sau khi chạy

Nén thư mục `results_rev2/`. Trong đó `summary_rev2.md` là bản tóm tắt; các tệp `*.json` và `*.npz` là dữ liệu và quỹ đạo dùng để vẽ hình.

## Ghi chú khi chạy trên server (02/11/2026)

- Phải đặt `RAYON_NUM_THREADS=1` (CLARABEL mở một luồng mỗi lõi, xem `RESUME_UBUNTU.md` mục 11).
  `run_rev2.sh`, `rev2_common.py` và `run_queue.py` đã đặt sẵn.
- Task level bị ngắt sau bước Powell sẽ tái dùng `.npz` đã lưu; baseline đã có `.json` được bỏ qua.
- `--ao-reference` chạy 5 seed trong khoảng 13 phút mỗi seed, vì AO từ seed thẳng không chấp nhận
  cập nhật quỹ đạo nào (dịch chuyển 0 m); seed chỉ ảnh hưởng audit Monte Carlo.
