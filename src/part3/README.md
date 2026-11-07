# F3 Future High-Impact — Code + Tài liệu

Tài liệu chính: **[F3_FUTURE_HIGH_IMPACT.md](F3_FUTURE_HIGH_IMPACT.md)** (đầy đủ mục tiêu, phân chia dữ liệu, công thức, ràng buộc, triển khai, đánh giá, giới hạn).

## Bắt đầu nhanh

Từ **thư mục gốc repo**, nơi đã có `data/processed/G_cite.graphml`, đặt `run_f3_future.py`, `make_demo_data.py`, thư mục `f3_future` và `requirements.txt` ở cùng cấp:

```powershell
pip install -r requirements.txt
python -m pytest -q
python run_f3_future.py --graph data/processed/G_cite.graphml --output outputs/f3_future
```

Chạy thử dữ liệu giả:

```powershell
python make_demo_data.py --output data/demo_G_cite.graphml
python run_f3_future.py --graph data/demo_G_cite.graphml --output outputs/demo --trees 50 --disable-novelty
```

Test: `T=2021 -> nhãn đến 2024`, không phải bài báo xuất bản năm 2024. Nếu dữ liệu thật không bao phủ đến 2024, cần đổi cấu hình.
