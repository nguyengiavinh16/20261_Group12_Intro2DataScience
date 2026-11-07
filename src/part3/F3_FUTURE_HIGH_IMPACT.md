# F3 — Early Prediction & Ranking of Future High-Impact Research Papers

> **Tài liệu thiết kế và hướng dẫn triển khai** (Python 3.10+, Windows PowerShell).  
> Phạm vi: **bài toán 2 của F3**, được đổi tên từ *Breakthrough/Seminal Detection* thành **Future High-Impact Paper Prediction & Ranking**. Không đưa ra kết luận rằng một bài báo có đóng góp phương pháp mang tính đột phá.

## 1. Mục tiêu và câu hỏi nghiên cứu

**Câu hỏi:** Với thông tin sẵn có tại thời điểm chốt dữ liệu `T`, liệu có thể dự đoán và xếp hạng những công trình **mới xuất bản** sẽ có **ảnh hưởng trích dẫn cao** trong `Δ=3` năm sau, so với các bài cùng **năm xuất bản và lĩnh vực**?

Đây là dự đoán một **đại lượng có thể quan sát** (citation impact trong tập dữ liệu), **không phải** đo chất lượng khoa học nội tại, tính mới về lý thuyết hay “đột phá” trong phần Method.

Hệ thống có hai đầu (dual-head):

- **Growth head:** `LGBMRegressor(objective='huber')` dự đoán tăng trưởng trích dẫn logarit trong ba năm.
- **Ranking head:** `LGBMRanker(objective='lambdarank')` xếp hạng bài báo theo số trích dẫn tích lũy tương lai, được chuẩn hóa thành mức relevance trong cùng nhóm `publication_year × main_domain`.

Kết quả chính là bảng xếp hạng **trong từng nhóm năm–lĩnh vực** và các chỉ số NDCG@10/20, MAP@20, Precision@Top-5%. Chạy cả hai head cho phép ablation và hợp nhất điểm số.

## 2. Dữ liệu đầu vào và những gì được giả định

### 2.1. Nguồn dữ liệu

Hai loại JSONL ban đầu:

**Metadata:** `paperId`, `title`, `abstract`, `year`, `conference` (venue), `raw_keywords` (nếu có).  
**References:** `paperId`, `references=[cited_paper_id,...]`.

**Đầu vào trực tiếp của mã F3:** `data/processed/G_cite.graphml` **đã tạo ở pipeline tiền xử lý trước**. Mã không chạy lại bước MinHash/Entity Resolution hay `G_flow`, vì F3 chỉ cần thông tin từ `G_cite`. Cấu trúc node:

| Thuộc tính | Kiểu | Bắt buộc khi chạy F3? |
|---|---|---|
| `paper_id` (hoặc node ID) | string | Có |
| `title` | string | Có thể rỗng, x17 sẽ kém tin cậy |
| `abstract` | string | Có thể rỗng, x17 sẽ kém tin cậy |
| `venue` | string | Có thể rỗng, x15 dùng prior |
| `year` | int | **Có**: node thiếu năm bị bỏ |
| `raw_keywords` | list[str] hoặc chuỗi JSON list | Không, mặc định `[]` |

`G_cite` là đồ thị có hướng: `u -> v` nghĩa là **bài u trích dẫn bài v**. `year` trên GraphML có thể được ghi thành chuỗi và sẽ được ép về số nguyên.

> Lưu ý đặc biệt: dataset của bạn hiện có 130.692 node và 2.471.854 cạnh sau khi gộp trùng. Mã đã kiểm tra được bằng dữ liệu giả lập, **chưa được đo thời gian chạy với tập thật**; GraphML NetworkX và k-core có thể cần nhiều RAM/CPU.

### 2.2. Quy tắc quan sát cạnh (bắt buộc ghi vào báo cáo)

Hai file JSONL không có *timestamp thực tế của mỗi sự kiện trích dẫn*. Vì vậy triển khai **xấp xỉ** thời điểm quan sát cạnh bằng **năm xuất bản của bài báo trích dẫn**:

\[
V_T=\{u:year(u)\le T\},\qquad
E_T=\{(u,v)\in E_{cite}:year(v)\le year(u)\le T\}.
\]

Nếu metadata ghi bài A năm 2021 trích dẫn B năm 2018, quan hệ được coi là xuất hiện ở snapshot 2021. Các cạnh "bài cũ trích dẫn bài tương lai" bị loại khỏi snapshot. Không sử dụng trường `citationCount` của Semantic Scholar để tạo các snapshot lịch sử vì trường đó thường là số trích dẫn tại **thời điểm thu thập**.

Đây là **publication-year-based retrospective benchmark**: kết quả không bảo đảm mô phỏng chính xác các lần cập nhật reference hoặc metadata sau khi bài báo được xuất bản. Việc Entity Resolution đã làm trên toàn bộ dữ liệu cũng có thể ảnh hưởng thông tin thời gian; muốn nghiêm ngặt tuyệt đối cần metadata version/as-of-time.

## 3. Strict Temporal Split: quy tắc chung cho 5 tác vụ

Mục 0.2 của dự án có thể duy trì **giao thức toàn cục**:

1. Tại snapshot `T`, chỉ nhìn thấy node/cạnh/thuộc tính khả dụng đến `T`.
2. Không tính degree, PageRank, HITS, k-core, tốc độ trích dẫn từ graph tương lai.
3. Không fit TF-IDF/SVD/scaler hay target encoder từ các dữ liệu chưa được phép dùng.
4. Mô hình và siêu tham số phải được quyết định trước khi mở nhãn Test.
5. Nếu dùng đầu ra F1/F2 làm đặc trưng F3, mô hình sinh đầu ra đó cũng phải có provenance và mốc quan sát hợp lệ.
6. Năm chia dữ liệu dùng cho các tác vụ khác có thể vẫn là train `<=2021`, validation `2022–2023`, test `2024–2026`. **F3 có benchmark riêng** vì nhãn F3 cần đủ ba năm mới trưởng thành; không nên ép trùng năm của mẫu huấn luyện giữa mọi tác vụ.

Đặc biệt, **split theo thời gian xuất bản không đồng nghĩa với split theo thời điểm nhãn có sẵn**.

## 4. Temporal split dành riêng cho F3

### 4.1. Giao thức mặc định được cài đặt

| Tập | Snapshot features `T` | Chốt nhãn `T+3` | Năm xuất bản ứng viên | Thời điểm nhãn có thể dùng |
|---|---:|---:|---|---|
| Train | **2015** | **2018** | 2013, 2014, 2015 | Cuối 2018 |
| Validation | **2018** | **2021** | 2016, 2017, 2018 | Cuối 2021 |
| Test | **2021** | **2024** | 2019, 2020, 2021 | Cuối 2024 |

Chốt mốc năm tại **cuối năm** để tránh mơ hồ. Tập ứng viên của mỗi snapshot là:

\[
\mathcal U_T=\{u:T-2\le year(u)\le T\}.
\]

**Lưu ý quan trọng:** Graph tại thời điểm `T` chứa **tất cả** bài đến `T`, không chỉ bài trong `\mathcal U_T`. Nhờ vậy các bài cũ vẫn có thể đóng vai trò bài tham khảo/được trích dẫn khi tạo features cho ứng viên mới. Nhưng **chỉ ứng viên** mới trở thành dòng của dataset dự đoán.

### 4.2. Tại sao không dùng train=2019/label=2022, val=2020/label=2023, test=2021/label=2024?

Vì mô hình dùng tại snapshot validation 2020 không thể đã được học bằng nhãn train của năm 2022. Để benchmark mô phỏng dự đoán chronologically strict, cần:

\[
T_{train}+\Delta\le T_{val},\quad
T_{val}+\Delta\le T_{test},\quad
T_{test}+\Delta\le T_{max,observed}.
\]

Mặc định: `2015+3=2018`, `2018+3=2021`, `2021+3=2024`. Nếu ngày thu thập dữ liệu không đủ đến 2024, **không thể đánh giá nhãn Test đầy đủ** dù tệp có tên các năm tương lai.

### 4.3. Chống chồng lấn mẫu

Các cohort năm xuất bản không giao nhau: `2013–2015`, `2016–2018`, `2019–2021`. Cùng một bài báo cũ có thể **xuất hiện trong đồ thị ngữ cảnh** của nhiều snapshot mà **không trở thành nhiều mẫu có nhãn**. Đây là cách làm hợp lệ với bài toán citation-graph có tính transductive, miễn không lộ thông tin tương lai.

### 4.4. Các mốc là cấu hình, không phải giả định dữ liệu thật

Dùng `--train-T`, `--val-T`, `--test-T`, `--horizon`, `--candidate-window`. Chương trình từ chối cấu hình nếu nhãn train chưa khả dụng trước val, nhãn val chưa khả dụng trước test hoặc năm mới nhất trong graph nhỏ hơn `test-T+horizon`. **Mã không tự bịa citation count của tương lai.**

## 5. Xây dựng nhãn (không đồng nhất với "breakthrough")

### 5.1. Số trích dẫn tích lũy

\[
C_T(u)=|\{w:(w\to u)\in E_T\}|.
\]

Để tiết kiệm RAM, mã đếm cạnh trích dẫn có `year(w) <= T+3` trực tiếp trên full graph, **không tạo bản sao toàn bộ graph tương lai** khi tính nhãn.

### 5.2. Head 1 — Citation Growth Regression

\[
y_{growth}(u,T)=\log(1+C_{T+3}(u))-\log(1+C_T(u)).
\]

Lượng này khác số trích dẫn tăng tuyệt đối `C_(T+3)-C_T`. Mục tiêu hồi quy là số thực, dùng Huber regression để giảm ảnh hưởng của phân phối lệch phải.

### 5.3. Head 2 — Cohort-Normalized High-Impact Ranker

Mỗi **query group**:

\[
q(u)=(year(u), main\_domain(u)).
\]

Với các bài thuộc cùng cohort, tính `y_pct` là **percentile mid-rank** của `C_(T+3)`. Phân vị được tính trong **tập ứng viên cùng split**, không dùng bài Test khi tạo nhãn Train. Trường hợp nhiều bài có cùng citation count, dùng `rank(method='average', pct=True)` để tránh phá hòa tuỳ tiện bằng paper ID.

**Nhãn nhị phân:** `y_high_impact = 1` khi `y_pct > 0.95` (Top 5% theo phân vị, không phải đúng tuyệt đối 5% trong trường hợp hòa).

**Relevance 5 mức cho LambdaMART:**

| Percentile | `y_grade` |
|---|---:|
| `<=50%` | 0 |
| `(50%, 75%]` | 1 |
| `(75%, 90%]` | 2 |
| `(90%, 95%]` | 3 |
| `>95%` | 4 |

**Không loại Survey ra khỏi nhãn:** bài survey có thể có ảnh hưởng trích dẫn lớn và không phải mặc định bị loại. Đặc trưng P_survey chỉ được dùng nếu cung cấp prediction từ F1-A hợp lệ.

**Nhóm nhỏ / quá nhiều ties:** mặc định chỉ xếp hạng và đánh giá nhóm có ít nhất `--min-group 10` bài. Nếu toàn bộ nhóm có cùng `C_(T+3)`, nhãn High-Impact có thể toàn 0; mô hình ranker bỏ nhóm **không có biến thiên mức relevance** ở train. MAP@20 bỏ qua những nhóm không có positive và báo số nhóm có positive; NDCG bỏ các nhóm gain lý tưởng bằng 0. Không được diễn giải precision như "chính xác tuyệt đối Top 5%" nếu có ties lớn.

## 6. Xác định `main_domain` khi thiếu raw_keywords

`raw_keywords` thiếu ở đa số node, nên không thể coi tất cả node đều có nhãn lĩnh vực chuẩn. **Ưu tiên cung cấp CSV do nhóm gán nhãn/thu thập từ nguồn tin cậy**:

```csv
paper_id,main_domain
0034c8439a1329ef9c0d4de7a30a3cfea7f954f8,CV
...
```

Chạy với `--domain-map path/to/domains.csv`. Khi không có file này, code sử dụng **heuristic minh bạch**: raw_keywords → venue → title với nhóm `CV`, `NLP`, `ML`, `Robotics`, `DataMining`, `Theory`, `Other`. Đây **chỉ là bản chạy được đầu tiên**, không phải bộ gán nhãn lĩnh vực đã được thẩm định. Cần thống kê nhóm `Other`, xem tỷ lệ gán nhãn sai và tiến hành ablation hoặc thay bằng domain classifier train-only nếu kết quả bị ảnh hưởng.

Không sử dụng abstract/full-text tương lai để học bộ phân loại lĩnh vực tại snapshot quá khứ.

## 7. Mô tả 18 đặc trưng dự báo (tất cả tính trên `G_cite(<=T)`)

Quy ước `u→v` nghĩa là u trích dẫn v. `Cit_in_year(t,u)` là số **bài có year=t** trích dẫn u (cạnh phải tồn tại ở snapshot T).

| Feature | Công thức/triển khai | Trạng thái |
|---|---|---|
| `x01` | `log(1+in_degree_T(u))` | Có |
| `x02` | `log(1+out_degree_T(u))` | Có |
| `x03` | `log(1e-8+PageRank_T(u))`, `alpha=.85` | Có |
| `x04` | `log(1e-8+CiteRank_T(u))`, teleport `p_i∝exp(-(T-year_i)/3)` | Có, 50 power iterations |
| `x05` | HITS Authority, `A^T A` power iteration | Có |
| `x06` | k-core trên phiên bản **vô hướng** của citation graph | Có |
| `x07` | `Cit_in_year(T,u)-Cit_in_year(T-1,u)` | Có |
| `x08` | `(Cit_in_year(T,u)+Cit_in_year(T-1,u))/2` | Có |
| `x09` | `Cit_in_year(T,u)-2*Cit_in_year(T-1,u)+Cit_in_year(T-2,u)` | Có |
| `x10` | `in_degree_T(u)/max(1,T-year(u)+1)` | Có |
| `x11` | `(Citations in publication year + next year if already observed)/(in_degree_T(u)+1)` | Có |
| `x12` | Tổng `weight_F1(w→u)` của bài đang trích dẫn u | **Optional**: mặc định 0 nếu thiếu F1 |
| `x13` | `max_{v∈Ref(u)} PageRank_T(v)`, rỗng→0 | Có |
| `x14` | Trung bình `x08(v)` với `v∈Ref(u)`, rỗng→0 | Có |
| `x15` | **Thay đổi có chủ ý:** historical venue prestige, Bayes smoothing `m=20` trên `log1p(in_degree)` của bài xuất bản `<=T-3` | Có, **không phải Venue Target Encoding y_growth** |
| `x16` | Entropy `-Σ_d p_d log p_d` của các domain của bài trích dẫn u | Có |
| `x17` | `1-max_{w:year(u)-2<=year(w)<=year(u)-1} cosine(LSA(u),LSA(w))` | Có, trừ khi tắt novelty |
| `x18` | Xác suất Survey từ F1-A, đầu vào ngoài | **Optional**: mặc định 0 |

**Cần hiểu đúng x15:** đặc tả cũ định nghĩa x15 là *Out-of-Fold Venue Target Encoding* dựa trên `y_growth`. Với chỉ một cohort train (snapshot 2015, nhãn chỉ biết đến 2018), việc tạo historical label-driven feature cho snapshot 2015 đòi hỏi dữ liệu nhãn của những cohort còn cũ hơn. Để không lén dùng nhãn năm 2018 trong feature ở năm 2015, mã thay bằng **historical venue prestige proxy không dùng nhãn tương lai**. Đây là **thay đổi phương pháp có ghi nhận**, không được báo cáo là đã cài đặt y_growth OOF venue target encoder theo nguyên bản.

**x12 và x18:** Nếu bạn chưa có F1 thì hai cột này luôn bằng 0. Để chạy thực sự với đủ 18 đặc trưng độc lập, cần cung cấp kết quả F1 chỉ dựa trên thông tin tại T. Vì trọng số F1 có thể phụ thuộc mô hình riêng, file F1 tự cung cấp **không đủ để chứng minh temporal safety**; kiểm tra thời điểm huấn luyện nó trước.

**x17:** TF-IDF và TruncatedSVD chỉ fit trên corpus bài báo có năm `<=2015`; cả Val/Test dùng lại đúng bộ biến đổi. Độ mới tính so với hai năm *trước năm công bố*, không dùng bài cùng năm hoặc sau năm công bố. Chạy `--disable-novelty` khi cần kiểm thử nhanh; khi đó `x17=0` và phải ghi rõ ablation trong báo cáo.

`PageRank`, `CiteRank`, HITS dùng sparse-matrix power iteration; `k-core` dùng NetworkX. `G_flow` không cần thiết cho tập 18 đặc trưng này, nhưng vẫn có thể được dùng cho F1/F2 và nghiên cứu tiếp.

## 8. Mô hình và baseline

### 8.1. Heuristic (không dùng nhãn)

- H1: In-Degree `x01`.
- H2: PageRank `x03` và HITS Authority `x05`.
- H3: CiteRank `x04`, time-decay `tau=3`.

### 8.2. ML Baselines

Mặc định: `RobustScaler+Ridge(alpha=10)`, `RobustScaler+ElasticNet(alpha=.05,l1_ratio=.5)`, `RobustScaler+LogisticRegression(class_weight='balanced')` (nếu train có cả hai nhãn).

`--extra-baselines`: Random Forest Regressor (400 cây), Extra Trees Classifier (400 cây), SVR-RBF (C=2, epsilon=.1) trên **tối đa 8.000** mẫu train để hạn chế độ phức tạp bậc hai. **Không cài Ranking SVM** và không chạy Optuna tự động trong bản này; đó là những thực nghiệm mở rộng riêng. Những baseline dự đoán `y_growth` được dùng như score xếp hạng, không mặc định là classifier.

### 8.3. Dual-Head chính

- **Growth:** `LGBMRegressor(objective='huber', alpha=0.9, n_estimators=400, learning_rate=.03, num_leaves=31, subsample=.8, subsample_freq=1, colsample_bytree=.8)`.
- **Rank:** `LGBMRanker(objective='lambdarank', n_estimators=400, learning_rate=.03, num_leaves=31, min_child_samples=15, label_gain=[0,1,3,7,15])`. Query group được sắp liên tiếp đúng yêu cầu của LightGBM; `sum(group)==n_training_rows`.

Đặc tả cũ dùng `alpha=1.35` cho Huber; bản cài đặt chọn `alpha=.9` là giá trị phổ biến và kiểm tra với LightGBM 4.x. Trong LightGBM, `alpha` là tham số của objective Huber, **không nên hiểu máy móc là ngưỡng Huber δ** trong công thức loss báo cáo.

### 8.4. Fusion không sử dụng query Q

Vì F3 đang xếp hạng **trong cohort năm–lĩnh vực** chứ chưa có truy vấn văn bản người dùng `Q`, mã **không giả tạo `Sim_Hybrid(Q,u)`**. Điểm hợp nhất:

\[
S(u)=\lambda\cdot PercentileGroup(s_{rank}(u))
+(1-\lambda)\cdot PercentileGroup(\hat y_{growth}(u)).
\]

Chọn `λ ∈ {0,.25,.5,.6,.75,1}` bằng NDCG@20 trên **Validation**, sau đó **đóng băng λ và đánh giá Test đúng một lần**. PercentileGroup chỉ dùng dự đoán model và danh sách ứng viên cùng group, **không đọc nhãn**.

## 9. Đánh giá benchmark

Đánh giá **macro average qua các cohort năm–lĩnh vực đủ lớn** (mỗi group có trọng số ngang nhau, không để domain lớn lấn át domain nhỏ):

| Chỉ số | Mục đích |
|---|---|
| NDCG@10, NDCG@20 | Ranking graded relevance 0–4, gain `2^rel-1` |
| MAP@20 | Average Precision tại 20 theo nhãn High-Impact (`denominator=min(#positive,20)`) |
| Precision@Top-5% | Trong k=ceil(0.05*n_group) bài đứng đầu, tỷ lệ nhãn High-Impact |
| Spearman ρ, Kendall τ | Tương quan thứ bậc giữa score và `C_(T+3)` thực tế |
| RMSE, MAE, Huber Loss | Đo riêng chất lượng hồi quy `y_growth` |

Không báo chỉ số NDCG/MAP giả nếu group không có nhãn đủ đa dạng; số nhóm được tính và số nhóm có positive được ghi riêng. Ở nhóm rất nhỏ, NDCG@20 thực tế chỉ dùng số bài trong group.

**Không đưa Wilcoxon Signed-Rank Test vào bản chạy mặc định.** Nếu muốn kiểm định có ý nghĩa thống kê, cần lưu score theo từng group cho từng phương pháp và kiểm định theo cùng một tập group, báo số nhóm, mức ý nghĩa và cách hiệu chỉnh nhiều phép thử.

## 10. Cấu trúc mã nguồn

```text
20261_Group12_Intro2DataScience/
├── data/
│   └── processed/
│       └── G_cite.graphml       # đã có từ pipeline trước
├── f3_future/
│   ├── __init__.py
│   ├── data.py                  # GraphML, domains, snapshot, as-of counts
│   ├── labels.py                # y_growth, y_pct, y_grade, y_high_impact
│   ├── features.py              # x01..x18, PageRank/CiteRank/HITS/LSA
│   ├── evaluation.py            # NDCG, MAP, Precision, Spearman, Kendall
│   └── modeling.py              # baselines + dual-head + validation fusion
├── run_f3_future.py             # lệnh chạy toàn pipeline
├── make_demo_data.py            # tạo graph GIẢ để smoke-test
├── requirements.txt
└── F3_FUTURE_HIGH_IMPACT.md
```

Bạn có thể **copy các file từ gói ZIP vào thư mục gốc repo** (nơi có `data/processed/G_cite.graphml`), không cần chỉnh các file `src/process_data` trước đó.

## 11. Hướng dẫn chạy chi tiết trên Windows PowerShell

### Bước 1 — Chuẩn bị môi trường

Tại thư mục gốc repo:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu đã dùng `.venv` của dự án có sẵn các thư viện tương thích, không bắt buộc tạo môi trường mới.

### Bước 2 — Test code nhỏ (không đụng vào dữ liệu thật)

```powershell
python -m pytest -q
python make_demo_data.py --output data/demo_G_cite.graphml --papers-per-year 90
python run_f3_future.py --graph data/demo_G_cite.graphml --output outputs/f3_demo --trees 50 --disable-novelty
```

Demo có các bài giả xuất bản giai đoạn 2010–2024; các chỉ số chỉ xác nhận pipeline chạy, **không phải kết quả học thuật thật**. Bỏ `--disable-novelty` nếu muốn kiểm tra cả TF-IDF/LSA.

### Bước 3 — Chạy dữ liệu thật

```powershell
python run_f3_future.py --graph data/processed/G_cite.graphml --output outputs/f3_future
```

Nếu máy có ít RAM hoặc muốn xác minh pipeline trước:

```powershell
python run_f3_future.py --graph data/processed/G_cite.graphml --output outputs/f3_quick --trees 100 --disable-novelty
```

**Không** xem kết quả `--disable-novelty` là benchmark đủ 18 đặc trưng; nó là ablation bỏ x17.

### Bước 4 — Tùy chọn cung cấp nhãn domain/F1

```powershell
python run_f3_future.py `
  --graph data/processed/G_cite.graphml `
  --domain-map data/processed/domains.csv `
  --f1-weights data/processed/f1_salience.csv `
  --survey-scores data/processed/survey_scores.csv `
  --output outputs/f3_future
```

Định dạng:

```csv
# domains.csv
paper_id,main_domain
paperA,CV
paperB,NLP
```

```csv
# f1_salience.csv
source,target,weight
paperA,paperB,0.9
```

```csv
# survey_scores.csv
paper_id,p_survey
paperA,0.03
```

(Dòng bắt đầu `#` chỉ để chú thích trong tài liệu, **không được đưa vào file CSV thực**.)

### Bước 5 — Chạy thêm baseline nặng

```powershell
python run_f3_future.py --graph data/processed/G_cite.graphml --extra-baselines --output outputs/f3_full
```

Không có GPU bắt buộc. NetworkX và phần lớn feature engineering chạy CPU; RTX 3060 không được sử dụng trong pipeline này.

### Bước 6 — Tùy chỉnh thời gian khi dataset không phù hợp

Ví dụ nếu có dữ liệu đến 2025, các mốc train=2016, val=2019, test=2022 cũng thỏa điều kiện, nếu các cohort có đủ số lượng:

```powershell
python run_f3_future.py --train-T 2016 --val-T 2019 --test-T 2022 --horizon 3
```

## 12. File đầu ra

```text
outputs/f3_future/
├── dataset_T2015.csv                 # features tại 2015 + labels chốt 2018
├── dataset_T2018.csv                 # features tại 2018 + labels chốt 2021
├── dataset_T2021.csv                 # features tại 2021 + labels chốt 2024
├── lightgbm_growth.joblib
├── lightgbm_ranker.joblib
├── ridge.joblib
├── elasticnet.joblib
├── logistic_high_impact.joblib       # chỉ có nếu Train có 2 lớp
├── predictions_validation.csv
├── predictions_test.csv
├── cohort_diagnostics.csv
├── models_config.json
├── run_config.json
└── metrics.json
```

Các `dataset_T*.csv` chứa `paper_id`, `publication_year`, `main_domain`, `snapshot_year`, `c_t`, `c_t3`, `y_growth`, `y_pct`, `y_grade`, `y_high_impact`, `group_id`, `group_size`, `eligible_rank_group`, `x01...x18`.

**Cấm** đưa các cột `c_t3`, `y_growth`, `y_grade`, `y_high_impact`, `y_pct` hoặc `group_id` (như một target encoding) vào model input. Mã chỉ đưa **18 cột x01..x18** vào mô hình; `group_id` chỉ để nhóm các query cho LambdaMART và đánh giá.

`predictions_test.csv` lưu đầy đủ các score heuristic, score ML, score của hai head và score_fused để phân tích, so sánh hoặc vẽ biểu đồ.

## 13. Kiểm tra bắt buộc trước khi kết luận khoa học

- [ ] GraphML có đầy đủ metadata và năm bao phủ ít nhất đến 2024.
- [ ] Metadata của các tài liệu được trích dẫn không quá thiếu; số trích dẫn tính được **chỉ** thuộc tập node đang có.
- [ ] Không dùng `citationCount` toàn kỳ làm `c_t` tại snapshot lịch sử.
- [ ] Không có feature nào phụ thuộc cạnh tương lai.
- [ ] `main_domain` có chất lượng tốt, tránh toàn `Other`.
- [ ] Cohort có đủ bài và có nhãn High-Impact không quá thưa.
- [ ] Không sử dụng nhãn Test để chọn λ, siêu tham số, mô hình hay ngưỡng.
- [ ] Nếu dùng F1-A/F1 salience, phải xác minh mô hình nguồn không học từ tương lai.
- [ ] Chỉ dùng kết quả benchmark thật sau khi chạy trên dữ liệu thật; các số demo không có giá trị khoa học.
- [ ] Báo cáo tỷ lệ bài báo không citation trong 3 năm và mức độ thiếu dữ liệu.

## 14. Các giới hạn và những phần cố ý chưa cài đặt

1. **Lịch sử cạnh trích dẫn chỉ xấp xỉ theo năm xuất bản.** Không thể bảo đảm tuyệt đối as-of-time nếu dữ liệu được chỉnh sửa về sau.
2. **Entity Resolution toàn kỳ** có thể gây sai khác metadata theo snapshot; nếu muốn benchmark lịch sử nghiêm ngặt phải duy trì metadata theo phiên bản và thời gian.
3. **Domain từ heuristic** chỉ là fallback cho dữ liệu thiếu raw_keywords; nên thay bằng nhãn domain đáng tin cậy.
4. `x12` và `x18` là **feature hook** chứ không phải các mô hình F1 được mã này huấn luyện.
5. `x15` là venue prestige proxy không giám sát; **không phải** OOF venue target encoding của đặc tả cũ.
6. `--extra-baselines` không bao gồm Ranking SVM; SVR dùng sample cap 8.000.
7. Chưa có full nested Optuna, xác suất calibration, confidence interval hoặc Wilcoxon; các phần này cần thêm thiết kế thực nghiệm.
8. Fusion không dùng `Sim_Hybrid(Q,u)` vì bài toán chưa có truy vấn `Q`; nếu muốn query-aware ranking cần thêm bước retrieval trước khi áp dụng F3.
9. **Mô hình không chứng minh “breakthrough”, “seminal” hay tính mới ở cấp phương pháp.** Nó chỉ dự đoán ảnh hưởng **trong mạng trích dẫn quan sát được**.
10. Với đồ thị ~2.5 triệu cạnh, RAM/CPU có thể là nút thắt. Chương trình tránh copy snapshot ở `T+3`, nhưng NetworkX core-number, GraphML loading và LSA vẫn tiêu tốn tài nguyên. Nếu thiếu RAM, cân nhắc chuyển phần graph sang sparse-native pipeline và lưu snapshots trung gian.

## 15. Tài liệu kỹ thuật tham khảo

- LightGBM `LGBMRanker`, tham số `group`, `label_gain`, `eval_at`: https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html
- LightGBM parameter reference (`huber`, `lambdarank`, `alpha`): https://lightgbm.readthedocs.io/en/latest/Parameters.html
- scikit-learn `ndcg_score`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html

**Kết luận:** Đây là F3 *Future High-Impact* dạng thực nghiệm hoàn chỉnh cho hai đầu dự đoán + baseline có thể chạy, nhưng *chỉ* là hệ thống dự đoán ảnh hưởng trích dẫn. Muốn tuyên bố benchmark có tính khoa học cao, cần đặc biệt kiểm tra provenance thời gian và chất lượng nhãn lĩnh vực.
