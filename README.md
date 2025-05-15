## 🎒 Dự án: Dự đoán giá cặp (Backpack Price Prediction)

Dự án này nhằm xây dựng một hệ thống Machine Learning để dự đoán giá bán của các loại cặp dựa trên thông tin mô tả như thương hiệu, chất liệu, màu sắc, số ngăn, khả năng chống nước,... Dữ liệu được lấy từ cuộc thi Backpack Prediction Challenge trên Kaggle. Dự án sử dụng các mô hình hồi quy như Linear Regression, XGBoost và LightGBM, kèm theo quy trình xử lý dữ liệu, đánh giá K-Fold và lưu mô hình.

## 📄 Mô tả dữ liệu

Bộ dữ liệu được lấy từ cuộc thi Kaggle có tên **Backpack Prediction Challenge**, tổ chức từ ngày 01/02/2025 đến 01/03/2025.

Link: https://www.kaggle.com/competitions/playground-series-s5e2

### 📌 Các đặc trưng trong dữ liệu:

- `id`: Số thứ tự định danh của chiếc cặp.
- `Brand`: Thương hiệu (ví dụ: `Nike`, `Adidas`, `Jansport`).
- `Material`: Vật liệu chính (ví dụ: `Polyester`, `Nylon`, `Canvas`).
- `Size`: Kích cỡ của cặp (`Small`, `Medium`, `Large`).
- `Compartments`: Số ngăn chứa (`1` đến `10`).
- `Laptop Compartment`: Có ngăn đựng laptop không (`Yes`/`No`).
- `Waterproof`: Có khả năng chống nước không (`Yes`/`No`).
- `Style`: Kiểu dáng (ví dụ: `Backpack`, `Tote`, ...).
- `Color`: Màu sắc (ví dụ: `Red`, `Blue`, `Green`, ...).
- `Weight Capacity`: Trọng lượng tối đa có thể chứa (kg).
- `Price` (**target**): Giá bán của chiếc cặp (USD).

### 📌 Nhận xét

- Nhiều giá trị NULL ở một số cột, cần có **phương pháp phù hợp** để xử lý.
- Các đặc trưng **đa dạng** kiểu dữ liệu bao gồm **chuỗi**, **số** đến **nhị phân**.

## 🎯Mô tả bài toán

### 🎯 Mục tiêu bài toán

Bài toán đặt ra là dự đoán giá bán của một chiếc cặp dựa trên các đặc trưng đầu vào, bao gồm thương hiệu, chất liệu, kích thước, màu sắc, kiểu dáng, v.v.

Do giá là **một biến liên tục** (continuous variable), đây là bài toán hồi quy (regression). Mục tiêu là xây dựng mô hình có khả năng **ước lượng giá** (Price) gần đúng với **giá thực tế** trong dữ liệu kiểm tra.

- **Input**: Một dòng dữ liệu mô tả chiếc cặp với các đặc trưng như Brand, Material, Size, ...
- **Output**: Giá dự đoán (Price) – đơn vị USD.

### 🎯 Lợi ích bài toán và ứng dụng

- ✅ Với gần **300.000 mẫu dữ liệu** và **11 đặc trưng**, mô hình có thể học tốt và dự đoán hiệu quả trên tập kiểm tra.
- ✅ Bài toán dự đoán giá cặp có ứng dụng thực tiễn trong **thương mại điện tử và bán lẻ**, hỗ trợ định giá phù hợp.
- ✅ Dataset bao gồm đa dạng các loại đặc trưng: **chuỗi, số lượng, nhị phân**, giúp khai thác toàn diện các yếu tố ảnh hưởng đến giá.
- ✅ Quy mô dữ liệu lớn cho phép thử nghiệm nhiều mô hình khác nhau, từ đó **tinh chỉnh và chọn mô hình tối ưu**.
- ✅ Có thể trực quan hóa dữ liệu bằng biểu đồ để **hiểu rõ các mối liên hệ giữa các đặc trưng**.

## 📁 Cấu trúc thư mục dự án

```
Backpack_Prediction_Challenge/
│
├── environment.yml
├── main.py
├── README.md
├── requirement.txt
├── .gitignore
│
│
├── data/
│   ├── raw/
│   │   └── data.csv
│   ├── processed/
│   └── submit/
│       ├── submission_lgbm.csv
│       ├── submission_lr.csv
│       └── submission_xgb.csv
│
├── models/
│   └── (Lưu các file mô hình .pkl sau khi train)
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature.ipynb
│   ├── 03_modeling.ipynb
│   └── data_preprocessed.csv
│
├── reports/
│   ├── eda/
│   │   └── (eda figures)
│   └── k_fold/
│       └── (Kết quả đánh giá mô hình)
│
└── src/
    ├── data/
    │   └── make_dataset.py
    ├── evaluation/
    │   └── evaluate_model.py
    ├── features/
    │   └── build_features.py
    └── models/
        ├── predict_model.py
        └── train_model.py
```

## 🚀Cách chạy dự án

### Cài đặt môi trường

- Cài đặt các thư viện cần thiết
<pre> <code> pip install -r requirements.txt</code> </pre>
- Hoặc cài đặt môi trường `ml-env` với Conda
<pre> <code> conda env create -f environment.yml
 conda activate ml-env</code> </pre>

### Cài biến môi trường `MODEL_DIR` (link dẫn đến folder project)

- Windows
<pre><code>set MODEL_DIR=.\Backpack_Prediction_Challenge</code></pre>

- macOS/Linux
<pre><code>export MODEL_DIR=.\Backpack_Prediction_Challenge</code> </pre>

### Chạy chương trình chính

 <pre><code> python main.py </code></pre>

## 📁 Sau khi chạy `main.py`, chương trình sẽ tự động tạo ra:

### Dữ liệu sau tiền xử lý

- `data/data_preprocessed.csv`: dữ liệu đã qua xử lý (EDA, encoding, sclaing, ...)

### Các Models đẫ huấn luyện

- `models/lr_model.pkl` - Logictis Regression
- `models/xgb_model.pkl` - XGBoost
- `model/lgbm_model.pkl` - LightBGM

### Kết quả đánh giá K-Fold (cv = 5)

- `reports/k_fold/kfold_lr.csv` – Kết quả từng fold của Logistic Regression
- `reports/k_fold/kfold_xgb.csv` – Kết quả từng fold của XGBoost
- `reports/k_fold/fold_lgbm.csv` – Kết quả từng fold của LightGBM
