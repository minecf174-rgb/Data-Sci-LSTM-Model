import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# =============================
# 1) โหลดข้อมูล + เตรียมเบื้องต้น
# =============================
df = pd.read_csv("dataclean/co2_data_2000_2024_clear_dev_level.csv")

# เลือกกลุ่มความเจริญ: "No-Develop", "Developing", "Developed"
dev_group = "No-Develop"   # <-- เปลี่ยนกลุ่มได้ตรงนี้

# เลือกเฉพาะกลุ่ม dev_level นั้น ๆ
df_g = df[df["dev_level"] == dev_group].copy()

# เลือกช่วงปีที่อยากใช้กับโมเดล (2000–2022)
df_g = df_g[(df_g["year"] >= 2000) & (df_g["year"] <= 2022)].copy()
df_g = df_g.sort_values(["country", "year"])

print("จำนวนแถวในกลุ่ม", dev_group, ":", len(df_g))
print("ประเทศในกลุ่มนี้:", df_g["country"].unique())

# =============================
# 2) เลือก feature / target (ใช้ log)
# =============================
feature_cols = ["gdp_per_capita", "energy_per_capita", "co2_per_unit_energy"]
raw_target_col = "co2_per_capita"          # ค่าเป้าหมายจริง
log_target_col = "log_co2_per_capita"      # ค่าเป้าหมายที่แปลง log แล้ว

# ตัด NaN ทิ้ง (สำคัญมาก)
df_g = df_g.dropna(subset=feature_cols + [raw_target_col])

# สร้างคอลัมน์ log1p(target)
df_g[log_target_col] = np.log1p(df_g[raw_target_col])

print("ปีทั้งหมดหลัง dropna:", sorted(df_g["year"].unique()))

# สเกลข้อมูล
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_all = df_g[feature_cols].values.astype("float32")
y_all_log = df_g[log_target_col].values.astype("float32").reshape(-1, 1)

X_scaled = X_scaler.fit_transform(X_all)
y_scaled = y_scaler.fit_transform(y_all_log)

# ใส่กลับเข้า df_g เพื่อใช้ groupby country
for i, col in enumerate(feature_cols):
    df_g[col + "_scaled"] = X_scaled[:, i]
df_g[log_target_col + "_scaled"] = y_scaled[:, 0]

# =============================
# 3) ฟังก์ชันสร้าง sequence แยกทีละประเทศ
# =============================
def create_sequences_by_country(df_group, feature_cols_scaled, target_col_scaled, look_back=5):
    X_list, y_list, y_years = [], [], []

    for country_name, df_c in df_group.groupby("country"):
        df_c = df_c.sort_values("year")
        years_c = df_c["year"].values

        X_data = df_c[feature_cols_scaled].values
        y_data = df_c[target_col_scaled].values.reshape(-1, 1)

        if len(df_c) <= look_back:
            continue

        for i in range(len(df_c) - look_back):
            X_list.append(X_data[i : i + look_back, :])
            y_list.append(y_data[i + look_back, :])
            y_years.append(years_c[i + look_back])

    return np.array(X_list), np.array(y_list), np.array(y_years)

# ฟังก์ชันสร้างโมเดล (reuse ได้หลายรอบ)
def build_model(look_back, n_features):
    model = Sequential([
        Input(shape=(look_back, n_features)),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# =============================
# 4) ลองหลาย parameter แล้วเลือกตัวที่ดีที่สุด
# =============================
look_back_list = [3, 5, 7]
batch_sizes = [16, 32, 64]

feature_cols_scaled = [c + "_scaled" for c in feature_cols]
target_col_scaled = log_target_col + "_scaled"

best_rmse = np.inf
best_info = None
best_y_true = None
best_y_pred = None
best_test_years = None

for look_back in look_back_list:
    # สร้าง sequence สำหรับ look_back นี้
    X, y, y_years = create_sequences_by_country(
        df_g, feature_cols_scaled, target_col_scaled, look_back
    )

    if len(X) == 0:
        print(f"[SKIP] look_back={look_back} -> ไม่มี sequence เลย")
        continue

    # แบ่ง train / test ตามปีของ y
    train_mask = (y_years <= 2018)
    test_mask  = (y_years >= 2019)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]
    test_years       = y_years[test_mask]

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print(f"[SKIP] look_back={look_back} -> train/test ว่าง")
        continue

    n_features = len(feature_cols)

    for batch_size in batch_sizes:
        print(f"\n=== ลอง config: look_back={look_back}, batch_size={batch_size} ===")
        model = build_model(look_back, n_features)

        es = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            epochs=80,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[es],
            verbose=0  # ถ้าอยากดู epoch print เปลี่ยนเป็น 2
        )

        # ทำนาย
        y_pred_scaled = model.predict(X_test)
        y_pred_log = y_scaler.inverse_transform(y_pred_scaled)
        y_true_log = y_scaler.inverse_transform(y_test)

        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_true_log)

        # metric รวม
        mse  = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_true, y_pred)
        y_mean = y_true.mean()
        rel_rmse = rmse / y_mean * 100

        print(f"Mean(y_true)={y_mean:.3f} | MSE={mse:.4f} | RMSE={rmse:.4f} | RelRMSE={rel_rmse:.2f}% | R^2={r2:.4f}")

        # เก็บตัวที่ดีที่สุด
        if rmse < best_rmse:
            best_rmse = rmse
            best_info = {
                "look_back": look_back,
                "batch_size": batch_size,
                "mse": mse,
                "rmse": rmse,
                "rel_rmse": rel_rmse,
                "r2": r2,
                "y_mean": y_mean,
            }
            best_y_true = y_true
            best_y_pred = y_pred
            best_test_years = test_years

# =============================
# 5) สรุป config ที่ดีที่สุด
# =============================
if best_info is None:
    raise RuntimeError("ไม่มี config ไหน train/test ได้เลย ลองเช็คช่วงปี / dev_group")

print("\n\n================ BEST CONFIG ================")
print(f"look_back   = {best_info['look_back']}")
print(f"batch_size  = {best_info['batch_size']}")
print("--------------- Metrics ---------------------")
print(f"Mean(y_true)   = {best_info['y_mean']:.3f}")
print(f"MSE            = {best_info['mse']:.6f}")
print(f"RMSE           = {best_info['rmse']:.6f}")
print(f"Relative RMSE  = {best_info['rel_rmse']:.2f}%")
print(f"R^2            = {best_info['r2']:.6f}")
print("=============================================")

# =============================
# 6) สร้าง summary รายปีของ "โมเดลที่ดีที่สุด"
# =============================
results = pd.DataFrame({
    "year": best_test_years,
    "true": best_y_true[:, 0],
    "pred": best_y_pred[:, 0],
})

print("\nค่าเฉลี่ยรายปีในกลุ่ม", dev_group, "(best config)")
for year, group in results.groupby("year"):
    print(
        f"ปี {int(year)} | จริงเฉลี่ย = {group['true'].mean():.3f} | ทำนายเฉลี่ย = {group['pred'].mean():.3f}"
    )

def calc_year_metrics(group):
    y_t = group["true"].values
    y_p = group["pred"].values
    mse_y  = mean_squared_error(y_t, y_p)
    rmse_y = np.sqrt(mse_y)
    r2_y   = r2_score(y_t, y_p) if len(group) > 1 else np.nan
    return pd.Series({"mse": mse_y, "rmse": rmse_y, "r2": r2_y})

metrics_by_year = results.groupby("year").apply(calc_year_metrics).reset_index()
print("\n=== Metrics by year (best config) ===")
print(metrics_by_year.to_string(index=False))

# =============================
# 7) Plot กราฟจริง vs ทำนาย (เฉลี่ยต่อปี) ของ config ที่ดีที่สุด
# =============================
mean_by_year = results.groupby("year").mean().reset_index()

plt.figure(figsize=(8, 4))
plt.plot(mean_by_year["year"], mean_by_year["true"], marker="o",
         label="Actual co2_per_capita (avg per year)")
plt.plot(mean_by_year["year"], mean_by_year["pred"], marker="x", linestyle="--",
         label="Predicted co2_per_capita (avg per year)")
plt.title(f"[BEST] Dev group: {dev_group} - co2_per_capita (Test: >= 2019)")
plt.xlabel("Year")
plt.ylabel("co2_per_capita")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
