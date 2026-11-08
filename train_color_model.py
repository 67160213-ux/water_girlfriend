"""
Training & Evaluation Pipeline: ทำนายค่าความเข้มข้นสารละลาย (Phosphate & Ammonia) จากสีของน้ำ
รองรับ:
1. สกัดฟีเจอร์สี 16 มิติ (RGB, HSV, CIELAB, Color Ratios) จากภาพมวลน้ำโปร่งใส (_liquid_exact_shape.png)
2. กรองแสงสะท้อนเงาวาว (Specular Reflection) ออกอัตโนมัติ
3. เทรนและเปรียบเทียบโมเดล: PLS Regression (Gold Standard), SVR, Ridge Regression, Random Forest
4. วัดผลบน Test Set ด้วย R², RMSE, MAE พร้อมบันทึกกราฟ Calibration Curve และโมเดล (.pkl)
"""

import os
import sys
import re
import cv2
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def parse_label(folder_str: str) -> tuple[str, float, int]:
    """แยกประเภทสาร (Phosphate/Ammonia), ค่าความเข้มข้น (mg/L), และความเค็ม (Salinity ppt) จากชื่อโฟลเดอร์"""
    folder_norm = folder_str.replace('/', '\\')
    
    # 1. ประเภทสารเคมี
    chemical = "Phosphate" if "phosphate" in folder_norm.lower() else "Ammonia"
    
    # 2. ค่าความเข้มข้น (mg/L) - ดึงค่าจากโฟลเดอร์ชั้นในสุด (Leaf)
    conc_matches = list(re.finditer(r'([0-9]+(?:\.[0-9]+)?)\s*(?:mg_l|mg/l|ppm)', folder_norm, re.IGNORECASE))
    conc = float(conc_matches[-1].group(1)) if conc_matches else None
    
    # 3. ค่าความเค็ม (Salinity: 0 ppt หรือ 30 ppt)
    sal_matches = list(re.finditer(r'([0-9]+)\s*ppt', folder_norm, re.IGNORECASE))
    salinity = int(sal_matches[-1].group(1)) if sal_matches else 0
    
    return chemical, conc, salinity

def extract_features_from_liquid(liquid_png_path: str) -> dict:
    """สกัดฟีเจอร์สีเฉพาะเนื้อน้ำจริง โดยตัดแสงสะท้อนจ้า (Specular Glare) ออก"""
    img = cv2.imread(liquid_png_path, cv2.IMREAD_UNCHANGED)
    if img is None or len(img.shape) != 3 or img.shape[2] != 4:
        return None
        
    b_ch, g_ch, r_ch, a_ch = cv2.split(img)
    mask = a_ch > 0
    if np.sum(mask) < 100:
        return None
        
    # กรองแสงสะท้อนบนผิวกระจก (Specular Reflection)
    hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    glare = (val > 240) & (sat < 30) & mask
    pure_mask = mask & (~glare)
    if np.sum(pure_mask) < 50:
        pure_mask = mask # fallback ถ้าแสงสะท้อนทั้งภาพ
        
    pure_b = b_ch[pure_mask]
    pure_g = g_ch[pure_mask]
    pure_r = r_ch[pure_mask]
    
    # แปลงเป็น CIELAB
    lab = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2LAB)
    pure_l = lab[:, :, 0][pure_mask]
    pure_a = lab[:, :, 1][pure_mask]
    pure_b_star = lab[:, :, 2][pure_mask]
    pure_h = hsv[:, :, 0][pure_mask]
    pure_s = sat[pure_mask]
    pure_v = val[pure_mask]
    
    # ค่าสถิติ RGB
    mean_r, median_r, std_r = float(np.mean(pure_r)), float(np.median(pure_r)), float(np.std(pure_r))
    mean_g, median_g, std_g = float(np.mean(pure_g)), float(np.median(pure_g)), float(np.std(pure_g))
    mean_b, median_b, std_b = float(np.mean(pure_b)), float(np.median(pure_b)), float(np.std(pure_b))
    
    # อัตราส่วนสี (Color Ratios - ช่วยลดผลกระทบจากแสงแวดล้อมไม่คงที่)
    rg_ratio = mean_r / (mean_g + 1e-5)
    rb_ratio = mean_r / (mean_b + 1e-5)
    gb_ratio = mean_g / (mean_b + 1e-5)
    norm_r = mean_r / (mean_r + mean_g + mean_b + 1e-5)
    norm_b = mean_b / (mean_r + mean_g + mean_b + 1e-5)
    
    return {
        "mean_R": mean_r, "median_R": median_r, "std_R": std_r,
        "mean_G": mean_g, "median_G": median_g, "std_G": std_g,
        "mean_B": mean_b, "median_B": median_b, "std_B": std_b,
        "mean_H": float(np.mean(pure_h)), "median_H": float(np.median(pure_h)),
        "mean_S": float(np.mean(pure_s)), "median_S": float(np.median(pure_s)),
        "mean_V": float(np.mean(pure_v)), "median_V": float(np.median(pure_v)),
        "mean_L": float(np.mean(pure_l)), "mean_a": float(np.mean(pure_a)), "mean_b": float(np.mean(pure_b_star)),
        "rg_ratio": rg_ratio, "rb_ratio": rb_ratio, "gb_ratio": gb_ratio,
        "norm_R": norm_r, "norm_B": norm_b,
        "liquid_area_px": int(np.sum(pure_mask))
    }

def scan_and_extract_dataset(crop_dir: str) -> pd.DataFrame:
    """สแกนค้นหาไฟล์ _liquid_exact_shape.png ทั้งหมดในโฟลเดอร์ และสกัดฟีเจอร์สีลง DataFrame"""
    records = []
    print(f"[Scan] กำลังสแกนไฟล์ภาพน้ำใน: {crop_dir} ...")
    
    for root, dirs, files in os.walk(crop_dir):
        for f in files:
            if f.endswith("_liquid_exact_shape.png"):
                p = os.path.join(root, f)
                rel_folder = os.path.relpath(root, crop_dir)
                chemical, conc, sal = parse_label(rel_folder)
                
                if conc is None:
                    continue
                    
                feats = extract_features_from_liquid(p)
                if feats is not None:
                    row = {
                        "file": f,
                        "folder": rel_folder,
                        "chemical": chemical,
                        "concentration": conc,
                        "salinity": sal,
                        "path": p,
                        **feats
                    }
                    records.append(row)
                    
    df = pd.DataFrame(records)
    print(f"[Scan] สกัดฟีเจอร์สำเร็จ: {len(df)} ตัวอย่าง\n")
    return df

def train_and_evaluate(train_crop_dir: str = "output/crop_train", test_crop_dir: str = "output/crop_result", output_models_dir: str = "models"):
    """เทรนโมเดลทำนายความเข้มข้น และประเมินผลบน Test Set"""
    os.makedirs(output_models_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # 1. ตรวจสอบว่ามีโฟลเดอร์ Crop หรือยัง
    if not os.path.exists(train_crop_dir):
        print(f"[Error] ไม่พบโฟลเดอร์ภาพ Train ที่ครอปแล้ว: {train_crop_dir}")
        print(f"  กรุณารันคำสั่งครอปภาพ Train ก่อน:")
        print(f"  python crop_bottle_and_liquid.py \"d:\\my_girlfriend_work\\dataset_split\\train\" \"{train_crop_dir}\"")
        return
        
    if not os.path.exists(test_crop_dir):
        print(f"[Error] ไม่พบโฟลเดอร์ภาพ Test ที่ครอปแล้ว: {test_crop_dir}")
        return
        
    # 2. สกัดฟีเจอร์
    df_train = scan_and_extract_dataset(train_crop_dir)
    df_test = scan_and_extract_dataset(test_crop_dir)
    
    if df_train.empty or df_test.empty:
        print("[Error] ไม่พบข้อมูลฟีเจอร์สำหรับการเทรน")
        return
        
    # บันทึกตารางฟีเจอร์เป็น CSV สำหรับใช้งาน
    df_train.to_csv("output/train_features.csv", index=False)
    df_test.to_csv("output/test_features.csv", index=False)
    print("บันทึกตารางฟีเจอร์สีที่: output/train_features.csv และ output/test_features.csv")
    
    # คอลัมน์ฟีเจอร์ที่นำมาใช้ทำนาย
    feature_cols = [
        "mean_R", "median_R", "std_R",
        "mean_G", "median_G", "std_G",
        "mean_B", "median_B", "std_B",
        "mean_H", "mean_S", "mean_V",
        "mean_L", "mean_a", "mean_b",
        "rg_ratio", "rb_ratio", "gb_ratio", "norm_R", "norm_B"
    ]
    
    chemicals = ["Phosphate", "Ammonia"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, chem in enumerate(chemicals):
        print(f"\n{'='*50}")
        print(f"=== กำลังเทรนและประเมินผลโมเดล: {chem} ===")
        print(f"{'='*50}")
        
        tr_sub = df_train[df_train["chemical"] == chem]
        te_sub = df_test[df_test["chemical"] == chem]
        
        if tr_sub.empty or te_sub.empty:
            print(f"[Warning] ไม่มีข้อมูลสำหรับ {chem}")
            continue
            
        X_train = tr_sub[feature_cols].values
        y_train = tr_sub["concentration"].values
        X_test = te_sub[feature_cols].values
        y_test = te_sub["concentration"].values
        
        print(f"จำนวนตัวอย่าง: Train = {len(X_train)} ภาพ, Test = {len(X_test)} ภาพ")
        print(f"ช่วงความเข้มข้น: {np.min(y_train):.2f} ถึง {np.max(y_train):.2f} mg/L")
        
        # รายการโมเดลที่นำมาแข่งขัน
        models = {
            "PLS Regression (n=3)": Pipeline([("scaler", StandardScaler()), ("pls", PLSRegression(n_components=3))]),
            "SVR (RBF Kernel)": Pipeline([("scaler", StandardScaler()), ("svr", SVR(C=10.0, epsilon=0.02))]),
            "Ridge Regression": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        best_name = None
        best_r2 = -float("inf")
        best_model = None
        best_pred = None
        
        for name, m in models.items():
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            if len(pred.shape) > 1:
                pred = pred.flatten()
                
            r2 = r2_score(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            mae = mean_absolute_error(y_test, pred)
            
            print(f"  - {name:22s} | R²: {r2:.4f} | RMSE: {rmse:.4f} mg/L | MAE: {mae:.4f} mg/L")
            
            if r2 > best_r2:
                best_r2 = r2
                best_name = name
                best_model = m
                best_pred = pred
                
        print(f"\n>> โมเดลที่ดีที่สุดสำหรับ {chem}: {best_name} (R² = {best_r2:.4f})")
        
        # บันทึกโมเดลที่ดีที่สุด
        model_save_path = os.path.join(output_models_dir, f"{chem.lower()}_model.pkl")
        with open(model_save_path, "wb") as f:
            pickle.dump({
                "model": best_model,
                "model_name": best_name,
                "feature_cols": feature_cols,
                "chemical": chem,
                "r2_test": best_r2
            }, f)
        print(f"บันทึกไฟล์โมเดลไปที่: {model_save_path}")
        
        # วาดกราฟ Calibration Curve (Actual vs Predicted)
        ax = axes[idx]
        ax.scatter(y_test, best_pred, color="royalblue" if chem == "Phosphate" else "forestgreen", alpha=0.7, edgecolors="k", s=50, label=f"Test Samples (N={len(y_test)})")
        
        min_v = min(np.min(y_test), np.min(best_pred))
        max_v = max(np.max(y_test), np.max(best_pred))
        ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2, label="Ideal 1:1 Line")
        
        ax.set_title(f"{chem} Calibration Curve\n({best_name}, R²={best_r2:.4f})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Actual Concentration (mg/L)", fontsize=11)
        ax.set_ylabel("Predicted Concentration (mg/L)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        
    plt.tight_layout()
    chart_path = "output/calibration_results.png"
    plt.savefig(chart_path, dpi=300)
    print(f"\nบันทึกกราฟ Calibration Curve ไปที่: {chart_path}")
    print(f"\n{'='*50}")
    print("=== เสร็จสิ้นกระบวนการเทรนและประเมินผลโมเดลทั้งหมด! ===")
    print(f"{'='*50}")

if __name__ == "__main__":
    # รองรับการระบุโฟลเดอร์แบบยืดหยุ่น:
    # 1. ระบุโฟลเดอร์หลัก เช่น "dataset_cropped" (ค้นหา train/test อัตโนมัติ)
    # 2. หรือระบุแยก 2 โฟลเดอร์ เช่น "dataset_cropped/train" "dataset_cropped/test"
    if len(sys.argv) == 2:
        base_dir = sys.argv[1]
        train_cand = os.path.join(base_dir, "train")
        test_cand = os.path.join(base_dir, "test")
        if os.path.exists(train_cand) and os.path.exists(test_cand):
            train_dir = train_cand
            test_dir = test_cand
        else:
            train_dir = base_dir
            test_dir = os.path.join("dataset_cropped", "test") if os.path.exists(os.path.join("dataset_cropped", "test")) else os.path.join("output", "crop_result")
    elif len(sys.argv) > 2:
        train_dir = sys.argv[1]
        test_dir = sys.argv[2]
        if os.path.isdir(train_dir) and os.path.exists(os.path.join(train_dir, "train")):
            train_dir = os.path.join(train_dir, "train")
        if os.path.isdir(test_dir) and os.path.exists(os.path.join(test_dir, "test")):
            test_dir = os.path.join(test_dir, "test")
    else:
        if os.path.exists(os.path.join("dataset_cropped", "train")) and os.path.exists(os.path.join("dataset_cropped", "test")):
            train_dir = os.path.join("dataset_cropped", "train")
            test_dir = os.path.join("dataset_cropped", "test")
        else:
            train_dir = os.path.join("output", "crop_train")
            test_dir = os.path.join("output", "crop_result")

    print(f"[Config] ใช้ชุดข้อมูล Train จาก: {os.path.abspath(train_dir)}")
    print(f"[Config] ใช้ชุดข้อมูล Test จาก:  {os.path.abspath(test_dir)}\n")
    train_and_evaluate(train_dir, test_dir)
