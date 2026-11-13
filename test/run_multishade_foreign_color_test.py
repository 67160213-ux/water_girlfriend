import os
import sys
import cv2
import numpy as np

sys.path.insert(0, r"d:\my_girlfriend_work\test")
from test_foreign_color_pipeline import dynamic_foreign_color_cleaner

def make_checker(h, w, s=10):
    y, x = np.ogrid[:h, :w]
    c = ((x // s) ^ (y // s)) & 1
    return (c * 30 + 20).astype(np.uint8)

def render_on_checker(rgba):
    H, W = rgba.shape[:2]
    bg = cv2.merge([make_checker(H, W), make_checker(H, W), make_checker(H, W)])
    alpha = (rgba[:, :, 3] / 255.0)[:, :, None]
    out = (rgba[:, :, :3] * alpha + bg * (1.0 - alpha)).astype(np.uint8)
    return out

def run_multishade_test():
    samples = [
        ("Yellow Reagent (Ammonia 0)", r"d:\my_girlfriend_work\test\crops_ready\Ammonia 0 mg_l 0 ppt\Ammonia 0 mg_l 0 ppt_02_clean.png"),
        ("Green Reagent (Ammonia 5)", r"d:\my_girlfriend_work\test\crops_ready\Ammonia 5 mg_l 0 ppt\Ammonia 5 mg_l 0 ppt_01_clean.png"),
        ("Clear Reagent (Phosphate 0)", r"d:\my_girlfriend_work\test\crops_ready\Phosphate 0 mg_l\Phosphate 0 mg_l_01_clean.png"),
        ("Blue Reagent (Phosphate 1.5)", r"d:\my_girlfriend_work\test\crops_ready\Phosphate 1.5 mg_l\Phosphate 1.5 mg_l_01_clean.png"),
    ]
    
    target_h = 320
    target_w = 200
    
    rows = []
    for label, path in samples:
        if not os.path.exists(path):
            print(f"Skipping {label}, file not found: {path}")
            continue
            
        rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        t_rgba, h_rgba, f_mask, stats = dynamic_foreign_color_cleaner(rgba)
        print(f"[{label}] Foreign cleaned: {stats['foreign_pct']:.1f}% | Tau: {stats['tau_chroma']:.1f}")
        
        # 1. Original
        d1 = render_on_checker(rgba)
        d1_s = cv2.resize(d1, (target_w, target_h))
        
        # 2. Foreign Mask
        d2 = d1.copy()
        d2[f_mask > 0] = [0, 0, 255]
        d2_s = cv2.resize(d2, (target_w, target_h))
        
        # 3. Approach 1 (Trimmed)
        d3 = render_on_checker(t_rgba)
        d3_s = cv2.resize(d3, (target_w, target_h))
        
        # 4. Approach 2 (Harmonized)
        d4 = render_on_checker(h_rgba)
        d4_s = cv2.resize(d4, (target_w, target_h))
        
        # Label banner on left
        banner = np.zeros((target_h, 160, 3), dtype=np.uint8) + 25
        cv2.putText(banner, label.split(" (")[0], (10, target_h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(banner, f"({label.split('(')[1]}", (10, target_h // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 220, 255), 1)
        cv2.putText(banner, f"Foreign: {stats['foreign_pct']:.1f}%", (10, target_h // 2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 0), 1)
        
        row = np.hstack([banner, d1_s, d2_s, d3_s, d4_s])
        rows.append(row)
        
    total_w = rows[0].shape[1]
    header = np.zeros((55, total_w, 3), dtype=np.uint8) + 15
    cv2.putText(header, "Sample Shade", (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
    cv2.putText(header, "[1] Original (ติดขอบ/ฉากหลัง)", (160 + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
    cv2.putText(header, "[2] Foreign Mask (ตรวจจับสีเพี้ยน)", (160 + target_w + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 255), 1)
    cv2.putText(header, "[3] Approach 1: Trim (Alpha=0)", (160 + target_w*2 + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1)
    cv2.putText(header, "[4] Approach 2: Harmonize (Inpaint)", (160 + target_w*3 + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0), 1)
    
    divider = np.zeros((6, total_w, 3), dtype=np.uint8) + 40
    
    canvas_parts = [header]
    for r in rows:
        canvas_parts.append(divider)
        canvas_parts.append(r)
        
    final_canvas = np.vstack(canvas_parts)
    out_path = r"d:\my_girlfriend_work\test\multishade_foreign_color_comparison.png"
    cv2.imwrite(out_path, final_canvas)
    print(f"Saved multi-shade comparison to: {out_path}")

if __name__ == "__main__":
    run_multishade_test()

