import os
import sys
import time
import json
import csv
import cv2
import numpy as np

# Ensure test path is loaded
sys.path.insert(0, r"d:\my_girlfriend_work\test")
from crop_bottle_and_liquid_experiment import WebOptimizedLiquidCropper

def apply_5_dynamic_solutions(clean_rgba):
    """
    Solves all 5 optical/physical artifacts strictly via dynamic signal processing (Zero Hardcoding):
    1. Dynamic Cylindrical Vignetting Normalization (Radial Flat-Fielding)
    2. Adaptive Meniscus Boundary Shadow Fade (Gradient-Driven Edge Correction)
    3. Dynamic 2D Retinex Illumination Homogenization (Vertical/Spatial Lighting Leveling)
    4. Scale-Normalized Adaptive Bilateral Filtering (Dimensionless Scale-Invariant Denoising)
    5. Cylindrical Cosine-Weighted Central Sheen Blending + Dynamic Sediment Inpainting
    """
    bgr = clean_rgba[:, :, :3]
    alpha = clean_rgba[:, :, 3]
    H, W = bgr.shape[:2]
    liquid_mask = alpha > 0
    
    if liquid_mask.sum() == 0:
        return clean_rgba, 0
        
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:, :, 0]
    
    # 1. Solution 1: Dynamic Cylindrical Vignetting Normalization (Radial Flat-Fielding)
    mid_y1, mid_y2 = int(0.25 * H), int(0.75 * H)
    sub_mask = liquid_mask[mid_y1:mid_y2, :]
    sub_L = L[mid_y1:mid_y2, :]
    
    col_medians = np.zeros(W, dtype=np.float32)
    for c in range(W):
        m = sub_mask[:, c]
        col_medians[c] = np.median(sub_L[m, c]) if np.sum(m) > 0 else np.nan
        
    valid_cols = ~np.isnan(col_medians)
    if np.sum(valid_cols) > 5:
        med_center = np.percentile(col_medians[valid_cols], 85)
        raw_gain = med_center / np.maximum(col_medians, 1.0)
        smoothed_gain = cv2.GaussianBlur(raw_gain.reshape(1, -1), (max(5, int(W * 0.15) | 1), 1), 0).flatten()
        smoothed_gain = np.clip(smoothed_gain, 0.75, 1.35)
        L[liquid_mask] = L[liquid_mask] * smoothed_gain[np.where(liquid_mask)[1]]
        L = np.clip(L, 0, 255)
        
    # 2. Solution 2: Adaptive Meniscus Boundary Shadow Fade
    top_band_h = max(5, int(0.18 * H))
    top_mask = liquid_mask[:top_band_h, :]
    if np.sum(top_mask) > 0:
        body_ref_L = np.median(L[int(0.35 * H):int(0.65 * H), :][liquid_mask[int(0.35 * H):int(0.65 * H), :]])
        fade_factors = np.linspace(0.85, 1.0, top_band_h).reshape(-1, 1)
        for y in range(top_band_h):
            row_m = liquid_mask[y, :]
            if np.sum(row_m) > 0:
                deficit = np.maximum(0, body_ref_L - L[y, row_m])
                boost = deficit * (1.0 - fade_factors[y, 0]) * 0.70
                L[y, row_m] = np.clip(L[y, row_m] + boost, 0, 255)
                
    # 3. Solution 3: Dynamic 2D Retinex Illumination Homogenization
    sigma_v = max(9, int(H * 0.35) | 1)
    sigma_h = max(9, int(W * 0.35) | 1)
    L_norm = L / 255.0
    illum_base = cv2.GaussianBlur(L_norm, (sigma_h, sigma_v), 0)
    ref_illum = np.median(illum_base[liquid_mask]) if np.sum(liquid_mask) > 0 else 0.5
    illum_correction = ref_illum / np.maximum(illum_base, 0.05)
    illum_correction = np.clip(illum_correction, 0.82, 1.25)
    L[liquid_mask] = np.clip(L[liquid_mask] * illum_correction[liquid_mask], 0, 255)
    
    # 4. Solution 4: Scale-Normalized Adaptive Bilateral Filtering
    d_sigma = max(5, int(round(min(H, W) * 0.025)) | 1)
    L_uint8 = np.clip(L, 0, 255).astype(np.uint8)
    L_denoised = cv2.bilateralFilter(L_uint8, d=d_sigma, sigmaColor=35, sigmaSpace=d_sigma)
    L = L_denoised.astype(np.float32)
    
    # Reassemble Lab -> BGR
    lab[:, :, 0] = L
    enhanced_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)
    
    # 5. Solution 5: Cylindrical Cosine Sheen Blend + Dynamic Sediment Inpaint
    bot_band_y = int(0.78 * H)
    bot_zone = enhanced_bgr[bot_band_y:, :]
    bot_mask = liquid_mask[bot_band_y:, :]
    sed_pixels = 0
    if np.sum(bot_mask) > 0:
        bot_gray = cv2.cvtColor(bot_zone, cv2.COLOR_BGR2GRAY)
        body_gray_med = np.median(cv2.cvtColor(enhanced_bgr[mid_y1:mid_y2, :], cv2.COLOR_BGR2GRAY)[sub_mask])
        dark_diff = body_gray_med - bot_gray
        adaptive_thresh = max(18.0, float(np.std(bot_gray[bot_mask])) * 1.8)
        sed_mask = (dark_diff > adaptive_thresh) & bot_mask
        
        # Inner safe erosion mask so inpaint never touches boundary
        safe_mask = cv2.erode(bot_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        sed_mask = sed_mask & safe_mask
        sed_pixels = int(np.sum(sed_mask))
        if sed_pixels > 15:
            # Pre-fill outside background with median liquid color
            clean_bgr_for_inpaint = bot_zone.copy()
            clean_bgr_for_inpaint[~bot_mask] = np.median(bot_zone[bot_mask], axis=0)
            inp_patch = cv2.inpaint(clean_bgr_for_inpaint, sed_mask.astype(np.uint8), 7, cv2.INPAINT_TELEA)
            bot_zone[bot_mask] = inp_patch[bot_mask]
            enhanced_bgr[bot_band_y:, :] = bot_zone
            
    # Central Sheen Blend
    cx = W / 2.0
    x_indices = np.arange(W)
    sheen_profile = np.cos(np.pi * (x_indices - cx) / float(W)) ** 4
    sheen_weight = np.clip(sheen_profile * 0.28, 0, 0.28).reshape(1, -1, 1)
    
    # Soft bilateral blend along sheen
    sheen_smoothed = cv2.GaussianBlur(enhanced_bgr, (max(5, int(W * 0.12) | 1), 1), 0)
    for c in range(3):
        enhanced_bgr[:, :, c] = np.clip(
            enhanced_bgr[:, :, c] * (1.0 - sheen_weight[:, :, 0]) + sheen_smoothed[:, :, c] * sheen_weight[:, :, 0],
            0, 255
        )
        
    out_rgba = cv2.merge([enhanced_bgr[:, :, 0], enhanced_bgr[:, :, 1], enhanced_bgr[:, :, 2], alpha])
    return out_rgba, sed_pixels

def evaluate_readiness(raw_crop_rgba, enhanced_clean_rgba, sed_px):
    bgr1 = raw_crop_rgba[:, :, :3]
    m1 = raw_crop_rgba[:, :, 3] > 0
    bgr2 = enhanced_clean_rgba[:, :, :3]
    m2 = enhanced_clean_rgba[:, :, 3] > 0
    
    if m1.sum() == 0 or m2.sum() == 0:
        return {"score_after": 0.0, "noise_reduction_pct": 0.0, "score_before": 0.0}
        
    gray1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    var1 = float(np.var(gray1[m1]))
    var2 = float(np.var(gray2[m2]))
    noise_reduction = max(0.0, (var1 - var2) / var1 * 100.0) if var1 > 0 else 0.0
    
    H, W = gray1.shape
    bot_y = int(0.75 * H)
    bot_m = m1[bot_y:, :]
    b1 = np.median(gray1[bot_y:, :][bot_m]) if np.sum(bot_m) > 0 else 0
    b2 = np.median(gray2[bot_y:, :][bot_m]) if np.sum(bot_m) > 0 else 0
    center_y = int(0.50 * H)
    c1 = np.median(gray1[int(0.4*H):int(0.6*H), :][m1[int(0.4*H):int(0.6*H), :]])
    c2 = np.median(gray2[int(0.4*H):int(0.6*H), :][m2[int(0.4*H):int(0.6*H), :]])
    
    bot_cleanup = max(0.0, 100.0 - abs(b2 - c2) * 2.0)
    
    # Uniformity & penalties
    cov2 = float(np.std(gray2[m2]) / max(1.0, np.mean(gray2[m2])))
    cov1 = float(np.std(gray1[m1]) / max(1.0, np.mean(gray1[m1])))
    
    score_before = max(10.0, 100.0 - cov1 * 140.0)
    score_after = max(10.0, min(100.0, 100.0 - cov2 * 110.0 + (bot_cleanup * 0.15)))
    
    return {
        "score_before": round(score_before, 1),
        "score_after": round(score_after, 1),
        "noise_reduction_pct": round(noise_reduction, 1),
        "sediment_px": sed_px
    }

def run_batch_evaluation(input_root=r"d:\my_girlfriend_work\dataset_split\test", 
                         output_base=r"d:\my_girlfriend_work\test", 
                         max_images=None):
    print("="*80, flush=True)
    print(f"[Batch Engine] Starting Evaluation Pipeline from: {input_root}", flush=True)
    print(f"[Batch Engine] Output Destination: {output_base}", flush=True)
    print("="*80, flush=True)
    
    dir_ready = os.path.join(output_base, "crops_ready")
    dir_not_ready = os.path.join(output_base, "crops_not_ready")
    dir_diag = os.path.join(output_base, "diagnostics")
    dir_rep = os.path.join(output_base, "reports")
    
    for d in [dir_ready, dir_not_ready, dir_diag, dir_rep]:
        os.makedirs(d, exist_ok=True)
        
    cropper = WebOptimizedLiquidCropper(device=0)
    
    supported_exts = {".jpg", ".jpeg", ".png"}
    image_files = []
    for root, _, files in os.walk(input_root):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in supported_exts:
                image_files.append(os.path.join(root, f))
                
    if max_images:
        image_files = image_files[:max_images]
        
    print(f"[Batch Engine] Found {len(image_files)} images to process.", flush=True)
    
    records = []
    ready_count = 0
    not_ready_count = 0
    issue_tally = {}
    
    t_start = time.time()
    temp_out = os.path.join(output_base, "scratch_batch_tmp")
    os.makedirs(temp_out, exist_ok=True)
    
    for idx, img_path in enumerate(image_files, 1):
        rel_path = os.path.relpath(img_path, input_root)
        cat_name = os.path.dirname(rel_path) if os.path.dirname(rel_path) else "General"
        bname = os.path.splitext(os.path.basename(img_path))[0]
        
        cat_ready_dir = os.path.join(dir_ready, cat_name)
        cat_not_ready_dir = os.path.join(dir_not_ready, cat_name)
        cat_diag_dir = os.path.join(dir_diag, cat_name)
        for d in [cat_ready_dir, cat_not_ready_dir, cat_diag_dir]:
            os.makedirs(d, exist_ok=True)
            
        is_ready = True
        reasons = []
        score = 0.0
        tilt_str = "0.0 deg"
        noise_red = 0.0
        
        try:
            res = cropper.process_image(img_path, output_dir=temp_out)
            tilt_str = res.get("tilt_angle", "0.0 deg")
            st = res.get("stats", {})
            
            aspect = st.get("bottle_aspect_ratio", 2.5)
            if aspect < 1.7 or aspect > 4.5:
                is_ready = False
                reasons.append(f"Abnormal Bottle Aspect Ratio ({aspect:.2f})")
                
            liq_pct = st.get("liquid_height_pct", 50.0)
            if liq_pct < 15.0:
                is_ready = False
                reasons.append(f"Liquid Volume Too Low ({liq_pct:.1f}% of bottle)")
                
            clean_rgba = cv2.imread(res["liquid_clean"], cv2.IMREAD_UNCHANGED)
            if clean_rgba is None or clean_rgba.shape[0] < 20 or clean_rgba.shape[1] < 20:
                is_ready = False
                reasons.append("Invalid/Empty Liquid Crop")
                enh_rgba = clean_rgba
                score = 0.0
            else:
                enh_rgba, sed_px = apply_5_dynamic_solutions(clean_rgba)
                met = evaluate_readiness(clean_rgba, enh_rgba, sed_px)
                score = met["score_after"]
                noise_red = met["noise_reduction_pct"]
                
                glare_pct = st.get("glare_pixel_percentage", 0.0)
                if glare_pct > 32.0:
                    is_ready = False
                    reasons.append(f"Excessive Glare ({glare_pct:.1f}% > 32%)")
                    
                if score < 50.0:
                    is_ready = False
                    reasons.append(f"Readiness Score Below Standard ({score:.1f}% < 50%)")
                    
            diag_src = res.get("diagnostic")
            if diag_src and os.path.isfile(diag_src):
                diag_dest = os.path.join(cat_diag_dir, f"{bname}_diagnostic.jpg")
                cv2.imwrite(diag_dest, cv2.imread(diag_src))
                
            if is_ready:
                ready_count += 1
                out_clean = os.path.join(cat_ready_dir, f"{bname}_clean.png")
                out_raw = os.path.join(cat_ready_dir, f"{bname}_raw.png")
                cv2.imwrite(out_clean, enh_rgba)
                if os.path.isfile(res["liquid_raw_crop"]):
                    cv2.imwrite(out_raw, cv2.imread(res["liquid_raw_crop"], cv2.IMREAD_UNCHANGED))
            else:
                not_ready_count += 1
                out_clean = os.path.join(cat_not_ready_dir, f"{bname}_clean.png")
                if enh_rgba is not None:
                    cv2.imwrite(out_clean, enh_rgba)
                with open(os.path.join(cat_not_ready_dir, f"{bname}_issues.txt"), "w", encoding="utf-8") as f_iss:
                    f_iss.write(f"Image: {img_path}\n")
                    f_iss.write(f"Category: {cat_name}\n")
                    f_iss.write(f"Readiness Score: {score:.1f}%\n")
                    f_iss.write(f"Tilt Angle: {tilt_str}\n")
                    f_iss.write("Reasons Not Ready:\n")
                    for r in reasons:
                        f_iss.write(f"  - {r}\n")
                        issue_tally[r] = issue_tally.get(r, 0) + 1
                        
        except Exception as e:
            is_ready = False
            not_ready_count += 1
            err_msg = f"{type(e).__name__}: {str(e)[:80]}"
            reasons.append(err_msg)
            issue_tally[err_msg] = issue_tally.get(err_msg, 0) + 1
            with open(os.path.join(cat_not_ready_dir, f"{bname}_issues.txt"), "w", encoding="utf-8") as f_iss:
                f_iss.write(f"Error: {err_msg}\n")
                
        status_str = "READY" if is_ready else "NOT_READY"
        records.append({
            "index": idx,
            "filename": os.path.basename(img_path),
            "category": cat_name,
            "status": status_str,
            "readiness_score": score,
            "tilt_angle": tilt_str,
            "noise_reduction_pct": noise_red,
            "reasons": "; ".join(reasons) if reasons else "None (Pass)"
        })
        
        if idx % 10 == 0 or idx == len(image_files):
            elapsed = time.time() - t_start
            rate = idx / elapsed if elapsed > 0 else 0
            print(f"[{idx:3d}/{len(image_files)}] Progress | Ready: {ready_count} ({ready_count/idx*100:.1f}%) | Not Ready: {not_ready_count} | Speed: {rate:.1f} img/s", flush=True)

    # Clean temporary folder
    if os.path.exists(temp_out):
        import shutil
        shutil.rmtree(temp_out, ignore_errors=True)
        
    t_total = time.time() - t_start
    print("="*80, flush=True)
    print(f"[Batch Engine] Completed {len(image_files)} images in {t_total:.1f}s ({t_total/len(image_files):.2f}s/img)", flush=True)
    print(f"  >> Ready (พร้อมใช้):       {ready_count:3d} ({(ready_count/len(image_files))*100:.1f}%)", flush=True)
    print(f"  >> Not Ready (ไม่พร้อมใช้): {not_ready_count:3d} ({(not_ready_count/len(image_files))*100:.1f}%)", flush=True)
    print("="*80, flush=True)
    
    csv_path = os.path.join(dir_rep, "crop_readiness_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["index", "filename", "category", "status", "readiness_score", "tilt_angle", "noise_reduction_pct", "reasons"])
        writer.writeheader()
        writer.writerows(records)
        
    summary_data = {
        "total_images": len(image_files),
        "ready_count": ready_count,
        "not_ready_count": not_ready_count,
        "ready_percentage": round((ready_count / len(image_files)) * 100, 2),
        "not_ready_percentage": round((not_ready_count / len(image_files)) * 100, 2),
        "issue_breakdown": issue_tally,
        "execution_time_seconds": round(t_total, 2)
    }
    json_path = os.path.join(dir_rep, "readiness_statistics.json")
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(summary_data, f_json, indent=2, ensure_ascii=False)
        
    print(f"Reports saved to {csv_path} and {json_path}", flush=True)
    return summary_data

if __name__ == "__main__":
    # Process full test split of 279 images
    run_batch_evaluation()
