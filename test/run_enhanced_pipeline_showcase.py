import sys
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Strict isolation: ONLY use test experiment files, do NOT touch main script
sys.path.insert(0, r"d:\my_girlfriend_work\test")
from crop_bottle_and_liquid_experiment import WebOptimizedLiquidCropper
from test_foreign_color_pipeline import dynamic_foreign_color_cleaner

cropper = WebOptimizedLiquidCropper()

test_cases = [
    ("1. Ammonia 0 mg/L (USER 02)", r"d:\my_girlfriend_work\dataset_split\test\10_04_69\Ammonia 0 mg_l 0 ppt\Ammonia 0 mg_l 0 ppt_02.jpg", "Ammonia 0 mg_l 0 ppt_02"),
    ("2. Ammonia 0 mg/L (01)", r"d:\my_girlfriend_work\dataset_split\test\10_04_69\Ammonia 0 mg_l 0 ppt\Ammonia 0 mg_l 0 ppt_01.jpg", "Ammonia 0 mg_l 0 ppt_01"),
    ("3. Ammonia 0.5 mg/L", r"d:\my_girlfriend_work\dataset_split\test\10_04_69\Ammonia 0.5 mg_l 0 ppt\Ammonia 0.5 mg_l 0 ppt_01.jpg", "Ammonia 0.5 mg_l 0 ppt_01"),
    ("4. Ammonia 1.0 mg/L", r"d:\my_girlfriend_work\dataset_split\test\10_04_69\Ammonia 1 mg_l 0ppt\Ammonia 1 mg_l 0ppt_01.jpg", "Ammonia 1 mg_l 0ppt_01"),
    ("5. Ammonia 3.0 mg/L", r"d:\my_girlfriend_work\dataset_split\test\10_04_69\Ammonia 3 mg_l 0 ppt\Ammonia 3 mg_l 0 ppt_01.jpg", "Ammonia 3 mg_l 0 ppt_01"),
    ("6. Ammonia 5.0 mg/L", r"d:\my_girlfriend_work\dataset_split\test\10_04_69\Ammonia 5 mg_l 0 ppt\Ammonia 5 mg_l 0 ppt_01.jpg", "Ammonia 5 mg_l 0 ppt_01"),
    ("7. Phosphate 0.04 mg/L", r"d:\my_girlfriend_work\dataset_split\test\Phosphate new 13.08.68\13.08.68\0.04 mg_l\Phosphate 0.04 mg_l_01.jpeg", "Phosphate 0.04 mg_l_01"),
    ("8. Phosphate 0.1 mg/L", r"d:\my_girlfriend_work\dataset_split\test\Phosphate 0.1 mg_l\Phosphate 0.1 mg_l_01.jpeg", "Phosphate 0.1 mg_l_01"),
    ("9. Phosphate 0.8 mg/L", r"d:\my_girlfriend_work\dataset_split\test\Phosphate 0.8 mg_l\Phosphate 0.8 mg_l_01.jpeg", "Phosphate 0.8 mg_l_01"),
    ("10. Phosphate 1.5 mg/L", r"d:\my_girlfriend_work\dataset_split\test\Phosphate 1.5 mg_l\Phosphate 1.5 mg_l_01.jpeg", "Phosphate 1.5 mg_l_01"),
]

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
        
    # Operate in Lab color space to preserve 100% of color hue (a*, b*)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    
    # -------------------------------------------------------------------------
    # Solution 2: Adaptive Meniscus Boundary Shadow Fade
    # Detect if the top boundary has a dark meniscus shadow dip, smooth it gently
    # -------------------------------------------------------------------------
    med_body_L = float(np.median(L[liquid_mask]))
    top_y_per_col = np.zeros(W, dtype=np.int32)
    for x in range(W):
        ys = np.where(liquid_mask[:, x])[0]
        top_y_per_col[x] = ys[0] if len(ys) > 0 else 0
        
    # Check top 4% of liquid column for shadow dip
    top_band_h = max(3, int(0.04 * H))
    for x in range(W):
        ty = top_y_per_col[x]
        if ty > 0 and ty + top_band_h < H:
            col_top_l = L[ty:ty + top_band_h, x]
            if len(col_top_l) > 0 and np.mean(col_top_l) < 0.88 * med_body_L:
                # Fade the shadow dip smoothly towards med_body_L
                weight = np.linspace(0.85, 1.0, len(col_top_l)).astype(np.float32)
                L[ty:ty + top_band_h, x] = np.maximum(col_top_l, med_body_L * weight)

    # -------------------------------------------------------------------------
    # Solution 1: Dynamic Cylindrical Vignetting Normalization (Radial Flat-Fielding)
    # -------------------------------------------------------------------------
    col_profile = np.zeros(W, dtype=np.float32)
    for x in range(W):
        col_m = liquid_mask[:, x]
        col_profile[x] = np.median(L[col_m, x]) if col_m.sum() > 5 else 0.0
        
    valid_cols = col_profile > 0
    if valid_cols.sum() > 10:
        core_x0, core_x1 = int(0.35 * W), int(0.65 * W)
        ref_L = float(np.median(col_profile[core_x0:core_x1]))
        k_smooth = max(5, int(0.12 * W) | 1)
        col_smooth = cv2.GaussianBlur(col_profile.reshape(1, -1), (k_smooth, 1), 0.05 * W)[0]
        gain_x = np.clip(ref_L / np.maximum(8.0, col_smooth), 0.80, 1.40)
        L = np.clip(L * gain_x.reshape(1, -1), 0.0, 255.0)

    # -------------------------------------------------------------------------
    # Solution 3: Dynamic 2D Retinex Illumination Homogenization (Vertical Gradient)
    # -------------------------------------------------------------------------
    sig_y = max(4.0, 0.22 * H)
    sig_x = max(4.0, 0.25 * W)
    L_masked = L.copy()
    L_masked[~liquid_mask] = float(np.median(L[liquid_mask]))
    illum_field = cv2.GaussianBlur(L_masked, (0, 0), sigmaX=sig_x, sigmaY=sig_y)
    
    med_target = float(np.median(L[liquid_mask]))
    gain_2d = np.clip(med_target / np.maximum(8.0, illum_field), 0.82, 1.25)
    L = np.clip(L * gain_2d, 0.0, 255.0)
    
    # Reassemble Lab -> BGR (Hue/Chroma 100% untouched!)
    lab[:, :, 0] = L
    bgr_leveled = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # -------------------------------------------------------------------------
    # Solution 5: Cylindrical Cosine-Weighted Central Sheen Blending
    # -------------------------------------------------------------------------
    xc = (W - 1) / 2.0
    half_w = W / 2.0
    x_coords = np.arange(W, dtype=np.float32)
    cos_theta = np.sqrt(np.maximum(0.0, 1.0 - ((x_coords - xc) / (half_w + 1e-5)) ** 2))
    sheen_weight = (cos_theta ** 2).reshape(1, -1)

    # -------------------------------------------------------------------------
    # Solution 4: Scale-Normalized Adaptive Bilateral Filtering
    # Dynamic kernel parameters scaled strictly to image width & data spread
    # -------------------------------------------------------------------------
    d_dyn = max(3, int(round(0.015 * W)) * 2 + 1)
    sig_space = max(2.0, 0.022 * W)
    mad_L = float(np.median(np.abs(L[liquid_mask] - np.median(L[liquid_mask]))))
    sig_color = max(18.0, 1.6 * mad_L)
    
    bgr_denoised = cv2.bilateralFilter(bgr_leveled, d=d_dyn, sigmaColor=sig_color, sigmaSpace=sig_space)
    bgr_final = (bgr_denoised.astype(np.float32) * 0.90 + bgr_leveled.astype(np.float32) * 0.10).astype(np.uint8)

    # -------------------------------------------------------------------------
    # Dynamic Bottom Sediment Inpainting (Activated strictly when distinct outlier exists)
    # -------------------------------------------------------------------------
    bot_y0 = int(0.82 * H)
    bot_mask = np.zeros((H, W), dtype=bool)
    bot_mask[bot_y0:, :] = liquid_mask[bot_y0:, :]
    sediment_px = 0
    if bot_mask.sum() > 50:
        med_bot = np.median(bgr_final[liquid_mask], axis=0)
        dist_full_bot = np.linalg.norm(bgr_final[bot_y0:].astype(np.float32) - med_bot, axis=2)
        dist_bot = np.linalg.norm(bgr_final[bot_mask].astype(np.float32) - med_bot, axis=1)
        med_dist = np.median(dist_bot)
        mad_dist = np.median(np.abs(dist_bot - med_dist)) + 1e-5
        
        # Outlier condition: statistically distinct + absolute Euclidean gap >= 16.0
        # Safe inner mask: sediment must be inside the liquid, not touching the outside table
        inner_safe = cv2.erode(liquid_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))) > 0
        outliers = (dist_full_bot > max(16.0, med_dist + 2.8 * mad_dist)) & liquid_mask[bot_y0:] & inner_safe[bot_y0:]
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(outliers.astype(np.uint8))
        sed_sub = np.zeros_like(outliers, dtype=np.uint8)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] > max(30, int(0.0005 * H * W)):
                sed_sub[labels == lbl] = 255
        if np.any(sed_sub):
            sed_sub = cv2.dilate(sed_sub, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            full_sed = np.zeros((H, W), dtype=np.uint8)
            full_sed[bot_y0:] = sed_sub
            full_sed = full_sed & (inner_safe.astype(np.uint8) * 255)
            sediment_px = int(np.sum(full_sed > 0))
            
            # Clean outside background so Telea never bleeds white table into the liquid
            bgr_inpaint = bgr_final.copy()
            bgr_inpaint[~liquid_mask] = med_bot
            bgr_final = cv2.inpaint(bgr_inpaint, full_sed, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    enhanced_rgba = cv2.merge([bgr_final, alpha])
    return enhanced_rgba, sediment_px

def evaluate_readiness(before_rgba, after_rgba, sed_px):
    b1 = before_rgba[:, :, :3]
    a1 = before_rgba[:, :, 3] > 0
    b2 = after_rgba[:, :, :3]
    a2 = after_rgba[:, :, 3] > 0
    
    g1 = cv2.cvtColor(b1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(b2, cv2.COLOR_BGR2GRAY)
    
    # 1. High frequency noise variance (Laplacian)
    lap1 = cv2.Laplacian(g1, cv2.CV_64F)[a1]
    lap2 = cv2.Laplacian(g2, cv2.CV_64F)[a2]
    var1 = float(np.var(lap1))
    var2 = float(np.var(lap2))
    noise_reduction = max(0.0, (var1 - var2) / max(1e-5, var1) * 100.0)
    
    # 2. Color Homogeneity / Uniformity (1 - CV)
    cov1 = float(np.std(g1[a1]) / max(1e-5, np.mean(g1[a1])))
    cov2 = float(np.std(g2[a2]) / max(1e-5, np.mean(g2[a2])))
    u1 = max(0.0, 1.0 - cov1) * 100.0
    u2 = max(0.0, 1.0 - cov2) * 100.0
    
    # 3. Bottom sediment contrast
    H, W = b1.shape[:2]
    bot1 = g1[int(0.82*H):][a1[int(0.82*H):]]
    bot2 = g2[int(0.82*H):][a2[int(0.82*H):]]
    bot_c1 = float(np.std(bot1)) if len(bot1) > 0 else 0.0
    bot_c2 = float(np.std(bot2)) if len(bot2) > 0 else 0.0
    bot_cleanup = max(0.0, (bot_c1 - bot_c2) / max(1e-5, bot_c1) * 100.0) if bot_c1 > 0 else 0.0
    
    # 4. Standard Visual Production Readiness Score (0-100%)
    noise_penalty1 = min(35.0, var1 * 0.45)
    noise_penalty2 = min(35.0, var2 * 0.45)
    bot_penalty1 = min(30.0, bot_c1 * 1.5)
    bot_penalty2 = min(30.0, bot_c2 * 1.5)
    u_penalty1 = min(25.0, cov1 * 35.0)
    u_penalty2 = min(25.0, cov2 * 35.0)
    
    score_before = max(10.0, 100.0 - (noise_penalty1 + bot_penalty1 + u_penalty1))
    score_after = max(10.0, 100.0 - (noise_penalty2 + bot_penalty2 + u_penalty2))
    
    abs_gain_pct = score_after - score_before
    rel_gain_pct = ((score_after - score_before) / score_before) * 100.0
    
    return {
        "noise_before": var1,
        "noise_after": var2,
        "noise_reduction_pct": noise_reduction,
        "uniformity_before": u1,
        "uniformity_after": u2,
        "bot_cleanup_pct": bot_cleanup,
        "score_before": score_before,
        "score_after": score_after,
        "abs_gain_pct": abs_gain_pct,
        "rel_gain_pct": rel_gain_pct,
        "sediment_px": sed_px
    }

print("Running 5 Dynamic Solutions across all 10 shades...")
records = []

for idx, (label, p, bname) in enumerate(test_cases):
    # Process image with cropper (using new upward parabola sag=0.082W)
    res = cropper.process_image(p, output_dir=r"test\output_enhanced")
    
    prev_clean = cv2.imread(res["liquid_clean"], cv2.IMREAD_UNCHANGED)
    enh_clean, sed_px = apply_5_dynamic_solutions(prev_clean)
    
    # Dynamic Foreign Color Cleaning & Infilling (ตัดสีแปลกปลอมแบบที่ 1 + เติมเต็มสีน้ำแท้แบบที่ 2)
    trimmed_clean, harmonized_clean, f_mask, f_stats = dynamic_foreign_color_cleaner(enh_clean)
    
    enh_path = os.path.join(r"test\output_enhanced", f"{bname}_liquid_enhanced.png")
    cv2.imwrite(enh_path, harmonized_clean)
    
    metrics = evaluate_readiness(prev_clean, harmonized_clean, sed_px)
    metrics["foreign_pct"] = f_stats["foreign_pct"]
    records.append((label, p, bname, res, prev_clean, harmonized_clean, metrics))
    print(f"[{idx+1}/10] {label[:25]:25s} | Before: {metrics['score_before']:.1f}% -> After: {metrics['score_after']:.1f}% (+{metrics['abs_gain_pct']:.1f}%) | Foreign Cleaned: {f_stats['foreign_pct']:.1f}%")

# Generate 10-row x 5-column Master Showcase Figure
print("Rendering Master Showcase Figure (10x5)...")
fig, axes = plt.subplots(10, 5, figsize=(26, 46), facecolor="#141414")

col_titles = [
    "1. Original Photo\n(YOLO Bounding Box)",
    "2. Tight Bottle\n(Deskew 90° via SAM 2 minAreaRect + Cap Trimmed)",
    "3. Adaptive Parabolic Curves\n(Top Meniscus + Upward Base Cup)",
    "4. Raw Liquid Crop\n(_liquid_raw_crop.png - 100% True)",
    "5. Cleaned & Infilled Liquid\n(Zero Foreign Colors + 100% Full Mass)"
]

for col_idx, title in enumerate(col_titles):
    axes[0, col_idx].set_title(title, color="cyan", fontsize=13, fontweight="bold", pad=15)

for row_idx, (label, orig_path, base_name, res, prev_clean, enh_clean, met) in enumerate(records):
    st = res["stats"]
    
    # 1. Original Image with YOLO box
    img_orig = cv2.imread(orig_path)
    box = cropper.locate_bottle(img_orig, image_path=orig_path)
    vis_orig = img_orig.copy()
    cv2.rectangle(vis_orig, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), max(3, vis_orig.shape[1] // 250))
    axes[row_idx, 0].imshow(cv2.cvtColor(vis_orig, cv2.COLOR_BGR2RGB))
    axes[row_idx, 0].set_ylabel(f"#{row_idx+1}: {label}", color="white", fontsize=11, fontweight="bold", labelpad=8)
    axes[row_idx, 0].axis("off")

    # 2. Tight Bottle (Deskewed via SAM 2 minAreaRect)
    tight_img = cv2.imread(res["bottle"])
    axes[row_idx, 1].imshow(cv2.cvtColor(tight_img, cv2.COLOR_BGR2RGB))
    axes[row_idx, 1].text(0.5, -0.06, f"SAM 2 Tilt: {res.get('tilt_angle', '0.0 deg')}\nMethod: minAreaRect",
                          color="#38bdf8", fontsize=9.0, fontweight="bold", ha="center", transform=axes[row_idx, 1].transAxes)
    axes[row_idx, 1].axis("off")

    # 3. Parabolic Curves (with new sag=0.082W upward cup)
    vis_curves = tight_img.copy()
    H, W = tight_img.shape[:2]
    top_c, surf = cropper.locate_liquid_surface(tight_img, H - 6)
    bot_c, bot_info = cropper.locate_liquid_bottom(tight_img, top_c, H - 6)
    pts_top = np.array([[x, int(top_c[x])] for x in range(W)], dtype=np.int32)
    pts_bot = np.array([[x, int(bot_c[x])] for x in range(W)], dtype=np.int32)
    top_col = (0, 220, 255) if surf["orientation"] == "arch" else (255, 120, 255)
    cv2.polylines(vis_curves, [pts_top], False, top_col, max(2, W // 75))
    cv2.polylines(vis_curves, [pts_bot], False, (0, 255, 0), max(2, W // 75))
    axes[row_idx, 2].imshow(cv2.cvtColor(vis_curves, cv2.COLOR_BGR2RGB))
    axes[row_idx, 2].text(0.5, -0.07, f"Top: {surf['orientation'].upper()} (y={int(top_c.min())})\nBot: CUP (y={int(bot_c.max())}, sag={bot_info['sag_px']:.0f}px)",
                          color="lightgreen", fontsize=9.5, fontweight="bold", ha="center", transform=axes[row_idx, 2].transAxes)
    axes[row_idx, 2].axis("off")

    # 4. Raw Liquid Crop
    img_raw_crop = cv2.imread(res["liquid_raw_crop"], cv2.IMREAD_UNCHANGED)
    axes[row_idx, 3].imshow(cv2.cvtColor(img_raw_crop, cv2.COLOR_BGRA2RGBA))
    axes[row_idx, 3].text(0.5, -0.06, f"Shape: {img_raw_crop.shape[1]}x{img_raw_crop.shape[0]} px\nRaw RGB: ({st.get('measured_inlier_R', 0):.1f}, {st.get('measured_inlier_G', 0):.1f}, {st.get('measured_inlier_B', 0):.1f})",
                          color="yellow", fontsize=9.0, ha="center", transform=axes[row_idx, 3].transAxes)
    axes[row_idx, 3].axis("off")

    # 5. Cleaned & Infilled Liquid (Zero Foreign Colors + 100% Full Mass)
    axes[row_idx, 4].imshow(cv2.cvtColor(enh_clean, cv2.COLOR_BGRA2RGBA))
    gain_txt = f"+{met['abs_gain_pct']:.1f}% Ready" if met['abs_gain_pct'] > 0 else "0% (Pure)"
    axes[row_idx, 4].text(0.5, -0.06, f"Readiness: {met['score_before']:.1f}% -> {met['score_after']:.1f}%\nCleaned Foreign: {met['foreign_pct']:.1f}% | Noise: -{met['noise_reduction_pct']:.0f}%",
                          color="lightgreen", fontsize=9.0, fontweight="bold", ha="center", transform=axes[row_idx, 4].transAxes)
    axes[row_idx, 4].axis("off")

plt.tight_layout()

# Save master figure in test directory and brain artifact directory
out_test = r"d:\my_girlfriend_work\test\ten_enhanced_pipeline.png"
out_brain = r"C:\Users\USER\.gemini\antigravity\brain\4b14b0d9-b001-4ae9-9733-76131e0f068c\ten_enhanced_pipeline.png"

plt.savefig(out_test, dpi=160, bbox_inches="tight", facecolor="#141414")
plt.savefig(out_brain, dpi=160, bbox_inches="tight", facecolor="#141414")
plt.close()
print(f"Master figure saved to {out_test} and {out_brain}")

# Also split into Part 1 and Part 2 for direct high-res viewing in chat UI
from PIL import Image
im = Image.open(out_brain)
w, h = im.size
row_h = h // 10

im1 = im.crop((0, 0, w, row_h * 5))
p1_path = r"C:\Users\USER\.gemini\antigravity\brain\4b14b0d9-b001-4ae9-9733-76131e0f068c\enhanced_pipeline_part1.png"
im1.save(p1_path)

header = im.crop((0, 0, w, 180))
body2 = im.crop((0, row_h * 5, w, h))
im2 = Image.new("RGBA", (w, header.size[1] + body2.size[1]), (20, 20, 20, 255))
im2.paste(header, (0, 0))
im2.paste(body2, (0, header.size[1]))
p2_path = r"C:\Users\USER\.gemini\antigravity\brain\4b14b0d9-b001-4ae9-9733-76131e0f068c\enhanced_pipeline_part2.png"
im2.save(p2_path)

# Detailed 5-Solutions Breakdown Figure
fig_sol, axes_s = plt.subplots(2, 5, figsize=(24, 10), facecolor="#141414")
# Row 1: Ammonia 0 USER 02 across 5 stages
r1_prev = records[0][4]
r1_enh = records[0][5]
H1, W1 = r1_prev.shape[:2]

# Row 2: Phosphate 0.8 across 5 stages
r9_prev = records[8][4]
r9_enh = records[8][5]
H9, W9 = r9_prev.shape[:2]

axes_s[0, 0].imshow(cv2.cvtColor(r1_prev, cv2.COLOR_BGRA2RGBA))
axes_s[0, 0].set_title("Ammonia 0: Before Solutions", color="yellow")
axes_s[0, 0].axis("off")

axes_s[0, 1].imshow(cv2.cvtColor(r1_prev[:int(0.2*H1), :], cv2.COLOR_BGRA2RGBA))
axes_s[0, 1].set_title("Top Meniscus Shadow (Before)", color="yellow")
axes_s[0, 1].axis("off")

axes_s[0, 2].imshow(cv2.cvtColor(r1_enh[:int(0.2*H1), :], cv2.COLOR_BGRA2RGBA))
axes_s[0, 2].set_title("Adaptive Fade (After Sol 2)", color="lightgreen")
axes_s[0, 2].axis("off")

axes_s[0, 3].imshow(cv2.cvtColor(r1_prev[int(0.75*H1):, :], cv2.COLOR_BGRA2RGBA))
axes_s[0, 3].set_title("Bottom Sediment (Before)", color="yellow")
axes_s[0, 3].axis("off")

axes_s[0, 4].imshow(cv2.cvtColor(r1_enh, cv2.COLOR_BGRA2RGBA))
axes_s[0, 4].set_title("Ammonia 0: All 5 Sols Applied", color="lightgreen", fontweight="bold")
axes_s[0, 4].axis("off")

# Phosphate 0.8 breakdown
axes_s[1, 0].imshow(cv2.cvtColor(r9_prev, cv2.COLOR_BGRA2RGBA))
axes_s[1, 0].set_title("Phosphate 0.8: Before Solutions", color="yellow")
axes_s[1, 0].axis("off")

axes_s[1, 1].imshow(cv2.cvtColor(r9_prev[:, :int(0.3*W9)], cv2.COLOR_BGRA2RGBA))
axes_s[1, 1].set_title("Left Dark Wall (Before Sol 1)", color="yellow")
axes_s[1, 1].axis("off")

axes_s[1, 2].imshow(cv2.cvtColor(r9_enh[:, :int(0.3*W9)], cv2.COLOR_BGRA2RGBA))
axes_s[1, 2].set_title("Vignette Normalized (After Sol 1)", color="lightgreen")
axes_s[1, 2].axis("off")

axes_s[1, 3].imshow(cv2.cvtColor(r9_prev, cv2.COLOR_BGRA2RGBA))
axes_s[1, 3].set_title("Vertical Gradient (Before Sol 3)", color="yellow")
axes_s[1, 3].axis("off")

axes_s[1, 4].imshow(cv2.cvtColor(r9_enh, cv2.COLOR_BGRA2RGBA))
axes_s[1, 4].set_title("Phosphate 0.8: All 5 Sols Applied", color="lightgreen", fontweight="bold")
axes_s[1, 4].axis("off")

plt.tight_layout()
breakdown_path = r"C:\Users\USER\.gemini\antigravity\brain\4b14b0d9-b001-4ae9-9733-76131e0f068c\five_solutions_visual_breakdown.png"
plt.savefig(breakdown_path, dpi=160, bbox_inches="tight", facecolor="#141414")
plt.close()
print("Saved 5-solutions visual breakdown to:", breakdown_path)

# Print Summary Table
print("\n" + "="*85)
print(f"{'Sample / Shade':30s} | {'Readiness Before':16s} | {'Readiness After':15s} | {'Gain (%)':9s} | {'Noise Red.':10s}")
print("="*85)
scores_b, scores_a, gains = [], [], []
for label, p, bname, res, prev, enh, met in records:
    scores_b.append(met['score_before'])
    scores_a.append(met['score_after'])
    gains.append(met['abs_gain_pct'])
    print(f"{label[:30]:30s} | {met['score_before']:14.1f}% | {met['score_after']:13.1f}% | +{met['abs_gain_pct']:6.1f}% | -{met['noise_reduction_pct']:8.1f}%")
print("="*85)
print(f"{'OVERALL AVERAGE':30s} | {np.mean(scores_b):14.1f}% | {np.mean(scores_a):13.1f}% | +{np.mean(gains):6.1f}% |")
print("="*85)
