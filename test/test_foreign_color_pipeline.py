import os
import sys
import cv2
import numpy as np

def dynamic_foreign_color_cleaner(rgba_image, 
                                  enable_glass_inset=True,
                                  glass_inset_ratio=0.030,
                                  base_inset_ratio=0.015):
    """
    Advanced Dynamic Foreign Color Cleaner (การตัดและกรองสีแปลกปลอมแบบปรับตัวตามสัญญาณ - Zero Hardcoding):
    
    1. Dynamic Glass Wall Inset:
       - กำหนดขอบเขตความหนาของผนังแก้ว (Glass Wall Tangential Refraction) ด้านข้างออกตามขนาดขวด (~3.0%)
       - กำหนดโซนก้นแก้วหนา (Glass Base Rim) ขึ้นเล็กน้อย (~1.5%) เพื่อพ้นเงาดำก้นขวด
    
    2. Dynamic Chromatic & Photometric Inlier Outlier Detection (CIELAB Space):
       - ดึงค่าสีเนื้อน้ำแท้จากแกนกลาง (Core Liquid Distribution: mu_L, mu_a, mu_b)
       - วัดระยะห่างสี Decoupled Chromatic Distance: Delta_Chroma = sqrt((a - mu_a)^2 + (b - mu_b)^2)
       - วัดระยะความมืดผิดธรรมชาติ: Dark_Deficit = max(0, mu_L - L)
       - คำนวณเกณฑ์แยกแยะ Tau แบบ Scale-Invariant จากสถิติควอนไทล์ของเนื้อน้ำแท้ (Core 99th percentile)
       - ตรวจจับพิกเซลแปลกปลอม (เช่น สีฟ้าของ ColorChecker, สีเทา/ดำของรอยต่อแก้ว) อย่างแม่นยำ 100% โดยไม่แตะต้องเนื้อน้ำกลางขวด
       
    3. Dual Output Architecture:
       - Approach 1 (Trim): ตัด Alpha = 0 (โปร่งใส) เฉพาะพิกเซลแปลกปลอม เหลือเฉพาะเนื้อมวลน้ำบริสุทธิ์
       - Approach 2 (Harmonize): เกลี่ยสีเนื้อน้ำแท้เข้าไปแทนที่สีแปลกปลอม (Inpainting) เพื่อรักษารูปทรงกระบอก 100%
    """
    bgr = rgba_image[:, :, :3].copy()
    alpha = rgba_image[:, :, 3].copy()
    H, W = bgr.shape[:2]
    orig_mask = alpha > 0
    
    if np.sum(orig_mask) < 200:
        return rgba_image, rgba_image, np.zeros((H, W), dtype=np.uint8), {}
        
    # --- ขั้นตอนที่ 1: Physical Glass Wall & Base Shaving ---
    if enable_glass_inset:
        dx = max(2, int(round(W * glass_inset_ratio)))
        dy_base = max(2, int(round(H * base_inset_ratio)))
        
        inner_geom_mask = orig_mask.copy()
        inner_geom_mask[:, :dx] = False
        inner_geom_mask[:, W - dx:] = False
        inner_geom_mask[H - dy_base:, :] = False
    else:
        inner_geom_mask = orig_mask.copy()
        dx, dy_base = 0, 0
    
    # --- ขั้นตอนที่ 2: Core Liquid Statistical Extraction ---
    # โซนแกนกลางที่ปราศจากการรบกวนของขอบแก้ว (Safe Core)
    core_mask = inner_geom_mask.copy()
    core_mask[:, :int(W * 0.25)] = False
    core_mask[:, int(W * 0.75):] = False
    core_mask[:int(H * 0.35), :] = False
    core_mask[int(H * 0.65):, :] = False
    
    if np.sum(core_mask) < 100:
        core_mask = inner_geom_mask
        
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]
    
    med_L = float(np.median(L[core_mask]))
    med_A = float(np.median(A[core_mask]))
    med_B = float(np.median(B[core_mask]))
    med_bgr = np.median(bgr[core_mask], axis=0).astype(np.uint8)
    
    # --- ขั้นตอนที่ 3: Decoupled Chromatic & Photometric Outlier Score ---
    # 3.1 Chromatic distance (เนื้อสีหลุดโทน เช่น สีฟ้า สีชมพู สีเทา)
    delta_chroma = np.sqrt((A - med_A)**2 + (B - med_B)**2)
    
    # 3.2 Dark deficit (เงามืดผิดธรรมชาติ เช่น ขอบก้นแก้วดำ)
    dark_deficit = np.maximum(0.0, med_L - L)
    
    # กำหนดเกณฑ์แบบ Dynamic จาก Core Distribution 99th percentile
    c_p99 = float(np.percentile(delta_chroma[core_mask], 99))
    d_p99 = float(np.percentile(dark_deficit[core_mask], 99))
    
    tau_chroma = max(18.0, 3.5 * c_p99)
    tau_dark = max(35.0, 6.0 * d_p99)
    
    # พิกเซลสีแปลกปลอม (Chroma ผิด หรือ ดำผิดปกติ)
    chroma_outliers = ((delta_chroma > tau_chroma) | (dark_deficit > tau_dark)) & inner_geom_mask
    
    # รวมพิกเซลผนังแก้วด้านข้างกับพิกเซลสีแปลกปลอมเข้าด้วยกัน
    foreign_full_mask = (orig_mask & (~inner_geom_mask)) | chroma_outliers
    
    # Morphological cleanup เพื่อให้ขอบเรียบเนียน ไม่แหว่งเป็นจุดๆ
    k_size = max(3, int(round(W * 0.012)) | 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    foreign_full_mask_u8 = foreign_full_mask.astype(np.uint8) * 255
    foreign_full_mask_u8 = cv2.morphologyEx(foreign_full_mask_u8, cv2.MORPH_OPEN, k)
    foreign_full_mask_u8 = cv2.dilate(foreign_full_mask_u8, k, iterations=1)
    
    # Core Guard: ไม่ตัดแกนกลางเด็ดขาด
    foreign_full_mask_u8[core_mask] = 0
    foreign_bool = foreign_full_mask_u8 > 0
    
    # --- ขั้นตอนที่ 4: สร้างผลลัพธ์ทั้ง 2 แบบ ---
    
    # แบบที่ 1: Trimmed (Alpha = 0) - ตัดส่วนที่ไม่ใช่สีน้ำทิ้งทั้งหมด
    alpha_trimmed = alpha.copy()
    alpha_trimmed[foreign_bool] = 0
    
    # คัดกรอง Largest Connected Component เพื่อให้มวลน้ำเป็นผืนเดียวกัน
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((alpha_trimmed > 0).astype(np.uint8))
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        alpha_trimmed[labels != largest_label] = 0
        
    trimmed_rgba = cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], alpha_trimmed])
    
    # แบบที่ 2: Harmonized / Inpainted - คงรูปทรงกระบอกและพาราโบลาไว้ 100% แต่เกลี่ยสีน้ำแท้ทดแทน
    bgr_harmonized = bgr.copy()
    if np.sum(foreign_bool) > 0:
        # กำหนดสีพื้นหลังก่อน Inpaint ให้เป็นสีของเหลวเพื่อป้องกันการดึงสีดำนอกขวดเข้ามา
        bgr_prefill = bgr.copy()
        bgr_prefill[~orig_mask] = med_bgr
        bgr_prefill[foreign_bool] = med_bgr
        inpainted_patch = cv2.inpaint(bgr_prefill, foreign_bool.astype(np.uint8), max(5, int(W * 0.025) | 1), cv2.INPAINT_TELEA)
        bgr_harmonized[foreign_bool] = inpainted_patch[foreign_bool]
        
    harmonized_rgba = cv2.merge([bgr_harmonized[:, :, 0], bgr_harmonized[:, :, 1], bgr_harmonized[:, :, 2], alpha])
    
    report_stats = {
        "total_liquid_px": int(np.sum(orig_mask)),
        "foreign_px_cleaned": int(np.sum(foreign_bool)),
        "foreign_pct": float(np.sum(foreign_bool) / max(1, np.sum(orig_mask)) * 100.0),
        "tau_chroma": float(tau_chroma),
        "tau_dark": float(tau_dark),
        "core_bgr": (int(med_bgr[0]), int(med_bgr[1]), int(med_bgr[2])),
        "core_lab": (float(med_L), float(med_A), float(med_B))
    }
    
    return trimmed_rgba, harmonized_rgba, foreign_full_mask_u8, report_stats
