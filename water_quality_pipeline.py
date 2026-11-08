"""
Water Quality Colorimetric Analysis System
ระบบตรวจสอบคุณภาพน้ำจากภาพถ่ายขวดตัวอย่าง เพื่อทำนายความเข้มข้นสารละลายจากสี
ตามสถาปัตยกรรม Computer Vision และ Analytical Chemistry
เวอร์ชันประมวลผลความเร็วสูงพิเศษ (Ultra-Fast Standardized Pipeline)
"""

import os
import sys
import json
import pickle
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import torch
from ultralytics import YOLO

# ตรวจสอบและตั้งค่าอุปกรณ์ประมวลผล (GPU / CPU)
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print(f"[Hardware Engine] ตรวจพบ GPU: {torch.cuda.get_device_name(0)} (เปิดใช้งาน CUDA Acceleration)", flush=True)
else:
    # หากไม่มี CUDA ให้จำกัด Thread เพื่อป้องกัน CPU Contention
    torch.set_num_threads(2)
    print("[Hardware Engine] ใช้งาน CPU Processing Mode", flush=True)

def remove_yellow_annotations(image: np.ndarray) -> np.ndarray:
    """ตรวจจับและลบเส้นสีเหลืองที่วาดทับบนภาพ (Inpainting) เพื่อคืนค่าน้ำสีเขียวที่แท้จริง"""
    if image is None:
        return None
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    yellow_mask = ((r > 160) & (g > 160) & (b < 100)).astype(np.uint8) * 255
    if np.sum(yellow_mask > 0) > 100:
        kernel = np.ones((3, 3), np.uint8)
        yellow_mask_dil = cv2.dilate(yellow_mask, kernel, iterations=1)
        return cv2.inpaint(image, yellow_mask_dil, 3, cv2.INPAINT_TELEA)
    return image

def load_standard_image(path: str, max_dim: int = 960) -> np.ndarray:
    """โหลดภาพ ปรับสเกลมาตรฐานไม่เกิน 960px และลบเส้นวาดตกแต่ง (ถ้ามี)"""
    img = cv2.imread(path)
    if img is None:
        return None
    img = remove_yellow_annotations(img)
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

# ==============================================================================
# ขั้นตอนที่ 0: Color Calibration & White Balance
# ==============================================================================
class ColorCalibrator:
    @staticmethod
    def detect_reference_patch(image: np.ndarray) -> tuple[np.ndarray, tuple]:
        h, w = image.shape[:2]
        small = cv2.resize(image, (300, 225))
        hsv_s = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        s_chan = hsv_s[:, :, 1]
        v_chan = hsv_s[:, :, 2]
        
        candidates = [
            (int(300 * 0.02), int(225 * 0.02), int(300 * 0.15), int(225 * 0.15)),
            (int(300 * 0.83), int(225 * 0.02), int(300 * 0.15), int(225 * 0.15)),
            (int(300 * 0.02), int(225 * 0.83), int(300 * 0.15), int(225 * 0.15)),
            (int(300 * 0.83), int(225 * 0.83), int(300 * 0.15), int(225 * 0.15)),
            (int(300 * 0.35), int(225 * 0.02), int(300 * 0.30), int(225 * 0.10)),
        ]
        
        best_cand = candidates[0]
        lowest_sat = 255.0
        for (cx, cy, cw, ch) in candidates:
            sub_s = s_chan[cy:cy+ch, cx:cx+cw]
            sub_v = v_chan[cy:cy+ch, cx:cx+cw]
            valid = (sub_v > 120) & (sub_v < 248)
            if np.sum(valid) > (cw * ch * 0.4):
                m_s = np.mean(sub_s[valid])
                if m_s < lowest_sat:
                    lowest_sat = m_s
                    best_cand = (cx, cy, cw, ch)
                    
        scale_x = w / 300.0
        scale_y = h / 225.0
        rx = int(best_cand[0] * scale_x)
        ry = int(best_cand[1] * scale_y)
        rw = int(best_cand[2] * scale_x)
        rh = int(best_cand[3] * scale_y)
        ref_roi = image[ry:ry+rh, rx:rx+rw]
        return ref_roi, (rx, ry, rw, rh)

    @classmethod
    def apply_white_balance(cls, image: np.ndarray) -> tuple[np.ndarray, tuple]:
        ref_roi, ref_box = cls.detect_reference_patch(image)
        ref_mean = np.mean(ref_roi, axis=(0, 1)) # [B, G, R]
        target = np.mean(ref_mean)
        scale = np.clip(target / (ref_mean + 1e-6), 0.6, 1.6)
        
        # ปรับค่าอย่างรวดเร็วด้วย cv2.multiply
        balanced = cv2.multiply(image.astype(np.float32), scale)
        balanced = np.clip(balanced, 0, 255).astype(np.uint8)
        return balanced, ref_box

    @staticmethod
    def srgb_to_linear(bgr_image: np.ndarray) -> np.ndarray:
        img_norm = bgr_image.astype(np.float32) / 255.0
        linear = np.where(
            img_norm <= 0.04045,
            img_norm / 12.92,
            np.power((img_norm + 0.055) / 1.055, 2.4)
        )
        return linear

# ==============================================================================
# ขั้นตอนที่ 1: ตรวจจับขวดด้วย YOLO (Object Detection & Deskew)
# ==============================================================================
class YOLOBottleDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf_thresh: float = 0.08, device=None):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.device = DEVICE if device is None else device

    def detect_and_crop(self, image: np.ndarray) -> dict:
        h, w = image.shape[:2]
        mid_x = w / 2.0
        
        # 1. รัน YOLO บนภาพสเกลมาตรฐานโดยตรง
        results = self.model.predict(source=image, conf=self.conf_thresh, device=self.device, verbose=False)[0]
        
        vial_cands = []
        for b in results.boxes:
            cls_name = self.model.names[int(b.cls[0].item())]
            conf = float(b.conf[0].item())
            if cls_name in ["bottle", "cup", "vase", "wine glass", "toilet"]:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                bw = x2 - x1
                bh = y2 - y1
                cx = (x1 + x2) / 2.0
                # ต้องอยู่ในช่วงกึ่งกลางภาพ และมีความกว้างเหมาะสมกับขวดทดลอง (~100-200px)
                if abs(cx - mid_x) < (w * 0.28) and (w * 0.07) < bw < (w * 0.24):
                    aspect = bh / float(bw) if bw > 0 else 0
                    vial_cands.append({
                        "box": (x1, y1, x2, y2),
                        "conf": conf,
                        "name": cls_name,
                        "aspect": aspect,
                        "cx": cx,
                        "bw": bw, "bh": bh
                    })
                    
        if vial_cands:
            # กรณีที่ 1: ตรวจพบทั้งขวดเต็มใบ (Aspect Ratio H/W >= 2.5)
            tall = [c for c in vial_cands if c["aspect"] >= 2.5]
            if tall:
                tall.sort(key=lambda x: x["conf"], reverse=True)
                best_box = tall[0]["box"]
            elif len(vial_cands) >= 2:
                # กรณีที่ 2: YOLO แยกตรวจจับฝากับตัวน้ำออกจากกัน (เช่น cup สองชิ้น) ให้รวมกัน
                min_x = min(c["box"][0] for c in vial_cands)
                min_y = min(c["box"][1] for c in vial_cands)
                max_x = max(c["box"][2] for c in vial_cands)
                max_y = max(c["box"][3] for c in vial_cands)
                merged_w = max_x - min_x
                merged_h = max_y - min_y
                if 2.4 <= (merged_h / float(merged_w)) <= 3.8:
                    best_box = (min_x, min_y, max_x, max_y)
                else:
                    vial_cands.sort(key=lambda x: x["conf"], reverse=True)
                    bx1, by1, bx2, by2 = vial_cands[0]["box"]
                    bh = by2 - by1
                    best_box = (bx1, max(0, int(by1 - bh * 0.40)), bx2, by2)
            else:
                # กรณีที่ 3: ตรวจพบเฉพาะลำตัวน้ำด้านล่าง ให้ขยายขอบบนขึ้นไปครอบคลุมฝาขวด
                vial_cands.sort(key=lambda x: x["conf"], reverse=True)
                bx1, by1, bx2, by2 = vial_cands[0]["box"]
                bh = by2 - by1
                best_box = (bx1, max(0, int(by1 - bh * 0.40)), bx2, by2)
        else:
            # กรณีสำรอง: หาขวดจากช่องว่างระหว่าง ColorChecker (Corridor Locator)
            best_box = self._geometric_corridor_locator(image)
            
        x1, y1, x2, y2 = best_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        best_box = (x1, y1, x2, y2)
        
        raw_crop = image[y1:y2, x1:x2].copy()
        aligned_crop, angle = self.deskew_bottle(raw_crop)
        
        return {
            "bbox": best_box,
            "crop": aligned_crop,
            "deskew_angle": angle
        }

    @staticmethod
    def _geometric_corridor_locator(image: np.ndarray) -> tuple:
        """ค้นหาตำแหน่งขวดจากทางเดินกึ่งกลางระหว่างชาร์ตสีซ้าย-ขวาอย่างแม่นยำ"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        c_start = int(w * 0.32)
        c_end = int(w * 0.68)
        corridor = gray[int(h * 0.20):int(h * 0.80), c_start:c_end]
        
        sobel_x = np.abs(cv2.Sobel(corridor, cv2.CV_32F, 1, 0, ksize=3))
        col_sums = np.sum(sobel_x, axis=0)
        
        best_pair = (10, 140)
        max_score = -1
        for l_idx in range(5, len(col_sums) - 90):
            for r_idx in range(l_idx + 95, min(len(col_sums), l_idx + 160)):
                score = col_sums[l_idx] + col_sums[r_idx]
                if score > max_score:
                    max_score = score
                    best_pair = (l_idx, r_idx)
                    
        x1 = c_start + best_pair[0]
        x2 = c_start + best_pair[1]
        
        strip = gray[:, x1:x2]
        sobel_y = np.abs(cv2.Sobel(strip, cv2.CV_32F, 0, 1, ksize=3))
        row_sums = np.sum(sobel_y, axis=1)
        
        top_range = row_sums[int(h*0.15):int(h*0.45)]
        y1 = int(h*0.15) + (int(np.argmax(top_range)) if len(top_range) > 0 else 0)
        
        bot_range = row_sums[int(h*0.65):int(h*0.90)]
        y2 = int(h*0.65) + (int(np.argmax(bot_range)) if len(bot_range) > 0 else int(h*0.25))
        y2 = min(h, y2 + 15)
        
        return (x1, y1, x2, y2)

    @staticmethod
    def deskew_bottle(crop: np.ndarray) -> tuple[np.ndarray, float]:
        ch, cw = crop.shape[:2]
        scale = 200.0 / max(ch, cw)
        small_c = cv2.resize(crop, (max(10, int(cw * scale)), max(10, int(ch * scale))))
        gray = cv2.cvtColor(small_c, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=small_c.shape[0] * 0.3, maxLineGap=10)
        
        angles = []
        if lines is not None:
            for line in lines:
                l_flat = np.array(line).reshape(-1)
                if len(l_flat) >= 4:
                    lx1, ly1, lx2, ly2 = l_flat[:4]
                    ang = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))
                    if 70 <= abs(ang) <= 110:
                        dev = 90 - abs(ang) if ang > 0 else -90 - ang
                        angles.append(dev)
                        
        rot_angle = float(np.median(angles)) if angles and abs(np.median(angles)) < 15.0 else 0.0
        if abs(rot_angle) > 0.5:
            M = cv2.getRotationMatrix2D((cw / 2, ch / 2), rot_angle, 1.0)
            crop = cv2.warpAffine(crop, M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            
        return crop, rot_angle

# ==============================================================================
# ขั้นตอนที่ 2: Segmentation ด้วย Horizontal Projection Profile
# ==============================================================================
class HorizontalProjectionSegmenter:
    def __init__(self, filter_window: int = 15, poly_order: int = 3):
        self.window = filter_window
        self.poly_order = poly_order

    def segment(self, bottle_crop: np.ndarray) -> dict:
        ch, cw = bottle_crop.shape[:2]
        gray = cv2.cvtColor(bottle_crop, cv2.COLOR_BGR2GRAY)
        
        # Sobel-Y ตรวจจับขอบแนวนอน
        sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        raw_proj = np.sum(sobel_y, axis=1)
        
        w_len = min(self.window, ch if ch % 2 != 0 else ch - 1)
        if w_len < self.poly_order + 2:
            w_len = self.poly_order + 3 if (self.poly_order + 3) % 2 != 0 else self.poly_order + 4
        smooth_proj = savgol_filter(raw_proj, window_length=w_len, polyorder=self.poly_order)
        
        # (ก) ขอบล่างฝาขวด (ช่วง 18% - 36% ของความสูงขวด)
        cap_search = smooth_proj[int(ch * 0.18):int(ch * 0.36)]
        cap_y = int(ch * 0.18) + (int(np.argmax(cap_search)) if len(cap_search) > 0 else int(ch * 0.25))
        
        # (ค) ก้นขวด (ช่วง 85% - 98% ของความสูงขวด)
        bot_search = smooth_proj[int(ch * 0.85):]
        bottom_y = int(ch * 0.85) + (int(np.argmax(bot_search)) if len(bot_search) > 0 else int(ch * 0.95))
        
        # (ข) ผิวน้ำ / ระดับสารละลาย (Meniscus) ระหว่างฝาขวดถึงก้นขวด
        if bottom_y - 30 > cap_y + 15:
            liq_search = smooth_proj[cap_y + 15:bottom_y - 30]
            meniscus_y = (cap_y + 15) + int(np.argmax(liq_search))
        else:
            meniscus_y = int(ch * 0.45)
            
        return {
            "cap_y": int(cap_y),
            "meniscus_y": int(meniscus_y),
            "bottom_y": int(bottom_y),
            "profile": smooth_proj
        }
            
        return {
            "cap_y": int(cap_y),
            "meniscus_y": int(meniscus_y),
            "bottom_y": int(bottom_y),
            "profile": smooth_proj
        }

# ==============================================================================
# ขั้นตอนที่ 3: เลือกพื้นที่ของเหลว (Liquid ROI)
# ==============================================================================
class LiquidROIExtractor:
    def __init__(self, v_margin: float = 0.02, h_margin: float = 0.04):
        # ปรับ Margin ให้น้อยลงเพื่อครอบคลุมมวลสารละลายทั้งหมดในขวด (Full Liquid Volume)
        self.v_margin = v_margin
        self.h_margin = h_margin

    def extract_liquid_roi(self, bottle_crop: np.ndarray, seg: dict) -> tuple[np.ndarray, np.ndarray, tuple]:
        ch, cw = bottle_crop.shape[:2]
        meniscus_y = seg["meniscus_y"]
        bottom_y = seg["bottom_y"]
        
        liq_h = max(30, bottom_y - meniscus_y)
        y_top = max(0, int(meniscus_y + liq_h * self.v_margin))
        y_bot = min(ch, int(bottom_y - liq_h * self.v_margin))
        if y_bot <= y_top:
            y_top = int(meniscus_y + 2)
            y_bot = int(bottom_y - 2)
            
        x_left = max(0, int(cw * self.h_margin))
        x_right = min(cw, int(cw * (1.0 - self.h_margin)))
        liquid_roi = bottle_crop[y_top:y_bot, x_left:x_right].copy()
        
        # Highlight mask (Specular Glare: V > 245 & S < 25) ตัดแสงสะท้อนจ้าผิดปกติ
        hsv = cv2.cvtColor(liquid_roi, cv2.COLOR_BGR2HSV)
        v_chan = hsv[:, :, 2]
        s_chan = hsv[:, :, 1]
        valid_mask = ~((v_chan > 245) & (s_chan < 25) | (v_chan < 20))
        if np.sum(valid_mask) < 20:
            valid_mask = np.ones((liquid_roi.shape[0], liquid_roi.shape[1]), dtype=bool)
            
        roi_box = (x_left, y_top, x_right - x_left, y_bot - y_top)
        return liquid_roi, valid_mask, roi_box

# ==============================================================================
# ขั้นตอนที่ 4: สกัดคุณลักษณะสี (Color Feature Extraction)
# ==============================================================================
class ColorFeatureExtractor:
    @staticmethod
    def extract_features(liquid_bgr: np.ndarray, mask: np.ndarray, blank_linear_rgb: np.ndarray = None) -> dict:
        valid_bgr = liquid_bgr[mask] if np.sum(mask) > 0 else liquid_bgr.reshape(-1, 3)
        
        p10, p90 = np.percentile(valid_bgr, [10, 90], axis=0)
        inliers = np.all((valid_bgr >= p10) & (valid_bgr <= p90), axis=1)
        trimmed_bgr = np.mean(valid_bgr[inliers], axis=0) if np.any(inliers) else np.median(valid_bgr, axis=0)
        trimmed_rgb = trimmed_bgr[[2, 1, 0]]
        
        lab_img = cv2.cvtColor(liquid_bgr, cv2.COLOR_BGR2LAB)
        valid_lab = lab_img[mask] if np.sum(mask) > 0 else lab_img.reshape(-1, 3)
        med_lab = np.median(valid_lab, axis=0)
        L_val = med_lab[0] * (100.0 / 255.0)
        a_val = float(med_lab[1]) - 128.0
        b_val = float(med_lab[2]) - 128.0
        chroma = np.sqrt(a_val**2 + b_val**2)
        
        hsv_img = cv2.cvtColor(liquid_bgr, cv2.COLOR_BGR2HSV)
        valid_hsv = hsv_img[mask] if np.sum(mask) > 0 else hsv_img.reshape(-1, 3)
        med_hsv = np.median(valid_hsv, axis=0)
        H_val = med_hsv[0] * 2.0
        S_val = med_hsv[1] / 255.0
        V_val = med_hsv[2] / 255.0
        
        linear_sample_bgr = ColorCalibrator.srgb_to_linear(np.array([[trimmed_bgr]]))[0, 0]
        linear_sample_rgb = linear_sample_bgr[[2, 1, 0]]
        
        eps = 1e-4
        if blank_linear_rgb is not None:
            transmittance = np.clip(linear_sample_rgb / (blank_linear_rgb + eps), eps, 1.0)
            absorbance = -np.log10(transmittance)
        else:
            absorbance = -np.log10(np.clip(linear_sample_rgb, eps, 1.0))
            
        lh = liquid_bgr.shape[0]
        step = max(1, lh // 3)
        patch_means = []
        for i in range(3):
            sub_p = liquid_bgr[i*step:(i+1)*step, :]
            sub_m = mask[i*step:(i+1)*step, :]
            if np.sum(sub_m) > 0:
                patch_means.append(np.mean(sub_p[sub_m], axis=0))
            else:
                patch_means.append(trimmed_bgr)
        vertical_drift = float(np.linalg.norm(patch_means[0] - patch_means[-1]))
        
        # Radial ratio (เปรียบเทียบความเข้มแสงจุดกึ่งกลางเทียบกับขอบ)
        lw = liquid_bgr.shape[1]
        mid = lw // 2
        center_strip = liquid_bgr[:, max(0, mid-5):min(lw, mid+5)]
        edge_strip = np.hstack([liquid_bgr[:, :max(1, lw//6)], liquid_bgr[:, -max(1, lw//6):]])
        m_c = np.mean(center_strip) if center_strip.size > 0 else 1.0
        m_e = np.mean(edge_strip) if edge_strip.size > 0 else 1.0
        radial_ratio = float(m_c / (m_e + 1e-5))
        
        return {
            "rgb_R": trimmed_rgb[0],
            "rgb_G": trimmed_rgb[1],
            "rgb_B": trimmed_rgb[2],
            "linear_R": linear_sample_rgb[0],
            "linear_G": linear_sample_rgb[1],
            "linear_B": linear_sample_rgb[2],
            "lab_L": L_val,
            "lab_a": a_val,
            "lab_b": b_val,
            "lab_chroma": chroma,
            "hsv_H": H_val,
            "hsv_S": S_val,
            "hsv_V": V_val,
            "abs_R": absorbance[0],
            "abs_G": absorbance[1],
            "abs_B": absorbance[2],
            "abs_max": float(np.max(absorbance)),
            "vertical_drift": vertical_drift,
            "radial_ratio": radial_ratio,
            "liquid_pixels": int(np.sum(mask))
        }

# ==============================================================================
# บันทึกภาพผลลัพธ์การ Crop ตอนทดสอบ
# ==============================================================================
class CropResultSaver:
    @staticmethod
    def save_diagnostic_panel(save_path: str, orig_img: np.ndarray, bottle_crop: np.ndarray, 
                              seg_dict: dict, liquid_roi: np.ndarray, mask: np.ndarray, 
                              vial_box: tuple, ref_box: tuple, roi_box: tuple, 
                              true_conc: float, pred_conc: float, abs_max: float,
                              liquid_pixels: int = None):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        annotated_bottle = bottle_crop.copy()
        cw = annotated_bottle.shape[1]
        cap_y = seg_dict["cap_y"]
        men_y = seg_dict["meniscus_y"]
        bot_y = seg_dict["bottom_y"]
        rx, ry, rw, rh = roi_box
        
        cv2.line(annotated_bottle, (0, cap_y), (cw, cap_y), (255, 100, 0), 2)
        cv2.putText(annotated_bottle, "Cap", (10, max(15, cap_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
        cv2.line(annotated_bottle, (0, men_y), (cw, men_y), (255, 255, 0), 2)
        cv2.putText(annotated_bottle, "Meniscus", (10, max(15, men_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.line(annotated_bottle, (0, bot_y), (cw, bot_y), (0, 255, 0), 2)
        cv2.putText(annotated_bottle, "Bottom", (10, max(15, bot_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(annotated_bottle, (rx, ry), (rx + rw, ry + rh), (0, 215, 255), 2)
        cv2.putText(annotated_bottle, "Liquid ROI", (rx, ry - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1)
        
        canvas = np.zeros((460, 1150, 3), dtype=np.uint8)
        canvas[:] = (35, 35, 35)
        
        # 1. Full image
        h_orig, w_orig = orig_img.shape[:2]
        small_orig = cv2.resize(orig_img, (300, 225))
        sx, sy = 300.0 / w_orig, 225.0 / h_orig
        cv2.rectangle(small_orig, (int(ref_box[0]*sx), int(ref_box[1]*sy)), 
                      (int((ref_box[0]+ref_box[2])*sx), int((ref_box[1]+ref_box[3])*sy)), (255, 255, 255), 2)
        cv2.rectangle(small_orig, (int(vial_box[0]*sx), int(vial_box[1]*sy)), 
                      (int((vial_box[0]+vial_box[2])*sx), int((vial_box[1]+vial_box[3])*sy)), (0, 255, 0), 2)
        canvas[50:275, 25:325] = small_orig
        cv2.putText(canvas, "1. Full Scene + Detection", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        
        # 2. Segmented bottle
        bh, bw = annotated_bottle.shape[:2]
        sc_b = 360.0 / bh
        scaled_b = cv2.resize(annotated_bottle, (max(10, int(bw * sc_b)), 360))
        bw_act = min(200, scaled_b.shape[1])
        canvas[50:410, 355:355+bw_act] = scaled_b[:, :bw_act]
        cv2.putText(canvas, "2. Segmented Bottle", (355, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        
        # 3. Full Liquid ROI
        scaled_l = cv2.resize(liquid_roi, (170, 200))
        canvas[50:250, 600:770] = scaled_l
        cv2.putText(canvas, "3. Full Liquid Column", (600, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        
        # 4. Glare Mask
        mask_vis = (mask.astype(np.uint8) * 255)
        mask_c = cv2.cvtColor(cv2.resize(mask_vis, (170, 200)), cv2.COLOR_GRAY2BGR)
        canvas[50:250, 810:980] = mask_c
        cv2.putText(canvas, "4. Valid Mask (No Glare)", (810, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        
        tot_pix = liquid_pixels if liquid_pixels is not None else int(np.sum(mask))
        if true_conc is not None:
            err = abs(pred_conc - true_conc)
            txt_res = f"True: {true_conc:.2f} mg/L | Pred: {pred_conc:.2f} mg/L (Err: {err:.2f})"
        else:
            txt_res = f"Predicted Concentration: {pred_conc:.2f} mg/L"
        cv2.putText(canvas, txt_res, (600, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(canvas, f"Absorbance: {abs_max:.4f} | Analyzed Pixels: {tot_pix:,} px", 
                    (600, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
        
        cv2.imwrite(save_path, canvas)

# ==============================================================================
# ฟังก์ชันรันทั้งระบบ
# ==============================================================================
def run_full_pipeline_with_crop_saving():
    manifest_path = r'd:\my_girlfriend_work\dataset_split\manifest.json'
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
        
    train_manifest = manifest['train']
    test_manifest = manifest['test']
    
    detector = YOLOBottleDetector()
    segmenter = HorizontalProjectionSegmenter()
    roi_extractor = LiquidROIExtractor()
    
    crop_results_base = r'd:\my_girlfriend_work\test_crop_results'
    os.makedirs(crop_results_base, exist_ok=True)
    
    print("====================================================================", flush=True)
    print(" รันระบบตาม Flow เต็มรูปแบบ พร้อมบันทึกภาพ Crop ผลลัพธ์", flush=True)
    print("====================================================================", flush=True)
    
    feature_cols = ['abs_R', 'abs_G', 'abs_B', 'abs_max', 'lab_L', 'lab_a', 'lab_b', 'lab_chroma', 'hsv_H', 'hsv_S', 'hsv_V']
    
    assays = ["Phosphate", "Ammonia_0ppt", "Ammonia_30ppt"]
    for assay_name in assays:
        if assay_name == "Phosphate":
            tr_items = [it for it in train_manifest if it['assay'] == 'Phosphate']
            te_items = [it for it in test_manifest if it['assay'] == 'Phosphate']
        elif assay_name == "Ammonia_0ppt":
            tr_items = [it for it in train_manifest if it['assay'] == 'Ammonia' and it['salinity'] == 0]
            te_items = [it for it in test_manifest if it['assay'] == 'Ammonia' and it['salinity'] == 0]
        else:
            tr_items = [it for it in train_manifest if it['assay'] == 'Ammonia' and it['salinity'] == 30]
            te_items = [it for it in test_manifest if it['assay'] == 'Ammonia' and it['salinity'] == 30]
            
        print(f"\n--- เริ่มประมวลผล: {assay_name} (Train={len(tr_items)}, Test={len(te_items)}) ---", flush=True)
        
        # 1. Blank I0 จากชุด Train
        blank_items = [it for it in tr_items if it['conc'] == 0.0]
        if not blank_items:
            min_c = min(it['conc'] for it in tr_items)
            blank_items = [it for it in tr_items if it['conc'] == min_c]
            
        blank_lin_list = []
        for it in blank_items[:10]:
            img = load_standard_image(it['full_path'])
            if img is not None:
                wb_img, _ = ColorCalibrator.apply_white_balance(img)
                det = detector.detect_and_crop(wb_img)
                seg = segmenter.segment(det['crop'])
                l_roi, mask, _ = roi_extractor.extract_liquid_roi(det['crop'], seg)
                if np.sum(mask) > 0:
                    m_bgr = np.mean(l_roi[mask], axis=0)
                    lin_rgb = ColorCalibrator.srgb_to_linear(np.array([[m_bgr]]))[0, 0][[2, 1, 0]]
                    blank_lin_list.append(lin_rgb)
        blank_ref_rgb = np.mean(blank_lin_list, axis=0) if blank_lin_list else np.array([0.5, 0.5, 0.5])
        
        # 2. สกัด Feature จากชุด Train
        train_feats = []
        for it in tr_items:
            img = load_standard_image(it['full_path'])
            if img is None:
                continue
            wb_img, _ = ColorCalibrator.apply_white_balance(img)
            det = detector.detect_and_crop(wb_img)
            seg = segmenter.segment(det['crop'])
            l_roi, mask, _ = roi_extractor.extract_liquid_roi(det['crop'], seg)
            f = ColorFeatureExtractor.extract_features(l_roi, mask, blank_ref_rgb)
            f['conc'] = it['conc']
            train_feats.append(f)
            
        df_tr = pd.DataFrame(train_feats)
        X_tr = df_tr[feature_cols].values
        y_tr = df_tr['conc'].values
        
        # 3. เทรนโมเดล PLS Regression บน Train
        pls = PLSRegression(n_components=2)
        pls.fit(X_tr, y_tr)
        
        # บันทึกโมเดลสำหรับนำไปใช้งานในอนาคต (Inference)
        os.makedirs(r'd:\my_girlfriend_work\saved_models', exist_ok=True)
        model_save_path = os.path.join(r'd:\my_girlfriend_work\saved_models', f"{assay_name}_pls.pkl")
        with open(model_save_path, 'wb') as mf:
            pickle.dump({
                "model": pls,
                "feature_cols": feature_cols,
                "blank_ref": blank_ref_rgb,
                "lod": float(np.std(y_tr) * 0.1),
                "loq": float(np.std(y_tr) * 0.3)
            }, mf)
        
        # 4. ทดสอบและบันทึกภาพ Crop ของชุด TEST
        assay_crop_dir = os.path.join(crop_results_base, assay_name)
        os.makedirs(assay_crop_dir, exist_ok=True)
        
        print(f"  กำลังประมวลผล Test Set ({len(te_items)} ภาพ) พร้อมบันทึกภาพ Crop Results...", flush=True)
        saved_count = 0
        test_preds = []
        test_trues = []
        
        for idx, it in enumerate(te_items):
            img = load_standard_image(it['full_path'])
            if img is None:
                continue
            wb_img, ref_box = ColorCalibrator.apply_white_balance(img)
            det = detector.detect_and_crop(wb_img)
            seg = segmenter.segment(det['crop'])
            seg['deskew_angle'] = det['deskew_angle']
            l_roi, mask, roi_box = roi_extractor.extract_liquid_roi(det['crop'], seg)
            feats = ColorFeatureExtractor.extract_features(l_roi, mask, blank_ref_rgb)
            
            X_sample = np.array([[feats[c] for c in feature_cols]])
            pred_val = max(0.0, float(pls.predict(X_sample).flatten()[0]))
            
            test_preds.append(pred_val)
            test_trues.append(it['conc'])
            
            # บันทึกภาพ Diagnostic Panel ตอน Crop
            filename_base = os.path.splitext(os.path.basename(it['full_path']))[0]
            crop_save_path = os.path.join(assay_crop_dir, f"crop_{it['conc']}mg_{filename_base}.jpg")
            
            CropResultSaver.save_diagnostic_panel(
                save_path=crop_save_path,
                orig_img=img,
                bottle_crop=det['crop'],
                seg_dict=seg,
                liquid_roi=l_roi,
                mask=mask,
                vial_box=det['bbox'],
                ref_box=ref_box,
                roi_box=roi_box,
                true_conc=it['conc'],
                pred_conc=pred_val,
                abs_max=feats['abs_max']
            )
            saved_count += 1
            
        r2 = r2_score(test_trues, test_preds)
        rmse = np.sqrt(mean_squared_error(test_trues, test_preds))
        print(f"  [สำเร็จ] บันทึกผลลัพธ์การ Crop ของ {assay_name} รวม {saved_count} ภาพ ที่: {assay_crop_dir}", flush=True)
        print(f"  Unbiased Test Performance: R² = {r2:.4f}, RMSE = {rmse:.4f} mg/L", flush=True)

if __name__ == '__main__':
    run_full_pipeline_with_crop_saving()
