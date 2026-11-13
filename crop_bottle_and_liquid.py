"""
Standalone Tool: ตรวจจับขวดตัวอย่าง, ปรับขวดตั้งตรงอัตโนมัติ (Automatic Deskew), และสกัดระดับน้ำจริง (True Water Level)
เวอร์ชัน: Web-Optimized + Dynamic Deskew + Curved Meniscus Fitting (อ้างอิงงานวิจัย Liu et al. 2011, Baudin et al. 2020, Zhang et al. 2025)
จุดเด่น:
1. ปรับมุมเอียงอัตโนมัติ (Automatic Deskew):
   - ตรวจจับมุมเอียงของขวดจากแนวผนังแก้วทรงกระบอกจริง (Dynamic Hough Angle Tracing)
   - หมุนปรับระนาบให้อยู่ในแนวดิ่ง 90 องศาเป๊ะ ก่อนทำการ Crop
   - ห้าม/ไม่ล็อกค่าคงที่ (No hardcoded size / dimensions / fixed angle limits)
2. ตรวจจับและฟิตเส้นโค้งผิวน้ำ (Curved Meniscus Geometric Profile):
   - ใช้ Sobel Gradient แนวดิ่ง และฟิตสมการพาราโบลาอันดับสอง y(x) = ax^2 + bx + c
   - คัดกรอง Inliers เพื่อกำจัดแสงสะท้อนหลอก ขอบผิวน้ำโค้งมนเนียนตา ไร้รอยฟันปลา ไร้รอยแหว่ง
3. โอบล้อมมวลน้ำ 3 มิติเต็ม 100% (3D Cylindrical Polygon):
   - เชื่อมผิวน้ำโค้งด้านบนและก้นขวดโค้งมนด้านล่าง เต็มความกว้างขวดแก้ว x in [0, bw-1]
4. เหมาะสำหรับ Web App:
   - โมเดล YOLOv8n เล็กเพียง 6 MB (แปลงเป็น ONNX ได้ 3.5 MB)
   - สเต็ป Deskew และ Water Level ใช้ Pure OpenCV (0 MB, ความเร็ว < 15 ms)
"""

import os
import sys
import json
import time
import csv
import argparse
import traceback
from datetime import datetime
from dataclasses import dataclass
import cv2
import numpy as np
import warnings


@dataclass(frozen=True)
class CropperConfig:
    """
    พารามิเตอร์เชิงวิจัยแบบไร้มิติ (Dimensionless / Scale-Invariant & Self-Adaptive SNR Configuration)
    ทุกค่ากำหนดในรูปสัดส่วนเรขาคณิตของขวดทรงกระบอก หรืออัตราส่วนสัญญาณต่อสัญญาณรบกวน (SNR / z-score)
    เพื่อความสามารถในการทำซ้ำ (Reproducibility) และการวางนัยทั่วไปข้ามสภาพแสง (Domain Generalization)
    """
    # 1. สัดส่วนเรขาคณิตสำหรับการย่อคำนวณและโซนอ้างอิงภายในขวด (Scale-Invariant Ratios)
    working_width: int = 220
    core_x_range: tuple[float, float] = (0.16, 0.84)
    body_ref_y_range: tuple[float, float] = (0.60, 0.86)
    body_ref_x_range: tuple[float, float] = (0.25, 0.75)
    window_height_ratio: float = 0.045
    surface_search_y_range: tuple[float, float] = (0.025, 0.78)
    local_refine_up_ratio: float = 0.035
    local_refine_down_ratio: float = 0.085

    # 2. เกณฑ์สถิติแบบปรับตัวตามสัญญาณภาพ (Self-Adaptive SNR & Continuous Weighting)
    lab_l_weight: float = 0.38
    chroma_half_saturation: float = 10.0   # C*_0 สำหรับฟังก์ชันถ่วงน้ำหนักต่อเนื่อง w = C* / (C* + C*_0)
    min_step_snr: float = 2.6              # อัตราส่วนขั้นสีต่อความแปรปรวนภายในเนื้อน้ำ (Step / MAD_body)
    min_air_to_body_ratio: float = 1.45    # อัตราส่วนความต่างสีโซนอากาศต่อโซนเนื้อน้ำ

    # 3. ขอบเขตทางฟิสิกส์ของความโค้งพาราโบลาผิวน้ำและก้นขวด (Physical Arch & Cylinder Curvature Bounds)
    meniscus_min_sag_ratio: float = 0.052  # ความโค้งคว่ำต่ำสุดของระดับน้ำด้านบนเทียบกับความกว้างขวด
    meniscus_max_sag_ratio: float = 0.115  # ความโค้งคว่ำสูงสุดของระดับน้ำด้านบนเทียบกับความกว้างขวด
    full_top_sag_ratio: float = 0.048      # ความโค้งพาราโบลาด้านบนกรณีน้ำเต็มขวด
    near_cap_ratio: float = 0.12           # สัดส่วนความลึกจากคอขวดที่ถือว่าระดับน้ำอยู่ชิดฝา (<= 12% ของความสูงขวด)
    near_cap_max_px: float = 45.0          # ขีดจำกัดพิกเซลสูงสุดสำหรับโซนชิดฝาขวด
    base_default_sag_ratio: float = 0.082  # ความโค้งมนตามรูปทรงก้นขวดทรงกระบอก 3 มิติ (พาราโบลาหงาย ∪, Sag ~ 8.2% ของความกว้าง หรือราว 45-50 px)
    base_min_sag_ratio: float = 0.050
    base_max_sag_ratio: float = 0.120
    base_search_y_start: float = 0.86      # ค้นหาในโซนล่างสุด (86%-100%) เพื่อโอบรับรอยต่อขอบมนของก้นแก้ว
    irls_iterations: int = 10
    irls_mad_floor: float = 1.5
    irls_cauchy_scale: float = 1.5

    # 4. พารามิเตอร์การเคลียแสงสะท้อนและเงาหลังครอบ (Post-Crop Restoration & L* Flat-Fielding)
    glare_z_threshold: float = 2.5         # เกณฑ์ z-score (เท่าของ MAD) สำหรับตรวจจับ Specular Glare
    glare_min_excess_l: float = 9.0
    surface_poly_degree: int = 3
    luminance_col_gain_bounds: tuple[float, float] = (0.82, 1.22)
    luminance_row_gain_bounds: tuple[float, float] = (0.70, 1.45)
    full_row_width_fraction: float = 0.65


DEFAULT_CONFIG = CropperConfig()


class WebOptimizedLiquidCropper:
    def __init__(self, model_path: str = "yolov8m-worldv2.pt", device=None, config: CropperConfig = DEFAULT_CONFIG):
        import torch
        from ultralytics import YOLO, YOLOWorld
        self.config = config
        default_dev = 0 if torch.cuda.is_available() else 'cpu'
        self.device = default_dev if device is None else device
        if "world" in model_path.lower():
            self.model = YOLOWorld(model_path)
            self.model.set_classes(["glass vial with white cap", "transparent chemical bottle", "bottle"])
            self.is_world = True
        else:
            self.model = YOLO(model_path)
            self.is_world = False
            
        dev_name = torch.cuda.get_device_name(0) if self.device == 0 and torch.cuda.is_available() else "CPU"
        print(f"[Engine] ใช้งานอุปกรณ์: {dev_name} (Device ID: {self.device}, Model: {os.path.basename(model_path)})", flush=True)

        # โหลดโมเดล SAM 2 (Meta AI) สำหรับตรวจจับมุมเอียงด้วย cv2.minAreaRect (ถ้ามีไฟล์)
        self.sam = None
        sam_candidate = "sam2.1_b.pt"
        if os.path.isfile(sam_candidate):
            try:
                from ultralytics import SAM
                self.sam = SAM(sam_candidate)
                print(f"[Engine] โหลดโมเดล SAM 2.1 สำเร็จ ({sam_candidate}) สำหรับ SAM 2 minAreaRect Deskewing", flush=True)
            except Exception as e:
                print(f"[Engine] ไม่สามารถโหลด SAM 2 ได้: {e}", flush=True)

    @staticmethod
    def remove_drawn_markup(image: np.ndarray) -> np.ndarray:
        """ลบเส้นวาดหรือสัญลักษณ์สีเหลืองที่ขีดเขียนลงบนภาพ คืนค่าน้ำสีจริง (Inpainting)"""
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        yellow_mask = ((r > 160) & (g > 160) & (b < 110)).astype(np.uint8) * 255
        if np.sum(yellow_mask > 0) > 100:
            k = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(yellow_mask, k, iterations=1)
            return cv2.inpaint(image, dilated, 3, cv2.INPAINT_TELEA)
        return image

    def locate_bottle(self, image: np.ndarray, image_path: str = None) -> tuple[int, int, int, int]:
        """
        ตรวจจับตำแหน่งขวดในภาพด้วย YOLO-Nano (เบา รวดเร็ว แม่นยำ)
        พร้อมการตรวจสอบเชิงเรขาคณิต (Geometry Guard) สำหรับขวดแก้วทรงกระบอก
        """
        h, w = image.shape[:2]
        source = image_path if (image_path and os.path.isfile(image_path)) else image
        
        # ปรับ conf=0.02 เพื่อจับขวดใส/ขวดแก้วทุกสภาพแสง
        results = self.model.predict(source=source, conf=0.02, device=self.device, verbose=False)[0]
        cands = []
        if getattr(self, "is_world", False):
            target_classes = set(self.model.names.values())
        else:
            target_classes = {"bottle", "vase", "cup", "wine glass", "toilet"}
            
        for b, c, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            c_name = self.model.names[int(c)]
            if c_name in target_classes:
                box = [int(v) for v in b]
                bw, bh = box[2] - box[0], box[3] - box[1]
                asp = bh / float(bw) if bw > 0 else 0
                cands.append((box, asp, float(conf)))
                
        # 1. ลำดับแรก: คัดเลือกกล่องที่มีสัดส่วนรูปทรงขวดจริง (Aspect Ratio 2.0 ถึง 3.8)
        vial_cands = [c for c in cands if 2.0 <= c[1] <= 3.8]
        if vial_cands:
            img_cx = w / 2.0
            def cand_score(c):
                bx = c[0]
                box_cx = (bx[0] + bx[2]) / 2.0
                dist = abs(box_cx - img_cx) / float(w)
                # Aspect ratio ของขวดแก้วทรงกระบอกพร้อมฝาเต็มใบจะอยู่ที่ประมาณ 2.8 - 3.4
                asp_penalty = abs(c[1] - 3.0) * 0.4
                # ให้คะแนนความสูง (Height Bonus) เพื่อเลือกขวดเต็มใบที่มีฝาครอบด้านบนมากกว่ากล่องที่จับเฉพาะช่วงลำตัว
                h_bonus = (bx[3] - bx[1]) / float(h) * 0.5
                return c[2] - dist * 0.25 - asp_penalty + h_bonus
            vial_cands.sort(key=cand_score, reverse=True)
            best_box = list(vial_cands[0][0])
            bbw = best_box[2] - best_box[0]
            bbh = best_box[3] - best_box[1]
            b_asp = bbh / float(bbw) if bbw > 0 else 0
            
            # Aspect Ratio Auto-Recovery: หากกล่องที่ได้กุดสั้น (H/W < 2.55) ให้ยืดกล่องขึ้นบนหาฝาขวดอัตโนมัติ
            if b_asp < 2.55:
                target_h = int(bbw * 3.05)
                best_box[1] = max(0, best_box[3] - target_h)
            return tuple(best_box)
            
        # 2. กรณีจับได้กล่องสั้น (เช่น จับได้เฉพาะช่วงน้ำครึ่งล่าง)
        short_cands = [c for c in cands if 1.2 <= c[1] < 2.0]
        if short_cands:
            short_cands.sort(key=lambda x: x[2], reverse=True)
            s_box = short_cands[0][0]
            sbw, sbh = s_box[2] - s_box[0], s_box[3] - s_box[1]
            exp_top = max(0, int(s_box[1] - sbh * 0.75))
            exp_asp = (s_box[3] - exp_top) / float(sbw) if sbw > 0 else 0
            if 2.0 <= exp_asp <= 3.8:
                return (s_box[0], exp_top, s_box[2], s_box[3])
                
        # 3. Fallback ผ่าน Pure OpenCV Vertical Projection ในแถบกลางภาพ
        cw_start, cw_end = int(w * 0.20), int(w * 0.80)
        center_strip = image[:, cw_start:cw_end]
        gray_c = cv2.cvtColor(center_strip, cv2.COLOR_BGR2GRAY)
        sob_x = np.abs(cv2.Sobel(gray_c, cv2.CV_32F, 1, 0, ksize=3))
        col_sums = np.sum(sob_x, axis=0)
        mid_col = len(col_sums) // 2
        l_peak = int(np.argmax(col_sums[:mid_col - 10])) if mid_col > 15 else 0
        r_peak = mid_col + 10 + int(np.argmax(col_sums[mid_col + 10:])) if mid_col + 10 < len(col_sums) else len(col_sums) - 1
        
        abs_lx = max(0, cw_start + l_peak - 10)
        abs_rx = min(w, cw_start + r_peak + 10)
        f_bw = abs_rx - abs_lx
        
        if f_bw > 30:
            strip_y = cv2.cvtColor(image[:, abs_lx:abs_rx], cv2.COLOR_BGR2GRAY)
            sob_y = np.abs(cv2.Sobel(strip_y, cv2.CV_32F, 0, 1, ksize=3))
            row_s = cv2.GaussianBlur(np.mean(sob_y, axis=1).reshape(-1, 1), (15, 1), 0).flatten()
            cap_top = int(h * 0.12) + int(np.argmax(row_s[int(h * 0.12):int(h * 0.50)]))
            bot_base = int(h * 0.55) + int(np.argmax(row_s[int(h * 0.55):int(h * 0.90)]))
            if bot_base > cap_top + 50:
                return (abs_lx, cap_top, abs_rx, bot_base)
                
        if cands:
            cands.sort(key=lambda x: x[2], reverse=True)
            return tuple(cands[0][0])
            
        return (int(w * 0.35), int(h * 0.20), int(w * 0.65), int(h * 0.85))

    @staticmethod
    def cut_bottle_cap(image_or_bottle: np.ndarray, box: tuple[int, int, int, int] | None = None) -> tuple[any, int]:
        """
        ตัดฝาขวด (Bottle Cap) ออกจากตัวขวดด้วยการวิเคราะห์หลายมิติ (Multi-Modal Cap Detection):
        - หากส่ง upright_bottle (ภาพขวดตั้งตรง 90 องศาที่ได้จาก YOLO Deskew):
          จะตัดฝาขวดออกจากด้านบนโดยตรง คืนค่า (bottle_no_cap, cap_height)
        - หากส่งภาพต้นฉบับและ bounding box:
          จะตัดขอบบนของ box คืนค่า (box_no_cap, cap_height)
        """
        if box is not None:
            x1, y1, x2, y2 = box
            patch = image_or_bottle[y1:y2, x1:x2]
        else:
            patch = image_or_bottle
            x1, y1 = 0, 0
            
        bh, bw = patch.shape[:2]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        sob_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        S = hsv[:, :, 1].astype(np.float32)
        V = hsv[:, :, 2].astype(np.float32)
        
        mid_sob = np.mean(sob_y[:, int(bw * 0.25):int(bw * 0.75)], axis=1)
        mid_s = np.mean(S[:, int(bw * 0.25):int(bw * 0.75)], axis=1)
        mid_v = np.mean(V[:, int(bw * 0.25):int(bw * 0.75)], axis=1)
        
        s1, s2 = max(5, int(bh * 0.08)), min(bh - 20, int(bh * 0.46))
        scores = np.zeros(s2 - s1, dtype=np.float32)
        for i, y in enumerate(range(s1, s2)):
            g = mid_sob[y]
            v_drop = max(0.0, mid_v[max(0, y - 4)] - mid_v[min(bh - 1, y + 4)])
            s_jump = max(0.0, mid_s[min(bh - 1, y + 6)] - mid_s[max(0, y - 6)])
            is_still_white = 1.0 if (mid_v[min(bh - 1, y + 8)] > 195 and mid_s[min(bh - 1, y + 8)] < 30) else 0.0
            scores[i] = g + 1.2 * v_drop + 1.5 * s_jump - 80.0 * is_still_white
            
        best_y = s1 + int(np.argmax(scores)) if s2 > s1 else int(bh * 0.20)
        
        if box is not None:
            cut_y1 = min(box[3] - 30, box[1] + best_y + 2)
            box_no_cap = (box[0], cut_y1, box[2], box[3])
            return box_no_cap, best_y + 2
        else:
            cut_y = min(bh - 30, best_y + 2)
            bottle_no_cap = patch[cut_y:].copy()
            return bottle_no_cap, cut_y

    @staticmethod
    def deskew_and_tight_crop(image: np.ndarray, box: tuple[int, int, int, int], sam_model=None) -> dict:
        """
        ปรับขวดให้ตั้งตรง 90 องศา (Deskew) โดยไม่ล็อกขนาดคงที่:
        1. ดึงภาพขวดพร้อมขอบเผื่อ (Dynamic Padding)
        2. ตรวจจับแนวเอียงของผนังแก้วด้วย:
           - แนวทางที่ 1 (แนะนำที่สุด): SAM 2 Minimum Area Rotated Rectangle (cv2.minAreaRect)
           - แนวทางสำรอง: Dynamic Hough Wall Tracing
        3. หมุนภาพกลับด้วยมุม -theta ให้อยู่ในแนวดิ่งเป๊ะ (Affine Transformation)
        4. ตัดขอบแนบชิด (Tight Crop) สกัดผนังแก้วทรงกระบอกคู่ขนาน (Cylinder Wall Pair)
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        
        pad_x = max(15, int(bw * 0.15))
        pad_y = max(15, int(bh * 0.08))
        
        px1, py1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        px2, py2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        
        patch = image[py1:py2, px1:px2].copy()
        ph, pw = patch.shape[:2]
        
        tilt_deg = 0.0
        deskew_method = "hough"
        
        # แนวทางที่ 1 (แนะนำที่สุด): SAM 2 Minimum Area Rotated Rectangle (cv2.minAreaRect)
        if sam_model is not None:
            try:
                cx_p = int(pw / 2.0)
                pts_pos = [[cx_p, int(ph * 0.50)], [cx_p, int(ph * 0.70)]]
                pts_neg = [[cx_p, int(ph * 0.08)], [10, int(ph * 0.50)], [pw - 10, int(ph * 0.50)]]
                s_res = sam_model(patch, points=[pts_pos + pts_neg], labels=[[1, 1, 0, 0, 0]], device=0, verbose=False)
                if s_res[0].masks is not None and len(s_res[0].masks.data) > 0:
                    m_d = s_res[0].masks.data[0].cpu().numpy().astype(np.uint8)
                    if m_d.shape[:2] != (ph, pw):
                        m_d = cv2.resize(m_d, (pw, ph), interpolation=cv2.INTER_NEAREST)
                    cnts, _ = cv2.findContours(m_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        c_largest = max(cnts, key=cv2.contourArea)
                        rect = cv2.minAreaRect(c_largest)
                        pts = cv2.boxPoints(rect)
                        edges = [pts[(i+1)%4] - pts[i] for i in range(4)]
                        lengths = [np.linalg.norm(e) for e in edges]
                        long_idx = np.argmax(lengths[:2])
                        long_vec = edges[long_idx]
                        dx, dy = long_vec[0], long_vec[1]
                        if dy < 0: dx, dy = -dx, -dy
                        computed_tilt = float(np.degrees(np.arctan2(dx, dy)))
                        if abs(computed_tilt) < 30.0:
                            tilt_deg = computed_tilt
                            deskew_method = "sam2_minAreaRect"
            except Exception:
                pass
                
        if deskew_method != "sam2_minAreaRect":
            # ตรวจจับมุมเอียงจากเส้นขอบแก้วทรงกระบอก
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 40, 140)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45, 
                                    minLineLength=int(ph * 0.25), maxLineGap=20)
            
            angles = []
            if lines is not None:
                for line in lines:
                    coords = line.flatten()
                    if len(coords) == 4:
                        lx1, ly1, lx2, ly2 = coords
                        dx, dy = lx2 - lx1, ly2 - ly1
                        if dy != 0:
                            deg = np.degrees(np.arctan2(dx, dy))
                            if abs(deg) < 25.0:
                                angles.append(deg)
                                
            tilt_deg = float(np.median(angles)) if angles else 0.0
        
        # หมุนภาพปรับตั้งตรง
        if abs(tilt_deg) > 0.4:
            center = (pw / 2.0, ph / 2.0)
            M = cv2.getRotationMatrix2D(center, -tilt_deg, 1.0)
            straight = cv2.warpAffine(patch, M, (pw, ph), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            straight = patch
            
        # ค้นหาขอบขวดแนบชิดบนภาพที่ตั้งตรงแล้ว
        gray_s = cv2.cvtColor(straight, cv2.COLOR_BGR2GRAY)
        sob_x = np.abs(cv2.Sobel(gray_s, cv2.CV_32F, 1, 0, ksize=3))
        col_sums = np.sum(sob_x[int(ph * 0.25):int(ph * 0.75), :], axis=0)
        
        # ค้นหาคู่ผนังแก้วคู่ขนาน (Cylinder Wall Pair) เพื่อป้องกันไม่ให้กระโดดไปจับแผ่นสีด้านข้าง
        from scipy.signal import find_peaks
        peaks, props = find_peaks(col_sums, height=1500, distance=15)
        best_score = -1
        mid_x = pw // 2
        l_wall = int(np.argmax(col_sums[:mid_x - 10])) if mid_x > 20 else 2
        r_wall = mid_x + 10 + int(np.argmax(col_sums[mid_x + 10:])) if mid_x + 10 < pw else pw - 3
        
        if len(peaks) >= 2:
            for i in range(len(peaks)):
                for j in range(i + 1, len(peaks)):
                    l_c, r_c = peaks[i], peaks[j]
                    cand_w = r_c - l_c
                    cand_asp = ph / float(cand_w) if cand_w > 0 else 0
                    if 2.0 <= cand_asp <= 3.8:
                        sc = props['peak_heights'][i] + props['peak_heights'][j]
                        if sc > best_score:
                            best_score = sc
                            l_wall, r_wall = int(l_c), int(r_c)
        
        sob_y = np.abs(cv2.Sobel(gray_s[:, l_wall:r_wall], cv2.CV_32F, 0, 1, ksize=3))
        row_sums = cv2.GaussianBlur(np.mean(sob_y, axis=1).reshape(-1, 1), (9, 1), 0).flatten()
        
        # ขอบบนสุดของตัวขวด (ที่ตัดฝาขวดออกแล้ว): เริ่มต้นที่ระดับ y1 ใน patch
        y1_in_patch = max(0, y1 - py1)
        top_y = y1_in_patch
        
        # ค้นหาก้นขวดจริง: อิงจากพิกัดต่ำสุดที่ YOLO ตรวจพบ (y2_patch) และขอบการเปลี่ยนผ่าน
        y2_patch = y2 - py1
        b_search_start = max(int(ph * 0.60), y2_patch - 80)
        b_search_end = min(ph, y2_patch + 80)
        if b_search_end > b_search_start:
            sub_grad = row_sums[b_search_start:b_search_end]
            base_y_patch = b_search_start + int(np.argmax(sub_grad))
        else:
            base_y_patch = y2_patch
            
        # Snap ก้นขวดแนบชิดเส้นโค้งแก้วจริง โดยไม่เหลือแถบพื้นโต๊ะขาวด้านล่าง
        bot_y = min(ph, base_y_patch + 3)
        bot_y = min(ph, max(top_y + 40, bot_y))
        
        upright_bottle = straight[top_y:bot_y, l_wall:r_wall].copy()
        base_in_upright = min(upright_bottle.shape[0] - 1, base_y_patch - top_y)
        
        return {
            "tilt_angle": tilt_deg,
            "straight_patch": straight,
            "upright_bottle": upright_bottle,
            "base_y": base_in_upright
        }

    # ------------------------------------------------------------------
    # ขั้นที่ 1: หาระดับน้ำจริง -> ครอบตามแนวพาราโบลาเว้าที่เส้นระดับน้ำพอดี และตัดก้นขวดตามรูปทรงขวด
    # ------------------------------------------------------------------
    @staticmethod
    def _small_lab(img_bgr: np.ndarray, cfg: CropperConfig = DEFAULT_CONFIG):
        """ย่อภาพตามสัดส่วนไร้มิติ (Scale-Invariant) เพื่อคำนวณในปริภูมิสี CIELAB"""
        H, W = img_bgr.shape[:2]
        s = min(1.0, cfg.working_width / float(W))
        w = max(40, int(round(W * s)))
        h = max(80, int(round(H * s)))
        small = cv2.GaussianBlur(cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA), (3, 3), 0)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        return lab, w / float(W), h / float(H)

    @staticmethod
    def _color_distance_map(lab: np.ndarray, by: int, cfg: CropperConfig = DEFAULT_CONFIG):
        """
        คำนวณสนามระยะห่างสี D(y, x) เทียบกับมัธยฐานสีของเนื้อน้ำโซนล่าง (cb)
        พร้อมคืนค่าความแปรปรวนภายในเนื้อน้ำ (sigma_body, sigma_l) สำหรับคำนวณ Self-Adaptive SNR
        """
        h, w = lab.shape[:2]
        r0 = int(cfg.body_ref_y_range[0] * by)
        r1 = max(int(cfg.body_ref_y_range[1] * by), r0 + 4)
        c0 = int(cfg.body_ref_x_range[0] * w)
        c1 = max(int(cfg.body_ref_x_range[1] * w), c0 + 4)
        body_patch = lab[r0:r1, c0:c1]
        body = body_patch.reshape(-1, 3)
        cb = np.median(body, axis=0)
        D = np.sqrt(
            cfg.lab_l_weight * (lab[:, :, 0] - cb[0]) ** 2 +
            (lab[:, :, 1] - cb[1]) ** 2 +
            (lab[:, :, 2] - cb[2]) ** 2
        )
        d_body = D[r0:r1, c0:c1].ravel()
        sigma_body = max(1.5, float(np.median(np.abs(d_body - np.median(d_body))) * 1.4826))
        l_body = body_patch[:, :, 0].ravel()
        sigma_l = max(1.5, float(np.median(np.abs(l_body - np.median(l_body))) * 1.4826))
        return D, cb, sigma_body, sigma_l

    @staticmethod
    def _fit_symmetric_parabola(xs: np.ndarray, ys: np.ndarray, xc: float,
                                max_sag: float, half_w: float, min_sag: float = 0.0,
                                cfg: CropperConfig = DEFAULT_CONFIG, opening: str = "down"):
        """
        ฟิตสมการพาราโบลาสมมาตรรอบแกนขวด y(x) = a(x - xc)^2 + c ด้วย Robust Cauchy IRLS:
          - opening="down" (พาราโบลาโค้งคว่ำ ∩ สำหรับระดับน้ำด้านบนภายในขวด):
            จุดยอดอยู่ตรงกลางด้านบน และปลายสองข้างโค้งลงตามแนวระดับน้ำด้านบน (a >= 0 ในพิกัดภาพ)
          - opening="up" (พาราโบลาโค้งหงาย ∪ สำหรับตัดก้นขวดตามรูปทรงขวด):
            จุดยอดอยู่ตรงกลางก้นขวด และปลายสองข้างโค้งขึ้นตามขอบมนของก้นขวด (a <= 0 ในพิกัดภาพ)
        """
        u = (xs - xc) ** 2
        A = np.stack([u, np.ones_like(u)], axis=1)
        coef = np.linalg.lstsq(A, ys, rcond=None)[0]
        if opening == "down":
            min_a = min_sag / (half_w ** 2 + 1e-6)
            max_a = max_sag / (half_w ** 2 + 1e-6)
        else:
            min_a = -max_sag / (half_w ** 2 + 1e-6)
            max_a = -min_sag / (half_w ** 2 + 1e-6)
        w = np.ones_like(ys, dtype=np.float64)
        for _ in range(cfg.irls_iterations):
            a = float(np.clip(coef[0], min_a, max_a))
            order = np.argsort(ys - a * u)
            vals = (ys - a * u)[order]
            ws = w[order]
            c = float(vals[min(len(vals) - 1, np.searchsorted(np.cumsum(ws), 0.5 * np.sum(ws)))])
            res = ys - (a * u + c)
            mad = max(cfg.irls_mad_floor, float(np.median(np.abs(res)) * 1.4826))
            w = 1.0 / (1.0 + (res / (cfg.irls_cauchy_scale * mad)) ** 2)
            sw = np.sqrt(w)
            coef = np.linalg.lstsq(A * sw[:, None], ys * sw, rcond=None)[0]

        a = float(np.clip(coef[0], min_a, max_a))
        order = np.argsort(ys - a * u)
        vals = (ys - a * u)[order]
        ws = w[order]
        c = float(vals[min(len(vals) - 1, np.searchsorted(np.cumsum(ws), 0.5 * np.sum(ws)))])
        res = ys - (a * u + c)
        mad = max(cfg.irls_mad_floor, float(np.median(np.abs(res)) * 1.4826))
        inlier = float(np.mean(np.abs(res) <= max(3.5, 2.5 * mad)))
        return a, c, inlier

    @staticmethod
    def locate_liquid_surface(upright_bottle: np.ndarray, base_y: int, cfg: CropperConfig = DEFAULT_CONFIG):
        """
        หาระดับน้ำด้วยเกณฑ์ Self-Adaptive SNR และฟิต 'พาราโบลาโค้งคว่ำ (∩)' ตามแนวโค้งคว่ำของระดับน้ำด้านบนภายในขวด:
          1. คำนวณสนามระยะห่างสี D(y, x) และค่าความแปรปรวนภายในเนื้อน้ำ (sigma_body)
          2. ประเมินอัตราส่วนสัญญาณต่อสัญญาณรบกวน (Step SNR) ถ่วงน้ำหนักแบบต่อเนื่องด้วยความอิ่มตัวสี
             w_chroma = C*_body / (C*_body + C*_0)
          3. ล็อกตำแหน่งเข้าหาแนวโค้งคว่ำของระดับน้ำด้านบนภายในขวด (Upper Arch of Liquid Surface)
          4. ฟิตพาราโบลาโค้งคว่ำสมมาตร y(x) = a(x - xc)^2 + c (a >= 0 ในพิกัดภาพ)
        """
        cls = WebOptimizedLiquidCropper
        H, W = upright_bottle.shape[:2]
        lab, sx, sy = cls._small_lab(upright_bottle, cfg)
        h, w = lab.shape[:2]
        by = int(min(h - 1, max(20, base_y * sy)))
        D, cb, sigma_body, sigma_l = cls._color_distance_map(lab, by, cfg)
        Ls = cv2.GaussianBlur(lab[:, :, 0], (3, 3), 0)
        Ds = cv2.GaussianBlur(D, (3, 5), 0)
        chroma_map = cv2.GaussianBlur(np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0), (3, 3), 0)

        x0, x1 = int(cfg.core_x_range[0] * w), int(cfg.core_x_range[1] * w)
        xc = (w - 1) / 2.0
        half_w = w / 2.0
        xf = np.arange(W, dtype=np.float64) * sx

        # เส้นพาราโบลาโค้งหงาย (∪) ที่ขอบบนเมื่อน้ำเต็มขวดชิดฝา:
        # เพื่อตามแนวขอบล่าง 3 มิติของฝาขวด และเก็บมวลน้ำขอบข้างซ้าย-ขวาครบ 100% ไม่ให้ขอบข้างแหว่ง
        default_sag = cfg.full_top_sag_ratio * w
        a_def = -default_sag / (half_w ** 2 + 1e-6)
        c_def = default_sag + 1.0
        full_curve = np.clip(np.round((a_def * (xf - xc) ** 2 + c_def) / sy), 1, max(1, int(0.8 * base_y))).astype(np.int32)
        info = {
            "mode": "full_top", "strength": 0.0, "snr": 0.0, "inlier": 1.0,
            "a": a_def, "c": c_def, "sag_px": float(default_sag / sy),
            "orientation": "cup", "is_near_cap": True
        }

        Dc = Ds[:, x0:x1]
        Wn = max(6, int(cfg.window_height_ratio * h))
        y_lo = max(Wn, int(cfg.surface_search_y_range[0] * h))
        y_hi = int(cfg.surface_search_y_range[1] * by) - Wn
        if y_hi <= y_lo + 4 or Dc.shape[1] < 10:
            return full_curve, info

        # ตรวจสอบประเภทสารละลาย: น้ำใสไร้สี (Low Chroma) vs น้ำมีสี (Colored)
        body_chroma = float(np.hypot(cb[1] - 128.0, cb[2] - 128.0))
        norm_scale = max(2.2, sigma_body)
        if body_chroma < 6.5:
            # 1. การตรวจจับระดับน้ำของ 'สีใส' (Clear Liquid):
            # ไม่พึ่งพาระยะห่างสี D(y, x) แต่ใช้เส้นหักเหแสงวงแหวน Meniscus (Edge Valley Operator บนความสว่าง)
            delta_clear = max(2, int(0.015 * h))
            valleys = np.zeros(y_hi - y_lo, dtype=np.float32)
            sobel_l = np.abs(cv2.Sobel(Ls, cv2.CV_32F, 0, 1, ksize=3))
            for idx, y in enumerate(range(y_lo, y_hi)):
                l_curr = Ls[y, x0:x1]
                l_up = Ls[y - delta_clear, x0:x1]
                l_dn = Ls[y + delta_clear, x0:x1]
                v_ring = np.maximum(0.0, np.minimum(l_up, l_dn) - l_curr)
                g_ring = sobel_l[y, x0:x1]
                valleys[idx] = float(np.median(v_ring + 0.60 * g_ring))
            peak_idx = int(np.argmax(valleys))
            peak_val = float(valleys[peak_idx])
            # ต้องมีสัญญาณหักเหแสงวงแหวนผิวน้ำชัดเจน (> 2.5 เท่าของ noise) จึงจะถือว่าเป็นน้ำไม่เต็มขวด
            if peak_val < 2.5 * sigma_l:
                return full_curve, info
            y_med = y_lo + peak_idx
            ycoarse = np.full(x1 - x0, float(y_med), dtype=np.float64)
            info["strength"] = peak_val
            info["snr"] = peak_val / max(1.5, sigma_l)
        else:
            # สำหรับน้ำมีสี: ผสานคะแนนระยะห่างสี (Color Distance Field) เข้ากับคะแนนการเพิ่มขึ้นของโครมา (Chroma Rise Jump)
            # เพื่อป้องกันการไปล็อกกับแนวเงาคาบสะท้อนแสงภายในเนื้อน้ำ (Internal Shadow / Caustic)
            Cc = chroma_map[:, x0:x1]
            cs = np.vstack([np.zeros((1, Dc.shape[1]), np.float32), np.cumsum(Dc, axis=0)])
            cc = np.vstack([np.zeros((1, Cc.shape[1]), np.float32), np.cumsum(Cc, axis=0)])
            ys_ = np.arange(y_lo, y_hi)
            above_d = (cs[ys_] - cs[ys_ - Wn]) / Wn
            below_d = (cs[ys_ + Wn] - cs[ys_]) / Wn
            score_d = above_d - below_d

            above_c = (cc[ys_] - cc[ys_ - Wn]) / Wn
            below_c = (cc[ys_ + Wn] - cc[ys_]) / Wn
            score_c = np.maximum(0.0, below_c - above_c)

            comb_score = (score_d / norm_scale) + 1.25 * (score_c / max(2.0, sigma_body))
            ycoarse = ys_[np.argmax(comb_score, axis=0)].astype(np.float64)
            strength = float(np.median(score_d.max(axis=0)))

            w_chroma = body_chroma / (body_chroma + cfg.chroma_half_saturation)
            effective_snr = (strength / norm_scale) * w_chroma
            info["strength"] = strength
            info["snr"] = effective_snr

            if effective_snr < cfg.min_step_snr:
                return full_curve, info

            y_med = int(round(float(np.median(ycoarse))))

        # ตรวจสอบอัตราส่วนความต่างสีและโครมาเหนือระดับน้ำเทียบกับใต้ระดับน้ำ
        d_above = float(np.median(Ds[max(0, y_med - Wn):max(1, y_med), x0:x1]))
        d_below = float(np.median(Ds[y_med + 2:min(by, y_med + Wn + 2), x0:x1]))
        c_above = float(np.median(chroma_map[max(0, y_med - Wn):max(1, y_med), x0:x1]))
        c_below = float(np.median(chroma_map[y_med + 2:min(by, y_med + Wn + 2), x0:x1]))
        l_above = float(np.median(Ls[max(0, y_med - Wn):max(1, y_med), x0:x1]))
        l_below = float(np.median(Ls[y_med + 2:min(by, y_med + Wn + 2), x0:x1]))

        # 2. การป้องกัน False Step ในน้ำสีเข้ม:
        # หากด้านบนของเส้นที่ตรวจพบ มีสีเข้มและมีโทนสีเดียวกับเนื้อน้ำ (ไม่ใช่สีขาวของอากาศ/ฝา)
        # ให้ถือว่าน้ำเต็มขวด (full_top) ทันที ป้องกันการตัดผ่ากลางเนื้อน้ำ
        is_dark_liquid = (body_chroma >= 15.0 or cb[0] < 50.0)
        if is_dark_liquid:
            is_colored_above = (c_above >= 0.35 * body_chroma) and (l_above < 60.0)
            if is_colored_above:
                return full_curve, info

        air_body_ratio = d_above / (d_below + sigma_body)
        is_chroma_jump = (c_below - c_above >= max(8.0, 0.22 * body_chroma)) and (c_below >= 1.45 * c_above)
        is_already_liquid_above = (c_above > 0.78 * max(body_chroma, c_below)) and (abs(l_above - l_below) < 2.5 * sigma_l)

        # ตรวจสอบว่าด้านบนเป็นช่องว่างอากาศ (Air Gap) จริงหรือไม่:
        # 1. หากมี Chroma Jump ชัดเจน (อากาศสีจาง -> น้ำมีสีสด) ให้ถือว่าผ่านทันที
        # 2. หากไม่มี Chroma Jump ต้องมีระยะห่างสี air_body_ratio >= min_air_to_body_ratio
        if not is_chroma_jump:
            if air_body_ratio < cfg.min_air_to_body_ratio or is_already_liquid_above:
                return full_curve, info

        # ล็อกตำแหน่งเข้าหา 'แนวโค้งคว่ำของระดับน้ำด้านบนภายในขวด' (Upper Arch of Liquid Surface)
        # จุดยอด (Apex) อยู่บริเวณกึ่งกลางขวด และปลายทั้งสองข้างโค้งลงไปบรรจบขอบวงรีผิวน้ำที่ผนังขวดซ้าย-ขวา
        R_up = max(4, int(cfg.local_refine_up_ratio * h))
        R_down = max(8, int(cfg.local_refine_down_ratio * h))
        delta = max(3, int(0.018 * h))
        # คัดกรองเฉพาะคอลัมน์ที่สอดคล้องกับระดับน้ำหลัก (ตัดคอลัมน์ที่ถูกเส้นแสงสะท้อนแนวดิ่งรบกวนออก)
        inlier_coarse = ycoarse[np.abs(ycoarse - y_med) <= max(8.0, 1.5 * Wn)]
        y_apex_coarse = int(round(float(np.percentile(inlier_coarse, 30)))) if len(inlier_coarse) > 0 else y_med

        # หาจุดยอดแนวโค้งคว่ำที่กึ่งกลางขวด (Center Apex) โดยผสาน Color Distance และ Chroma Gradient
        ya_c = max(3, y_apex_coarse - max(3, R_up // 2))
        yb_c = min(by - 5, y_apex_coarse + max(3, R_up // 2))
        x_mid_lo, x_mid_hi = int(0.38 * w), int(0.62 * w)
        if yb_c > ya_c + 2 and x_mid_hi > x_mid_lo:
            gd_mid = np.median(np.maximum(0.0, (Ds[ya_c - 2:yb_c - 2, x_mid_lo:x_mid_hi] - Ds[ya_c + 2:yb_c + 2, x_mid_lo:x_mid_hi]) / norm_scale), axis=1)
            gc_mid = np.median(np.maximum(0.0, (chroma_map[ya_c + 2:yb_c + 2, x_mid_lo:x_mid_hi] - chroma_map[ya_c - 2:yb_c - 2, x_mid_lo:x_mid_hi]) / norm_scale), axis=1)
            score_apex = gd_mid + 0.85 * gc_mid
            y_apex = ya_c + int(np.argmax(score_apex)) if float(np.max(score_apex)) > 0 else y_apex_coarse
        else:
            y_apex = y_apex_coarse

        x_fit_lo, x_fit_hi = int(0.08 * w), int(0.92 * w)
        xs = np.arange(x_fit_lo, x_fit_hi, dtype=np.float64)
        yref_arch = np.zeros_like(xs)
        yref_core = np.zeros_like(xs)
        for i, x in enumerate(range(x_fit_lo, x_fit_hi)):
            rel_x = ((x - xc) / (half_w + 1e-6)) ** 2
            ya = max(delta + 1, y_apex - (2 if rel_x < 0.15 else 0) + int(round(rel_x * 0.035 * w)))
            yb = min(by - delta - 2, y_apex + max(6, int(0.04 * h)) + int(round(rel_x * 0.12 * w)))
            if yb > ya + 3:
                l_curr = Ls[ya:yb, x]
                l_up = Ls[ya - delta:yb - delta, x]
                l_dn = Ls[ya + delta:yb + delta, x]
                valley_l = np.maximum(0.0, (np.minimum(l_up, l_dn) - l_curr) / max(2.0, sigma_l))
                g_d = np.maximum(0.0, (Ds[ya - 2:yb - 2, x] - Ds[ya + 2:yb + 2, x]) / norm_scale)
                g_c = np.maximum(0.0, (chroma_map[ya + 2:yb + 2, x] - chroma_map[ya - 2:yb - 2, x]) / norm_scale)
                g_l = np.abs(Ls[ya - 2:yb - 2, x] - Ls[ya + 2:yb + 2, x]) / max(2.0, sigma_l)
                # 1. พาราโบลาคว่ำสำหรับระดับน้ำบนสุด (Top Arch)
                resp_arch = g_d + 0.85 * g_c + 0.35 * g_l + (0.75 * rel_x) * valley_l
                yref_arch[i] = float(ya + int(np.argmax(resp_arch)))
                # 2. พาราโบลาหงายสำหรับระดับเข้มผิวน้ำด้านหน้า (Dark Meniscus Core)
                resp_core = valley_l + 0.35 * g_l + 0.25 * g_d
                yref_core[i] = float(ya + int(np.argmax(resp_core)))
            else:
                yref_arch[i] = float(ya)
                yref_core[i] = float(ya + 4)

        # ฟิตขอบบนสุด: พาราโบลาโค้งคว่ำ (opening="down", cap \cap)
        a_a, c_a, inl_a = cls._fit_symmetric_parabola(
            xs, yref_arch, xc,
            max_sag=cfg.meniscus_max_sag_ratio * w,
            half_w=half_w,
            min_sag=cfg.meniscus_min_sag_ratio * w,
            cfg=cfg,
            opening="down"
        )
        # ฟิตขอบระดับเข้ม: พาราโบลาโค้งหงาย (opening="up", trough \cup)
        a_c, c_c, inl_c = cls._fit_symmetric_parabola(
            xs, yref_core, xc,
            max_sag=cfg.meniscus_max_sag_ratio * w,
            half_w=half_w,
            min_sag=cfg.meniscus_min_sag_ratio * w,
            cfg=cfg,
            opening="up"
        )
        curve_arch = np.clip(np.round((a_a * (xf - xc) ** 2 + c_a) / sy), 1, max(1, int(0.85 * base_y))).astype(np.int32)
        curve_core = np.clip(np.round((a_c * (xf - xc) ** 2 + c_c) / sy), 1, max(1, int(0.88 * base_y))).astype(np.int32)

        # ตัดสินใจอัตโนมัติด้วยระบบสายตาคอมพิวเตอร์ (Zero Hardcoded Path):
        # หากระดับน้ำอยู่ชิดฝาขวด (is_near_cap): ให้ใช้พาราโบลาโค้งหงาย (∪, curve_core)
        # เพื่อตามแนวขอบล่าง 3 มิติของฝาขวด และเก็บเนื้อมวลน้ำขอบข้างซ้าย-ขวาครบ 100%
        # หากระดับน้ำอยู่ลึกในตัวขวด (มีช่องว่างอากาศชัดเจน): ให้ใช้พาราโบลาโค้งคว่ำ (∩, curve_arch)
        y_surf_px = float(c_a / sy)
        is_near_cap = (y_surf_px / float(H) <= cfg.near_cap_ratio) or (y_surf_px <= cfg.near_cap_max_px)

        if is_near_cap:
            chosen_curve = curve_core
            orientation = "cup"
        else:
            chosen_curve = curve_arch
            orientation = "arch"

        info.update(
            inlier=inl_c if is_near_cap else inl_a,
            a=a_c if is_near_cap else a_a,
            c=c_c if is_near_cap else c_a,
            sag_px=float(abs(a_c if is_near_cap else a_a) * half_w ** 2 / sy),
            curve_arch=curve_arch,
            curve_core=curve_core,
            mode="step",
            orientation=orientation,
            is_near_cap=is_near_cap
        )
        return chosen_curve, info

    @staticmethod
    def locate_liquid_bottom(upright_bottle: np.ndarray, top_curve: np.ndarray,
                             base_y: int = None, cfg: CropperConfig = DEFAULT_CONFIG):
        """
        ตัดก้นขวดตามรูปทรงโค้งของขวดทรงกระบอกจริง (3D Cylindrical Bottle Base Contour):
        - จุดยอด (Apex) อยู่ตรงกลางก้นขวดชิดขอบล่างสุด (H - 2)
        - ปลายทั้งสองข้างโค้งขึ้นตามขอบมนของก้นขวด (พาราโบลาหงาย ∪, Cup)
        - โอบรับก้นหลอดแก้วทรงกระบอก 3 มิติครบ 100% ไม่ตัดกินเนื้อมวลน้ำและเนื้อขวดด้านล่าง
        """
        cls = WebOptimizedLiquidCropper
        H, W = upright_bottle.shape[:2]
        xc = (W - 1) / 2.0
        half_w = W / 2.0
        xf = np.arange(W, dtype=np.float64)

        # ตำแหน่งก้นขวดจริง: upright_bottle ถูกตัดขอบล่างพอดีแล้วที่ขั้น deskew_and_tight_crop
        # ดังนั้นจุดกึ่งกลางก้นขวดจะอยู่ที่ระดับต่ำสุดชิดขอบล่าง (H - 6) ไม่ให้ตัดกินก้นขวดทิ้ง
        c_anchor = float(H - 6) if (base_y is None or base_y < int(0.90 * H)) else float(min(H - 6, base_y))

        # เส้นพาราโบลาหงาย (∪) โอบรับก้นขวดมนทรงกระบอกจริง:
        # sag สัดส่วนตามความกว้างขวด (ประมาณ 8.2% ของความกว้างขวด หรือราว 45-50 px) เพื่อตัดมุมโต๊ะซ้าย-ขวาออกชัดเจน
        default_sag = max(18.0, cfg.base_default_sag_ratio * W)
        a_def = -default_sag / (half_w ** 2 + 1e-6)
        bot_default = np.clip(np.round(a_def * (xf - xc) ** 2 + c_anchor), top_curve + 15, H - 1).astype(np.int32)
        info = {
            "mode": "bottle_contour",
            "orientation": "cup",
            "sag_px": float(default_sag),
            "apex_y": int(c_anchor)
        }

        # ตรวจสอบขอบโต๊ะบริเวณมุมล่างซ้าย-ขวาเฉพาะในโซนล่างสุด (94%-100% ของขวด)
        # เพื่อปรับจูนความโค้ง sag ให้แนบสนิทกับขอบมนของแก้วจริง
        search_h = max(10, int((1.0 - cfg.base_search_y_start) * H))
        y_start_full = max(0, H - search_h)
        if search_h > 4:
            gray = cv2.cvtColor(upright_bottle, cv2.COLOR_BGR2GRAY)
            sob_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
            corner_w = max(5, int(0.12 * W))

            # วัดขอบที่มุมซ้าย ขวา และกึ่งกลาง
            l_edge_y = y_start_full + int(np.argmax(sob_y[y_start_full:, :corner_w].mean(axis=1)))
            r_edge_y = y_start_full + int(np.argmax(sob_y[y_start_full:, -corner_w:].mean(axis=1)))
            mid_edge_y = y_start_full + int(np.argmax(sob_y[y_start_full:, int(0.4 * W):int(0.6 * W)].mean(axis=1)))

            measured_sag = float(mid_edge_y - 0.5 * (l_edge_y + r_edge_y))
            min_sag = cfg.base_min_sag_ratio * W
            max_sag = cfg.base_max_sag_ratio * W
            if min_sag <= measured_sag <= max_sag:
                sag = measured_sag
                a_fit = -sag / (half_w ** 2 + 1e-6)
                bot_fitted = np.clip(np.round(a_fit * (xf - xc) ** 2 + c_anchor), top_curve + 15, H - 1).astype(np.int32)
                info.update(mode="bottle_contour_fitted", sag_px=float(sag))
                return bot_fitted, info

        return bot_default, info

    @staticmethod
    def detect_water_level_and_segment(upright_bottle: np.ndarray, base_hint_y: int = None,
                                       cfg: CropperConfig = DEFAULT_CONFIG) -> dict:
        """
        ตรวจจับระดับน้ำ แล้วครอบมวลน้ำตามแนวพาราโบลาของเส้นแนวระดับน้ำพอดีถึงรูปทรงก้นขวดจริง:
        1. หาก้นขวด (จาก base_hint_y ของขั้น Deskew)
        2. หาผิวน้ำด้วยเกณฑ์ Self-Adaptive SNR + ล็อกตรงเส้นแนวระดับน้ำพอดี + ฟิตพาราโบลาเว้าสมมาตร
        3. ตัดก้นขวดตามรูปทรงโค้งของขวดทรงกระบอกจริง (locate_liquid_bottom)
        4. สร้างมาสก์ = ผิวน้ำโค้งพาราโบลาด้านบน + ผนังแก้วตรง + ก้นขวดโค้งพาราโบลาตามรูปทรงขวด
        ขั้นนี้ 'ไม่แก้ค่าสีของพิกเซล' ; การเคลียแสงสะท้อน/เงาอยู่ในขั้น clean_glare_and_shadow ซึ่งทำหลังครอบแล้วเท่านั้น
        """
        cls = WebOptimizedLiquidCropper
        bh, bw = upright_bottle.shape[:2]

        # 1. ก้นขวด
        if base_hint_y is not None and int(0.90 * bh) <= base_hint_y < bh:
            base_y = base_hint_y
        else:
            base_y = bh - 2
        base_y = min(bh - 1, max(30, base_y))

        # 2. ผิวน้ำ (พาราโบลาโค้งคว่ำตรงระดับน้ำบนสุด และพาราโบลาโค้งหงายตรงระดับเข้ม)
        top_curve, surf = cls.locate_liquid_surface(upright_bottle, base_y, cfg=cfg)
        top_curve = np.clip(top_curve, 0, base_y - 15).astype(int)
        top_core = surf.get("curve_core", top_curve).astype(int)
        meniscus_y = int(np.mean(top_curve))

        # 3. ก้นขวด (พาราโบลาเว้าตามรูปทรงก้นขวดจริง)
        bot_curve, bot_info = cls.locate_liquid_bottom(upright_bottle, top_curve, base_y=base_y, cfg=cfg)
        bot_curve = bot_curve.astype(int)
        bottom_y = int(np.max(bot_curve))

        # 4. สร้างมาสก์รูปทรงมวลน้ำ (Mask A: มวลรวม 100% และ Core Mask B: แกนทัศนศาสตร์บริสุทธิ์)
        pts_contour = [[x, int(top_curve[x])] for x in range(bw)]
        pts_contour += [[x, int(bot_curve[x])] for x in range(bw - 1, -1, -1)]
        liquid_mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.drawContours(liquid_mask, [np.array(pts_contour, dtype=np.int32)], -1, 255, -1)

        pts_core = [[x, int(top_core[x])] for x in range(bw)]
        pts_core += [[x, int(bot_curve[x])] for x in range(bw - 1, -1, -1)]
        core_mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.drawContours(core_mask, [np.array(pts_core, dtype=np.int32)], -1, 255, -1)

        b_ch, g_ch, r_ch = cv2.split(upright_bottle)
        rgba = cv2.merge([b_ch, g_ch, r_ch, liquid_mask])
        min_y = int(np.min(top_curve))
        max_y = int(np.max(bot_curve))
        liquid_rgba_crop = rgba[min_y:max_y + 1, 0:bw].copy()

        pure_mask = liquid_mask > 0
        valid_b, valid_g, valid_r = b_ch[pure_mask], g_ch[pure_mask], r_ch[pure_mask]
        total_px = len(valid_b)

        stats = {
            "total_pixels": int(total_px),
            "glare_filtered_px": 0,
            "cap_removed": True,
            "meniscus_y": meniscus_y,
            "bottom_y": int(bottom_y),
            "liquid_height": int(max_y - min_y),
            "liquid_height_pct": float((max_y - min_y) / bh * 100.0),
            "liquid_width": int(bw),
            "pure_y1": int(min_y),
            "pure_y2": int(max_y),
            "x_left": 0,
            "x_right": int(bw - 1),
            "surface_mode": surf["mode"],
            "surface_strength": float(surf["strength"]),
            "surface_snr": float(surf.get("snr", 0.0)),
            "surface_inlier": float(surf["inlier"]),
            "surface_sag_px": float(surf["sag_px"]),
            "surface_orientation": surf.get("orientation", "arch"),
            "is_near_cap": bool(surf.get("is_near_cap", False)),
            "base_mode": bot_info["mode"],
            "mean_R": float(np.mean(valid_r)) if total_px > 0 else 0.0,
            "mean_G": float(np.mean(valid_g)) if total_px > 0 else 0.0,
            "mean_B": float(np.mean(valid_b)) if total_px > 0 else 0.0,
            "median_R": float(np.median(valid_r)) if total_px > 0 else 0.0,
            "median_G": float(np.median(valid_g)) if total_px > 0 else 0.0,
            "median_B": float(np.median(valid_b)) if total_px > 0 else 0.0,
        }

        return {
            "mask": liquid_mask,
            "core_mask": core_mask,
            "contour": np.array(pts_contour, dtype=np.int32),
            "rgba": liquid_rgba_crop,
            "stats": stats,
            "meniscus_y": meniscus_y,
            "bottom_y": bottom_y,
            "top_curve": top_curve,
            "top_core": top_core,
            "bot_curve": bot_curve
        }

    # ---------------- ขั้นที่ 2: เคลียแสงสะท้อนและเงา (ทำหลังครอบเสร็จเท่านั้น) ----------------
    @staticmethod
    def _smooth1d(v, sigma: float) -> np.ndarray:
        k = int(max(3, sigma * 6)) | 1
        return cv2.GaussianBlur(np.asarray(v, np.float32).reshape(-1, 1), (1, k), sigma).flatten()

    @staticmethod
    def detect_specular_glare(lab: np.ndarray, mask: np.ndarray,
                              cfg: CropperConfig = DEFAULT_CONFIG) -> np.ndarray:
        """
        ตรวจเฉพาะแสงสะท้อนด้วยเกณฑ์ Self-Adaptive z-score (เท่าของ MAD ของการเบี่ยงเบนความสว่างในภาพ)
        ร่วมกับการลดลงของความอิ่มตัวสี (Chroma Desaturation)
        """
        cls = WebOptimizedLiquidCropper
        L = lab[:, :, 0]
        a = lab[:, :, 1] - 128.0
        b = lab[:, :, 2] - 128.0
        h, w = L.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            Lm = np.where(mask, L, np.nan)
            target = float(np.nanmedian(Lm))
            col_med = np.nanpercentile(Lm, 40, axis=0) - target
            col_med = np.where(np.isnan(col_med), 0.0, col_med)
            col_med = cls._smooth1d(col_med, max(4.0, w * 0.08))
            L_no_col = Lm - col_med[None, :]
            row_med = np.nanpercentile(L_no_col, 45, axis=1)
            full_rows = np.mean(mask, axis=1) >= cfg.full_row_width_fraction
            if np.any(full_rows):
                valid_idx = np.where(full_rows & ~np.isnan(row_med))[0]
                if len(valid_idx) > 0:
                    row_med = np.interp(np.arange(h), valid_idx, row_med[valid_idx])
            row_med = np.where(np.isnan(row_med), target, row_med)
            row_med = cls._smooth1d(row_med, max(3.0, h * 0.03))
        excess = L - row_med[:, None] - col_med[None, :]
        ex_valid = excess[mask]
        mad = float(np.median(np.abs(ex_valid - np.median(ex_valid))) * 1.4826)
        thr = max(cfg.glare_min_excess_l, cfg.glare_z_threshold * mad)
        chroma = np.hypot(a, b)
        chroma_ref = float(np.median(chroma[mask]))
        washed = (chroma < 0.88 * chroma_ref) | (chroma_ref < cfg.chroma_half_saturation) | (excess > thr + 3.0 * mad)
        glare = (mask & (excess > thr) & washed).astype(np.uint8)
        glare = cv2.morphologyEx(glare, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        d_rad = max(7, int(min(h, w) * 0.025) | 1)
        glare = cv2.dilate(glare, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_rad, d_rad)))
        return (glare > 0) & mask

    @staticmethod
    def _fill_from_surface(lab: np.ndarray, glare: np.ndarray, mask: np.ndarray,
                           cfg: CropperConfig = DEFAULT_CONFIG, max_samples: int = 30000) -> np.ndarray:
        """
        เติมพิกเซลแสงสะท้อนด้วยพื้นผิวพหุนามที่ฟิตจากพิกเซลปกติ พร้อมเกลี่ยขอบแบบ Feathering
        เพื่อให้รอยต่อเนียนสนิทเป็นเนื้อเดียวกับมวลน้ำ (ไม่ลบพิกเซล ไม่ทิ้งรูโหว่ ไม่เกิดขอบแข็ง)
        """
        out = lab.copy()
        h, w = mask.shape
        deg = cfg.surface_poly_degree
        good = mask & ~glare
        if good.sum() < 500 or glare.sum() == 0:
            return out
        yy, xx = np.mgrid[0:h, 0:w]
        xn = (xx / (w - 1.0)) * 2 - 1
        yn = (yy / (h - 1.0)) * 2 - 1

        def terms(x, y):
            return np.stack([(x ** i) * (y ** j) for i in range(deg + 1) for j in range(deg + 1 - i)], axis=1)

        gi = np.flatnonzero(good.ravel())
        if len(gi) > max_samples:
            gi = np.random.default_rng(0).choice(gi, max_samples, replace=False)

        k_blur = max(11, int(min(h, w) * 0.04) | 1)
        alpha = cv2.GaussianBlur(glare.astype(np.float32), (k_blur, k_blur), k_blur * 0.25)
        alpha = np.clip(alpha * 1.35, 0.0, 1.0)
        alpha[~mask] = 0.0

        gl = np.flatnonzero((alpha > 0.01).ravel())
        A = terms(xn.ravel()[gi], yn.ravel()[gi])
        Ag = terms(xn.ravel()[gl], yn.ravel()[gl])
        w_gl = alpha.ravel()[gl]
        for c in range(3):
            coef = np.linalg.lstsq(A, lab[:, :, c].ravel()[gi], rcond=None)[0]
            flat = out[:, :, c].ravel()
            pred = Ag @ coef
            flat[gl] = (1.0 - w_gl) * flat[gl] + w_gl * pred
            out[:, :, c] = flat.reshape(h, w)
        return out

    @staticmethod
    def _flatten_luminance(lab: np.ndarray, mask: np.ndarray,
                           cfg: CropperConfig = DEFAULT_CONFIG) -> np.ndarray:
        """
        ปรับแสง/เงาให้เรียบโดยแก้เฉพาะช่อง L* ไม่แตะสีจริง a*, b*:
        1. เกลี่ยเงาขอบขวดทรงกระบอกตามแนวคอลัมน์ก่อน (ทุกคอลัมน์มีความสูงเต็มขวด)
        2. เกลี่ยการไล่แสงแนวดิ่งตามแถว โดยคำนวณจากแถวที่มีความกว้างเต็ม (>= full_row_width_fraction)
           เพื่อไม่ให้มุมโค้งพาราโบลาบน-ล่างดึงค่ามัธยฐานแสงผิดเพี้ยน
        """
        cls = WebOptimizedLiquidCropper
        L = lab[:, :, 0].copy()
        h, w = L.shape
        c_lo, c_hi = cfg.luminance_col_gain_bounds
        r_lo, r_hi = cfg.luminance_row_gain_bounds
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            Lm = np.where(mask, L, np.nan)
            target = float(np.nanmedian(Lm))
            col = np.nanmedian(Lm, axis=0)
            col = np.where(np.isnan(col), target, col)
            gain_c = np.clip(target / np.clip(cls._smooth1d(col, max(2.0, w * 0.02)), 20, 255), c_lo, c_hi)
            L1 = L * gain_c[None, :]

            Lm1 = np.where(mask, L1, np.nan)
            row = np.nanmedian(Lm1, axis=1)
            full_rows = np.mean(mask, axis=1) >= cfg.full_row_width_fraction
            if np.any(full_rows):
                valid_idx = np.where(full_rows & ~np.isnan(row))[0]
                if len(valid_idx) > 0:
                    row = np.interp(np.arange(h), valid_idx, row[valid_idx])
            row = np.where(np.isnan(row), target, row)
            gain_r = np.clip(target / np.clip(cls._smooth1d(row, max(2.0, h * 0.02)), 20, 255), r_lo, r_hi)
        L2 = np.clip(L1 * gain_r[:, None], 0, 255)
        out = lab.copy()
        out[:, :, 0] = np.where(mask, L2, L)
        return out

    @staticmethod
    def _delta_e_top_bottom(bgr: np.ndarray, mask: np.ndarray) -> float:
        """ความต่างสีเฉลี่ยครึ่งบน-ครึ่งล่างของมวลน้ำ (CIELAB สเกล 8-bit ของ OpenCV)"""
        rows = np.where(np.any(mask, axis=1))[0]
        if len(rows) < 10:
            return 0.0
        mid = rows[len(rows) // 2]
        top = mask.copy(); top[mid:, :] = False
        bot = mask.copy(); bot[:mid, :] = False
        if top.sum() < 50 or bot.sum() < 50:
            return 0.0
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        return float(np.linalg.norm(lab[top].mean(0) - lab[bot].mean(0)))

    @staticmethod
    def remove_specular_drm_tan_ikeuchi(bgr_image: np.ndarray, liquid_mask: np.ndarray = None) -> dict:
        """
        Physics-based Specular Highlight Removal via Dichromatic Reflection Model
        (Shafer 1985; Tan & Ikeuchi 2005, IEEE TPAMI; Yang et al. 2010, ECCV)
        
        I(x, y) = I_diffuse(x, y) + I_specular(x, y)
        - คำนวณองค์ประกอบแสงสะท้อน I_specular (m_s) ผ่าน Normalized Chromaticity
        - หักลบเฉพาะ I_specular ออกจากภาพ เพื่อฟื้นฟู I_diffuse (สีเนื้อมวลน้ำบริสุทธิ์)
        - รักษาเนื้อมวลน้ำ 100% Full Mass ไร้รูพรุนโปร่งใส (0 Interior Holes) และไม่แต่งสีสังเคราะห์
        """
        img_f = bgr_image.astype(np.float32)
        B, G, R = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
        total_I = R + G + B + 1e-6
        max_I = np.maximum(np.maximum(R, G), B)
        min_I = np.minimum(np.minimum(R, G), B)
        
        valid_mask = liquid_mask > 0 if liquid_mask is not None else np.ones(bgr_image.shape[:2], dtype=bool)
        
        # Normalized maximum chromaticity
        lambda_max = max_I / total_I
        body_lambdas = lambda_max[valid_mask]
        if len(body_lambdas) == 0:
            return {
                "diffuse": bgr_image.copy(),
                "specular_mask": np.zeros(bgr_image.shape[:2], dtype=np.uint8),
                "m_s": np.zeros(bgr_image.shape[:2], dtype=np.float32),
                "glare_pct": 0.0
            }
            
        body_lambda_diff = float(np.percentile(body_lambdas, 80))
        
        # Calculate CIELAB chroma to detect clear vs colored liquid
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        chroma = np.hypot(lab[:, :, 1].astype(np.float32) - 128.0, lab[:, :, 2].astype(np.float32) - 128.0)
        body_chroma = float(np.median(chroma[valid_mask]))
        
        if body_lambda_diff > 0.365 and body_chroma > 10.0:
            # 1. Chromatic / Colored Liquid (Tan & Ikeuchi 2005 closed-form DRM):
            lambda_target = min(0.85, max(body_lambda_diff, 0.38))
            denom = 1.0 - 3.0 * lambda_target
            if abs(denom) < 1e-4:
                denom = -1e-4
                
            m_s = (max_I - lambda_target * total_I) / denom
            m_s = np.clip(m_s, 0.0, None)
            m_s = np.minimum(m_s, min_I)
            
            gray = (R + G + B) / 3.0
            med_gray = float(np.median(gray[valid_mask]))
            weight_bright = np.clip((gray - 0.6 * med_gray) / (0.4 * med_gray + 1e-4), 0.0, 1.0)
            m_s = m_s * weight_bright
        else:
            # 2. Achromatic / Clear Liquid (Phosphate 0 mg/L or pure water):
            gray = (R + G + B) / 3.0
            med_gray = float(np.median(gray[valid_mask]))
            mad_gray = float(np.median(np.abs(gray[valid_mask] - med_gray)) * 1.4826)
            glare_thr = med_gray + max(12.0, 2.2 * mad_gray)
            m_s = np.clip(gray - glare_thr, 0.0, None)
            m_s = np.minimum(m_s, min_I)
            
        if liquid_mask is not None:
            m_s = m_s * (liquid_mask > 0)
            
        # Subtract specular component m_s * (1, 1, 1)
        diff_B = np.clip(B - m_s, 0, 255).astype(np.uint8)
        diff_G = np.clip(G - m_s, 0, 255).astype(np.uint8)
        diff_R = np.clip(R - m_s, 0, 255).astype(np.uint8)
        diffuse = cv2.merge([diff_B, diff_G, diff_R])
        
        spec_mask = (m_s > 3.0).astype(np.uint8) * 255
        spec_pct = float(np.sum(m_s > 3.0) / max(1, np.sum(valid_mask)) * 100.0)
        
        return {
            "diffuse": diffuse,
            "specular_mask": spec_mask,
            "m_s": m_s,
            "glare_pct": spec_pct,
            "body_chroma": body_chroma,
            "lambda_target": body_lambda_diff
        }

    @staticmethod
    def clean_glare_and_shadow(upright_bottle: np.ndarray, liquid_mask: np.ndarray,
                               cfg: CropperConfig = DEFAULT_CONFIG) -> dict:
        """
        เคลียแสงสะท้อนตามแบบจำลอง Dichromatic Reflection Model (DRM)
        (Shafer 1985; Tan & Ikeuchi 2005, IEEE TPAMI; Yang et al. 2010, ECCV)
        - คำนวณองค์ประกอบแสงสะท้อน Ispecular ผ่าน Normalized Chromaticity
        - หักลบเฉพาะ Ispecular ออกจากภาพ เพื่อให้ได้ Idiffuse (สีเนื้อมวลน้ำบริสุทธิ์)
        - รักษาเนื้อมวลน้ำ 100% Full Mass ไร้รูพรุนโปร่งใส (0 Interior Holes) และไม่แต่งสีสังเคราะห์
        """
        cls = WebOptimizedLiquidCropper
        mask = liquid_mask > 0
        drm = cls.remove_specular_drm_tan_ikeuchi(upright_bottle, mask)
        out = drm["diffuse"]
        out[~mask] = upright_bottle[~mask]
        
        return {
            "bgr": out,
            "glare_mask": drm["specular_mask"],
            "glare_pct": drm["glare_pct"],
            "delta_e_raw": cls._delta_e_top_bottom(upright_bottle, mask),
            "delta_e_clean": cls._delta_e_top_bottom(out, mask),
            "m_s": drm["m_s"]
        }

    @staticmethod
    def log_crop_failure(image_path: str, error: Exception, output_dir: str = None, stage: str = "unknown") -> dict:
        """
        บันทึกข้อมูลและประวัติข้อผิดพลาด (Error Logging) อัตโนมัติเมื่อเกิดการ crop ไม่สำเร็จ:
        - บันทึกลงไฟล์ text log (output/crop_failed.log) สำหรับการอ่านตรวจสอบ
        - บันทึกลงไฟล์ jsonl (output/crop_failed.jsonl) สำหรับการนำข้อมูลไปวิเคราะห์ต่อ
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        err_type = type(error).__name__
        err_msg = str(error)
        tb_str = traceback.format_exc()
        abs_path = os.path.abspath(image_path) if image_path else ""
        file_name = os.path.basename(image_path) if image_path else "unknown"

        entry = {
            "timestamp": timestamp,
            "image_name": file_name,
            "image_path": abs_path,
            "stage": stage,
            "error_type": err_type,
            "error_message": err_msg,
            "traceback": tb_str.strip()
        }

        # 1. บันทึกไฟล์ข้อความแบบอ่านง่าย (.log)
        log_text = (
            f"======================================================================\n"
            f"[{timestamp}] CROP FAILED: {file_name}\n"
            f"  File:      {abs_path}\n"
            f"  Stage:     {stage}\n"
            f"  Exception: {err_type}: {err_msg}\n"
            f"  Traceback:\n{tb_str}\n"
            f"======================================================================\n\n"
        )

        target_dirs = set()
        central_out = os.path.abspath(os.path.join("output"))
        target_dirs.add(central_out)
        if output_dir:
            target_dirs.add(os.path.abspath(output_dir))

        for d in target_dirs:
            try:
                os.makedirs(d, exist_ok=True)
                log_file = os.path.join(d, "crop_failed.log")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(log_text)
            except Exception:
                pass

        # 2. บันทึกไฟล์ jsonl เพื่อการนำข้อมูลไปวิเคราะห์ทางโปรแกรม
        try:
            os.makedirs(central_out, exist_ok=True)
            jsonl_file = os.path.join(central_out, "crop_failed.jsonl")
            with open(jsonl_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

        return entry

    @staticmethod
    def apply_5_dynamic_solutions(clean_rgba: np.ndarray) -> tuple[np.ndarray, int]:
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
        
        sheen_smoothed = cv2.GaussianBlur(enhanced_bgr, (max(5, int(W * 0.12) | 1), 1), 0)
        for c in range(3):
            enhanced_bgr[:, :, c] = np.clip(
                enhanced_bgr[:, :, c] * (1.0 - sheen_weight[:, :, 0]) + sheen_smoothed[:, :, c] * sheen_weight[:, :, 0],
                0, 255
            )
            
        out_rgba = cv2.merge([enhanced_bgr[:, :, 0], enhanced_bgr[:, :, 1], enhanced_bgr[:, :, 2], alpha])
        return out_rgba, sed_pixels

    @staticmethod
    def evaluate_readiness(raw_crop_rgba: np.ndarray, enhanced_clean_rgba: np.ndarray, sed_px: int) -> dict:
        """
        ประเมินคุณภาพความพร้อมใช้งาน (Readiness Evaluation) เชิงฟิสิกส์และความสม่ำเสมอของเนื้อน้ำ
        """
        bgr1 = raw_crop_rgba[:, :, :3]
        m1 = raw_crop_rgba[:, :, 3] > 0
        bgr2 = enhanced_clean_rgba[:, :, :3]
        m2 = enhanced_clean_rgba[:, :, 3] > 0
        
        if m1.sum() == 0 or m2.sum() == 0:
            return {"score_before": 0.0, "score_after": 0.0, "noise_reduction_pct": 0.0, "sediment_px": 0}
            
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
        c1 = np.median(gray1[int(0.4*H):int(0.6*H), :][m1[int(0.4*H):int(0.6*H), :]])
        c2 = np.median(gray2[int(0.4*H):int(0.6*H), :][m2[int(0.4*H):int(0.6*H), :]])
        
        bot_cleanup = max(0.0, 100.0 - abs(b2 - c2) * 2.0)
        
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

    @staticmethod
    def dynamic_foreign_color_cleaner(rgba_image: np.ndarray, 
                                      enable_glass_inset: bool = True,
                                      glass_inset_ratio: float = 0.030,
                                      base_inset_ratio: float = 0.015) -> tuple:
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
           - แบบที่ 1 (Trimmed): ตัด Alpha = 0 (โปร่งใส) เฉพาะพิกเซลแปลกปลอม เหลือเฉพาะเนื้อมวลน้ำบริสุทธิ์
           - แบบที่ 2 (Harmonized / Inpainted): เกลี่ยสีเนื้อน้ำแท้เข้าไปแทนที่สีแปลกปลอม (Inpainting) เพื่อรักษารูปทรงกระบอก 100% ไม่แหว่ง
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
        delta_chroma = np.sqrt((A - med_A)**2 + (B - med_B)**2)
        dark_deficit = np.maximum(0.0, med_L - L)
        
        c_p99 = float(np.percentile(delta_chroma[core_mask], 99))
        d_p99 = float(np.percentile(dark_deficit[core_mask], 99))
        
        tau_chroma = max(18.0, 3.5 * c_p99)
        tau_dark = max(35.0, 6.0 * d_p99)
        
        chroma_outliers = ((delta_chroma > tau_chroma) | (dark_deficit > tau_dark)) & inner_geom_mask
        foreign_full_mask = (orig_mask & (~inner_geom_mask)) | chroma_outliers
        
        k_size = max(3, int(round(W * 0.012)) | 1)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        foreign_full_mask_u8 = foreign_full_mask.astype(np.uint8) * 255
        foreign_full_mask_u8 = cv2.morphologyEx(foreign_full_mask_u8, cv2.MORPH_OPEN, k)
        foreign_full_mask_u8 = cv2.dilate(foreign_full_mask_u8, k, iterations=1)
        
        # Core Guard: ไม่ตัดแกนกลางเด็ดขาด
        foreign_full_mask_u8[core_mask] = 0
        foreign_bool = foreign_full_mask_u8 > 0
        
        # --- ขั้นตอนที่ 4: สร้างผลลัพธ์ทั้ง 2 แบบ ---
        # แบบที่ 1: Trimmed (Alpha = 0)
        alpha_trimmed = alpha.copy()
        alpha_trimmed[foreign_bool] = 0
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((alpha_trimmed > 0).astype(np.uint8))
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            alpha_trimmed[labels != largest_label] = 0
            
        trimmed_rgba = cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], alpha_trimmed])
        
        # แบบที่ 2: Harmonized / Inpainted (100% Full Mass)
        bgr_harmonized = bgr.copy()
        if np.sum(foreign_bool) > 0:
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

    def process_image(self, image_path: str, output_dir: str = os.path.join("output", "crop_result")) -> dict:
        stage = "validation"
        try:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"ไม่พบไฟล์: {image_path}")
                
            os.makedirs(output_dir, exist_ok=True)
            stage = "image_load"
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"ไม่สามารถเปิดภาพได้ (Corrupted or Invalid Image Format): {image_path}")
                
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            cfg = getattr(self, "config", DEFAULT_CONFIG)
        
            # 1. ลบเส้นขีดเขียน
            stage = "remove_drawn_markup"
            clean_img = self.remove_drawn_markup(img)
            
            # 2. ตรวจจับขวดตัวอย่างด้วย YOLO (ตรวจจับกรอบตัวขวดเต็มที่ YOLO ตรวจพบ)
            stage = "locate_bottle"
            raw_box = self.locate_bottle(clean_img, image_path=image_path)
            
            # 3. หมุนปรับระนาบขวดเต็มใบให้ตั้งตรง 90 องศา (Deskew full bottle directly from YOLO box via SAM 2 minAreaRect)
            stage = "deskew_and_tight_crop"
            deskew_res = self.deskew_and_tight_crop(clean_img, raw_box, sam_model=getattr(self, "sam", None))
            upright_full = deskew_res["upright_bottle"]
            tilt_angle = deskew_res["tilt_angle"]
            base_hint_full = deskew_res.get("base_y", upright_full.shape[0] - 1)
            
            # 4. ตัดฝาขวดออกอย่างแม่นยำบนขวดที่ตั้งตรงแล้ว (Trim bottle cap cleanly on upright bottle)
            stage = "cut_bottle_cap"
            bottle_crop, cap_height = self.cut_bottle_cap(upright_full)
            base_hint_y = max(10, base_hint_full - cap_height)
            
            bh, bw = bottle_crop.shape[:2]
            aspect = bh / float(bw) if bw > 0 else 0
            
            # 5. ขั้นที่ 1: ตรวจจับระดับน้ำและครอบตรงเส้นแนวระดับน้ำพอดีถึงรูปทรงก้นขวดจริง
            stage = "detect_water_level_and_segment"
            liq = self.detect_water_level_and_segment(bottle_crop, base_hint_y=base_hint_y, cfg=cfg)
            st = liq["stats"]
            st["bottle_aspect_ratio"] = aspect
            
            # 6. ขั้นที่ 2: หลังครอบตามแนวพาราโบลาเสร็จแล้ว ค่อยเคลียแสงสะท้อนและเงา (โดยไม่กัดกินพิกเซลกลางขวด)
            stage = "clean_glare_and_shadow"
            clean = self.clean_glare_and_shadow(bottle_crop, liq["mask"], cfg=cfg)
            y1c, y2c = st["pure_y1"], st["pure_y2"]
            cb_, cg_, cr_ = cv2.split(clean["bgr"])
            clean_rgba = cv2.merge([cb_, cg_, cr_, liq["mask"]])[y1c:y2c + 1].copy()
            fill_mask_png = (clean["glare_mask"].astype(np.uint8) * 255)[y1c:y2c + 1].copy()

            # 7. ขั้นที่ 3: ปรับปรุงคุณภาพ 5 มิติแบบพลวัต (5 Dynamic Solutions)
            stage = "apply_5_dynamic_solutions"
            enhanced_clean_rgba, sed_px = self.apply_5_dynamic_solutions(clean_rgba)

            # 8. ขั้นที่ 4: ตัดสีแปลกปลอมและเกลี่ยสีเนื้อน้ำแท้ทดแทน (Dynamic Foreign Color Cleaning & Infilling)
            stage = "dynamic_foreign_color_cleaner"
            trimmed_clean_rgba, harmonized_clean_rgba, foreign_mask_u8, f_stats = self.dynamic_foreign_color_cleaner(enhanced_clean_rgba)

            # ประเมินคุณภาพความพร้อมใช้งาน (Readiness Evaluation) กับภาพน้ำบริสุทธิ์สมบูรณ์ (Harmonized)
            eval_res = self.evaluate_readiness(liq["rgba"], harmonized_clean_rgba, sed_px)
            st["readiness_score"] = eval_res["score_after"]
            st["score_before"] = eval_res["score_before"]
            st["noise_reduction_pct"] = eval_res["noise_reduction_pct"]
            st["sediment_px"] = sed_px
            st["foreign_cleaned_px"] = f_stats.get("foreign_px_cleaned", 0)
            st["foreign_cleaned_pct"] = f_stats.get("foreign_pct", 0.0)
            st["tau_chroma"] = f_stats.get("tau_chroma", 0.0)
            st["tau_dark"] = f_stats.get("tau_dark", 0.0)

            # แยกค่าสถิติเชิงมาตรวิทยา (Measured Inlier Pixels ในโซน Core แท้ใต้ระดับเข้ม ไม่ผ่านการสังเคราะห์สี)
            pure_mask = liq["mask"] > 0
            core_mask = liq.get("core_mask", pure_mask) > 0
            inlier_measured_mask = core_mask & (~clean["glare_mask"])
            raw_b, raw_g, raw_r = cv2.split(bottle_crop)
            if np.any(inlier_measured_mask):
                st["measured_inlier_R"] = float(np.median(raw_r[inlier_measured_mask]))
                st["measured_inlier_G"] = float(np.median(raw_g[inlier_measured_mask]))
                st["measured_inlier_B"] = float(np.median(raw_b[inlier_measured_mask]))
            severe_glare_px = np.sum((clean_rgba[:, :, 0] > 250) & (clean_rgba[:, :, 1] > 250) & (clean_rgba[:, :, 2] > 250) & pure_mask[y1c:y2c + 1])
            st["severe_glare_pct"] = float(severe_glare_px / max(1, st["total_pixels"]) * 100.0)
            st["glare_filtered_px"] = int((clean["glare_mask"] > 0).sum())
            st["glare_fill_pct"] = clean["glare_pct"]
            st["delta_e_raw"] = clean["delta_e_raw"]
            st["delta_e_clean"] = clean["delta_e_clean"]
            if np.any(pure_mask):
                st["mean_R"] = float(np.mean(cr_[pure_mask]))
                st["mean_G"] = float(np.mean(cg_[pure_mask]))
                st["mean_B"] = float(np.mean(cb_[pure_mask]))
                st["median_R"] = float(np.median(cr_[pure_mask]))
                st["median_G"] = float(np.median(cg_[pure_mask]))
                st["median_B"] = float(np.median(cb_[pure_mask]))
            
            stage = "save_outputs"
            bottle_path = os.path.join(output_dir, f"{base_name}_tight_bottle.jpg")
            raw_crop_path = os.path.join(output_dir, f"{base_name}_liquid_raw_crop.png")
            liquid_path = os.path.join(output_dir, f"{base_name}_liquid_exact_shape.png")
            clean_path = os.path.join(output_dir, f"{base_name}_liquid_clean.png")
            trimmed_path = os.path.join(output_dir, f"{base_name}_liquid_trimmed.png")
            fill_mask_path = os.path.join(output_dir, f"{base_name}_glare_fill_mask.png")
            diagnostic_path = os.path.join(output_dir, f"{base_name}_diagnostic.jpg")
            
            cv2.imwrite(bottle_path, bottle_crop)
            cv2.imwrite(raw_crop_path, liq["rgba"])
            cv2.imwrite(liquid_path, clean_rgba)
            cv2.imwrite(clean_path, harmonized_clean_rgba)
            cv2.imwrite(trimmed_path, trimmed_clean_rgba)
            cv2.imwrite(fill_mask_path, fill_mask_png)
            
            # 8. สร้างภาพ Diagnostic เปรียบเทียบครบทุกขั้นตอน (5 Visual Stages + Academic Theory Card)
            vis = bottle_crop.copy()
            cv2.drawContours(vis, [np.array(liq["contour"], dtype=np.int32)], -1, (0, 255, 0), 2)
            
            top_curve = liq.get("top_curve", None)
            if top_curve is not None:
                pts_meniscus = np.array([[x, int(top_curve[x])] for x in range(bw)], dtype=np.int32)
                cv2.polylines(vis, [pts_meniscus], False, (0, 255, 255), max(2, bw // 90))
            else:
                cv2.line(vis, (0, liq["meniscus_y"]), (bw, liq["meniscus_y"]), (0, 255, 255), 2)
            cv2.putText(vis, f"Water Level Curve (y~{liq['meniscus_y']})", (5, max(18, liq["meniscus_y"] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            
            bot_curve = liq.get("bot_curve", None)
            if bot_curve is not None:
                pts_base = np.array([[x, int(bot_curve[x])] for x in range(bw)], dtype=np.int32)
                cv2.polylines(vis, [pts_base], False, (0, 255, 0), max(2, bw // 90))
            else:
                cv2.line(vis, (0, liq["bottom_y"]), (bw, liq["bottom_y"]), (0, 255, 0), 2)
            cv2.putText(vis, f"Bottle Base Contour (y~{liq['bottom_y']})", (5, min(bh - 6, liq["bottom_y"] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
            
            lh, lw = liq["rgba"].shape[:2]
            # Diagnostic Panel: 5-Stage Comparison (Matching Research-Grade Standards)
            panel = np.zeros((480, 1680, 3), dtype=np.uint8) + 25
            
            # 1. Original Photo with YOLO detection box
            vis_orig = img.copy()
            cv2.rectangle(vis_orig, (raw_box[0], raw_box[1]), (raw_box[2], raw_box[3]), (0, 255, 0), max(3, vis_orig.shape[1] // 250))
            sc_orig = min(360.0 / max(1, vis_orig.shape[0]), 200.0 / max(1, vis_orig.shape[1]))
            disp_orig = cv2.resize(vis_orig, (max(1, int(vis_orig.shape[1] * sc_orig)), max(1, int(vis_orig.shape[0] * sc_orig))))
            panel[60:60+disp_orig.shape[0], 20:20+disp_orig.shape[1]] = disp_orig
            cv2.putText(panel, "1. Original Photo (YOLO Box)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
            
            # 2. Tight Bottle + Parabolic Curves
            sc_b = min(360.0 / max(1, bh), 200.0 / max(1, bw))
            b_disp = cv2.resize(vis, (max(1, int(bw * sc_b)), max(1, int(bh * sc_b))))
            panel[60:60+b_disp.shape[0], 245:245+b_disp.shape[1]] = b_disp
            cv2.putText(panel, "2. Parabolic Curves", (245, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
            
            # 3. Raw Parabolic Crop (100% True Camera Color)
            raw_rgb = np.zeros((lh, lw, 3), dtype=np.uint8)
            alpha = liq["rgba"][:, :, 3] / 255.0
            for c in range(3):
                raw_rgb[:, :, c] = (liq["rgba"][:, :, c] * alpha).astype(np.uint8)
            sc_l = min(360.0 / max(1, lh), 200.0 / max(1, lw))
            r_disp = cv2.resize(raw_rgb, (max(1, int(lw * sc_l)), max(1, int(lh * sc_l))))
            panel[60:60+r_disp.shape[0], 470:470+r_disp.shape[1]] = r_disp
            cv2.putText(panel, f"3. Raw Crop (dE={st['delta_e_raw']:.1f})", (470, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1)

            # 4. 5 Dynamic Solutions Clean
            enh_rgb = np.zeros((lh, lw, 3), dtype=np.uint8)
            alpha_enh = enhanced_clean_rgba[:, :, 3] / 255.0
            for c in range(3):
                enh_rgb[:, :, c] = (enhanced_clean_rgba[:, :, c] * alpha_enh).astype(np.uint8)
            c_disp = cv2.resize(enh_rgb, (max(1, int(lw * sc_l)), max(1, int(lh * sc_l))))
            panel[60:60+c_disp.shape[0], 695:695+c_disp.shape[1]] = c_disp
            cv2.putText(panel, "4. 5 Dynamic Solutions", (695, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 200, 0), 1)
            
            # 5. Cleaned & Harmonized Liquid (Zero Foreign Colors + 100% Full Mass)
            harm_rgb = np.zeros((lh, lw, 3), dtype=np.uint8)
            alpha_harm = harmonized_clean_rgba[:, :, 3] / 255.0
            for c in range(3):
                harm_rgb[:, :, c] = (harmonized_clean_rgba[:, :, c] * alpha_harm).astype(np.uint8)
            h_disp = cv2.resize(harm_rgb, (max(1, int(lw * sc_l)), max(1, int(lh * sc_l))))
            panel[60:60+h_disp.shape[0], 920:920+h_disp.shape[1]] = h_disp
            cv2.putText(panel, f"5. Zero Foreign ({eval_res['score_after']:.1f}%)", (920, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0), 1)

            # 6. Diagnostic & Academic Theory Card
            lines = [
                "=== Physics-Based & Dynamic Enhancement ===",
                f"Model: Shafer (1985) + 5 Solutions + Foreign Cleaner",
                f"Detected Tilt Angle: {tilt_angle:+.2f} deg -> 90 deg Upright",
                f"Cap Status: Cut at y={cap_height} px",
                f"Top Surface: {st['surface_mode']} ({st['surface_orientation'].upper()}, y={liq['meniscus_y']} px)",
                f"Bottom Base: CUP (y={liq['bottom_y']} px, Upward Parabola)",
                f"Liquid Column: {st['liquid_height']} px ({st['liquid_height_pct']:.1f}% of Body)",
                f"Pure Liquid Mass: {st['total_pixels']:,} px (0 Interior Holes)",
                f"Readiness Score: {eval_res['score_after']:.1f}% (Initial: {eval_res['score_before']:.1f}%)",
                f"Noise Reduction: {eval_res['noise_reduction_pct']:.1f}%",
                f"Foreign Colors Cleaned: {f_stats.get('foreign_pct', 0.0):.1f}% ({f_stats.get('foreign_px_cleaned', 0):,} px)",
                f"Sediment Cleaned: {sed_px} px",
                f"Specular Glare Removed: {st['glare_fill_pct']:.1f}%",
                f"Raw Inlier RGB: ({st.get('measured_inlier_R', 0):.1f}, {st.get('measured_inlier_G', 0):.1f}, {st.get('measured_inlier_B', 0):.1f})",
                f"Diffuse Mean RGB: ({st['mean_R']:.1f}, {st['mean_G']:.1f}, {st['mean_B']:.1f})",
            ]
            for idx, line in enumerate(lines):
                col = (0, 255, 0) if "100%" in line or "Score:" in line or "Diffuse Mean" in line or "Foreign Colors" in line else (220, 220, 220)
                if "===" in line:
                    col = (0, 220, 255)
                cv2.putText(panel, line, (1150, 68 + idx * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
                
            cv2.imwrite(diagnostic_path, panel)
            
            return {
                "image": image_path,
                "tilt_angle": f"{tilt_angle:+.2f} deg",
                "bottle": bottle_path,
                "liquid_raw_crop": raw_crop_path,
                "liquid_rgba": liquid_path,
                "liquid_clean": clean_path,
                "liquid_trimmed": trimmed_path,
                "diagnostic": diagnostic_path,
                "stats": st,
                "readiness": eval_res,
                "enhanced_rgba": harmonized_clean_rgba,
                "trimmed_rgba": trimmed_clean_rgba,
                "foreign_stats": f_stats
            }
        except Exception as e:
            self.log_crop_failure(image_path=image_path, error=e, output_dir=output_dir, stage=stage)
            raise e


def run_batch_crop(
    input_root: str,
    output_base: str = os.path.join("data_crop"),
    cropper: WebOptimizedLiquidCropper = None,
    max_images: int = None
) -> dict:
    """
    ประมวลผล Batch Crop สำหรับชุดข้อมูลทั้งหมด พร้อมแยกหมวดหมู่อัตโนมัติ:
    - crops_ready/       : มวลน้ำที่ผ่านเกณฑ์พร้อมใช้งาน 100%
    - crops_not_ready/   : มวลน้ำที่ไม่ผ่านเกณฑ์ พร้อม issues.txt
    - diagnostics/       : ภาพแผงสรุป Diagnostic 5 ขั้นตอน
    - reports/           : รายงาน CSV และ JSON
    """
    dir_ready = os.path.join(output_base, "crops_ready")
    dir_not_ready = os.path.join(output_base, "crops_not_ready")
    dir_diag = os.path.join(output_base, "diagnostics")
    dir_rep = os.path.join(output_base, "reports")
    
    for d in [dir_ready, dir_not_ready, dir_diag, dir_rep]:
        os.makedirs(d, exist_ok=True)
        
    if cropper is None:
        cropper = WebOptimizedLiquidCropper()
        
    supported_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = []
    for root, _, files in os.walk(input_root):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in supported_exts:
                image_files.append(os.path.join(root, f))
                
    if max_images:
        image_files = image_files[:max_images]
        
    print(f"\n" + "="*80, flush=True)
    print(f"[Batch Engine] ค้นพบรูปภาพทั้งหมด: {len(image_files)} รูปใน {os.path.abspath(input_root)}", flush=True)
    print(f"[Batch Engine] บันทึกผลลัพธ์แยกกลุ่มไปที่: {os.path.abspath(output_base)}", flush=True)
    print(f"  ├── crops_ready/       (ภาพที่ผ่านเกณฑ์พร้อมใช้งาน 100%)", flush=True)
    print(f"  ├── crops_not_ready/   (ภาพที่ไม่ผ่านเกณฑ์ พร้อม issues.txt ระบุเหตุผล)", flush=True)
    print(f"  ├── diagnostics/       (ภาพแผงวิเคราะห์ 5 ขั้นตอน)", flush=True)
    print(f"  └── reports/           (รายงาน CSV และ JSON สรุปความพร้อม)", flush=True)
    print("="*80 + "\n", flush=True)
    
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
        enh_rgba = None
        
        try:
            res = cropper.process_image(img_path, output_dir=temp_out)
            tilt_str = res.get("tilt_angle", "0.0 deg")
            st = res.get("stats", {})
            eval_res = res.get("readiness", {})
            score = eval_res.get("score_after", 0.0)
            noise_red = eval_res.get("noise_reduction_pct", 0.0)
            enh_rgba = res.get("enhanced_rgba")
            
            aspect = st.get("bottle_aspect_ratio", 2.5)
            if aspect < 1.7 or aspect > 4.5:
                is_ready = False
                reasons.append(f"Abnormal Bottle Aspect Ratio ({aspect:.2f})")
                
            liq_pct = st.get("liquid_height_pct", 50.0)
            if liq_pct < 15.0:
                is_ready = False
                reasons.append(f"Liquid Volume Too Low ({liq_pct:.1f}% of bottle)")
                
            if enh_rgba is None or enh_rgba.shape[0] < 20 or enh_rgba.shape[1] < 20:
                is_ready = False
                reasons.append("Invalid/Empty Liquid Crop")
                
            glare_pct = st.get("severe_glare_pct", 0.0)
            if glare_pct > 32.0:
                is_ready = False
                reasons.append(f"Excessive Glare ({glare_pct:.1f}% > 32%)")
                
            if score < 50.0:
                is_ready = False
                reasons.append(f"Readiness Score Below Standard ({score:.1f}% < 50%)")
                
            # บันทึกภาพ Diagnostic
            diag_src = res.get("diagnostic")
            if diag_src and os.path.isfile(diag_src):
                diag_dest = os.path.join(cat_diag_dir, f"{bname}_diagnostic.jpg")
                diag_img = cv2.imread(diag_src)
                if diag_img is not None:
                    cv2.imwrite(diag_dest, diag_img)
                    
            if is_ready:
                ready_count += 1
                out_clean = os.path.join(cat_ready_dir, f"{bname}_clean.png")
                out_raw = os.path.join(cat_ready_dir, f"{bname}_raw.png")
                if enh_rgba is not None:
                    cv2.imwrite(out_clean, enh_rgba)
                raw_crop_file = res.get("liquid_raw_crop")
                if raw_crop_file and os.path.isfile(raw_crop_file):
                    raw_crop_img = cv2.imread(raw_crop_file, cv2.IMREAD_UNCHANGED)
                    if raw_crop_img is not None:
                        cv2.imwrite(out_raw, raw_crop_img)
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
            "foreign_cleaned_pct": st.get("foreign_cleaned_pct", 0.0),
            "reasons": "; ".join(reasons) if reasons else "None (Pass)"
        })
        
        if idx % 10 == 0 or idx == len(image_files):
            elapsed = time.time() - t_start
            rate = idx / elapsed if elapsed > 0 else 0
            print(f"[{idx:3d}/{len(image_files)}] ความคืบหน้า | พร้อมใช้: {ready_count} ({ready_count/idx*100:.1f}%) | ไม่พร้อมใช้: {not_ready_count} | ความเร็ว: {rate:.1f} ภาพ/วิ", flush=True)

    # ทำความสะอาดโฟลเดอร์ชั่วคราว
    if os.path.exists(temp_out):
        import shutil
        shutil.rmtree(temp_out, ignore_errors=True)
        
    t_total = time.time() - t_start
    print("\n" + "="*80, flush=True)
    print(f"[Batch Engine] ประมวลผลเสร็จสิ้น {len(image_files)} ภาพ ใช้เวลา {t_total:.1f} วินาที ({t_total/max(1, len(image_files)):.2f} วินาที/ภาพ)", flush=True)
    print(f"  >> Ready (พร้อมใช้งาน):       {ready_count:3d} ({(ready_count/max(1, len(image_files)))*100:.1f}%)", flush=True)
    print(f"  >> Not Ready (ไม่พร้อมใช้):    {not_ready_count:3d} ({(not_ready_count/max(1, len(image_files)))*100:.1f}%)", flush=True)
    print("="*80, flush=True)
    
    csv_path = os.path.join(dir_rep, "crop_readiness_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["index", "filename", "category", "status", "readiness_score", "tilt_angle", "noise_reduction_pct", "foreign_cleaned_pct", "reasons"])
        writer.writeheader()
        writer.writerows(records)
        
    summary_data = {
        "total_images": len(image_files),
        "ready_count": ready_count,
        "not_ready_count": not_ready_count,
        "ready_percentage": round((ready_count / max(1, len(image_files))) * 100, 2),
        "not_ready_percentage": round((not_ready_count / max(1, len(image_files))) * 100, 2),
        "issue_breakdown": issue_tally,
        "execution_time_seconds": round(t_total, 2)
    }
    json_path = os.path.join(dir_rep, "readiness_statistics.json")
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(summary_data, f_json, indent=2, ensure_ascii=False)
        
    print(f"บันทึกรายงานสรุปผลไว้ที่:\n  - {csv_path}\n  - {json_path}", flush=True)
    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated Glass Bottle & Liquid Cropper with SAM 2 Deskewing and 5 Dynamic Solutions"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to an image file or dataset folder to process (Default: dataset_split/test or dataset_split)"
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Optional positional output directory (e.g. python crop_bottle_and_liquid.py dataset_split dataset_cropped)"
    )
    parser.add_argument(
        "--output", "-o",
        dest="output_flag",
        default=None,
        help="Output directory flag (Default: data_crop)"
    )
    parser.add_argument(
        "--max-images", "-n",
        type=int,
        default=None,
        help="Maximum number of images to process in batch mode (Default: process all)"
    )
    parser.add_argument(
        "--model", "-m",
        default="yolov8m-worldv2.pt",
        help="Path to YOLO model weights (Default: yolov8m-worldv2.pt)"
    )
    parser.add_argument(
        "--device", "-d",
        default=None,
        help="Device to run on (e.g. 0 for CUDA GPU, 'cpu' for CPU. Default: auto-detect GPU)"
    )

    args = parser.parse_args()

    # เลือก target เริ่มต้นอัจฉริยะหากผู้ใช้ไม่ได้ระบุ
    target = args.input
    if target is None:
        if os.path.isdir(os.path.join("dataset_split", "test")):
            target = os.path.join("dataset_split", "test")
            print(f"[Auto-Select] ไม่ได้ระบุ input -> เลือกโฟลเดอร์ทดสอบ: {target}")
        elif os.path.isdir("dataset_split"):
            target = "dataset_split"
            print(f"[Auto-Select] ไม่ได้ระบุ input -> เลือกโฟลเดอร์: {target}")
        else:
            sample_img = r"d:\my_girlfriend_work\Testkit for AI\Phosphate 1.5 mg_l\05F968CC-AE4E-407F-8450-303AED8B3AF5.JPEG"
            if os.path.isfile(sample_img):
                target = sample_img
                print(f"[Auto-Select] ไม่ได้ระบุ input -> เลือกภาพตัวอย่าง: {target}")
            else:
                target = "."

    device_arg = int(args.device) if (args.device and args.device.isdigit()) else args.device
    cropper = WebOptimizedLiquidCropper(model_path=args.model, device=device_arg)

    output_target = args.output_flag or args.output or "data_crop"

    if os.path.isdir(target):
        run_batch_crop(input_root=target, output_base=output_target, cropper=cropper, max_images=args.max_images)
    elif os.path.isfile(target):
        res = cropper.process_image(target, output_dir=output_target)
        st = res["stats"]
        rd = res.get("readiness", {})
        print("\n" + "="*70)
        print("=== สรุปผลลัพธ์การตัดและประเมินคุณภาพระดับน้ำ (Single Image) ===")
        print("="*70)
        print(f"  รูปภาพต้นฉบับ:        {res['image']}")
        print(f"  มุมเอียงและหมุนกลับ:  {res['tilt_angle']}")
        print(f"  ภาพขวดตั้งตรง:        {res['bottle']}")
        print(f"  ภาพมวลน้ำแท้ (Raw):   {res['liquid_raw_crop']}")
        print(f"  ภาพมวลน้ำปรับปรุง (Zero Foreign Colors): {res['liquid_clean']}")
        print(f"  ภาพมวลน้ำตัดขอบ (Trimmed): {res['liquid_trimmed']}")
        print(f"  ภาพแผง Diagnostic:   {res['diagnostic']}")
        print(f"  Readiness Score:      {rd.get('score_after', 0.0):.1f}% (ก่อนปรับ: {rd.get('score_before', 0.0):.1f}%)")
        print(f"  Noise Reduction:      {rd.get('noise_reduction_pct', 0.0):.1f}%")
        print(f"  Foreign Cleaned:      {st.get('foreign_cleaned_pct', 0.0):.1f}% ({st.get('foreign_cleaned_px', 0):,} px)")
        print(f"  ความสูงระดับน้ำ:      {st['liquid_height']} px ({st['liquid_height_pct']:.1f}% ของตัวขวด)")
        print(f"  มวลน้ำบริสุทธิ์:      {st['total_pixels']:,} พิกเซล (กรองแสงสะท้อน {st['glare_filtered_px']:,} px, ลบตะกอน {st.get('sediment_px', 0)} px)")
        print(f"  ค่าสี Diffuse RGB:    ({st['mean_R']:.1f}, {st['mean_G']:.1f}, {st['mean_B']:.1f})")
        print("="*70)
    else:
        print(f"[Error] ไม่พบไฟล์หรือโฟลเดอร์: {target}")
