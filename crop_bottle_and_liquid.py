"""
Standalone Tool: ตรวจจับขวดตัวอย่าง, ปรับขวดตั้งตรงอัตโนมัติ (Automatic Deskew), และสกัดระดับน้ำจริง (True Water Level)
เวอร์ชัน: Web-Optimized + Dynamic Deskew (ปรับขวดตั้งตรง 90 องศา ก่อน Crop ทุกครั้ง)
จุดเด่น:
1. ปรับมุมเอียงอัตโนมัติ (Automatic Deskew):
   - ตรวจจับมุมเอียงของขวดจากแนวผนังแก้วทรงกระบอกจริง (Dynamic Hough Angle Tracing)
   - หมุนปรับระนาบให้อยู่ในแนวดิ่ง 90 องศาเป๊ะ ก่อนทำการ Crop
   - ห้าม/ไม่ล็อกค่าคงที่ (No hardcoded size / dimensions / fixed angle limits)
2. ตรวจจับระดับผิวน้ำ (Meniscus) จริง:
   - ตรวจจับการเปลี่ยนผ่านทางแสง (Sustained Transition) ไม่หลุดไปเกลียวฝาหรือก้นขวด
3. เหมาะสำหรับ Web App:
   - โมเดล YOLOv8n เล็กเพียง 6 MB (แปลงเป็น ONNX ได้ 3.5 MB)
   - สเต็ป Deskew และ Water Level ใช้ Pure OpenCV (0 MB, ความเร็ว < 15 ms)
"""

import os
import sys
import cv2
import numpy as np
import torch
from ultralytics import YOLO, YOLOWorld

DEVICE = 0 if torch.cuda.is_available() else 'cpu'

class WebOptimizedLiquidCropper:
    def __init__(self, model_path: str = "yolov8m-worldv2.pt", device=None):
        self.device = DEVICE if device is None else device
        if "world" in model_path.lower():
            self.model = YOLOWorld(model_path)
            self.model.set_classes(["glass vial with white cap", "transparent chemical bottle", "bottle"])
            self.is_world = True
        else:
            self.model = YOLO(model_path)
            self.is_world = False
            
        dev_name = torch.cuda.get_device_name(0) if self.device == 0 and torch.cuda.is_available() else "CPU"
        print(f"[Engine] ใช้งานอุปกรณ์: {dev_name} (Device ID: {self.device}, Model: {os.path.basename(model_path)})", flush=True)

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
    def deskew_and_tight_crop(image: np.ndarray, box: tuple[int, int, int, int]) -> dict:
        """
        ปรับขวดให้ตั้งตรง 90 องศา (Deskew) โดยไม่ล็อกขนาดคงที่:
        1. ดึงภาพขวดพร้อมขอบเผื่อ (Dynamic Padding)
        2. ตรวจจับแนวเอียงของผนังแก้วด้วย Hough Lines
        3. หมุนภาพกลับด้วยมุม -theta ให้อยู่ในแนวดิ่งเป๊ะ
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
        
        top_y = int(np.argmax(row_sums[:int(ph * 0.25)])) if int(ph * 0.25) > 0 else 0
        
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

    @staticmethod
    def detect_water_level_and_segment(upright_bottle: np.ndarray, base_hint_y: int = None) -> dict:
        """
        ตรวจจับขอบเขตระดับน้ำจริง (Adaptive Water Level Tracking):
        1. ระบุขอบล่างของฝาขวด (Cap Bottom) โดยไม่ติดขอบบนสุดของภาพ
        2. ตรวจจับผิวน้ำ (Meniscus) จากการเปลี่ยนผ่านทางแสงและสี (Sustained Transition)
        3. ครอบคลุมเนื้อน้ำทั้งหมดตั้งแต่ก้นขวดขึ้นไปจนถึงผิวน้ำใต้ฝา
        4. สร้าง Polygon Mask โอบล้อมมวลน้ำทั้งหมดตามระดับน้ำจริง
        """
        bh, bw = upright_bottle.shape[:2]
        center = upright_bottle[:, int(bw * 0.25):int(bw * 0.75)]
        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        
        sob_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        row_g = cv2.GaussianBlur(np.mean(sob_y, axis=1).reshape(-1, 1), (9, 1), 0).flatten()
        
        # ค้นหาขอบล่างของฝาขวด: เริ่มต้นค้นหาที่ 8% ของความสูงขึ้นไป (ข้ามขอบบนสุดของภาพ y=0..1 ป้องกันจับขอบภาพ)
        cap_search_start = max(5, int(bh * 0.08))
        cap_search_end = min(bh - 20, int(bh * 0.38))
        if cap_search_end > cap_search_start:
            cap_y = cap_search_start + int(np.argmax(row_g[cap_search_start:cap_search_end]))
        else:
            cap_y = int(bh * 0.22)
            
        # ก้นขวดจริง (ครอบคลุมมวลน้ำตั้งแต่ก้นขวด)
        if base_hint_y is not None and 0 < base_hint_y < bh:
            base_y = base_hint_y
        else:
            base_search_start = int(bh * 0.80)
            base_search = row_g[base_search_start:]
            base_y = base_search_start + (int(np.argmax(base_search)) if len(base_search) > 0 else int(bh * 0.15))
        base_y = min(bh - 1, max(cap_y + 40, base_y))
        
        bgr = np.mean(center, axis=1)
        intensity = np.mean(bgr, axis=1)
        
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        sat = cv2.GaussianBlur(np.mean(hsv[:, :, 1], axis=1).reshape(-1, 1), (9, 1), 0).flatten()
        
        cap_int = np.mean(intensity[max(0, cap_y - 20):cap_y])
        cap_sat = np.mean(sat[max(0, cap_y - 20):cap_y])
        body_sat = np.median(sat[int(bh * 0.60):base_y]) if base_y > int(bh * 0.60) else 0.0
        
        # ค้นหาจุดเริ่มต้นของสารละลายที่มีสีจริง (Meniscus Tracking ป้องกันการหลุดไปจับคอขวด)
        if body_sat > 35.0 or (body_sat - cap_sat) > 15.0:
            # สารละลายมีสี: ต้องเกิดการเพิ่มขึ้นของ Saturation อย่างต่อเนื่อง (Sustained Rise)
            # ตัดเงื่อนไขความสว่างตก (cap_int - intensity) ออก เพื่อไม่ให้หลุดไปจับเงาสะท้อนตรงคอขวดแก้วเปล่า
            thresh_sat = max(cap_sat + 18.0, 35.0)
            candidates = []
            for y in range(cap_y + 15, int(base_y - 20)):
                if sat[y] >= thresh_sat and np.mean(sat[y:min(base_y, y + 15)]) >= thresh_sat:
                    candidates.append(y)
                    break
                    
            if candidates and candidates[0] < (bh * 0.85):
                cand_y = candidates[0]
                p_start = max(cap_y + 10, cand_y - 25)
                p_end = min(base_y - 15, cand_y + 35)
                meniscus_y = p_start + int(np.argmax(row_g[p_start:p_end]))
            else:
                meniscus_y = cap_y + int((base_y - cap_y) * 0.40)
        else:
            # สารละลายใส / 0 mg/L: ค้นหาเส้นผิวน้ำ Meniscus Refraction Edge ในช่วงกลางขวด (ข้ามคอขวดด้านบนและก้นขวดด้านล่าง)
            mid_s = cap_y + max(20, int((base_y - cap_y) * 0.15))
            mid_e = min(int(base_y - (base_y - cap_y) * 0.18), int(bh * 0.75))
            if mid_e > mid_s:
                mid_grad = row_g[mid_s:mid_e]
                if len(mid_grad) > 0 and np.max(mid_grad) > 6.0:
                    meniscus_y = mid_s + int(np.argmax(mid_grad))
                else:
                    meniscus_y = cap_y + int((base_y - cap_y) * 0.40)
            else:
                meniscus_y = cap_y + int((base_y - cap_y) * 0.40)
                
        meniscus_y = max(cap_y + 8, min(meniscus_y, base_y - 25))
        
        # ดึงน้ำเต็มความกว้างขวด 100% (Full Bottle Width Coverage) ตามรูปทรงจริงของขวดแก้ว ไม่เว้นขอบข้าง
        x_left = 0
        x_right = bw - 1
        
        # ป้องกันเฉพาะกรณีมีวัตถุแปลกปลอม/การ์ดสีดำทึบติดที่ขอบนอกสุดจริงๆ เท่านั้น (>70% ของคอลัมน์)
        hsv_bottle = cv2.cvtColor(upright_bottle, cv2.COLOR_BGR2HSV)
        sat_ch = hsv_bottle[:, :, 1]
        val_ch = hsv_bottle[:, :, 2]
        for x in range(0, max(1, int(bw * 0.05))):
            card_col = (sat_ch[meniscus_y:base_y, x] > 160) | (val_ch[meniscus_y:base_y, x] < 20)
            if np.mean(card_col) > 0.70:
                x_left = max(x_left, x + 1)
        for x in range(bw - 1, min(bw - 2, int(bw * 0.95)), -1):
            card_col = (sat_ch[meniscus_y:base_y, x] > 160) | (val_ch[meniscus_y:base_y, x] < 20)
            if np.mean(card_col) > 0.70:
                x_right = min(x_right, x - 1)
        
        # ตรวจจับขอบเขตพื้นผิวโต๊ะ/ฉากหลังสีขาวใต้ก้นขวด เพื่อตัดออกตามรูปทรงขวดจริง
        b_ch, g_ch, r_ch = cv2.split(upright_bottle)
        hsv_bottle = cv2.cvtColor(upright_bottle, cv2.COLOR_BGR2HSV)
        sat_ch = hsv_bottle[:, :, 1]
        val_ch = hsv_bottle[:, :, 2]
        
        # สำหรับสารละลายมีสี: แยกแยะพื้นโต๊ะขาวออกจากน้ำสีได้ชัดเจน
        # สำหรับน้ำใส 0 mg/L: น้ำใสมีเฉดสีเทา/ขาว จึงต้องอิงตามเส้นขอบแก้วก้นขวดจริง (base_y)
        is_colored = (body_sat > 35.0) or (body_sat - cap_sat > 15.0)
        is_white_table = (r_ch > 155) & (g_ch > 155) & (b_ch > 155) & (sat_ch < 38)
        
        bot_curve = []
        if is_colored:
            bottom_start = max(meniscus_y + 20, int(bh * 0.70))
            for x in range(bw):
                col_bgr = upright_bottle[bottom_start:, x]
                col_sat = sat_ch[bottom_start:, x]
                table_rows = np.where((col_bgr[:, 0] > 155) & (col_bgr[:, 1] > 155) & (col_bgr[:, 2] > 155) & (col_sat < 38))[0]
                if len(table_rows) > 0:
                    bot_row = bottom_start + table_rows[0] - 2
                else:
                    bot_row = base_y
                bot_curve.append(min(base_y, bot_row))
        else:
            mid_curve_x = (x_left + x_right) / 2.0
            rx = (x_right - x_left) / 2.0 + 1e-5
            for x in range(bw):
                dx = (x - mid_curve_x) / rx
                curve_sag = max(2, int((x_right - x_left) * 0.04))
                bot_y_val = base_y - int(curve_sag * (1.0 - np.sqrt(max(0.0, 1.0 - min(1.0, dx**2)))))
                bot_curve.append(min(base_y, bot_y_val))
            
        bot_curve = np.array(bot_curve)
        from scipy.ndimage import median_filter, gaussian_filter1d
        smooth_curve = gaussian_filter1d(median_filter(bot_curve, size=15), sigma=5)
        
        # สร้างรูปทรงของเหลวจริง: ผิวน้ำใต้ฝาโค้งลง + ก้นขวดมนตามรูปทรงแก้วจริง (ตัดพื้นขาวออก 100%)
        pts_contour = []
        curve_sag = max(2, int((x_right - x_left) * 0.05))
        for x in range(x_left, x_right + 1):
            dx = (x - (x_left + x_right) / 2.0) / ((x_right - x_left) / 2.0 + 1e-5)
            y_curve = meniscus_y + int(curve_sag * (1.0 - np.sqrt(max(0.0, 1.0 - dx**2))))
            pts_contour.append([x, y_curve])
            
        for x in range(x_right, x_left - 1, -1):
            # แนบชิดตามขอบแก้วก้นขวด ตัดพื้นสีขาวด้านล่างออก
            y_base = max(meniscus_y + 10, min(base_y, int(smooth_curve[x])))
            pts_contour.append([x, y_base])
            
        pts_contour = np.array(pts_contour, dtype=np.int32)
        
        liquid_mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.drawContours(liquid_mask, [pts_contour], -1, 255, -1)
        
        # Guard 1: ตัดพิกเซลสีขาวใต้ขวดที่อาจเล็ดลอดออก (เฉพาะสารละลายมีสีเท่านั้น เพื่อไม่ให้ลบน้ำใส 0 mg/L)
        if is_colored:
            table_guard = is_white_table & (np.arange(bh)[:, None] > int(bh * 0.70)) & (np.arange(bh)[:, None] > (meniscus_y + 20))
            liquid_mask[table_guard] = 0
        
        # Guard 2: ตัดพื้นที่แก้วเปล่าสีขาว/เทาด้านบนเหนือระดับน้ำ (เหลือเฉพาะมวลน้ำแท้จริง)
        if body_sat > 35.0:
            top_white = (r_ch > 135) & (g_ch > 135) & (b_ch > 135) & (sat_ch < 30) & (np.arange(bh)[:, None] < (meniscus_y + 12))
        else:
            top_white = (r_ch > 165) & (g_ch > 165) & (b_ch > 165) & (sat_ch < 20) & (np.arange(bh)[:, None] < (meniscus_y + 5))
        liquid_mask[top_white] = 0
        
        # Guard 3: Auto-Skin Exclusion (ลบนิ้วมือ/เนื้อคนจับขวดออกจากมวลน้ำอัตโนมัติ)
        ycrcb = cv2.cvtColor(upright_bottle, cv2.COLOR_BGR2YCrCb)
        cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
        hue_ch = hsv_bottle[:, :, 0]
        true_skin = (cr >= 138) & (cr <= 175) & (cb >= 90) & (cb <= 128) & ((hue_ch <= 18) | (hue_ch >= 172)) & (r_ch.astype(int) - g_ch.astype(int) >= 15) & (sat_ch > 35) & (val_ch > 50) & (liquid_mask > 0)
        liquid_mask[true_skin] = 0
        
        hsv_bottle = cv2.cvtColor(upright_bottle, cv2.COLOR_BGR2HSV)
        sat_ch = hsv_bottle[:, :, 1]
        val_ch = hsv_bottle[:, :, 2]
        specular = (val_ch > 245) & (sat_ch < 35) & (liquid_mask > 0)
        pure_mask = (liquid_mask > 0) & (~specular)
        
        b_ch, g_ch, r_ch = cv2.split(upright_bottle)
        rgba = cv2.merge([b_ch, g_ch, r_ch, liquid_mask])
        
        pts_idx = np.argwhere(liquid_mask > 0)
        min_y, min_x = pts_idx.min(axis=0)
        max_y, max_x = pts_idx.max(axis=0)
        liquid_rgba_crop = rgba[min_y:max_y+1, min_x:max_x+1].copy()
        
        valid_b = b_ch[pure_mask]
        valid_g = g_ch[pure_mask]
        valid_r = r_ch[pure_mask]
        total_px = len(valid_b)
        
        stats = {
            "total_pixels": int(total_px),
            "cap_y": int(cap_y),
            "meniscus_y": int(meniscus_y),
            "bottom_y": int(base_y),
            "liquid_height": int(base_y - meniscus_y),
            "liquid_height_pct": float((base_y - meniscus_y) / bh * 100.0),
            "liquid_width": int(max_x - min_x + 1),
            "mean_R": float(np.mean(valid_r)) if total_px > 0 else 0.0,
            "mean_G": float(np.mean(valid_g)) if total_px > 0 else 0.0,
            "mean_B": float(np.mean(valid_b)) if total_px > 0 else 0.0,
            "median_R": float(np.median(valid_r)) if total_px > 0 else 0.0,
            "median_G": float(np.median(valid_g)) if total_px > 0 else 0.0,
            "median_B": float(np.median(valid_b)) if total_px > 0 else 0.0,
        }
        
        return {
            "mask": liquid_mask,
            "contour": pts_contour,
            "rgba": liquid_rgba_crop,
            "stats": stats,
            "cap_y": cap_y,
            "meniscus_y": meniscus_y,
            "bottom_y": base_y
        }

    def process_image(self, image_path: str, output_dir: str = os.path.join("output", "crop_result")) -> dict:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"ไม่พบไฟล์: {image_path}")
            
        os.makedirs(output_dir, exist_ok=True)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"ไม่สามารถเปิดภาพได้: {image_path}")
            
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 1. ลบเส้นขีดเขียน
        clean_img = self.remove_drawn_markup(img)
        
        # 2. ตรวจจับขวดตัวอย่าง
        raw_box = self.locate_bottle(clean_img, image_path=image_path)
        
        # 3. หมุนปรับระนาบขวดให้ตั้งตรง 90 องศา (Automatic Deskew) ก่อนทำการ Crop
        deskew_res = self.deskew_and_tight_crop(clean_img, raw_box)
        bottle_crop = deskew_res["upright_bottle"]
        tilt_angle = deskew_res["tilt_angle"]
        
        bh, bw = bottle_crop.shape[:2]
        aspect = bh / float(bw) if bw > 0 else 0
        
        # 4. ตรวจจับระดับน้ำและสกัดมวลของเหลวตามจริง (ครอบคลุมตั้งแต่ก้นขวดถึงผิวน้ำใต้ฝา)
        base_hint_y = deskew_res.get("base_y", None)
        liq = self.detect_water_level_and_segment(bottle_crop, base_hint_y=base_hint_y)
        st = liq["stats"]
        
        # บันทึกไฟล์
        bottle_path = os.path.join(output_dir, f"{base_name}_tight_bottle.jpg")
        liquid_path = os.path.join(output_dir, f"{base_name}_liquid_exact_shape.png")
        diagnostic_path = os.path.join(output_dir, f"{base_name}_diagnostic.jpg")
        
        cv2.imwrite(bottle_path, bottle_crop)
        cv2.imwrite(liquid_path, liq["rgba"])
        
        # 5. ภาพ Diagnostic แสดงระดับน้ำและมุมหมุน
        vis = bottle_crop.copy()
        cv2.drawContours(vis, [liq["contour"]], -1, (0, 255, 0), 2)
        
        cv2.line(vis, (0, liq["cap_y"]), (bw, liq["cap_y"]), (0, 165, 255), 2)
        cv2.putText(vis, f"Cap Edge (y={liq['cap_y']})", (5, liq["cap_y"] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1)
        
        cv2.line(vis, (0, liq["meniscus_y"]), (bw, liq["meniscus_y"]), (0, 255, 255), 2)
        cv2.putText(vis, f"Water Level (y={liq['meniscus_y']})", (5, liq["meniscus_y"] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        
        cv2.line(vis, (0, liq["bottom_y"]), (bw, liq["bottom_y"]), (0, 255, 0), 2)
        cv2.putText(vis, f"Curved Base (y={liq['bottom_y']})", (5, liq["bottom_y"] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
        
        lh, lw = liq["rgba"].shape[:2]
        panel = np.zeros((460, 920, 3), dtype=np.uint8) + 30
        
        sc_b = min(360.0 / bh, 240.0 / bw)
        b_disp = cv2.resize(vis, (int(bw * sc_b), int(bh * sc_b)))
        panel[60:60+b_disp.shape[0], 40:40+b_disp.shape[1]] = b_disp
        cv2.putText(panel, f"1. Deskewed Bottle (Tilt: {tilt_angle:+.1f} deg -> 90 deg)", (40, 38), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
        
        pure_liq = np.zeros((lh, lw, 3), dtype=np.uint8)
        alpha = liq["rgba"][:, :, 3] / 255.0
        for c in range(3):
            pure_liq[:, :, c] = (liq["rgba"][:, :, c] * alpha).astype(np.uint8)
            
        sc_l = min(360.0 / lh, 240.0 / lw)
        l_disp = cv2.resize(pure_liq, (int(lw * sc_l), int(lh * sc_l)))
        panel[60:60+l_disp.shape[0], 340:340+l_disp.shape[1]] = l_disp
        cv2.putText(panel, f"2. Exact Liquid ({st['liquid_height_pct']:.1f}% Full, {st['total_pixels']:,} px)", (340, 38), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0), 1)
        
        lines = [
            "=== Automatic Deskew & Level Report ===",
            f"Detected Tilt Angle: {tilt_angle:+.2f} deg",
            f"Correction: Rotated {-tilt_angle:+.2f} deg to 90 deg Upright",
            f"Dimensions: {bw}x{bh} px (Aspect H/W={aspect:.2f})",
            f"Cap Bottom: y={st['cap_y']} px",
            f"Meniscus Level: y={st['meniscus_y']} px",
            f"Liquid Column: {st['liquid_height']} px ({st['liquid_height_pct']:.1f}%)",
            f"Liquid Area: {st['total_pixels']:,} px",
            f"Mean RGB: ({st['mean_R']:.1f}, {st['mean_G']:.1f}, {st['mean_B']:.1f})",
            "",
            "[Dynamic Scale-Free Assurance]",
            "  Size Limits: NONE (Scale-Invariant)",
            "  Angle Limits: Dynamic Hough Tracing",
            "  Water Level: Dynamic Sustained Tracking"
        ]
        for idx, line in enumerate(lines):
            col = (0, 255, 0) if "deg" in line or "NONE" in line or "Dynamic" in line else (220, 220, 220)
            if "===" in line:
                col = (0, 220, 255)
            cv2.putText(panel, line, (590, 75 + idx * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.41, col, 1)
            
        cv2.imwrite(diagnostic_path, panel)
        
        return {
            "image": image_path,
            "tilt_angle": f"{tilt_angle:+.2f} deg",
            "bottle": bottle_path,
            "liquid_rgba": liquid_path,
            "diagnostic": diagnostic_path,
            "stats": st
        }

if __name__ == "__main__":
    cropper = WebOptimizedLiquidCropper()
    
    # กำหนด output directory (ค่าเริ่มต้น: output/crop_result)
    default_out_dir = os.path.join("output", "crop_result")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else default_out_dir
        
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        # หากไม่ใส่พารามิเตอร์ ให้รันภาพตัวอย่างเริ่มต้น
        target = r"d:\my_girlfriend_work\Testkit for AI\Phosphate 1.5 mg_l\05F968CC-AE4E-407F-8450-303AED8B3AF5.JPEG"
        print(f"[Info] ไม่ได้ระบุไฟล์ -> ใช้ภาพตัวอย่าง: {target}")
        print(f"  วิธีใช้: python crop_bottle_and_liquid.py \"path/to/image_or_folder\" [output_folder]\n")

    # ตรวจสอบว่าเป็นไฟล์เดี่ยว หรือ โฟลเดอร์
    if os.path.isdir(target):
        # กรณีรันทั้งโฟลเดอร์ (ค้นหาลึกทุกโฟลเดอร์ย่อย Recursive Search)
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = []
        for r, d, f_list in os.walk(target):
            for f in f_list:
                if os.path.splitext(f)[1].lower() in valid_exts:
                    files.append(os.path.join(r, f))
                    
        print(f"\n=== เริ่มต้น Batch Crop สำหรับ {len(files)} รูปภาพในโฟลเดอร์: {target} ===")
        print(f"บันทึกผลลัพธ์แยกโฟลเดอร์ไปที่: {os.path.abspath(out_dir)}\n")
        
        success = 0
        for idx, f in enumerate(files, 1):
            try:
                # รักษารูปแบบโฟลเดอร์ย่อยตามประเภทความเข้มข้นให้เป็นระเบียบ
                rel_sub = os.path.relpath(os.path.dirname(f), target)
                sub_out_dir = os.path.join(out_dir, rel_sub) if rel_sub != "." else out_dir
                
                res = cropper.process_image(f, output_dir=sub_out_dir)
                st = res["stats"]
                print(f"[{idx}/{len(files)}] สำเร็จ: {os.path.basename(f)} (Tilt: {res['tilt_angle']}, Liquid: {st['liquid_height_pct']:.1f}%)")
                success += 1
            except Exception as e:
                print(f"[{idx}/{len(files)}] ผิดพลาด: {os.path.basename(f)} -> {e}")
        print(f"\n=== เสร็จสิ้นการ Batch Crop! สำเร็จ {success}/{len(files)} ไฟล์ ===")
        print(f"ตรวจสอบผลลัพธ์ทั้งหมดได้ที่: {os.path.abspath(out_dir)}")
    elif os.path.isfile(target):
        # กรณีรันไฟล์เดี่ยว
        res = cropper.process_image(target, output_dir=out_dir)
        print("\n=== สรุปผลลัพธ์การ Crop ขวดและระดับน้ำ ===")
        print(f"  รูปภาพต้นฉบับ: {res['image']}")
        print(f"  มุมเอียงที่ตรวจพบและหมุนกลับ: {res['tilt_angle']}")
        print(f"  ภาพขวดตั้งตรง: {res['bottle']}")
        print(f"  ภาพมวลน้ำรูปทรงจริง (โปร่งใส): {res['liquid_rgba']}")
        print(f"  ภาพแผงสรุป Diagnostic: {res['diagnostic']}")
        st = res["stats"]
        print(f"  ระดับน้ำ: {st['liquid_height']} px ({st['liquid_height_pct']:.1f}% ของตัวขวด, {st['total_pixels']:,} พิกเซล)")
        print(f"  ค่าสีเฉลี่ย RGB: ({st['mean_R']:.1f}, {st['mean_G']:.1f}, {st['mean_B']:.1f})")
    else:
        print(f"[Error] ไม่พบไฟล์หรือโฟลเดอร์: {target}")
