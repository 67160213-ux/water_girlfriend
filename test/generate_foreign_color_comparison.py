import os
import sys
import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, r"d:\my_girlfriend_work\test")
from test_foreign_color_pipeline import dynamic_foreign_color_cleaner

def get_font(size=18, bold=False):
    win_dir = os.environ.get('WINDIR', 'C:\\Windows')
    font_file = 'tahomabd.ttf' if bold else 'tahoma.ttf'
    font_path = os.path.join(win_dir, 'Fonts', font_file)
    if not os.path.exists(font_path):
        font_path = os.path.join(win_dir, 'Fonts', 'tahoma.ttf')
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()

def draw_text_pil(cv_img, text, pos, font_size=18, color=(255, 255, 255), bold=False):
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = get_font(font_size, bold)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def make_checker(h, w, s=12):
    y, x = np.ogrid[:h, :w]
    c = ((x // s) ^ (y // s)) & 1
    return (c * 35 + 25).astype(np.uint8)

def render_on_checker(rgba):
    H, W = rgba.shape[:2]
    bg = cv2.merge([make_checker(H, W), make_checker(H, W), make_checker(H, W)])
    alpha = (rgba[:, :, 3] / 255.0)[:, :, None]
    out = (rgba[:, :, :3] * alpha + bg * (1.0 - alpha)).astype(np.uint8)
    return out

def run_showcase():
    test_sample = r"d:\my_girlfriend_work\test\crops_ready\Ammonia 0 mg_l 0 ppt\Ammonia 0 mg_l 0 ppt_02_clean.png"
    orig_rgba = cv2.imread(test_sample, cv2.IMREAD_UNCHANGED)
    H, W = orig_rgba.shape[:2]
    
    trimmed_rgba, harmonized_rgba, foreign_mask, stats = dynamic_foreign_color_cleaner(orig_rgba)
    print("Clean stats:", stats)
    
    # Render full images on checker
    orig_disp = render_on_checker(orig_rgba)
    trim_disp = render_on_checker(trimmed_rgba)
    harm_disp = render_on_checker(harmonized_rgba)
    
    mask_disp = orig_disp.copy()
    mask_disp[foreign_mask > 0] = [0, 0, 255] # Red foreign mask
    
    # Full height comparison panel
    disp_h = 600
    sc = disp_h / float(H)
    disp_w = int(W * sc)
    
    f1 = cv2.resize(orig_disp, (disp_w, disp_h))
    f2 = cv2.resize(mask_disp, (disp_w, disp_h))
    f3 = cv2.resize(trim_disp, (disp_w, disp_h))
    f4 = cv2.resize(harm_disp, (disp_w, disp_h))
    
    # Zoom Meniscus Top (first 16% height)
    zoom_top_h = int(H * 0.16)
    zt1 = render_on_checker(orig_rgba[:zoom_top_h, :])
    zt2 = render_on_checker(orig_rgba[:zoom_top_h, :])
    zt2[foreign_mask[:zoom_top_h, :] > 0] = [0, 0, 255]
    zt3 = render_on_checker(trimmed_rgba[:zoom_top_h, :])
    zt4 = render_on_checker(harmonized_rgba[:zoom_top_h, :])
    
    zt_disp_w = 340
    zt_disp_h = int(zoom_top_h * (zt_disp_w / float(W)))
    ztop_row = np.hstack([
        cv2.resize(zt1, (zt_disp_w, zt_disp_h)),
        cv2.resize(zt2, (zt_disp_w, zt_disp_h)),
        cv2.resize(zt3, (zt_disp_w, zt_disp_h)),
        cv2.resize(zt4, (zt_disp_w, zt_disp_h))
    ])
    
    # Zoom Base Bottom (last 18% height)
    zoom_bot_y = int(H * 0.82)
    zb1 = render_on_checker(orig_rgba[zoom_bot_y:, :])
    zb2 = render_on_checker(orig_rgba[zoom_bot_y:, :])
    zb2[foreign_mask[zoom_bot_y:, :] > 0] = [0, 0, 255]
    zb3 = render_on_checker(trimmed_rgba[zoom_bot_y:, :])
    zb4 = render_on_checker(harmonized_rgba[zoom_bot_y:, :])
    
    zb_disp_w = 340
    zb_disp_h = int((H - zoom_bot_y) * (zb_disp_w / float(W)))
    zbot_row = np.hstack([
        cv2.resize(zb1, (zb_disp_w, zb_disp_h)),
        cv2.resize(zb2, (zb_disp_w, zb_disp_h)),
        cv2.resize(zb3, (zb_disp_w, zb_disp_h)),
        cv2.resize(zb4, (zb_disp_w, zb_disp_h))
    ])
    
    full_row = np.hstack([f1, f2, f3, f4])
    total_w = ztop_row.shape[1]
    full_sc = total_w / float(full_row.shape[1])
    full_row = cv2.resize(full_row, (total_w, int(full_row.shape[0] * full_sc)))
    
    # Build complete diagnostic showcase canvas with PIL Thai Typography
    header = np.zeros((75, total_w, 3), dtype=np.uint8) + 20
    col_w = total_w // 4
    header = draw_text_pil(header, "[1] Original (ภาพเดิมมีสีเพี้ยน/ฉากหลัง)", (15, 25), 15, (255, 255, 255), True)
    header = draw_text_pil(header, "[2] Foreign Mask (ตรวจจับสีแปลกปลอม)", (col_w + 10, 25), 15, (0, 100, 255), True)
    header = draw_text_pil(header, "[3] Approach 1: Trim (ตัดโปร่งใส Alpha=0)", (col_w*2 + 10, 25), 15, (0, 255, 255), True)
    header = draw_text_pil(header, "[4] Approach 2: Harmonize (เกลี่ยสีน้ำแท้)", (col_w*3 + 10, 25), 15, (0, 255, 120), True)
    
    divider1 = np.zeros((36, total_w, 3), dtype=np.uint8) + 32
    divider1 = draw_text_pil(divider1, "--- TOP MENISCUS ZOOM (ซูมผิวน้ำด้านบน: สังเกตสีฟ้า/เทาของ ColorChecker ที่มุมซ้าย-ขวาหายไปเรียบร้อย) ---", 
                             (20, 7), 15, (0, 220, 255), True)
                
    divider2 = np.zeros((36, total_w, 3), dtype=np.uint8) + 32
    divider2 = draw_text_pil(divider2, "--- BOTTOM BASE ZOOM (ซูมก้นขวดด้านล่าง: สังเกตเงามืดขอบแก้ว และจุดสีฟ้าที่ปลายก้นถูกเคลียร์สะอาด) ---", 
                             (20, 7), 15, (0, 220, 255), True)
                
    divider3 = np.zeros((36, total_w, 3), dtype=np.uint8) + 32
    divider3 = draw_text_pil(divider3, "--- FULL LIQUID BODY COMPARISON (เปรียบเทียบมวลน้ำเต็มขวดทั้งใบ: เลือกได้ทั้งแบบตัดทิ้ง หรือแบบเกลี่ยสีเนียน) ---", 
                             (20, 7), 15, (0, 220, 255), True)

    canvas = np.vstack([
        header,
        divider1,
        ztop_row,
        divider2,
        zbot_row,
        divider3,
        full_row
    ])
    
    out_file = r"d:\my_girlfriend_work\test\foreign_color_solutions_showcase.png"
    cv2.imwrite(out_file, canvas)
    print(f"Showcase saved successfully to: {out_file}")

if __name__ == "__main__":
    run_showcase()
