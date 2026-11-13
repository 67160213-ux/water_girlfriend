"""
สคริปต์ Re-crop ชุดข้อมูลทั้งหมด (1,390 ภาพ) ด้วยอัลกอริทึมใหม่ล่าสุด:
1. YOLO ตรวจจับขวด
2. ตัดฝาขวดออกก่อน (Cut Bottle Cap First)
3. ปรับขวดตั้งตรง 90 องศา (Automatic Deskew)
4. ฟิตเส้นโค้งผิวน้ำ Meniscus และก้นขวด 3 มิติเต็ม 100% (Curved Meniscus Profile)
5. บันทึกผลลัพธ์ลงใน dataset_cropped และซิงค์ _liquid_exact_shape.png ไปที่ output/liquid_ready_to_use โดยอัตโนมัติ
"""

import os
import sys
import shutil
import time
from crop_bottle_and_liquid import WebOptimizedLiquidCropper

def recrop_entire_dataset(
    input_root: str = "dataset_split",
    crop_output: str = "dataset_cropped",
    ready_to_use_dir: str = os.path.join("output", "liquid_ready_to_use")
):
    if not os.path.exists(input_root):
        print(f"[Error] ไม่พบโฟลเดอร์ภาพต้นฉบับ: {os.path.abspath(input_root)}")
        return
        
    os.makedirs(crop_output, exist_ok=True)
    os.makedirs(ready_to_use_dir, exist_ok=True)
    
    # รวบรวมไฟล์ภาพต้นทางทั้งหมด
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_files = []
    for root, dirs, files in os.walk(input_root):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_exts:
                all_files.append(os.path.join(root, f))
                
    total = len(all_files)
    print(f"======================================================================")
    print(f"     เริ่มต้น RE-CROP ภาพทั้งหมด {total:,} ภาพ ด้วยโมเดลใหม่ล่าสุด")
    print(f"======================================================================")
    print(f"• โฟลเดอร์ภาพต้นทาง:       {os.path.abspath(input_root)}")
    print(f"• โฟลเดอร์เก็บภาพ Crop เต็ม: {os.path.abspath(crop_output)}")
    print(f"• โฟลเดอร์เก็บภาพพร้อมใช้:   {os.path.abspath(ready_to_use_dir)}")
    print(f"• ฟังก์ชันใหม่ที่เปิดใช้งาน:")
    print(f"   [1] Cut Bottle Cap First  (ตัดฝาขวดออกก่อน)")
    print(f"   [2] Automatic 90° Deskew  (หมุนตั้งตรง)")
    print(f"   [3] Curved Meniscus Arc   (ฟิตเส้นโค้งผิวน้ำ ไร้รอยแหว่ง)")
    print(f"======================================================================\n")
    
    cropper = WebOptimizedLiquidCropper()
    
    t0 = time.time()
    success = 0
    errors = 0
    
    for idx, img_path in enumerate(all_files, 1):
        rel_dir = os.path.relpath(os.path.dirname(img_path), input_root)
        target_sub_dir = os.path.join(crop_output, rel_dir) if rel_dir != "." else crop_output
        ready_sub_dir = os.path.join(ready_to_use_dir, rel_dir) if rel_dir != "." else ready_to_use_dir
        
        try:
            res = cropper.process_image(img_path, output_dir=target_sub_dir)
            
            # คัดลอกไฟล์ _liquid_exact_shape.png ไปยัง output/liquid_ready_to_use อัตโนมัติ
            liquid_png = res["liquid_rgba"]
            os.makedirs(ready_sub_dir, exist_ok=True)
            dst_ready_png = os.path.join(ready_sub_dir, os.path.basename(liquid_png))
            shutil.copy2(liquid_png, dst_ready_png)
            
            success += 1
            if idx % 100 == 0 or idx == total:
                elapsed = time.time() - t0
                speed = idx / elapsed if elapsed > 0 else 0
                st = res["stats"]
                print(f"[{idx:,}/{total:,}] สำเร็จ ({speed:.1f} ภาพ/วิ) - {os.path.basename(img_path)} (Liquid: {st['liquid_height_pct']:.1f}%)")
        except Exception as e:
            errors += 1
            print(f"[{idx:,}/{total:,}] ผิดพลาด: {os.path.basename(img_path)} -> {e}")
            
    total_elapsed = time.time() - t0
    print(f"\n======================================================================")
    print(f"                    เสร็จสิ้นการ RE-CROP ทั้งหมด!")
    print(f"======================================================================")
    print(f"• สำเร็จ:   {success:,} / {total:,} ภาพ")
    if errors > 0:
        print(f"• ผิดพลาด: {errors:,} ภาพ")
        print(f"• ตรวจสอบสาเหตุข้อผิดพลาดและ Traceback ได้ที่:")
        print(f"   - Text Log:  {os.path.abspath(os.path.join('output', 'crop_failed.log'))}")
        print(f"   - JSON Line: {os.path.abspath(os.path.join('output', 'crop_failed.jsonl'))}")
    print(f"• เวลาที่ใช้: {total_elapsed:.1f} วินาที (เฉลี่ย {total/total_elapsed:.1f} ภาพ/วินาที)")
    print(f"• ภาพมวลน้ำพร้อมใช้งานทั้งหมดถูกบันทึกไปที่: {os.path.abspath(ready_to_use_dir)}")
    print(f"======================================================================")

if __name__ == "__main__":
    recrop_entire_dataset()

