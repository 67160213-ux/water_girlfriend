"""
Script สำหรับคัดลอกเฉพาะไฟล์ที่ลงท้ายด้วย _liquid_exact_shape.png
จาก dataset_cropped ไปยัง output/liquid_ready_to_use โดยรักษาโครงสร้างโฟลเดอร์เดิม (train/test)
"""

import os
import shutil
import time

def copy_liquid_files(src_root: str = "dataset_cropped", dst_root: str = os.path.join("output", "liquid_ready_to_use")):
    if not os.path.exists(src_root):
        print(f"[Error] ไม่พบโฟลเดอร์ต้นทาง: {os.path.abspath(src_root)}")
        return
        
    os.makedirs(dst_root, exist_ok=True)
    
    print(f"=== เริ่มต้นคัดลอกไฟล์ _liquid_exact_shape ===")
    print(f"  ต้นทาง (Source):      {os.path.abspath(src_root)}")
    print(f"  ปลายทาง (Destination): {os.path.abspath(dst_root)}\n")
    
    files_to_copy = []
    for dirpath, dirs, filenames in os.walk(src_root):
        for f in filenames:
            if f.endswith("_liquid_exact_shape.png"):
                src_path = os.path.join(dirpath, f)
                rel_path = os.path.relpath(src_path, src_root)
                dst_path = os.path.join(dst_root, rel_path)
                files_to_copy.append((src_path, dst_path))
                
    total = len(files_to_copy)
    print(f"พบไฟล์ที่ตรงเงื่อนไขทั้งหมด: {total:,} ไฟล์")
    if total == 0:
        print("[Warning] ไม่พบไฟล์ที่ลงท้ายด้วย _liquid_exact_shape.png")
        return
        
    t0 = time.time()
    for idx, (src_f, dst_f) in enumerate(files_to_copy, 1):
        os.makedirs(os.path.dirname(dst_f), exist_ok=True)
        shutil.copy2(src_f, dst_f)
        if idx % 300 == 0 or idx == total:
            print(f"  คัดลอกแล้ว: [{idx:,}/{total:,}] ไฟล์...")
            
    elapsed = time.time() - t0
    print(f"\n[สำเร็จ] คัดลอกไฟล์ทั้งหมด {total:,} ไฟล์ ไปยัง {dst_root} เรียบร้อยแล้ว (ใช้เวลา {elapsed:.2f} วินาที)")

if __name__ == "__main__":
    copy_liquid_files()

