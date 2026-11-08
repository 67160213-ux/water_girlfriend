"""
Inference Script สำหรับนำภาพถ่ายขวดตัวอย่างภาพใหม่มาทำนายความเข้มข้นสารละลาย
รองรับการประมวลผลทั้ง CPU และ GPU (CUDA) อัตโนมัติ พร้อมสกัดลำน้ำสมบูรณ์ 100%
"""

import sys
import os
import pickle
import cv2
import numpy as np
import torch

from water_quality_pipeline import (
    ColorCalibrator,
    YOLOBottleDetector,
    HorizontalProjectionSegmenter,
    LiquidROIExtractor,
    ColorFeatureExtractor,
    CropResultSaver,
    load_standard_image,
    DEVICE
)

def predict_sample(image_path: str, model_path: str, save_crop_result: bool = True, output_dir: str = "inference_results") -> dict:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ภาพ: {image_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {model_path}")
        
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        
    model = model_data["model"]
    feature_cols = model_data["feature_cols"]
    blank_ref = model_data.get("blank_ref", model_data.get("blank_linear_ref", None))
    lod = model_data.get("lod", np.nan)
    loq = model_data.get("loq", np.nan)
    
    img = load_standard_image(image_path)
    if img is None:
        raise ValueError("ไม่สามารถอ่านภาพได้")
        
    # 0. Color Calibration & White Balance
    wb_img, ref_box = ColorCalibrator.apply_white_balance(img)
    
    # 1. YOLO Detection with Glass Wall Refinement
    detector = YOLOBottleDetector(device=DEVICE)
    det = detector.detect_and_crop(wb_img)
    
    # 2. Horizontal Projection Profile
    segmenter = HorizontalProjectionSegmenter()
    seg = segmenter.segment(det['crop'])
    seg['deskew_angle'] = det['deskew_angle']
    
    # 3. Full Liquid Column ROI & Geometry Mask
    roi_extractor = LiquidROIExtractor()
    l_roi, mask, roi_box = roi_extractor.extract_liquid_roi(det['crop'], seg)
    
    # 4. Color Feature Extraction from all liquid pixels
    feats = ColorFeatureExtractor.extract_features(l_roi, mask, blank_ref)
    
    # 5. Model Prediction
    X = np.array([[feats[col] for col in feature_cols]])
    pred_conc = float(model.predict(X).flatten()[0])
    pred_conc = max(0.0, pred_conc)
    
    # Quality Control Evaluation
    qc_status = "PASS"
    qc_notes = []
    if not np.isnan(lod) and pred_conc < lod:
        qc_notes.append(f"ต่ำกว่าขีดจำกัดการตรวจหา (Below LOD: {lod:.2f} mg/L)")
    elif not np.isnan(loq) and pred_conc < loq:
        qc_notes.append(f"ตรวจพบแต่ต่ำกว่าขีดจำกัดการหาปริมาณ (Between LOD and LOQ: {loq:.2f} mg/L)")
        
    if feats["vertical_drift"] > 25.0:
        qc_notes.append("ตรวจพบการไล่ระดับแสงผิดปกติหรืออาจมีการตกตะกอน (High Vertical Drift)")
        qc_status = "WARNING"
        
    # บันทึกภาพ Diagnostic Crop Result หากเปิดใช้งาน
    saved_panel_path = None
    if save_crop_result:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        saved_panel_path = os.path.join(output_dir, f"result_{base_name}.jpg")
        CropResultSaver.save_diagnostic_panel(
            save_path=saved_panel_path,
            orig_img=img,
            bottle_crop=det['crop'],
            seg_dict=seg,
            liquid_roi=l_roi,
            mask=mask,
            vial_box=det['bbox'],
            ref_box=ref_box,
            roi_box=roi_box,
            true_conc=None,
            pred_conc=pred_conc,
            abs_max=feats['abs_max'],
            liquid_pixels=feats['liquid_pixels']
        )

    return {
        "image_path": image_path,
        "device_used": f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU",
        "predicted_concentration_mg_l": round(pred_conc, 3),
        "absorbance_max": round(feats["abs_max"], 4),
        "CIE_L": round(feats["lab_L"], 2),
        "CIE_a": round(feats["lab_a"], 2),
        "CIE_b": round(feats["lab_b"], 2),
        "total_liquid_pixels": feats["liquid_pixels"],
        "deskew_angle_deg": round(det["deskew_angle"], 2),
        "qc_status": qc_status,
        "qc_notes": "; ".join(qc_notes) if qc_notes else "ปกติ",
        "diagnostic_panel_saved": saved_panel_path
    }

if __name__ == '__main__':
    if len(sys.argv) >= 3:
        sample_img = os.path.normpath(sys.argv[1])
        model_file = os.path.normpath(sys.argv[2])
    else:
        sample_img = r'd:\my_girlfriend_work\dataset_split\test\Phosphate 1.5 mg_l\05F968CC-AE4E-407F-8450-303AED8B3AF5.JPEG'
        model_file = r'd:\my_girlfriend_work\saved_models\Phosphate_pls.pkl'
        
    if os.path.exists(sample_img) and os.path.exists(model_file):
        res = predict_sample(sample_img, model_file)
        print("\n=== ผลการทำนายตัวอย่างทดสอบเดี่ยว ===")
        for k, v in res.items():
            print(f"  {k}: {v}")
    else:
        print(f"ไม่พบไฟล์: {sample_img} หรือ {model_file}")
