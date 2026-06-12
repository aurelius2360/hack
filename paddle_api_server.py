import os
import sys
import json
from pathlib import Path
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np

# Apply dynamic python patch to bypass the paddle optimization level attribute error
try:
    import paddle.inference as paddle_inf
    if not hasattr(paddle_inf.Config, "set_optimization_level"):
        paddle_inf.Config.set_optimization_level = lambda self, level: None
        print("[Patch] Applied set_optimization_level patch to paddle.inference.Config")
except Exception as e:
    print("[Patch] Config patch skipped:", e)

try:
    import paddle.base.libpaddle as libpaddle
    if hasattr(libpaddle, "AnalysisConfig") and not hasattr(libpaddle.AnalysisConfig, "set_optimization_level"):
        libpaddle.AnalysisConfig.set_optimization_level = lambda self, level: None
        print("[Patch] Applied set_optimization_level patch to libpaddle.AnalysisConfig")
except Exception as e:
    print("[Patch] AnalysisConfig patch skipped:", e)

# Import PaddleOCR and PaddleOCRVL after patching
from paddleocr import PaddleOCR, PaddleOCRVL
import torch

app = FastAPI(title="Paddle OCR & Layout Server")

def create_text_ocr(lang="en"):
    use_gpu = torch.cuda.is_available()
    option_sets = [
        {"use_doc_orientation_classify": False, "use_doc_unwarping": False, "use_textline_orientation": False, "lang": lang, "use_gpu": use_gpu},
        {"use_angle_cls": False, "lang": lang, "use_gpu": use_gpu},
        {"lang": lang, "use_gpu": use_gpu}
    ]
    for options in option_sets:
        try:
            return PaddleOCR(**options)
        except Exception:
            continue
    if lang != "en":
        return create_text_ocr(lang="en")
    return PaddleOCR(use_angle_cls=False, lang="en", use_gpu=use_gpu)

@app.post("/api/layout")
async def api_layout(file: UploadFile = File(...)):
    """
    Performs layout analysis on the uploaded document image and returns the layout JSON structure.
    """
    temp_img_path = Path("temp_api_layout.png")
    with open(temp_img_path, "wb") as f:
        f.write(await file.read())
        
    try:
        print(f"[API] Running PaddleOCRVL layout parser on {file.filename}...")
        layout_pipeline = PaddleOCRVL()
        output = layout_pipeline.predict(input=str(temp_img_path))
        structured_output = layout_pipeline.restructure_pages(list(output))
        
        if not structured_output:
            raise HTTPException(status_code=500, detail="Layout analysis failed")
            
        # Save layout JSON to disk temporarily and read it back
        temp_json_dir = Path("./temp_json_out")
        temp_json_dir.mkdir(exist_ok=True)
        structured_output[0].save_to_json(save_path=str(temp_json_dir))
        
        json_files = list(temp_json_dir.glob("*.json"))
        if not json_files:
            raise HTTPException(status_code=500, detail="Failed to save layout JSON")
            
        with open(json_files[0], "r", encoding="utf-8") as f:
            layout_data = json.load(f)
            
        # Clean up
        json_files[0].unlink()
        try:
            temp_json_dir.rmdir()
        except Exception:
            pass
            
        return layout_data
    except Exception as e:
        print(f"[API] Error in api_layout: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_img_path.exists():
            temp_img_path.unlink()

@app.post("/api/ocr")
async def api_ocr(file: UploadFile = File(...), lang: str = "en"):
    """
    Performs OCR character detection on the uploaded cropped block image and returns the raw results.
    """
    temp_crop_path = Path("temp_api_crop.png")
    with open(temp_crop_path, "wb") as f:
        f.write(await file.read())
        
    try:
        text_ocr = create_text_ocr(lang=lang)
        raw_result = text_ocr.ocr(str(temp_crop_path), cls=False)
        return {"raw_result": raw_result}
    except Exception as e:
        print(f"[API] Error in api_ocr: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_crop_path.exists():
            temp_crop_path.unlink()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8011)
