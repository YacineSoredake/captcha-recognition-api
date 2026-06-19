from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import easyocr
from PIL import Image
from collections import Counter
import io
import re
import numpy as np

reader = easyocr.Reader(['en'], gpu=False)

app = FastAPI(title="OCR Number Detection API")


def extract_number_from_image(image_file):
    try:
        
        img = Image.open(io.BytesIO(image_file))
        img_array = np.array(img)
 
        results = reader.readtext(img_array, allowlist='0123456789')
        
        if not results:
            return None
        
        detected_numbers = []
        for (bbox, text, confidence) in results:
            cleaned = re.sub(r'[^0-9]', '', text)
            if cleaned:
                detected_numbers.append((cleaned, confidence))
        
        if not detected_numbers:
            return None
        
        best_number = max(detected_numbers, key=lambda x: x[1])[0]
        
        if len(best_number) == 3:
            return best_number
        
        return best_number
    
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")


def find_most_common_number(numbers):
    """
    Find the most common number from a list.
    
    Args:
        numbers: List of numbers
    
    Returns:
        dict: Contains the most common number and statistics
    """
    if not numbers:
        return {
            "most_common_number": None,
            "count": 0
        }
    
    counter = Counter(numbers)
    most_common = counter.most_common(1)[0]
    
    return {
        "most_common_number": most_common[0],
        "count": most_common[1]
    }


@app.post("/recognize")
async def recognizeBatch(files: list[UploadFile] = File(...)):
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        extracted_numbers = []          
        indexed_results = []            
        failed_images = []
        
        for idx, file in enumerate(files):
            try:
                contents = await file.read()

                try:
                    Image.open(io.BytesIO(contents))
                except Exception:
                    failed_images.append({
                        "filename": file.filename or "unknown",
                        "error": "Invalid image file"
                    })
                    continue
                
                number = extract_number_from_image(contents)

                if number:
                    extracted_numbers.append(number)
                    indexed_results.append((number, idx))   
                else:
                    failed_images.append({
                        "filename": file.filename or "unknown",
                        "error": "No number detected"
                    })

            except Exception as e:
                failed_images.append({
                    "filename": file.filename or "unknown",
                    "error": str(e)
                })

        result = find_most_common_number(extracted_numbers)

        indices = []

        if result["most_common_number"] is not None:
            most_common = result["most_common_number"]

            indices = [
                idx + 1 for (num, idx) in indexed_results
                if num == most_common
            ]


        return JSONResponse(content={
            "success": True,
            "result": result,
            "indices_containing_number": indices ,
            "failed_images": failed_images if failed_images else None
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "OCR Number Detection API is running",
        "endpoints": {
            "POST /detect-most-common-number/": "Upload multiple images to find most common number"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)