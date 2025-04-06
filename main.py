from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from inference_sdk import InferenceHTTPClient
import shutil
import uuid
import os

app = FastAPI()

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="VJmICXJRnj9bYjhmsktT"
)

translation_dict = {
    "T-shirt": "Футболка",
    "Jacket": "Куртка",
    "Dress": "Платье",
    "Jeans": "Джинсы",
}

@app.get("/")
def read_root():
    return {"message": "Clothing API is running!"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = CLIENT.infer(temp_filename, model_id="clothing-detection-scn9m/1")
        predictions = result.get("predictions", [])
        items = []

        for prediction in predictions:
            cls = prediction["class"]
            conf = prediction["confidence"]
            translated_cls = translation_dict.get(cls, cls)
            items.append({
                "class": translated_cls,
                "confidence": round(conf, 2)
            })

        return JSONResponse(content={"results": items})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.remove(temp_filename)