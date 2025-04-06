
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from inference_sdk import InferenceHTTPClient
import uvicorn
import shutil
import os

app = FastAPI()

# API-клиент Roboflow (скрыт от клиента, безопасно)
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="VJmICXJRnj9bYjhmsktT"  # Заменить на свой рабочий ключ
)

@app.post("/detect")
async def detect_image(image: UploadFile = File(...)):
    try:
        # Сохраняем временно файл
        temp_path = f"temp_{image.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Отправляем на Roboflow
        result = CLIENT.infer(temp_path, model_id="clothing-detection-scn9m/1")

        # Удаляем временный файл
        os.remove(temp_path)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Для локального теста
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
