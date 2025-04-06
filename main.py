from fastapi import FastAPI, File, UploadFile, Request, Query
from fastapi.responses import JSONResponse
from inference_sdk import InferenceHTTPClient
from deep_translator import GoogleTranslator
import shutil
import uuid
import os
import httpx

app = FastAPI()

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="YOUR_API_KEY"
)

# Получение языка по IP
async def get_language_by_ip(ip: str) -> str:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://ipapi.co/{ip}/json/")
            data = response.json()
            country = data.get("country", "").lower()

        country_to_lang = {
            "de": "de",     # Германия
            "nl": "nl",     # Нидерланды
            "es": "es",     # Испания
            "fr": "fr",     # Франция
            "it": "it",     # Италия
            "us": "en",     # США
            "gb": "en",     # Англия
            "ru": "ru",     # Россия
            "ua": "uk",     # Украина
            "in": "hi",     # Индия
            "cn": "zh-CN",  # Китай
            "jp": "ja",     # Япония
            "kr": "ko",     # Корея
            "vn": "vi",     # Вьетнам
            "no": "no",     # Норвегия
            "is": "is",     # Исландия
            "ch": "de",     # Швейцария → немецкий
            "il": "he",     # Израиль → иврит
            "sa": "ar",     # Саудовская Аравия → арабский
            "ae": "ar",     # ОАЭ → арабский
            "eg": "ar",     # Египет → арабский
            "kz": "kk"      # Казахстан
        }
        return country_to_lang.get(country, "en")  # по умолчанию английский
    except:
        return "en"

# Перевод названия класса
def translate_class(cls: str, target_lang: str) -> str:
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(cls)
    except:
        return cls

@app.get("/")
def read_root():
    return {"message": "Clothing Detection API with Translation is Running!"}

@app.post("/detect")
async def detect(request: Request, file: UploadFile = File(...), lang: str = Query(None)):
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if not lang:
            client_host = request.client.host
            lang = await get_language_by_ip(client_host)

        result = CLIENT.infer(temp_filename, model_id="clothing-detection-scn9m/1")
        predictions = result.get("predictions", [])
        items = []

        for prediction in predictions:
            cls = prediction["class"]
            conf = prediction["confidence"]
            translated_cls = translate_class(cls, target_lang=lang)
            items.append({
                "class": translated_cls,
                "confidence": f"{round(conf * 100)}%"
            })

        return JSONResponse(content={"results": items, "language": lang})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.remove(temp_filename)