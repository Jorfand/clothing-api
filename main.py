from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from deep_translator import GoogleTranslator
import httpx
from roboflow import Roboflow
import os
from typing import Optional

app = FastAPI()

rf = Roboflow(api_key="VJmICXJRnj9bYjhmsktT")
project = rf.workspace().project("clothing-detection-sc9m1")
model = project.version(1).model

CLIENT = model

clothing_to_brands = {
    "shirt": ["Gucci", "Nike", "Adidas"],
    "pants": ["Zara", "H&M", "Levis"],
    "jacket": ["North Face", "Columbia", "Canada Goose"],
    # Добавь больше классов по аналогии
}

brand_store_links = {
    "Gucci": "https://www.gucci.com/",
    "Nike": "https://www.nike.com/",
    "Adidas": "https://www.adidas.com/",
    "Zara": "https://www.zara.com/",
    "H&M": "https://www2.hm.com/",
    "Levis": "https://www.levi.com/",
    "North Face": "https://www.thenorthface.com/",
    "Columbia": "https://www.columbia.com/",
    "Canada Goose": "https://www.canadagoose.com/"
    # Добавь больше брендов и ссылок по аналогии
}

def get_language_by_ip(ip: str) -> str:
    try:
        response = httpx.get(f"https://ipinfo.io/{ip}/json")
        if response.status_code == 200:
            country = response.json().get("country", "EN")
            return {
                "DE": "de", "NL": "nl", "ES": "es", "FR": "fr", "IT": "it",
                "GB": "en", "US": "en", "UA": "uk", "RU": "ru", "IN": "hi",
                "CN": "zh", "JP": "ja", "KR": "ko", "VN": "vi", "NO": "no",
                "IS": "is", "CH": "de", "AE": "ar", "IL": "he", "KZ": "kk"
            }.get(country, "en")
    except:
        return "en"

def translate_class(class_name: str, target_lang: str) -> str:
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(class_name)
    except:
        return class_name

@app.post("/detect")
async def detect(request: Request, file: UploadFile = File(...), lang: Optional[str] = None):
    try:
        if not lang:
            client_host = request.client.host
            lang = await get_language_by_ip(client_host)

        contents = await file.read()
        with open("temp.jpg", "wb") as f:
            f.write(contents)

        result = CLIENT.predict("temp.jpg", confidence=40, overlap=30).json()
        predictions = result.get("predictions", [])
        items = []

        for prediction in predictions:
            cls = prediction["class"]
            conf = prediction["confidence"]
            translated_cls = translate_class(cls, target_lang=lang)
            brands = clothing_to_brands.get(cls.lower(), [])
            stores = []
            for brand in brands:
                stores.append({
                    "brand": brand,
                    "store_link": brand_store_links.get(brand, f"https://www.google.com/search?q={brand}+{cls}")
                })
            items.append({
                "class": translated_cls,
                "confidence": f"{round(conf * 100)}%",
                "brands": brands,
                "stores": stores
            })

        return JSONResponse(content={"results": items, "language": lang})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})