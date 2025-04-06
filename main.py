from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from deep_translator import GoogleTranslator
from inference_sdk import InferenceHTTPClient
import httpx
import os

app = FastAPI()

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="VJmICXJRnj9bYjhmsktT"
)

clothing_to_brands = {
    "shirt": ["Nike", "Adidas", "Puma", "Gucci", "Zara", "H&M"],
    "pants": ["Levi's", "Diesel", "Wrangler", "Gucci", "Armani"],
    "shoes": ["Nike", "Adidas", "Reebok", "Puma", "Converse", "Vans"],
    "shorts": ["Nike", "Adidas", "Gucci", "Zara", "H&M"],
    "hoodie": ["Champion", "Nike", "Adidas", "Puma", "H&M"],
    "hat": ["New Era", "Nike", "Adidas", "Puma", "Gucci"],
    "jacket": ["The North Face", "Columbia", "Zara", "Gucci", "H&M"],
    "dress": ["Zara", "H&M", "Gucci", "Versace", "Chanel"],
    "t-shirt": ["Nike", "Adidas", "Gucci", "Uniqlo", "H&M"],
    "sneakers": ["Nike", "Adidas", "Reebok", "Puma", "New Balance"]
}

brand_store_links = {
    "Nike": "https://www.nike.com",
    "Adidas": "https://www.adidas.com",
    "Puma": "https://www.puma.com",
    "Gucci": "https://www.gucci.com",
    "Zara": "https://www.zara.com",
    "H&M": "https://www2.hm.com",
    "Levi's": "https://www.levi.com",
    "Diesel": "https://global.diesel.com",
    "Wrangler": "https://www.wrangler.com",
    "Champion": "https://www.champion.com",
    "New Era": "https://www.neweracap.com",
    "The North Face": "https://www.thenorthface.com",
    "Columbia": "https://www.columbia.com",
    "Versace": "https://www.versace.com",
    "Chanel": "https://www.chanel.com",
    "Uniqlo": "https://www.uniqlo.com",
    "Reebok": "https://www.reebok.com",
    "Converse": "https://www.converse.com",
    "Vans": "https://www.vans.com",
    "New Balance": "https://www.newbalance.com"
}

async def get_language_by_ip(ip: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://ip-api.com/json/{ip}")
            data = response.json()
            country_code = data.get("countryCode", "").lower()

            language_map = {
                "de": "de", "nl": "nl", "es": "es", "it": "it", "fr": "fr",
                "en": "en", "ru": "ru", "ua": "uk", "in": "hi", "cn": "zh-CN",
                "jp": "ja", "kr": "ko", "vn": "vi", "no": "no", "is": "is",
                "ch": "de", "ae": "ar", "il": "iw", "kz": "kk"
            }

            return language_map.get(country_code, "en")
    except Exception:
        return "en"

def translate_class(class_name: str, target_lang: str):
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(class_name)
    except Exception:
        return class_name

@app.post("/detect")
async def detect(request: Request, file: UploadFile = File(...), lang: str = None):
    try:
        if not lang:
            client_host = request.client.host
            lang = await get_language_by_ip(client_host)

        result = CLIENT.infer(file, model_id="clothing-detection-sc9m9/1")
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