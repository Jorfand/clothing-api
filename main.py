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
    api_key="VJmICXJRnj9bYjhmsktT"
)

# Универсальный список брендов
ALL_BRANDS = [
    "Gucci", "Chanel", "Dior", "Nike", "Adidas", "Puma", "Reebok", "Fila", "Zara",
    "H&M", "New Yorker", "ASICS", "Saucony", "Palombier", "Givenchy", "Saint Laurent",
    "Valentino", "Prada", "Fendi", "Versace", "Balenciaga", "Kenzo", "Dolce&Gabbana",
    "Pierre Cardin", "Polo Ralph Lauren", "Gant", "Stone Island", "Brioni", "Bugatti",
    "Hugo", "Boss", "Burberry", "Canada Goose", "Lacoste", "Emporio Armani", "Moncler",
    "Balmain", "Columbia", "Hermès", "Bottega Veneta", "Louis Vuitton", "Dr. Martens",
    "Crocs", "Salomon", "Timberland", "New Balance", "Converse", "Ecco", "Vans",
    "Breguet", "Rolex", "Omega", "Cartier", "Blancpain"
]

# Универсальный список магазинов (пример)
ALL_STORES = [
    "Zalando", "Farfetch", "ASOS", "AliExpress", "Amazon", "Shein", "Uniqlo", "Zara",
    "Nike", "Adidas", "H&M", "Gucci", "Chanel", "Dior"
]

# Словарь: категория одежды -> список брендов (все бренды)
clothing_to_brands = {
    "jacket": ALL_BRANDS, "coat": ALL_BRANDS, "shirt": ALL_BRANDS, "t-shirt": ALL_BRANDS,
    "hoodie": ALL_BRANDS, "sweater": ALL_BRANDS, "shorts": ALL_BRANDS, "pants": ALL_BRANDS,
    "jeans": ALL_BRANDS, "leggings": ALL_BRANDS, "underwear": ALL_BRANDS, "socks": ALL_BRANDS,
    "bra": ALL_BRANDS, "hat": ALL_BRANDS, "cap": ALL_BRANDS, "beanie": ALL_BRANDS,
    "glasses": ALL_BRANDS, "skirt": ALL_BRANDS, "dress": ALL_BRANDS, "suit": ALL_BRANDS,
    "blazer": ALL_BRANDS, "pajamas": ALL_BRANDS, "boots": ALL_BRANDS, "sneakers": ALL_BRANDS,
    "shoes": ALL_BRANDS, "loafers": ALL_BRANDS, "slippers": ALL_BRANDS, "sandals": ALL_BRANDS,
    "heels": ALL_BRANDS, "flip-flops": ALL_BRANDS
}

# Получение языка по IP
async def get_language_by_ip(ip: str) -> str:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://ipapi.co/{ip}/json/")
            data = response.json()
            country = data.get("country", "").lower()
        return {
            "de": "de", "nl": "nl", "es": "es", "fr": "fr", "it": "it", "us": "en", "gb": "en",
            "ru": "ru", "ua": "uk", "in": "hi", "cn": "zh-CN", "jp": "ja", "kr": "ko", "vn": "vi",
            "no": "no", "is": "is", "ch": "de", "il": "he", "sa": "ar", "ae": "ar", "eg": "ar",
            "kz": "kk"
        }.get(country, "en")
    except:
        return "en"

# Перевод
def translate_class(cls: str, target_lang: str) -> str:
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(cls)
    except:
        return cls

@app.get("/")
def read_root():
    return {"message": "AI Fashion Finder is running."}

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
    finally:
        os.remove(temp_filename)

@app.get("/search")
def search(query: str = Query(...)):
    query_lower = query.lower()
    category = None
    brand = None

    for c in clothing_to_brands.keys():
        if c in query_lower:
            category = c
            break

    for b in ALL_BRANDS:
        if b.lower() in query_lower:
            brand = b
            break

    shops = []
    if category and brand:
        shops.append({"brand": brand, "url": f"https://www.google.com/search?q={brand}+{category}"})
    elif category:
        for b in clothing_to_brands.get(category, []):
            shops.append({"brand": b, "url": f"https://www.google.com/search?q={b}+{category}"})
    elif brand:
        shops.append({"brand": brand, "url": f"https://www.google.com/search?q={brand}+clothing"})

    return {
        "query": query,
        "detected_category": category,
        "detected_brand": brand,
        "shop_results": shops,
        "brands": ALL_BRANDS,
        "stores": ALL_STORES
    }