from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

app = FastAPI(title="جستجوی هوشمند مقاله - جشنواره")

# مدل سبک و سریع (برای جلوگیری از crash رم)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# کش ساده
cache = {}
CACHE_LIMIT = 100

class QueryRequest(BaseModel):
    query: str

def search_wikipedia(query: str, lang: str = "fa", max_results: int = 300):
    url = f"https://{lang}.wikipedia.org/w/api.php"
    titles = []
    cont = {}

    headers = {
        'User-Agent': 'FestivalProject/1.0 (k.sharafie@gmail.com)'
    }

    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srwhat": "text",
            "srlimit": "50",
            **cont
        }

        resp = requests.get(url, params=params, headers=headers).json()

        if 'query' in resp and 'search' in resp['query']:
            new_titles = [item['title'] for item in resp['query']['search']]
            titles.extend(new_titles)

        if 'continue' in resp:
            cont = resp['continue']
        else:
            break

        time.sleep(0.5)

        if len(titles) >= max_results:
            break

    return list(set(titles))[:max_results]

def get_text(title: str, lang: str):
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "exsentences": 8,
        "titles": title
    }
    resp = requests.get(url, params=params).json()
    pages = resp.get("query", {}).get("pages", {})
    for p in pages.values():
        return p.get("extract", "")
    return ""

@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "<h1>فایل index.html پیدا نشد</h1>"

@app.get("/kaithhealth")
async def health_check():
    return {"status": "healthy"}

@app.post("/search")
def search(req: QueryRequest):
    q = req.query.strip()
    if not q:
        return []

    if q in cache:
        return cache[q]

    fa_titles = search_wikipedia(q, "fa")
    en_titles = search_wikipedia(q, "en")

    candidates = list(set(fa_titles[:10] + en_titles[:10]))

    results = []

    for title in candidates:
        lang = "fa" if title in fa_titles else "en"
        text = get_text(title, lang)
        if not text:
            continue

        q_emb = embed_model.encode(q)
        t_emb = embed_model.encode(text)

        sim = cosine_similarity([q_emb], [t_emb])[0][0]
        percent = round(float(sim * 100), 2)

        if percent > 75:
            help_msg = "بله، احتمالاً پاسخ کامل دارد."
        elif percent > 50:
            help_msg = "تا حد خوبی مرتبط است."
        else:
            help_msg = "ارتباط کم – منابع دیگر را هم چک کنید."

        summary = text[:250] + "..." if text else "خلاصه موجود نیست."

        results.append({
            "title": title,
            "percent": percent,
            "help": help_msg,
            "summary": summary,
            "url": f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "lang": lang.upper()
        })

    top = sorted(results, key=lambda x: x["percent"], reverse=True)[:5]

    if len(cache) >= CACHE_LIMIT:
        cache.pop(next(iter(cache)), None)
    cache[q] = top

    return top
