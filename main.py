from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import time
from functools import lru_cache

app = FastAPI(title="جستجوی هوشمند مقاله - جشنواره")

# مدل embedding (دقت بالا + multilingual شامل فارسی)
embed_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')

# مدل خلاصه‌سازی (چندزبانه)
summarizer = pipeline("summarization", model="facebook/mbart-large-50")

# کش نتایج جستجو (تا ۱۰۰ مورد اخیر)
cache = {}
CACHE_LIMIT = 100

class QueryRequest(BaseModel):
    query: str

def search_wikipedia_titles(query: str, lang: str = "fa", max_results: int = 800):
    """
    جستجوی گسترده با pagination و srlimit بالا
    """
    url = f"https://{lang}.wikipedia.org/w/api.php"
    titles = []
    continue_token = {}

    headers = {
        'User-Agent': 'FestivalSemanticSearch/1.0 (k.sharafie@gmail.com)'  # ← ایمیل واقعی خودت را بگذار
    }

    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srwhat": "text",           # جستجو در متن مقاله → نتایج خیلی بیشتر
            "srlimit": "500",           # حداکثر مجاز
            "srprop": "snippet",
            **continue_token
        }

        resp = requests.get(url, params=params, headers=headers).json()

        if 'query' in resp and 'search' in resp['query']:
            new_titles = [item['title'] for item in resp['query']['search']]
            titles.extend(new_titles)

        if 'continue' in resp:
            continue_token = resp['continue']
        else:
            break

        time.sleep(0.6)  # تأخیر برای جلوگیری از بلاک شدن

        if len(titles) >= max_results:
            break

    return list(set(titles))[:max_results]  # حذف تکراری + محدود

def fetch_article_text(title: str, lang: str = "fa"):
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "exsentences": 10,          # فقط چند جمله اول برای سرعت
        "titles": title
    }
    resp = requests.get(url, params=params).json()
    pages = resp.get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", "")
    return ""

def summarize_text(text: str):
    if len(text.strip()) < 50:
        return text.strip() or "خلاصه موجود نیست."
    try:
        summary = summarizer(text[:1200], max_length=90, min_length=40, do_sample=False)[0]['summary_text']
        return summary
    except:
        return text[:300] + "..."  # fallback

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>فایل index.html پیدا نشد!</h1>"

@app.post("/search")
def perform_search(req: QueryRequest):
    q = req.query.strip()
    if not q:
        return []

    # کش
    if q in cache:
        return cache[q]

    # جستجوی گسترده
    fa_titles = search_wikipedia_titles(q, "fa", 800)
    en_titles = search_wikipedia_titles(q, "en", 800)

    # ترکیب + محدود به ۲۰-۳۰ مورد برتر برای embedding (سرعت مهم است)
    combined = list(set(fa_titles[:15] + en_titles[:15]))

    results = []

    for title in combined:
        lang = "fa" if title in fa_titles else "en"
        text = fetch_article_text(title, lang)
        if not text:
            continue

        q_emb = embed_model.encode(q)
        t_emb = embed_model.encode(text)

        sim = cosine_similarity([q_emb], [t_emb])[0][0]
        percent = round(float(sim * 100), 2)

        if percent > 82:
            help_text = "بله – احتمالاً پاسخ کامل سوال شما در این مقاله است."
        elif percent > 58:
            help_text = "تا حد خوبی مرتبط است – بخش‌هایی از پاسخ را پوشش می‌دهد."
        else:
            help_text = "ارتباط کم – بهتر است منابع دیگری هم بررسی کنید."

        summary = summarize_text(text)

        results.append({
            "title": title,
            "percent": percent,
            "help": help_text,
            "summary": summary,
            "url": f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "lang": lang.upper()
        })

    # مرتب‌سازی descending + فقط ۵ مورد برتر
    top5 = sorted(results, key=lambda x: x["percent"], reverse=True)[:5]

    # ذخیره در کش
    if len(cache) >= CACHE_LIMIT:
        oldest = next(iter(cache))
        del cache[oldest]
    cache[q] = top5


    return top5
