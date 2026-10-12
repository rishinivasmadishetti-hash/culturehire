import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import os
import math

# --- Config ---
QLOO_BASE = "https://hackathon.api.qloo.com"
QLOO_KEY  = os.environ.get("QLOO_API_KEY", "hack_a5affd6c0eb23ff8ba6520ac7d11d87a93eb7386")
HEADERS   = {"X-Api-Key": QLOO_KEY}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Qloo helpers ──────────────────────────────────────────────────────────────

# Try multiple entity types per category for best search hit
SEARCH_TYPES = {
    "music": [
        "urn:entity:music_genre",
        "urn:entity:artist",
        "urn:entity:album",
    ],
    "movie": [
        "urn:entity:movie",
        "urn:entity:tv_show",
    ],
    "food": [
        "urn:entity:place_category",
        "urn:entity:place",
        "urn:entity:cuisine",
    ],
}

def search_entity(query: str, category: str) -> tuple[str | None, str | None]:
    """
    Try multiple entity types for a category.
    Returns (entity_id, entity_type) of first hit.
    Falls back to no-type search if typed search fails.
    """
    types = SEARCH_TYPES.get(category, ["urn:entity:movie"])

    # First: try each specific type
    for etype in types:
        try:
            r = requests.get(
                f"{QLOO_BASE}/search",
                headers=HEADERS,
                params={"q": query, "types": etype},
                timeout=8,
            )
            if r.status_code != 200:
                continue
            data = r.json()
            results = data.get("results") or data.get("entities") or []
            for item in results:
                eid = item.get("entity_id") or item.get("id")
                if eid:
                    print(f"  ✓ Found '{query}' as {etype}: {eid[:8]}...")
                    return eid, etype
        except Exception as e:
            print(f"  search error {etype}: {e}")

    # Fallback: search without type filter
    try:
        r = requests.get(
            f"{QLOO_BASE}/search",
            headers=HEADERS,
            params={"q": query},
            timeout=8,
        )
        if r.status_code == 200:
            data = r.json()
            results = data.get("results") or data.get("entities") or []
            for item in results:
                eid  = item.get("entity_id") or item.get("id")
                etype = (item.get("types") or ["urn:entity:movie"])[0]
                if eid:
                    print(f"  ✓ Found '{query}' (no-type fallback) as {etype}: {eid[:8]}...")
                    return eid, etype
    except Exception as e:
        print(f"  fallback search error: {e}")

    print(f"  ✗ No entity found for '{query}' in category '{category}'")
    return None, None


def get_insights_vector(entity_id: str, signal_type: str) -> dict[str, float]:
    """
    Get Qloo insights using same-type results for discriminative scoring.
    music → music space, movie → movie space, food → food space.
    Falls back to movie space if same-type returns nothing.
    """
    # Use same-type first for more discriminative comparison
    result_types = [signal_type, "urn:entity:movie"]
    # Deduplicate
    seen = set()
    unique_types = []
    for t in result_types:
        if t not in seen:
            seen.add(t)
            unique_types.append(t)

    for result_type in unique_types:
        try:
            r = requests.get(
                f"{QLOO_BASE}/v2/insights",
                headers=HEADERS,
                params={
                    "signal.interests.entities": entity_id,
                    "filter.type": result_type,
                    "take": 30,
                },
                timeout=10,
            )
            if r.status_code != 200:
                continue
            data = r.json()
            results = data.get("results") or []
            if results:
                vec = {}
                for item in results:
                    eid = item.get("entity_id") or item.get("id")
                    aff = float(item.get("affinity") or item.get("score") or 0.5)
                    if eid:
                        vec[eid] = aff
                if vec:
                    print(f"  ✓ Got {len(vec)} insights [{result_type.split(':')[-1]}] for {entity_id[:8]}...")
                    return vec
        except Exception as e:
            print(f"  insights error: {e}")
    print(f"  ✗ No insights for {entity_id[:8]}...")
    return {}


def cosine(v1: dict, v2: dict) -> float:
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot  = sum(v1[k] * v2[k] for k in common)
    mag1 = math.sqrt(sum(x**2 for x in v1.values()))
    mag2 = math.sqrt(sum(x**2 for x in v2.values()))
    return dot / (mag1 * mag2) if (mag1 and mag2) else 0.0


def text_similarity(a: str, b: str) -> float:
    """Simple word-overlap similarity as fallback."""
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    if not a_words or not b_words:
        return 0.0
    overlap = a_words & b_words
    union   = a_words | b_words
    return len(overlap) / len(union)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "CultureHire running ✅",
        "qloo_key_set": QLOO_KEY != "hack_a5affd6c0eb23ff8ba6520ac7d11d87a93eb7386",
        "qloo_base": QLOO_BASE,
    }


@app.get("/debug/{category}/{query}")
def debug_search(category: str, query: str):
    """Debug: see exactly what Qloo returns for a search."""
    results = {}
    types = SEARCH_TYPES.get(category, ["urn:entity:movie"])
    for etype in types:
        r = requests.get(
            f"{QLOO_BASE}/search",
            headers=HEADERS,
            params={"q": query, "types": etype},
            timeout=8,
        )
        results[etype] = {
            "status": r.status_code,
            "data": r.json() if r.status_code == 200 else r.text
        }
    return results


@app.post("/fingerprint")
def build_fingerprint(data: dict):
    """
    Input:  { "music": "jazz", "food": "sushi", "movie": "Inception" }
    Output: { "fingerprint": [...], "vectors": {...}, "raw_queries": {...} }
    """
    fingerprint = []
    vectors     = {}
    raw_queries = {}

    for category, query in data.items():
        if not query or not query.strip():
            continue
        q = query.strip()
        raw_queries[category] = q
        print(f"\n[{category}] Searching for: '{q}'")

        eid, etype = search_entity(q, category)

        if eid:
            fingerprint.append({
                "category": category,
                "query": q,
                "entity_id": eid,
                "entity_type": etype,
            })
            vec = get_insights_vector(eid, etype)
            vectors[category] = {
                "entity_id": eid,
                "affinity": vec,
            }
        else:
            # No Qloo entity found — store query text for text-similarity fallback
            fingerprint.append({
                "category": category,
                "query": q,
                "entity_id": None,
                "entity_type": None,
            })
            vectors[category] = {
                "entity_id": None,
                "affinity": {},
                "text": q,  # fallback
            }

    return {
        "fingerprint": fingerprint,
        "vectors": vectors,
        "raw_queries": raw_queries,
    }


@app.post("/match")
def match_culture(data: dict):
    """
    Scoring strategy (per category):
      1. Same entity_id found on both sides → 100
      2. Both have insight vectors → cosine similarity × 100
      3. No vectors → text similarity × 100
      4. Only one side has a vector → 20 (partial)
    Final score = weighted average (movie=40%, music=35%, food=25%)
    """
    WEIGHTS = {"movie": 0.40, "music": 0.35, "food": 0.25}

    c_vecs = data["company"].get("vectors", {})
    p_vecs = data["candidate"].get("vectors", {})

    c_fp   = {i["category"]: i for i in data["company"].get("fingerprint", [])}
    p_fp   = {i["category"]: i for i in data["candidate"].get("fingerprint", [])}

    categories = set(c_vecs) | set(p_vecs)
    category_scores = {}

    for cat in categories:
        cv = c_vecs.get(cat, {})
        pv = p_vecs.get(cat, {})

        c_eid   = cv.get("entity_id")
        p_eid   = pv.get("entity_id")
        c_aff   = cv.get("affinity", {})
        p_aff   = pv.get("affinity", {})
        c_text  = cv.get("text") or c_fp.get(cat, {}).get("query", "")
        p_text  = pv.get("text") or p_fp.get(cat, {}).get("query", "")

        if c_eid and p_eid and c_eid == p_eid:
            # Exact same entity
            sim = 1.0
            print(f"  [{cat}] Exact entity match → 100%")
        elif c_aff and p_aff:
            # Compare insight vectors
            sim = cosine(c_aff, p_aff)
            print(f"  [{cat}] Cosine similarity → {sim:.2%}")
        elif c_text and p_text:
            # Text fallback
            sim = text_similarity(c_text, p_text)
            print(f"  [{cat}] Text similarity → {sim:.2%}")
        elif c_aff or p_aff:
            sim = 0.2  # one side only
        else:
            sim = 0.0

        category_scores[cat] = round(sim * 100)

    # Weighted average
    total_weight = 0.0
    weighted_sum = 0.0
    for cat, pct in category_scores.items():
        w = WEIGHTS.get(cat, 0.33)
        weighted_sum += pct * w
        total_weight += w

    score = round(weighted_sum / total_weight) if total_weight else 0
    score = min(100, score)

    # Calibrate — apply S-curve to spread scores more naturally
    # cosine tends to cluster high; push middle scores down
    def adjust(s):
        if s >= 95: return 92
        if s >= 80: return round(65 + (s - 80) * 1.35)
        if s >= 60: return round(40 + (s - 60) * 1.25)
        if s >= 40: return round(20 + (s - 40) * 1.0)
        return round(s * 0.6)
    score = adjust(score)

    if score >= 70:
        verdict     = "Strong Culture Fit"
        explanation = "Qloo's taste graph shows significant cultural overlap. This candidate is likely to feel right at home with your team's vibe across music, food, and entertainment."
    elif score >= 50:
        verdict     = "Good Alignment"
        explanation = "Meaningful shared taste in several dimensions. Some differences exist but the cultural foundation is solid — worth a conversation."
    elif score >= 30:
        verdict     = "Some Common Ground"
        explanation = "A few cultural touchpoints in common, but notable differences. Could work with intentional onboarding and team integration."
    else:
        verdict     = "Different Cultural Profiles"
        explanation = "This candidate and team have distinct cultural profiles. Diversity of taste can be a strength — but worth exploring in interviews."

    return {
        "score": score,
        "verdict": verdict,
        "explanation": explanation,
        "company_taste":   [i["query"] for i in data["company"].get("fingerprint", [])],
        "candidate_taste": [i["query"] for i in data["candidate"].get("fingerprint", [])],
        "category_scores": category_scores,
    }


# Serve frontend (must be last)
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
