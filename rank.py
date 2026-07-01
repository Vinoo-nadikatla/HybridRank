#!/usr/bin/env python3
"""
HybridRank — Optimized with multiprocessing for 100K candidates in <30s
"""

import argparse, csv, json, math, re, sys, os
from datetime import date, datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count

# ─── Pre-compile regexes ────────────────────────────────────────────────────
RANKING_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\branking\b", r"\bretrieval\b", r"\brecommendation\b", r"\bsearch\b",
    r"\bembedding", r"\bvector\b", r"re.?rank", r"\bhybrid search\b",
    r"\brelevance\b", r"\bndcg\b", r"\bmrr\b", r"\brecall@", r"\bprecision@",
    r"\bproduction ml\b", r"\bml pipeline\b", r"\bml platform\b",
    r"\bmodel serv", r"\bllm\b", r"\brag\b", r"\bfine.tun",
    r"\bpinecone\b", r"\bweaviate\b", r"\bqdrant\b", r"\bfaiss\b",
    r"\belasticsearch\b", r"\bopensearch\b", r"\ba/b test",
    r"\bonline experiment\b", r"\bfeature store\b",
]]

REFERENCE_DATE = date(2026, 6, 25)

# ─── Core skill sets ────────────────────────────────────────────────────────
CORE_SKILLS = frozenset({
    "sentence-transformers","text embeddings","embeddings","semantic search",
    "dense retrieval","bi-encoder","cross-encoder","embedding models",
    "bge","e5","openai embeddings","cohere embeddings",
    "pinecone","weaviate","qdrant","milvus","faiss","chroma",
    "elasticsearch","opensearch","vector database","vector search",
    "hybrid search","ann","approximate nearest neighbour",
    "information retrieval","learning to rank","ltr","ndcg","mrr","map",
    "ranking","retrieval","reranking","re-ranking","bm25",
    "retrieval-augmented generation","rag",
    "python","nlp","natural language processing","transformers","bert","roberta",
    "hugging face","huggingface",
})
BONUS_SKILLS = frozenset({
    "lora","qlora","peft","fine-tuning llms","fine-tuning","rlhf","instruction tuning",
    "llm","llms","gpt","large language models","pytorch","tensorflow","keras","jax",
    "mlflow","weights & biases","wandb","kubeflow","ray","triton","onnx","model serving",
    "recommendation systems","collaborative filtering","personalization","search ranking",
    "relevance","click-through rate prediction","mlops","feature store","model registry",
    "a/b testing","experimentation","online learning","streaming ml",
    "spark","kafka","airflow","dbt","data pipelines",
})
WRONG_DOMAIN = frozenset({
    "photoshop","illustrator","figma","indesign","premiere","after effects",
    "autocad","solidworks","ansys","catia","revit","accounting","tally",
    "sap fi","quickbooks","crm","salesforce","marketing automation","hubspot",
})

TITLE_SCORES = {
    "ml engineer":1.0,"machine learning engineer":1.0,"senior machine learning engineer":1.0,
    "senior ml engineer":1.0,"staff machine learning engineer":1.0,"lead ml engineer":1.0,
    "ai engineer":1.0,"senior ai engineer":1.0,"lead ai engineer":1.0,
    "applied ml engineer":1.0,"applied scientist":1.0,"senior applied scientist":1.0,
    "nlp engineer":1.0,"senior nlp engineer":1.0,"search engineer":1.0,
    "ranking engineer":1.0,"recommendation systems engineer":1.0,"ai research engineer":0.95,
    "ai specialist":0.90,
    "data scientist":0.80,"senior data scientist":0.85,"lead data scientist":0.85,
    "staff data scientist":0.85,"research scientist":0.80,"junior ml engineer":0.75,
    "backend engineer":0.55,"software engineer":0.50,"data engineer":0.55,
    "analytics engineer":0.50,"full stack developer":0.40,"backend developer":0.45,
    "software developer":0.45,"cloud engineer":0.40,"devops engineer":0.30,
    "platform engineer":0.40,"infrastructure engineer":0.35,
    "senior software engineer":0.50,"senior backend engineer":0.55,
    "frontend engineer":0.20,"java developer":0.20,".net developer":0.15,
    "mobile developer":0.15,"qa engineer":0.15,"qa analyst":0.15,
    "marketing manager":0.03,"content writer":0.02,"graphic designer":0.02,
    "hr manager":0.05,"business analyst":0.10,"project manager":0.08,
    "operations manager":0.05,"sales executive":0.02,"accountant":0.01,
    "civil engineer":0.02,"mechanical engineer":0.02,"customer support":0.02,
}
CONSULTING_FIRMS = frozenset({
    "tata consultancy","tcs","infosys","wipro","hcl technologies","hcl",
    "accenture","cognizant","capgemini","tech mahindra","mphasis","hexaware",
    "l&t infotech","ltimindtree","mindtree","deloitte","kpmg","ey","ernst",
    "pwc","ibm global","dxc technology","ntt data",
})
PRODUCT_SIGNALS = frozenset({
    "google","meta","microsoft","amazon","apple","netflix","uber","airbnb",
    "stripe","openai","anthropic","cohere","deepmind","paytm","flipkart",
    "ola","swiggy","zomato","meesho","razorpay","zepto","cred","phonepe",
    "groww","navi","slice","redrob","freshworks","zoho","chargebee",
    "browserstack","atlassian","databricks","snowflake",
})
PREFERRED_CITIES = frozenset({
    "pune","noida","mumbai","delhi","gurugram","gurgaon",
    "hyderabad","bengaluru","bangalore","ncr",
})
PROF_W = {"expert":1.0,"advanced":0.85,"intermediate":0.65,"beginner":0.40}

def days_since(s):
    try:
        return (REFERENCE_DATE - datetime.strptime(s, "%Y-%m-%d").date()).days
    except: return 999

def norm_skill(s): return s.lower().strip()

def score_candidate(candidate):
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills_raw = candidate.get("skills", [])

    # ── Skill sets ──
    skill_map = {norm_skill(s["name"]): s for s in skills_raw}
    skill_names = frozenset(skill_map.keys())

    # ── TECHNICAL SCORE ──
    core_matches = []
    for sk in CORE_SKILLS:
        if sk in skill_names:
            s = skill_map[sk]
            pw = PROF_W.get(s.get("proficiency","beginner"), 0.4)
            eb = min(0.15, s.get("endorsements",0)/200)
            db = min(0.10, s.get("duration_months",0)/240)
            core_matches.append(pw + eb + db)
    core_score = min(1.0, sum(core_matches)/4.0) if core_matches else 0.0
    breadth_bonus = min(0.2, max(0, len(core_matches)-3)*0.05)
    bonus_count = sum(1 for s in BONUS_SKILLS if s in skill_names)
    bonus_score = min(0.3, bonus_count*0.04)

    assess = signals.get("skill_assessment_scores", {})
    if assess:
        rel_scores = [v/100 for k,v in assess.items() if norm_skill(k) in CORE_SKILLS|BONUS_SKILLS]
        assess_score = sum(rel_scores)/len(rel_scores) if rel_scores else 0.5
    else:
        assess_score = 0.5

    github = signals.get("github_activity_score", -1)
    github_score = github/100 if github >= 0 else 0.3
    wrong_count = sum(1 for s in WRONG_DOMAIN if s in skill_names)
    wrong_penalty = max(0.6, 1.0 - wrong_count*0.08)

    tech_score = min(1.0, (
        0.55*core_score + 0.10*breadth_bonus + 0.15*bonus_score +
        0.10*assess_score + 0.10*github_score
    ) * wrong_penalty)

    # ── CAREER SCORE ──
    current_title = profile.get("current_title","").lower().strip()
    total_yoe = profile.get("years_of_experience", 0)
    companies = [j.get("company","").lower() for j in career]

    title_score = TITLE_SCORES.get(current_title, 0.10)
    if title_score == 0.10:
        for pat, sc in [("machine learning",0.90),("ml ",0.85),("ai ",0.85),
                        ("data scien",0.80),("nlp",0.90),("search",0.70),
                        ("recommend",0.80),("ranking",0.80),("applied scientist",0.90),
                        ("backend",0.45),("software",0.40),("data engineer",0.45)]:
            if pat in current_title: title_score = max(title_score, sc)

    if 4 <= total_yoe <= 10: exp_score = 1.0
    elif 3 <= total_yoe < 4 or 10 < total_yoe <= 14: exp_score = 0.85
    elif 2 <= total_yoe < 3 or 14 < total_yoe <= 18: exp_score = 0.70
    else: exp_score = 0.50

    # AI YoE
    ai_title_kw = {"ml","machine learning","ai ","data scientist","nlp","applied scientist",
                   "search","recommendation","ranking","deep learning","research"}
    ai_months = 0
    for j in career:
        t = j.get("title","").lower()
        d = j.get("description","").lower()
        m = j.get("duration_months",0)
        if any(k in t for k in ai_title_kw): ai_months += m
        elif any(k in d for k in {"machine learning","neural network","embedding","nlp","vector"}):
            ai_months += m*0.5
    if ai_months/12 >= 3: exp_score = min(1.0, exp_score+0.10)

    consulting_only = bool(companies) and all(
        any(cf in co for cf in CONSULTING_FIRMS) for co in companies if co.strip()
    )
    has_product = any(
        any(pc in j.get("company","").lower() for pc in PRODUCT_SIGNALS)
        or (any(ind in j.get("industry","").lower() for ind in
                {"technology","saas","fintech","edtech","healthtech","ai","software product"})
            and not any(cf in j.get("company","").lower() for cf in CONSULTING_FIRMS))
        for j in career
    )
    company_score = 0.30 if consulting_only else (1.0 if has_product else 0.65)

    combined_desc = " ".join(j.get("description","") for j in career).lower()
    desc_matches = sum(1 for p in RANKING_PATTERNS if p.search(combined_desc))
    desc_relevance = min(1.0, desc_matches/5.0)

    short_stints = sum(1 for j in career if j.get("duration_months",36)<18 and not j.get("is_current",False))
    hop_penalty = 0.70 if short_stints>=3 else (0.85 if short_stints==2 else 1.0)

    curr_industry = profile.get("current_industry","").lower()
    industry_score = 1.0 if any(i in curr_industry for i in {
        "technology","software","ai","ml","saas","fintech","internet","data","analytics","e-commerce"
    }) else 0.60

    career_score = min(1.0, (
        0.35*title_score + 0.15*exp_score + 0.25*company_score +
        0.20*desc_relevance + 0.05*industry_score
    ) * hop_penalty)

    # ── BEHAVIORAL SCORE ──
    days_inactive = days_since(signals.get("last_active_date","2020-01-01"))
    recency = (1.0 if days_inactive<=14 else 0.90 if days_inactive<=30 else
               0.75 if days_inactive<=60 else 0.55 if days_inactive<=90 else
               0.35 if days_inactive<=180 else 0.15)
    otw = 1.0 if signals.get("open_to_work_flag",False) else 0.40
    rr = signals.get("recruiter_response_rate",0.0)
    h = signals.get("avg_response_time_hours",48)
    speed = (1.0 if h<=4 else 0.90 if h<=12 else 0.80 if h<=24 else
             0.65 if h<=48 else 0.45 if h<=96 else 0.25)
    iv_rate = signals.get("interview_completion_rate",0.5)
    completeness = signals.get("profile_completeness_score",50)/100
    verified = (signals.get("verified_email",False)*0.5 +
                signals.get("verified_phone",False)*0.3 +
                signals.get("linkedin_connected",False)*0.2)
    saved = min(1.0, signals.get("saved_by_recruiters_30d",0)/20)
    views = min(1.0, signals.get("profile_views_received_30d",0)/100)
    market = 0.6*saved + 0.4*views

    behav_score = min(1.0, 0.25*recency + 0.20*otw + 0.20*rr + 0.10*speed +
                      0.10*iv_rate + 0.07*completeness + 0.05*verified + 0.03*market)

    # ── LOGISTICS SCORE ──
    country = profile.get("country","").strip()
    location = profile.get("location","").lower()
    relocate = signals.get("willing_to_relocate",False)
    in_india = country == "India"
    in_pref = any(c in location for c in PREFERRED_CITIES)

    loc_score = (1.0 if in_india and in_pref else
                 0.90 if in_india and relocate else
                 0.75 if in_india else 0.35 if relocate else 0.10)

    n = signals.get("notice_period_days",90)
    notice_score = (1.0 if n==0 else 0.97 if n<=15 else 0.90 if n<=30 else
                    0.65 if n<=60 else 0.45 if n<=90 else 0.25)

    wm = signals.get("preferred_work_mode","flexible")
    wm_score = 1.0 if wm in ("hybrid","flexible","onsite") else 0.60

    sal = signals.get("expected_salary_range_inr_lpa",{})
    smin, smax = sal.get("min",0), sal.get("max",200)
    sal_score = (1.0 if in_india and smin<=60 and smax>=20 else
                 0.80 if in_india and smin<=80 else 0.50 if in_india else 0.70)

    logistics_score = min(1.0, 0.50*loc_score + 0.30*notice_score + 0.12*wm_score + 0.08*sal_score)

    # ── WEIGHTED COMBINATION ──
    raw = (0.30*tech_score + 0.35*career_score + 0.25*behav_score + 0.10*logistics_score)

    # ── DISQUALIFIER MULTIPLIERS ──
    mult = 1.0
    if title_score < 0.10 and desc_relevance < 0.10: mult *= 0.15  # keyword stuffer
    if consulting_only: mult *= 0.65
    if not in_india and not relocate: mult *= 0.30
    if days_inactive > 180 and not signals.get("open_to_work_flag",False): mult *= 0.40
    if rr < 0.10: mult *= 0.70

    final = min(1.0, raw * mult)

    # ── REASONING ──
    parts = [f"{profile.get('current_title','?')} with {total_yoe:.1f} yrs"]
    if len(core_matches): parts.append(f"{len(core_matches)} core AI/retrieval skills")
    if desc_relevance >= 0.4: parts.append("ranking/retrieval experience in history")
    if consulting_only: parts.append("consulting-only background (penalized)")
    elif company_score >= 0.9: parts.append("product-company background")
    if github >= 70: parts.append(f"GitHub score {github}")
    if rr >= 0.70: parts.append(f"response rate {rr:.0%}")
    elif rr <= 0.15: parts.append(f"low response rate {rr:.0%}")
    if signals.get("open_to_work_flag"): parts.append("open to work")
    else: parts.append("not open to work")
    if n <= 15: parts.append(f"immediate joiner ({n}d notice)")
    elif n > 60: parts.append(f"long notice ({n}d)")
    if not in_india: parts.append(f"outside India ({country})" + ("" if relocate else ", won't relocate"))

    return {
        "candidate_id": candidate["candidate_id"],
        "score": final,
        "reasoning": "; ".join(parts) + ".",
    }


def process_chunk(lines):
    results = []
    for line in lines:
        if not line.strip(): continue
        try:
            results.append(score_candidate(json.loads(line)))
        except Exception: pass
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="./candidates.jsonl")
    parser.add_argument("--out", default="./submission.csv")
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()

    print(f"[HybridRank] Reading {args.candidates}...")
    with open(args.candidates, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    n_cpu = max(1, cpu_count() - 1)
    chunk_size = max(1000, len(all_lines) // n_cpu)
    chunks = [all_lines[i:i+chunk_size] for i in range(0, len(all_lines), chunk_size)]
    print(f"[HybridRank] Processing {len(all_lines):,} candidates across {len(chunks)} chunks ({n_cpu} CPUs)...")

    with Pool(processes=n_cpu) as pool:
        chunk_results = pool.map(process_chunk, chunks)

    results = [r for chunk in chunk_results for r in chunk]
    print(f"[HybridRank] Scored {len(results):,} candidates. Selecting top {args.top}...")

    top = sorted(results, key=lambda r: -r["score"])[:args.top]
    mn, mx = top[-1]["score"], top[0]["score"]
    rng = mx - mn if mx > mn else 1.0
    for r in top:
        r["norm"] = round(0.2000 + (r["score"]-mn)/rng*(0.9950-0.2000), 4)

    top.sort(key=lambda r: (-r["norm"], r["candidate_id"]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id","rank","score","reasoning"])
        for rank, r in enumerate(top, 1):
            writer.writerow([r["candidate_id"], rank, f"{r['norm']:.4f}", r["reasoning"]])

    print(f"[HybridRank] Done! → {out_path}")
    print(f"\nTop 10:")
    for i, r in enumerate(top[:10], 1):
        print(f"  {i:>3}. {r['candidate_id']}  {r['norm']:.4f}  {r['reasoning'][:75]}")

if __name__ == "__main__":
    main()
