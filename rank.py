#!/usr/bin/env python3
"""
HybridRank v3 - Team Abhijayati
india.runs Data & AI Challenge

Four-dimensional scoring tuned for NDCG@10:
  Technical Fit  30%  -- core skills weighted by proficiency x endorsements x duration
  Career Quality 35%  -- title fit, product vs consulting, 31 regex patterns on descriptions
  Behavioral     25%  -- all 23 Redrob signals (recency, responsiveness, market demand)
  Logistics      10%  -- location (tiered: Pune/Noida > metro > rest), notice period, work mode, salary

Six hard disqualifier multipliers + honeypot detection.

v3 additions:
  - CV/speech/robotics without NLP/IR penalty (x0.75): explicit JD disqualifier
  - Tiered location: Pune/Noida=1.0, metro=0.92, rest India+relocate=0.80, rest India=0.70
  - Salary sanity: handles min > max data bug in real candidates

Run: python rank.py --candidates ./candidates.jsonl --out ./submission.csv
Requirements: Python 3.8+ stdlib only. No pip installs needed.
Runtime: ~90 seconds for 100,000 candidates on a single CPU core.
"""

import argparse, csv, json, re
from datetime import date, datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count

REFERENCE_DATE = date(2026, 6, 25)

# Pre-compiled career description patterns (31 total)
RANKING_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\branking\b", r"\bretrieval\b", r"\brecommendation\b", r"\bsearch\b",
    r"\bembedding", r"\bvector\b", r"re.?rank", r"\bhybrid search\b",
    r"\brelevance\b", r"\bndcg\b", r"\bmrr\b", r"\brecall@", r"\bprecision@",
    r"\bproduction ml\b", r"\bml pipeline\b", r"\bml platform\b",
    r"\bmodel serv", r"\bllm\b", r"\brag\b", r"\bfine.tun",
    r"\bpinecone\b", r"\bweaviate\b", r"\bqdrant\b", r"\bfaiss\b",
    r"\belasticsearch\b", r"\bopensearch\b", r"\ba/b test",
    r"\bonline experiment\b", r"\bfeature store\b",
    r"\blearning to rank\b", r"\bltr\b", r"\bquery understanding\b",
    r"\bdense retrieval\b", r"\bsparse retrieval\b",
]]

CORE_SKILLS = frozenset({
    "sentence-transformers", "text embeddings", "embeddings", "semantic search",
    "dense retrieval", "bi-encoder", "cross-encoder", "embedding models",
    "bge", "e5", "openai embeddings", "cohere embeddings",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "chroma", "pgvector",
    "elasticsearch", "opensearch", "vector database", "vector search",
    "hybrid search", "ann", "approximate nearest neighbour", "approximate nearest neighbor",
    "information retrieval", "learning to rank", "ltr", "ndcg", "mrr", "map",
    "ranking", "retrieval", "reranking", "re-ranking", "bm25", "sparse retrieval",
    "retrieval-augmented generation", "rag",
    "python", "nlp", "natural language processing", "transformers", "bert", "roberta",
    "hugging face", "huggingface",
})
BONUS_SKILLS = frozenset({
    "lora", "qlora", "peft", "fine-tuning llms", "fine-tuning", "rlhf", "instruction tuning",
    "llm", "llms", "gpt", "large language models", "pytorch", "tensorflow", "keras", "jax",
    "mlflow", "weights & biases", "wandb", "kubeflow", "ray", "triton", "onnx", "model serving",
    "recommendation systems", "collaborative filtering", "personalization", "search ranking",
    "relevance", "click-through rate prediction", "mlops", "feature store", "model registry",
    "a/b testing", "experimentation", "online learning", "streaming ml",
    "spark", "kafka", "airflow", "dbt", "data pipelines",
    "langchain", "llamaindex", "openai api",
})
WRONG_DOMAIN = frozenset({
    "photoshop", "illustrator", "figma", "indesign", "premiere", "after effects",
    "autocad", "solidworks", "ansys", "catia", "revit", "accounting", "tally",
    "sap fi", "quickbooks", "crm", "salesforce", "marketing automation", "hubspot",
    "corel draw", "adobe xd", "sketch",
})
# IR foundation skills - absence of these alongside LangChain = vibe-coding flag
IR_FOUNDATION_SKILLS = frozenset({
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "bm25",
    "elasticsearch", "opensearch", "learning to rank", "ltr",
    "information retrieval", "dense retrieval", "hybrid search",
    "ranking", "reranking", "retrieval",
})
# CV/speech/robotics skills - JD explicit disqualifier when no NLP/IR exposure
CV_SPEECH_SKILLS = frozenset({
    "computer vision", "image classification", "object detection", "object recognition",
    "opencv", "yolo", "yolov5", "yolov8", "image segmentation", "image processing",
    "speech recognition", "automatic speech recognition", "asr", "speech synthesis",
    "text-to-speech", "tts", "voice recognition", "speaker identification",
    "robotics", "ros", "robot operating system", "slam",
    "autonomous driving", "lidar", "point cloud",
})

TITLE_SCORES = {
    # Tier A: pure AI/ML engineering
    "ml engineer": 1.0, "machine learning engineer": 1.0,
    "senior machine learning engineer": 1.0, "senior ml engineer": 1.0,
    "staff machine learning engineer": 1.0, "lead ml engineer": 1.0,
    "principal ml engineer": 1.0,
    "ai engineer": 1.0, "senior ai engineer": 1.0, "lead ai engineer": 1.0,
    "applied ml engineer": 1.0, "applied scientist": 0.95,
    "senior applied scientist": 1.0, "principal applied scientist": 1.0,
    "nlp engineer": 1.0, "senior nlp engineer": 1.0,
    "search engineer": 1.0, "senior search engineer": 1.0,
    "ranking engineer": 1.0, "recommendation systems engineer": 1.0,
    "ai research engineer": 0.95, "research engineer": 0.90,
    "staff research engineer": 0.95, "ml specialist": 0.90, "ai specialist": 0.90,
    # Tier B: data science adjacent
    "data scientist": 0.80, "senior data scientist": 0.85,
    "lead data scientist": 0.85, "staff data scientist": 0.85,
    "principal data scientist": 0.88, "research scientist": 0.75,
    # Tier C: junior
    "junior ml engineer": 0.75, "junior data scientist": 0.65,
    # Tier D: SWE with ML potential
    "backend engineer": 0.55, "senior backend engineer": 0.55,
    "software engineer": 0.50, "senior software engineer": 0.50,
    "data engineer": 0.55, "senior data engineer": 0.55,
    "analytics engineer": 0.50, "full stack developer": 0.40,
    "backend developer": 0.45, "software developer": 0.45,
    "cloud engineer": 0.40, "devops engineer": 0.30,
    "platform engineer": 0.40, "infrastructure engineer": 0.35,
    # Tier E: low relevance
    "frontend engineer": 0.20, "java developer": 0.20, ".net developer": 0.15,
    "mobile developer": 0.15, "ios developer": 0.15, "android developer": 0.15,
    "qa engineer": 0.15, "qa analyst": 0.15, "test engineer": 0.15,
    "business intelligence analyst": 0.25, "bi analyst": 0.20,
    # Tier F: disqualifying
    "marketing manager": 0.03, "content writer": 0.02, "graphic designer": 0.02,
    "hr manager": 0.05, "business analyst": 0.10, "project manager": 0.08,
    "operations manager": 0.05, "sales executive": 0.02, "accountant": 0.01,
    "civil engineer": 0.02, "mechanical engineer": 0.02, "customer support": 0.02,
    "scrum master": 0.05, "product manager": 0.08, "ux designer": 0.10,
    "recruiter": 0.02, "hr executive": 0.03,
}

CONSULTING_FIRMS = frozenset({
    "tata consultancy", "tcs", "infosys", "wipro", "hcl technologies", "hcl",
    "accenture", "cognizant", "capgemini", "tech mahindra", "mphasis", "hexaware",
    "l&t infotech", "ltimindtree", "mindtree", "deloitte", "kpmg", "ey", "ernst",
    "pwc", "ibm global", "dxc technology", "ntt data", "niit technologies",
    "zensar", "cyient", "sonata software", "mastech", "igate", "persistent systems",
})
PRODUCT_SIGNALS = frozenset({
    "google", "meta", "microsoft", "amazon", "apple", "netflix", "uber", "airbnb",
    "stripe", "openai", "anthropic", "cohere", "deepmind", "mistral", "stability ai",
    "paytm", "flipkart", "ola", "swiggy", "zomato", "meesho", "razorpay", "zepto",
    "cred", "phonepe", "groww", "navi", "slice", "redrob", "freshworks", "zoho",
    "chargebee", "browserstack", "atlassian", "databricks", "snowflake", "hugging face",
    "sarvam", "niramai", "haptik", "uniphore", "vernacular", "kognitos",
    "yellow.ai", "observe.ai", "genpact ai", "sigmoid", "fractal analytics",
    "mu sigma", "tiger analytics", "latentview", "sprinklr", "leadsquared",
})
RESEARCH_ORGS = frozenset({
    "university", "iit ", "iim ", "iisc", "iiser", "nit ", "bits ",
    "research institute", "research lab", "research center", "laboratory",
    "college of engineering", "institute of technology",
})
NON_CODING_TITLES = frozenset({
    "architect", "solution architect", "enterprise architect", "cloud architect",
    "chief", "vp ", "vice president", "director of", "head of",
    "cto", "ceo", "coo", "chief technology",
})
# JD says "Pune/Noida preferred" -> tier-1; other metros -> tier-2
TIER1_CITIES = frozenset({"pune", "noida"})
TIER2_CITIES = frozenset({
    "mumbai", "delhi", "gurugram", "gurgaon",
    "hyderabad", "bengaluru", "bangalore", "ncr", "chennai",
})
PREFERRED_CITIES = TIER1_CITIES | TIER2_CITIES  # backward compat
RELEVANT_CERTS = frozenset({
    "aws certified machine learning", "google professional machine learning",
    "tensorflow developer", "azure ai engineer", "databricks certified",
    "deeplearning.ai", "coursera machine learning", "pytorch",
})

PROF_W = {"expert": 1.0, "advanced": 0.85, "intermediate": 0.65, "beginner": 0.40}
EDU_TIER = {"tier_1": 1.0, "tier_2": 0.80, "tier_3": 0.60, "tier_4": 0.40, "unknown": 0.50}


def days_since(s):
    try:
        return (REFERENCE_DATE - datetime.strptime(s, "%Y-%m-%d").date()).days
    except Exception:
        return 999


def nrm(s):
    return s.lower().strip() if s else ""


def is_honeypot(candidate):
    """
    Detect candidates with impossible profile combinations.
    From submission spec: "expert proficiency in 10 skills with 0 years used"
    or "8 years at a company founded 3 years ago".
    Returns (bool, reason_string).
    """
    skills = candidate.get("skills", [])
    stated_yoe = candidate.get("profile", {}).get("years_of_experience", 0)

    # Pattern 1: Multiple expert/advanced skills with duration_months = 0
    expert_zero = [
        s for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and s.get("duration_months", -1) == 0
    ]
    if len(expert_zero) >= 3:
        return True, (
            str(len(expert_zero)) + " advanced-level skills with 0 months duration"
        )

    # Pattern 2: Any skill duration exceeds entire career
    if stated_yoe > 0:
        max_skill_months = max((s.get("duration_months", 0) for s in skills), default=0)
        career_months = stated_yoe * 12
        if max_skill_months > career_months + 24:
            return True, (
                "skill duration " + str(max_skill_months) +
                "mo exceeds career length " + str(career_months) + "mo"
            )

    # Pattern 3: 8+ expert skills all under 12 months
    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    if len(expert_skills) >= 8 and all(s.get("duration_months", 0) < 12 for s in expert_skills):
        return True, (
            str(len(expert_skills)) + " expert-level skills all under 12 months"
        )

    return False, ""


def score_candidate(candidate):
    cid = candidate["candidate_id"]
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills_raw = candidate.get("skills", [])
    education = candidate.get("education", [])
    certifications = candidate.get("certifications", [])

    # Honeypot gate
    hp, hp_desc = is_honeypot(candidate)
    if hp:
        return {
            "candidate_id": cid,
            "score": 0.0005,
            "reasoning": (
                "Profile contains impossible signal combination (" + hp_desc + "); "
                "forced to relevance tier 0 per honeypot detection."
            ),
        }

    skill_map = {nrm(s["name"]): s for s in skills_raw}
    skill_names = frozenset(skill_map.keys())

    # ===========================================================================
    # DIMENSION 1 - TECHNICAL FIT (30%)
    # ===========================================================================
    core_matched = []
    for sk in CORE_SKILLS:
        if sk in skill_names:
            s = skill_map[sk]
            dur = s.get("duration_months", 0)
            prof = s.get("proficiency", "beginner")
            if prof in ("expert", "advanced") and dur == 0:
                continue  # partial honeypot signal - skip
            pw = PROF_W.get(prof, 0.4)
            eb = min(0.15, s.get("endorsements", 0) / 200)
            db = min(0.10, dur / 240)
            core_matched.append(pw + eb + db)

    core_score = min(1.0, sum(core_matched) / 4.0) if core_matched else 0.0
    breadth_bonus = min(0.20, max(0, len(core_matched) - 3) * 0.05)
    bonus_count = sum(1 for sk in BONUS_SKILLS if sk in skill_names)
    bonus_score = min(0.30, bonus_count * 0.04)

    assess = signals.get("skill_assessment_scores", {}) or {}
    rel_assess = [v / 100 for k, v in assess.items()
                  if nrm(k) in CORE_SKILLS | BONUS_SKILLS]
    assess_score = (sum(rel_assess) / len(rel_assess)) if rel_assess else 0.5
    high_core_assess = [v for k, v in assess.items() if nrm(k) in CORE_SKILLS and v >= 80]
    assess_depth = min(0.10, len(high_core_assess) * 0.033)

    github = signals.get("github_activity_score", -1)
    github_score = github / 100 if github >= 0 else 0.30

    wrong_count = sum(1 for sk in WRONG_DOMAIN if sk in skill_names)
    wrong_penalty = max(0.60, 1.0 - wrong_count * 0.08)

    has_rag_wrappers = "langchain" in skill_names or "llamaindex" in skill_names
    has_ir_foundations = bool(IR_FOUNDATION_SKILLS & skill_names)
    lc_penalty = 0.82 if (has_rag_wrappers and not has_ir_foundations) else 1.0

    # CV/speech/robotics without NLP/IR exposure - explicit JD disqualifier
    # "People whose primary expertise is computer vision, speech, or robotics without
    #  significant NLP/IR exposure. We respect your work but you'd be re-learning here."
    has_cv_speech = bool(CV_SPEECH_SKILLS & skill_names)
    cv_penalty = 0.75 if (has_cv_speech and not has_ir_foundations) else 1.0

    tech_score = min(1.0, (
        0.50 * core_score +
        0.10 * breadth_bonus +
        0.12 * bonus_score +
        0.08 * assess_score +
        0.10 * github_score +
        0.10 * assess_depth
    ) * wrong_penalty * lc_penalty * cv_penalty)

    # ===========================================================================
    # DIMENSION 2 - CAREER QUALITY (35%)
    # ===========================================================================
    current_title = nrm(profile.get("current_title", ""))
    total_yoe = profile.get("years_of_experience", 0)
    companies = [nrm(j.get("company", "")) for j in career]

    title_score = TITLE_SCORES.get(current_title, 0.10)
    if title_score == 0.10:
        for pat, sc in [
            ("machine learning", 0.90), ("ml ", 0.85), ("ai ", 0.85),
            ("data scien", 0.80), ("nlp", 0.90), ("search", 0.70),
            ("recommend", 0.80), ("ranking", 0.80), ("applied scientist", 0.90),
            ("research engineer", 0.85), ("backend", 0.45), ("software", 0.40),
            ("data engineer", 0.45),
        ]:
            if pat in current_title:
                title_score = max(title_score, sc)
    if any(t in current_title for t in NON_CODING_TITLES):
        title_score = min(title_score, 0.50)

    # Experience score - JD wants 5+ years NLP/IR
    if 5 <= total_yoe <= 10:
        exp_score = 1.0
    elif 4 <= total_yoe < 5 or 10 < total_yoe <= 13:
        exp_score = 0.90
    elif 3 <= total_yoe < 4 or 13 < total_yoe <= 16:
        exp_score = 0.75
    elif 2 <= total_yoe < 3 or 16 < total_yoe <= 20:
        exp_score = 0.60
    else:
        exp_score = 0.40

    # AI-specific YoE from career history
    ai_kw = {"ml", "machine learning", "ai ", "data scientist", "nlp",
              "applied scientist", "search", "recommendation", "ranking",
              "deep learning", "research"}
    ai_months = 0
    for j in career:
        t = nrm(j.get("title", ""))
        d = nrm(j.get("description", ""))
        m = j.get("duration_months", 0)
        if any(k in t for k in ai_kw):
            ai_months += m
        elif any(k in d for k in
                 {"machine learning", "neural network", "embedding", "nlp", "vector search"}):
            ai_months += m * 0.5
    ai_yoe = ai_months / 12
    if ai_yoe >= 3:
        exp_score = min(1.0, exp_score + 0.10)
    if ai_yoe >= 5:
        exp_score = min(1.0, exp_score + 0.05)

    # Company type
    consulting_flags = [any(cf in co for cf in CONSULTING_FIRMS) for co in companies if co]
    n_jobs = len(consulting_flags)
    n_consulting = sum(consulting_flags)
    consulting_only = bool(n_jobs) and n_consulting == n_jobs
    consulting_heavy = bool(n_jobs) and n_consulting >= max(1, n_jobs * 0.75)

    has_product = any(
        any(pc in nrm(j.get("company", "")) for pc in PRODUCT_SIGNALS)
        or (
            any(ind in nrm(j.get("industry", "")) for ind in
                {"technology", "saas", "fintech", "edtech", "healthtech",
                 "ai", "software product", "internet", "e-commerce"})
            and not any(cf in nrm(j.get("company", "")) for cf in CONSULTING_FIRMS)
        )
        for j in career
    )

    is_pure_research = bool(career) and all(
        any(r in nrm(j.get("company", "")) for r in RESEARCH_ORGS)
        or any(r in nrm(j.get("industry", "")) for r in {"research", "education", "academic"})
        for j in career
    )

    if consulting_only:
        company_score = 0.30
    elif consulting_heavy:
        company_score = 0.50
    elif is_pure_research:
        company_score = 0.60
    elif has_product:
        # Product company bonus is gated on title relevance:
        # A Frontend/Java dev at Google is still not a fit for this JD
        title_gate = min(1.0, max(0.60, title_score / 0.70))
        company_score = 1.0 * title_gate
    else:
        company_score = 0.72

    # Description scan — career descriptions + profile summary + headline (all rich text)
    summary_text = profile.get("summary", "") + " " + profile.get("headline", "")
    combined_desc = " ".join(j.get("description", "") for j in career) + " " + summary_text
    desc_matches = sum(1 for p in RANKING_PATTERNS if p.search(combined_desc))
    desc_relevance = min(1.0, desc_matches / 6.0)

    # Job-hopping — use consistent default (36mo = normal) for missing duration_months
    past_roles = [j for j in career if not j.get("is_current", False)]
    durations = [j.get("duration_months") for j in past_roles]
    durations = [d if d is not None else 36 for d in durations]
    short_stints = sum(1 for d in durations if d < 18)
    ultra_short = sum(1 for d in durations if d < 6)
    avg_tenure = sum(durations) / len(durations) if durations else 36
    if ultra_short >= 2:
        hop_penalty = 0.65
    elif short_stints >= 3 or avg_tenure < 14:
        hop_penalty = 0.75
    elif short_stints >= 2:
        hop_penalty = 0.88
    else:
        hop_penalty = 1.0

    curr_industry = nrm(profile.get("current_industry", ""))
    industry_score = (1.0 if any(i in curr_industry for i in {
        "technology", "software", "ai", "ml", "saas", "fintech", "internet",
        "data", "analytics", "e-commerce", "healthtech", "edtech",
    }) else 0.60)

    edu_score = 0.60
    for e in education:
        tier = e.get("tier", "unknown")
        edu_score = max(edu_score, EDU_TIER.get(tier, 0.50))

    cert_bonus = 0.0
    for cert in certifications:
        if any(rc in nrm(cert.get("name", "")) for rc in RELEVANT_CERTS):
            cert_bonus = min(0.05, cert_bonus + 0.025)

    career_score = min(1.0, (
        0.30 * title_score +
        0.15 * exp_score +
        0.22 * company_score +
        0.20 * desc_relevance +
        0.05 * industry_score +
        0.05 * edu_score +
        cert_bonus
    ) * hop_penalty)

    # ===========================================================================
    # DIMENSION 3 - BEHAVIORAL AVAILABILITY (25%) - All 23 Redrob signals
    # ===========================================================================
    days_inactive = days_since(signals.get("last_active_date", "2020-01-01"))
    recency = (
        1.0 if days_inactive <= 7 else
        0.95 if days_inactive <= 14 else
        0.85 if days_inactive <= 30 else
        0.70 if days_inactive <= 60 else
        0.50 if days_inactive <= 90 else
        0.30 if days_inactive <= 180 else
        0.10
    )

    otw = 1.0 if signals.get("open_to_work_flag", False) else 0.40
    rr = signals.get("recruiter_response_rate", 0.0)

    h = signals.get("avg_response_time_hours", 48)
    speed = (
        1.0 if h <= 4 else 0.90 if h <= 12 else
        0.80 if h <= 24 else 0.65 if h <= 48 else
        0.45 if h <= 96 else 0.25
    )

    iv_rate = signals.get("interview_completion_rate", 0.5)
    oa_rate = signals.get("offer_acceptance_rate", -1)
    offer_signal = oa_rate if oa_rate >= 0 else 0.60

    completeness = signals.get("profile_completeness_score", 50) / 100

    verified = (
        signals.get("verified_email", False) * 0.50 +
        signals.get("verified_phone", False) * 0.30 +
        signals.get("linkedin_connected", False) * 0.20
    )

    saved = min(1.0, signals.get("saved_by_recruiters_30d", 0) / 15)
    views = min(1.0, signals.get("profile_views_received_30d", 0) / 80)
    search_app = min(1.0, signals.get("search_appearance_30d", 0) / 100)
    market_signal = 0.40 * saved + 0.30 * views + 0.30 * search_app

    apps = signals.get("applications_submitted_30d", 0)
    if apps == 0:
        activity = 0.50
    elif apps <= 5:
        activity = 0.75
    elif apps <= 15:
        activity = 1.0
    elif apps <= 25:
        activity = 0.85
    else:
        activity = 0.60

    connections = signals.get("connection_count", 0)
    endorsements_recv = signals.get("endorsements_received", 0)
    social = min(1.0, connections / 300 * 0.5 + endorsements_recv / 50 * 0.5)

    behav_score = min(1.0, (
        0.22 * recency +
        0.17 * otw +
        0.14 * rr +
        0.08 * speed +
        0.08 * iv_rate +
        0.06 * offer_signal +
        0.06 * completeness +
        0.05 * verified +
        0.07 * market_signal +
        0.04 * activity +
        0.03 * social
    ))

    # ===========================================================================
    # DIMENSION 4 - LOGISTICS (10%)
    # ===========================================================================
    country = profile.get("country", "").strip()
    location = nrm(profile.get("location", ""))
    relocate = signals.get("willing_to_relocate", False)
    in_india = country == "India"
    in_tier1 = any(c in location for c in TIER1_CITIES)   # Pune/Noida preferred
    in_tier2 = any(c in location for c in TIER2_CITIES)   # other metros

    # Tiered location: JD explicitly says "Pune/Noida preferred"
    loc_score = (
        1.00 if in_india and in_tier1 else        # Pune/Noida
        0.92 if in_india and in_tier2 else        # other India metros
        0.80 if in_india and relocate else         # rest of India, will relocate
        0.70 if in_india else                      # rest of India, won't say
        0.35 if relocate else 0.10                 # abroad
    )

    n_notice = signals.get("notice_period_days", 90)
    notice_score = (
        1.0 if n_notice == 0 else 0.97 if n_notice <= 15 else
        0.90 if n_notice <= 30 else 0.65 if n_notice <= 60 else
        0.45 if n_notice <= 90 else 0.25
    )

    wm = signals.get("preferred_work_mode", "flexible")
    wm_score = 1.0 if wm in ("hybrid", "flexible", "onsite") else 0.60

    sal = signals.get("expected_salary_range_inr_lpa", {}) or {}
    s_min = sal.get("min", 0)
    s_max = sal.get("max", 200)
    # Sanity: real data bug where min > max - swap them
    if s_min > s_max and s_max > 0:
        s_min, s_max = s_max, s_min
    sal_score = (
        1.0 if in_india and s_min <= 60 and s_max >= 20 else
        0.80 if in_india and s_min <= 80 else
        0.50 if in_india else 0.70
    )

    logistics_score = min(1.0,
        0.50 * loc_score + 0.30 * notice_score +
        0.12 * wm_score + 0.08 * sal_score
    )

    # ===========================================================================
    # COMPOSITE SCORE
    # ===========================================================================
    raw = (
        0.30 * tech_score +
        0.35 * career_score +
        0.25 * behav_score +
        0.10 * logistics_score
    )

    # Hard disqualifier multipliers
    mult = 1.0
    if title_score < 0.10 and desc_relevance < 0.10:
        mult *= 0.15   # Keyword stuffer
    if consulting_only:
        mult *= 0.65   # Consulting-only
    if not in_india and not relocate:
        mult *= 0.30   # Outside India, won't relocate
    if days_inactive > 180 and not signals.get("open_to_work_flag", False):
        mult *= 0.40   # Ghost candidate
    if rr < 0.10:
        mult *= 0.70   # Recruiter black hole

    final = min(1.0, raw * mult)

    # ===========================================================================
    # REASONING - specific, factual, varied per candidate
    # Stage 4 checks: specific facts, JD connection, honest concerns,
    #                 no hallucination, variation, rank consistency
    # ===========================================================================
    ct = profile.get("current_title", "Unknown")
    cc = profile.get("current_company", "")
    at_co = (" at " + cc) if cc else ""

    # Sentence 1: who they are + technical/career strengths
    s1 = []
    yoe_str = str(int(total_yoe)) + " yrs total"
    if ai_yoe >= 1:
        yoe_str += " (" + str(int(ai_yoe)) + " AI/ML)"
    s1.append(ct + at_co + "; " + yoe_str)

    if len(core_matched) > 0:
        sample_core = [sk for sk in list(CORE_SKILLS) if sk in skill_names][:2]
        s1.append(str(len(core_matched)) + " core retrieval skills (" + ", ".join(sample_core) + ")")

    if desc_relevance >= 0.5:
        s1.append("strong ranking/retrieval signals in career history")
    elif desc_relevance >= 0.3:
        s1.append("some IR experience in career descriptions")

    if consulting_only:
        s1.append("consulting-only background (x0.65 penalty)")
    elif has_product:
        prod_names = [nrm(j.get("company", "")) for j in career
                      if any(pc in nrm(j.get("company", "")) for pc in PRODUCT_SIGNALS)]
        if prod_names:
            s1.append("product-company track record (" + prod_names[0] + ")")
    elif is_pure_research:
        s1.append("academic/research background only")

    if edu_score >= 0.90:
        s1.append("tier-1 institution")

    # Sentence 2: availability + concerns
    s2 = []
    s2.append("open to work" if signals.get("open_to_work_flag", False) else "not open-to-work")

    if rr >= 0.80:
        s2.append(str(int(rr * 100)) + "% recruiter response rate")
    elif rr < 0.20:
        s2.append("low response rate (" + str(int(rr * 100)) + "%)")

    if days_inactive <= 14:
        s2.append("active recently")
    elif days_inactive > 180:
        s2.append("inactive " + str(days_inactive // 30) + "+ months")

    if hop_penalty < 0.85:
        if short_stints >= 2:
            s2.append(str(short_stints) + " stints under 18mo (job-hopping risk)")
        elif avg_tenure < 14:
            s2.append("avg tenure " + str(int(avg_tenure)) + "mo across past roles (short)")

    if not in_india:
        reloc_note = "" if relocate else ", no relocation"
        s2.append("outside India (" + country + reloc_note + ")")

    if n_notice > 90:
        s2.append(str(n_notice) + "d notice period")

    saved_count = signals.get("saved_by_recruiters_30d", 0)
    if saved_count >= 5:
        s2.append("saved by " + str(saved_count) + " recruiters (30d)")

    if github >= 70:
        s2.append("GitHub score " + str(github) + "/100")

    if has_rag_wrappers and not has_ir_foundations:
        s2.append("LangChain only - no IR fundamentals")

    if has_cv_speech and not has_ir_foundations:
        cv_skills = list(CV_SPEECH_SKILLS & skill_names)[:2]
        s2.append("CV/speech background (" + ", ".join(cv_skills) + ") without NLP/IR foundation")

    reasoning = "; ".join(s1) + ". " + "; ".join(s2) + "."
    return {
        "candidate_id": cid,
        "score": final,
        "reasoning": reasoning,
    }


def process_chunk(lines):
    results = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            results.append(score_candidate(json.loads(line)))
        except Exception:
            pass
    return results


def main():
    parser = argparse.ArgumentParser(description="HybridRank v3 - Team Abhijayati")
    parser.add_argument("--candidates", default="./candidates.jsonl")
    parser.add_argument("--out", default="./submission.csv")
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()

    print("[HybridRank v3] Reading " + args.candidates + " ...")
    with open(args.candidates, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    print("[HybridRank v3] Loaded " + str(len(all_lines)) + " candidates.")

    n_cpu = max(1, cpu_count() - 1)
    chunk_size = max(1000, len(all_lines) // n_cpu)
    chunks = [all_lines[i:i + chunk_size]
              for i in range(0, len(all_lines), chunk_size)]
    print("[HybridRank v3] Scoring across " + str(len(chunks)) +
          " chunks on " + str(n_cpu) + " CPU cores...")

    with Pool(processes=n_cpu) as pool:
        chunk_results = pool.map(process_chunk, chunks)

    results = [r for chunk in chunk_results for r in chunk]
    print("[HybridRank v3] Scored " + str(len(results)) +
          " candidates. Selecting top " + str(args.top) + "...")

    top = sorted(results, key=lambda r: (-r["score"], r["candidate_id"]))[:args.top]

    mn, mx = top[-1]["score"], top[0]["score"]
    rng = mx - mn if mx > mn else 1.0
    for r in top:
        r["norm"] = round(0.2000 + (r["score"] - mn) / rng * (0.9950 - 0.2000), 4)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_pos, r in enumerate(top, 1):
            writer.writerow([
                r["candidate_id"],
                rank_pos,
                "{:.4f}".format(r["norm"]),
                r["reasoning"],
            ])

    print("[HybridRank v3] Done! -> " + str(out_path))
    print("\nTop 10:")
    for i, r in enumerate(top[:10], 1):
        print("  " + str(i).rjust(3) + ". " + r["candidate_id"] +
              "  " + "{:.4f}".format(r["norm"]) +
              "  " + r["reasoning"][:85] + "...")

    hp_in_top = sum(1 for r in top if r["score"] <= 0.001)
    print("\nHoneypot rate in top " + str(args.top) + ": " +
          str(hp_in_top) + "/" + str(args.top))
    if hp_in_top == 0:
        print("No honeypots in top " + str(args.top) + ".")
    else:
        print("WARNING: " + str(hp_in_top) + " honeypot(s) in top 100 - review immediately!")

    scores = [r["norm"] for r in top]
    print("Score range: " + "{:.4f}".format(min(scores)) + " - " + "{:.4f}".format(max(scores)))
    print("Candidates >= 0.80: " + str(sum(1 for s in scores if s >= 0.80)))
    print("Candidates >= 0.60: " + str(sum(1 for s in scores if s >= 0.60)))


if __name__ == "__main__":
    main()
