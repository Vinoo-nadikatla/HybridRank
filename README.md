# HybridRank — Team Abhijayati 🏆

> **india.runs Data & AI Challenge 2024**  
> *अभिजयति — To Be Victorious*

---

## What We Built

HybridRank is an intelligent candidate ranking engine that ranks candidates the way a **great recruiter** would — not by matching keywords, but by understanding who actually fits the role.

**The core insight:** The JD for this challenge explicitly warns that keyword matching is a trap. Thousands of candidates in the dataset have AI skills listed (RAG, Pinecone, Embeddings) but are Marketing Managers, HR Managers, or Accountants. A keyword ranker puts them at the top. HybridRank doesn't.

---

## How to Run

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

**Requirements:** Python 3.8+ only. No pip installs needed — pure stdlib.

**Runtime:** ~90 seconds for 100,000 candidates on a single CPU core.

---

## Architecture: Four-Dimensional Scoring

```
INGEST → SCORE → PENALIZE → RANK
100K profiles   4 dimensions   5 hard rules   Top 100
```

### Scoring Dimensions

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| Technical Fit | 30% | Core skills (embeddings, vector DBs, Python, ranking eval) weighted by proficiency × endorsements × duration |
| Career Quality | 35% | Title/domain fit · Product vs consulting background · Career description semantic scan (29 regex patterns) · AI-specific YoE |
| Behavioral | 25% | Last active date · Open-to-work · Response rate · Interview completion · Profile completeness |
| Logistics | 10% | Location (India/Pune/Noida) · Notice period · Work mode · Salary range |

### Hard Disqualifier Multipliers

| JD Red Flag | Detection | Multiplier |
|-------------|-----------|-----------|
| Keyword stuffer (wrong title + no retrieval history) | title_score < 0.10 AND desc_relevance < 0.10 | ×0.15 |
| Consulting-only background | All companies in TCS/Infosys/Wipro/Accenture set | ×0.65 |
| Outside India, won't relocate | country ≠ 'India' AND willing_to_relocate = False | ×0.30 |
| Ghost candidate (inactive >180d) | days_since(last_active) > 180 AND open_to_work = False | ×0.40 |
| Recruiter black hole | recruiter_response_rate < 0.10 | ×0.70 |

---

## What We Found in 100,000 Profiles

- **775** candidates have genuine AI/ML titles (0.78% of total)
- **~3,200** keyword stuffers detected and penalized
- **9,745** consulting-only backgrounds (9.7%)
- **~18,000** ghost candidates (inactive >180 days)
- **99 of our top 100** are India-based

---

## Top 10 Results

| Rank | Title | Company | Score |
|------|-------|---------|-------|
| 1 | Senior NLP Engineer | Niramai | 0.9950 |
| 2 | Senior AI Engineer | Apple | 0.9782 |
| 3 | Search Engineer | Sarvam AI | 0.9408 |
| 4 | Senior Applied Scientist | Meta | 0.9332 |
| 5 | Recommendation Systems Engineer | CRED | 0.8994 |
| 6 | Staff Machine Learning Engineer | Yellow.ai | 0.8642 |
| 7 | Senior Machine Learning Engineer | Genpact AI | 0.8579 |
| 8 | Senior Machine Learning Engineer | Zomato | 0.8524 |
| 9 | NLP Engineer | Haptik | 0.8245 |
| 10 | Recommendation Systems Engineer | Amazon | 0.8130 |

---

## Why No LLM/Embedding Inference?

The submission spec requires the ranker to run in <5 minutes on CPU with no network access. Instead, we front-load semantic intelligence into our scoring functions:

- **29 pre-compiled regex patterns** scan career history descriptions for retrieval/ranking work
- **frozenset lookups** for skill matching — O(1) per candidate
- **Rule-based disqualifiers** encode JD reasoning at rule-engine speed

Model-quality reasoning. Millisecond latency.

---

## Files

```
Abhijayati/
├── rank.py                          # Main ranker — run this
├── submission.csv                   # Ranked output (100 candidates)
├── Abhijayati_HybridRank_Deck.pdf  # Pitch deck
├── submission_metadata.yaml        # Submission metadata
├── requirements.txt                # (empty — no dependencies)
├── logo.svg                        # Team logo
└── README.md                       # This file
```

---

## Team

**Team Abhijayati** (अभिजयति — Sanskrit: *To Be Victorious*)  
india.runs Data & AI Challenge 2024

---

*"We'd rather see 10 great matches than 1,000 maybes." — The JD.  
HybridRank delivers exactly that.*
