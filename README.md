# HybridRank v2 — Team Abhijayati

> *To Be Victorious*

---

## What We Built

HybridRank ranks candidates the way a great recruiter would — not by matching keywords, but by understanding who actually fits the role.

The core insight: thousands of candidates in the dataset have AI skills listed (RAG, Pinecone, Embeddings) but are Marketing Managers, HR Managers, or Accountants. A keyword ranker puts them at the top. HybridRank doesn't. It also detects the ~80 honeypot candidates the challenge embeds to catch naive systems.

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
INGEST -> SCORE -> PENALIZE -> RANK
100K profiles   4 dimensions   5 hard rules   Top 100
```

### Scoring Dimensions

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| Technical Fit | 30% | Core skills weighted by proficiency x endorsements x duration; Redrob assessment scores; GitHub activity |
| Career Quality | 35% | Title/domain fit; product vs consulting background; 31 regex patterns on career descriptions; AI-specific YoE; education tier |
| Behavioral | 25% | All 23 Redrob signals: recency, open-to-work, response rate, interview completion, offer acceptance, market demand (saved/viewed/searched), social proof |
| Logistics | 10% | Location (India/preferred cities); notice period; work mode; salary range |

### Hard Disqualifier Multipliers

| JD Red Flag | Detection | Multiplier |
|-------------|-----------|-----------|
| Keyword stuffer | title_score < 0.10 AND desc_relevance < 0.10 | x0.15 |
| Consulting-only background | All companies in consulting set | x0.65 |
| Outside India, won't relocate | country != India AND willing_to_relocate = False | x0.30 |
| Ghost candidate | Inactive >180d AND not open to work | x0.40 |
| Recruiter black hole | recruiter_response_rate < 10% | x0.70 |

### Honeypot Detection

The challenge embeds ~80 honeypot candidates with impossible profile combinations. HybridRank detects three patterns:

1. **Expert/advanced skills with 0 months duration** — 3+ such skills triggers disqualification
2. **Skill duration exceeds career** — any skill claimed for longer than total career length
3. **Mass expert claims with trivial duration** — 8+ expert skills all under 12 months

Honeypots are scored 0.0005 (effectively rank 100+) and reasoning explicitly states the impossible combination found.

### v2 Improvements Over v1

- Honeypot detection (protects against disqualification at Stage 3)
- All 23 Redrob behavioral signals used (v1 used ~11)
- LangChain-only penalty: has LangChain but no IR foundations = vibe-coder flag
- Pure research background penalty (no production deployment evidence)
- Education tier scoring (tier_1 = IIT/IISc gets bonus)
- Improved job-hopping analysis (avg tenure + ultra-short stints)
- Consistent missing-field handling
- Specific, factual, varied reasoning per candidate (Stage 4 quality)

---

## What We Found in 100,000 Profiles

- **775** candidates have genuine AI/ML titles (0.78% of total)
- **~3,200** keyword stuffers detected and penalized
- **9,745** consulting-only backgrounds (9.7%)
- **~18,000** ghost candidates (inactive >180 days)
- **99 of our top 100** are India-based

---

## Top 10 Results (v1 run — v2 results require running on full dataset)

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

- **31 pre-compiled regex patterns** scan career history for retrieval/ranking work
- **frozenset O(1) lookups** for skill matching — no linear scans
- **Rule-based disqualifiers** encode JD reasoning at rule-engine speed

Model-quality reasoning. Millisecond latency.

---

## Files

```
Abhijayati/
├── rank.py                   # Main ranker v2 — run this
├── submission.csv            # Ranked output (100 candidates)
├── submission_metadata.yaml  # Submission metadata
├── requirements.txt          # (empty — zero dependencies)
├── Dockerfile                # For HuggingFace Space
├── app.py                    # Streamlit demo
└── README.md                 # This file
```

---

## Team

**Team Abhijayati** — To Be Victorious

*"We'd rather see 10 great matches than 1,000 maybes." — The JD.
HybridRank delivers exactly that.*
