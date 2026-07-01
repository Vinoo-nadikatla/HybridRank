"""
HybridRank — Streamlit Demo App
Team Abhijayati · india.runs Data & AI Challenge 2024

Deploy to HuggingFace Spaces:
  1. Create a new Space (Streamlit)
  2. Upload this file + rank.py
  3. Set title: HybridRank by Team Abhijayati

pip install streamlit
streamlit run app.py
"""

import streamlit as st
import json
import csv
import io
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import rank as R

st.set_page_config(
    page_title="HybridRank — Team Abhijayati",
    page_icon="🏆",
    layout="wide",
)

st.markdown("""
<style>
.big-score { font-size: 2rem; font-weight: 700; color: #7B2FBE; }
.rank-badge { background: #0F0F23; color: #E9C46A; padding: 4px 12px;
              border-radius: 6px; font-weight: 700; font-size: 1.1rem; }
.title-text { font-size: 1rem; font-weight: 600; color: #2D3E50; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 5])
with col_title:
    st.title("🏆 HybridRank")
    st.markdown("**Team Abhijayati** · india.runs Data & AI Challenge 2024  \n"
                "*Intelligent Candidate Ranking — Beyond Keywords*")

st.divider()

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_n = st.slider("Candidates to rank", 10, 100, 20, step=10)
    st.divider()
    st.markdown("**Scoring weights**")
    w_tech = st.slider("Technical Fit", 0.0, 1.0, 0.30, step=0.05)
    w_career = st.slider("Career Quality", 0.0, 1.0, 0.35, step=0.05)
    w_behav = st.slider("Behavioral", 0.0, 1.0, 0.25, step=0.05)
    w_log = st.slider("Logistics", 0.0, 1.0, 0.10, step=0.05)
    total = w_tech + w_career + w_behav + w_log
    if abs(total - 1.0) > 0.01:
        st.warning(f"Weights sum to {total:.2f} — should be 1.0")
    st.divider()
    st.caption("HybridRank runs fully on CPU — no API calls, no embeddings model needed.")

# ── File uploader ────────────────────────────────────────────────────────────
st.subheader("📂 Upload Candidates")
uploaded = st.file_uploader(
    "Upload candidates.jsonl (or a sample subset)",
    type=["jsonl", "json"],
    help="Upload the full candidates.jsonl or a subset for demo"
)

if uploaded is not None:
    with st.spinner("Scoring candidates…"):
        content = uploaded.read().decode("utf-8")
        lines = [l for l in content.splitlines() if l.strip()]

        # Override weights from sidebar
        R.WEIGHTS = {"technical": w_tech, "career": w_career,
                     "behavioral": w_behav, "logistics": w_log}

        results = []
        errors = 0
        for line in lines:
            try:
                c = json.loads(line)
                results.append(R.score_candidate(c))
            except Exception:
                errors += 1

    st.success(f"✅ Scored **{len(results):,}** candidates ({errors} skipped)")

    # Normalize and pick top N
    top = sorted(results, key=lambda r: -r["score"])[:top_n]
    mn, mx = top[-1]["score"], top[0]["score"]
    rng = mx - mn if mx > mn else 1.0
    for r in top:
        r["norm"] = round(0.20 + (r["score"] - mn) / rng * (0.995 - 0.20), 4)
    top.sort(key=lambda r: (-r["norm"], r["candidate_id"]))

    # ── Stats row ──
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total scored", f"{len(results):,}")
    c2.metric("Shortlisted", top_n)
    c3.metric("#1 score", f"{top[0]['norm']:.4f}")
    c4.metric("#1 candidate", top[0]["candidate_id"])

    # ── Results table ──
    st.divider()
    st.subheader(f"🥇 Top {top_n} Candidates")

    for i, r in enumerate(top):
        rank = i + 1
        with st.container():
            col_rank, col_id, col_score, col_reason = st.columns([1, 2, 1.5, 6])
            with col_rank:
                st.markdown(f'<span class="rank-badge">#{rank}</span>', unsafe_allow_html=True)
            with col_id:
                st.markdown(f'`{r["candidate_id"]}`')
            with col_score:
                st.markdown(f'<span class="big-score">{r["norm"]:.4f}</span>', unsafe_allow_html=True)
            with col_reason:
                st.markdown(f'<span class="title-text">{r["reasoning"]}</span>', unsafe_allow_html=True)
        if rank < top_n:
            st.divider()

    # ── Download button ──
    st.divider()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["candidate_id", "rank", "score", "reasoning"])
    for rank, r in enumerate(top, 1):
        w.writerow([r["candidate_id"], rank, f"{r['norm']:.4f}", r["reasoning"]])

    st.download_button(
        label="⬇️ Download submission.csv",
        data=buf.getvalue(),
        file_name="submission.csv",
        mime="text/csv",
    )

else:
    st.info("👆 Upload a candidates.jsonl file to start ranking. "
            "You can use a sample of 100–1000 candidates for a quick demo.")

    st.divider()
    st.subheader("How HybridRank Works")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Four scoring dimensions:**
- 🔬 **Technical Fit (30%)** — Core skills, proficiency, GitHub activity
- 📈 **Career Quality (35%)** — Title fit, product vs consulting, career history semantic scan
- 📡 **Behavioral (25%)** — Last active, response rate, open-to-work, interview completion
- 📍 **Logistics (10%)** — Location, notice period, work mode
        """)
    with col2:
        st.markdown("""
**Five hard disqualifier rules:**
- ×0.15 — Keyword stuffer (wrong title + no retrieval history)
- ×0.65 — Consulting-only background
- ×0.30 — Outside India, won't relocate
- ×0.40 — Ghost candidate (inactive >180d)
- ×0.70 — Recruiter black hole (<10% response rate)
        """)
