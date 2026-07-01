"""
HybridRank — Demo App
Team Abhijayati · india.runs Challenge
"""

import streamlit as st
import json, csv, io, sys, os

st.set_page_config(
    page_title="HybridRank · Team Abhijayati",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #1a0533 0%, #2d1052 50%, #0f0f23 100%);
    border-radius: 16px; padding: 40px 48px; margin-bottom: 32px;
    border: 1px solid #4a1a8a;
}
.hero h1 {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(90deg, #e9c46a, #f4a261, #e76f51);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0;
}
.hero p { color: #c9b8e8; font-size: 1.05rem; margin: 0; }
.tag {
    display: inline-block; background: rgba(233,196,106,0.15);
    color: #e9c46a; border: 1px solid rgba(233,196,106,0.3);
    border-radius: 20px; padding: 4px 14px; font-size: 0.82rem;
    font-weight: 600; margin-top: 14px; margin-right: 8px;
}
.stat-box {
    background: linear-gradient(135deg, #1a0533, #2d1052);
    border: 1px solid #4a1a8a; border-radius: 12px;
    padding: 20px 24px; text-align: center;
}
.stat-num { font-size: 2rem; font-weight: 800; color: #e9c46a; }
.stat-label { font-size: 0.78rem; color: #c9b8e8; margin-top: 4px; font-weight: 500; }
.section-title {
    font-size: 1.3rem; font-weight: 700; color: #e9c46a;
    margin: 32px 0 16px 0; border-bottom: 1px solid #2d1052; padding-bottom: 8px;
}
.cand-card {
    background: linear-gradient(135deg, #0f0f23, #1a0533);
    border: 1px solid #2d1052; border-radius: 12px;
    padding: 18px 22px; margin-bottom: 12px;
}
.score-bar-bg { background: #1a0533; border-radius: 6px; height: 8px; overflow: hidden; margin-top: 6px; }
.score-bar-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #7b2fbe, #e9c46a); }
.badge {
    display: inline-block; border-radius: 20px;
    padding: 2px 10px; font-size: 0.72rem; font-weight: 600;
    margin-right: 5px; margin-top: 6px;
}
.badge-ai { background: rgba(123,47,190,0.25); color: #c084fc; border: 1px solid rgba(123,47,190,0.4); }
.badge-open { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.badge-co { background: rgba(233,196,106,0.15); color: #e9c46a; border: 1px solid rgba(233,196,106,0.3); }
.badge-notice { background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
.insight-box {
    background: linear-gradient(135deg, #0f0f23, #1a0533);
    border: 1px solid #2d1052; border-radius: 12px; padding: 20px 24px;
}
.insight-label { color: #c9b8e8; font-size: 0.82rem; font-weight: 600; margin-bottom: 6px; }
.insight-bar-bg { background: #0f0f23; border-radius: 4px; height: 10px; overflow: hidden; }
.insight-bar { height: 100%; border-radius: 4px; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Pre-loaded top results ────────────────────────────────────────────────────
TOP_RESULTS = [
    {"rank":1,"id":"CAND_0011687","score":0.9950,"title":"Senior NLP Engineer","company":"Niramai","yrs":7.8,"skills":5,"github":76.3,"response":89,"open":True,"notice":15},
    {"rank":2,"id":"CAND_0002025","score":0.9782,"title":"Senior AI Engineer","company":"Apple","yrs":5.9,"skills":6,"github":96.9,"response":80,"open":True,"notice":None},
    {"rank":3,"id":"CAND_0064326","score":0.9408,"title":"Search Engineer","company":"Sarvam AI","yrs":7.6,"skills":6,"github":None,"response":94,"open":True,"notice":None},
    {"rank":4,"id":"CAND_0039754","score":0.9332,"title":"Senior Applied Scientist","company":"Meta","yrs":16.2,"skills":7,"github":77.5,"response":81,"open":True,"notice":None},
    {"rank":5,"id":"CAND_0041669","score":0.8994,"title":"Recommendation Systems Engineer","company":"CRED","yrs":8.0,"skills":6,"github":70.9,"response":77,"open":True,"notice":None},
    {"rank":6,"id":"CAND_0088025","score":0.8642,"title":"Staff ML Engineer","company":"Yellow.ai","yrs":8.6,"skills":7,"github":74.6,"response":83,"open":True,"notice":90},
    {"rank":7,"id":"CAND_0046525","score":0.8579,"title":"Senior ML Engineer","company":"Genpact AI","yrs":6.1,"skills":4,"github":None,"response":88,"open":True,"notice":None},
    {"rank":8,"id":"CAND_0018499","score":0.8524,"title":"Senior ML Engineer","company":"Zomato","yrs":7.2,"skills":8,"github":94.8,"response":None,"open":True,"notice":15},
    {"rank":9,"id":"CAND_0027691","score":0.8245,"title":"NLP Engineer","company":"Haptik","yrs":6.5,"skills":5,"github":None,"response":None,"open":True,"notice":15},
    {"rank":10,"id":"CAND_0052328","score":0.8130,"title":"Recommendation Systems Engineer","company":"Amazon","yrs":6.5,"skills":4,"github":77.6,"response":79,"open":True,"notice":None},
]

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🏆 HybridRank</h1>
  <p>AI Candidate Ranking · <strong style="color:#e9c46a">india.runs Challenge</strong></p>
  <p style="color:#9d7fd4; font-size:0.9rem; margin-top:6px;">अभिजयति — To Be Victorious · Team Abhijayati</p>
  <span class="tag">⚡ Pure Python · No GPU · No API</span>
  <span class="tag">📊 100K Candidates Ranked</span>
  <span class="tag">🎯 4-Dimensional Scoring</span>
  <span class="tag">🛡️ 5 Anti-Fraud Rules</span>
</div>
""", unsafe_allow_html=True)

# ── STATS ROW ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
for col, (num, label) in zip([c1,c2,c3,c4,c5], [
    ("100,000", "Candidates Processed"),
    ("775", "Real AI/ML Profiles · 0.78%"),
    ("~3,200", "Keyword Stuffers Caught"),
    ("9,745", "Consulting-Only Filtered"),
    ("< 90s", "Runtime · Single CPU"),
]):
    col.markdown(f'<div class="stat-box"><div class="stat-num">{num}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

# ── TOP CANDIDATES ────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🥇 Top 10 Ranked Candidates — Senior AI Engineer · Redrob AI</div>', unsafe_allow_html=True)

for r in TOP_RESULTS:
    medal = "🥇" if r["rank"]==1 else "🥈" if r["rank"]==2 else "🥉" if r["rank"]==3 else str(r["rank"])
    rank_color = "#e9c46a" if r["rank"]==1 else "#b0bec5" if r["rank"]==2 else "#cd7f32" if r["rank"]==3 else "#7b2fbe"
    badges = f'<span class="badge badge-ai">🧠 {r["skills"]} AI Skills</span>'
    if r["open"]: badges += '<span class="badge badge-open">✅ Open to Work</span>'
    badges += f'<span class="badge badge-co">🏢 {r["company"]}</span>'
    if r["notice"] is not None:
        badges += f'<span class="badge badge-notice">📅 {"Immediate" if r["notice"]<=15 else str(r["notice"])+"d notice"}</span>'
    if r["github"]: badges += f'<span class="badge badge-ai">⚡ GitHub {r["github"]}</span>'

    st.markdown(f"""
    <div class="cand-card">
      <div style="display:flex; align-items:center; gap:16px;">
        <div style="background:linear-gradient(135deg,{rank_color}33,{rank_color}11); color:{rank_color};
             font-weight:800; font-size:1.2rem; width:42px; height:42px; border-radius:50%;
             display:flex; align-items:center; justify-content:center; flex-shrink:0;
             border:2px solid {rank_color}55;">{medal}</div>
        <div style="flex:1;">
          <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <div>
              <span style="font-weight:700; color:#f1f5f9; font-size:1.05rem;">{r["title"]}</span>
              <span style="color:#9d7fd4; font-size:0.88rem; margin-left:10px;">{r["yrs"]} yrs</span>
            </div>
            <span style="font-size:1.5rem; font-weight:800; color:#e9c46a;">{r["score"]:.4f}</span>
          </div>
          <div class="score-bar-bg"><div class="score-bar-fill" style="width:{int(r['score']*100)}%;"></div></div>
          <div style="margin-top:8px;">{badges}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

# ── WHY NOT KEYWORDS ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🛡️ Why Keyword Matching Fails — What HybridRank Catches</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="insight-box">
      <div style="font-weight:700; color:#e9c46a; margin-bottom:14px;">Dataset Reality (100K candidates)</div>
      <div class="insight-label">Real AI/ML Engineers · 0.78%</div>
      <div class="insight-bar-bg"><div class="insight-bar" style="width:5%; background:#7b2fbe;"></div></div>
      <div class="insight-label" style="margin-top:12px;">Keyword Stuffers (wrong title + AI skills) · 3.2%</div>
      <div class="insight-bar-bg"><div class="insight-bar" style="width:16%; background:#e76f51;"></div></div>
      <div class="insight-label" style="margin-top:12px;">Consulting-Only Background · 9.7%</div>
      <div class="insight-bar-bg"><div class="insight-bar" style="width:48%; background:#f4a261;"></div></div>
      <div class="insight-label" style="margin-top:12px;">Ghost Candidates (inactive >180d) · 18%</div>
      <div class="insight-bar-bg"><div class="insight-bar" style="width:90%; background:#457b9d;"></div></div>
      <div style="margin-top:16px; padding:12px; background:rgba(233,196,106,0.08); border-radius:8px; border-left:3px solid #e9c46a;">
        <span style="color:#e9c46a; font-weight:600;">A keyword ranker</span>
        <span style="color:#c9b8e8;"> puts Marketing Managers listing "RAG, Pinecone, Embeddings" at #1. HybridRank collapses them to near-zero.</span>
      </div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="insight-box">
      <div style="font-weight:700; color:#e9c46a; margin-bottom:14px;">5 Hard Disqualifier Multipliers</div>
      """ + "".join([f"""
      <div style="margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between;">
          <span style="color:#c9b8e8; font-size:0.88rem;">{label}</span>
          <span style="color:{color}; font-weight:700;">{mult}</span>
        </div>
        <div class="insight-bar-bg" style="margin-top:4px;"><div class="insight-bar" style="width:{pct}%; background:{color};"></div></div>
      </div>""" for label, mult, pct, color in [
          ("🚨 Keyword Stuffer (wrong title + no retrieval history)", "×0.15", 15, "#e76f51"),
          ("🏢 Consulting-Only (TCS / Infosys / Wipro / Accenture)", "×0.65", 65, "#f4a261"),
          ("👻 Ghost Candidate (inactive >180d + not open to work)", "×0.40", 40, "#457b9d"),
          ("🌍 Outside India, won't relocate", "×0.30", 30, "#9d7fd4"),
          ("📵 Recruiter Black Hole (<10% response rate)", "×0.70", 70, "#e9c46a"),
      ]]) + """
    </div>""", unsafe_allow_html=True)

# ── SCORING ARCHITECTURE ─────────────────────────────────────────────────────
st.markdown('<div class="section-title">⚙️ Scoring Architecture</div>', unsafe_allow_html=True)
cols = st.columns(4)
for col, (icon, title, weight, color, desc) in zip(cols, [
    ("🔬","Technical Fit","30%","#7b2fbe","Core skills weighted by proficiency × endorsements × duration. frozenset O(1) lookup over 40+ AI/retrieval skills."),
    ("📈","Career Quality","35%","#e9c46a","Title fit · Product vs consulting · 29 regex patterns scan career descriptions for real retrieval/ranking work."),
    ("📡","Behavioral","25%","#4ade80","Last active · Open-to-work · Response rate · Interview completion · Profile completeness score."),
    ("📍","Logistics","10%","#60a5fa","India/Pune/Noida location · Notice period · Work mode preference · Salary range fit."),
]):
    col.markdown(f"""
    <div class="insight-box" style="height:100%;">
      <div style="font-size:1.8rem; margin-bottom:8px;">{icon}</div>
      <div style="font-weight:800; color:{color}; font-size:1.3rem;">{weight}</div>
      <div style="font-weight:600; color:#f1f5f9; margin:4px 0 10px;">{title}</div>
      <div style="color:#9d7fd4; font-size:0.82rem; line-height:1.5;">{desc}</div>
    </div>""", unsafe_allow_html=True)

# ── LIVE UPLOAD ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🚀 Try It Live</div>', unsafe_allow_html=True)
st.markdown('<div class="insight-box" style="margin-bottom:16px;"><span style="color:#c9b8e8;">Upload your own <code style="background:#1a0533;padding:2px 6px;border-radius:4px;color:#e9c46a;">candidates.jsonl</code> to rank candidates in real time:</span></div>', unsafe_allow_html=True)

sys.path.insert(0, os.path.dirname(__file__))
try:
    import rank as R
    uploaded = st.file_uploader("Upload candidates.jsonl", type=["jsonl","json"], label_visibility="collapsed")
    if uploaded:
        with st.spinner("Scoring candidates…"):
            lines = uploaded.read().decode("utf-8").splitlines()
            results, errors = [], 0
            for line in lines:
                if not line.strip(): continue
                try: results.append(R.score_candidate(json.loads(line)))
                except: errors += 1
        results.sort(key=lambda r: -r["score"])
        mn, mx = results[-1]["score"], results[0]["score"]
        rng = mx - mn if mx > mn else 1.0
        st.success(f"✅ Scored {len(results):,} candidates ({errors} skipped)")
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["candidate_id","rank","score","reasoning"])
        for i, r in enumerate(results[:100], 1):
            norm = round(0.20 + (r["score"]-mn)/rng*(0.995-0.20), 4)
            w.writerow([r["candidate_id"], i, f"{norm:.4f}", r["reasoning"]])
        st.download_button("⬇️ Download submission.csv", buf.getvalue(), "submission.csv", "text/csv")
        for i, r in enumerate(results[:20], 1):
            norm = round(0.20 + (r["score"]-mn)/rng*(0.995-0.20), 4)
            st.markdown(f"`#{i}` **{r['candidate_id']}** — `{norm:.4f}` — {r['reasoning']}")
except Exception:
    st.info("Upload a candidates.jsonl to rank live.")

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:48px; padding:24px; text-align:center; border-top:1px solid #2d1052;">
  <div style="color:#9d7fd4; font-size:0.85rem;">
    <strong style="color:#e9c46a;">Team Abhijayati</strong> · अभिजयति · india.runs Challenge<br>
    <span style="font-size:0.78rem;">Pure Python stdlib · No GPU · No API calls · Fully deterministic · &lt;90s for 100K candidates</span>
  </div>
</div>
""", unsafe_allow_html=True)
