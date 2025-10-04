# to run the app:
# streamlit run app.py

# app.py — Patient → Therapist Matching (Journey schemas)
import re
import streamlit as st
import pandas as pd
import random

# --- Move these arrays to the top, right after imports ---
NEEDS_KEYWORDS = [
    "anxiety", "depression", "trauma", "cbt", "ptsd", "grief", "stress", "anger", "adhd", "autism",
    "relationship", "family", "addiction", "eating disorder", "bipolar", "panic", "phobia", "sleep",
    "self-esteem", "mindfulness", "career", "parenting", "divorce", "abuse", "lgbtq", "identity",
    "social skills", "emotional regulation", "coping skills", "life transitions", "chronic illness",
    "pain management", "obsessive-compulsive", "psychosis", "schizophrenia", "self-harm", "suicidal ideation",
    "substance use", "codependency", "perfectionism", "spirituality", "anger management", "stress management",
    "assertiveness", "communication", "boundaries", "motivation", "goal setting", "peer relationships",
    "school issues", "work stress"
]

NOTES_PHRASES = [
    "Looking for a therapist experienced with trauma and PTSD.",
    "Prefers CBT and practical coping strategies.",
    "Needs support for anxiety and panic attacks.",
    "Interested in mindfulness and stress reduction.",
    "Wants help with relationship and communication skills.",
    "Seeking guidance for career and life transitions.",
    "Struggling with depression and low motivation.",
    "Needs family therapy and parenting support.",
    "Looking for LGBTQ-affirming care.",
    "Wants to address substance use and addiction.",
    "Needs help with emotional regulation and anger.",
    "Interested in support for chronic illness and pain.",
    "Looking for help with self-esteem and confidence.",
    "Needs support for grief and loss.",
    "Prefers telehealth sessions in the evenings.",
    "Wants to work on assertiveness and boundaries.",
    "Needs help with obsessive thoughts and compulsions.",
    "Looking for culturally competent care.",
    "Interested in group therapy options.",
    "Needs support for eating disorder recovery.",
    "Wants to address perfectionism and work stress.",
    "Needs help with school-related anxiety.",
    "Looking for a therapist who accepts Aetna insurance.",
    "Prefers in-person sessions.",
    "Needs support for identity and self-discovery."
]
# --- End move ---

# Initialize patient_version in session state if not present
if "patient_version" not in st.session_state:
    st.session_state["patient_version"] = 0

# sklearn optional (falls back to 0 text similarity if missing)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

st.set_page_config(page_title="Therapist Matching POC", layout="wide")

# ----------------- helpers -----------------
def parse_set(x):
    """Split messy list strings into a set."""
    if pd.isna(x) or str(x).strip() == "":
        return set()
    s = re.sub(r"[\[\]\(\)\{\}'\"\\]", " ", str(x))
    toks = re.split(r"[,\|;/]+", s)
    return {t.strip() for t in toks if t.strip()}

def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))

STATE2 = {
 "AL":"AL","Alabama":"AL","AK":"AK","Alaska":"AK","AZ":"AZ","Arizona":"AZ","AR":"AR","Arkansas":"AR",
 "CA":"CA","California":"CA","CO":"CO","Colorado":"CO","CT":"CT","Connecticut":"CT","DE":"DE","Delaware":"DE",
 "FL":"FL","Florida":"FL","GA":"GA","Georgia":"GA","HI":"HI","Hawaii":"HI","ID":"ID","Idaho":"ID","IL":"IL","Illinois":"IL",
 "IN":"IN","Indiana":"IN","IA":"IA","Iowa":"IA","KS":"KS","Kansas":"KS","KY":"KY","Kentucky":"KY","LA":"LA","Louisiana":"LA",
 "ME":"ME","Maine":"ME","MD":"MD","Maryland":"MD","MA":"MA","Massachusetts":"MA","MI":"MI","Michigan":"MI",
 "MN":"MN","Minnesota":"MN","MS":"MS","Mississippi":"MS","MO":"MO","Missouri":"MO","MT":"MT","Montana":"MT",
 "NE":"NE","Nebraska":"NE","NV":"NV","Nevada":"NV","NH":"NH","New Hampshire":"NH","NJ":"NJ","New Jersey":"NJ",
 "NM":"NM","New Mexico":"NM","NY":"NY","New York":"NY","NC":"NC","North Carolina":"NC","ND":"ND","North Dakota":"ND",
 "OH":"OH","Ohio":"OH","OK":"OK","Oklahoma":"OK","OR":"OR","Oregon":"OR","PA":"PA","Pennsylvania":"PA",
 "RI":"RI","Rhode Island":"RI","SC":"SC","South Carolina":"SC","SD":"SD","South Dakota":"SD","TN":"TN","Tennessee":"TN",
 "TX":"TX","Texas":"TX","UT":"UT","Utah":"UT","VT":"VT","Vermont":"VT","VA":"VA","Virginia":"VA",
 "WA":"WA","Washington":"WA","WV":"WV","West Virginia":"WV","WI":"WI","Wisconsin":"WI","WY":"WY","Wyoming":"WY",
 "DC":"DC","District of Columbia":"DC"
}
FIPS2 = {
 "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","11":"DC",
 "12":"FL","13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY",
 "22":"LA","23":"ME","24":"MD","25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT",
 "31":"NE","32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND","39":"OH",
 "40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD","47":"TN","48":"TX","49":"UT",
 "50":"VT","51":"VA","53":"WA","54":"WV","55":"WI","56":"WY"
}
def norm_state(x):
    """Normalize full names, 2-letter codes, or numeric codes like 5 / 5.0 → FIPS."""
    if x is None:
        return ""
    s = str(x).strip()
    if re.fullmatch(r"\d+(\.0+)?", s):
        n = str(int(float(s))).zfill(2)
        return FIPS2.get(n, n)
    return STATE2.get(s, STATE2.get(s.title(), STATE2.get(s.upper(), s.upper())))

def parse_bool(x):
    s = str(x).strip().lower()
    return any(k in s for k in ["true","t","yes","y","1","open","accept"])

def tfidf_sim(a, b):
    if not HAS_SKLEARN:
        return 0.0
    a = "" if pd.isna(a) else str(a)
    b = "" if pd.isna(b) else str(b)
    if not a and not b:
        return 0.0
    vec = TfidfVectorizer(min_df=1, stop_words="english")
    X = vec.fit_transform([a, b])
    return float(cosine_similarity(X[0], X[1])[0, 0])

def col_or(df, row_mask, col, fallback=""):
    if col in df.columns:
        v = df.loc[row_mask, col].astype(str).fillna("").iloc[0]
        return v
    return fallback

# ----------------- sidebar -----------------
st.sidebar.header("Upload CSVs (Journey schemas)")

with st.sidebar.expander("Show Instructions", expanded=False):
    st.markdown(
        """
        **How to use this app:**
        1. Upload both CSVs (patients and therapists).
        2. Adjust weights as needed.
        3. Select a patient and review top therapist matches.
        4. Approve or reject matches and export your decisions.

        ---
        **How does matching work?**
        - **Hard filters:** Therapist must be credentialed in the patient's state and accepting new patients.
        - **Scoring:** Each feasible match is scored using:
            - **Tag overlap:** Jaccard similarity between patient needs and therapist specialties.
            - **Text similarity:** TF-IDF cosine similarity between patient notes and therapist bio (using scikit-learn).
            - **Miscellaneous boosts:** +0.1 for matching modality, +0.1 for matching insurance.
        - **Formula:**  
          `score = 0.1 + w_tags * tag_score + w_text * text_score + w_misc * misc_boost`
          - `tag_score = max(Jaccard(needs, specialties), 0.01)`
          - `text_score = max(TF-IDF cosine similarity, 0.01)`
          - `misc_boost = min(modality_boost + insurance_boost, 0.2)`

        **Tech stack:**  
        - Python, Streamlit, Pandas, scikit-learn (TF-IDF), Jaccard similarity.
        """
    )

p_file = st.sidebar.file_uploader(
    "Patients-schema.csv", type=["csv"],
    help="Upload the patient data file (CSV, Journey schema)."
)
t_file = st.sidebar.file_uploader(
    "Therapists-schema.csv", type=["csv"],
    help="Upload the therapist data file (CSV, Journey schema)."
)

st.sidebar.header("Weights")
w_tags = st.sidebar.slider(
    "Specialty/Needs overlap", 0.0, 10.0, 5.0, 0.1
)
w_text = st.sidebar.slider(
    "Text similarity (notes ↔ bio)", 0.0, 10.0, 5.0, 0.1
)
w_misc = st.sidebar.slider(
    "Misc boosts (modality + insurance)", 0.0, 10.0, 3.0, 0.1
)

if not (p_file and t_file):
    st.info("Upload both CSVs to begin.")
    st.stop()

patients = pd.read_csv(p_file)
therapists = pd.read_csv(t_file, engine="python")

# ----------------- normalize -----------------
state_col = "US_STATE_ID" if "US_STATE_ID" in patients.columns else (
    "PATIENT_STATE_NAME" if "PATIENT_STATE_NAME" in patients.columns else None
)
if state_col is None:
    st.error("Patients file must include US_STATE_ID or PATIENT_STATE_NAME.")
    st.stop()

patients["_PID"]   = patients["PATIENT_ID"] if "PATIENT_ID" in patients.columns else pd.Series(range(len(patients)))
patients["_STATE"] = patients[state_col].apply(norm_state)

therapists["_TID"]         = therapists["THERAPIST_ID"] if "THERAPIST_ID" in therapists.columns else pd.Series(range(len(therapists)))
therapists["_ACCEPT"]      = therapists.get("ACCEPTING_NEW_PATIENTS", False).apply(parse_bool)
therapists["_STATES"]      = therapists.get("CREDENTIALED_STATES", "").apply(lambda s: {norm_state(tok) for tok in parse_set(s)})
therapists["_INS"]         = therapists.get("THERAPIST_INSURANCES", "").apply(parse_set)
therapists["_MODS"]        = therapists.get("KAP_MODALITIES", "").apply(parse_set)
therapists["_BIO"]         = therapists.get("BIO", "").astype(str)
therapists["_SPECIAL"]     = therapists.get("INTEREST_SPECIALITIES", "").astype(str) + ";" + therapists.get("EXPERTISE_SPECIALITIES", "").astype(str)
therapists["_SPECIAL_SET"] = therapists["_SPECIAL"].apply(parse_set)

# ----------------- UI -----------------
c1, c2 = st.columns([1, 2])

with c1:
    st.subheader("Select Patient")

    pid = st.selectbox(
        "Patient ID", patients["_PID"], key="pid_select"
    )

    # Increment version on patient change
    if st.session_state.get("prev_pid") != pid:
        st.session_state["patient_version"] += 1
        # Pick random 3-5 needs and 1 note
        random_needs = ", ".join(random.sample(NEEDS_KEYWORDS, k=random.randint(3, 5)))
        random_note = random.choice(NOTES_PHRASES)
        # Get insurance from CSV if present, else blank
        ins_col = None
        for col in ["INSURANCE", "PATIENT_INSURANCE", "INSURANCES"]:
            if col in patients.columns:
                ins_col = col
                break
        if ins_col:
            mask = patients["_PID"] == pid
            patient_ins = patients.loc[mask, ins_col].astype(str).fillna("").iloc[0]
        else:
            patient_ins = ""
        st.session_state[f"needs_{pid}"] = random_needs
        st.session_state[f"notes_{pid}"] = random_note
        st.session_state[f"ins_{pid}"] = patient_ins
        st.session_state["prev_pid"] = pid

    version = st.session_state["patient_version"]

    # Use composite keys to force widget refresh
    needs_txt = st.text_input(
        "Patient needs/tags (comma-separated)",
        key=f"needs_{pid}_{version}",
        value=st.session_state.get(f"needs_{pid}", "")
    )
    notes_txt = st.text_area(
        "Patient notes (free-text for similarity)",
        key=f"notes_{pid}_{version}",
        value=st.session_state.get(f"notes_{pid}", ""),
        height=120
    )
    needs_set = {s.strip() for s in needs_txt.split(",") if s.strip()}

    p_modality = st.text_input(
        "Preferred modality (optional, e.g., Telehealth, In-person)",
        value=""
    ).strip()
    p_ins = st.text_input(
        "Insurance (optional, e.g., Aetna)",
        key=f"ins_{pid}_{version}",
        value=st.session_state.get(f"ins_{pid}", "")
    ).strip()

# Now update mask and p_state_norm for the selected patient
mask = patients["_PID"] == pid
p_state_norm = patients.loc[mask, "_STATE"].iloc[0] if mask.any() else ""

st.title("Patient → Therapist Matching (Journey schemas)")
if mask.any():
    st.markdown(
        f"**Hard matching parameters:**\n"
        f"- Therapist must be credentialed in the patient's state (**{p_state_norm}**)\n"
        f"- Therapist must be accepting new patients"
    )
else:
    st.markdown(
        "**Hard matching parameters:**\n"
        "- Therapist must be credentialed in the patient's state\n"
        "- Therapist must be accepting new patients"
    )

# Hard filters
def feasible(df, p_state_code):
    state_ok  = df["_STATES"].apply(lambda S: p_state_code in S if p_state_code else False)
    accept_ok = df["_ACCEPT"]
    return df[state_ok & accept_ok].copy()

feas = feasible(therapists, p_state_norm)

with c2:
    st.subheader("Top Matches")
    with st.spinner("Finding best therapist matches..."):
        if feas.empty:
            st.error("No therapists credentialed in the patient state (and accepting new patients).")
        else:
            rows = []
            for _, t in feas.iterrows():
                base_score = 0.1  # Minimum score for any feasible match

                tag_score  = jaccard(needs_set, t["_SPECIAL_SET"])
                if tag_score == 0.0:
                    tag_score = 0.01

                text_score = (0.0 if not HAS_SKLEARN else
                              (lambda a,b: float(cosine_similarity(TfidfVectorizer(min_df=1, stop_words="english")
                                                                   .fit_transform([a,b]))[0,1]))(notes_txt, t["_BIO"])) if w_text > 0 else 0.0
                if text_score == 0.0:
                    text_score = 0.01

                mod_boost  = 0.1 if (p_modality and (p_modality in t["_MODS"])) else 0.0
                ins_boost  = 0.1 if (p_ins and (p_ins in t["_INS"])) else 0.0
                misc       = min(mod_boost + ins_boost, 0.2)

                score      = base_score + w_tags*tag_score + w_text*text_score + w_misc*misc

                parts = [f"tags {tag_score:.2f}"]
                if w_text > 0: parts.append(f"text {text_score:.2f}")
                if mod_boost:  parts.append(f"mod:{p_modality}")
                if ins_boost:  parts.append(f"ins:{p_ins}")
                reason = " | ".join(parts)

                rows.append({
                    "therapist_id": t["_TID"],
                    "score": round(float(score), 4),
                    "tag": round(tag_score, 3),
                    "text": round(text_score, 3),
                    "misc": round(misc, 3),
                    "reason": reason
                })

            ranked = pd.DataFrame(rows).sort_values("score", ascending=False).head(3)
            st.dataframe(ranked, use_container_width=True)

            # HITL approve/reject
            if "decisions" not in st.session_state:
                st.session_state["decisions"] = []

            for _, r in ranked.iterrows():
                with st.expander(f"Therapist {r['therapist_id']} • score {r['score']}"):
                    tt = therapists.loc[therapists["_TID"] == r["therapist_id"]].iloc[0]
                    st.markdown("**States:** " + (", ".join(sorted(tt["_STATES"])) or "—"))
                    st.markdown("**Specialties:** " + (", ".join(sorted(tt["_SPECIAL_SET"])) or "—"))
                    st.markdown("**Modalities:** " + (", ".join(sorted(tt["_MODS"])) or "—"))
                    st.markdown("**Insurances:** " + (", ".join(sorted(tt["_INS"])) or "—"))
                    st.caption("Bio:")
                    st.write(tt["_BIO"][:900])

                    cA, cB = st.columns([0.2, 0.8])
                    if cA.button("Approve", key=f"approve_{pid}_{tt['_TID']}"):
                        st.session_state["decisions"].append({
                            "patient_id": pid, "therapist_id": tt["_TID"],
                            "decision": "approve", "score": r["score"], "reason": r["reason"]
                        })
                        st.success("Approved")
                    rej = cB.text_input("Reject reason", key=f"rej_{pid}_{tt['_TID']}")
                    if cB.button("Reject", key=f"reject_{pid}_{tt['_TID']}"):
                        st.session_state["decisions"].append({
                            "patient_id": pid, "therapist_id": tt["_TID"],
                            "decision": "reject", "score": r["score"], "reason": rej or "—"
                        })
                        st.warning("Rejected")

# ----------------- decisions/export -----------------
st.markdown("---")
st.subheader("Decisions")
dec = pd.DataFrame(st.session_state.get("decisions", []))
if dec.empty:
    st.info("No decisions yet.")
else:
    st.dataframe(dec, use_container_width=True)
    st.download_button(
        "Download decisions.csv",
        dec.to_csv(index=False).encode("utf-8"),
        file_name="decisions.csv",
        mime="text/csv"
    )

st.caption(
    "Rules: state credentialing + accepting new patients. "
    "Score = tags overlap + TF-IDF text similarity (+ small boosts for modality/insurance). "
    "Per-patient widget keys ensure inputs change with Patient ID."
)