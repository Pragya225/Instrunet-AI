import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import io
import base64
import json
import yaml
import os
from datetime import datetime
from yaml.loader import SafeLoader

st.set_page_config(
    page_title="InstruNet AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

QUALITY_LABELS   = ['excellent', 'good', 'fair', 'poor']
CONDITION_LABELS = ['modern', 'clean', 'noisy', 'vintage']

CONFIG_FILE = "auth_config.yaml"

def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return yaml.load(f, Loader=SafeLoader)
    try:
        seed_users = dict(st.secrets.get("seed_users", {}))
    except Exception:
        seed_users = {}
    config = {
        "credentials": {"usernames": {}},
        "cookie": {
            "name":     "instrunet_auth",
            "key":      st.secrets.get("COOKIE_KEY", "instrunet_super_secret_key_2026"),
            "expiry_days": 30
        },
        "pre-authorized": {"emails": []}
    }
    for uname, hpw in seed_users.items():
        config["credentials"]["usernames"][uname] = {
            "name": uname.capitalize(),
            "email": f"{uname}@instrunet.ai",
            "password": hpw,
            "failed_login_attempts": 0,
            "logged_in": False
        }
    save_config(config)
    return config

def save_config(config: dict):
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

@st.cache_resource(show_spinner=False)
def load_model_and_encoder():
    from huggingface_hub import hf_hub_download
    from tensorflow import keras
    token   = st.secrets["HF_TOKEN"]
    repo_id = st.secrets["HF_REPO_ID"]
    model_path   = hf_hub_download(repo_id=repo_id, filename="best_multitask_model.keras", token=token)
    encoder_path = hf_hub_download(repo_id=repo_id, filename="label_encoder.pkl", token=token)
    model = keras.models.load_model(model_path)
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

INSTRUMENT_ICONS = {
    'brass':'🎺','guitar':'🎸','keyboard':'🎹','mallet':'🪘',
    'organ':'🎹','reed':'🎷','string':'🎻','synth':'🎛️',
    'vocal':'🎤','flute':'🪈','bass':'🎸','drum':'🥁'
}
def get_icon(name):
    n = (name or '').lower()
    for k, v in INSTRUMENT_ICONS.items():
        if k in n: return v
    return '🎵'


# ─────────────────────────────────────────────────────────────────────────────
# REDESIGNED CSS — Clean, Editorial, Professional
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Fraunces:ital,wght@0,300;0,600;1,300&display=swap');

/* ── RESET & BASE ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #F6F5F1 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── COLOR TOKENS ── */
:root {
    --ink:       #1A1917;
    --ink-2:     #4A4845;
    --ink-3:     #9A9793;
    --rule:      #E2E0DB;
    --rule-2:    #D0CEC8;
    --surface:   #FFFFFF;
    --surface-2: #F6F5F1;
    --surface-3: #EDECEA;
    --accent:    #2C5282;
    --accent-lt: #EBF2FF;
    --green:     #1A6B4A;
    --green-lt:  #E6F4EE;
    --amber:     #92500A;
    --amber-lt:  #FEF3E2;
    --red:       #9B1C1C;
    --red-lt:    #FEF2F2;
    --purple:    #5B21B6;
    --purple-lt: #F3EEFF;
    --radius:    10px;
    --radius-lg: 16px;
    --shadow:    0 1px 3px rgba(26,25,23,.07), 0 4px 12px rgba(26,25,23,.05);
    --shadow-lg: 0 2px 8px rgba(26,25,23,.08), 0 12px 32px rgba(26,25,23,.08);
}

/* ── ALL TEXT ── */
* { font-family: 'Plus Jakarta Sans', sans-serif !important; color: var(--ink); }

/* ── STREAMLIT WIDGET OVERRIDES ── */
.stTextInput > div > div > input {
    background: var(--surface) !important;
    border: 1.5px solid var(--rule-2) !important;
    border-radius: var(--radius) !important;
    color: var(--ink) !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 0.9rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(44,82,130,0.12) !important;
    outline: none !important;
}
.stTextInput label {
    color: var(--ink-2) !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.02em !important;
    text-transform: uppercase !important;
}
.stCheckbox label { color: var(--ink-2) !important; font-size: 0.88rem !important; font-weight: 500 !important; }

[data-testid="stFileUploadDropzone"] {
    border: 2px dashed var(--rule-2) !important;
    border-radius: var(--radius-lg) !important;
    background: var(--surface-2) !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--accent) !important;
    background: var(--accent-lt) !important;
}
[data-testid="stFileUploadDropzone"] label,
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] span {
    color: var(--ink-3) !important;
    font-weight: 500 !important;
}

.stButton > button {
    background: var(--ink) !important;
    color: #F6F5F1 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border-radius: 9999px !important;
    border: none !important;
    padding: 0.7rem 2rem !important;
    letter-spacing: 0.01em !important;
    transition: background 0.15s, transform 0.1s !important;
}
.stButton > button:hover {
    background: #2D2C2A !important;
    transform: translateY(-1px) !important;
}
.stButton > button:disabled {
    background: var(--rule) !important;
    color: var(--ink-3) !important;
    transform: none !important;
}

.stDownloadButton > button {
    border-radius: var(--radius) !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.2rem !important;
    border: 1.5px solid var(--rule-2) !important;
    background: var(--surface) !important;
    color: var(--ink) !important;
    transition: all 0.15s !important;
    font-size: 0.85rem !important;
}
.stDownloadButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-lt) !important;
}

.stSpinner > div { border-top-color: var(--accent) !important; }
.stAlert { border-radius: var(--radius) !important; }
audio { border-radius: var(--radius); width: 100%; margin-top: 0.5rem; }

/* ── AUTH PAGE STYLES ── */
.auth-page-bg {
    min-height: 100vh;
    background: var(--surface-2);
    display: flex;
    align-items: center;
    justify-content: center;
}
.auth-grid-bg {
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        linear-gradient(var(--rule) 1px, transparent 1px),
        linear-gradient(90deg, var(--rule) 1px, transparent 1px);
    background-size: 48px 48px;
    opacity: 0.6;
}
.auth-accent-blob {
    position: fixed; pointer-events: none;
    width: 500px; height: 500px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(44,82,130,0.07) 0%, transparent 70%);
    top: -100px; right: -100px;
}

/* ── MAIN APP LAYOUT ── */
.app-topbar {
    background: var(--surface);
    border-bottom: 1px solid var(--rule);
    padding: 0.8rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
}
.app-brand {
    font-family: 'Fraunces', serif !important;
    font-size: 1.35rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--ink);
}
.app-brand span { color: var(--accent); }

.app-hero {
    background: var(--ink);
    color: #F6F5F1;
    padding: 3.5rem 2.5rem 3rem;
    position: relative;
    overflow: hidden;
}
.app-hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 350px; height: 350px;
    border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.06);
}
.app-hero::after {
    content: '';
    position: absolute;
    bottom: -60px; right: 120px;
    width: 200px; height: 200px;
    border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.04);
}
.app-hero h1 {
    font-family: 'Fraunces', serif !important;
    font-size: 3rem;
    font-weight: 300;
    letter-spacing: -0.04em;
    line-height: 1.05;
    color: #F6F5F1;
    margin-bottom: 0.75rem;
}
.app-hero h1 em {
    font-style: italic;
    color: rgba(246,245,241,0.5);
}
.app-hero p {
    font-size: 0.9rem;
    color: rgba(246,245,241,0.55);
    font-weight: 400;
    margin-bottom: 1.25rem;
}
.app-hero-pills {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.app-hero-pill {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 9999px;
    padding: 0.22rem 0.75rem;
    font-size: 0.7rem;
    font-weight: 600;
    color: rgba(246,245,241,0.75);
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.app-content { padding: 2rem 2.5rem; }
.app-section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-3);
    margin-bottom: 0.75rem;
}

/* ── CARD COMPONENT ── */
.card {
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: var(--shadow);
}
.card-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--ink-3);
    margin-bottom: 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--rule);
}

/* ── RESULT HERO CARD ── */
.result-hero {
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius-lg);
    overflow: hidden;
    margin-bottom: 1.25rem;
    box-shadow: var(--shadow-lg);
}
.result-hero-banner {
    background: var(--ink);
    padding: 2rem 2rem 1.75rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.result-hero-icon {
    font-size: 3rem;
    line-height: 1;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    width: 72px; height: 72px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.result-hero-name {
    font-family: 'Fraunces', serif !important;
    font-size: 2.2rem;
    font-weight: 300;
    letter-spacing: -0.03em;
    color: #F6F5F1;
    text-transform: capitalize;
    line-height: 1.1;
}
.result-hero-sub {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(246,245,241,0.4);
    margin-top: 0.3rem;
}
.result-hero-conf {
    margin-top: 0.55rem;
    font-size: 0.82rem;
    font-weight: 600;
}
.conf-high  { color: #4ADE80; }
.conf-mid   { color: #FCD34D; }
.conf-low   { color: #F87171; }
.result-hero-body { padding: 1.5rem 2rem; }

/* ── PREDICTION CARDS GRID ── */
.pred-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-top: 0.5rem;
}
.pred-card {
    border: 1px solid var(--rule);
    border-radius: var(--radius);
    padding: 1rem;
    background: var(--surface-2);
    transition: border-color 0.15s;
}
.pred-card.is-top {
    border-color: var(--accent);
    background: var(--accent-lt);
}
.pred-card-rank {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink-3);
    margin-bottom: 0.5rem;
}
.pred-card.is-top .pred-card-rank { color: var(--accent); }
.pred-card-name {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--ink);
    text-transform: capitalize;
    margin-bottom: 0.65rem;
}
.pred-card-pct {
    font-family: 'Fraunces', serif !important;
    font-size: 1.6rem;
    font-weight: 300;
    color: var(--ink);
    letter-spacing: -0.02em;
    line-height: 1;
}
.pred-card.is-top .pred-card-pct { color: var(--accent); }

/* ── BARS ── */
.bar-row { margin-bottom: 0.6rem; }
.bar-meta {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 0.8rem;
    margin-bottom: 0.28rem;
    font-weight: 500;
    color: var(--ink-2);
}
.bar-meta span:last-child { font-weight: 700; font-size: 0.78rem; color: var(--ink); }
.bar-track {
    height: 6px;
    background: var(--surface-3);
    border-radius: 9999px;
    overflow: hidden;
}
.bar-fill {
    height: 6px;
    border-radius: 9999px;
    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── QUALITY BADGE ── */
.quality-badge {
    display: inline-block;
    padding: 0.25rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.qb-excellent { background: var(--green-lt);  color: var(--green); }
.qb-good      { background: var(--accent-lt); color: var(--accent); }
.qb-fair      { background: var(--amber-lt);  color: var(--amber); }
.qb-poor      { background: var(--red-lt);    color: var(--red); }

/* ── CONDITION TAGS ── */
.cond-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1.5px solid;
}
.cond-modern  { background: var(--accent-lt); border-color: #B9D4F7; color: var(--accent); }
.cond-vintage { background: var(--amber-lt);  border-color: #F9CC83; color: var(--amber); }
.cond-clean   { background: var(--green-lt);  border-color: #86CFAF; color: var(--green); }
.cond-noisy   { background: var(--red-lt);    border-color: #F9C4C4; color: var(--red); }

/* ── STAT TILES ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
}
.stat-tile {
    background: var(--surface-2);
    border: 1px solid var(--rule);
    border-radius: var(--radius);
    padding: 1.1rem 1.2rem;
}
.stat-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; color: var(--ink-3); margin-bottom: 0.4rem; }
.stat-value { font-family: 'Fraunces', serif !important; font-size: 1.55rem; font-weight: 300; color: var(--ink); letter-spacing: -0.02em; }
.stat-unit  { font-size: 0.7rem; font-weight: 600; color: var(--ink-3); margin-top: 0.12rem; }

/* ── UPLOAD CONFIRMED ── */
.upload-confirmed {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 0.75rem 1rem;
    background: var(--green-lt);
    border: 1px solid #86CFAF;
    border-radius: var(--radius);
    margin-top: 0.75rem;
}
.upload-filename { font-weight: 600; font-size: 0.85rem; color: var(--green); }
.upload-size { font-size: 0.72rem; color: var(--green); opacity: 0.75; margin-top: 0.1rem; }

/* ── VIZ IMAGES ── */
.viz-img {
    width: 100%;
    border-radius: var(--radius);
    border: 1px solid var(--rule);
}

/* ── FOOTER ── */
.app-footer {
    text-align: center;
    padding: 2rem;
    border-top: 1px solid var(--rule);
    font-size: 0.75rem;
    color: var(--ink-3);
    margin-top: 1rem;
}

/* ── AUTH CARD ── */
.auth-wordmark {
    font-family: 'Fraunces', serif !important;
    font-size: 1.6rem; font-weight: 600;
    letter-spacing: -0.03em;
    margin-bottom: 0.2rem;
    color: var(--ink);
}
.auth-wordmark span { color: var(--accent); }
.auth-tagline {
    font-size: 0.75rem; font-weight: 500;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--ink-3);
}

/* ── RESPONSIVE ── */
@media (max-width: 640px) {
    .pred-grid { grid-template-columns: 1fr; }
    .stat-grid { grid-template-columns: 1fr 1fr; }
    .app-hero h1 { font-size: 2rem; }
    .app-content { padding: 1.25rem 1rem; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    for k, v in {
        'auth_page': 'login',
        'result': None,
        'auth_status': None,
        'auth_name': '',
        'auth_username': '',
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_authenticator():
    import streamlit_authenticator as stauth
    config = load_config()
    auth = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
    )
    return auth, config

def hash_password(plain: str) -> str:
    import bcrypt
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# AUTH PAGE
# ─────────────────────────────────────────────────────────────────────────────
def render_auth():
    st.markdown("""
<div class="auth-grid-bg"></div>
<div class="auth-accent-blob"></div>
""", unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("""
<div style="text-align:center; padding: 2.5rem 0 1.5rem; position: relative; z-index: 1;">
  <div style="width:52px;height:52px;background:#1A1917;border-radius:12px;display:inline-flex;align-items:center;justify-content:center;font-size:1.4rem;margin-bottom:1rem;">🎵</div>
  <div class="auth-wordmark">Instru<span>Net</span></div>
  <div class="auth-tagline">AI Instrument Recognition</div>
</div>
""", unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("""
<div style="background:#fff;border:1px solid #E2E0DB;border-radius:16px;padding:2rem;
     box-shadow:0 4px 24px rgba(26,25,23,.08);position:relative;z-index:1;">
""", unsafe_allow_html=True)

        active_login = st.session_state.auth_page == 'login'
        t1, t2 = st.columns(2)
        with t1:
            if st.button("Sign In", key="tab_login", use_container_width=True,
                         type="primary" if active_login else "secondary"):
                st.session_state.auth_page = 'login'; st.rerun()
        with t2:
            if st.button("Register", key="tab_reg", use_container_width=True,
                         type="primary" if not active_login else "secondary"):
                st.session_state.auth_page = 'register'; st.rerun()

        st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

        if st.session_state.auth_page == 'login':
            st.markdown("""
<p style="font-family:'Fraunces',serif;font-size:1.15rem;font-weight:300;letter-spacing:-.02em;margin:0 0 .1rem">Welcome back</p>
<p style="font-size:.78rem;color:#9A9793;margin:0 0 1.25rem">Sign in to your account to continue.</p>
""", unsafe_allow_html=True)

            auth, config = get_authenticator()
            login_result = auth.login(
                fields={'Form name':'Sign In','Username':'Username',
                        'Password':'Password','Login':'Sign In'},
                key='login_widget'
            )
            if isinstance(login_result, tuple) and len(login_result) == 3:
                name, auth_status, username = login_result
            else:
                name, auth_status, username = None, None, None

            if auth_status is True:
                st.session_state.auth_status   = True
                st.session_state.auth_name     = name
                st.session_state.auth_username = username
                save_config(config)
                st.rerun()
            elif auth_status is False:
                st.error("Incorrect username or password.")

            st.markdown('<p style="text-align:center;font-size:.74rem;color:#9A9793;margin-top:.75rem">No account? Click <strong>Register</strong> above.</p>', unsafe_allow_html=True)

        else:
            st.markdown("""
<p style="font-family:'Fraunces',serif;font-size:1.15rem;font-weight:300;letter-spacing:-.02em;margin:0 0 .1rem">Create an account</p>
<p style="font-size:.78rem;color:#9A9793;margin:0 0 1.25rem">Join InstruNet AI and start analyzing.</p>
""", unsafe_allow_html=True)

            new_name  = st.text_input("Full Name",        placeholder="Alex Johnson",    key="reg_name")
            new_user  = st.text_input("Username",         placeholder="alexj",           key="reg_user")
            new_email = st.text_input("Email",            placeholder="you@example.com", key="reg_email")
            new_pass  = st.text_input("Password",         placeholder="Min. 6 characters", type="password", key="reg_pass")
            new_pass2 = st.text_input("Confirm Password", placeholder="Repeat password",   type="password", key="reg_pass2")

            if st.button("Create Account", use_container_width=True, key="reg_submit"):
                config = load_config()
                users  = config["credentials"]["usernames"]
                err = None
                if not all([new_name, new_user, new_email, new_pass, new_pass2]):
                    err = "Please fill in all fields."
                elif new_user in users:
                    err = f"Username '{new_user}' is already taken."
                elif any(u.get('email') == new_email for u in users.values()):
                    err = "An account with this email already exists."
                elif len(new_pass) < 6:
                    err = "Password must be at least 6 characters."
                elif new_pass != new_pass2:
                    err = "Passwords do not match."

                if err:
                    st.error(err)
                else:
                    hashed = hash_password(new_pass)
                    config["credentials"]["usernames"][new_user] = {
                        "name": new_name, "email": new_email,
                        "password": hashed, "failed_login_attempts": 0, "logged_in": False
                    }
                    save_config(config)
                    st.success("Account created! Sign in to continue.")
                    st.balloons()

            st.markdown('<p style="text-align:center;font-size:.74rem;color:#9A9793;margin-top:.75rem">Have an account? Click <strong>Sign In</strong> above.</p>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<p style="text-align:center;font-size:.72rem;color:#C8C6C0;margin-top:1rem;position:relative;z-index:1;">Passwords are bcrypt-hashed · InstruNet AI © 2026</p>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def process_audio_file(audio_bytes, sr=22050, duration=3.0):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr, duration=duration, mono=True)
    target = int(sr * duration)
    y = np.pad(y, (0, max(0, target - len(y))), mode='constant')[:target]
    return y, sr

def extract_mel_spectrogram(y, sr, n_mels=128, n_fft=2048, hop_length=512):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    if mel_norm.shape[1] > 128:
        mel_norm = mel_norm[:, :128]
    elif mel_norm.shape[1] < 128:
        mel_norm = np.pad(mel_norm, ((0,0),(0, 128 - mel_norm.shape[1])), mode='constant')
    return mel_norm


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def fig_to_b64(fig):
    fig.patch.set_facecolor('#FFFFFF')
    for ax in fig.axes:
        ax.set_facecolor('#FFFFFF')
        ax.tick_params(colors='#9A9793', labelsize=9)
        ax.xaxis.label.set_color('#4A4845')
        ax.yaxis.label.set_color('#4A4845')
        ax.title.set_color('#1A1917')
        for sp in ax.spines.values():
            sp.set_edgecolor('#E2E0DB')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor='#FFFFFF')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"

def gen_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.plot(np.arange(len(y))/sr, y, color='#2C5282', linewidth=0.6, alpha=0.85)
    ax.fill_between(np.arange(len(y))/sr, y, alpha=0.08, color='#2C5282')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.set_title('Waveform', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_xlim(0, len(y)/sr)
    plt.tight_layout()
    return fig_to_b64(fig)

def gen_spectrogram(mel, sr):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    img = librosa.display.specshow(mel, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='Blues')
    ax.set_title('Mel-Spectrogram', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Frequency (Hz)', fontsize=10)
    cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.set_label('Intensity (dB)', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='#9A9793', labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#9A9793')
    plt.tight_layout()
    return fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_audio(audio_bytes, model, label_encoder):
    y, sr = process_audio_file(audio_bytes)
    mel   = extract_mel_spectrogram(y, sr)
    preds = model.predict(mel.reshape(1,128,128,1), verbose=0)
    inst_p = preds['instrument'][0]
    top3   = [{'name': label_encoder.classes_[i], 'confidence': float(inst_p[i]*100)}
               for i in np.argsort(inst_p)[::-1][:3]]
    qi = np.argmax(preds['quality'][0])
    quality = {
        'label': QUALITY_LABELS[qi],
        'confidence': float(preds['quality'][0][qi]*100),
        'all_scores': {QUALITY_LABELS[i]: float(preds['quality'][0][i]*100) for i in range(4)}
    }
    ci = np.argmax(preds['condition'][0])
    condition = {
        'label': CONDITION_LABELS[ci],
        'confidence': float(preds['condition'][0][ci]*100),
        'all_scores': {CONDITION_LABELS[i]: float(preds['condition'][0][i]*100) for i in range(4)}
    }
    return {
        'instrument': top3[0], 'top_instruments': top3,
        'quality': quality, 'condition': condition,
        'visualizations': {'waveform': gen_waveform(y, sr), 'spectrogram': gen_spectrogram(mel, sr)},
        'audio_info': {'duration': float(len(y)/sr), 'sample_rate': int(sr), 'samples': len(y)},
        'timestamp': datetime.now().isoformat()
    }


# ─────────────────────────────────────────────────────────────────────────────
# PDF EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf_bytes(result):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.units import inch
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    story, styles = [], getSampleStyleSheet()
    ts = ParagraphStyle('T', parent=styles['Heading1'], fontSize=22,
                        textColor=colors.HexColor('#1A1917'), spaceAfter=26, alignment=1)
    story.append(Paragraph("Audio Analysis Report — InstruNet AI", ts))
    story.append(Spacer(1, .28*inch))
    story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, .2*inch))
    def mkt(data, cw, hc):
        t = Table(data, colWidths=cw)
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor(hc)),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,0),11),
            ('BOTTOMPADDING',(0,0),(-1,0),10),
            ('BACKGROUND',(0,1),(-1,-1),colors.beige),
            ('GRID',(0,0),(-1,-1),1,colors.HexColor('#E2E0DB')),
        ]))
        return t
    ai = result['audio_info']
    for heading, data, cw, hc in [
        ("INSTRUMENT PREDICTION",
         [['Instrument','Confidence'],[result['instrument']['name'].upper(), f"{result['instrument']['confidence']:.1f}%"]],
         [3*inch,2*inch],'#2C5282'),
        ("AUDIO QUALITY",
         [['Quality','Confidence'],[result['quality']['label'].upper(), f"{result['quality']['confidence']:.1f}%"]],
         [3*inch,2*inch],'#1A6B4A'),
        ("AUDIO CONDITION",
         [['Condition','Confidence'],[result['condition']['label'].upper(), f"{result['condition']['confidence']:.1f}%"]],
         [3*inch,2*inch],'#92500A'),
        ("AUDIO INFORMATION",
         [['Property','Value'],['Duration',f"{ai['duration']:.2f}s"],
          ['Sample Rate',f"{ai['sample_rate']} Hz"],['Samples',f"{ai['samples']:,}"]],
         [3*inch,2*inch],'#4A4845'),
    ]:
        story.append(Paragraph(f"<b>{heading}</b>", styles['Heading2']))
        story.append(mkt(data, cw, hc)); story.append(Spacer(1,.25*inch))
    story.append(Paragraph("<b>Top 3 Predictions</b>", styles['Heading3']))
    t3 = [['Rank','Instrument','Confidence']]
    for i, x in enumerate(result['top_instruments'],1):
        t3.append([str(i), x['name'].capitalize(), f"{x['confidence']:.1f}%"])
    story.append(mkt(t3,[1*inch,2*inch,2*inch],'#5B21B6'))
    doc.build(story); buf.seek(0); return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — RESULT SECTIONS
# ─────────────────────────────────────────────────────────────────────────────
def bar_html(value, color, height=6):
    return (f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{value:.1f}%;background:{color};height:{height}px;"></div></div>')

def render_results(r):
    inst = r['instrument']
    icon = get_icon(inst['name'])
    conf = inst['confidence']
    conf_cls = 'conf-high' if conf >= 80 else 'conf-mid' if conf >= 50 else 'conf-low'
    conf_lbl = 'High confidence' if conf >= 80 else 'Moderate confidence' if conf >= 50 else 'Low confidence'

    # Top prediction cards
    cards_html = ''
    for i, item in enumerate(r['top_instruments']):
        top_cls = 'is-top' if i == 0 else ''
        cards_html += f"""
<div class="pred-card {top_cls}">
  <div class="pred-card-rank">{'Top pick' if i==0 else f'#{i+1}'}</div>
  <div style="font-size:1.5rem;margin-bottom:.5rem">{get_icon(item['name'])}</div>
  <div class="pred-card-name">{item['name']}</div>
  <div class="pred-card-pct">{item['confidence']:.1f}%</div>
  <div style="margin-top:.65rem">{bar_html(item['confidence'], '#2C5282' if i==0 else '#D0CEC8')}</div>
</div>"""

    st.markdown(f"""
<div class="result-hero">
  <div class="result-hero-banner">
    <div class="result-hero-icon">{icon}</div>
    <div>
      <div class="result-hero-name">{inst['name']}</div>
      <div class="result-hero-sub">Detected Instrument</div>
      <div class="result-hero-conf {conf_cls}">{conf:.1f}% · {conf_lbl}</div>
    </div>
  </div>
  <div class="result-hero-body">
    <div class="app-section-label">All predictions</div>
    <div class="pred-grid">{cards_html}</div>
  </div>
</div>
""", unsafe_allow_html=True)

def render_quality_card(q):
    lbl  = q['label']
    conf = q['confidence']
    badge_cls = f'qb-{lbl.lower()}'
    COLORS = {
        'excellent': '#1A6B4A',
        'good':      '#2C5282',
        'fair':      '#92500A',
        'poor':      '#9B1C1C',
    }
    bars = ''
    for l, s in q['all_scores'].items():
        is_active = l == lbl
        col = COLORS.get(l, '#9A9793')
        fw = 'font-weight:700' if is_active else ''
        bars += f"""
<div class="bar-row">
  <div class="bar-meta">
    <span style="{fw};text-transform:capitalize">{l}</span>
    <span>{s:.1f}%</span>
  </div>
  {bar_html(s, col if is_active else '#D0CEC8')}
</div>"""

    st.markdown(f"""
<div class="card" style="height:100%">
  <div class="card-title">Audio Quality</div>
  <div style="display:flex;align-items:center;gap:.75rem;margin-bottom:1.25rem">
    <span class="quality-badge {badge_cls}">{lbl.upper()}</span>
    <span style="font-size:.82rem;color:#4A4845;font-weight:600">{conf:.1f}% confidence</span>
  </div>
  {bars}
</div>
""", unsafe_allow_html=True)

def render_condition_card(cond):
    cs  = cond.get('all_scores', {})
    lbl = cond['label']
    modern  = cs.get('modern', 0)  + cs.get('clean', 0)
    vintage = cs.get('vintage', 0) + cs.get('noisy', 0)
    total   = modern + vintage or 1
    is_modern  = modern >= vintage
    is_clean   = cs.get('clean', 0) >= cs.get('noisy', 0)

    era_pill = ('cond-modern' if is_modern else 'cond-vintage')
    era_label = ('Modern' if is_modern else 'Vintage')
    sig_pill = ('cond-clean' if is_clean else 'cond-noisy')
    sig_label = ('Clean signal' if is_clean else 'Noisy signal')

    bars = ''
    pairs = [('Modern era', (modern/total)*100, '#2C5282'), ('Vintage era', (vintage/total)*100, '#92500A')]
    for l, v, c in pairs:
        is_a = (l.startswith('Modern') and is_modern) or (l.startswith('Vintage') and not is_modern)
        bars += f"""
<div class="bar-row">
  <div class="bar-meta">
    <span style="{'font-weight:700' if is_a else ''}">{l}</span>
    <span>{v:.1f}%</span>
  </div>
  {bar_html(v, c if is_a else '#D0CEC8', 8)}
</div>"""

    clean_v = cs.get('clean', 0)
    noisy_v = cs.get('noisy', 0)
    sig_bars = ''
    for l, v, c in [('Clean', clean_v, '#1A6B4A'), ('Noisy', noisy_v, '#9B1C1C')]:
        sig_bars += f"""
<div class="bar-row">
  <div class="bar-meta"><span>{l}</span><span>{v:.1f}%</span></div>
  {bar_html(v, c, 6)}
</div>"""

    st.markdown(f"""
<div class="card" style="height:100%">
  <div class="card-title">Recording Condition</div>
  <div style="display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1.25rem">
    <span class="cond-pill {era_pill}">{era_label}</span>
    <span class="cond-pill {sig_pill}">{sig_label}</span>
  </div>
  <div class="app-section-label" style="margin-bottom:.6rem">Recording Era</div>
  {bars}
  <div class="app-section-label" style="margin-top:1rem;margin-bottom:.6rem">Signal Quality</div>
  {sig_bars}
</div>
""", unsafe_allow_html=True)

def render_info_card(result):
    ai = result['audio_info']
    st.markdown(f"""
<div class="card">
  <div class="card-title">Audio Information</div>
  <div class="stat-grid">
    <div class="stat-tile">
      <div class="stat-label">Duration</div>
      <div class="stat-value">{ai['duration']:.2f}</div>
      <div class="stat-unit">seconds</div>
    </div>
    <div class="stat-tile">
      <div class="stat-label">Sample Rate</div>
      <div class="stat-value">{ai['sample_rate']:,}</div>
      <div class="stat-unit">Hz</div>
    </div>
    <div class="stat-tile">
      <div class="stat-label">Total Samples</div>
      <div class="stat-value">{ai['samples']//1000}k</div>
      <div class="stat-unit">samples</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("↓ Download JSON",
            data=json.dumps({k:v for k,v in result.items() if k!='visualizations'}, indent=2),
            file_name=f"instrunet_{ts}.json", mime="application/json", use_container_width=True)
    with c2:
        st.download_button("↓ Download PDF",
            data=generate_pdf_bytes(result),
            file_name=f"instrunet_{ts}.pdf", mime="application/pdf", use_container_width=True)

def render_viz_card(result):
    st.markdown("""
<div class="card">
  <div class="card-title">Visualizations</div>
  <div class="app-section-label" style="margin-bottom:.5rem">Waveform</div>
""", unsafe_allow_html=True)
    st.markdown(f'<img src="{result["visualizations"]["waveform"]}" class="viz-img" style="margin-bottom:1.25rem"/>', unsafe_allow_html=True)
    st.markdown('<div class="app-section-label" style="margin-bottom:.5rem">Mel-Spectrogram</div>', unsafe_allow_html=True)
    st.markdown(f'<img src="{result["visualizations"]["spectrogram"]}" class="viz-img"/>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def render_app():
    load_model_and_encoder()

    # ── TOPBAR ──
    name = st.session_state.get('auth_name') or st.session_state.get('name', 'User')
    col_brand, col_user, col_out = st.columns([4, 2.5, 1])
    with col_brand:
        st.markdown("""
<div class="app-topbar" style="border-bottom:1px solid #E2E0DB;padding:.8rem 2.5rem">
  <span class="app-brand">Instru<span>Net</span></span>
</div>
""", unsafe_allow_html=True)
    with col_user:
        st.markdown(f"""
<div style="display:flex;align-items:center;height:100%;padding:.5rem 0;
     font-size:.8rem;color:#9A9793;border-bottom:1px solid #E2E0DB">
  <span style="color:#4A4845;font-weight:600">{name}</span>
</div>""", unsafe_allow_html=True)
    with col_out:
        if st.button("Sign out", key="signout_btn"):
            st.session_state.auth_status   = None
            st.session_state.auth_name     = ''
            st.session_state.auth_username = ''
            for k in ["authentication_status", "name", "username", "logout"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    # ── HERO ──
    st.markdown("""
<div class="app-hero">
  <h1>Instrument<br><em>Recognition</em></h1>
  <p>Upload an audio file — the model identifies the instrument, grades quality, and classifies recording condition.</p>
  <div class="app-hero-pills">
    <span class="app-hero-pill">Instrument</span>
    <span class="app-hero-pill">Quality</span>
    <span class="app-hero-pill">Condition</span>
    <span class="app-hero-pill">Multi-task CNN</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── UPLOAD ──
    st.markdown('<div class="app-content">', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">Upload Audio</div>', unsafe_allow_html=True)

    af = st.file_uploader(
        "Drag and drop or click to browse — WAV · MP3 · OGG · FLAC — first 3 seconds analyzed",
        type=['wav','mp3','ogg','flac','m4a'],
        label_visibility='visible'
    )

    if af:
        st.audio(af, format=af.type)
        st.markdown(f"""
<div class="upload-confirmed">
  <span style="font-size:1.1rem">✓</span>
  <div>
    <div class="upload-filename">{af.name}</div>
    <div class="upload-size">{af.size/1024:.1f} KB</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── ANALYZE BUTTON ──
    _, cc, _ = st.columns([3, 2, 3])
    with cc:
        go = st.button("Analyze Audio →", disabled=af is None, use_container_width=True)

    if go and af:
        with st.spinner("Analyzing audio…"):
            model, le = load_model_and_encoder()
            result = predict_audio(af.read(), model, le)
        st.session_state.result = result

    # ── RESULTS ──
    if st.session_state.result:
        res = st.session_state.result
        st.markdown('<div style="margin-top:.75rem">', unsafe_allow_html=True)
        render_results(res)
        c1, c2 = st.columns(2)
        with c1: render_quality_card(res['quality'])
        with c2: render_condition_card(res['condition'])
        render_info_card(res)
        render_viz_card(res)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-footer">© 2026 InstruNet AI · Multi-task CNN · Instrument · Quality · Condition</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    init_state()
    inject_css()

    is_auth = (st.session_state.get("authentication_status") is True
               or st.session_state.auth_status is True)

    if is_auth:
        render_app()
    else:
        render_auth()


if __name__ == '__main__':
    main()
