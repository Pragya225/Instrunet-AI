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

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InstruNet AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

QUALITY_LABELS   = ['excellent', 'good', 'fair', 'poor']
CONDITION_LABELS = ['modern', 'clean', 'noisy', 'vintage']

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG FILE
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
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
# CSS  — fully rewritten, fixes all layout & visual issues
# ─────────────────────────────────────────────────────────────────────────────
def inject_css(dark: bool):
    if dark:
        bg        = "#0d0d14"
        surface   = "#15151f"
        surface2  = "#1c1c28"
        border    = "#2a2a3d"
        text      = "#e4e4f0"
        muted     = "#64648a"
        accent    = "#7c6af7"
        accent2   = "#a78bfa"
        hgrad     = "linear-gradient(135deg,#1e1b4b 0%,#312e81 100%)"
        shadow    = "0 8px 32px rgba(0,0,0,.55)"
        btrk      = "#252535"
        snbg      = "#1a0c0c"; snbd = "#7f1d1d"
        scbg      = "#0c1a15"; scbd = "#064e3b"
        fbg       = "#08080e"
        lcrd      = "#13131d"; lbrd = "#25253a"
        ibg       = "#1a1a28"; ibd  = "#35355a"
        ifsh      = "rgba(124,106,247,.25)"
        tblue     = "#0d1120"; tpurp = "#120d20"; tindi = "#0e1020"
        lbg_body  = "radial-gradient(ellipse at 30% 20%, #1e1b4b 0%, #0d0d14 55%, #1a0d2e 100%)"
    else:
        bg        = "#f4f4fb"
        surface   = "#ffffff"
        surface2  = "#f7f7fc"
        border    = "#e2e2ee"
        text      = "#111118"
        muted     = "#8888aa"
        accent    = "#5b50e8"
        accent2   = "#764ba2"
        hgrad     = "linear-gradient(135deg,#667eea 0%,#764ba2 100%)"
        shadow    = "0 10px 40px rgba(0,0,0,.08)"
        btrk      = "#e2e2ee"
        snbg      = "#fff1f2"; snbd = "#fecaca"
        scbg      = "#f0fff9"; scbd = "#99f6e4"
        fbg       = "#111118"
        lcrd      = "#ffffff"; lbrd = "#e2e2ee"
        ibg       = "#ffffff"; ibd  = "#c8c8dd"
        ifsh      = "rgba(91,80,232,.18)"
        tblue     = "#eef3ff"; tpurp = "#f3eeff"; tindi = "#eef0ff"
        lbg_body  = "radial-gradient(ellipse at 30% 20%, #667eea 0%, #4f46e5 40%, #764ba2 100%)"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&display=swap');

/* ── RESET & BASE ── */
*, *::before, *::after {{
  font-family: 'DM Sans', sans-serif !important;
  box-sizing: border-box;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}
section[data-testid="stSidebar"] {{ display: none !important; }}
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {{
  background: {bg} !important;
  min-height: 100vh;
}}

/* ── AUTH PAGE BACKGROUND ── */
.auth-bg-override [data-testid="stAppViewContainer"],
.auth-bg-override [data-testid="stMain"] {{
  background: transparent !important;
}}

/* ── AUTH PAGE: fixed full-screen gradient behind everything ── */
.auth-backdrop {{
  position: fixed;
  inset: 0;
  z-index: 0;
  background: {lbg_body};
  pointer-events: none;
}}

/* Animated orbs — contained, no overflow scroll ── */
.orb {{
  position: absolute;
  border-radius: 50%;
  filter: blur(80px);
  opacity: .22;
  pointer-events: none;
  animation: orbDrift 16s ease-in-out infinite alternate;
}}
.orb1 {{ width:340px; height:340px; background:#7c6af7; top:-100px;  left:-60px;   animation-delay:0s; }}
.orb2 {{ width:240px; height:240px; background:#a855f7; bottom:-60px; right:-40px;  animation-delay:-6s; }}
.orb3 {{ width:150px; height:150px; background:#3b82f6; top:42%;     right:14%;    animation-delay:-11s; }}
@keyframes orbDrift {{
  from {{ transform: translate(0,0) scale(1); }}
  to   {{ transform: translate(18px,24px) scale(1.06); }}
}}

/* ── AUTH CARD OUTER WRAPPER ── */
.auth-outer {{
  position: relative;
  z-index: 10;
  display: flex;
  flex-direction: column;
  align-items: center;
  min-height: 100vh;
  padding: 2rem 1rem 3rem;
}}

/* ── AUTH LOGO ── */
.auth-logo {{
  text-align: center;
  margin-bottom: 1.6rem;
}}
.auth-logo-icon {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 64px; height: 64px;
  border-radius: 1.1rem;
  font-size: 1.8rem;
  background: linear-gradient(135deg, {accent}, {accent2});
  box-shadow: 0 8px 28px {accent}55;
  margin-bottom: .75rem;
}}
.auth-brand {{
  font-family: 'Syne', sans-serif !important;
  font-size: 1.65rem;
  font-weight: 800;
  color: #ffffff;
  letter-spacing: -.04em;
  margin: 0;
  text-shadow: 0 2px 12px rgba(0,0,0,.3);
}}
.auth-brand span {{ color: {accent2}; }}
.auth-tag {{
  font-size: .69rem;
  color: rgba(255,255,255,.55);
  margin: .28rem 0 0;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: .08em;
}}

/* ── AUTH CARD ── */
.auth-card {{
  background: {lcrd};
  border: 1px solid {lbrd};
  border-radius: 1.4rem;
  padding: 2.2rem 2rem 1.8rem;
  width: 100%;
  max-width: 420px;
  box-shadow: 0 28px 80px rgba(0,0,0,.38);
  animation: cardIn .55s cubic-bezier(.16,1,.3,1) both;
}}
@keyframes cardIn {{
  from {{ opacity:0; transform: translateY(22px) scale(.97); }}
  to   {{ opacity:1; transform: none; }}
}}

/* ── TAB SWITCHER ── */
.auth-tabs {{
  display: flex;
  background: {"#1a1a2e" if dark else "#ededf8"};
  border-radius: .85rem;
  padding: .28rem;
  gap: .25rem;
  margin-bottom: 1.4rem;
  border: 1px solid {border};
}}
.auth-tab {{
  flex: 1;
  padding: .52rem .5rem;
  border-radius: .62rem;
  border: none;
  cursor: pointer;
  font-weight: 600;
  font-size: .84rem;
  transition: all .2s;
  background: transparent;
  color: {muted};
  font-family: 'DM Sans', sans-serif;
}}
.auth-tab.active {{
  background: {accent};
  color: white;
  box-shadow: 0 4px 14px {accent}44;
}}

/* ── FORM SECTION TITLES ── */
.form-title {{
  font-family: 'Syne', sans-serif !important;
  font-size: .97rem;
  font-weight: 700;
  color: {text};
  margin: 0 0 .12rem;
}}
.form-sub {{
  font-size: .77rem;
  color: {muted};
  margin: 0 0 1.1rem;
}}

/* ── THEME TOGGLE (auth) ── */
.auth-theme-row {{
  text-align: center;
  margin-top: 1.1rem;
}}
.auth-footer {{
  text-align: center;
  font-size: .69rem;
  color: rgba(255,255,255,.25);
  margin-top: .6rem;
}}

/* ════════════════════════════════════
   MAIN APP LAYOUT
   ════════════════════════════════════ */
.hdr {{
  background: {hgrad};
  color: white;
  padding: 1.3rem 2rem 1.4rem;
}}
.hdr h1 {{
  font-family: 'Syne', sans-serif !important;
  font-size: 1.9rem;
  font-weight: 800;
  margin: 0 0 .1rem;
  letter-spacing: -.04em;
  line-height: 1.1;
}}
.hdr h1 span {{ opacity: .5; font-weight: 400; }}
.hdr p {{ color: rgba(255,255,255,.6); margin: 0; font-size: .85rem; }}
.tag-row {{ display:flex; gap:.4rem; margin-top:.6rem; flex-wrap:wrap; }}
.tag-pill {{
  background: rgba(255,255,255,.15);
  border: 1px solid rgba(255,255,255,.22);
  border-radius: 9999px;
  padding: .16rem .7rem;
  font-size: .69rem;
  font-weight: 600;
  color: white;
  letter-spacing: .04em;
}}

/* ── USER BAR ── */
.ubar {{
  display: flex;
  align-items: center;
  gap: .5rem;
  padding: .55rem 2rem;
  background: {surface2};
  border-bottom: 1px solid {border};
  flex-wrap: wrap;
}}
.ubar-left {{
  flex: 1;
  min-width: 160px;
  font-size: .79rem;
  color: {muted};
}}
.ubar-left strong {{ color: {text}; }}

/* ── MAIN CONTENT WRAPPER ── */
.wrap {{
  padding: 1.5rem 2rem 2rem;
  background: {bg};
}}
@media (max-width: 640px) {{
  .wrap {{ padding: 1rem; }}
  .ubar {{ padding: .5rem 1rem; }}
  .hdr {{ padding: 1rem; }}
  .hdr h1 {{ font-size: 1.5rem; }}
}}

/* ── CARD ── */
.card {{
  background: {surface};
  border: 1px solid {border};
  border-radius: 1rem;
  box-shadow: {shadow};
  padding: 1.3rem;
  margin-bottom: 1.3rem;
}}
.card h3 {{
  font-family: 'Syne', sans-serif !important;
  font-size: .94rem;
  font-weight: 700;
  color: {text};
  margin: 0 0 1rem;
}}

/* ── PROGRESS BARS ── */
.bar-row {{ margin-bottom: .62rem; }}
.bar-label {{
  display: flex;
  justify-content: space-between;
  font-size: .77rem;
  margin-bottom: .18rem;
}}
.bar-track {{
  background: {btrk};
  border-radius: 9999px;
  overflow: hidden;
}}
.bar-fill {{
  border-radius: 9999px;
  transition: width 1.2s cubic-bezier(.4,0,.2,1);
}}

/* ── QUALITY BADGE ── */
.qbadge {{
  display: inline-block;
  padding: .28rem .85rem;
  border-radius: 9999px;
  font-weight: 700;
  font-size: .72rem;
  letter-spacing: .05em;
}}
.q-excellent {{ background:#10b981; color:white; }}
.q-good      {{ background:#3b82f6; color:white; }}
.q-fair      {{ background:#f59e0b; color:white; }}
.q-poor      {{ background:#ef4444; color:white; }}

/* ── HERO SECTION ── */
.hero-wrap {{
  background: {"linear-gradient(135deg,#13132a,#1a1635,#13132a)" if dark else "linear-gradient(135deg,#eff0ff,#f5f0ff,#eef0ff)"};
  border-radius: .9rem;
  padding: 1.8rem 1.5rem;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 2rem;
  margin-bottom: 1.2rem;
  text-align: center;
  border: 1px solid {border};
}}
.hero-icon {{ font-size: 4rem; line-height: 1; margin-bottom: .3rem; }}
.hero-name {{
  font-family: 'Syne', sans-serif !important;
  font-size: clamp(1.8rem, 5vw, 2.6rem);
  font-weight: 800;
  color: {text};
  letter-spacing: -.04em;
  line-height: 1;
  word-break: break-word;
}}
.hero-sub {{
  font-size: .67rem;
  font-weight: 600;
  color: {muted};
  text-transform: uppercase;
  letter-spacing: .1em;
  margin-top: .3rem;
}}
.ch {{ color:#10b981; font-size:1rem; font-weight:700; margin-top:.38rem; }}
.cm {{ color:#f59e0b; font-size:1rem; font-weight:700; margin-top:.38rem; }}
.cl {{ color:#ef4444; font-size:1rem; font-weight:700; margin-top:.38rem; }}

/* ── PREDICTION CARDS GRID ── */
.pgrid {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: .75rem;
  margin-top: .75rem;
}}
@media (max-width: 560px) {{
  .pgrid {{ grid-template-columns: 1fr; }}
}}
.pcard {{
  border-radius: .8rem;
  padding: .95rem;
  border: 1.5px solid {border};
  background: {surface2};
}}
.pcard.win {{
  border-color: {accent};
  background: {"#13103a" if dark else "#f0eeff"};
}}
.rbadge {{
  display: inline-block;
  padding: .1rem .48rem;
  border-radius: 9999px;
  font-size: .67rem;
  font-weight: 700;
  margin-bottom: .6rem;
}}
.r1 {{ background:{accent}; color:white; }}
.rn {{ background:{btrk}; color:{muted}; }}

/* ── SIGNAL BOXES ── */
.sigbox {{
  border-radius: .8rem;
  padding: .82rem 1rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: .62rem;
  gap: .5rem;
}}
.snoisy {{ background:{snbg}; border:1px solid {snbd}; }}
.sclean {{ background:{scbg}; border:1px solid {scbd}; }}

/* ── INFO TILES ── */
.igrid {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: .75rem;
  margin-bottom: 1.1rem;
}}
@media (max-width: 460px) {{
  .igrid {{ grid-template-columns: 1fr; }}
}}
.itile {{
  border-radius: .8rem;
  padding: .85rem;
  text-align: center;
  border: 1px solid {border};
}}
.t-ico {{ font-size:1.18rem; margin-bottom:.18rem; }}
.t-lbl {{ font-size:.64rem; text-transform:uppercase; letter-spacing:.07em; color:{muted}; font-weight:600; }}
.t-val {{ font-size:.95rem; font-weight:800; color:{text}; margin-top:.08rem; }}
.tb {{ background:{tblue}; }}
.tp {{ background:{tpurp}; }}
.ti {{ background:{tindi}; }}

/* ── VISUALIZATIONS ── */
.viz-img {{
  width: 100%;
  border-radius: .8rem;
  border: 1px solid {border};
  display: block;
}}

/* ── FOOTER ── */
.ftr {{
  background: {fbg};
  color: {"#3a3a5a" if dark else "#6b7280"};
  text-align: center;
  padding: 1.2rem;
  font-size: .75rem;
  margin-top: 0;
  border-top: 1px solid {border};
}}

/* ════════════════════════════════════
   STREAMLIT WIDGET OVERRIDES
   ════════════════════════════════════ */

/* Text inputs */
.stTextInput > div > div > input {{
  background: {ibg} !important;
  border: 1.5px solid {ibd} !important;
  border-radius: .75rem !important;
  color: {text} !important;
  font-size: .9rem !important;
  padding: .68rem .95rem !important;
  transition: border-color .2s, box-shadow .2s !important;
}}
.stTextInput > div > div > input:focus {{
  border-color: {accent} !important;
  box-shadow: 0 0 0 3px {ifsh} !important;
  outline: none !important;
}}
.stTextInput label {{
  color: {text} !important;
  font-weight: 600 !important;
  font-size: .8rem !important;
}}

/* Checkbox */
.stCheckbox label {{ color:{text} !important; font-size:.83rem !important; font-weight:500 !important; }}
.stCheckbox > div > label > span:first-child {{
  border-color: {ibd} !important;
  background: {ibg} !important;
  border-radius: .38rem !important;
}}

/* File uploader */
[data-testid="stFileUploadDropzone"] {{
  border: 2px dashed {"#35355a" if dark else "#c8c8dd"} !important;
  border-radius: .9rem !important;
  background: {surface2} !important;
  transition: all .3s !important;
}}
[data-testid="stFileUploadDropzone"]:hover {{
  border-color: {accent} !important;
  background: {"#171730" if dark else "#eeeeff"} !important;
}}
[data-testid="stFileUploadDropzone"] label,
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] span {{
  color: {muted} !important;
  font-weight: 500 !important;
}}

/* Primary buttons */
.stButton > button {{
  background: linear-gradient(135deg, {accent}, {accent2}) !important;
  color: white !important;
  font-weight: 700 !important;
  font-size: .9rem !important;
  border-radius: 9999px !important;
  border: none !important;
  padding: .75rem 2.4rem !important;
  transition: transform .15s, box-shadow .15s !important;
  font-family: 'Syne', sans-serif !important;
  letter-spacing: .02em !important;
  white-space: nowrap !important;
}}
.stButton > button:hover {{
  transform: scale(1.04) !important;
  box-shadow: 0 8px 28px {accent}55 !important;
}}
.stButton > button:disabled {{
  background: {"#252535" if dark else "#e2e2ee"} !important;
  color: {"#45455a" if dark else "#a0a0b8"} !important;
  transform: none !important;
  box-shadow: none !important;
}}

/* Download buttons */
.stDownloadButton > button {{
  border-radius: .8rem !important;
  font-weight: 700 !important;
  padding: .62rem 1.2rem !important;
  border: 1.5px solid {border} !important;
  background: {surface2} !important;
  color: {text} !important;
  transition: all .15s !important;
  font-size: .84rem !important;
  width: 100% !important;
}}
.stDownloadButton > button:hover {{
  transform: scale(1.02) !important;
  border-color: {accent} !important;
  color: {accent} !important;
  box-shadow: 0 4px 16px {accent}33 !important;
}}

/* Spinner */
.stSpinner > div {{ border-top-color: {accent} !important; }}

/* Alerts */
.stAlert {{ border-radius: .8rem !important; }}

/* Audio player */
audio {{ border-radius: .7rem; width: 100%; margin-top: .45rem; }}

/* Remove extra padding Streamlit adds around columns */
[data-testid="column"] > div:first-child {{ padding: 0 !important; }}

/* Fix gaps on column layouts in auth card area */
.auth-cols [data-testid="column"] {{
  padding: 0 .2rem !important;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    for k, v in {
        'auth_page':     'login',
        'dark_mode':     True,
        'result':        None,
        'auth_status':   None,
        'auth_name':     '',
        'auth_username': '',
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# AUTH HELPERS
# ─────────────────────────────────────────────────────────────────────────────
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
# AUTH PAGE — clean, fixed layout
# ─────────────────────────────────────────────────────────────────────────────
def render_auth(dark: bool):
    accent = '#7c6af7' if dark else '#5b50e8'
    text   = '#e4e4f0' if dark else '#111118'
    muted  = '#64648a' if dark else '#9090aa'

    # Full-screen backdrop + orbs (fixed, won't cause scroll)
    st.markdown("""
<style>
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: transparent !important;
}
</style>
<div class="auth-backdrop">
  <div class="orb orb1"></div>
  <div class="orb orb2"></div>
  <div class="orb orb3"></div>
</div>
""", unsafe_allow_html=True)

    # Center column layout
    _, mid, _ = st.columns([1, 1.6, 1])
    with mid:
        # Logo
        st.markdown(f"""
<div class="auth-logo" style="margin-top:2.5rem">
  <div class="auth-logo-icon">🎵</div>
  <h1 class="auth-brand">InstruNet <span>AI</span></h1>
  <p class="auth-tag">AI Music Instrument Recognition</p>
</div>
""", unsafe_allow_html=True)

        # Card start
        st.markdown(f'<div class="auth-card">', unsafe_allow_html=True)

        # Tab switcher using Streamlit buttons
        active_login = st.session_state.auth_page == 'login'
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "🔑  Sign In",
                key="tab_login",
                use_container_width=True,
                type="primary" if active_login else "secondary"
            ):
                st.session_state.auth_page = 'login'
                st.rerun()
        with col2:
            if st.button(
                "✨  Register",
                key="tab_reg",
                use_container_width=True,
                type="primary" if not active_login else "secondary"
            ):
                st.session_state.auth_page = 'register'
                st.rerun()

        st.markdown(
            f"<hr style='border:none;border-top:1px solid {'#25253a' if dark else '#e2e2ee'};margin:.8rem 0 1rem'>",
            unsafe_allow_html=True
        )

        # ── LOGIN ──
        if st.session_state.auth_page == 'login':
            st.markdown(f"""
<p class="form-title">Welcome back 👋</p>
<p class="form-sub">Sign in to your InstruNet AI account</p>
""", unsafe_allow_html=True)

            auth, config = get_authenticator()
            login_result = auth.login(
                fields={
                    'Form name': 'Sign In',
                    'Username':  'Username',
                    'Password':  'Password',
                    'Login':     'Sign In'
                },
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
                st.error("⚠️ Incorrect username or password.")

            st.markdown(
                f"<p style='text-align:center;font-size:.72rem;color:{muted};margin-top:.7rem'>"
                f"No account? Click <b>Register</b> above.</p>",
                unsafe_allow_html=True
            )

        # ── REGISTER ──
        else:
            st.markdown(f"""
<p class="form-title">Create your account ✨</p>
<p class="form-sub">Join InstruNet AI — it's free</p>
""", unsafe_allow_html=True)

            new_name  = st.text_input("Full Name",        placeholder="e.g. Alex Johnson", key="reg_name")
            new_user  = st.text_input("Username",         placeholder="e.g. alexj",        key="reg_user")
            new_email = st.text_input("Email",            placeholder="you@example.com",   key="reg_email")
            new_pass  = st.text_input("Password",         placeholder="Min. 6 characters", type="password", key="reg_pass")
            new_pass2 = st.text_input("Confirm Password", placeholder="Repeat password",   type="password", key="reg_pass2")

            if st.button("Create Account →", use_container_width=True, key="reg_submit"):
                config = load_config()
                users  = config["credentials"]["usernames"]
                err    = None
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
                    st.error(f"⚠️ {err}")
                else:
                    hashed = hash_password(new_pass)
                    config["credentials"]["usernames"][new_user] = {
                        "name": new_name, "email": new_email,
                        "password": hashed, "failed_login_attempts": 0, "logged_in": False
                    }
                    save_config(config)
                    st.success("✅ Account created! Click Sign In to log in.")
                    st.balloons()

            st.markdown(
                f"<p style='text-align:center;font-size:.72rem;color:{muted};margin-top:.7rem'>"
                f"Have an account? Click <b>Sign In</b> above.</p>",
                unsafe_allow_html=True
            )

        # Card end
        st.markdown("</div>", unsafe_allow_html=True)

        # Theme toggle
        st.markdown("<div style='text-align:center;margin-top:.9rem'>", unsafe_allow_html=True)
        if st.button("☀️ Light mode" if dark else "🌙 Dark mode", key="auth_theme"):
            st.session_state.dark_mode = not dark
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<p class='auth-footer'>🔒 Passwords are bcrypt-hashed &nbsp;·&nbsp; InstruNet AI © 2026</p>",
            unsafe_allow_html=True
        )


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
    mel_db   = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    if mel_norm.shape[1] > 128:
        mel_norm = mel_norm[:, :128]
    elif mel_norm.shape[1] < 128:
        mel_norm = np.pad(mel_norm, ((0,0),(0, 128 - mel_norm.shape[1])), mode='constant')
    return mel_norm


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def fig_to_b64(fig, dark):
    bg_c = '#15151f' if dark else 'white'
    tc   = '#64648a' if dark else '#6b7280'
    lc   = '#a0a0c0' if dark else '#374151'
    ttc  = '#e4e4f0' if dark else '#111118'
    sc   = '#2a2a3d' if dark else '#e2e2ee'
    fig.patch.set_facecolor(bg_c)
    for ax in fig.axes:
        ax.set_facecolor(bg_c)
        ax.tick_params(colors=tc)
        ax.xaxis.label.set_color(lc)
        ax.yaxis.label.set_color(lc)
        ax.title.set_color(ttc)
        for sp in ax.spines.values():
            sp.set_edgecolor(sc)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=bg_c)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"

def gen_waveform(y, sr, dark):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(np.arange(len(y)) / sr, y,
            color='#7c6af7' if dark else '#3b82f6', linewidth=0.5)
    ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    ax.set_title('Audio Waveform', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=.12 if dark else .25)
    ax.set_xlim(0, len(y) / sr)
    plt.tight_layout()
    return fig_to_b64(fig, dark)

def gen_spectrogram(mel, sr, dark):
    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(mel, sr=sr, x_axis='time', y_axis='mel', ax=ax,
                                   cmap='magma' if dark else 'viridis')
    ax.set_title('Mel-Spectrogram', fontsize=13, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency (Hz)', fontsize=11, fontweight='bold')
    cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.set_label('Intensity (dB)', fontsize=9)
    tc = '#64648a' if dark else '#6b7280'
    cbar.ax.yaxis.set_tick_params(color=tc)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=tc)
    plt.tight_layout()
    return fig_to_b64(fig, dark)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_audio(audio_bytes, model, label_encoder, dark):
    y, sr = process_audio_file(audio_bytes)
    mel   = extract_mel_spectrogram(y, sr)
    preds = model.predict(mel.reshape(1, 128, 128, 1), verbose=0)

    inst_p = preds['instrument'][0]
    top3   = [
        {'name': label_encoder.classes_[i], 'confidence': float(inst_p[i] * 100)}
        for i in np.argsort(inst_p)[::-1][:3]
    ]

    qi      = np.argmax(preds['quality'][0])
    quality = {
        'label':      QUALITY_LABELS[qi],
        'confidence': float(preds['quality'][0][qi] * 100),
        'all_scores': {QUALITY_LABELS[i]: float(preds['quality'][0][i] * 100) for i in range(4)}
    }
    ci        = np.argmax(preds['condition'][0])
    condition = {
        'label':      CONDITION_LABELS[ci],
        'confidence': float(preds['condition'][0][ci] * 100),
        'all_scores': {CONDITION_LABELS[i]: float(preds['condition'][0][i] * 100) for i in range(4)}
    }
    return {
        'instrument':     top3[0],
        'top_instruments': top3,
        'quality':        quality,
        'condition':      condition,
        'visualizations': {
            'waveform':    gen_waveform(y, sr, dark),
            'spectrogram': gen_spectrogram(mel, sr, dark)
        },
        'audio_info': {
            'duration':    float(len(y) / sr),
            'sample_rate': int(sr),
            'samples':     len(y)
        },
        'timestamp': datetime.now().isoformat()
    }


# ─────────────────────────────────────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf_bytes(result):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.units import inch

    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=letter)
    story, styles = [], getSampleStyleSheet()
    ts   = ParagraphStyle('T', parent=styles['Heading1'], fontSize=22,
                          textColor=colors.HexColor('#1e40af'), spaceAfter=26, alignment=1)
    story.append(Paragraph("Audio Analysis Report — InstruNet AI", ts))
    story.append(Spacer(1, .28 * inch))
    story.append(Paragraph(
        f"<b>Date:</b> {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}",
        styles['Normal']
    ))
    story.append(Spacer(1, .2 * inch))

    def mkt(data, cw, hc):
        t = Table(data, colWidths=cw)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor(hc)),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,0), 11),
            ('BOTTOMPADDING', (0,0), (-1,0), 10),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID',       (0,0), (-1,-1), 1, colors.black),
        ]))
        return t

    ai = result['audio_info']
    for heading, data, cw, hc in [
        ("INSTRUMENT PREDICTION",
         [['Instrument', 'Confidence'],
          [result['instrument']['name'].upper(), f"{result['instrument']['confidence']:.1f}%"]],
         [3*inch, 2*inch], '#3b82f6'),
        ("AUDIO QUALITY",
         [['Quality', 'Confidence'],
          [result['quality']['label'].upper(), f"{result['quality']['confidence']:.1f}%"]],
         [3*inch, 2*inch], '#10b981'),
        ("AUDIO CONDITION",
         [['Condition', 'Confidence'],
          [result['condition']['label'].upper(), f"{result['condition']['confidence']:.1f}%"]],
         [3*inch, 2*inch], '#f59e0b'),
        ("AUDIO INFORMATION",
         [['Property', 'Value'],
          ['Duration', f"{ai['duration']:.2f}s"],
          ['Sample Rate', f"{ai['sample_rate']} Hz"],
          ['Samples', f"{ai['samples']:,}"]],
         [3*inch, 2*inch], '#6b7280'),
    ]:
        story.append(Paragraph(f"<b>{heading}</b>", styles['Heading2']))
        story.append(mkt(data, cw, hc))
        story.append(Spacer(1, .25 * inch))

    story.append(Paragraph("<b>Top 3 Predictions</b>", styles['Heading3']))
    t3 = [['Rank', 'Instrument', 'Confidence']]
    for i, x in enumerate(result['top_instruments'], 1):
        t3.append([str(i), x['name'].capitalize(), f"{x['confidence']:.1f}%"])
    story.append(mkt(t3, [1*inch, 2*inch, 2*inch], '#6366f1'))
    doc.build(story)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def gauge_svg(val, size=200, stroke=15, label='', dark=True):
    import math
    r   = (size - stroke) / 2
    cx  = cy = size / 2
    c   = 2 * math.pi * r
    arc = c * .75
    off = arc - (val / 100) * arc
    col = '#10b981' if val >= 80 else '#f59e0b' if val >= 50 else '#ef4444'
    trk = '#252535' if dark else '#e2e2ee'
    rot = f"rotate(-225 {cx} {cy})"
    lb  = (
        f'<text x="{cx}" y="{size - 5}" text-anchor="middle" font-size="10" font-weight="700" '
        f'fill="{trk}" font-family="DM Sans,sans-serif">{label.upper()}</text>'
    ) if label else ''
    return (
        f'<svg width="{size}" height="{size}" style="overflow:visible">'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{trk}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-dasharray="{arc:.2f} {c:.2f}" transform="{rot}"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{col}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-dasharray="{arc:.2f} {c:.2f}" stroke-dashoffset="{off:.2f}" '
        f'transform="{rot}" style="filter:drop-shadow(0 0 6px {col}99)"/>'
        f'<text x="{cx}" y="{cy - 4}" text-anchor="middle" font-size="21" font-weight="800" '
        f'fill="{col}" font-family="Syne,sans-serif">{val:.1f}%</text>'
        f'<text x="{cx}" y="{cy + 14}" text-anchor="middle" font-size="10" font-weight="500" '
        f'fill="{trk}" font-family="DM Sans,sans-serif">CONFIDENCE</text>{lb}</svg>'
    )

def abar(v, col, h=8):
    return (
        f'<div class="bar-track" style="height:{h}px">'
        f'<div class="bar-fill" style="height:{h}px;width:{v:.1f}%;'
        f'background:{col};box-shadow:0 0 7px {col}55"></div></div>'
    )

def render_hero(r, dark):
    inst = r['instrument']
    icon = get_icon(inst['name'])
    conf = inst['confidence']
    ccls = 'ch' if conf >= 80 else 'cm' if conf >= 50 else 'cl'
    clbl = '✅ High Confidence' if conf >= 80 else '⚠️ Moderate Confidence' if conf >= 50 else '❌ Low Confidence'
    gsvg = gauge_svg(conf, 200, 16, dark=dark)
    acc  = '#7c6af7' if dark else '#5b50e8'
    tc   = '#e4e4f0' if dark else '#111118'
    mu   = '#64648a' if dark else '#9090aa'

    cards = ''
    for i, item in enumerate(r['top_instruments']):
        wc = 'win' if i == 0 else ''
        rc = 'r1'  if i == 0 else 'rn'
        bc = acc   if i == 0 else ('#35355a' if dark else '#c8c8dd')
        cards += (
            f'<div class="pcard {wc}">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.6rem">'
            f'<span class="rbadge {rc}">#{i+1}</span>'
            f'<span style="font-size:1.3rem">{get_icon(item["name"])}</span></div>'
            f'<p style="font-family:Syne,sans-serif;font-weight:700;color:{tc};font-size:.94rem;'
            f'text-transform:capitalize;margin:.15rem 0">{item["name"]}</p>'
            f'<div style="display:flex;justify-content:space-between;font-size:.75rem;margin-bottom:.32rem">'
            f'<span style="color:{mu}">Confidence</span>'
            f'<span style="font-weight:800;color:{tc}">{item["confidence"]:.1f}%</span></div>'
            f'{abar(item["confidence"], bc, 8)}</div>'
        )

    st.markdown(
        f'<div class="card">'
        f'<h3 style="font-size:1.1rem;border-bottom:1px solid var(--border);'
        f'padding-bottom:.75rem;margin-bottom:1.1rem">📊 Analysis Results</h3>'
        f'<div class="hero-wrap">'
        f'<div>{gsvg}</div>'
        f'<div><div class="hero-icon">{icon}</div>'
        f'<div class="hero-name">{inst["name"].upper()}</div>'
        f'<div class="hero-sub">Top Prediction</div>'
        f'<div class="{ccls}">{clbl}</div></div></div>'
        f'<p style="font-size:.72rem;font-weight:700;color:{mu};text-transform:uppercase;'
        f'letter-spacing:.09em;margin-bottom:.65rem">🔢 All Predictions</p>'
        f'<div class="pgrid">{cards}</div></div>',
        unsafe_allow_html=True
    )

def render_quality(q, dark):
    lbl  = q['label']
    conf = q['confidence']
    gsvg = gauge_svg(conf, 170, 14, lbl, dark)
    mu   = '#64648a' if dark else '#9090aa'
    bars = ''
    for l, s in q['all_scores'].items():
        w   = l == lbl
        col = '#10b981' if w else ('#35355a' if dark else '#d1d5db')
        fw  = 'font-weight:700;color:#10b981' if w else f'color:{mu}'
        bars += (
            f'<div class="bar-row">'
            f'<div class="bar-label">'
            f'<span style="{fw};text-transform:capitalize;font-size:.77rem">{"▶ " if w else ""}{l}</span>'
            f'<span style="{fw};font-size:.77rem">{s:.1f}%</span></div>'
            f'{abar(s, col, 8)}</div>'
        )
    st.markdown(
        f'<div class="card" style="height:100%">'
        f'<h3>🎚️ Audio Quality</h3>'
        f'<div style="display:flex;flex-direction:column;align-items:center;margin-bottom:.82rem">'
        f'{gsvg}'
        f'<span class="qbadge q-{lbl.lower()}" style="margin-top:.62rem">{lbl.upper()}</span>'
        f'</div>{bars}</div>',
        unsafe_allow_html=True
    )

def render_condition(cond, dark):
    cs  = cond.get('all_scores', {})
    ms  = cs.get('modern', 0) + cs.get('clean', 0) + cs.get('noisy', 0)
    vs  = cs.get('vintage', 0)
    et  = ms + vs or 1
    mp  = (ms / et) * 100
    vp  = (vs / et) * 100
    im  = ms >= vs
    nr  = cs.get('noisy', 0)
    cr  = cs.get('clean', 0)
    isn = nr > cr
    mu  = '#64648a' if dark else '#9090aa'
    sc  = '#ef4444' if isn else '#10b981'
    mc  = '#818cf8' if dark else '#3b82f6'

    meb = (
        f'<span style="margin-left:.26rem;padding:.07rem .44rem;'
        f'background:{"#1e1b4b" if dark else "#dbeafe"};'
        f'color:{"#818cf8" if dark else "#2563eb"};'
        f'border-radius:9999px;font-size:.64rem;font-weight:700">ERA</span>'
    ) if im else ''
    veb = (
        f'<span style="margin-left:.26rem;padding:.07rem .44rem;'
        f'background:{"#431407" if dark else "#fef3c7"};'
        f'color:#fb923c;border-radius:9999px;font-size:.64rem;font-weight:700">ERA</span>'
    ) if not im else ''

    sub = ''
    for l, s, c, tc2 in [
        ('Clean signal', cr, '#2dd4bf', '#0d9488'),
        ('Noisy signal', nr, '#f87171', '#dc2626'),
    ]:
        sub += (
            f'<div style="display:flex;align-items:center;gap:.44rem;margin-bottom:.28rem">'
            f'<span style="font-size:.69rem;color:{mu};width:82px;flex-shrink:0">{l}</span>'
            f'<div style="flex:1">{abar(s, c, 6)}</div>'
            f'<span style="font-size:.69rem;font-weight:700;color:{tc2};'
            f'width:32px;text-align:right">{s:.1f}%</span></div>'
        )

    st.markdown(
        f'<div class="card" style="height:100%">'
        f'<h3>⏳ Audio Condition</h3>'
        f'<p style="font-size:.64rem;font-weight:700;color:{mu};text-transform:uppercase;'
        f'letter-spacing:.08em;margin-bottom:.62rem">🕰️ Recording Era</p>'
        f'<div class="bar-row">'
        f'<div class="bar-label">'
        f'<span style="font-weight:700;color:{"#818cf8" if im and dark else "#2563eb" if im else mu};'
        f'font-size:.77rem">💿 Modern {meb}</span>'
        f'<span style="font-weight:800;color:{"#818cf8" if im and dark else "#2563eb" if im else mu};'
        f'font-size:.77rem">{mp:.1f}%</span></div>{abar(mp, mc, 14)}</div>'
        f'<div class="bar-row" style="margin-top:.62rem">'
        f'<div class="bar-label">'
        f'<span style="font-weight:700;color:{"#fb923c" if not im else mu};font-size:.77rem">'
        f'🕰️ Vintage {veb}</span>'
        f'<span style="font-weight:800;color:{"#fb923c" if not im else mu};font-size:.77rem">'
        f'{vp:.1f}%</span></div>{abar(vp, "#fb923c", 14)}</div>'
        f'<p style="font-size:.64rem;font-weight:700;color:{mu};text-transform:uppercase;'
        f'letter-spacing:.08em;margin:1rem 0 .5rem">🔊 Signal Quality</p>'
        f'<div class="sigbox {"snoisy" if isn else "sclean"}">'
        f'<div style="display:flex;align-items:center;gap:.62rem">'
        f'<span style="font-size:1.25rem">{"📻" if isn else "✨"}</span>'
        f'<div>'
        f'<p style="font-size:.86rem;font-weight:800;color:{sc};margin:0">'
        f'{"Noisy" if isn else "Clean"}</p>'
        f'<p style="font-size:.68rem;color:{mu};margin:.08rem 0 0">'
        f'{"Background noise or artifacts detected" if isn else "Clear signal with minimal noise"}'
        f'</p></div></div>'
        f'<span style="font-size:1.25rem">{"⚠️" if isn else "✅"}</span></div>'
        f'<div style="padding:0 .1rem">{sub}</div></div>',
        unsafe_allow_html=True
    )

def render_info(result):
    ai = result['audio_info']
    st.markdown(
        f'<div class="card"><h3>ℹ️ Audio Information</h3><div class="igrid">'
        f'<div class="itile tb"><div class="t-ico">⏱️</div>'
        f'<div class="t-lbl">Duration</div>'
        f'<div class="t-val">{ai["duration"]:.2f}s</div></div>'
        f'<div class="itile tp"><div class="t-ico">〰️</div>'
        f'<div class="t-lbl">Sample Rate</div>'
        f'<div class="t-val">{ai["sample_rate"]} Hz</div></div>'
        f'<div class="itile ti"><div class="t-ico">💾</div>'
        f'<div class="t-lbl">Samples</div>'
        f'<div class="t-val">{ai["samples"]:,}</div></div>'
        f'</div></div>',
        unsafe_allow_html=True
    )
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️  Download JSON",
            data=json.dumps({k: v for k, v in result.items() if k != 'visualizations'}, indent=2),
            file_name=f"instrunet_{ts}.json",
            mime="application/json",
            use_container_width=True
        )
    with c2:
        st.download_button(
            "📄  Download PDF",
            data=generate_pdf_bytes(result),
            file_name=f"instrunet_{ts}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

def render_viz(result):
    st.markdown(
        f'<div class="card"><h3>〰️ Audio Visualizations</h3>'
        f'<p style="font-size:.64rem;font-weight:700;color:var(--muted);'
        f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem">Waveform</p>'
        f'<img src="{result["visualizations"]["waveform"]}" class="viz-img" '
        f'style="margin-bottom:1.2rem"/>'
        f'<p style="font-size:.64rem;font-weight:700;color:var(--muted);'
        f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem">Mel-Spectrogram</p>'
        f'<img src="{result["visualizations"]["spectrogram"]}" class="viz-img"/></div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def render_app(dark: bool):
    text  = '#e4e4f0' if dark else '#111118'
    muted = '#64648a' if dark else '#9090aa'

    load_model_and_encoder()  # pre-load (cached)

    # ── Header ──
    st.markdown(
        '<div class="hdr">'
        '<h1>🎵 InstruNet <span>AI</span></h1>'
        '<p>AI-Powered Music Instrument Recognition &amp; Analysis</p>'
        '<div class="tag-row">'
        '<span class="tag-pill">Instrument</span>'
        '<span class="tag-pill">Quality</span>'
        '<span class="tag-pill">Condition</span>'
        '<span class="tag-pill">Multi-Task CNN</span>'
        '</div></div>',
        unsafe_allow_html=True
    )

    # ── User bar: name | theme toggle | sign out ──
    cu, ct, clo = st.columns([5, 1.5, 1.2])
    with cu:
        name = st.session_state.get('auth_name') or st.session_state.get('name', 'User')
        st.markdown(
            f'<div class="ubar">'
            f'<span class="ubar-left">👤 Signed in as <strong>{name}</strong></span>'
            f'</div>',
            unsafe_allow_html=True
        )
    with ct:
        if st.button("☀️ Light" if dark else "🌙 Dark", key="app_theme"):
            st.session_state.dark_mode = not dark
            # NOTE: don't wipe result on theme toggle
            st.rerun()
    with clo:
        if st.button("Sign Out", key="signout_btn"):
            st.session_state.auth_status   = None
            st.session_state.auth_name     = ''
            st.session_state.auth_username = ''
            st.session_state.result        = None
            for k in ["authentication_status", "name", "username", "logout"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    st.markdown('<div class="wrap">', unsafe_allow_html=True)

    # ── Upload card ──
    st.markdown('<div class="card"><h3>☁️ Upload Audio File</h3>', unsafe_allow_html=True)
    af = st.file_uploader(
        "Drop audio file here or click to browse — WAV · MP3 · OGG · First 3 seconds analyzed",
        type=['wav', 'mp3', 'ogg', 'flac', 'm4a'],
        label_visibility='visible'
    )
    if af:
        st.audio(af, format=af.type)
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:.6rem;padding:.58rem .9rem;'
            f'background:{"#0c1a10" if dark else "#f0fff4"};border-radius:.8rem;'
            f'border:1px solid {"#166534" if dark else "#86efac"};margin-top:.62rem">'
            f'<span style="font-size:1rem">✅</span>'
            f'<div>'
            f'<p style="font-weight:600;color:{text};font-size:.82rem;margin:0">{af.name}</p>'
            f'<p style="font-size:.68rem;color:{muted};margin:0">{af.size / 1024:.2f} KB</p>'
            f'</div></div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Analyze button ──
    _, cc, _ = st.columns([2, 2, 2])
    with cc:
        go = st.button("🧠  Analyze Audio", disabled=af is None, use_container_width=True)

    if go and af:
        with st.spinner("🔬 Analyzing audio…"):
            model, le = load_model_and_encoder()
            result = predict_audio(af.read(), model, le, dark)
        st.session_state.result = result

    # ── Results ──
    if st.session_state.result:
        res = st.session_state.result
        render_hero(res, dark)
        c1, c2 = st.columns(2)
        with c1:
            render_quality(res['quality'], dark)
        with c2:
            render_condition(res['condition'], dark)
        render_info(res)
        render_viz(res)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ftr">© 2026 InstruNet AI &nbsp;·&nbsp; '
        'Multi-Task CNN · Instrument · Quality · Condition</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    init_state()
    dark = st.session_state.dark_mode
    inject_css(dark)

    is_auth = (
        st.session_state.get("authentication_status") is True
        or st.session_state.auth_status is True
    )

    if is_auth:
        render_app(dark)
    else:
        render_auth(dark)


if __name__ == '__main__':
    main()
