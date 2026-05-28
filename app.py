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

def load_config():
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
            "name": "instrunet_auth",
            "key": st.secrets.get("COOKIE_KEY", "instrunet_super_secret_key_2026"),
            "expiry_days": 30
        },
        "pre-authorized": {"emails": []}
    }
    for uname, hpw in seed_users.items():
        config["credentials"]["usernames"][uname] = {
            "name": uname.capitalize(), "email": f"{uname}@instrunet.ai",
            "password": hpw, "failed_login_attempts": 0, "logged_in": False
        }
    save_config(config)
    return config

def save_config(config):
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

def init_state():
    defaults = {
        'auth_page': 'login', 'dark_mode': True, 'result': None,
        'auth_status': None, 'auth_name': '', 'auth_username': '',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def get_authenticator():
    import streamlit_authenticator as stauth
    config = load_config()
    auth = stauth.Authenticate(
        config['credentials'], config['cookie']['name'],
        config['cookie']['key'], config['cookie']['expiry_days'],
    )
    return auth, config

def hash_password(plain):
    import bcrypt
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

# ─── THEME TOKENS ────────────────────────────────────────────────────────────
def get_theme(dark):
    if dark:
        return {
            'bg':       '#0d0d14',
            'surface':  '#16161f',
            'surface2': '#1d1d2c',
            'border':   '#2c2c42',
            'text':     '#eaeaf5',
            'muted':    '#6b6b95',
            'accent':   '#7c6af7',
            'accent2':  '#a78bfa',
            'green':    '#10b981',
            'amber':    '#f59e0b',
            'red':      '#ef4444',
            'blue':     '#3b82f6',
            'input_bg': '#1a1a2c',
            'input_bd': '#3a3a58',
            'track':    '#2a2a40',
            'card_bg':  '#16161f',
            'hdr1':     '#b91c1c',
            'hdr2':     '#ea580c',
            'auth_bg1': '#0d0d14',
            'auth_bg2': '#1a0d2e',
            'auth_card':'#141420',
            'auth_bd':  '#28283d',
            'success_bg':'#0c1a10',
            'success_bd':'#166534',
        }
    else:
        return {
            'bg':       '#f2f2fa',
            'surface':  '#ffffff',
            'surface2': '#f8f8fd',
            'border':   '#e4e4f0',
            'text':     '#0f0f1a',
            'muted':    '#8888aa',
            'accent':   '#5b50e8',
            'accent2':  '#764ba2',
            'green':    '#10b981',
            'amber':    '#f59e0b',
            'red':      '#ef4444',
            'blue':     '#3b82f6',
            'input_bg': '#ffffff',
            'input_bd': '#cccce0',
            'track':    '#e0e0f0',
            'card_bg':  '#ffffff',
            'hdr1':     '#b91c1c',
            'hdr2':     '#ea580c',
            'auth_bg1': '#4f46e5',
            'auth_bg2': '#7c3aed',
            'auth_card':'#ffffff',
            'auth_bd':  '#e4e4f0',
            'success_bg':'#ecfdf5',
            'success_bd':'#6ee7b7',
        }

# ─── MASTER CSS ──────────────────────────────────────────────────────────────
def inject_css(t, is_auth=False):
    auth_page_bg = f"linear-gradient(160deg, {t['auth_bg1']} 0%, #0a0a1a 50%, {t['auth_bg2']} 100%)" if t['bg'] == '#0d0d14' else f"linear-gradient(160deg, #4338ca 0%, #5b50e8 40%, #7c3aed 100%)"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* GLOBAL RESET */
html, body, * {{ box-sizing: border-box; }}
* {{ font-family: 'Plus Jakarta Sans', sans-serif !important; }}
#MainMenu, footer, header {{ visibility: hidden !important; display: none !important; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; background: transparent !important; }}
section[data-testid="stSidebar"] {{ display: none !important; }}

/* APP BACKGROUND */
[data-testid="stAppViewContainer"] {{
    background: {auth_page_bg if is_auth else t['bg']} !important;
    min-height: 100vh;
}}
[data-testid="stMain"] {{ background: transparent !important; }}
[data-testid="stHeader"] {{ display: none !important; }}
[data-testid="stDecoration"] {{ display: none !important; }}

/* REMOVE DEFAULT STREAMLIT TOP PADDING */
.main .block-container {{ padding-top: 0 !important; }}
div[data-testid="stVerticalBlock"] > div:first-child {{ margin-top: 0 !important; }}

/* ── INPUT FIELDS ── */
div[data-testid="stTextInput"] > label {{
    color: {t['text']} !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    margin-bottom: 0.3rem !important;
    display: block !important;
}}
div[data-testid="stTextInput"] > div > div > input {{
    background: {t['input_bg']} !important;
    border: 1.5px solid {t['input_bd']} !important;
    border-radius: 10px !important;
    color: {t['text']} !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 0.9rem !important;
    width: 100% !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    box-shadow: none !important;
}}
div[data-testid="stTextInput"] > div > div > input:focus {{
    border-color: {t['accent']} !important;
    box-shadow: 0 0 0 3px {t['accent']}30 !important;
    outline: none !important;
}}
div[data-testid="stTextInput"] > div > div > input::placeholder {{
    color: {t['muted']} !important;
    opacity: 0.7 !important;
}}

/* ── BUTTONS ── */
div[data-testid="stButton"] > button {{
    background: linear-gradient(135deg, {t['accent']} 0%, {t['accent2']} 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    border-radius: 50px !important;
    border: none !important;
    padding: 0.65rem 1.8rem !important;
    letter-spacing: 0.01em !important;
    transition: opacity 0.15s, transform 0.15s !important;
    box-shadow: 0 4px 15px {t['accent']}40 !important;
    cursor: pointer !important;
    width: 100% !important;
}}
div[data-testid="stButton"] > button:hover {{
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px {t['accent']}55 !important;
}}
div[data-testid="stButton"] > button:disabled,
div[data-testid="stButton"] > button[disabled] {{
    background: {t['track']} !important;
    color: {t['muted']} !important;
    box-shadow: none !important;
    transform: none !important;
    opacity: 1 !important;
    cursor: not-allowed !important;
}}
/* secondary buttons: theme toggle + sign out */
div[data-testid="stButton"] > button[kind="secondary"] {{
    background: {t['surface2']} !important;
    color: {t['text']} !important;
    border: 1.5px solid {t['border']} !important;
    box-shadow: none !important;
    font-weight: 600 !important;
}}
div[data-testid="stButton"] > button[kind="secondary"]:hover {{
    border-color: {t['accent']} !important;
    color: {t['accent']} !important;
    background: {t['surface2']} !important;
}}

/* ── DOWNLOAD BUTTONS ── */
div[data-testid="stDownloadButton"] > button {{
    background: {t['surface2']} !important;
    color: {t['text']} !important;
    border: 1.5px solid {t['border']} !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    padding: 0.6rem 1.2rem !important;
    width: 100% !important;
    transition: all 0.15s !important;
    box-shadow: none !important;
}}
div[data-testid="stDownloadButton"] > button:hover {{
    border-color: {t['accent']} !important;
    color: {t['accent']} !important;
    transform: translateY(-1px) !important;
}}

/* ── FILE UPLOADER ── */
section[data-testid="stFileUploadDropzone"] {{
    background: {t['surface2']} !important;
    border: 2px dashed {t['border']} !important;
    border-radius: 14px !important;
    padding: 1.2rem !important;
    transition: all 0.2s !important;
}}
section[data-testid="stFileUploadDropzone"]:hover {{
    border-color: {t['accent']} !important;
    background: {t['accent']}08 !important;
}}
div[data-testid="stFileUploaderDropzoneInstructions"] span,
div[data-testid="stFileUploaderDropzoneInstructions"] p,
div[data-testid="stFileUploaderDropzoneInstructions"] small {{
    color: {t['muted']} !important;
}}
/* Hide the duplicate "Browse files" button text overlap */
div[data-testid="stFileUploadDropzone"] label {{
    color: {t['muted']} !important;
}}

/* ── ALERTS ── */
div[data-testid="stAlert"] {{
    border-radius: 12px !important;
    font-size: 0.86rem !important;
}}

/* ── SPINNER ── */
div[data-testid="stSpinner"] > div {{
    border-top-color: {t['accent']} !important;
}}

/* ── AUDIO PLAYER ── */
audio {{
    width: 100% !important;
    border-radius: 10px !important;
    margin-top: 0.5rem !important;
}}

/* ── COLUMN GAPS ── */
div[data-testid="stHorizontalBlock"] {{
    gap: 0.75rem !important;
    align-items: stretch !important;
}}

/* ── METRIC ELEMENTS ── */
div[data-testid="stMetric"] {{
    background: {t['surface2']} !important;
    border: 1px solid {t['border']} !important;
    border-radius: 12px !important;
    padding: 0.9rem 1rem !important;
}}
div[data-testid="stMetricLabel"] p {{
    color: {t['muted']} !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}}
div[data-testid="stMetricValue"] {{
    color: {t['text']} !important;
    font-size: 1.35rem !important;
    font-weight: 800 !important;
}}

/* REMOVE STRAY IFRAME/TOOLBAR */
[data-testid="stToolbar"] {{ display: none !important; }}
iframe[title="streamlit_authenticator.authenticate"] {{ border: none !important; }}
</style>
""", unsafe_allow_html=True)

# ─── AUTH PAGE ────────────────────────────────────────────────────────────────
def render_auth(t):
    dark = t['bg'] == '#0d0d14'

    # Centered layout
    left, center, right = st.columns([1, 1.4, 1])
    with center:
        st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)

        # ── LOGO ──
        st.markdown(f"""
<div style="text-align:center; margin-bottom:1.5rem;">
  <div style="
    display:inline-flex; align-items:center; justify-content:center;
    width:62px; height:62px; border-radius:18px; font-size:1.75rem;
    background:linear-gradient(135deg,{t['accent']},{t['accent2']});
    box-shadow:0 8px 24px {t['accent']}50;
    margin-bottom:0.9rem;
  ">🎵</div>
  <h1 style="
    font-family:'Space Grotesk',sans-serif !important;
    font-size:1.9rem; font-weight:700; color:#ffffff;
    margin:0; letter-spacing:-0.03em;
    text-shadow:0 2px 16px rgba(0,0,0,0.4);
  ">InstruNet <span style="color:{t['accent2']}">AI</span></h1>
  <p style="
    font-size:0.68rem; color:rgba(255,255,255,0.45);
    margin:0.3rem 0 0; font-weight:600;
    text-transform:uppercase; letter-spacing:0.1em;
  ">AI Music Instrument Recognition</p>
</div>
""", unsafe_allow_html=True)

        # ── CARD ──
        st.markdown(f"""
<div style="
  background:{t['auth_card']};
  border:1px solid {t['auth_bd']};
  border-radius:20px;
  padding:1.8rem 1.7rem 1.5rem;
  box-shadow:0 24px 80px rgba(0,0,0,0.45);
">
""", unsafe_allow_html=True)

        # Tab row
        active = st.session_state.auth_page == 'login'
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔑  Sign In", key="tab_login", use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.auth_page = 'login'; st.rerun()
        with c2:
            if st.button("✨  Register", key="tab_reg", use_container_width=True,
                         type="primary" if not active else "secondary"):
                st.session_state.auth_page = 'register'; st.rerun()

        st.markdown(f"""<hr style="
          border:none; border-top:1px solid {t['border']};
          margin:1rem 0 1.2rem;
        ">""", unsafe_allow_html=True)

        # ── LOGIN FORM ──
        if active:
            st.markdown(f"""
<p style="font-family:'Space Grotesk',sans-serif !important;
   font-size:1.05rem; font-weight:700; color:{t['text']}; margin:0 0 0.1rem">
   Welcome back 👋</p>
<p style="font-size:0.78rem; color:{t['muted']}; margin:0 0 1rem">
   Sign in to your InstruNet AI account</p>
""", unsafe_allow_html=True)
            auth, config = get_authenticator()
            login_result = auth.login(
                fields={'Form name': 'Sign In', 'Username': 'Username',
                        'Password': 'Password', 'Login': 'Sign In'},
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
                save_config(config); st.rerun()
            elif auth_status is False:
                st.error("⚠️ Incorrect username or password.")

            st.markdown(f"""
<p style="text-align:center; font-size:0.73rem; color:{t['muted']}; margin-top:0.8rem">
  No account? Click <b style="color:{t['accent']}">Register</b> above.</p>
""", unsafe_allow_html=True)

        # ── REGISTER FORM ──
        else:
            st.markdown(f"""
<p style="font-family:'Space Grotesk',sans-serif !important;
   font-size:1.05rem; font-weight:700; color:{t['text']}; margin:0 0 0.1rem">
   Create your account ✨</p>
<p style="font-size:0.78rem; color:{t['muted']}; margin:0 0 1rem">
   Join InstruNet AI — free forever</p>
""", unsafe_allow_html=True)
            new_name  = st.text_input("Full Name",        placeholder="Alex Johnson",    key="reg_name")
            new_user  = st.text_input("Username",         placeholder="alexj",           key="reg_user")
            new_email = st.text_input("Email",            placeholder="you@example.com", key="reg_email")
            new_pass  = st.text_input("Password",         placeholder="Min 6 chars",     type="password", key="reg_pass")
            new_pass2 = st.text_input("Confirm Password", placeholder="Repeat password", type="password", key="reg_pass2")

            if st.button("Create Account →", use_container_width=True, key="reg_submit"):
                config = load_config()
                users  = config["credentials"]["usernames"]
                err    = None
                if not all([new_name, new_user, new_email, new_pass, new_pass2]):
                    err = "Please fill in all fields."
                elif new_user in users:
                    err = f"Username '{new_user}' is already taken."
                elif any(u.get('email') == new_email for u in users.values()):
                    err = "That email is already registered."
                elif len(new_pass) < 6:
                    err = "Password must be at least 6 characters."
                elif new_pass != new_pass2:
                    err = "Passwords do not match."
                if err:
                    st.error(f"⚠️ {err}")
                else:
                    config["credentials"]["usernames"][new_user] = {
                        "name": new_name, "email": new_email,
                        "password": hash_password(new_pass),
                        "failed_login_attempts": 0, "logged_in": False
                    }
                    save_config(config)
                    st.success("✅ Account created! Click Sign In to log in.")
                    st.balloons()

            st.markdown(f"""
<p style="text-align:center; font-size:0.73rem; color:{t['muted']}; margin-top:0.8rem">
  Have an account? Click <b style="color:{t['accent']}">Sign In</b> above.</p>
""", unsafe_allow_html=True)

        # Close card div
        st.markdown("</div>", unsafe_allow_html=True)

        # ── THEME TOGGLE ──
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        if st.button("☀️  Light mode" if dark else "🌙  Dark mode",
                     key="auth_theme", use_container_width=True, type="secondary"):
            st.session_state.dark_mode = not dark; st.rerun()

        st.markdown(f"""
<p style="text-align:center; font-size:0.67rem; color:rgba(255,255,255,0.22);
   margin-top:0.7rem; padding-bottom:2rem">
  🔒 Bcrypt-secured &nbsp;·&nbsp; InstruNet AI © 2026</p>
""", unsafe_allow_html=True)


# ─── AUDIO PROCESSING ─────────────────────────────────────────────────────────
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

def fig_to_b64(fig, dark):
    bg = '#16161f' if dark else '#ffffff'
    tc = '#6b6b95' if dark else '#6b7280'
    lc = '#a0a0c0' if dark else '#374151'
    fig.patch.set_facecolor(bg)
    for ax in fig.axes:
        ax.set_facecolor(bg)
        ax.tick_params(colors=tc)
        ax.xaxis.label.set_color(lc); ax.yaxis.label.set_color(lc)
        ax.title.set_color('#eaeaf5' if dark else '#0f0f1a')
        for sp in ax.spines.values(): sp.set_edgecolor('#2c2c42' if dark else '#e4e4f0')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=bg)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"

def gen_waveform(y, sr, dark):
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.plot(np.arange(len(y))/sr, y, color='#7c6af7' if dark else '#5b50e8', linewidth=0.6)
    ax.set_xlabel('Time (s)', fontsize=10); ax.set_ylabel('Amplitude', fontsize=10)
    ax.set_title('Waveform', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.1 if dark else 0.2); ax.set_xlim(0, len(y)/sr)
    plt.tight_layout(); return fig_to_b64(fig, dark)

def gen_spectrogram(mel, sr, dark):
    fig, ax = plt.subplots(figsize=(12, 3.8))
    img = librosa.display.specshow(mel, sr=sr, x_axis='time', y_axis='mel', ax=ax,
                                   cmap='magma' if dark else 'viridis')
    ax.set_title('Mel-Spectrogram', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=10); ax.set_ylabel('Frequency', fontsize=10)
    cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
    tc = '#6b6b95' if dark else '#6b7280'
    cbar.ax.yaxis.set_tick_params(color=tc)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=tc)
    plt.tight_layout(); return fig_to_b64(fig, dark)

def predict_audio(audio_bytes, model, label_encoder, dark):
    y, sr = process_audio_file(audio_bytes)
    mel   = extract_mel_spectrogram(y, sr)
    preds = model.predict(mel.reshape(1,128,128,1), verbose=0)
    inst_p = preds['instrument'][0]
    top3   = [{'name': label_encoder.classes_[i], 'confidence': float(inst_p[i]*100)}
               for i in np.argsort(inst_p)[::-1][:3]]
    qi = np.argmax(preds['quality'][0])
    quality = {
        'label': QUALITY_LABELS[qi], 'confidence': float(preds['quality'][0][qi]*100),
        'all_scores': {QUALITY_LABELS[i]: float(preds['quality'][0][i]*100) for i in range(4)}
    }
    ci = np.argmax(preds['condition'][0])
    condition = {
        'label': CONDITION_LABELS[ci], 'confidence': float(preds['condition'][0][ci]*100),
        'all_scores': {CONDITION_LABELS[i]: float(preds['condition'][0][i]*100) for i in range(4)}
    }
    return {
        'instrument': top3[0], 'top_instruments': top3,
        'quality': quality, 'condition': condition,
        'visualizations': {'waveform': gen_waveform(y, sr, dark), 'spectrogram': gen_spectrogram(mel, sr, dark)},
        'audio_info': {'duration': float(len(y)/sr), 'sample_rate': int(sr), 'samples': len(y)},
        'timestamp': datetime.now().isoformat()
    }

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
                        textColor=colors.HexColor('#1e40af'), spaceAfter=26, alignment=1)
    story.append(Paragraph("Audio Analysis Report — InstruNet AI", ts))
    story.append(Spacer(1, .25*inch))
    story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, .18*inch))
    def mkt(data, cw, hc):
        t = Table(data, colWidths=cw)
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor(hc)),('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,0),11),('BOTTOMPADDING',(0,0),(-1,0),10),
            ('BACKGROUND',(0,1),(-1,-1),colors.beige),('GRID',(0,0),(-1,-1),1,colors.black),
        ])); return t
    ai = result['audio_info']
    for heading, data, cw, hc in [
        ("INSTRUMENT PREDICTION",[['Instrument','Confidence'],[result['instrument']['name'].upper(),f"{result['instrument']['confidence']:.1f}%"]],[3*inch,2*inch],'#3b82f6'),
        ("AUDIO QUALITY",[['Quality','Confidence'],[result['quality']['label'].upper(),f"{result['quality']['confidence']:.1f}%"]],[3*inch,2*inch],'#10b981'),
        ("AUDIO CONDITION",[['Condition','Confidence'],[result['condition']['label'].upper(),f"{result['condition']['confidence']:.1f}%"]],[3*inch,2*inch],'#f59e0b'),
        ("AUDIO INFO",[['Property','Value'],['Duration',f"{ai['duration']:.2f}s"],['Sample Rate',f"{ai['sample_rate']} Hz"],['Samples',f"{ai['samples']:,}"]],[3*inch,2*inch],'#6b7280'),
    ]:
        story.append(Paragraph(f"<b>{heading}</b>",styles['Heading2']))
        story.append(mkt(data,cw,hc)); story.append(Spacer(1,.2*inch))
    story.append(Paragraph("<b>Top 3 Predictions</b>",styles['Heading3']))
    t3=[['Rank','Instrument','Confidence']]
    for i,x in enumerate(result['top_instruments'],1): t3.append([str(i),x['name'].capitalize(),f"{x['confidence']:.1f}%"])
    story.append(mkt(t3,[1*inch,2*inch,2*inch],'#6366f1'))
    doc.build(story); buf.seek(0); return buf.read()

# ─── RENDER HELPERS ───────────────────────────────────────────────────────────
def gauge_svg(val, dark, size=180, stroke=14, label=''):
    import math
    r=( size-stroke)/2; cx=cy=size/2; c=2*math.pi*r; arc=c*.75
    off=arc-(val/100)*arc
    col='#10b981' if val>=80 else '#f59e0b' if val>=50 else '#ef4444'
    trk='#2a2a40' if dark else '#e0e0f0'
    rot=f"rotate(-225 {cx} {cy})"
    lb=(f'<text x="{cx}" y="{size-4}" text-anchor="middle" font-size="9" font-weight="700" '
        f'fill="{col if label else trk}" font-family="Plus Jakarta Sans,sans-serif">{label.upper()}</text>') if label else ''
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{trk}" stroke-width="{stroke}" '
            f'stroke-linecap="round" stroke-dasharray="{arc:.2f} {c:.2f}" transform="{rot}"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{col}" stroke-width="{stroke}" '
            f'stroke-linecap="round" stroke-dasharray="{arc:.2f} {c:.2f}" stroke-dashoffset="{off:.2f}" '
            f'transform="{rot}" style="filter:drop-shadow(0 0 5px {col}88)"/>'
            f'<text x="{cx}" y="{cy-2}" text-anchor="middle" font-size="20" font-weight="800" '
            f'fill="{col}" font-family="Space Grotesk,sans-serif">{val:.1f}%</text>'
            f'<text x="{cx}" y="{cy+13}" text-anchor="middle" font-size="9" font-weight="500" '
            f'fill="{trk}" font-family="Plus Jakarta Sans,sans-serif">CONFIDENCE</text>{lb}</svg>')

def prog_bar(val, col, h=7, dark=True):
    trk = '#2a2a40' if dark else '#e0e0f0'
    return (f'<div style="background:{trk};border-radius:999px;overflow:hidden;height:{h}px">'
            f'<div style="height:{h}px;width:{val:.1f}%;background:{col};'
            f'border-radius:999px;box-shadow:0 0 6px {col}66"></div></div>')

def section_label(txt, t):
    return (f'<p style="font-size:0.63rem;font-weight:700;color:{t["muted"]};'
            f'text-transform:uppercase;letter-spacing:0.09em;margin:0 0 0.55rem">{txt}</p>')

def card_wrap(content, t, extra_style=""):
    return (f'<div style="background:{t["card_bg"]};border:1px solid {t["border"]};'
            f'border-radius:16px;padding:1.3rem;margin-bottom:1.2rem;{extra_style}">'
            f'{content}</div>')

def card_title(txt, t):
    return (f'<p style="font-family:Space Grotesk,sans-serif;font-size:0.92rem;font-weight:700;'
            f'color:{t["text"]};margin:0 0 1rem;padding-bottom:0.7rem;'
            f'border-bottom:1px solid {t["border"]}">{txt}</p>')

# ─── MAIN APP ─────────────────────────────────────────────────────────────────
def render_app(t):
    dark = t['bg'] == '#0d0d14'
    load_model_and_encoder()

    # ── HEADER ──
    st.markdown(f"""
<div style="
  background:linear-gradient(135deg,{t['hdr1']} 0%,{t['hdr2']} 100%);
  padding:1.4rem 2.2rem 1.5rem;
">
  <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.25rem">
    <span style="font-size:1.4rem">🎵</span>
    <h1 style="font-family:'Space Grotesk',sans-serif !important;font-size:1.85rem;
               font-weight:700;color:white;margin:0;letter-spacing:-0.03em">
      InstruNet <span style="opacity:0.5;font-weight:400">AI</span></h1>
  </div>
  <p style="color:rgba(255,255,255,0.55);margin:0;font-size:0.84rem">
    AI-Powered Music Instrument Recognition &amp; Analysis</p>
  <div style="display:flex;gap:0.45rem;margin-top:0.65rem;flex-wrap:wrap">
    {"".join(f'<span style="background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.2);border-radius:999px;padding:0.15rem 0.7rem;font-size:0.68rem;font-weight:600;color:white;letter-spacing:0.04em">{x}</span>' for x in ['Instrument','Quality','Condition','Multi-Task CNN'])}
  </div>
</div>
""", unsafe_allow_html=True)

    # ── USER BAR ──
    name = st.session_state.get('auth_name') or st.session_state.get('name', 'User')
    ub_left, ub_mid, ub_right = st.columns([5, 1.4, 1.2])
    with ub_left:
        st.markdown(f"""
<div style="background:{t['surface2']};border-bottom:1px solid {t['border']};
            padding:0.6rem 2.2rem;font-size:0.8rem;color:{t['muted']}">
  👤 Signed in as <strong style="color:{t['text']}">{name}</strong>
</div>
""", unsafe_allow_html=True)
    with ub_mid:
        st.markdown(f"<div style='background:{t['surface2']};border-bottom:1px solid {t['border']};padding:0.35rem 0.5rem'>", unsafe_allow_html=True)
        if st.button("☀️ Light" if dark else "🌙 Dark", key="app_theme", use_container_width=True, type="secondary"):
            st.session_state.dark_mode = not dark; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with ub_right:
        st.markdown(f"<div style='background:{t['surface2']};border-bottom:1px solid {t['border']};padding:0.35rem 0.5rem'>", unsafe_allow_html=True)
        if st.button("Sign Out", key="signout_btn", use_container_width=True, type="secondary"):
            for k in ['auth_status','auth_name','auth_username','authentication_status','name','username','logout','result']:
                if k in st.session_state: del st.session_state[k]
            st.session_state.auth_status = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # ── MAIN CONTENT ──
    st.markdown(f"<div style='padding:1.4rem 2.2rem 2rem;background:{t['bg']}'>", unsafe_allow_html=True)

    # Upload section
    st.markdown(card_wrap(
        card_title("☁️ Upload Audio File", t) +
        "<p style='font-size:0.78rem;color:" + t['muted'] + ";margin:0 0 0.7rem'>Supports WAV · MP3 · OGG · FLAC · M4A — first 3 seconds analyzed</p>",
        t
    ), unsafe_allow_html=True)

    # File uploader OUTSIDE the HTML card (Streamlit must render it natively)
    with st.container():
        af = st.file_uploader(
            "Drop audio file or click to browse",
            type=['wav','mp3','ogg','flac','m4a'],
            label_visibility='collapsed'
        )

    if af:
        st.audio(af, format=af.type)
        st.markdown(f"""
<div style="
  display:flex;align-items:center;gap:0.65rem;
  background:{t['success_bg']};border:1px solid {t['success_bd']};
  border-radius:12px;padding:0.65rem 1rem;margin-top:0.6rem
">
  <span style="font-size:1rem">✅</span>
  <div>
    <p style="font-weight:700;color:{t['text']};font-size:0.84rem;margin:0">{af.name}</p>
    <p style="font-size:0.7rem;color:{t['muted']};margin:0">{af.size/1024:.1f} KB</p>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)

    # Analyze button
    _, bc, _ = st.columns([1.5, 2, 1.5])
    with bc:
        go = st.button("🧠  Analyze Audio", disabled=af is None, use_container_width=True)

    if go and af:
        with st.spinner("🔬 Analyzing your audio…"):
            model, le = load_model_and_encoder()
            st.session_state.result = predict_audio(af.read(), model, le, dark)

    # ── RESULTS ──
    if st.session_state.result:
        res = st.session_state.result
        _render_results(res, t, dark)

    st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown(f"""
<div style="
  background:{'#08080e' if dark else '#1a1a2e'};
  color:{'#3a3a58' if dark else '#6b7280'};
  text-align:center;padding:1.1rem;font-size:0.74rem;
  border-top:1px solid {t['border']}
">© 2026 InstruNet AI · Multi-Task CNN · Instrument · Quality · Condition</div>
""", unsafe_allow_html=True)


def _render_results(res, t, dark):
    # ── HERO: instrument + gauge ──
    inst = res['instrument']
    icon = get_icon(inst['name'])
    conf = inst['confidence']
    conf_color = t['green'] if conf>=80 else t['amber'] if conf>=50 else t['red']
    conf_label = ('✅ High Confidence' if conf>=80 else '⚠️ Moderate Confidence' if conf>=50 else '❌ Low Confidence')

    # Top 3 cards
    pred_cards_html = ""
    for i, item in enumerate(res['top_instruments']):
        win = i == 0
        bg = (f"linear-gradient(135deg,{t['accent']}22,{t['accent2']}11)" if win
              else f"{t['surface2']}")
        bd = t['accent'] if win else t['border']
        bc_fill = t['accent'] if win else t['muted']
        pred_cards_html += f"""
<div style="background:{bg};border:1.5px solid {bd};border-radius:14px;padding:1rem;flex:1;min-width:0">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.55rem">
    <span style="background:{t['accent'] if win else t['track']};color:{'white' if win else t['muted']};
                 font-size:0.65rem;font-weight:700;padding:0.1rem 0.5rem;border-radius:999px">
      #{i+1}</span>
    <span style="font-size:1.25rem">{get_icon(item['name'])}</span>
  </div>
  <p style="font-family:'Space Grotesk',sans-serif;font-weight:700;color:{t['text']};
             font-size:0.9rem;text-transform:capitalize;margin:0 0 0.5rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{item['name']}</p>
  <div style="display:flex;justify-content:space-between;font-size:0.73rem;margin-bottom:0.3rem">
    <span style="color:{t['muted']}">Confidence</span>
    <span style="font-weight:800;color:{t['text']}">{item['confidence']:.1f}%</span>
  </div>
  {prog_bar(item['confidence'], t['accent'] if win else t['muted'], 6, dark)}
</div>"""

    st.markdown(card_wrap(
        card_title("📊 Analysis Results", t) +
        f"""<div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:center;
                        gap:2rem;background:{t['surface2']};border-radius:14px;
                        padding:1.6rem;margin-bottom:1.1rem;border:1px solid {t['border']}">
          <div style="text-align:center">{gauge_svg(conf, dark, 185, 15)}</div>
          <div style="text-align:center">
            <div style="font-size:3.8rem;line-height:1;margin-bottom:0.3rem">{icon}</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:clamp(1.6rem,4vw,2.4rem);
                        font-weight:800;color:{t['text']};letter-spacing:-0.03em;text-transform:uppercase">{inst['name']}</div>
            <div style="font-size:0.65rem;font-weight:600;color:{t['muted']};
                        text-transform:uppercase;letter-spacing:0.1em;margin-top:0.28rem">Top Prediction</div>
            <div style="font-size:0.9rem;font-weight:700;color:{conf_color};margin-top:0.35rem">{conf_label}</div>
          </div>
        </div>
        {section_label("🔢 Top 3 Predictions", t)}
        <div style="display:flex;gap:0.75rem;flex-wrap:wrap">{pred_cards_html}</div>""",
        t
    ), unsafe_allow_html=True)

    # ── QUALITY + CONDITION ──
    qcol, ccol = st.columns(2)
    with qcol:
        q    = res['quality']
        bars = ""
        for l, s in q['all_scores'].items():
            w = l == q['label']
            col = t['green'] if w else t['muted']
            fw  = f"color:{t['green']};font-weight:700" if w else f"color:{t['muted']}"
            bars += (f'<div style="margin-bottom:0.55rem">'
                     f'<div style="display:flex;justify-content:space-between;font-size:0.75rem;margin-bottom:0.18rem">'
                     f'<span style="{fw};text-transform:capitalize">{"▶ " if w else ""}{l}</span>'
                     f'<span style="{fw}">{s:.1f}%</span></div>'
                     f'{prog_bar(s, col, 7, dark)}</div>')
        q_badge_colors = {'excellent':'#10b981','good':'#3b82f6','fair':'#f59e0b','poor':'#ef4444'}
        q_bc = q_badge_colors.get(q['label'], t['accent'])
        st.markdown(card_wrap(
            card_title("🎚️ Audio Quality", t) +
            f'<div style="display:flex;flex-direction:column;align-items:center;margin-bottom:1rem">'
            f'{gauge_svg(q["confidence"], dark, 165, 13, q["label"])}'
            f'<span style="margin-top:0.55rem;padding:0.25rem 0.85rem;border-radius:999px;'
            f'background:{q_bc};color:white;font-size:0.7rem;font-weight:700;'
            f'letter-spacing:0.05em">{q["label"].upper()}</span></div>' + bars,
            t, "height:100%"
        ), unsafe_allow_html=True)

    with ccol:
        cond = res['condition']
        cs   = cond.get('all_scores', {})
        ms   = cs.get('modern',0)+cs.get('clean',0)+cs.get('noisy',0)
        vs   = cs.get('vintage',0)
        et   = ms+vs or 1
        mp   = (ms/et)*100; vp = (vs/et)*100
        im   = ms >= vs
        nr   = cs.get('noisy',0); cr = cs.get('clean',0); isn = nr > cr
        sc   = t['red'] if isn else t['green']
        mc   = '#818cf8' if dark else t['blue']

        sub = ""
        for l, s, col in [('Clean signal', cr, '#2dd4bf'), ('Noisy signal', nr, '#f87171')]:
            sub += (f'<div style="display:flex;align-items:center;gap:0.45rem;margin-bottom:0.28rem">'
                    f'<span style="font-size:0.69rem;color:{t["muted"]};width:80px;flex-shrink:0">{l}</span>'
                    f'<div style="flex:1">{prog_bar(s, col, 6, dark)}</div>'
                    f'<span style="font-size:0.69rem;font-weight:700;color:{col};width:32px;text-align:right">{s:.1f}%</span></div>')

        st.markdown(card_wrap(
            card_title("⏳ Audio Condition", t) +
            section_label("🕰️ Recording Era", t) +
            f'<div style="margin-bottom:0.55rem">'
            f'<div style="display:flex;justify-content:space-between;font-size:0.76rem;margin-bottom:0.18rem">'
            f'<span style="font-weight:700;color:{mc if im else t["muted"]}">💿 Modern {"✓" if im else ""}</span>'
            f'<span style="font-weight:800;color:{mc if im else t["muted"]}">{mp:.1f}%</span></div>'
            f'{prog_bar(mp, mc, 13, dark)}</div>'
            f'<div style="margin-bottom:1rem">'
            f'<div style="display:flex;justify-content:space-between;font-size:0.76rem;margin-bottom:0.18rem">'
            f'<span style="font-weight:700;color:{"#fb923c" if not im else t["muted"]}">🕰️ Vintage {"✓" if not im else ""}</span>'
            f'<span style="font-weight:800;color:{"#fb923c" if not im else t["muted"]}">{vp:.1f}%</span></div>'
            f'{prog_bar(vp, "#fb923c", 13, dark)}</div>'
            + section_label("🔊 Signal Quality", t) +
            f'<div style="background:{"#1a0c0c" if (isn and dark) else "#0c1a15" if dark else ("#fff1f2" if isn else "#f0fff9")};'
            f'border:1px solid {"#7f1d1d" if isn else "#064e3b"};border-radius:12px;'
            f'padding:0.75rem 0.9rem;margin-bottom:0.7rem;display:flex;align-items:center;justify-content:space-between">'
            f'<div style="display:flex;align-items:center;gap:0.6rem">'
            f'<span style="font-size:1.15rem">{"📻" if isn else "✨"}</span>'
            f'<div><p style="font-weight:800;color:{sc};font-size:0.85rem;margin:0">{"Noisy" if isn else "Clean"}</p>'
            f'<p style="font-size:0.67rem;color:{t["muted"]};margin:0.06rem 0 0">'
            f'{"Noise/artifacts detected" if isn else "Clear minimal-noise signal"}</p></div></div>'
            f'<span>{"⚠️" if isn else "✅"}</span></div>' + sub,
            t, "height:100%"
        ), unsafe_allow_html=True)

    # ── AUDIO INFO + EXPORTS ──
    ai = res['audio_info']
    ic1, ic2, ic3 = st.columns(3)
    tile_style = f"background:{t['surface2']};border:1px solid {t['border']};border-radius:14px;padding:1rem;text-align:center"
    for col, emoji, label, val in [
        (ic1, "⏱️", "Duration", f"{ai['duration']:.2f}s"),
        (ic2, "〰️", "Sample Rate", f"{ai['sample_rate']} Hz"),
        (ic3, "💾", "Samples", f"{ai['samples']:,}"),
    ]:
        with col:
            st.markdown(f"""
<div style="{tile_style}">
  <div style="font-size:1.1rem;margin-bottom:0.18rem">{emoji}</div>
  <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.07em;color:{t['muted']};font-weight:600">{label}</div>
  <div style="font-size:0.94rem;font-weight:800;color:{t['text']};margin-top:0.1rem">{val}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button("⬇️  Download JSON",
            data=json.dumps({k:v for k,v in res.items() if k!='visualizations'}, indent=2),
            file_name=f"instrunet_{ts}.json", mime="application/json", use_container_width=True)
    with dl2:
        st.download_button("📄  Download PDF", data=generate_pdf_bytes(res),
            file_name=f"instrunet_{ts}.pdf", mime="application/pdf", use_container_width=True)

    # ── VISUALIZATIONS ──
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    st.markdown(card_wrap(
        card_title("〰️ Audio Visualizations", t) +
        f'{section_label("Waveform", t)}'
        f'<img src="{res["visualizations"]["waveform"]}" style="width:100%;border-radius:12px;border:1px solid {t["border"]};display:block;margin-bottom:1.1rem"/>'
        f'{section_label("Mel-Spectrogram", t)}'
        f'<img src="{res["visualizations"]["spectrogram"]}" style="width:100%;border-radius:12px;border:1px solid {t["border"]};display:block"/>',
        t
    ), unsafe_allow_html=True)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
def main():
    init_state()
    dark = st.session_state.dark_mode
    t    = get_theme(dark)
    is_auth = (st.session_state.get("authentication_status") is True
               or st.session_state.get("auth_status") is True)
    inject_css(t, is_auth=not is_auth)
    if is_auth:
        render_app(t)
    else:
        render_auth(t)

if __name__ == '__main__':
    main()
