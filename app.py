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
CONFIG_FILE      = "auth_config.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG / AUTH HELPERS
# ─────────────────────────────────────────────────────────────────────────────
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
            "expiry_days": 30,
        },
        "pre-authorized": {"emails": []},
    }
    for uname, hpw in seed_users.items():
        config["credentials"]["usernames"][uname] = {
            "name": uname.capitalize(), "email": f"{uname}@instrunet.ai",
            "password": hpw, "failed_login_attempts": 0, "logged_in": False,
        }
    save_config(config)
    return config

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def get_authenticator():
    import streamlit_authenticator as stauth
    config = load_config()
    auth = stauth.Authenticate(
        config["credentials"], config["cookie"]["name"],
        config["cookie"]["key"], config["cookie"]["expiry_days"],
    )
    return auth, config

def hash_password(plain):
    import bcrypt
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_encoder():
    from huggingface_hub import hf_hub_download
    from tensorflow import keras
    token    = st.secrets["HF_TOKEN"]
    repo_id  = st.secrets["HF_REPO_ID"]
    model_path   = hf_hub_download(repo_id=repo_id, filename="best_multitask_model.keras", token=token)
    encoder_path = hf_hub_download(repo_id=repo_id, filename="label_encoder.pkl", token=token)
    model = keras.models.load_model(model_path)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

INSTRUMENT_ICONS = {
    "brass":"🎺","guitar":"🎸","keyboard":"🎹","mallet":"🪘",
    "organ":"🎹","reed":"🎷","string":"🎻","synth":"🎛️",
    "vocal":"🎤","flute":"🪈","bass":"🎸","drum":"🥁",
}
def get_icon(name):
    n = (name or "").lower()
    for k, v in INSTRUMENT_ICONS.items():
        if k in n: return v
    return "🎵"

# ─────────────────────────────────────────────────────────────────────────────
# THEME  — all colors resolved as plain strings, never as CSS vars
# ─────────────────────────────────────────────────────────────────────────────
def T(dark=True):
    if dark:
        return dict(
            bg="#0d0d14", surface="#15151f", surface2="#1c1c28", border="#2a2a3d",
            text="#e4e4f0", muted="#64648a", accent="#7c6af7", accent2="#a78bfa",
            track="#252535", inp_bg="#1a1a28", inp_bd="#35355a",
            hdr_grad="linear-gradient(135deg,#1e1b4b,#312e81)",
            hero_bg="#13132a", win_bg="#13103a", win_bd="#7c6af7",
            tb="#0d1120", tp="#120d20", ti="#0e1020",
            snbg="#1a0c0c", snbd="#7f1d1d", scbg="#0c1a15", scbd="#064e3b",
            footer_bg="#08080e", footer_txt="#3a3a5a",
            upload_ok_bg="#0c1a10", upload_ok_bd="#166534",
        )
    return dict(
        bg="#f0f0f7", surface="#ffffff", surface2="#f7f7fc", border="#e2e2ee",
        text="#111118", muted="#9090aa", accent="#5b50e8", accent2="#764ba2",
        track="#e2e2ee", inp_bg="#ffffff", inp_bd="#c8c8dd",
        hdr_grad="linear-gradient(135deg,#667eea,#764ba2)",
        hero_bg="#eff0ff", win_bg="#f0eeff", win_bd="#5b50e8",
        tb="#eef3ff", tp="#f3eeff", ti="#eef0ff",
        snbg="#fff1f2", snbd="#fecaca", scbg="#f0fff9", scbd="#99f6e4",
        footer_bg="#111118", footer_txt="#6b7280",
        upload_ok_bg="#f0fff4", upload_ok_bd="#86efac",
    )

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — widget styling only; layout via Streamlit columns/containers
# ─────────────────────────────────────────────────────────────────────────────
def inject_css(dark):
    c = T(dark)
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@400;500;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
* {{ font-family: 'DM Sans', sans-serif !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}
section[data-testid="stSidebar"] {{ display: none !important; }}
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {{ background: {c['bg']} !important; }}

/* ── inputs ── */
.stTextInput > div > div > input {{
    background: {c['inp_bg']} !important; border: 1.5px solid {c['inp_bd']} !important;
    border-radius: .75rem !important; color: {c['text']} !important;
    font-size: .92rem !important; padding: .7rem 1rem !important;
}}
.stTextInput > div > div > input:focus {{
    border-color: {c['accent']} !important;
    box-shadow: 0 0 0 3px {c['accent']}33 !important; outline: none !important;
}}
.stTextInput label {{ color: {c['text']} !important; font-weight: 600 !important; font-size: .82rem !important; }}

/* ── file uploader ── */
[data-testid="stFileUploadDropzone"] {{
    border: 2px dashed {c['inp_bd']} !important; border-radius: 1rem !important;
    background: {c['surface2']} !important;
}}
[data-testid="stFileUploadDropzone"]:hover {{ border-color: {c['accent']} !important; }}
[data-testid="stFileUploadDropzone"] label,
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] span {{ color: {c['muted']} !important; }}

/* ── primary button ── */
.stButton > button {{
    background: linear-gradient(135deg, {c['accent']}, {c['accent2']}) !important;
    color: white !important; font-weight: 700 !important; font-size: .9rem !important;
    border-radius: 9999px !important; border: none !important;
    padding: .72rem 2.2rem !important; transition: transform .15s, box-shadow .15s !important;
    font-family: 'Syne', sans-serif !important;
}}
.stButton > button:hover {{ transform: scale(1.04) !important; box-shadow: 0 8px 28px {c['accent']}55 !important; }}
.stButton > button:disabled {{ opacity: .45 !important; transform: none !important; }}

/* ── download button ── */
.stDownloadButton > button {{
    border-radius: .85rem !important; font-weight: 700 !important;
    padding: .65rem 1.3rem !important; border: 1.5px solid {c['border']} !important;
    background: {c['surface2']} !important; color: {c['text']} !important;
    font-size: .86rem !important; transition: all .15s !important;
}}
.stDownloadButton > button:hover {{
    border-color: {c['accent']} !important; color: {c['accent']} !important;
}}

/* ── divider ── */
hr {{ border-color: {c['border']} !important; }}
audio {{ border-radius: .75rem; width: 100%; margin-top: .5rem; }}

/* ── card helper class (used on st.container via markdown trick) ── */
.instrunet-card {{
    background: {c['surface']}; border: 1px solid {c['border']};
    border-radius: 1.1rem; padding: 1.4rem;
    box-shadow: 0 4px 24px rgba(0,0,0,.09); margin-bottom: 1.2rem;
}}
.instrunet-card-title {{
    font-family: 'Syne', sans-serif !important; font-size: .95rem; font-weight: 700;
    color: {c['text']}; margin: 0 0 1rem; padding-bottom: .75rem;
    border-bottom: 1px solid {c['border']};
}}
/* auth card */
.auth-card {{
    background: {c['surface']}; border: 1px solid {c['border']};
    border-radius: 1.4rem; padding: 1.8rem;
    box-shadow: 0 24px 80px rgba(0,0,0,.28);
}}
/* orbs */
.orb {{ position: fixed; border-radius: 50%; filter: blur(90px); opacity: .22; pointer-events: none;
        animation: drift 14s ease-in-out infinite alternate; }}
.orb1 {{ width:380px;height:380px;background:{c['accent']};top:-110px;left:-80px; }}
.orb2 {{ width:260px;height:260px;background:{c['accent2']};bottom:-70px;right:-50px;animation-delay:-5s; }}
.orb3 {{ width:160px;height:160px;background:#3b82f6;top:45%;right:16%;animation-delay:-9s; }}
@keyframes drift {{ from{{transform:translate(0,0)}} to{{transform:translate(22px,30px)}} }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    for k, v in {"auth_page":"login","dark_mode":True,"result":None,
                 "auth_status":None,"auth_name":"","auth_username":""}.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# HTML HELPERS  — self-contained blocks only, never split across st calls
# ─────────────────────────────────────────────────────────────────────────────
def pbar(val, col, track, h=8):
    return (f'<div style="background:{track};border-radius:9999px;overflow:hidden;height:{h}px">'
            f'<div style="height:{h}px;width:{val:.1f}%;background:{col};'
            f'border-radius:9999px;box-shadow:0 0 6px {col}55"></div></div>')

def gauge_svg(val, size=190, stroke=15, label="", dark=True):
    import math
    r=( size-stroke)/2; cx=cy=size/2; circ=2*math.pi*r; arc=circ*.75
    off=arc-(val/100)*arc
    col="#10b981" if val>=80 else "#f59e0b" if val>=50 else "#ef4444"
    trk="#252535" if dark else "#e2e2ee"
    rot=f"rotate(-225 {cx} {cy})"
    lb=(f'<text x="{cx}" y="{size-5}" text-anchor="middle" font-size="10" font-weight="700" '
        f'fill="{trk}" font-family="DM Sans">{label.upper()}</text>') if label else ""
    return (
        f'<svg width="{size}" height="{size}" style="display:block;margin:0 auto">'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{trk}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-dasharray="{arc:.2f} {circ:.2f}" transform="{rot}"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{col}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-dasharray="{arc:.2f} {circ:.2f}" stroke-dashoffset="{off:.2f}" '
        f'transform="{rot}" style="filter:drop-shadow(0 0 6px {col}99)"/>'
        f'<text x="{cx}" y="{cy-4}" text-anchor="middle" font-size="22" font-weight="800" '
        f'fill="{col}" font-family="Syne">{val:.1f}%</text>'
        f'<text x="{cx}" y="{cy+16}" text-anchor="middle" font-size="10" '
        f'fill="{trk}" font-family="DM Sans">CONFIDENCE</text>{lb}</svg>'
    )

# ─────────────────────────────────────────────────────────────────────────────
# AUTH PAGE  — every st.markdown is self-contained HTML
# ─────────────────────────────────────────────────────────────────────────────
def render_auth(dark):
    c = T(dark)

    # Background gradient
    st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: radial-gradient(ellipse at 20% 30%, {c['accent']}44, {c['bg']} 55%, {c['accent2']}33) !important;
}}
</style>
<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div class="orb orb3"></div>
""", unsafe_allow_html=True)

    # Logo — self-contained
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown(f"""
<div style="text-align:center;padding:2.5rem 0 1.5rem">
  <div style="display:inline-flex;align-items:center;justify-content:center;
              width:70px;height:70px;border-radius:1.2rem;font-size:2rem;
              background:linear-gradient(135deg,{c['accent']},{c['accent2']});
              box-shadow:0 8px 32px {c['accent']}55;margin-bottom:.9rem">🎵</div>
  <h1 style="font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;
             color:{c['text']};letter-spacing:-.04em;margin:0 0 .3rem">
    InstruNet <span style="color:{c['accent']}">AI</span></h1>
  <p style="font-size:.72rem;color:{c['muted']};font-weight:600;
            text-transform:uppercase;letter-spacing:.1em;margin:0">
    AI Music Instrument Recognition</p>
</div>
""", unsafe_allow_html=True)

    # Card — open
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown(f"""
<div style="background:{c['surface']};border:1px solid {c['border']};border-radius:1.4rem;
            padding:1.8rem 1.8rem .5rem;box-shadow:0 24px 80px rgba(0,0,0,.28)">
""", unsafe_allow_html=True)

        # Tab buttons
        active = st.session_state.auth_page == "login"
        t1, t2 = st.columns(2)
        with t1:
            if st.button("🔑  Sign In", key="tab_login", use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.auth_page = "login"; st.rerun()
        with t2:
            if st.button("✨  Register", key="tab_reg", use_container_width=True,
                         type="primary" if not active else "secondary"):
                st.session_state.auth_page = "register"; st.rerun()

        st.divider()

        if st.session_state.auth_page == "login":
            st.markdown(f"""
<p style="font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;color:{c['text']};margin:0 0 .15rem">
  Welcome back 👋</p>
<p style="font-size:.8rem;color:{c['muted']};margin:0 0 1rem">Sign in to your InstruNet AI account</p>
""", unsafe_allow_html=True)

            auth, config = get_authenticator()
            result = auth.login(
                fields={"Form name":"Sign In","Username":"Username",
                        "Password":"Password","Login":"Sign In"},
                key="login_widget"
            )
            name, auth_status, username = (result if isinstance(result, tuple) and len(result)==3
                                           else (None, None, None))
            if auth_status is True:
                st.session_state.auth_status   = True
                st.session_state.auth_name     = name
                st.session_state.auth_username = username
                save_config(config); st.rerun()
            elif auth_status is False:
                st.error("⚠️ Incorrect username or password.")

            st.markdown(f'<p style="text-align:center;font-size:.73rem;color:{c["muted"]};'
                        f'padding:.5rem 0 1rem">No account? Click <b>Register</b> above.</p>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f"""
<p style="font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;color:{c['text']};margin:0 0 .15rem">
  Create your account ✨</p>
<p style="font-size:.8rem;color:{c['muted']};margin:0 0 1rem">Sign up to start using InstruNet AI</p>
""", unsafe_allow_html=True)

            new_name  = st.text_input("Full Name",        placeholder="e.g. Alex Johnson", key="reg_name")
            new_user  = st.text_input("Username",         placeholder="e.g. alexj",        key="reg_user")
            new_email = st.text_input("Email",            placeholder="you@example.com",   key="reg_email")
            new_pass  = st.text_input("Password",         placeholder="Min. 6 characters", type="password", key="reg_pass")
            new_pass2 = st.text_input("Confirm Password", placeholder="Repeat password",   type="password", key="reg_pass2")

            if st.button("Create Account →", use_container_width=True, key="reg_submit"):
                config = load_config()
                users  = config["credentials"]["usernames"]
                err = None
                if not all([new_name, new_user, new_email, new_pass, new_pass2]):
                    err = "Please fill in all fields."
                elif new_user in users:
                    err = f"Username '{new_user}' is already taken."
                elif any(u.get("email") == new_email for u in users.values()):
                    err = "An account with this email already exists."
                elif len(new_pass) < 6:
                    err = "Password must be at least 6 characters."
                elif new_pass != new_pass2:
                    err = "Passwords do not match."
                if err:
                    st.error(f"⚠️ {err}")
                else:
                    config["credentials"]["usernames"][new_user] = {
                        "name": new_name, "email": new_email, "password": hash_password(new_pass),
                        "failed_login_attempts": 0, "logged_in": False,
                    }
                    save_config(config)
                    st.success("✅ Account created! Click Sign In to log in.")
                    st.balloons()

            st.markdown(f'<p style="text-align:center;font-size:.73rem;color:{c["muted"]};'
                        f'padding:.5rem 0 1rem">Have an account? Click <b>Sign In</b> above.</p>',
                        unsafe_allow_html=True)

        # Card — close
        st.markdown("</div>", unsafe_allow_html=True)

    # Theme toggle
    _, tc, _ = st.columns([4, 1.5, 4])
    with tc:
        if st.button("☀️ Light" if dark else "🌙 Dark", key="auth_theme"):
            st.session_state.dark_mode = not dark; st.rerun()

    st.markdown(f'<p style="text-align:center;font-size:.72rem;margin-top:.5rem;color:{c["muted"]}88">'
                f'🔒 Passwords are bcrypt-hashed · InstruNet AI © 2026</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def process_audio_file(audio_bytes, sr=22050, duration=3.0):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr, duration=duration, mono=True)
    target = int(sr * duration)
    y = np.pad(y, (0, max(0, target - len(y))), mode="constant")[:target]
    return y, sr

def extract_mel_spectrogram(y, sr, n_mels=128, n_fft=2048, hop_length=512):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    if mel_norm.shape[1] > 128:   mel_norm = mel_norm[:, :128]
    elif mel_norm.shape[1] < 128: mel_norm = np.pad(mel_norm, ((0,0),(0,128-mel_norm.shape[1])))
    return mel_norm

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
def fig_to_b64(fig, dark):
    bg = "#15151f" if dark else "white"
    tc = "#64648a" if dark else "#6b7280"
    lc = "#a0a0c0" if dark else "#374151"
    sc = "#2a2a3d" if dark else "#e2e2ee"
    fig.patch.set_facecolor(bg)
    for ax in fig.axes:
        ax.set_facecolor(bg); ax.tick_params(colors=tc)
        ax.xaxis.label.set_color(lc); ax.yaxis.label.set_color(lc)
        ax.title.set_color("#e4e4f0" if dark else "#111118")
        for sp in ax.spines.values(): sp.set_edgecolor(sc)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=bg)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"

def gen_waveform(y, sr, dark):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(np.arange(len(y))/sr, y, color="#7c6af7" if dark else "#3b82f6", linewidth=0.5)
    ax.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Amplitude", fontsize=11, fontweight="bold")
    ax.set_title("Audio Waveform", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.12 if dark else 0.25); ax.set_xlim(0, len(y)/sr)
    plt.tight_layout(); return fig_to_b64(fig, dark)

def gen_spectrogram(mel, sr, dark):
    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(mel, sr=sr, x_axis="time", y_axis="mel", ax=ax,
                                   cmap="magma" if dark else "viridis")
    ax.set_title("Mel-Spectrogram", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency (Hz)", fontsize=11, fontweight="bold")
    cbar = plt.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.set_label("Intensity (dB)", fontsize=9)
    tc = "#64648a" if dark else "#6b7280"
    cbar.ax.yaxis.set_tick_params(color=tc)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=tc)
    plt.tight_layout(); return fig_to_b64(fig, dark)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_audio(audio_bytes, model, label_encoder, dark):
    y, sr = process_audio_file(audio_bytes)
    mel   = extract_mel_spectrogram(y, sr)
    preds = model.predict(mel.reshape(1,128,128,1), verbose=0)
    inst_p = preds["instrument"][0]
    top3 = [{"name": label_encoder.classes_[i], "confidence": float(inst_p[i]*100)}
             for i in np.argsort(inst_p)[::-1][:3]]
    qi = np.argmax(preds["quality"][0])
    quality = {"label": QUALITY_LABELS[qi], "confidence": float(preds["quality"][0][qi]*100),
               "all_scores": {QUALITY_LABELS[i]: float(preds["quality"][0][i]*100) for i in range(4)}}
    ci = np.argmax(preds["condition"][0])
    condition = {"label": CONDITION_LABELS[ci], "confidence": float(preds["condition"][0][ci]*100),
                 "all_scores": {CONDITION_LABELS[i]: float(preds["condition"][0][i]*100) for i in range(4)}}
    return {
        "instrument": top3[0], "top_instruments": top3,
        "quality": quality, "condition": condition,
        "visualizations": {"waveform": gen_waveform(y, sr, dark), "spectrogram": gen_spectrogram(mel, sr, dark)},
        "audio_info": {"duration": float(len(y)/sr), "sample_rate": int(sr), "samples": len(y)},
        "timestamp": datetime.now().isoformat(),
    }

# ─────────────────────────────────────────────────────────────────────────────
# PDF
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
    ts = ParagraphStyle("T", parent=styles["Heading1"], fontSize=22,
                        textColor=colors.HexColor("#1e40af"), spaceAfter=26, alignment=1)
    story.append(Paragraph("Audio Analysis Report — InstruNet AI", ts))
    story.append(Spacer(1, .28*inch))
    story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, .2*inch))
    def mkt(data, cw, hc):
        t = Table(data, colWidths=cw)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor(hc)),
            ("TEXTCOLOR",(0,0),(-1,0),colors.whitesmoke),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,0),11),
            ("BOTTOMPADDING",(0,0),(-1,0),10),
            ("BACKGROUND",(0,1),(-1,-1),colors.beige),
            ("GRID",(0,0),(-1,-1),1,colors.black),
        ]))
        return t
    ai = result["audio_info"]
    for heading, data, cw, hc in [
        ("INSTRUMENT PREDICTION",
         [["Instrument","Confidence"],[result["instrument"]["name"].upper(),f"{result['instrument']['confidence']:.1f}%"]],
         [3*inch,2*inch],"#3b82f6"),
        ("AUDIO QUALITY",
         [["Quality","Confidence"],[result["quality"]["label"].upper(),f"{result['quality']['confidence']:.1f}%"]],
         [3*inch,2*inch],"#10b981"),
        ("AUDIO CONDITION",
         [["Condition","Confidence"],[result["condition"]["label"].upper(),f"{result['condition']['confidence']:.1f}%"]],
         [3*inch,2*inch],"#f59e0b"),
        ("AUDIO INFORMATION",
         [["Property","Value"],["Duration",f"{ai['duration']:.2f}s"],
          ["Sample Rate",f"{ai['sample_rate']} Hz"],["Samples",f"{ai['samples']:,}"]],
         [3*inch,2*inch],"#6b7280"),
    ]:
        story.append(Paragraph(f"<b>{heading}</b>", styles["Heading2"]))
        story.append(mkt(data, cw, hc)); story.append(Spacer(1,.25*inch))
    story.append(Paragraph("<b>Top 3 Predictions</b>", styles["Heading3"]))
    t3 = [["Rank","Instrument","Confidence"]]
    for i, x in enumerate(result["top_instruments"], 1):
        t3.append([str(i), x["name"].capitalize(), f"{x['confidence']:.1f}%"])
    story.append(mkt(t3, [1*inch,2*inch,2*inch], "#6366f1"))
    doc.build(story); buf.seek(0); return buf.read()

# ─────────────────────────────────────────────────────────────────────────────
# RESULT RENDER FUNCTIONS
# Each uses ONLY self-contained st.markdown blocks + native Streamlit widgets
# ─────────────────────────────────────────────────────────────────────────────
def render_hero(r, dark):
    c    = T(dark)
    inst = r["instrument"]
    icon = get_icon(inst["name"])
    conf = inst["confidence"]
    conf_col   = "#10b981" if conf>=80 else "#f59e0b" if conf>=50 else "#ef4444"
    conf_label = "✅ High Confidence" if conf>=80 else "⚠️ Moderate Confidence" if conf>=50 else "❌ Low Confidence"
    gsvg = gauge_svg(conf, 200, 16, dark=dark)

    # Top-3 cards HTML — all inline, no split tags
    cards_html = ""
    for i, item in enumerate(r["top_instruments"]):
        bg  = c["win_bg"] if i==0 else c["surface2"]
        bd  = c["win_bd"] if i==0 else c["border"]
        rbc = c["accent"] if i==0 else c["track"]
        rfc = "white"     if i==0 else c["muted"]
        bc  = c["accent"] if i==0 else c["track"]
        cards_html += f"""
<div style="background:{bg};border:1.5px solid {bd};border-radius:.85rem;padding:1rem">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.6rem">
    <span style="background:{rbc};color:{rfc};font-size:.7rem;font-weight:700;
                 padding:.15rem .55rem;border-radius:9999px">#{i+1}</span>
    <span style="font-size:1.3rem">{get_icon(item['name'])}</span>
  </div>
  <p style="font-family:Syne,sans-serif;font-weight:700;color:{c['text']};font-size:.93rem;
            text-transform:capitalize;margin:0 0 .45rem">{item['name']}</p>
  <div style="display:flex;justify-content:space-between;font-size:.76rem;margin-bottom:.32rem">
    <span style="color:{c['muted']}">Confidence</span>
    <span style="font-weight:800;color:{c['text']}">{item['confidence']:.1f}%</span>
  </div>
  {pbar(item['confidence'], bc, c['track'], 7)}
</div>"""

    st.markdown(f"""
<div style="background:{c['surface']};border:1px solid {c['border']};border-radius:1.1rem;
            padding:1.4rem;box-shadow:0 4px 24px rgba(0,0,0,.09);margin-bottom:1.2rem">
  <p style="font-family:Syne,sans-serif;font-size:.95rem;font-weight:700;color:{c['text']};
            margin:0 0 1rem;padding-bottom:.75rem;border-bottom:1px solid {c['border']}">
    📊 Analysis Results</p>
  <div style="background:{c['hero_bg']};border:1px solid {c['border']};border-radius:1rem;
              padding:2rem;display:flex;flex-wrap:wrap;align-items:center;justify-content:center;
              gap:2rem;margin-bottom:1.2rem;text-align:center">
    <div>{gsvg}</div>
    <div>
      <div style="font-size:4rem;line-height:1;margin-bottom:.4rem">{icon}</div>
      <div style="font-family:Syne,sans-serif;font-size:2.2rem;font-weight:800;
                  color:{c['text']};letter-spacing:-.04em;line-height:1">{inst['name'].upper()}</div>
      <div style="font-size:.68rem;font-weight:600;color:{c['muted']};text-transform:uppercase;
                  letter-spacing:.1em;margin-top:.3rem">Top Prediction</div>
      <div style="color:{conf_col};font-size:1rem;font-weight:700;margin-top:.4rem">{conf_label}</div>
    </div>
  </div>
  <p style="font-size:.72rem;font-weight:700;color:{c['muted']};text-transform:uppercase;
            letter-spacing:.09em;margin-bottom:.65rem">🔢 All Predictions</p>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem">{cards_html}</div>
</div>
""", unsafe_allow_html=True)

def render_quality(q, dark):
    c   = T(dark)
    lbl = q["label"]; conf = q["confidence"]
    gsvg = gauge_svg(conf, 170, 14, lbl, dark)
    badge_col = {"excellent":"#10b981","good":"#3b82f6","fair":"#f59e0b","poor":"#ef4444"}.get(lbl, c["accent"])
    bars = ""
    for label, score in q["all_scores"].items():
        is_w = label == lbl
        bc   = "#10b981" if is_w else c["track"]
        fw   = f"font-weight:700;color:#10b981" if is_w else f"color:{c['muted']}"
        bars += f"""
<div style="margin-bottom:.65rem">
  <div style="display:flex;justify-content:space-between;font-size:.78rem;margin-bottom:.25rem">
    <span style="{fw};text-transform:capitalize">{"▶ " if is_w else ""}{label}</span>
    <span style="{fw}">{score:.1f}%</span>
  </div>
  {pbar(score, bc, c['track'], 8)}
</div>"""
    st.markdown(f"""
<div style="background:{c['surface']};border:1px solid {c['border']};border-radius:1.1rem;
            padding:1.4rem;box-shadow:0 4px 24px rgba(0,0,0,.09);margin-bottom:1.2rem">
  <p style="font-family:Syne,sans-serif;font-size:.95rem;font-weight:700;color:{c['text']};
            margin:0 0 1rem;padding-bottom:.75rem;border-bottom:1px solid {c['border']}">
    🎚️ Audio Quality</p>
  <div style="text-align:center;margin-bottom:.9rem">
    {gsvg}
    <span style="display:inline-block;margin-top:.65rem;padding:.3rem .9rem;
                 background:{badge_col};color:white;border-radius:9999px;
                 font-weight:700;font-size:.74rem;letter-spacing:.05em">{lbl.upper()}</span>
  </div>
  {bars}
</div>
""", unsafe_allow_html=True)

def render_condition(cond, dark):
    c  = T(dark)
    cs = cond.get("all_scores", {})
    ms = cs.get("modern",0)+cs.get("clean",0)+cs.get("noisy",0)
    vs = cs.get("vintage",0); et = ms+vs or 1
    mp = (ms/et)*100; vp = (vs/et)*100; im = ms>=vs
    nr = cs.get("noisy",0); cr = cs.get("clean",0); isn = nr>cr
    mc = c["accent"] if im else c["muted"]
    vc = "#fb923c" if not im else c["muted"]
    sig_bg = c["snbg"] if isn else c["scbg"]
    sig_bd = c["snbd"] if isn else c["scbd"]
    sig_col= "#ef4444" if isn else "#10b981"
    era_m = (f'<span style="margin-left:.3rem;padding:.08rem .5rem;'
             f'background:{c["tb"]};color:{c["accent"]};border-radius:9999px;'
             f'font-size:.65rem;font-weight:700">ERA</span>') if im else ""
    era_v = (f'<span style="margin-left:.3rem;padding:.08rem .5rem;'
             f'background:{c["tp"]};color:#fb923c;border-radius:9999px;'
             f'font-size:.65rem;font-weight:700">ERA</span>') if not im else ""
    st.markdown(f"""
<div style="background:{c['surface']};border:1px solid {c['border']};border-radius:1.1rem;
            padding:1.4rem;box-shadow:0 4px 24px rgba(0,0,0,.09);margin-bottom:1.2rem">
  <p style="font-family:Syne,sans-serif;font-size:.95rem;font-weight:700;color:{c['text']};
            margin:0 0 1rem;padding-bottom:.75rem;border-bottom:1px solid {c['border']}">
    ⏳ Audio Condition</p>
  <p style="font-size:.66rem;font-weight:700;color:{c['muted']};text-transform:uppercase;
            letter-spacing:.08em;margin-bottom:.65rem">🕰️ Recording Era</p>
  <div style="margin-bottom:.65rem">
    <div style="display:flex;justify-content:space-between;font-size:.78rem;margin-bottom:.28rem">
      <span style="font-weight:700;color:{mc}">💿 Modern {era_m}</span>
      <span style="font-weight:800;color:{mc}">{mp:.1f}%</span>
    </div>
    {pbar(mp, c['accent'], c['track'], 16)}
  </div>
  <div style="margin-bottom:1rem">
    <div style="display:flex;justify-content:space-between;font-size:.78rem;margin-bottom:.28rem">
      <span style="font-weight:700;color:{vc}">🕰️ Vintage {era_v}</span>
      <span style="font-weight:800;color:{vc}">{vp:.1f}%</span>
    </div>
    {pbar(vp, '#fb923c', c['track'], 16)}
  </div>
  <p style="font-size:.66rem;font-weight:700;color:{c['muted']};text-transform:uppercase;
            letter-spacing:.08em;margin-bottom:.5rem">🔊 Signal Quality</p>
  <div style="background:{sig_bg};border:1px solid {sig_bd};border-radius:.85rem;
              padding:.85rem 1rem;display:flex;align-items:center;
              justify-content:space-between;margin-bottom:.75rem">
    <div style="display:flex;align-items:center;gap:.65rem">
      <span style="font-size:1.3rem">{"📻" if isn else "✨"}</span>
      <div>
        <p style="font-size:.88rem;font-weight:800;color:{sig_col};margin:0">{"Noisy" if isn else "Clean"}</p>
        <p style="font-size:.7rem;color:{c['muted']};margin:.1rem 0 0">
          {"Background noise or artifacts detected" if isn else "Clear signal with minimal noise"}</p>
      </div>
    </div>
    <span style="font-size:1.3rem">{"⚠️" if isn else "✅"}</span>
  </div>
  <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.35rem">
    <span style="font-size:.7rem;color:{c['muted']};width:90px;flex-shrink:0">Clean signal</span>
    <div style="flex:1">{pbar(cr,'#2dd4bf',c['track'],6)}</div>
    <span style="font-size:.7rem;font-weight:700;color:#0d9488;width:38px;text-align:right">{cr:.1f}%</span>
  </div>
  <div style="display:flex;align-items:center;gap:.5rem">
    <span style="font-size:.7rem;color:{c['muted']};width:90px;flex-shrink:0">Noisy signal</span>
    <div style="flex:1">{pbar(nr,'#f87171',c['track'],6)}</div>
    <span style="font-size:.7rem;font-weight:700;color:#dc2626;width:38px;text-align:right">{nr:.1f}%</span>
  </div>
</div>
""", unsafe_allow_html=True)

def render_info(result, dark):
    c  = T(dark)
    ai = result["audio_info"]
    st.markdown(f"""
<div style="background:{c['surface']};border:1px solid {c['border']};border-radius:1.1rem;
            padding:1.4rem;box-shadow:0 4px 24px rgba(0,0,0,.09);margin-bottom:1rem">
  <p style="font-family:Syne,sans-serif;font-size:.95rem;font-weight:700;color:{c['text']};
            margin:0 0 1rem;padding-bottom:.75rem;border-bottom:1px solid {c['border']}">
    ℹ️ Audio Information</p>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem">
    <div style="background:{c['tb']};border:1px solid {c['border']};border-radius:.85rem;
                padding:1rem;text-align:center">
      <div style="font-size:1.2rem;margin-bottom:.2rem">⏱️</div>
      <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.07em;color:{c['muted']};font-weight:600">Duration</div>
      <div style="font-size:.97rem;font-weight:800;color:{c['text']};margin-top:.1rem">{ai['duration']:.2f}s</div>
    </div>
    <div style="background:{c['tp']};border:1px solid {c['border']};border-radius:.85rem;
                padding:1rem;text-align:center">
      <div style="font-size:1.2rem;margin-bottom:.2rem">〰️</div>
      <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.07em;color:{c['muted']};font-weight:600">Sample Rate</div>
      <div style="font-size:.97rem;font-weight:800;color:{c['text']};margin-top:.1rem">{ai['sample_rate']} Hz</div>
    </div>
    <div style="background:{c['ti']};border:1px solid {c['border']};border-radius:.85rem;
                padding:1rem;text-align:center">
      <div style="font-size:1.2rem;margin-bottom:.2rem">💾</div>
      <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.07em;color:{c['muted']};font-weight:600">Samples</div>
      <div style="font-size:.97rem;font-weight:800;color:{c['text']};margin-top:.1rem">{ai['samples']:,}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇️  Download JSON",
            data=json.dumps({k:v for k,v in result.items() if k!="visualizations"}, indent=2),
            file_name=f"instrunet_{ts}.json", mime="application/json", use_container_width=True)
    with col2:
        st.download_button("📄  Download PDF", data=generate_pdf_bytes(result),
            file_name=f"instrunet_{ts}.pdf", mime="application/pdf", use_container_width=True)

def render_viz(result, dark):
    c = T(dark)
    st.markdown(f"""
<div style="background:{c['surface']};border:1px solid {c['border']};border-radius:1.1rem;
            padding:1.4rem;box-shadow:0 4px 24px rgba(0,0,0,.09);margin-bottom:1.2rem">
  <p style="font-family:Syne,sans-serif;font-size:.95rem;font-weight:700;color:{c['text']};
            margin:0 0 1rem;padding-bottom:.75rem;border-bottom:1px solid {c['border']}">
    〰️ Audio Visualizations</p>
  <p style="font-size:.66rem;font-weight:700;color:{c['muted']};text-transform:uppercase;
            letter-spacing:.08em;margin-bottom:.4rem">Waveform</p>
  <img src="{result['visualizations']['waveform']}"
       style="width:100%;border-radius:.85rem;border:1px solid {c['border']};margin-bottom:1.2rem"/>
  <p style="font-size:.66rem;font-weight:700;color:{c['muted']};text-transform:uppercase;
            letter-spacing:.08em;margin-bottom:.4rem">Mel-Spectrogram</p>
  <img src="{result['visualizations']['spectrogram']}"
       style="width:100%;border-radius:.85rem;border:1px solid {c['border']}"/>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def render_app(dark):
    c = T(dark)
    load_model_and_encoder()

    # Header — self-contained
    pills = "".join(
        f'<span style="background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.2);'
        f'border-radius:9999px;padding:.18rem .75rem;font-size:.7rem;font-weight:600;'
        f'color:white;letter-spacing:.04em">{t}</span>'
        for t in ["Instrument","Quality","Condition","Multi-Task CNN"]
    )
    st.markdown(f"""
<div style="background:{c['hdr_grad']};color:white;padding:1.4rem 2.5rem">
  <h1 style="font-family:Syne,sans-serif;font-size:2rem;font-weight:800;
             margin:0 0 .1rem;letter-spacing:-.04em">
    🎵 InstruNet <span style="opacity:.55;font-weight:400">AI</span></h1>
  <p style="color:rgba(255,255,255,.58);margin:0;font-size:.87rem">
    AI-Powered Music Instrument Recognition &amp; Analysis</p>
  <div style="display:flex;gap:.45rem;margin-top:.65rem;flex-wrap:wrap">{pills}</div>
</div>
""", unsafe_allow_html=True)

    # User bar + buttons
    u_col, t_col, lo_col = st.columns([6, 1.4, 1.2])
    with u_col:
        name = st.session_state.get("auth_name") or st.session_state.get("name","User")
        st.markdown(f"""
<div style="padding:.55rem 1rem;font-size:.79rem;color:{c['muted']};
            background:{c['surface2']};border-bottom:1px solid {c['border']}">
  👤 Signed in as <strong style="color:{c['text']}">{name}</strong>
</div>
""", unsafe_allow_html=True)
    with t_col:
        if st.button("☀️ Light" if dark else "🌙 Dark", key="app_theme"):
            st.session_state.dark_mode = not dark
            st.session_state.result = None; st.rerun()
    with lo_col:
        if st.button("Sign Out", key="signout_btn"):
            for k in ["auth_status","auth_name","auth_username",
                      "authentication_status","name","username","logout"]:
                st.session_state.pop(k, None)
            st.session_state.auth_status = None; st.rerun()

    # Padding wrapper — self-contained open+close in ONE markdown call
    st.markdown(f'<div style="padding:1.6rem 2.5rem;background:{c["bg"]}">', unsafe_allow_html=True)

    # ── Upload card — title in markdown, widget native, success in markdown ──
    st.markdown(f"""
<div style="background:{c['surface']};border:1px solid {c['border']};border-radius:1.1rem;
            padding:1.4rem 1.4rem .2rem;box-shadow:0 4px 24px rgba(0,0,0,.09);margin-bottom:.5rem">
  <p style="font-family:Syne,sans-serif;font-size:.95rem;font-weight:700;color:{c['text']};
            margin:0 0 1rem;padding-bottom:.75rem;border-bottom:1px solid {c['border']}">
    ☁️ Upload Audio File</p>
</div>
""", unsafe_allow_html=True)
    # File uploader sits outside the card div (Streamlit can't be inside HTML divs)
    af = st.file_uploader(
        "Drop an audio file here, or click to browse — WAV · MP3 · OGG · FLAC · M4A",
        type=["wav","mp3","ogg","flac","m4a"], label_visibility="visible"
    )
    if af:
        st.audio(af, format=af.type)
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:.65rem;padding:.6rem 1rem;
            background:{c['upload_ok_bg']};border-radius:.85rem;
            border:1px solid {c['upload_ok_bd']};margin-top:.5rem;margin-bottom:.5rem">
  <span style="font-size:1rem">✅</span>
  <div>
    <p style="font-weight:600;color:{c['text']};font-size:.84rem;margin:0">{af.name}</p>
    <p style="font-size:.7rem;color:{c['muted']};margin:0">{af.size/1024:.2f} KB</p>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

    # Analyze button
    _, bc, _ = st.columns([2, 2, 2])
    with bc:
        go = st.button("🧠  Analyze Audio", disabled=af is None, use_container_width=True)

    if go and af:
        with st.spinner("🔬 Analyzing audio…"):
            model, le = load_model_and_encoder()
            result = predict_audio(af.read(), model, le, dark)
        st.session_state.result = result

    if st.session_state.result:
        res = st.session_state.result
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        render_hero(res, dark)
        col1, col2 = st.columns(2)
        with col1: render_quality(res["quality"], dark)
        with col2: render_condition(res["condition"], dark)
        render_info(res, dark)
        render_viz(res, dark)

    st.markdown("</div>", unsafe_allow_html=True)  # padding wrapper close

    # Footer — self-contained
    st.markdown(f"""
<div style="background:{c['footer_bg']};color:{c['footer_txt']};text-align:center;
            padding:1.3rem;font-size:.77rem;border-top:1px solid {c['border']}">
  © 2026 InstruNet AI &nbsp;·&nbsp; Multi-Task CNN · Instrument · Quality · Condition
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    init_state()
    dark = st.session_state.dark_mode
    inject_css(dark)
    is_auth = (st.session_state.get("authentication_status") is True
               or st.session_state.auth_status is True)
    if is_auth:
        render_app(dark)
    else:
        render_auth(dark)

if __name__ == "__main__":
    main()
