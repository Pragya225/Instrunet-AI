import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import io
import base64
import json
import yaml
import os
from datetime import datetime
from yaml.loader import SafeLoader

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InstruNet AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# MINIMAL CSS  — only what Streamlit can't do natively
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none !important; }

/* Soft warm page background */
[data-testid="stAppViewContainer"],
[data-testid="stMain"] { background: #FAFAF8 !important; }

/* Card helper — used via st.markdown */
.sn-card {
    background: #FFFFFF;
    border: 1px solid #E8E6E1;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.sn-pill {
    display: inline-block;
    border-radius: 999px;
    padding: 3px 14px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-right: 6px;
}
.pill-excellent { background:#D1FAE5; color:#065F46; }
.pill-good      { background:#DBEAFE; color:#1E40AF; }
.pill-fair      { background:#FEF3C7; color:#92400E; }
.pill-poor      { background:#FEE2E2; color:#991B1B; }
.pill-modern    { background:#EDE9FE; color:#4C1D95; }
.pill-vintage   { background:#FEF3C7; color:#92400E; }
.pill-clean     { background:#D1FAE5; color:#065F46; }
.pill-noisy     { background:#FEE2E2; color:#991B1B; }

/* Progress bar track override */
.stProgress > div > div > div > div {
    border-radius: 999px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
QUALITY_LABELS   = ['excellent', 'good', 'fair', 'poor']
CONDITION_LABELS = ['modern', 'clean', 'noisy', 'vintage']
CONFIG_FILE      = "auth_config.yaml"

QUALITY_EMOJI = {
    'excellent': '🟢',
    'good':      '🔵',
    'fair':      '🟡',
    'poor':      '🔴',
}
CONDITION_EMOJI = {
    'modern':  '⚡',
    'vintage': '🎞️',
    'clean':   '✨',
    'noisy':   '📻',
}
INSTRUMENT_ICONS = {
    'brass':    '🎺', 'guitar':   '🎸', 'keyboard': '🎹',
    'mallet':   '🪘', 'organ':    '🎹', 'reed':     '🎷',
    'string':   '🎻', 'synth':    '🎛️', 'vocal':    '🎤',
    'flute':    '🪈', 'bass':     '🎸', 'drum':     '🥁',
}

# Bar colours for matplotlib charts
CHART_COLORS = ['#6366F1', '#A5B4FC', '#C7D2FE', '#E0E7FF']

def get_icon(name):
    n = (name or '').lower()
    for k, v in INSTRUMENT_ICONS.items():
        if k in n:
            return v
    return '🎵'


# ─────────────────────────────────────────────────────────────────────────────
# AUTH CONFIG
# ─────────────────────────────────────────────────────────────────────────────
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
            "name": "instrunet_auth",
            "key": st.secrets.get("COOKIE_KEY", "instrunet_super_secret_key_2026"),
            "expiry_days": 30,
        },
        "pre-authorized": {"emails": []},
    }
    for uname, hpw in seed_users.items():
        config["credentials"]["usernames"][uname] = {
            "name": uname.capitalize(),
            "email": f"{uname}@instrunet.ai",
            "password": hpw,
            "failed_login_attempts": 0,
            "logged_in": False,
        }
    save_config(config)
    return config


def save_config(config: dict):
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def hash_password(plain: str) -> str:
    import bcrypt
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


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


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        'auth_page': 'login',
        'result': None,
        'auth_status': None,
        'auth_name': '',
        'auth_username': '',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_encoder():
    from huggingface_hub import hf_hub_download
    from tensorflow import keras
    token    = st.secrets["HF_TOKEN"]
    repo_id  = st.secrets["HF_REPO_ID"]
    model_p  = hf_hub_download(repo_id=repo_id, filename="best_multitask_model.keras", token=token)
    enc_p    = hf_hub_download(repo_id=repo_id, filename="label_encoder.pkl", token=token)
    model    = keras.models.load_model(model_p)
    with open(enc_p, 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def process_audio(audio_bytes, sr=22050, duration=3.0):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr, duration=duration, mono=True)
    target = int(sr * duration)
    y = np.pad(y, (0, max(0, target - len(y))), mode='constant')[:target]
    return y, sr


def extract_mel(y, sr, n_mels=128, n_fft=2048, hop_length=512):
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_n  = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    if mel_n.shape[1] > 128:
        mel_n = mel_n[:, :128]
    elif mel_n.shape[1] < 128:
        mel_n = np.pad(mel_n, ((0, 0), (0, 128 - mel_n.shape[1])), mode='constant')
    return mel_n


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATIONS  (matplotlib, returned as pyplot figures)
# ─────────────────────────────────────────────────────────────────────────────
def _style_ax(ax, fig):
    """Apply consistent styling to a matplotlib axis."""
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FAFAF8')
    for sp in ax.spines.values():
        sp.set_edgecolor('#E8E6E1')
        sp.set_linewidth(0.8)
    ax.tick_params(colors='#6B7280', labelsize=9)
    ax.xaxis.label.set_color('#374151')
    ax.yaxis.label.set_color('#374151')
    ax.title.set_color('#111827')
    ax.grid(True, alpha=0.25, linewidth=0.5, color='#D1D5DB')


def fig_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 2.6))
    t = np.arange(len(y)) / sr
    ax.fill_between(t, y, alpha=0.18, color='#6366F1')
    ax.plot(t, y, color='#6366F1', linewidth=0.7, alpha=0.9)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Amplitude', fontsize=9)
    ax.set_title('Waveform', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlim(0, t[-1])
    _style_ax(ax, fig)
    plt.tight_layout()
    return fig


def fig_spectrogram(mel, sr):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    img = librosa.display.specshow(mel, sr=sr, x_axis='time', y_axis='mel',
                                   ax=ax, cmap='RdPu')
    ax.set_title('Mel-Spectrogram', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Frequency (Hz)', fontsize=9)
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label('Intensity (dB)', fontsize=8)
    cbar.ax.yaxis.set_tick_params(labelsize=7, color='#6B7280')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#6B7280')
    _style_ax(ax, fig)
    fig.patch.set_facecolor('#FFFFFF')
    plt.tight_layout()
    return fig


def fig_bar_chart(labels, values, title, colors=None):
    """Horizontal bar chart for quality / condition scores."""
    if colors is None:
        colors = CHART_COLORS
    fig, ax = plt.subplots(figsize=(5, 2.8))
    y_pos = np.arange(len(labels))
    bars  = ax.barh(y_pos, values, color=colors[:len(labels)],
                    height=0.55, zorder=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([l.capitalize() for l in labels], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Confidence (%)', fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=8, color='#374151')
    _style_ax(ax, fig)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


def fig_instrument_chart(top3):
    """Donut-style bar chart for top-3 instrument predictions."""
    names  = [x['name'].capitalize() for x in top3]
    vals   = [x['confidence'] for x in top3]
    cols   = ['#6366F1', '#A5B4FC', '#C7D2FE']
    fig, ax = plt.subplots(figsize=(5, 2.4))
    y_pos  = np.arange(len(names))
    bars   = ax.barh(y_pos, vals, color=cols, height=0.5, zorder=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlim(0, 105)
    ax.set_title('Top 3 Predictions', fontsize=10, fontweight='bold', pad=6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=8.5, fontweight='bold',
                color='#374151')
    _style_ax(ax, fig)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_audio(audio_bytes, model, label_encoder):
    y, sr  = process_audio(audio_bytes)
    mel    = extract_mel(y, sr)
    preds  = model.predict(mel.reshape(1, 128, 128, 1), verbose=0)

    inst_p = preds['instrument'][0]
    top3   = [
        {'name': label_encoder.classes_[i], 'confidence': float(inst_p[i] * 100)}
        for i in np.argsort(inst_p)[::-1][:3]
    ]

    qi      = np.argmax(preds['quality'][0])
    quality = {
        'label': QUALITY_LABELS[qi],
        'confidence': float(preds['quality'][0][qi] * 100),
        'all_scores': {QUALITY_LABELS[i]: float(preds['quality'][0][i] * 100) for i in range(4)},
    }

    ci        = np.argmax(preds['condition'][0])
    condition = {
        'label': CONDITION_LABELS[ci],
        'confidence': float(preds['condition'][0][ci] * 100),
        'all_scores': {CONDITION_LABELS[i]: float(preds['condition'][0][i] * 100) for i in range(4)},
    }

    return {
        'instrument':     top3[0],
        'top_instruments': top3,
        'quality':        quality,
        'condition':      condition,
        'audio_info': {
            'duration':    float(len(y) / sr),
            'sample_rate': int(sr),
            'samples':     len(y),
        },
        'waveform_fig':     fig_waveform(y, sr),
        'spectrogram_fig':  fig_spectrogram(mel, sr),
        'inst_chart_fig':   fig_instrument_chart(top3),
        'qual_chart_fig':   fig_bar_chart(
            list(quality['all_scores'].keys()),
            list(quality['all_scores'].values()),
            'Quality Breakdown',
            ['#10B981', '#3B82F6', '#F59E0B', '#EF4444'],
        ),
        'cond_chart_fig':   fig_bar_chart(
            list(condition['all_scores'].keys()),
            list(condition['all_scores'].values()),
            'Condition Breakdown',
            ['#8B5CF6', '#06B6D4', '#F97316', '#EC4899'],
        ),
        'timestamp': datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PDF EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf(result) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.units import inch

    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=letter,
                              leftMargin=0.8*inch, rightMargin=0.8*inch,
                              topMargin=0.8*inch, bottomMargin=0.8*inch)
    story  = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'Title2', parent=styles['Heading1'],
        fontSize=20, textColor=colors.HexColor('#111827'),
        spaceAfter=4, alignment=1,
    )
    sub_style = ParagraphStyle(
        'Sub', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#6B7280'),
        spaceAfter=20, alignment=1,
    )

    story.append(Paragraph("🎵 InstruNet AI — Analysis Report", title_style))
    story.append(Paragraph(datetime.now().strftime('%B %d, %Y  ·  %H:%M:%S'), sub_style))

    def make_table(data, col_widths, header_color):
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_color)),
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
            ('ALIGN',      (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING',    (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F9FAFB')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.HexColor('#F9FAFB'), colors.HexColor('#FFFFFF')]),
            ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
            ('ROWPADDING', (0, 0), (-1, -1), 6),
        ]))
        return t

    sections = [
        ("Instrument", [['Instrument', 'Confidence'],
                        [result['instrument']['name'].upper(),
                         f"{result['instrument']['confidence']:.1f}%"]], '#6366F1'),
        ("Audio Quality", [['Quality', 'Confidence'],
                           [result['quality']['label'].upper(),
                            f"{result['quality']['confidence']:.1f}%"]], '#10B981'),
        ("Recording Condition", [['Condition', 'Confidence'],
                                  [result['condition']['label'].upper(),
                                   f"{result['condition']['confidence']:.1f}%"]], '#F59E0B'),
        ("Audio Info", [['Property', 'Value'],
                        ['Duration', f"{result['audio_info']['duration']:.2f} s"],
                        ['Sample Rate', f"{result['audio_info']['sample_rate']:,} Hz"],
                        ['Samples', f"{result['audio_info']['samples']:,}"]], '#374151'),
    ]

    for heading, data, hcol in sections:
        story.append(Paragraph(f"<b>{heading}</b>", styles['Heading3']))
        story.append(make_table(data, [3*inch, 2.5*inch], hcol))
        story.append(Spacer(1, 0.22*inch))

    story.append(Paragraph("<b>Top 3 Predictions</b>", styles['Heading3']))
    t3_data = [['Rank', 'Instrument', 'Confidence']]
    for i, x in enumerate(result['top_instruments'], 1):
        t3_data.append([str(i), x['name'].capitalize(), f"{x['confidence']:.1f}%"])
    story.append(make_table(t3_data, [1*inch, 2.5*inch, 2*inch], '#6366F1'))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# AUTH PAGE  (pure Streamlit)
# ─────────────────────────────────────────────────────────────────────────────
def render_auth():
    # Centered narrow layout
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        st.markdown("<br>", unsafe_allow_html=True)

        # Logo / wordmark
        st.markdown(
            "<h1 style='text-align:center;font-size:2.4rem;margin-bottom:0;'>🎵</h1>"
            "<h2 style='text-align:center;margin-top:.2rem;margin-bottom:.1rem;"
            "font-size:1.8rem;font-weight:800;letter-spacing:-.03em;'>"
            "InstruNet <span style='color:#6366F1'>AI</span></h2>"
            "<p style='text-align:center;color:#9CA3AF;font-size:.85rem;"
            "margin-bottom:1.5rem;letter-spacing:.06em;text-transform:uppercase;'>"
            "Instrument Recognition</p>",
            unsafe_allow_html=True,
        )

        # Tab toggle
        tab_login, tab_reg = st.tabs(["Sign In", "Register"])

        # ── LOGIN ──
        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Welcome back 👋")
            st.caption("Sign in to start analysing audio.")

            auth, config = get_authenticator()
            login_result = auth.login(
                fields={
                    'Form name': 'Sign In', 'Username': 'Username',
                    'Password': 'Password', 'Login': 'Sign In',
                },
                key='login_widget',
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
                st.error("❌  Incorrect username or password.")

        # ── REGISTER ──
        with tab_reg:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Create an account 🎉")
            st.caption("Join InstruNet AI and start analysing.")

            new_name  = st.text_input("Full Name",         placeholder="Alex Johnson",      key="rn")
            new_user  = st.text_input("Username",          placeholder="alexj",             key="ru")
            new_email = st.text_input("Email",             placeholder="you@example.com",   key="re")
            new_pass  = st.text_input("Password",          placeholder="Min. 6 characters", type="password", key="rp1")
            new_pass2 = st.text_input("Confirm Password",  placeholder="Repeat password",   type="password", key="rp2")

            if st.button("Create Account ✨", use_container_width=True, key="reg_submit"):
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
                    config["credentials"]["usernames"][new_user] = {
                        "name": new_name, "email": new_email,
                        "password": hash_password(new_pass),
                        "failed_login_attempts": 0, "logged_in": False,
                    }
                    save_config(config)
                    st.success("✅  Account created! Sign in to continue.")
                    st.balloons()

        st.markdown(
            "<p style='text-align:center;font-size:.72rem;color:#D1D5DB;"
            "margin-top:1.5rem;'>Passwords are bcrypt-hashed · InstruNet AI © 2026</p>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# DIVIDER HELPER
# ─────────────────────────────────────────────────────────────────────────────
def section_header(emoji: str, title: str, subtitle: str = ""):
    st.markdown(
        f"<h3 style='margin-bottom:.15rem;margin-top:.25rem;font-size:1.05rem;"
        f"font-weight:700;'>{emoji} {title}</h3>"
        + (f"<p style='color:#9CA3AF;font-size:.82rem;margin-bottom:.5rem;'>{subtitle}</p>" if subtitle else ""),
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def render_app():
    # Warm-up model in background
    with st.spinner("🔧  Loading model…"):
        load_model_and_encoder()

    # ── TOP BAR ──
    user_name = st.session_state.get('auth_name') or st.session_state.get('name', 'User')
    col_logo, col_spacer, col_user, col_out = st.columns([2, 5, 2, 1])
    with col_logo:
        st.markdown(
            "<h2 style='margin:0;padding:.55rem 0;font-size:1.3rem;font-weight:800;"
            "letter-spacing:-.02em;'>🎵 Instru<span style='color:#6366F1'>Net</span></h2>",
            unsafe_allow_html=True,
        )
    with col_user:
        st.markdown(
            f"<p style='text-align:right;margin:0;padding:.7rem 0;"
            f"font-size:.82rem;color:#6B7280;'>👤 <b>{user_name}</b></p>",
            unsafe_allow_html=True,
        )
    with col_out:
        if st.button("Sign out", key="signout"):
            for k in ['auth_status', 'auth_name', 'auth_username',
                      'authentication_status', 'name', 'username', 'logout']:
                if k in st.session_state:
                    try:
                        del st.session_state[k]
                    except Exception:
                        pass
            st.session_state.auth_status = None
            st.rerun()

    st.markdown("<hr style='margin:.2rem 0 1.2rem;border:none;border-top:1px solid #E8E6E1;'>",
                unsafe_allow_html=True)

    # ── HERO BANNER ──
    st.markdown(
        "<div style='background:linear-gradient(135deg,#6366F1 0%,#8B5CF6 50%,#A78BFA 100%);"
        "border-radius:20px;padding:2.2rem 2.4rem 2rem;margin-bottom:1.6rem;'>"
        "<h1 style='color:#FFFFFF;font-size:2.2rem;font-weight:800;margin:0 0 .4rem;"
        "letter-spacing:-.03em;'>Instrument Recognition</h1>"
        "<p style='color:rgba(255,255,255,.8);font-size:.95rem;margin:0 0 1rem;'>"
        "Upload an audio clip — the multi-task CNN identifies the instrument, "
        "grades recording quality, and classifies condition.</p>"
        "<div style='display:flex;gap:.5rem;flex-wrap:wrap;'>"
        "<span style='background:rgba(255,255,255,.18);color:#fff;border-radius:999px;"
        "padding:.25rem .9rem;font-size:.72rem;font-weight:700;letter-spacing:.05em;"
        "text-transform:uppercase;'>🎸 Instrument</span>"
        "<span style='background:rgba(255,255,255,.18);color:#fff;border-radius:999px;"
        "padding:.25rem .9rem;font-size:.72rem;font-weight:700;letter-spacing:.05em;"
        "text-transform:uppercase;'>⭐ Quality</span>"
        "<span style='background:rgba(255,255,255,.18);color:#fff;border-radius:999px;"
        "padding:.25rem .9rem;font-size:.72rem;font-weight:700;letter-spacing:.05em;"
        "text-transform:uppercase;'>🎞️ Condition</span>"
        "<span style='background:rgba(255,255,255,.18);color:#fff;border-radius:999px;"
        "padding:.25rem .9rem;font-size:.72rem;font-weight:700;letter-spacing:.05em;"
        "text-transform:uppercase;'>🧠 Multi-task CNN</span>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    # ── TWO COLUMN LAYOUT ──
    left_col, right_col = st.columns([1.15, 1], gap="large")

    # ────────────────── LEFT — UPLOAD & PLAYBACK ──────────────────
    with left_col:
        section_header("📁", "Upload Audio",
                        "WAV · MP3 · OGG · FLAC  —  first 3 seconds are analysed")

        af = st.file_uploader(
            "Drag & drop or click to browse",
            type=['wav', 'mp3', 'ogg', 'flac', 'm4a'],
            label_visibility='collapsed',
        )

        if af:
            st.audio(af, format=af.type)
            c1, c2, c3 = st.columns(3)
            c1.metric("File", af.name[:18] + ("…" if len(af.name) > 18 else ""))
            c2.metric("Size", f"{af.size / 1024:.1f} KB")
            c3.metric("Type", af.type.split('/')[-1].upper())
            st.success(f"✅  **{af.name}** ready to analyse")

        st.markdown("<br>", unsafe_allow_html=True)

        col_btn, _ = st.columns([1, 1.5])
        with col_btn:
            go = st.button(
                "🔍  Analyse Audio",
                disabled=af is None,
                use_container_width=True,
                type="primary",
            )

        if go and af:
            with st.spinner("🎵  Analysing your audio…"):
                model, le = load_model_and_encoder()
                result = predict_audio(af.read(), model, le)
            st.session_state.result = result
            st.rerun()

        # ── PREVIOUS RESULT SUMMARY in left col ──
        if st.session_state.result:
            res = st.session_state.result

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("📊", "Audio Info")

            ai = res['audio_info']
            m1, m2, m3 = st.columns(3)
            m1.metric("⏱ Duration",    f"{ai['duration']:.2f} s")
            m2.metric("🔊 Sample Rate", f"{ai['sample_rate']:,} Hz")
            m3.metric("📦 Samples",     f"{ai['samples'] // 1000}k")

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("💾", "Export Results")

            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇️ JSON",
                    data=json.dumps(
                        {k: v for k, v in res.items()
                         if not k.endswith('_fig')},
                        indent=2,
                    ),
                    file_name=f"instrunet_{ts}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with dl2:
                st.download_button(
                    "⬇️ PDF Report",
                    data=generate_pdf(res),
                    file_name=f"instrunet_{ts}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    # ────────────────── RIGHT — RESULTS ──────────────────
    with right_col:
        if not st.session_state.result:
            # Empty state
            st.markdown(
                "<div style='text-align:center;padding:3.5rem 1rem;border:2px dashed #D1D5DB;"
                "border-radius:16px;background:#F9FAFB;'>"
                "<p style='font-size:2.5rem;margin-bottom:.5rem;'>🎧</p>"
                "<p style='font-size:1rem;font-weight:600;color:#374151;margin-bottom:.25rem;'>"
                "Results appear here</p>"
                "<p style='color:#9CA3AF;font-size:.85rem;'>"
                "Upload a file and click Analyse</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            res  = st.session_state.result
            inst = res['instrument']
            icon = get_icon(inst['name'])
            conf = inst['confidence']

            # ── Big instrument result ──
            conf_color = (
                "#10B981" if conf >= 80
                else "#F59E0B" if conf >= 50
                else "#EF4444"
            )
            conf_label = (
                "High confidence" if conf >= 80
                else "Moderate" if conf >= 50
                else "Low confidence"
            )

            st.markdown(
                f"<div style='background:#FFFFFF;border:1px solid #E8E6E1;border-radius:16px;"
                f"padding:1.4rem 1.6rem;margin-bottom:1rem;'>"
                f"<div style='display:flex;align-items:center;gap:1rem;margin-bottom:1rem;'>"
                f"<span style='font-size:2.8rem;'>{icon}</span>"
                f"<div>"
                f"<p style='font-size:1.6rem;font-weight:800;margin:0;letter-spacing:-.02em;"
                f"text-transform:capitalize;'>{inst['name']}</p>"
                f"<p style='margin:0;font-size:.78rem;color:#9CA3AF;text-transform:uppercase;"
                f"letter-spacing:.06em;font-weight:600;'>Detected Instrument</p>"
                f"</div>"
                f"</div>"
                f"<p style='margin:0;font-size:1rem;font-weight:700;color:{conf_color};'>"
                f"{conf:.1f}% · {conf_label}</p>"
                f"<div style='background:#F3F4F6;border-radius:999px;height:8px;margin-top:.5rem;'>"
                f"<div style='background:{conf_color};border-radius:999px;height:8px;"
                f"width:{conf:.0f}%;'></div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # ── Top 3 instrument chart ──
            st.pyplot(res['inst_chart_fig'], use_container_width=True)
            plt.close(res['inst_chart_fig'])

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Quality & Condition side by side ──
            q_col, c_col = st.columns(2)
            with q_col:
                section_header("⭐", "Quality")
                q = res['quality']
                q_emoji = QUALITY_EMOJI.get(q['label'], '⚪')
                pill_cls = f"pill-{q['label']}"
                st.markdown(
                    f"<span class='sn-pill {pill_cls}'>{q_emoji} {q['label'].upper()}</span>"
                    f"<span style='font-size:.8rem;color:#6B7280;'> {q['confidence']:.1f}%</span>",
                    unsafe_allow_html=True,
                )
                st.pyplot(res['qual_chart_fig'], use_container_width=True)
                plt.close(res['qual_chart_fig'])

            with c_col:
                section_header("🎞️", "Condition")
                cond = res['condition']
                c_emoji = CONDITION_EMOJI.get(cond['label'], '⚪')
                pill_cls = f"pill-{cond['label']}"
                st.markdown(
                    f"<span class='sn-pill {pill_cls}'>{c_emoji} {cond['label'].upper()}</span>"
                    f"<span style='font-size:.8rem;color:#6B7280;'> {cond['confidence']:.1f}%</span>",
                    unsafe_allow_html=True,
                )
                st.pyplot(res['cond_chart_fig'], use_container_width=True)
                plt.close(res['cond_chart_fig'])

    # ── VISUALISATIONS — full width below ──
    if st.session_state.result:
        res = st.session_state.result
        st.markdown(
            "<hr style='margin:1.5rem 0 1rem;border:none;border-top:1px solid #E8E6E1;'>",
            unsafe_allow_html=True,
        )
        section_header("📈", "Visualisations", "Waveform and Mel-Spectrogram of the analysed audio")

        v1, v2 = st.columns(2, gap="medium")
        with v1:
            st.markdown("**Waveform**")
            st.pyplot(res['waveform_fig'], use_container_width=True)
            plt.close(res['waveform_fig'])
        with v2:
            st.markdown("**Mel-Spectrogram**")
            st.pyplot(res['spectrogram_fig'], use_container_width=True)
            plt.close(res['spectrogram_fig'])

    # ── FOOTER ──
    st.markdown(
        "<hr style='margin:2rem 0 .75rem;border:none;border-top:1px solid #E8E6E1;'>"
        "<p style='text-align:center;font-size:.75rem;color:#D1D5DB;margin-bottom:1rem;'>"
        "© 2026 InstruNet AI · Multi-task CNN · Instrument · Quality · Condition</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    init_state()
    is_auth = (
        st.session_state.get("authentication_status") is True
        or st.session_state.auth_status is True
    )
    if is_auth:
        render_app()
    else:
        render_auth()


if __name__ == '__main__':
    main()
