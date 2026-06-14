import os
import tempfile
from pathlib import Path

import streamlit as st

from src.formatting import (
    analyze_silence,
    auto_assign_roles,
    format_segments_with_roles,
    get_unique_speakers,
    segments_to_speaker_text,
)
from src.predict import (
    generate_basic_comment,
    load_quality_model,
    manual_review_required,
    predict_quality_score,
    score_to_class,
)
from src.transcribe import transcribe_with_whisperx
from src.rule_engine import evaluate_rules


# Hugging Face token'ı ortam değişkeninden okunur (koda GÖMME!).
# Çalıştırmadan önce terminalde ayarla:  export HF_TOKEN="hf_xxx"
HF_TOKEN_HARDCODED = os.environ.get("HF_TOKEN", "")


def _hardcoded_hf_token():
    """Gömülü token'ı döndürür; placeholder hâlâ duruyorsa yok sayar."""
    token = (HF_TOKEN_HARDCODED or "").strip()
    if not token or token == "hf_BURAYA_TOKENINI_YAZ":
        return None
    return token


DEFAULT_TRANSCRIPT = """customer: Merhaba, faturamda anlamadigim bir ek ucret goruyorum.
representative: Merhaba, size hemen yardimci olayim. Hesabinizi kontrol edebilmem icin musteri numaranizi alabilir miyim?
customer: Tabii, 123456.
representative: Tesekkur ederim. Kisa bir kontrol sagliyorum. Ek ucretin gecen ayki paket degisikliginden kaynaklandigini goruyorum.
customer: Bunu bana kimse soylemedi, biraz sinirlendim acikcasi.
representative: Haklisiniz, bilgilendirmenin daha net yapilmasi gerekirdi. Durumu sizin icin aciklayayim ve uygun bir cozum olup olmadigini kontrol edeyim.
customer: Tamam, tesekkurler.
representative: Rica ederim. Bu ay icin duzeltme talebi olusturuyorum ve sonuc hakkinda SMS ile bilgilendirileceksiniz."""

CLASS_LABELS = {
    "Excellent": "Excellent",
    "Good": "Good",
    "Average": "Average",
    "Poor": "Poor",
}


@st.cache_resource(show_spinner=False)
def get_quality_model():
    return load_quality_model()


def apply_page_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        :root {
            --accent: #0d9488;
            --accent-strong: #0f766e;
            --accent-soft: #ccfbf1;
            --accent-tint: #f0fdfa;
            --ink: #0f172a;
            --muted: #64748b;
            --line: #e6ebf1;
            --panel: #ffffff;
            --page: #f5f8fa;
        }
        html, body, .stApp, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        .stApp {
            background:
                radial-gradient(1100px 500px at 12% -8%, var(--accent-tint) 0%, rgba(240,253,250,0) 55%),
                radial-gradient(900px 460px at 100% 0%, #eef6ff 0%, rgba(238,246,255,0) 50%),
                var(--page);
        }
        .main .block-container {
            max-width: 1200px;
            padding-top: 1.6rem;
            padding-bottom: 3.5rem;
        }
        h3 {
            color: var(--ink) !important;
            font-weight: 800 !important;
            letter-spacing: -0.01em;
        }

        /* HERO */
        .app-hero {
            position: relative;
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: center;
            background: linear-gradient(135deg, #ffffff 0%, #fbfffe 100%);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1.8rem 2rem;
            margin-bottom: 1.4rem;
            box-shadow: 0 18px 48px rgba(15, 23, 42, 0.07);
            overflow: hidden;
        }
        .app-hero::before {
            content: "";
            position: absolute;
            left: 0; top: 0; bottom: 0;
            width: 6px;
            background: linear-gradient(180deg, var(--accent) 0%, #2dd4bf 100%);
        }
        .app-hero .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            color: var(--accent-strong);
            background: var(--accent-soft);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            padding: 0.32rem 0.7rem;
            border-radius: 999px;
            margin-bottom: 0.7rem;
        }
        .app-hero h1 {
            margin: 0 0 0.5rem 0;
            color: var(--ink);
            font-size: 2.15rem;
            font-weight: 900;
            line-height: 1.18;
            letter-spacing: -0.02em;
        }
        .app-hero p {
            margin: 0;
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.55;
            max-width: 780px;
        }
        .hero-pill {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
            color: #ffffff;
            border-radius: 16px;
            padding: 0.95rem 1.2rem;
            font-weight: 700;
            font-size: 0.95rem;
            white-space: nowrap;
            box-shadow: 0 10px 24px rgba(13, 148, 136, 0.32);
        }
        .hero-pill span {
            display: block;
            color: var(--accent-soft);
            font-size: 0.78rem;
            font-weight: 600;
            opacity: 0.95;
        }
        .hero-art {
            flex: 0 0 auto;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* CARDS */
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1.1rem;
            margin: 1.1rem 0 1.4rem;
        }
        .dashboard-card,
        .section-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 16px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
            color: var(--ink);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        .dashboard-card {
            position: relative;
            min-height: 150px;
            padding: 1.4rem 1.3rem 1.25rem;
            overflow: hidden;
        }
        .dashboard-card::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--accent) 0%, #2dd4bf 100%);
        }
        .dashboard-card::after {
            content: "";
            position: absolute;
            right: -30px; top: -30px;
            width: 90px; height: 90px;
            border-radius: 50%;
            background: var(--accent-tint);
            opacity: 0.7;
            transition: transform 0.25s ease;
        }
        .dashboard-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 24px 48px rgba(15, 23, 42, 0.12);
            border-color: var(--accent-soft);
        }
        .dashboard-card:hover::after { transform: scale(1.25); }
        .dashboard-card .icon {
            position: relative;
            z-index: 1;
            width: 48px; height: 48px;
            display: flex; align-items: center; justify-content: center;
            border-radius: 14px;
            background: linear-gradient(135deg, var(--accent-soft) 0%, #ffffff 100%);
            border: 1px solid var(--accent-soft);
            color: var(--accent-strong);
            font-size: 1.35rem;
            margin-bottom: 1rem;
            box-shadow: 0 6px 16px rgba(13, 148, 136, 0.14);
        }
        .dashboard-card .label {
            position: relative; z-index: 1;
            color: var(--muted);
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.11em;
            text-transform: uppercase;
        }
        .dashboard-card .value {
            position: relative; z-index: 1;
            color: var(--ink);
            font-size: 1.7rem;
            font-weight: 900;
            letter-spacing: -0.025em;
            margin-top: 0.45rem;
            line-height: 1.1;
        }
        .dashboard-card .caption {
            position: relative; z-index: 1;
            color: var(--muted);
            font-size: 0.84rem;
            margin-top: 0.35rem;
        }
        .section-card {
            padding: 1.15rem 1.3rem;
            margin: 1rem 0;
            border-left: 4px solid var(--accent);
        }
        .section-card strong {
            color: var(--ink);
            font-size: 1.02rem;
        }
        .small-note {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.5;
        }

        /* RESULT */
        .result-grid {
            display: grid;
            grid-template-columns: 1.05fr 1fr 1fr;
            gap: 1.1rem;
            margin: 0.9rem 0 1.1rem;
        }
        .result-card {
            position: relative;
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1.25rem;
            min-height: 132px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        .result-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 20px 40px rgba(15, 23, 42, 0.1);
        }
        .result-card.highlight {
            background: linear-gradient(135deg, var(--accent-tint) 0%, #ffffff 70%);
            border-color: var(--accent-soft);
        }
        .result-card .label {
            color: var(--muted);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }
        .result-card .value {
            color: var(--ink);
            font-size: 2.1rem;
            font-weight: 900;
            letter-spacing: -0.02em;
            margin-top: 0.7rem;
        }
        .result-card.highlight .value { color: var(--accent-strong); }
        .result-card .subvalue {
            color: var(--muted);
            font-size: 0.88rem;
            margin-top: 0.15rem;
        }
        .comment-card {
            background: linear-gradient(135deg, #0f172a 0%, #134e4a 100%);
            border-radius: 16px;
            color: #e2e8f0;
            padding: 1.2rem 1.35rem;
            margin: 1rem 0;
            border-left: 6px solid var(--accent);
            line-height: 1.55;
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.18);
        }
        .comment-card strong {
            color: #5eead4;
            display: block;
            margin-bottom: 0.4rem;
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        /* BUTTONS */
        div.stButton > button {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
            color: #ffffff;
            border: none;
            border-radius: 12px;
            padding: 0.58rem 1.4rem;
            font-weight: 700;
            font-size: 0.95rem;
            box-shadow: 0 8px 20px rgba(13, 148, 136, 0.28);
            transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
        }
        div.stButton > button:hover {
            filter: brightness(1.06);
            transform: translateY(-1px);
            box-shadow: 0 12px 26px rgba(13, 148, 136, 0.36);
            color: #ffffff;
        }
        div.stButton > button:active { transform: translateY(0); }

        /* PROGRESS */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, var(--accent) 0%, #2dd4bf 100%) !important;
        }
        .stProgress > div > div {
            background: #e6ebf1 !important;
            border-radius: 999px;
        }

        /* TABS */
        div[data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: #eef2f6;
            padding: 0.35rem;
            border-radius: 14px;
            width: fit-content;
        }
        div[data-baseweb="tab"] {
            border-radius: 10px;
            padding: 0.5rem 1.15rem;
            background: transparent;
            border: none;
            color: var(--muted);
            font-weight: 700;
        }
        div[data-baseweb="tab"] p {
            color: var(--muted) !important;
            font-weight: 700;
        }
        div[data-baseweb="tab"][aria-selected="true"] {
            background: #ffffff;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
        }
        div[data-baseweb="tab"][aria-selected="true"] p {
            color: var(--accent-strong) !important;
        }
        div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
            display: none;
        }

        /* INPUTS */
        .stTextArea textarea, .stTextInput input {
            border-radius: 12px !important;
            border-color: var(--line) !important;
        }
        .stTextArea textarea:focus, .stTextInput input:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 2px var(--accent-soft) !important;
        }
        div[data-baseweb="select"] > div {
            border-radius: 12px !important;
            border-color: var(--line) !important;
        }
        div[data-testid="stAlert"] {
            border-radius: 12px;
            border: 1px solid var(--line);
        }
        div[data-testid="stExpander"] details {
            border-radius: 12px;
            border-color: var(--line);
        }

        /* RULE BREAKDOWN */
        .rule-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1rem;
            margin: 0.6rem 0 0.4rem;
        }
        .rule-cat {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
        }
        .rule-cat-title {
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: var(--accent-strong);
            margin-bottom: 0.6rem;
        }
        .rule-item {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.32rem 0;
            font-size: 0.92rem;
        }
        .rule-icon {
            width: 20px; height: 20px;
            flex: 0 0 20px;
            display: flex; align-items: center; justify-content: center;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 900;
        }
        .rule-pass .rule-icon { background: var(--accent-soft); color: var(--accent-strong); }
        .rule-fail .rule-icon { background: #fee2e2; color: #b91c1c; }
        .rule-label { flex: 1; color: var(--ink); }
        .rule-fail .rule-label { color: var(--muted); }
        .rule-pen {
            font-weight: 800;
            font-size: 0.85rem;
        }
        .rule-pass .rule-pen { color: var(--accent-strong); }
        .rule-fail .rule-pen { color: #b91c1c; }

        /* FEATURE CARDS */
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1.1rem;
            margin: 1.1rem 0 1.5rem;
        }
        .feature-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1.3rem 1.35rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        .feature-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 20px 40px rgba(15, 23, 42, 0.1);
        }
        .feature-card .f-icon {
            width: 44px; height: 44px;
            display: flex; align-items: center; justify-content: center;
            border-radius: 12px;
            background: var(--accent-soft);
            color: var(--accent-strong);
            font-size: 1.25rem;
            margin-bottom: 0.9rem;
        }
        .feature-card .f-title {
            font-size: 1.05rem;
            font-weight: 800;
            color: var(--ink);
            letter-spacing: -0.01em;
        }
        .feature-card .f-desc {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.5;
            margin-top: 0.3rem;
        }

        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }

        @media (max-width: 900px) {
            .app-hero { display: block; }
            .hero-pill { display: inline-block; margin-top: 1rem; }
            .dashboard-grid, .result-grid, .feature-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <div class="app-hero">
            <div>
                <span class="eyebrow">📞 NLP · Speech Analytics</span>
                <h1>Call Center Quality Scoring</h1>
                <p>Analyzes call center conversations with AI to produce a 0–100 quality score for representative performance.</p>
            </div>
            <div class="hero-art">
                <svg viewBox="0 0 220 190" width="180" height="160" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Customer service representative">
                    <circle cx="110" cy="95" r="80" fill="#f0fdfa"/>
                    <circle cx="110" cy="95" r="80" fill="none" stroke="#ccfbf1" stroke-width="2"/>
                    <path d="M52 180 Q58 132 110 132 Q162 132 168 180 Z" fill="#0d9488"/>
                    <path d="M94 124 h32 v14 q0 12 -16 12 q-16 0 -16 -12 Z" fill="#f8cda6"/>
                    <circle cx="110" cy="92" r="34" fill="#f8cda6"/>
                    <path d="M76 94 q-2 -40 34 -40 q36 0 34 40 q-7 -22 -34 -22 q-27 0 -34 22 Z" fill="#43352b"/>
                    <circle cx="99" cy="93" r="3.2" fill="#2c2620"/>
                    <circle cx="121" cy="93" r="3.2" fill="#2c2620"/>
                    <path d="M99 104 q11 9 22 0" fill="none" stroke="#2c2620" stroke-width="3" stroke-linecap="round"/>
                    <path d="M74 96 a36 36 0 0 1 72 0" fill="none" stroke="#0f766e" stroke-width="6" stroke-linecap="round"/>
                    <rect x="67" y="92" width="13" height="22" rx="6.5" fill="#0f766e"/>
                    <rect x="140" y="92" width="13" height="22" rx="6.5" fill="#0f766e"/>
                    <path d="M73 112 q-9 22 24 24" fill="none" stroke="#0f766e" stroke-width="5" stroke-linecap="round"/>
                    <circle cx="99" cy="136" r="5" fill="#0d9488"/>
                    <rect x="150" y="42" width="48" height="30" rx="11" fill="#ffffff" stroke="#ccfbf1" stroke-width="2"/>
                    <path d="M161 70 l-3 11 l13 -8 Z" fill="#ffffff"/>
                    <circle cx="163" cy="57" r="3.2" fill="#0d9488"/>
                    <circle cx="174" cy="57" r="3.2" fill="#2dd4bf"/>
                    <circle cx="185" cy="57" r="3.2" fill="#0d9488"/>
                </svg>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_prediction_result(rule_result, transcript):
    final_score = rule_result["final_score"]
    model_score = rule_result["model_score"]
    penalty_total = rule_result["penalty_total"]
    applied_cap = rule_result["applied_cap"]

    bonus_total = rule_result.get("bonus_total", 0)
    quality_class = score_to_class(final_score)
    quality_label = CLASS_LABELS.get(quality_class, quality_class)
    review_required = manual_review_required(final_score)

    st.markdown("### Analysis Result")
    review_label = "Required" if review_required else "Not Required"
    penalty_caption = (
        f"Model {model_score:.2f} + {bonus_total:.0f} bonus − {penalty_total:.0f} penalty"
        + (f", cap {applied_cap:.0f}" if applied_cap is not None else "")
    )
    st.markdown(
        f"""
        <div class="result-grid">
            <div class="result-card highlight">
                <div class="label">Final Quality Score</div>
                <div class="value">{final_score:.2f}/100</div>
                <div class="subvalue">{penalty_caption}</div>
            </div>
            <div class="result-card">
                <div class="label">Quality Level</div>
                <div class="value">{quality_label}</div>
                <div class="subvalue">{quality_class}</div>
            </div>
            <div class="result-card">
                <div class="label">Manual Review</div>
                <div class="value">{review_label}</div>
                <div class="subvalue">{rule_result['passed_count']}/{rule_result['passed_count'] + rule_result['failed_count']} checks passed</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(int(round(final_score)))
    st.markdown(
        f"""
        <div class="comment-card">
            <strong>Summary</strong>
            {generate_basic_comment(final_score)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_rule_breakdown(rule_result)

    st.markdown("### Transcript Sent to Model")
    st.text_area(
        "Analyzed text",
        value=transcript,
        height=230,
        disabled=True,
        label_visibility="collapsed",
    )


def _render_rule_breakdown(rule_result):
    """Kural kontrollerini kategori kategori, geçti/kaldı olarak gösterir."""
    st.markdown("### Rule-Based Evaluation")
    if rule_result["applied_cap"] is not None:
        st.markdown(
            f'<p class="small-note">⚠️ Critical violation detected — score is capped at '
            f'{rule_result["applied_cap"]:.0f} max.</p>',
            unsafe_allow_html=True,
        )

    silence = rule_result.get("silence")
    if silence:
        st.markdown(
            f'<p class="small-note">🕒 Silence analysis — total silence: '
            f'{silence["total_silence"]:g}s · longest gap: {silence["longest_gap"]:g}s · '
            f'dead air (>{silence["threshold"]:g}s): {silence["dead_air_count"]} event(s) '
            f'({silence["unannounced_count"]} unannounced, {silence["announced_count"]} announced).</p>',
            unsafe_allow_html=True,
        )

    # Kategoriye göre grupla
    categories = {}
    for check in rule_result["checks"]:
        categories.setdefault(check["category"], []).append(check)

    rows = []
    for category, checks in categories.items():
        items = ""
        for c in checks:
            if c["passed"]:
                bonus = c.get("bonus", 0)
                note = f"+{bonus:g}" if bonus else ""
                icon, cls = "✓", "rule-pass"
            else:
                icon, cls, note = "✕", "rule-fail", f"−{c['penalty']:g}"
            items += (
                f'<div class="rule-item {cls}">'
                f'<span class="rule-icon">{icon}</span>'
                f'<span class="rule-label">{c["label"]}</span>'
                f'<span class="rule-pen">{note}</span>'
                f'</div>'
            )
        rows.append(
            f'<div class="rule-cat"><div class="rule-cat-title">{category}</div>{items}</div>'
        )

    st.markdown(
        f'<div class="rule-grid">{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def run_quality_prediction(transcript, silence_stats=None):
    if not transcript.strip():
        st.warning("Please enter a conversation transcript to analyze.")
        return

    try:
        with st.spinner("Loading model and analyzing the conversation..."):
            tokenizer, model, device = get_quality_model()
            model_score = predict_quality_score(transcript, tokenizer, model, device)
            rule_result = evaluate_rules(transcript, model_score, silence_stats=silence_stats)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"An error occurred during analysis: {exc}")
        return

    show_prediction_result(rule_result, transcript)


def save_uploaded_file(uploaded_file):
    # st.file_uploader 'name' verir; st.audio_input (mikrofon) vermeyebilir.
    name = getattr(uploaded_file, "name", "") or ""
    suffix = Path(name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name


def transcribe_audio(audio_path):
    """Sesi yazıya çevirir ve WhisperX segmentlerini döndürür (hata olursa None)."""
    hf_token = (
        st.session_state.get("hf_token")
        or os.getenv("HF_TOKEN")
        or _hardcoded_hf_token()
    )
    if not hf_token:
        st.warning(
            "HF_TOKEN not found. Speaker diarization cannot run; all text will "
            "appear as a single speaker (SPEAKER_UNKNOWN). "
            "You can enter the token in the sidebar."
        )

    try:
        with st.spinner("Transcribing audio and separating speakers..."):
            return transcribe_with_whisperx(audio_path, hf_token=hf_token)
    except ImportError:
        st.error("WhisperX is not installed. Please run `pip install -r requirements.txt`.")
    except FileNotFoundError as exc:
        st.error(str(exc))
    except RuntimeError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"An error occurred while transcribing the audio: {exc}")
    return None


def run_audio_auto_analysis(uploaded_file):
    """Yüklenen sesi tek seferde uçtan uca işler: yazıya çevir -> konuşmacıları
    otomatik temsilci/müşteriye ata -> formatla -> kalite skoru üret ve göster."""
    audio_path = save_uploaded_file(uploaded_file)
    segments = transcribe_audio(audio_path)
    if not segments:
        return

    st.session_state["uploaded_segments"] = segments
    auto_roles = auto_assign_roles(segments)
    formatted_transcript = format_segments_with_roles(segments, auto_roles)
    st.session_state["uploaded_formatted"] = formatted_transcript

    # Ses analizinde sessizlik/ölü hava istatistiğini zaman damgalarından çıkar.
    silence_stats = analyze_silence(segments, role_map=auto_roles, threshold=10.0)
    run_quality_prediction(formatted_transcript, silence_stats=silence_stats)


def render_speaker_mapping_flow(state_key, form_key_prefix):
    segments = st.session_state.get(state_key)
    if not segments:
        return

    st.markdown("### Raw Speaker Transcript")
    st.text_area(
        "WhisperX output",
        value=segments_to_speaker_text(segments),
        height=260,
        disabled=True,
        label_visibility="collapsed",
    )

    speakers = get_unique_speakers(segments)
    auto_roles = auto_assign_roles(segments)
    role_options = ["customer", "representative"]

    st.markdown("### Speaker Mapping")
    st.markdown(
        '<p class="small-note">Speakers were auto-detected (greeting, identity verification, company phrases, resolution language and question-asking signals are weighted). If wrong, you can change them below.</p>',
        unsafe_allow_html=True,
    )

    speaker_role_map = {}
    columns = st.columns(min(len(speakers), 3) or 1)
    for index, speaker in enumerate(speakers):
        with columns[index % len(columns)]:
            default_role = auto_roles.get(speaker, "customer")
            role = st.selectbox(
                f"{speaker}",
                options=role_options,
                index=role_options.index(default_role),
                format_func=lambda value: "Customer" if value == "customer" else "Representative",
                key=f"{form_key_prefix}_{speaker}",
            )
            speaker_role_map[speaker] = role

    formatted_transcript = format_segments_with_roles(segments, speaker_role_map)

    st.markdown("### Mapped Transcript")
    st.text_area(
        "Transcript to send to the model",
        value=formatted_transcript,
        height=240,
        disabled=True,
        label_visibility="collapsed",
    )

    if st.button("Analyze Transcript", key=f"{form_key_prefix}_analyze", type="primary"):
        silence_stats = analyze_silence(segments, role_map=speaker_role_map, threshold=10.0)
        run_quality_prediction(formatted_transcript, silence_stats=silence_stats)


def render_text_tab():
    st.markdown(
        '<div class="section-card"><strong>Text Analysis</strong><br><span class="small-note">Enter the conversation transcript with customer and representative labels.</span></div>',
        unsafe_allow_html=True,
    )
    transcript = st.text_area(
        "Conversation transcript",
        value=DEFAULT_TRANSCRIPT,
        height=300,
        placeholder="customer: ...\nrepresentative: ...",
    )

    if st.button("Analyze", type="primary", key="analyze_text"):
        run_quality_prediction(transcript)


def render_upload_tab():
    st.markdown(
        '<div class="section-card"><strong>Upload Audio File</strong><br><span class="small-note">'
        'Upload a call recording in wav, mp3, m4a or mp4 format and click <strong>Analyze</strong>. '
        'The system automatically transcribes it, separates speakers into representative/customer, and produces the quality score.'
        '</span></div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Audio file",
        type=["wav", "mp3", "m4a", "mp4"],
        accept_multiple_files=False,
    )

    st.info(
        "ℹ️ Speakers are detected automatically. If you think the diarization or "
        "speaker roles are not correct, you can change them manually in the advanced "
        "panel below the result."
    )

    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("Analyze", key="auto_analyze_upload", type="primary"):
            run_audio_auto_analysis(uploaded_file)

    # Advanced: if auto speaker detection is wrong, fix and re-analyze.
    if st.session_state.get("uploaded_segments"):
        with st.expander("Fix speaker mapping manually (advanced)"):
            render_speaker_mapping_flow("uploaded_segments", "upload_role")


def render_sidebar():
    with st.sidebar:
        st.markdown("### Settings")

        # If the token is hardcoded or in an env var, hide the input;
        # just show a confirmation.
        preset_token = _hardcoded_hf_token() or os.getenv("HF_TOKEN")

        if preset_token:
            st.success("Hugging Face token set — diarization active.")
            st.caption("Using the token embedded in the code.")
            return

        st.text_input(
            "Hugging Face Token",
            type="password",
            key="hf_token",
            placeholder="hf_xxxxxxxx",
            help=(
                "Required for speaker diarization. "
                "Create a 'read' token at huggingface.co/settings/tokens and accept the "
                "model licenses for pyannote/speaker-diarization-3.1 and "
                "pyannote/segmentation-3.0."
            ),
        )
        if st.session_state.get("hf_token"):
            st.success("Token entered — diarization active.")
        else:
            st.info("No token — speakers cannot be separated.")


def main():
    st.set_page_config(
        page_title="Call Center Quality Analysis",
        page_icon="📞",
        layout="wide",
    )
    apply_page_style()
    render_sidebar()
    render_header()

    tab_text, tab_upload = st.tabs(["Text Analysis", "Upload Audio File"])

    with tab_text:
        render_text_tab()
    with tab_upload:
        render_upload_tab()


if __name__ == "__main__":
    main()
