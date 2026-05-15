# ═══════════════════════════════════════════════════════════════════════════
#   Pothole Prahari: Smart Sight for Smoother Ride
#   app/app.py  |  Run: streamlit run app/app.py
#
#   SINGLE SOURCE OF TRUTH:
#     All condition / speed / warning decisions come from run_analysis()
#     in detection.py.  app.py is display-only — zero decision logic.
#
#   Folder structure:
#   root/
#   ├── app/
#   │   ├── app.py               ← this file
#   │   └── detection.py         ← ONLY decision engine
#   ├── assets/
#   │   └── alert.wav
#   └── best.pt
# ═══════════════════════════════════════════════════════════════════════════

import os
import sys
import base64
import tempfile
import time
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

# ── Resolve project root & add to path ────────────────────────────────────────
_APP_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_APP_DIR, ".."))
for _p in (_APP_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Import the ONE decision engine ────────────────────────────────────────────
from detection import (                           # noqa: E402
    Detection, DetectionSummary, RoadAnalysis,
    classify_severity, run_analysis,
)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pothole Prahari | AI Detection",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── MODEL ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        from ultralytics import YOLO
        for candidate in (
            os.path.join(_ROOT_DIR, "best.pt"),
            os.path.join(_APP_DIR,  "best.pt"),
            "best.pt",
        ):
            if os.path.exists(candidate):
                return YOLO(candidate)
    except Exception:
        pass
    return None

model = load_model()

# ── SESSION STATE ──────────────────────────────────────────────────────────────
_DEFAULTS = {
    "run_webcam":          False,
    "last_alert":          0.0,
    "prev_condition":      None,
    "last_frame":          None,
    "last_analysis":       None,
    "accumulated_dets":    [],
    "frame_summaries":     [],
    "last_img_name":       None,
    "last_vid_name":       None,
    "last_vid_key":        None,
    "vid_bytes":            b"",
    "vid_processed":        False,
    "vid_stopped_by_user":  False,
    "vid_duration_sec":     0,
    "vid_worst_frame":      0,
    "vid_worst_seen":       "Good",
    "vid_acc":              {"total": 0, "minor": 0, "moderate": 0, "severe": 0},
    "vid_frames_processed": 0,
    "vid_frames_potholes":  0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── CURRENT PAGE ───────────────────────────────────────────────────────────────
try:
    page = st.query_params.get("page", "home")
except Exception:
    page = "home"


# ══════════════════════════════════════════════════════════════════════════════
#   YOLO → Detection list  (only conversion — no logic)
# ══════════════════════════════════════════════════════════════════════════════

def yolo_boxes_to_detections(boxes, frame_h: int, frame_w: int) -> list[Detection]:
    """Convert raw YOLO boxes to Detection objects. No severity / condition logic."""
    detections = []
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        x1, y1, x2, y2 = map(float, xyxy)
        detections.append(Detection(
            bbox=(x1, y1, x2, y2),
            confidence=conf,
            frame_width=frame_w,
            frame_height=frame_h,
        ))
    return detections


def accumulate_analysis(acc: dict) -> RoadAnalysis:
    """
    Cumulative approach — sum of ALL detections across the entire session.

    acc = {"total": int, "minor": int, "moderate": int, "severe": int}

    Builds a synthetic DetectionSummary from cumulative counts and calls
    run_analysis() exactly once.
    """
    if not acc or acc.get("total", 0) == 0:
        return run_analysis([])

    synthetic: list[Detection] = []
    FW, FH     = 1280, 720
    frame_area = FW * FH

    for _ in range(acc.get("minor", 0)):
        side = (0.002 * frame_area) ** 0.5
        synthetic.append(Detection(bbox=(0.0, 0.0, side, side),
                                   confidence=0.9, frame_width=FW, frame_height=FH))
    for _ in range(acc.get("moderate", 0)):
        side = (0.012 * frame_area) ** 0.5
        synthetic.append(Detection(bbox=(0.0, 0.0, side, side),
                                   confidence=0.9, frame_width=FW, frame_height=FH))
    for _ in range(acc.get("severe", 0)):
        side = (0.030 * frame_area) ** 0.5
        synthetic.append(Detection(bbox=(0.0, 0.0, side, side),
                                   confidence=0.9, frame_width=FW, frame_height=FH))

    return run_analysis(synthetic)


# ══════════════════════════════════════════════════════════════════════════════
#   FRAME ANNOTATION  (visual only — reads analysis for colours, no decisions)
# ══════════════════════════════════════════════════════════════════════════════

_SEV_BGR   = {"Minor": (34, 197, 94), "Moderate": (30, 180, 245), "Severe": (60, 60, 238)}
_SEV_THICK = {"Minor": 2, "Moderate": 3, "Severe": 4}
_COND_BGR  = {"Good": (34, 197, 94), "Moderate": (30, 165, 245), "Poor": (60, 60, 238)}


def draw_boxes(frame: np.ndarray, analysis: RoadAnalysis,
               fps: float = 0.0, conf_thresh: float = 0.25) -> np.ndarray:
    """Draw bounding boxes + HUD bar. All values read from analysis — no new logic."""
    frame   = frame.copy()
    img_h, img_w = frame.shape[:2]

    for det in analysis.summary.detections:
        sev   = classify_severity(det)
        color = _SEV_BGR[sev]
        thick = _SEV_THICK[sev]
        x1, y1, x2, y2 = map(int, det.bbox)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        label = f"{sev} | {int(det.confidence * 100)}%"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        fsc, ftk = 0.50, 1
        (tw, th), bl = cv2.getTextSize(label, font, fsc, ftk)
        lx1, ly1 = x1, max(y1 - th - bl - 8, 0)
        cv2.rectangle(frame, (lx1, ly1), (lx1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (lx1 + 4, y1 - bl - 2),
                    font, fsc, (255, 255, 255), ftk, cv2.LINE_AA)

    # HUD top bar
    fps_str  = f"{fps:.0f}" if fps > 0 else "--"
    bar_text = (f"  FPS: {fps_str}   |   Potholes: {analysis.summary.total}"
                f"   |   Conf: {conf_thresh:.2f}  ")
    font     = cv2.FONT_HERSHEY_SIMPLEX
    fsc, ftk = 0.55, 2
    (_, bh), bl = cv2.getTextSize(bar_text, font, fsc, ftk)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (img_w, bh + bl + 14), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.putText(frame, bar_text, (8, bh + 8), font, fsc, (200, 200, 200), ftk, cv2.LINE_AA)

    tag = f"  {analysis.condition}  "
    (tw, th), tbl = cv2.getTextSize(tag, font, fsc, ftk)
    cv2.rectangle(frame, (img_w - tw - 12, 2), (img_w - 2, th + tbl + 12),
                  _COND_BGR[analysis.condition], -1)
    cv2.putText(frame, tag, (img_w - tw - 8, bh + 8),
                font, fsc, (255, 255, 255), ftk, cv2.LINE_AA)

    return frame


# ══════════════════════════════════════════════════════════════════════════════
#   ALERT SOUND
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _alert_b64() -> str:
    path = os.path.join(_ROOT_DIR, "assets", "alert.wav")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

_ALERT_CACHE: dict = {}

def play_alert(current_condition: str):
    """
    Fire alert sound when condition changes to Poor.
    3-second cooldown. Never links with any report generation.
    """
    now  = time.time()
    prev = st.session_state.get("prev_condition", None)

    should_alert = (current_condition == "Poor") and (prev != current_condition)
    cooldown_ok  = (now - st.session_state.last_alert > 3.0)

    if should_alert and cooldown_ok:
        if _ALERT_CACHE.get("html") is None:
            b64 = _alert_b64()
            if b64:
                _ALERT_CACHE["html"] = (
                    f'<script>(function(){{'
                    f'try{{var b=atob("{b64}");'
                    f'var a=new Uint8Array(b.length);'
                    f'for(var i=0;i<b.length;i++)a[i]=b.charCodeAt(i);'
                    f'var u=URL.createObjectURL(new Blob([a],{{type:"audio/wav"}}));'
                    f'var x=new Audio(u);x.volume=0.85;'
                    f'x.play().catch(function(){{}});'
                    f'}}catch(e){{}}}})()</script>'
                )
        if _ALERT_CACHE.get("html"):
            st.components.v1.html(_ALERT_CACHE["html"], height=0)
        st.session_state.last_alert = now

    st.session_state.prev_condition = current_condition


# ══════════════════════════════════════════════════════════════════════════════
#   DASHBOARD HTML  —  reads only RoadAnalysis fields, zero logic
# ══════════════════════════════════════════════════════════════════════════════

_C = {
    "Good":     {"hex": "#22c55e", "glow": "rgba(34,197,94,0.35)",  "bg": "rgba(34,197,94,.08)",  "icon": "🟢", "label_bg": "#052e16"},
    "Moderate": {"hex": "#f59e0b", "glow": "rgba(245,158,11,0.35)", "bg": "rgba(245,158,11,.08)", "icon": "🟡", "label_bg": "#1c1003"},
    "Poor":     {"hex": "#ef4444", "glow": "rgba(239,68,68,0.40)",  "bg": "rgba(239,68,68,.08)",  "icon": "🔴", "label_bg": "#1f0505"},
}


def _dashboard_html(analysis: RoadAnalysis) -> str:
    """
    Build dashboard HTML from RoadAnalysis fields.
    NO score. NO density. NO rule trace. NO FPS. NO logic — display only.

    4-part flow (data → diagnosis → action → alert):
      1. Pothole Counts  — Total / Minor / Moderate / Severe pills
      2. Road Condition  — condition value with colour glow
      3. Recommended Speed — km/h value
      4. Alert Message   — warning banner
    """
    s     = analysis.summary
    cond  = analysis.condition
    speed = analysis.speed_kmh
    warn  = analysis.warning

    cc  = _C[cond]
    ch  = cc["hex"]
    cg  = cc["glow"]
    cb  = cc["bg"]
    ci  = cc["icon"]
    clb = cc["label_bg"]

    sg   = "font-family:'Space Grotesk',system-ui,sans-serif;"
    mono = "font-family:'JetBrains Mono',monospace;"

    def section_label(text):
        return (
            f'<div style="font-size:.56rem;font-weight:700;color:#888;'
            f'letter-spacing:2.8px;text-transform:uppercase;margin-bottom:10px;">'
            f'{text}</div>'
        )

    def card(content, accent=False):
        border = f'border:1px solid {ch}33;box-shadow:0 0 18px {cg};' if accent else 'border:1px solid #1e1e1e;'
        return (
            f'<div style="background:#222226;{border}'
            f'border-radius:13px;padding:13px 15px;">'
            f'{content}'
            f'</div>'
        )

    # ── 1. Pothole Counts ─────────────────────────────────────────────────────
    def pill(val, lbl, col):
        return (
            f'<div style="flex:1;background:#1e1e22;border:1px solid #222;'
            f'border-radius:10px;padding:10px 6px;text-align:center;">'
            f'<div style="{mono}font-size:1.4rem;font-weight:600;line-height:1;'
            f'margin-bottom:4px;color:{col};">{val}</div>'
            f'<div style="font-size:.54rem;font-weight:600;letter-spacing:1.6px;'
            f'text-transform:uppercase;color:#888;">{lbl}</div>'
            f'</div>'
        )

    counts_content = (
        section_label("① Pothole Counts") +
        f'<div style="display:flex;gap:7px;">'
        f'{pill(s.total,    "Total",    "#c8c8c8")}'
        f'{pill(s.minor,    "Minor",    "#22c55e")}'
        f'{pill(s.moderate, "Moderate", "#f59e0b")}'
        f'{pill(s.severe,   "Severe",   "#ef4444")}'
        f'</div>'
    )

    # ── 2. Road Condition ─────────────────────────────────────────────────────
    condition_content = (
        section_label("② Road Condition") +
        f'<div style="display:flex;align-items:center;justify-content:space-between;">'
        f'<div style="{mono}font-size:2.4rem;font-weight:600;color:{ch};line-height:1;'
        f'letter-spacing:-1px;text-shadow:0 0 28px {cg};">{ci} {cond}</div>'
        f'<div style="font-size:.62rem;font-weight:700;color:{ch};letter-spacing:2.5px;'
        f'text-transform:uppercase;background:{clb};padding:4px 10px;'
        f'border-radius:6px;border:1px solid {ch}33;">{cond.upper()}</div>'
        f'</div>'
    )

    # ── 3. Recommended Speed ──────────────────────────────────────────────────
    speed_content = (
        section_label("③ Recommended Speed") +
        f'<div style="display:flex;align-items:flex-end;gap:6px;">'
        f'<div style="{mono}font-size:3.2rem;font-weight:600;color:{ch};line-height:1;'
        f'text-shadow:0 0 24px {cg};">{speed}</div>'
        f'<div style="display:flex;flex-direction:column;gap:1px;padding-bottom:4px;">'
        f'<div style="font-size:.62rem;font-weight:700;color:{ch};'
        f'letter-spacing:1.5px;text-transform:uppercase;">km/h</div>'
        f'<div style="font-size:.56rem;color:#888;letter-spacing:1px;'
        f'text-transform:uppercase;">max safe speed</div>'
        f'</div>'
        f'</div>'
    )

    # ── 4. Alert Message ──────────────────────────────────────────────────────
    alert_content = (
        section_label("④ Alert") +
        f'<div style="background:{cb};border:1.5px solid {ch}55;border-radius:9px;'
        f'padding:10px 13px;font-size:.82rem;font-weight:600;color:{ch};'
        f'line-height:1.5;">{warn}</div>'
    )

    return (
        f'<div style="display:flex;flex-direction:column;gap:8px;{sg}">'
        f'{card(counts_content)}'
        f'{card(condition_content, accent=True)}'
        f'{card(speed_content, accent=True)}'
        f'{card(alert_content)}'
        f'</div>'
    )


def render_dashboard(analysis: RoadAnalysis, ph=None):
    html = _dashboard_html(analysis)
    if ph is not None:
        ph.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


# Idle placeholder
_IDLE = RoadAnalysis(
    condition="Good", speed_kmh=60,
    warning="✅  Road is safe for driving.",
    summary=DetectionSummary(total=0, minor=0, moderate=0, severe=0),
    display_score=100.0,
)


# ══════════════════════════════════════════════════════════════════════════════
#   GLOBAL CSS + NAVBAR
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Exo+2:wght@700;800;900&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap"
      rel="stylesheet" media="print" onload="this.media='all'">
""", unsafe_allow_html=True)

_A = {"home": "", "demo": "", "about": ""}
_A[page] = "pp-active"

st.markdown(f"""
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html,body,[data-testid="stAppViewContainer"],[data-testid="stAppViewContainer"]>section>div{{
    font-family:'Space Grotesk',system-ui,sans-serif!important;
    background-color:#111111!important;color:#c8c8c8!important;}}
#MainMenu,footer,header{{visibility:hidden!important;height:0!important}}
[data-testid="collapsedControl"],[data-testid="stDecoration"],[data-testid="stToolbar"]{{display:none!important}}
.main .block-container{{padding-top:78px!important;padding-bottom:3rem!important;max-width:1200px!important;padding-left:2rem!important;padding-right:2rem!important;}}

/* ── Navbar ── */
.pp-nav{{position:fixed;top:0;left:0;right:0;height:62px;z-index:999999;
  background:rgba(10,10,10,0.85);border-bottom:1px solid rgba(255,255,255,0.04);
  backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);
  display:flex;align-items:center;justify-content:space-between;padding:0 2.8rem;
  box-shadow:0 1px 0 rgba(230,126,34,0.08),0 8px 40px rgba(0,0,0,0.7);}}
.pp-brand{{display:flex;flex-direction:column;gap:0px;font-family:'Exo 2',sans-serif;text-decoration:none;line-height:1}}
.pp-brand-main{{font-size:1.18rem;font-weight:900;color:#e67e22;letter-spacing:.3px;line-height:1.15}}
.pp-brand-sub{{font-size:.54rem;font-weight:500;color:#888;letter-spacing:2.5px;text-transform:uppercase;line-height:1}}
.pp-links{{display:flex;align-items:center;
  background:rgba(255,255,255,0.03);
  border:1px solid rgba(255,255,255,0.07);
  border-radius:100px;padding:4px;gap:2px;
  box-shadow:inset 0 1px 0 rgba(255,255,255,0.04),0 2px 12px rgba(0,0,0,0.4);}}
.pp-link{{font-family:'Space Grotesk',sans-serif;color:#888;text-decoration:none!important;
  padding:.32rem 1rem;border-radius:100px;font-size:.82rem;font-weight:500;
  border:1px solid transparent;transition:all .18s ease;}}
.pp-link:hover{{color:#c8c8c8;background:rgba(255,255,255,0.06);border-color:rgba(255,255,255,0.08)}}
.pp-link.pp-active{{color:#e67e22!important;background:rgba(230,126,34,0.12);border-color:rgba(230,126,34,0.3);font-weight:700}}

/* ── Misc ── */
.pp-badge{{display:inline-block;background:#222226;border:1px solid #2e2e2e;
  color:#e67e22;padding:.25rem .9rem;border-radius:100px;
  font-size:.68rem;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:1rem;}}
.pp-divider{{border:none;border-top:1px solid #1e1e1e;margin:2rem 0}}

/* ── Hero ── */
.hero{{text-align:center;padding:4rem 1rem 3rem;border-radius:18px;overflow:hidden;
  background:#1c1c1e;border:1px solid #1e1e1e;margin-bottom:.5rem;position:relative;}}
.hero::before{{content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse 70% 60% at 50% 0%, rgba(230,126,34,.07) 0%, transparent 70%);
  pointer-events:none;}}
.hero>*{{position:relative;z-index:2}}
.hero-logo-wrap{{display:inline-flex;align-items:center;gap:.65rem;margin-bottom:.5rem}}
.hero-logo-icon{{font-size:2.8rem;line-height:1}}
.hero-logo-text{{font-family:'Exo 2',sans-serif;font-size:clamp(2rem,5.5vw,3.6rem);
  font-weight:900;color:#e67e22;letter-spacing:-1px;line-height:1.1;}}
.hero-tagline{{font-size:.85rem;font-weight:700;color:#e67e2299;letter-spacing:4px;
  text-transform:uppercase;margin-bottom:.6rem;font-family:'Exo 2',sans-serif;}}
.hero-subtitle{{font-size:1.05rem;font-weight:500;color:#c8c8c8;margin-bottom:0;}}

/* ── Stats ── */
.stats-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:.9rem;margin:2.5rem 0 2rem}}
.stat-box{{background:#222226;border:1px solid #1e1e1e;border-radius:12px;padding:1.3rem 1rem;text-align:center;}}
.stat-val{{font-family:'Exo 2',sans-serif;font-size:1.5rem;font-weight:900;color:#e67e22;line-height:1}}
.stat-lbl{{font-size:.64rem;color:#888;margin-top:.45rem;text-transform:uppercase;letter-spacing:1.8px}}

/* ── Features ── */
.feat-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1rem 0 2.5rem}}
.feat-card{{background:#1e1e22;border:1px solid #1e1e1e;border-radius:14px;padding:1.6rem 1.3rem;text-align:center;transition:all .2s ease;}}
.feat-card:hover{{border-color:#e67e2230;background:#222226;}}
.feat-icon{{font-size:1.8rem;margin-bottom:.8rem}}
.feat-name{{font-size:.88rem;font-weight:700;color:#e67e22;margin-bottom:.4rem;font-family:'Exo 2',sans-serif;}}
.feat-txt{{font-size:.79rem;color:#c8c8c8;line-height:1.7}}

/* ── How it works ── */
.hiw-wrap{{display:flex;align-items:center;justify-content:center;gap:0;margin:1.5rem 0 2.5rem;flex-wrap:wrap;}}
.hiw-step{{background:#1e1e22;border:1px solid #1e1e1e;border-radius:14px;padding:1.6rem 1.2rem;text-align:center;width:185px;transition:all .2s ease;}}
.hiw-step:hover{{border-color:#e67e2235;}}
.hiw-num{{width:28px;height:28px;background:#e67e22;border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:'Exo 2',sans-serif;font-size:.75rem;font-weight:900;color:#111;margin:0 auto .8rem;}}
.hiw-icon{{font-size:1.8rem;margin-bottom:.6rem}}
.hiw-title{{font-family:'Exo 2',sans-serif;font-size:.85rem;font-weight:800;color:#e67e22;margin-bottom:.35rem;}}
.hiw-desc{{font-size:.73rem;color:#c8c8c8;line-height:1.6}}
.hiw-arrow{{font-size:1.4rem;color:#888;padding:0 .5rem;align-self:center;flex-shrink:0;}}

/* ── Model warning ── */
.model-warn{{background:#1a0505;border:1px solid rgba(220,38,38,.3);border-radius:10px;
  padding:.9rem 1.4rem;color:#f87171;font-size:.86rem;text-align:center;margin-bottom:1.4rem}}

/* ── About ── */
.dev-card{{background:linear-gradient(145deg,#1c1c1e 0%,#222226 100%);border:1px solid #242424;
  border-top:2px solid #e67e22;border-radius:16px;padding:2.5rem 2rem;max-width:640px;margin:0 auto 2rem;text-align:center;}}
.dev-avatar{{width:80px;height:80px;background:linear-gradient(135deg,#e67e22,#f39c12);border-radius:50%;
  display:flex;align-items:center;justify-content:center;font-size:2.2rem;margin:0 auto 1.2rem;}}
.dev-name{{font-family:'Exo 2',sans-serif;font-size:1.5rem;font-weight:900;color:#e8e8e8;}}
.dev-role{{font-size:.68rem;color:#e67e22;font-weight:700;letter-spacing:3px;text-transform:uppercase;margin:.3rem 0 .2rem}}
.dev-college{{font-size:.82rem;color:#c8c8c8;margin-bottom:1.3rem}}
.dev-links{{display:flex;justify-content:center;gap:.65rem;flex-wrap:wrap}}
.dev-link{{display:inline-flex;align-items:center;gap:.3rem;background:#222226;border:1px solid #2a2a2a;
  color:#c8c8c8!important;text-decoration:none!important;padding:.38rem 1rem;border-radius:7px;
  font-size:.78rem;font-weight:500;transition:all .18s ease;}}
.dev-link:hover{{border-color:#e67e22;color:#e67e22!important;}}
.info-card{{background:#1e1e22;border:1px solid #1e1e1e;border-left:3px solid #e67e22;border-radius:12px;padding:1.5rem 1.7rem;margin-bottom:.9rem;}}
.info-card h3{{font-family:'Exo 2',sans-serif;font-size:.95rem;font-weight:800;color:#e67e22;margin-bottom:.8rem;}}
.info-card p,.info-card li{{font-size:.85rem;color:#c8c8c8;line-height:1.8}}
.info-card ul{{padding-left:1.1rem;margin:0}}
.info-card li{{margin-bottom:.25rem}}
.info-card strong{{color:#e8e8e8;font-weight:600}}
.info-card code{{background:#222226;border:1px solid #2a2a2a;color:#e67e22;
  padding:.1rem .38rem;border-radius:4px;font-size:.8rem;font-family:'JetBrains Mono',monospace;}}
.tech-pill{{display:inline-block;background:#1a0e02;border:1px solid #e67e2255;
  color:#e67e22;padding:.28rem .85rem;border-radius:100px;font-size:.74rem;font-weight:600;margin:.2rem;
  letter-spacing:.3px;}}
.flow-step{{display:flex;align-items:flex-start;gap:.9rem;margin-bottom:.85rem}}
.flow-num{{min-width:26px;height:26px;background:#e67e22;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-family:'Exo 2',sans-serif;font-size:.7rem;font-weight:900;color:#111;flex-shrink:0;margin-top:.1rem;}}
.flow-body{{font-size:.85rem;color:#c8c8c8;line-height:1.7}}
.flow-body strong{{color:#e8e8e8}}

/* ── Streamlit overrides ── */
.stSelectbox label,.stSlider label{{color:#c8c8c8!important;font-weight:600!important;font-size:.84rem!important;}}
.stButton>button{{background:linear-gradient(135deg,#e67e22 0%,#f39c12 100%)!important;
  color:#111!important;border:none!important;border-radius:8px!important;font-weight:700!important;
  font-family:'Space Grotesk',sans-serif!important;font-size:.86rem!important;
  padding:.48rem 1.4rem!important;transition:all .18s ease!important;
  box-shadow:0 2px 12px rgba(230,126,34,.25)!important;}}
.stButton>button:hover{{box-shadow:0 4px 20px rgba(230,126,34,.4)!important;transform:translateY(-1px)!important}}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] section>div button{{
  background:rgba(230,126,34,.1)!important;color:#e67e22!important;
  border:1px solid rgba(230,126,34,.4)!important;border-radius:7px!important;
  font-weight:700!important;font-size:.84rem!important;padding:.42rem 1.2rem!important;}}
[data-testid="stFileUploader"]{{background:#1e1e22!important;
  border:1px dashed rgba(230,126,34,.25)!important;border-radius:10px!important;padding:.5rem!important;max-width:100%;}}
[data-testid="stFileUploader"] label{{color:#c8c8c8!important;font-size:.86rem!important}}
[data-testid="stImage"] img{{border-radius:10px;border:1px solid #1e1e1e;object-fit:contain;width:100%;max-height:480px;}}
div[data-baseweb="select"]>div{{background-color:#222226!important;border-color:#2a2a2a!important;color:#e8e8e8!important;border-radius:8px!important;}}
[data-testid="stSlider"] [role="slider"]{{background-color:#e67e22!important}}

/* ── Perf tags ── */
.sev-legend{{display:flex;align-items:center;gap:12px;background:#222226;border:1px solid #1e1e1e;
  border-radius:9px;padding:8px 14px;font-size:.72rem;font-weight:700;}}
.fps-tag{{display:inline-block;background:#222226;border:1px solid #1e1e1e;color:#c8c8c8;
  padding:.26rem .72rem;border-radius:6px;font-size:.72rem;font-family:'JetBrains Mono',monospace;margin-top:.5rem;}}
.col-label{{font-size:.64rem;font-weight:700;color:#888;letter-spacing:2.2px;text-transform:uppercase;margin-bottom:8px;}}

@media(max-width:700px){{
  .stats-row{{grid-template-columns:repeat(2,1fr)}}
  .feat-grid{{grid-template-columns:1fr}}
  .hiw-wrap{{flex-direction:column;align-items:center}}
  .hiw-arrow{{transform:rotate(90deg)}}
  .pp-nav{{padding:0 1rem}}
  .pp-brand-sub{{display:none}}
  .pp-link{{padding:.3rem .6rem;font-size:.78rem}}
  .main .block-container{{padding-left:1rem!important;padding-right:1rem!important}}
}}
</style>

<nav class="pp-nav">
  <a href="?page=home" target="_self" class="pp-brand" style="text-decoration:none;">
    <span class="pp-brand-main">🚧 Pothole Prahari</span>
    <span class="pp-brand-sub">Smart sight for smoother ride</span>
  </a>
  <div class="pp-links">
    <a href="?page=home"  target="_self" class="pp-link {_A['home']}">🏠 Home</a>
    <a href="?page=demo"  target="_self" class="pp-link {_A['demo']}">🎥 Live Demo</a>
    <a href="?page=about" target="_self" class="pp-link {_A['about']}">👤 About</a>
  </div>
</nav>
""", unsafe_allow_html=True)

FOOTER_HTML = """
<div style="margin-top:4rem;border-top:1px solid #1a1a1a;padding:2rem 1rem 1.8rem;text-align:center;background:#111;">
  <div style="font-family:'Exo 2',sans-serif;font-size:1rem;font-weight:900;color:#e67e22;margin-bottom:.3rem;">🚧 Pothole Prahari</div>
  <div style="font-size:.62rem;color:#888;letter-spacing:3px;text-transform:uppercase;margin-bottom:1.1rem;">Smart Sight for Smoother Ride</div>
  <div style="display:flex;justify-content:center;gap:1.6rem;margin-bottom:1.2rem;flex-wrap:wrap;">
    <a href="?page=home"  target="_self" style="color:#c8c8c8;text-decoration:none;font-size:.8rem;">Home</a>
    <a href="?page=demo"  target="_self" style="color:#c8c8c8;text-decoration:none;font-size:.8rem;">Live Demo</a>
    <a href="?page=about" target="_self" style="color:#c8c8c8;text-decoration:none;font-size:.8rem;">About</a>
  </div>
  <div style="font-size:.68rem;color:#888;letter-spacing:1.5px;text-transform:uppercase;">
    © 2026 &nbsp;·&nbsp; <span style="color:#e67e2266;">AI-Powered Pothole Detection</span> &nbsp;·&nbsp; Built for Safer Rides
  </div>
</div>
"""


# ╔══════════════════════════════════════════════════════════════════════════════╗
#   PAGE ▸ HOME
# ╚══════════════════════════════════════════════════════════════════════════════╝
if page == "home":
    st.markdown("""
    <div class="hero">
      <div class="hero-logo-wrap">
        <span class="hero-logo-icon">🚧</span>
        <span class="hero-logo-text">Pothole Prahari</span>
      </div>
      <div class="hero-tagline">Smart Sight for Smoother Ride</div>
      <div class="hero-subtitle">AI-Powered Pothole Detection &amp; Road Intelligence System</div>
    </div>
    <div class="stats-row">
      <div class="stat-box"><div class="stat-val">YOLOv8</div><div class="stat-lbl">Core Model</div></div>
      <div class="stat-box"><div class="stat-val">Real‑Time</div><div class="stat-lbl">Detection Speed</div></div>
      <div class="stat-box"><div class="stat-val">3 Modes</div><div class="stat-lbl">Image · Video · Webcam</div></div>
      <div class="stat-box"><div class="stat-val">95%+</div><div class="stat-lbl">Accuracy</div></div>
    </div>
    <div style="text-align:center;margin:2rem 0 .8rem;">
      <div class="pp-badge">⚙️ Workflow</div>
      <div style="font-family:'Exo 2',sans-serif;font-size:1.25rem;font-weight:800;color:#e8e8e8;">How It Works</div>
    </div>
    <div class="hiw-wrap">
      <div class="hiw-step"><div class="hiw-num">1</div><div class="hiw-icon">📤</div><div class="hiw-title">Upload / Stream</div><div class="hiw-desc">Image, video, or live webcam feed</div></div>
      <div class="hiw-arrow">›</div>
      <div class="hiw-step"><div class="hiw-num">2</div><div class="hiw-icon">🧠</div><div class="hiw-title">YOLO Detection</div><div class="hiw-desc">Bounding boxes for every pothole</div></div>
      <div class="hiw-arrow">›</div>
      <div class="hiw-step"><div class="hiw-num">3</div><div class="hiw-icon">⚙️</div><div class="hiw-title">Analysis Engine</div><div class="hiw-desc">Rule-based road condition, safe speed, and hazard warning</div></div>
      <div class="hiw-arrow">›</div>
      <div class="hiw-step"><div class="hiw-num">4</div><div class="hiw-icon">🔔</div><div class="hiw-title">Live Alert</div><div class="hiw-desc">Audio on Poor condition detection</div></div>
    </div>
    <hr class="pp-divider">
    <div style="text-align:center;margin-bottom:1rem;">
      <div class="pp-badge">✨ Features</div>
      <div style="font-family:'Exo 2',sans-serif;font-size:1.25rem;font-weight:800;color:#e8e8e8;">What It Can Do</div>
    </div>
    <div class="feat-grid">
      <div class="feat-card"><div class="feat-icon">📸</div><div class="feat-name">Image Detection</div><div class="feat-txt">Upload a road image for instant severity-based pothole detection.</div></div>
      <div class="feat-card"><div class="feat-icon">🎬</div><div class="feat-name">Video Analysis</div><div class="feat-txt">Frame-by-frame rule-based analysis on dashcam footage.</div></div>
      <div class="feat-card"><div class="feat-icon">📷</div><div class="feat-name">Live Webcam</div><div class="feat-txt">Real-time feed with live dashboard for field testing.</div></div>
      <div class="feat-card"><div class="feat-icon">📊</div><div class="feat-name">Road Intelligence</div><div class="feat-txt">Outputs road condition, safe speed, and hazard warnings from a unified rule engine.</div></div>
      <div class="feat-card"><div class="feat-icon">⚡</div><div class="feat-name">Consistent Decision Logic</div><div class="feat-txt">Deterministic rules ensure stable and predictable outputs.</div></div>
      <div class="feat-card"><div class="feat-icon">🔔</div><div class="feat-name">Audio Alerts</div><div class="feat-txt">Instant alert triggered on poor road condition detection.</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
#   PAGE ▸ LIVE DEMO
# ╚══════════════════════════════════════════════════════════════════════════════╝
elif page == "demo":

    st.markdown("""
    <div style="text-align:center;padding:2rem 0 .8rem;">
      <div class="pp-badge">🎥 Detection Engine</div>
      <div style="font-family:'Exo 2',sans-serif;font-size:1.8rem;font-weight:900;color:#e67e22;margin:.4rem 0 .2rem;">
        Road Intelligence Dashboard
      </div>

    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.markdown(
            '<div class="model-warn">⚠️ <strong>Model not loaded.</strong> '
            'Place <code>best.pt</code> in the project root and restart.</div>',
            unsafe_allow_html=True)

    c1, c2 = st.columns([1.5, 1.2])
    with c1:
        mode = st.selectbox("🎛️  Detection Mode",
                            ["📸  Image", "🎬  Video", "📷  Webcam"])
    with c2:
        confidence = st.slider("🎯  Confidence", 0.10, 1.00, 0.25, 0.05, format="%.2f")

    st.markdown("<hr class='pp-divider' style='margin:.8rem 0'>", unsafe_allow_html=True)

    _LL = '<p class="col-label">📷 Detection Output</p>'
    _RL = '<p class="col-label">📊 Analysis &amp; Decision</p>'
    _RL_PUSH = '<div style="margin-top:-3.8rem;"></div>'
    _SEV_LEGEND = (
        '<div style="height:100%;display:flex;align-items:center;justify-content:center;">'
        '<div style="display:flex;align-items:center;gap:22px;'
        'background:#222226;border:1px solid #222;border-radius:12px;'
        'padding:14px 28px;font-size:.88rem;font-weight:700;">'
        '<span style="color:#22c55e;">■ &nbsp;Minor</span>'
        '<span style="color:#f59e0b;">■ &nbsp;Moderate</span>'
        '<span style="color:#ef4444;">■ &nbsp;Severe</span>'
        '</div></div>'
    )

    # ══════════════════════════════════════════════════════════════════════════
    #   IMAGE MODE
    # ══════════════════════════════════════════════════════════════════════════
    if "Image" in mode:
        up_col, leg_col = st.columns([1, 1])
        with up_col:
            uploaded = st.file_uploader("Upload a road / pothole image",
                                        type=["jpg", "jpeg", "png"])
        with leg_col:
            st.markdown(_SEV_LEGEND, unsafe_allow_html=True)

        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown(_LL, unsafe_allow_html=True)
            frame_ph = st.empty()
        with right:
            st.markdown(_RL, unsafe_allow_html=True)
            st.markdown(_RL_PUSH, unsafe_allow_html=True)
            dash_ph = st.empty()
            if st.session_state.last_analysis is None:
                render_dashboard(_IDLE, ph=dash_ph)

        if uploaded and model:
            if st.session_state.get("last_img_name") != uploaded.name:
                st.session_state.last_analysis   = None
                st.session_state.last_img_name   = uploaded.name

            raw  = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img  = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            if w > 1100:
                img = cv2.resize(img, (1100, int(h * 1100 / w)))
                h, w = img.shape[:2]

            with st.spinner("Running detection…"):
                t0      = time.time()
                results = model.predict(img, conf=confidence, verbose=False)
                elapsed = time.time() - t0
            fps_val = 1.0 / max(elapsed, 1e-4)

            # ── SINGLE decision call ──────────────────────────────────────────
            detections = yolo_boxes_to_detections(results[0].boxes, h, w)
            analysis   = run_analysis(detections)
            # ─────────────────────────────────────────────────────────────────

            annotated = draw_boxes(img, analysis, fps_val, confidence)

            with frame_ph.container():
                st.image(annotated, channels="BGR", use_container_width=True)
                st.markdown(
                    f'<div style="display:flex;gap:8px;margin-top:6px;flex-wrap:wrap;">'
                    f'<span class="fps-tag">⚡ {fps_val:.1f} FPS</span>'
                    f'<span class="fps-tag">⏱ {elapsed*1000:.0f} ms</span>'
                    f'<span class="fps-tag">🎯 Conf: {confidence:.2f}</span>'
                    f'</div>', unsafe_allow_html=True)

            render_dashboard(analysis, ph=dash_ph)

            st.session_state.last_frame    = annotated
            st.session_state.last_analysis = analysis

            play_alert(analysis.condition)

        elif uploaded and model is None:
            frame_ph.markdown(
                '<div class="model-warn">Model not loaded.</div>',
                unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #   VIDEO MODE
    # ══════════════════════════════════════════════════════════════════════════
    elif "Video" in mode:
        up_col, leg_col = st.columns([1, 1])
        with up_col:
            vid_up = st.file_uploader("Upload road / dashcam footage",
                                      type=["mp4", "avi", "mov"])
        with leg_col:
            st.markdown(_SEV_LEGEND, unsafe_allow_html=True)

        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown(_LL, unsafe_allow_html=True)
            frame_ph = st.empty()
            perf_ph  = st.empty()
        with right:
            st.markdown(_RL, unsafe_allow_html=True)
            st.markdown(_RL_PUSH, unsafe_allow_html=True)
            dash_ph = st.empty()
            if st.session_state.last_analysis is not None:
                render_dashboard(st.session_state.last_analysis, ph=dash_ph)
            else:
                render_dashboard(_IDLE, ph=dash_ph)

        if vid_up and model:
            vid_key  = getattr(vid_up, "file_id", f"{vid_up.name}_{vid_up.size}")
            new_file = (st.session_state.get("last_vid_key") != vid_key)
            if new_file:
                st.session_state.vid_bytes           = vid_up.read()
                st.session_state.last_vid_key        = vid_key
                st.session_state.last_vid_name       = vid_up.name
                st.session_state.last_analysis       = None
                st.session_state.prev_condition      = None
                st.session_state.vid_processed       = False
                st.session_state.frame_summaries     = []
                st.session_state.vid_stopped_by_user  = False
                st.session_state.vid_duration_sec     = 0
                st.session_state.vid_worst_frame      = 0
                st.session_state.vid_worst_seen       = "Good"
                st.session_state.vid_acc              = {"total": 0, "minor": 0, "moderate": 0, "severe": 0}
                st.session_state.vid_frames_processed = 0
                st.session_state.vid_frames_potholes  = 0
                render_dashboard(_IDLE, ph=dash_ph)

            stop_col, _ = st.columns([1, 5])
            with stop_col:
                stop_video = st.button("⏹  Stop", key="stop_vid")

            if stop_video:
                st.session_state.vid_stopped_by_user = True

            if not st.session_state.get("vid_processed", False):
                vid_bytes = st.session_state.get("vid_bytes", b"")
                if not vid_bytes:
                    st.error("Video file could not be read. Please re-upload.")
                    st.stop()

                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(vid_bytes); tfile.close()
                cap = cv2.VideoCapture(tfile.name)

                vid_fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
                vid_total_frm  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                real_duration  = round(vid_total_frm / max(vid_fps, 1.0))
                st.session_state.vid_duration_sec = real_duration

                acc = {"total": 0, "minor": 0, "moderate": 0, "severe": 0}
                frames_with_potholes = 0
                worst_seen      = "Good"
                worst_frame_idx = 0
                frame_idx       = 0
                frames_processed = 0

                while cap.isOpened():
                    if st.session_state.get("vid_stopped_by_user", False):
                        break

                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    h, w = frame.shape[:2]
                    if w > 1100:
                        frame = cv2.resize(frame, (1100, int(h * 1100 / w)))
                        h, w  = frame.shape[:2]

                    t0      = time.time()
                    results = model.predict(frame, conf=confidence, verbose=False)
                    fps_val = 1.0 / max(time.time() - t0, 1e-4)

                    frame_dets     = yolo_boxes_to_detections(results[0].boxes, h, w)
                    frame_analysis = run_analysis(frame_dets)
                    fs             = frame_analysis.summary
                    frames_processed += 1

                    acc["total"]    += fs.total
                    acc["minor"]    += fs.minor
                    acc["moderate"] += fs.moderate
                    acc["severe"]   += fs.severe
                    if fs.total > 0:
                        frames_with_potholes += 1

                    fc = frame_analysis.condition
                    if fc == "Poor" and worst_seen != "Poor":
                        worst_seen = "Poor"; worst_frame_idx = frame_idx
                    elif fc == "Moderate" and worst_seen == "Good":
                        worst_seen = "Moderate"; worst_frame_idx = frame_idx

                    st.session_state.vid_acc             = acc.copy()
                    st.session_state.vid_frames_processed = frames_processed
                    st.session_state.vid_frames_potholes = frames_with_potholes
                    st.session_state.vid_worst_frame     = worst_frame_idx
                    st.session_state.vid_worst_seen      = worst_seen

                    annotated = draw_boxes(frame, frame_analysis, fps_val, confidence)

                    frame_ph.image(annotated, channels="BGR", use_container_width=True)
                    perf_ph.markdown(
                        f'<div style="display:flex;gap:8px;margin-top:4px;flex-wrap:wrap;">'
                        f'<span class="fps-tag">⚡ {fps_val:.1f} FPS</span>'
                        f'<span class="fps-tag">⏱ {1/fps_val*1000:.0f} ms</span>'
                        f'<span class="fps-tag">🎯 Conf: {confidence:.2f}</span>'
                        f'</div>', unsafe_allow_html=True)

                    render_dashboard(frame_analysis, ph=dash_ph)
                    play_alert(frame_analysis.condition)

                    st.session_state.last_frame = annotated

                cap.release()
                try: os.unlink(tfile.name)
                except Exception: pass

                if st.session_state.get("vid_stopped_by_user", False):
                    st.session_state.vid_duration_sec = round(
                        frames_processed / max(vid_fps, 1.0))

                # Reset dashboard to idle — cumulative counts are session artefacts,
                # not the current road state after stopping.
                st.session_state.last_analysis = _IDLE
                st.session_state.vid_processed = True

                render_dashboard(_IDLE, ph=dash_ph)

                stopped = st.session_state.get("vid_stopped_by_user", False)
                done_c  = "#888" if stopped else "#22c55e"
                done_m  = "Stopped by user." if stopped else "✔ Video processing complete."
                st.markdown(
                    f'<p style="text-align:center;font-size:.8rem;color:{done_c};'
                    f'margin-top:.5rem;">{done_m}</p>',
                    unsafe_allow_html=True)

                st.rerun()

            if st.session_state.get("vid_processed", False):
                render_dashboard(_IDLE, ph=dash_ph)
                st.markdown(
                    '<p style="text-align:center;font-size:.8rem;color:#22c55e;'
                    'margin-top:.5rem;">✔ Video processing complete.</p>',
                    unsafe_allow_html=True)

        elif vid_up and model is None:
            st.markdown('<div class="model-warn">Model not loaded.</div>',
                        unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #   WEBCAM MODE
    # ══════════════════════════════════════════════════════════════════════════
    elif "Webcam" in mode:
        st.markdown(
            '<p style="color:#c8c8c8;font-size:.82rem;margin-bottom:.8rem;text-align:center;">'
            'Allow camera access, then click <strong style="color:#e8e8e8;">▶ Start Webcam</strong>.</p>',
            unsafe_allow_html=True)

        bc1, bc2, bc3 = st.columns([1, 1, 4])
        with bc1:
            if st.button("▶  Start Webcam", key="start_wc"):
                st.session_state.run_webcam      = True
                st.session_state.prev_condition       = None
                st.session_state.last_analysis        = None
                st.session_state.frame_summaries      = []
                st.session_state.vid_acc              = {"total": 0, "minor": 0, "moderate": 0, "severe": 0}
                st.session_state.vid_frames_processed = 0
                st.session_state.vid_frames_potholes  = 0
                st.session_state.vid_worst_seen       = "Good"
                st.session_state.vid_worst_frame      = 0
                st.session_state.vid_duration_sec     = 0
        with bc2:
            if st.button("⏹  Stop Webcam", key="stop_wc"):
                st.session_state.run_webcam = False
        with bc3:
            st.markdown(_SEV_LEGEND, unsafe_allow_html=True)

        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown(_LL, unsafe_allow_html=True)
            frame_ph = st.empty()
            perf_ph  = st.empty()
        with right:
            st.markdown(_RL, unsafe_allow_html=True)
            st.markdown(_RL_PUSH, unsafe_allow_html=True)
            dash_ph = st.empty()
            if st.session_state.last_analysis is not None:
                render_dashboard(st.session_state.last_analysis, ph=dash_ph)
            else:
                render_dashboard(_IDLE, ph=dash_ph)

        if st.session_state.run_webcam and model:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.markdown(
                    '<div class="model-warn">❌ Camera not accessible.</div>',
                    unsafe_allow_html=True)
            else:
                wc_acc              = {"total": 0, "minor": 0, "moderate": 0, "severe": 0}
                wc_frames_processed = 0
                wc_frames_potholes  = 0
                wc_worst_seen       = "Good"
                wc_worst_frame      = 0
                wc_frame_idx        = 0
                wc_start_time       = time.time()

                while st.session_state.run_webcam:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    h, w = frame.shape[:2]
                    if w > 1100:
                        frame = cv2.resize(frame, (1100, int(h * 1100 / w)))
                        h, w  = frame.shape[:2]

                    t0      = time.time()
                    results = model.predict(frame, conf=confidence, verbose=False)
                    fps_val = 1.0 / max(time.time() - t0, 1e-4)

                    frame_dets     = yolo_boxes_to_detections(results[0].boxes, h, w)
                    frame_analysis = run_analysis(frame_dets)
                    fs             = frame_analysis.summary
                    wc_frame_idx  += 1
                    wc_frames_processed += 1

                    wc_acc["total"]    += fs.total
                    wc_acc["minor"]    += fs.minor
                    wc_acc["moderate"] += fs.moderate
                    wc_acc["severe"]   += fs.severe
                    if fs.total > 0:
                        wc_frames_potholes += 1

                    fc = frame_analysis.condition
                    if fc == "Poor" and wc_worst_seen != "Poor":
                        wc_worst_seen = "Poor";  wc_worst_frame = wc_frame_idx
                    elif fc == "Moderate" and wc_worst_seen == "Good":
                        wc_worst_seen = "Moderate"; wc_worst_frame = wc_frame_idx

                    st.session_state.vid_acc              = wc_acc.copy()
                    st.session_state.vid_frames_processed = wc_frames_processed
                    st.session_state.vid_frames_potholes  = wc_frames_potholes
                    st.session_state.vid_worst_seen       = wc_worst_seen
                    st.session_state.vid_worst_frame      = wc_worst_frame

                    annotated = draw_boxes(frame, frame_analysis, fps_val, confidence)

                    frame_ph.image(annotated, channels="BGR", use_container_width=True)
                    perf_ph.markdown(
                        f'<div style="display:flex;gap:8px;margin-top:4px;flex-wrap:wrap;">'
                        f'<span class="fps-tag">⚡ {fps_val:.1f} FPS</span>'
                        f'<span class="fps-tag">⏱ {1/fps_val*1000:.0f} ms</span>'
                        f'<span class="fps-tag">🎯 Conf: {confidence:.2f}</span>'
                        f'</div>', unsafe_allow_html=True)

                    render_dashboard(frame_analysis, ph=dash_ph)
                    play_alert(frame_analysis.condition)

                    st.session_state.last_frame = annotated

                cap.release()

                st.session_state.vid_duration_sec    = round(time.time() - wc_start_time)
                st.session_state.vid_stopped_by_user = True

                # Reset dashboard to idle on stop
                st.session_state.last_analysis = _IDLE
                render_dashboard(_IDLE, ph=dash_ph)
                st.rerun()

        # ── Show idle dashboard after webcam stopped (runs on rerun) ───────────
        if (not st.session_state.run_webcam
                and st.session_state.last_analysis is not None):
            render_dashboard(_IDLE, ph=dash_ph)

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
#   PAGE ▸ ABOUT
# ╚══════════════════════════════════════════════════════════════════════════════╝
elif page == "about":
    st.markdown("""
    <div style="text-align:center;padding:2.5rem 0 1.2rem;">
      <div class="pp-badge">👤 The Team &amp; Project</div>
      <div style="font-family:'Exo 2',sans-serif;font-size:1.9rem;font-weight:900;color:#e67e22;margin:.5rem 0 .3rem;">About Developer</div>
      <p style="color:#c8c8c8;font-size:.86rem;margin:0;">The mind behind Pothole Prahari and the ideas that power it.</p>
    </div>
    <div class="dev-card">
      <div class="dev-avatar">👨‍💻</div>
      <div class="dev-name">Akash Kumar</div>
      <div class="dev-role">Computer Engineer</div>
      <div class="dev-college">B.E. Computer &nbsp;·&nbsp; D.A. Degree Engineering &amp; Technology &nbsp;·&nbsp; Batch 2022-26</div>
      <div class="dev-links">
        <a href="https://www.linkedin.com/in/imakash45/" class="dev-link">💼 LinkedIn</a>
        <a href="#" class="dev-link">🐙 GitHub</a>
        <a href="#" class="dev-link">📧 Email</a>
        <a href="#" class="dev-link">🌐 Portfolio</a>
      </div>
    </div>
    <hr class="pp-divider">
    <div style="font-family:'Exo 2',sans-serif;font-size:1.25rem;font-weight:800;color:#e8e8e8;text-align:center;margin-bottom:1.4rem;">🚧 About the Project</div>
    <div class="info-card">
      <h3>🎯 Objective</h3>
      <p>Pothole Prahari identifies potholes in real time and translates them into actionable driving insights—road condition awareness, safe speed guidance, and hazard warnings—reducing delayed reactions and improving road safety.</p>
    </div>
    <div class="info-card">
      <h3>❗ Real Problem Being Solved</h3>
      <ul>
        <li><strong>No real-time awareness</strong> — Drivers detect potholes too late, increasing accident risk.</li>
        <li><strong>Vehicle damage costs</strong> — Hidden potholes cause tire, suspension, and alignment damage.</li>
        <li><strong>Inefficient reporting</strong> — Manual reporting is slow, inconsistent, and often not acted on.</li>
        <li><strong>No actionable guidance</strong> — Existing systems detect hazards but don’t guide driver response.</li>
        <li><strong>Lack of structured data</strong> — Authorities lack reliable, session-level evidence for timely action.</li>
      </ul>
    </div>
    <div class="info-card">
      <h3>✅ Key Advantages</h3>
      <ul>
        <li><strong>Real-time and actionable</strong> — Provides road condition, safe speed, and hazard warnings instantly.</li>
        <li><strong>Explainable decisions</strong> — Clear rule-based outputs without black-box ambiguity.</li>
        <li><strong>Multi-input support</strong> — Works seamlessly with image, video, and live webcam.</li>
        <li><strong>Session-based reporting</strong> — Outputs reflect full-session analysis, not single-frame results.</li>
        <li><strong>Deployment-ready</strong> — Lightweight pipeline suitable for demos and edge deployment.</li>
      </ul>
    </div>
    <div class="info-card">
      <h3>🛠️ Tools &amp; Technologies</h3>
      <div style="margin-top:.6rem;">
        <span class="tech-pill">Python 3.10+</span>
        <span class="tech-pill">YOLOv8 (Ultralytics)</span>
        <span class="tech-pill">OpenCV</span>
        <span class="tech-pill">PyTorch</span>
        <span class="tech-pill">NumPy</span>
        <span class="tech-pill">Streamlit</span>
        <span class="tech-pill">Juyter Notebook</span>
        <span class="tech-pill">Google Colab</span>
        <span class="tech-pill">VS Code</span>
      </div>
    </div>

    <div class="info-card">
      <h3>🚀 Future Scope</h3>
      <ul>
        <li><strong>GPS tagging &amp; heatmaps</strong> — Map pothole locations for city-level analysis and planning.</li>
        <li><strong>Authority dashboard</strong> — Enable automated reporting and prioritization for municipal teams.</li>
        <li><strong>Mobile app integration</strong> — Support real-time detection using smartphone cameras.</li>
        <li><strong>Edge deployment</strong> — Deploy on devices like Raspberry Pi or Jetson Nano for roadside use.</li>
        <li><strong>Robust detection</strong> — Improve performance under low light, rain, and fog conditions.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)