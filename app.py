# SmartScribe - single-file Flask application
# Converted from the supplied Colab notebook.
# Colab/ngrok startup commands have been removed.
# Do NOT put API keys, passwords, or ngrok tokens in this file.

import os
import subprocess
import shutil
import time
import threading
import hashlib
import uuid
import base64
import io
import re
import json
import sqlite3
import wave
import struct

import numpy as np
import cv2
import face_recognition

from datetime import datetime
from contextlib import contextmanager
from flask import Flask, render_template_string, request, jsonify, send_file
from flask_cors import CORS
import PyPDF2
from flask import make_response

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT


# ==================== ORIGINAL NOTEBOOK CELL 2 ====================
# ==================== CELL 2: IMPORTS & DATABASE ====================
from contextlib import contextmanager



from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ==================== DATABASE SETUP ====================
DB_PATH = 'smartscribe.db'

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS students (
                reg_no TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                image TEXT NOT NULL,
                face_encoding TEXT NOT NULL,
                voice_embedding TEXT,
                voice_wav_b64 TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
        for col in [('voice_embedding','TEXT'), ('voice_wav_b64','TEXT')]:
            try: c.execute(f'ALTER TABLE students ADD COLUMN {col[0]} {col[1]}')
            except: pass
        c.execute('''
            CREATE TABLE IF NOT EXISTS exams (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                pdf TEXT NOT NULL,
                questions TEXT NOT NULL,
                duration_minutes INTEGER DEFAULT 60,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
        try: c.execute('ALTER TABLE exams ADD COLUMN duration_minutes INTEGER DEFAULT 60')
        except: pass
        c.execute('''
            CREATE TABLE IF NOT EXISTS exam_submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reg_no TEXT NOT NULL,
                student_name TEXT NOT NULL,
                exam_id TEXT NOT NULL,
                exam_name TEXT NOT NULL,
                answers TEXT NOT NULL,
                pdf_data BLOB,
                pdf_filename TEXT,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
        for col in [('student_name','TEXT DEFAULT ""'),('exam_name','TEXT DEFAULT ""'),
                    ('pdf_data','BLOB'),('pdf_filename','TEXT')]:
            try: c.execute(f'ALTER TABLE exam_submissions ADD COLUMN {col[0]} {col[1]}')
            except: pass
        print('✅ Database initialized')

init_db()


# ==================== ORIGINAL NOTEBOOK CELL 3 ====================
# ==================== CELL 3: FLASK APP & ALL CORE FUNCTIONS ====================

app = Flask(__name__)
app.secret_key = 'smartscribe_key'
CORS(app)

# ─── FACE HELPERS ─────────────────────────────────────────
def extract_face_features(image_data):
    """Extract exactly one face encoding from a base64 image."""
    try:
        if not image_data:
            return None
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        img_bytes = base64.b64decode(image_data, validate=True)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Use HOG for fast browser verification and require exactly one face.
        locations = face_recognition.face_locations(rgb_img, model='hog')
        if len(locations) == 0:
            print('Face extraction: no face detected')
            return None
        if len(locations) > 1:
            print(f'Face extraction: {len(locations)} faces detected')
            return None

        encs = face_recognition.face_encodings(rgb_img, known_face_locations=locations, num_jitters=2)
        if not encs:
            return None
        return encs[0].tolist()
    except Exception as e:
        print(f"Face extraction error: {e}")
        return None

# IMPORTANT: lower than the old 0.60 threshold.
# 0.45 is deliberately stricter to reduce false acceptance of another person.
FACE_MATCH_THRESHOLD = 0.45

def verify_face(registered_encoding, captured_image):
    try:
        if not registered_encoding:
            return False, "No registered face"
        cap_enc = extract_face_features(captured_image)
        if cap_enc is None:
            return False, "Show only one face clearly in the camera"

        reg = np.asarray(registered_encoding, dtype=np.float64)
        cap = np.asarray(cap_enc, dtype=np.float64)
        if reg.shape != cap.shape:
            return False, "Registered face data is invalid"

        dist = float(np.linalg.norm(reg - cap))
        print(f"Face distance: {dist:.4f} | threshold: {FACE_MATCH_THRESHOLD}")

        if dist <= FACE_MATCH_THRESHOLD:
            return True, f"Face verified (distance={dist:.3f})"
        return False, f"Face does not match (distance={dist:.3f})"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Face error: {e}"

# ─── WAV CONVERSION ────────────────────────────────────────
# Browser records as audio/webm (or audio/wav on some browsers).
# We convert anything to raw PCM using pydub+ffmpeg so MFCC works reliably.
def audio_b64_to_pcm_numpy(audio_b64: str, target_sr: int = 16000):
    """
    Convert browser-recorded audio (WebM/Opus/WAV/OGG/etc.)
    into mono 16 kHz PCM using FFmpeg.
    Returns (numpy float32 PCM, clean WAV base64).
    """
    try:
        if shutil.which("ffmpeg") is None:
            print("ERROR: FFmpeg not found in PATH")
            return None, None

        if audio_b64.startswith("data:") and "," in audio_b64:
            audio_b64 = audio_b64.split(",", 1)[1]

        raw_bytes = base64.b64decode(audio_b64)
        if not raw_bytes:
            print("ERROR: Empty audio data")
            return None, None

        print(f"Received audio: {len(raw_bytes)} bytes")

        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-i", "pipe:0", "-ar", str(target_sr), "-ac", "1",
             "-sample_fmt", "s16", "-f", "wav", "pipe:1"],
            input=raw_bytes, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=True
        )

        wav_bytes = result.stdout
        if not wav_bytes:
            print("ERROR: FFmpeg produced empty WAV")
            return None, None

        print(f"FFmpeg converted audio to WAV: {len(wav_bytes)} bytes")

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.readframes(wf.getnframes())

        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        wav_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        print(f"Audio processed successfully: {len(pcm)} samples, {len(pcm)/target_sr:.2f} seconds")
        return pcm, wav_b64

    except subprocess.CalledProcessError as e:
        print("FFmpeg conversion error:")
        print(e.stderr.decode("utf-8", errors="ignore"))
        return None, None
    except Exception as e:
        print("audio_b64_to_pcm_numpy error:", e)
        return None, None

# ─── MFCC VOICE EMBEDDING ──────────────────────────────────
# Pure numpy: fast, no heavy ML library required.
# ─── IMPROVED VOICE VERIFICATION ───────────────────────────

def extract_mfcc_embedding(pcm, sr=16000, n_mfcc=20, n_fft=512, hop=160):
    """
    Create a more speaker-sensitive voice embedding.

    Uses:
    - MFCC mean
    - MFCC standard deviation
    - Delta MFCC mean
    - Delta MFCC standard deviation

    This is stronger than using only MFCC mean.
    """

    if pcm is None or len(pcm) < sr * 1.0:
        return None

    # Convert to float32
    pcm = np.asarray(pcm, dtype=np.float32)

    # Remove DC offset
    pcm = pcm - np.mean(pcm)

    # Normalize volume
    max_val = np.max(np.abs(pcm))
    if max_val > 0:
        pcm = pcm / max_val

    # Pre-emphasis
    pre = np.append(
        pcm[0],
        pcm[1:] - 0.97 * pcm[:-1]
    )

    # Framing
    frames = []

    for start in range(0, len(pre) - n_fft + 1, hop):
        frame = pre[start:start + n_fft]

        if len(frame) == n_fft:
            frames.append(
                frame * np.hamming(n_fft)
            )

    if len(frames) < 5:
        return None

    frames = np.asarray(frames, dtype=np.float32)

    # FFT
    spectrum = np.fft.rfft(frames, n=n_fft)
    magnitude = np.abs(spectrum)

    power = magnitude ** 2

    # ---------------------------------------------------------
    # MEL FILTER BANK
    # ---------------------------------------------------------

    n_mels = 40

    mel_min = 0
    mel_max = 2595 * np.log10(
        1 + (sr / 2) / 700
    )

    mel_points = np.linspace(
        mel_min,
        mel_max,
        n_mels + 2
    )

    hz_points = 700 * (
        10 ** (mel_points / 2595) - 1
    )

    bins = np.floor(
        (n_fft + 1) * hz_points / sr
    ).astype(int)

    filter_bank = np.zeros(
        (n_mels, n_fft // 2 + 1),
        dtype=np.float32
    )

    for m in range(1, n_mels + 1):

        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center > left:
            for k in range(left, center):
                filter_bank[m - 1, k] = (
                    (k - left) /
                    (center - left)
                )

        if right > center:
            for k in range(center, right):
                filter_bank[m - 1, k] = (
                    (right - k) /
                    (right - center)
                )

    # Mel energy
    mel_energy = np.dot(
        power,
        filter_bank.T
    )

    mel_energy = np.maximum(
        mel_energy,
        1e-10
    )

    log_mel = np.log(mel_energy)

    # ---------------------------------------------------------
    # DCT → MFCC
    # ---------------------------------------------------------

    dct_matrix = np.cos(
        np.pi / n_mels *
        np.outer(
            np.arange(n_mfcc),
            np.arange(0.5, n_mels)
        )
    )

    mfcc = np.dot(
        log_mel,
        dct_matrix.T
    )

    # Ignore energy coefficient
    mfcc = mfcc[:, 1:]

    # ---------------------------------------------------------
    # DELTA MFCC
    # ---------------------------------------------------------

    if len(mfcc) > 2:

        delta = np.gradient(
            mfcc,
            axis=0
        )

    else:
        delta = np.zeros_like(mfcc)

    # ---------------------------------------------------------
    # BUILD STRONGER EMBEDDING
    # ---------------------------------------------------------

    mfcc_mean = np.mean(
        mfcc,
        axis=0
    )

    mfcc_std = np.std(
        mfcc,
        axis=0
    )

    delta_mean = np.mean(
        delta,
        axis=0
    )

    delta_std = np.std(
        delta,
        axis=0
    )

    embedding = np.concatenate([
        mfcc_mean,
        mfcc_std,
        delta_mean,
        delta_std
    ]).astype(np.float32)

    # ---------------------------------------------------------
    # NORMALIZE
    # ---------------------------------------------------------

    norm = np.linalg.norm(
        embedding
    )

    if norm <= 1e-8:
        return None

    embedding = embedding / norm

    return embedding


def compute_voice_embedding(audio_b64: str):

    """
    Convert browser audio → PCM → voice embedding.
    """

    pcm, wav_b64 = audio_b64_to_pcm_numpy(
        audio_b64
    )

    if pcm is None:
        return None, None

    # Check minimum useful duration
    duration = len(pcm) / 16000

    if duration < 1.0:
        print(
            f"Voice too short: {duration:.2f}s"
        )
        return None, None

    embedding = extract_mfcc_embedding(
        pcm,
        sr=16000
    )

    if embedding is None:
        print("Could not create voice embedding")
        return None, None

    print(
        f"Voice embedding created: "
        f"{len(embedding)} dimensions"
    )

    return embedding, wav_b64


def cosine_sim(a, b):

    a = np.asarray(
        a,
        dtype=np.float32
    )

    b = np.asarray(
        b,
        dtype=np.float32
    )

    # Safety check
    if len(a) != len(b):
        print(
            f"Embedding size mismatch: "
            f"{len(a)} vs {len(b)}"
        )
        return 0.0

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm <= 1e-8 or b_norm <= 1e-8:
        return 0.0

    a = a / a_norm
    b = b / b_norm

    return float(
        np.dot(a, b)
    )


# ------------------------------------------------------------
# VOICE VERIFICATION THRESHOLDS
# ------------------------------------------------------------

VOICE_LOGIN_THRESHOLD = 0.90
VOICE_MONITOR_THRESHOLD = 0.85


def verify_voice_embedding(
    registered_list,
    audio_b64: str
):

    """
    Compare student's current voice
    against registered voice.
    """

    if not registered_list:
        return (
            False,
            "No registered voice found",
            0.0
        )

    emb, _ = compute_voice_embedding(
        audio_b64
    )

    if emb is None:
        return (
            False,
            "Voice sample too short or unclear",
            0.0
        )

    try:

        registered = np.asarray(
            registered_list,
            dtype=np.float32
        )

        # Check embedding size
        if registered.shape != emb.shape:

            print(
                "Registered embedding shape:",
                registered.shape
            )

            print(
                "Current embedding shape:",
                emb.shape
            )

            return (
                False,
                "Voice registration is outdated. Please register your voice again.",
                0.0
            )

        similarity = cosine_sim(
            registered,
            emb
        )

        print(
            f"VOICE SIMILARITY = {similarity:.4f}"
        )

        print(
            f"VOICE THRESHOLD = {VOICE_LOGIN_THRESHOLD:.2f}"
        )

        if similarity >= VOICE_LOGIN_THRESHOLD:

            return (
                True,
                f"Voice verified (similarity={similarity:.2f})",
                similarity
            )

        return (
            False,
            f"Voice does not match "
            f"(similarity={similarity:.2f})",
            similarity
        )

    except Exception as e:

        print(
            "Voice verification error:",
            e
        )

        return (
            False,
            "Voice verification failed",
            0.0
        )

def compute_voice_embedding(audio_b64: str):
    """Returns (embedding_array_or_None, clean_wav_b64_or_None)"""
    pcm, wav_b64 = audio_b64_to_pcm_numpy(audio_b64)
    if pcm is None: return None, None
    emb = extract_mfcc_embedding(pcm)
    return emb, wav_b64

def cosine_sim(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0

# Login threshold: strict. Monitor threshold: slightly looser.
# Raised thresholds for better speaker discrimination
VOICE_LOGIN_THRESHOLD   = 0.82   # must be clearly the same speaker
VOICE_MONITOR_THRESHOLD = 0.75   # during exam — slightly looser but still strict

def verify_voice_embedding(registered_list, audio_b64: str):
    emb, _ = compute_voice_embedding(audio_b64)
    if emb is None: return False, "Voice sample too short or silent", 0.0
    sim = cosine_sim(registered_list, emb.tolist())
    print(f"Voice cosine sim: {sim:.3f}")
    if sim >= VOICE_LOGIN_THRESHOLD:
        return True, "Voice verified", sim
    return False, f"Voice does not match (similarity={sim:.2f}, need {VOICE_LOGIN_THRESHOLD})", sim

# ─── PDF GENERATION ───────────────────────────────────────
def generate_exam_pdf(student_reg, student_name, exam_name, questions_answers):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    purple = colors.HexColor('#D082D9')
    story = [
        Paragraph('SmartScribe Exam Report',
            ParagraphStyle('T', parent=styles['Title'], fontSize=18, textColor=purple, spaceAfter=6, alignment=TA_CENTER)),
        HRFlowable(width='100%', thickness=2, color=purple, spaceAfter=10),
        Paragraph(f'<b>Student:</b> {student_name} ({student_reg})',
            ParagraphStyle('I', parent=styles['Normal'], fontSize=11, spaceAfter=4)),
        Paragraph(f'<b>Exam:</b> {exam_name}',
            ParagraphStyle('I2', parent=styles['Normal'], fontSize=11, spaceAfter=4)),
        Paragraph(f'<b>Date:</b> {datetime.now().strftime("%Y-%m-%d %H:%M")}',
            ParagraphStyle('I3', parent=styles['Normal'], fontSize=11, spaceAfter=4)),
        Spacer(1, 0.4*cm),
        HRFlowable(width='100%', thickness=1, color=colors.lightgrey, spaceAfter=10),
    ]
    qs = ParagraphStyle('Q', parent=styles['Normal'], fontSize=12, textColor=purple,
        spaceBefore=12, spaceAfter=4, fontName='Helvetica-Bold')
    as_ = ParagraphStyle('A', parent=styles['Normal'], fontSize=11, spaceAfter=6, leftIndent=10)
    def esc(t): return str(t).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    for i, qa in enumerate(questions_answers, 1):
        story.append(Paragraph(f'Q{i}: {esc(qa.get("question",""))}', qs))
        story.append(Paragraph(f'Ans: {esc(qa.get("answer","") or "[No answer]")}', as_))
    doc.build(story)
    return buffer.getvalue()

# ─── PDF / QUESTION HELPERS ────────────────────────────────
def extract_text_from_pdf(pdf_base64):
    try:
        data = base64.b64decode(pdf_base64.split(',')[1])
        r = PyPDF2.PdfReader(io.BytesIO(data))
        return ''.join(p.extract_text() or '' for p in r.pages)
    except Exception as e:
        print(f"PDF Error: {e}"); return ''

def parse_questions_from_text(text):
    questions = []
    lines = [l.strip() for l in text.splitlines()]

    # Matches:
    # 1 what is photosynthesis
    # 2 what is chlorophyll
    # 3. what is bacteria
    # 4) list diseases
    question_pattern = re.compile(
        r'^\s*(\d+)\s*(?:[.)])?\s+(.+)$'
    )

    # Matches options such as:
    # a) Oxygen
    # b) Carbon dioxide
    option_pattern = re.compile(
        r'^\s*[a-dA-D]\s*[.)]\s*(.+)$'
    )

    current = ''

    for line in lines:

        if not line:
            continue

        # Check for a numbered question
        q_match = question_pattern.match(line)

        if q_match:
            if current:
                questions.append(current.strip())

            current = q_match.group(1) + '. ' + q_match.group(2).strip()
            continue

        # Check for an option
        opt_match = option_pattern.match(line)

        if opt_match:
            if current:
                current += ' ' + line.strip()
            continue

        # Continuation of current question
        if current:
            current += ' ' + line.strip()

    # Save last question
    if current:
        questions.append(current.strip())

    # Remove duplicates
    unique = []
    seen = set()

    for q in questions:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    print("===================================")
    print("EXTRACTED QUESTIONS:", len(unique))

    for i, q in enumerate(unique, 1):
        print(f"{i}. {q}")

    print("===================================")

    return unique
def format_question_for_tts(q):
    pat = re.compile(r'(?:^|\s)([a-dA-D])\)\s*', re.IGNORECASE)
    matches = list(pat.finditer(q))
    if not matches: return q
    stem = q[:matches[0].start()].strip()
    result = stem
    for idx, m in enumerate(matches):
        letter = m.group(1).lower()
        start = m.end()
        end = matches[idx+1].start() if idx+1 < len(matches) else len(q)
        opt = q[start:end].strip().rstrip('.')
        result += f'. {letter}, {opt}'
    return result


# ==================== ORIGINAL NOTEBOOK CELL 4 ====================
# ==================== CELL 4: HOME, ADMIN LOGIN/REGISTER ====================

COMMON_CSS = """
* { margin:0; padding:0; box-sizing:border-box; font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif; }
body { background:white; min-height:100vh; display:flex; justify-content:center; align-items:flex-start; padding:20px; }
.container { width:100%; max-width:1200px; background:white; border-radius:20px; box-shadow:0 20px 60px rgba(0,0,0,0.1); overflow:hidden; border:1px solid #f0f0f0; margin:20px auto; }
.header { background:#D082D9; padding:20px 30px; display:flex; justify-content:space-between; align-items:center; color:white; }
.header h1 { font-size:24px; }
.header a { color:white; text-decoration:none; padding:8px 15px; background:rgba(255,255,255,0.2); border-radius:5px; }
.content { padding:30px; background:white; }
.form-container { max-width:500px; margin:0 auto; background:white; padding:30px; border-radius:15px; box-shadow:0 10px 30px rgba(0,0,0,0.1); border:1px solid #f0f0f0; }
.form-group { margin-bottom:20px; }
.form-group label { display:block; margin-bottom:8px; color:#333; font-weight:500; }
.form-group input,.form-group select { width:100%; padding:12px 15px; border:2px solid #e0e0e0; border-radius:8px; font-size:14px; }
.form-group input:focus,.form-group select:focus { border-color:#D082D9; outline:none; }
.btn { background:#D082D9; color:white; border:none; padding:12px 30px; border-radius:8px; font-size:16px; font-weight:600; cursor:pointer; width:100%; margin:5px 0; transition:all 0.2s; }
.btn:hover { background:#b86fc1; transform:translateY(-2px); }
.btn-secondary { background:white; color:#D082D9; border:2px solid #D082D9; }
.btn-secondary:hover { background:#f8f0fa; }
a { color:#D082D9; text-decoration:none; }
a:hover { text-decoration:underline; }
.text-center { text-align:center; margin-top:15px; }
"""

@app.route('/')
def home():
    return render_template_string("""
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
    <meta name="viewport" content="width=device-width,initial-scale=1.0">
    <title>SmartScribe</title>
    <style>""" + COMMON_CSS + """
        .role-selection{display:flex;justify-content:center;gap:40px;margin:50px 0;flex-wrap:wrap;}
        .role-card{background:white;border-radius:20px;padding:40px;text-align:center;width:300px;cursor:pointer;transition:all 0.3s;box-shadow:0 10px 30px rgba(0,0,0,0.1);border:2px solid #f0f0f0;}
        .role-card:hover{transform:translateY(-10px);border-color:#D082D9;}
        .role-icon{font-size:80px;margin-bottom:20px;}
        .role-card h3{font-size:28px;color:#333;margin-bottom:10px;}
        .role-card p{color:#666;font-size:16px;}
    </style></head><body>
    <div class="container">
        <div class="header"><h1>SmartScribe <span style="font-size:14px;opacity:.9;margin-left:10px;">AI Exam Assistant for Visually Impaired</span></h1></div>
        <div class="content">
            <div class="role-selection">
                <div class="role-card" onclick="window.location.href='/admin-login'">
                    <div class="role-icon">👨‍💼</div><h3>Admin</h3>
                    <p>Manage students, exams, submissions</p>
                </div>
                <div class="role-card" onclick="window.location.href='/student-verification'">
                    <div class="role-icon">👩‍🎓</div><h3>Student</h3>
                    <p>Take exams with voice assistance</p>
                </div>
            </div>
        </div>
    </div>
    </body></html>
    """)

@app.route('/admin-login')
def admin_login():
    return render_template_string("""
    <!DOCTYPE html><html><head><title>Admin Login</title><style>""" + COMMON_CSS + """</style></head><body>
    <div class="container">
        <div class="header"><h1>SmartScribe</h1><a href="/">← Back</a></div>
        <div class="content"><div class="form-container">
            <h2 style="color:#D082D9;text-align:center;margin-bottom:30px;">Admin Login</h2>
            <div class="form-group"><label>Username</label><input type="text" id="u" placeholder="Username"></div>
            <div class="form-group"><label>Password</label><input type="password" id="p" placeholder="Password"></div>
            <button class="btn" onclick="login()">Login</button>
            <p class="text-center">No account? <a href="/admin-register">Register</a></p>
        </div></div>
    </div>
    <script>
        function login(){
            fetch('/api/admin/login',{method:'POST',headers:{'Content-Type':'application/json'},
                body:JSON.stringify({username:document.getElementById('u').value,password:document.getElementById('p').value})})
            .then(r=>r.json()).then(d=>{if(d.success)window.location.href='/admin-dashboard';else alert('Invalid credentials');});
        }
        document.addEventListener('keydown',e=>{if(e.key==='Enter')login();});
    </script></body></html>
    """)

@app.route('/admin-register')
def admin_register():
    return render_template_string("""
    <!DOCTYPE html><html><head><title>Register</title><style>""" + COMMON_CSS + """</style></head><body>
    <div class="container">
        <div class="header"><h1>SmartScribe</h1><a href="/">← Back</a></div>
        <div class="content"><div class="form-container">
            <h2 style="color:#D082D9;text-align:center;margin-bottom:30px;">Admin Registration</h2>
            <div class="form-group"><label>Username</label><input type="text" id="u" placeholder="Username"></div>
            <div class="form-group"><label>Password</label><input type="password" id="p" placeholder="Password"></div>
            <div class="form-group"><label>Confirm Password</label><input type="password" id="c" placeholder="Confirm"></div>
            <button class="btn" onclick="reg()">Register</button>
            <p class="text-center">Have account? <a href="/admin-login">Login</a></p>
        </div></div>
    </div>
    <script>
        function reg(){
            const u=document.getElementById('u').value,p=document.getElementById('p').value,c=document.getElementById('c').value;
            if(p!==c){alert('Passwords do not match');return;}
            fetch('/api/admin/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:u,password:p})})
            .then(r=>r.json()).then(d=>{if(d.success){alert('Registered! Please login.');window.location.href='/admin-login';}else alert(d.message);});
        }
    </script></body></html>
    """)


# ==================== ORIGINAL NOTEBOOK CELL 5 ====================
# ==================== CELL 5: ADMIN DASHBOARD + SUBMISSIONS ====================

@app.route('/admin-dashboard')
def admin_dashboard():
    return render_template_string("""
    <!DOCTYPE html><html><head><title>Admin Dashboard</title>
    <style>""" + COMMON_CSS + """
        .dashboard-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:20px;margin-bottom:30px;}
        .stat-card{background:linear-gradient(135deg,#D082D9,#b86fc1);color:white;padding:25px;border-radius:15px;text-align:center;}
        .stat-number{font-size:36px;font-weight:700;}
        .action-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:20px;margin-top:30px;}
        .action-card{background:#f8f9fa;padding:25px;border-radius:12px;text-align:center;cursor:pointer;border:2px solid transparent;transition:all 0.2s;}
        .action-card:hover{border-color:#D082D9;background:#fdf6fe;transform:translateY(-3px);}
        .action-icon{font-size:36px;}
        .action-card h3{font-size:16px;color:#444;margin-top:8px;}
    </style></head><body>
    <div class="container">
        <div class="header"><h1>SmartScribe — Admin Dashboard</h1><a href="/">Logout</a></div>
        <div class="content">
            <div class="dashboard-cards">
                <div class="stat-card"><div class="stat-number" id="ts">0</div><div>Total Students</div></div>
                <div class="stat-card"><div class="stat-number" id="te">0</div><div>Total Exams</div></div>
                <div class="stat-card"><div class="stat-number" id="tsub">0</div><div>Submissions</div></div>
            </div>
            <div class="action-grid">
                <div class="action-card" onclick="window.location.href='/create-student'"><div class="action-icon">➕</div><h3>Add Student</h3></div>
                <div class="action-card" onclick="window.location.href='/view-students'"><div class="action-icon">👥</div><h3>View Students</h3></div>
                <div class="action-card" onclick="window.location.href='/create-exam'"><div class="action-icon">📝</div><h3>Add Exam</h3></div>
                <div class="action-card" onclick="window.location.href='/view-exams'"><div class="action-icon">📚</div><h3>View Exams</h3></div>
                <div class="action-card" onclick="window.location.href='/view-submissions'" style="border-color:#D082D9;background:#fdf6fe;">
                    <div class="action-icon">📥</div><h3>View Submissions</h3>
                </div>
            </div>
        </div>
    </div>
    <script>
        fetch('/api/admin/stats').then(r=>r.json()).then(d=>{
            document.getElementById('ts').textContent=d.totalStudents;
            document.getElementById('te').textContent=d.totalExams;
            document.getElementById('tsub').textContent=d.totalSubmissions;
        });
    </script></body></html>
    """)

@app.route('/view-submissions')
def view_submissions():
    with get_db() as conn:
        subs = conn.cursor().execute(
            "SELECT id,reg_no,student_name,exam_name,pdf_filename,submitted_at FROM exam_submissions ORDER BY submitted_at DESC"
        ).fetchall()
    rows = ''
    for s in subs:
        fname = s['pdf_filename'] or f"{s['student_name']}.{s['exam_name']}.pdf"
        rows += f"""<tr><td>{s['reg_no']}</td><td>{s['student_name']}</td><td>{s['exam_name']}</td>
        <td>{s['submitted_at']}</td>
        <td><a href="/api/admin/download-submission/{s['id']}" style="background:#D082D9;color:white;padding:5px 12px;border-radius:5px;text-decoration:none;font-size:13px;">⬇ Download</a></td></tr>"""
    if not rows:
        rows = '<tr><td colspan="5" style="text-align:center;padding:40px;color:#888;">No submissions yet.</td></tr>'
    return render_template_string(f"""
    <!DOCTYPE html><html><head><title>Submissions</title><style>
        {COMMON_CSS}
        table{{width:100%;border-collapse:collapse;margin-top:20px;}}
        th{{background:#D082D9;color:white;padding:12px 15px;text-align:left;}}
        td{{padding:11px 15px;border-bottom:1px solid #f0f0f0;}}
        tr:hover td{{background:#fdf6fe;}}
    </style></head><body>
    <div class="container">
        <div class="header"><h1>SmartScribe — Submissions</h1><a href="/admin-dashboard">← Back</a></div>
        <div class="content">
            <h2 style="color:#D082D9;margin-bottom:10px;">📥 Student Submitted Answer Sheets</h2>
            <table><thead><tr><th>Reg No</th><th>Student</th><th>Exam</th><th>Submitted At</th><th>PDF</th></tr></thead>
            <tbody>{rows}</tbody></table>
        </div>
    </div></body></html>
    """)


# ==================== ORIGINAL NOTEBOOK CELL 6 ====================
# ==================== CELL 6: CREATE/VIEW STUDENTS & EXAMS ====================

@app.route('/create-student')
def create_student():
    return render_template_string("""
    <!DOCTYPE html><html><head><title>Create Student</title>
    <style>""" + COMMON_CSS + """
        .preview-image{max-width:180px;max-height:180px;display:none;margin:10px auto;border-radius:10px;border:3px solid #D082D9;}
        .warning{color:#ff4757;font-size:13px;margin-top:4px;}
        .voice-section{background:#f0f7ff;border:2px dashed #80b3ff;border-radius:12px;padding:20px;margin-top:15px;text-align:center;}
        .rec-btn{background:#e74c3c;color:white;border:none;padding:10px 22px;border-radius:8px;font-size:14px;cursor:pointer;margin:4px;}
        .stop-btn{background:#27ae60;color:white;border:none;padding:10px 22px;border-radius:8px;font-size:14px;cursor:pointer;margin:4px;display:none;}
        .voice-status{font-size:13px;color:#555;margin-top:8px;min-height:20px;}
        .step-indicator{display:inline-block;background:#D082D9;color:white;border-radius:50%;width:26px;height:26px;line-height:26px;font-size:13px;font-weight:700;margin-right:6px;}
    </style></head><body>
    <div class="container">
        <div class="header"><h1>SmartScribe — Add Student</h1><a href="/admin-dashboard">← Back</a></div>
        <div class="content">
        <div style="max-width:640px;margin:0 auto;background:white;padding:30px;border-radius:15px;box-shadow:0 10px 30px rgba(0,0,0,.08);border:1px solid #f0f0f0;">
            <h2 style="color:#D082D9;text-align:center;margin-bottom:25px;">Register New Student</h2>

            <!-- Basic info -->
            <div class="form-group"><label>Registration Number</label><input type="text" id="regNo" placeholder="e.g., 2024001"></div>
            <div class="form-group"><label>Full Name</label><input type="text" id="studentName" placeholder="Enter full name"></div>

            <!-- Face photo -->
            <div class="form-group">
                <label><span class="step-indicator">1</span>Student Photo (for face recognition)</label>
                <input type="file" id="studentImage" accept="image/*" onchange="previewImg(event)">
                <img id="imgPreview" class="preview-image">
                <div class="warning">⚠️ Ensure face is clearly visible and well-lit</div>
            </div>

            <!-- Voice sample -->
            <div class="form-group">
                <label><span class="step-indicator">2</span>Voice Sample (for voice authentication)</label>
                <div class="voice-section">
                    <p style="color:#555;margin-bottom:10px;font-size:14px;">
                        Ask the student to speak clearly for <strong>5–8 seconds</strong>.<br>
                        Example: <em>"My name is [name] and my registration number is [reg no]"</em>
                    </p>
                    <button class="rec-btn" id="recBtn" onclick="startRec()">🎙️ Start Recording</button>
                    <button class="stop-btn" id="stopBtn" onclick="stopRec()">⏹ Stop Recording</button>
                    <div class="voice-status" id="voiceStatus">Press Start Recording to begin</div>
                    <audio id="voicePlayback" controls style="display:none;width:100%;margin-top:10px;"></audio>
                    <div id="wavNotice" style="display:none;font-size:12px;color:#27ae60;margin-top:4px;">✅ Voice recorded and will be stored as WAV</div>
                </div>
            </div>

            <button class="btn" onclick="registerStudent()" style="margin-top:10px;">✅ Register Student</button>
            <button class="btn btn-secondary" onclick="window.location.href='/admin-dashboard'">Cancel</button>
        </div>
        </div>
    </div>
    <script>
    let imageData=null, voiceB64=null, mediaRecorder=null, audioChunks=[], autoStopTimer=null;

    function previewImg(event){
        const file=event.target.files[0];
        if(!file) return;
        const reader=new FileReader();
        reader.onload=e=>{
            document.getElementById('imgPreview').src=e.target.result;
            document.getElementById('imgPreview').style.display='block';
            imageData=e.target.result;
        };
        reader.readAsDataURL(file);
    }

    async function startRec(){
        audioChunks=[]; voiceB64=null;
        document.getElementById('voicePlayback').style.display='none';
        document.getElementById('wavNotice').style.display='none';
        let stream;
        try{
            stream=await navigator.mediaDevices.getUserMedia({audio:{sampleRate:16000,channelCount:1,echoCancellation:true}});
        }catch(e){
            alert('Microphone access denied: '+e.message);
            return;
        }
        // Prefer wav, fall back to webm (server handles both)
        const mime=MediaRecorder.isTypeSupported('audio/wav')?'audio/wav':
                   MediaRecorder.isTypeSupported('audio/webm;codecs=opus')?'audio/webm;codecs=opus':'audio/webm';
        mediaRecorder=new MediaRecorder(stream,{mimeType:mime});
        mediaRecorder.ondataavailable=e=>audioChunks.push(e.data);
        mediaRecorder.onstop=()=>{
            const blob=new Blob(audioChunks,{type:mime});
            document.getElementById('voicePlayback').src=URL.createObjectURL(blob);
            document.getElementById('voicePlayback').style.display='block';
            const reader=new FileReader();
            reader.onload=e=>{
                voiceB64=e.target.result.split(',')[1];
                document.getElementById('voiceStatus').textContent='✅ Voice recorded ('+(blob.size/1024).toFixed(1)+' KB). You can play it back above.';
                document.getElementById('wavNotice').style.display='block';
            };
            reader.readAsDataURL(blob);
            stream.getTracks().forEach(t=>t.stop());
        };
        mediaRecorder.start();
        document.getElementById('recBtn').style.display='none';
        document.getElementById('stopBtn').style.display='inline-block';
        document.getElementById('voiceStatus').textContent='🔴 Recording... speak clearly now';
        // Auto-stop after 10 seconds
        autoStopTimer=setTimeout(()=>{ if(mediaRecorder&&mediaRecorder.state==='recording')stopRec(); },10000);
    }

    function stopRec(){
        if(autoStopTimer){clearTimeout(autoStopTimer);autoStopTimer=null;}
        if(mediaRecorder&&mediaRecorder.state==='recording')mediaRecorder.stop();
        document.getElementById('recBtn').style.display='inline-block';
        document.getElementById('stopBtn').style.display='none';
    }

    function registerStudent(){
        const regNo=document.getElementById('regNo').value.trim();
        const name=document.getElementById('studentName').value.trim();
        if(!regNo||!name){alert('Please fill registration number and name');return;}
        if(!imageData){alert('Please upload a student photo');return;}
        if(!voiceB64){alert('Please record a voice sample (Step 2)');return;}
        const btn=event.target;btn.disabled=true;btn.textContent='Registering...';
        fetch('/api/admin/create-student',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({regNo,name,image:imageData,voice:voiceB64})})
        .then(r=>r.json()).then(d=>{
            if(d.success){alert('Student registered successfully!');window.location.href='/view-students';}
            else{alert('Error: '+(d.message||'Unknown error'));btn.disabled=false;btn.textContent='✅ Register Student';}
        }).catch(e=>{alert('Network error: '+e.message);btn.disabled=false;btn.textContent='✅ Register Student';});
    }
    </script></body></html>
    """)

@app.route('/view-students')
def view_students():
    with get_db() as conn:
        students = conn.cursor().execute("SELECT reg_no,name,image,voice_embedding FROM students ORDER BY created_at DESC").fetchall()
    html = ''
    if students:
        for s in students:
            has_voice = '✅ Voice' if s['voice_embedding'] else '⚠️ No voice'
            html += f'''
            <div class="student-card">
                <div class="student-image"><img src="{s['image']}" alt="{s['name']}"></div>
                <div style="text-align:center;">
                    <h3>{s['name']}</h3><p>Reg: {s['reg_no']}</p>
                    <span style="font-size:12px;color:#666;">{has_voice}</span>
                </div>
                <div style="display:flex;gap:8px;margin-top:12px;">
                    <button onclick="delS('{s['reg_no']}')" style="flex:1;padding:8px;background:#ff4757;color:white;border:none;border-radius:5px;cursor:pointer;">Delete</button>
                </div>
            </div>'''
    else:
        html = '<p style="text-align:center;padding:50px;">No students yet. <a href="/create-student">Add one</a></p>'
    return render_template_string(f"""
    <!DOCTYPE html><html><head><title>Students</title><style>
        {COMMON_CSS}
        .students-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:20px;}}
        .student-card{{background:white;border-radius:15px;padding:20px;box-shadow:0 5px 15px rgba(0,0,0,0.1);border:1px solid #f0f0f0;}}
        .student-image{{width:110px;height:110px;border-radius:50%;margin:0 auto 12px;overflow:hidden;border:3px solid #D082D9;}}
        .student-image img{{width:100%;height:100%;object-fit:cover;}}
    </style>
    <script>function delS(r){{if(confirm('Delete student '+r+'?')){{fetch('/api/admin/delete-student',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{regNo:r}})}}).then(r=>r.json()).then(d=>{{if(d.success)location.reload();}});}}}}</script>
    </head><body>
    <div class="container">
        <div class="header"><h1>SmartScribe — Students</h1><a href="/admin-dashboard">← Back</a></div>
        <div class="content"><div class="students-grid">{html}</div></div>
    </div></body></html>
    """)



@app.route('/api/exam/pdf/<exam_id>')
def view_exam_pdf(exam_id):
    try:
        with get_db() as conn:
            exam = conn.cursor().execute(
                "SELECT pdf FROM exams WHERE id=?",
                (exam_id,)
            ).fetchone()

        if not exam:
            return "Exam not found", 404

        pdf_data = exam['pdf']

        if not pdf_data:
            return "Question paper not found", 404

        if ',' in pdf_data:
            pdf_data = pdf_data.split(',', 1)[1]

        pdf_bytes = base64.b64decode(pdf_data)

        response = make_response(pdf_bytes)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'inline'

        return response

    except Exception as e:
        print("PDF VIEW ERROR:", e)
        return f"Error opening PDF: {e}", 500

@app.route('/view-exams')
def view_exams():
    with get_db() as conn:
        exams = conn.cursor().execute("SELECT id,name,duration_minutes,created_at FROM exams ORDER BY created_at DESC").fetchall()
    html = ''
    if exams:
        for e in exams:
            html += f"""<div class="exam-card">
                <h3 style="color:#D082D9;">{e['name']}</h3>
                <p>⏱ Duration: {e['duration_minutes'] or 60} minutes</p>
                <p>📅 Created: {e['created_at']}</p>
                <button onclick="window.open('/api/exam/pdf/{e['id']}','_blank')" style="margin-top:10px;padding:8px 16px;background:#D082D9;color:white;border:none;border-radius:5px;cursor:pointer;">View PDF</button>
            </div>"""
    else:
        html = '<p style="text-align:center;padding:50px;">No exams yet. <a href="/create-exam">Create one</a></p>'
    return render_template_string(f"""
    <!DOCTYPE html><html><head><title>Exams</title><style>
        {COMMON_CSS}
        .exams-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:20px;margin-top:20px;}}
        .exam-card{{background:white;border-radius:10px;padding:20px;border-left:5px solid #D082D9;box-shadow:0 5px 15px rgba(0,0,0,.1);}}
    </style></head><body>
    <div class="container">
        <div class="header"><h1>SmartScribe — Exams</h1><a href="/admin-dashboard">← Back</a></div>
        <div class="content"><div class="exams-grid">{html}</div></div>
    </div></body></html>
    """)


# ==================== ORIGINAL NOTEBOOK CELL 7 ====================
# ==================== CELL 7: STUDENT VERIFICATION (Face + Voice, 2-step) ====================

@app.route('/student-verification')
def student_verification():
    return render_template_string("""
    <!DOCTYPE html><html><head><title>Student Verification</title>
    <style>""" + COMMON_CSS + """
        .box{max-width:540px;margin:0 auto;background:#f8f9fa;border-radius:12px;padding:22px;}
        .cam-prev{width:100%;height:280px;background:#ddd;border-radius:8px;margin:12px 0;overflow:hidden;}
        .cam-prev video,.cam-prev img{width:100%;height:100%;object-fit:cover;}
        .status-box{padding:10px;border-radius:5px;margin:10px 0;display:none;font-weight:500;}
        .ok{background:#d4edda;color:#155724;border:1px solid #c3e6cb;}
        .err{background:#f8d7da;color:#721c24;border:1px solid #f5c6cb;}
        .voice-box{max-width:540px;margin:18px auto;background:#f0f7ff;border:2px dashed #80b3ff;border-radius:12px;padding:20px;text-align:center;display:none;}
        .step-bar{display:flex;justify-content:center;gap:14px;margin:6px 0 18px;}
        .step{padding:5px 14px;border-radius:20px;font-size:13px;font-weight:600;background:#eee;color:#888;}
        .step.active{background:#D082D9;color:white;}
        .step.done{background:#d4edda;color:#155724;}
        .rec-btn{background:#e74c3c;color:white;border:none;padding:10px 20px;border-radius:8px;font-size:14px;cursor:pointer;margin:4px;}
        .stop-btn{background:#27ae60;color:white;border:none;padding:10px 20px;border-radius:8px;font-size:14px;cursor:pointer;margin:4px;display:none;}
    </style></head><body>
    <div class="container">
        <div class="header"><h1>SmartScribe — Verification</h1><a href="/">← Back</a></div>
        <div class="content">
            <div style="max-width:540px;margin:0 auto 18px;">
                <input type="text" id="regNo" placeholder="Enter Registration Number"
                    style="width:100%;padding:14px;border:2px solid #e0e0e0;border-radius:8px;font-size:16px;text-align:center;">
            </div>
            <div class="step-bar">
                <div class="step active" id="step1">1 — Face</div>
                <div class="step" id="step2">2 — Voice</div>
                <div class="step" id="step3">3 — Exam</div>
            </div>
            <div id="statusMsg" class="status-box"></div>

            <!-- FACE -->
            <div class="box" id="faceBox">
                <h3 style="color:#D082D9;text-align:center;margin-bottom:10px;">Step 1: Face Verification</h3>
                <div class="cam-prev">
                    <video id="cam" autoplay style="width:100%;height:100%;"></video>
                    <canvas id="camCanvas" style="display:none;"></canvas>
                </div>
                <div style="text-align:center;">
                    <button class="btn" id="capBtn" onclick="captureImg()" style="width:auto;padding:10px 22px;">📸 Capture Face</button>
                    <button class="btn btn-secondary" id="retakeBtn" onclick="retake()" style="display:none;width:auto;padding:10px 22px;">🔄 Retake</button>
                </div>
                <div id="faceStatus" style="text-align:center;margin-top:8px;color:#666;"></div>
                <button class="btn" id="verifyFaceBtn" onclick="verifyFace()" style="margin-top:14px;">Verify Face →</button>
            </div>

            <!-- VOICE -->
            <div class="voice-box" id="voiceBox">
                <h3 style="color:#4a90e2;margin-bottom:8px;">Step 2: Voice Verification</h3>
                <p style="color:#555;margin-bottom:12px;font-size:14px;">
                    Please say: <strong>"My name is [your name] and I am ready for my exam"</strong>
                </p>
                <button class="rec-btn" id="vRecBtn" onclick="startVoice()">🎙️ Record Voice</button>
                <button class="stop-btn" id="vStopBtn" onclick="stopVoice()">⏹ Stop</button>
                <div id="vStatus" style="font-size:14px;color:#555;margin-top:8px;">Press Record to begin</div>
                <audio id="vPlayback" controls style="display:none;width:100%;margin-top:10px;"></audio>
                <button class="btn" id="verifyVoiceBtn" onclick="verifyVoice()" style="margin-top:14px;display:none;">Verify Voice →</button>
            </div>
        </div>
    </div>
    <script>
    let capturedImg=null,faceCaptured=false,voiceB64=null,vMR=null,vChunks=[],vAutoStop=null;

    navigator.mediaDevices.getUserMedia({video:true})
        .then(s=>{document.getElementById('cam').srcObject=s;})
        .catch(()=>showStatus('Camera access denied','err'));

    function showStatus(msg,type){
        const d=document.getElementById('statusMsg');
        d.textContent=msg; d.className='status-box '+type; d.style.display='block';
        setTimeout(()=>d.style.display='none',6000);
    }
    function setStep(n){
        ['step1','step2','step3'].forEach((id,i)=>{
            document.getElementById(id).className='step'+(i+1<n?' done':i+1===n?' active':'');
        });
    }
    function captureImg(){
        const v=document.getElementById('cam'),c=document.getElementById('camCanvas');
        c.width=v.videoWidth;c.height=v.videoHeight;
        c.getContext('2d').drawImage(v,0,0);
        capturedImg=c.toDataURL('image/jpeg');
        const img=document.createElement('img');img.src=capturedImg;img.style='width:100%;height:100%;object-fit:cover;';
        v.style.display='none';v.parentNode.appendChild(img);
        document.getElementById('capBtn').style.display='none';
        document.getElementById('retakeBtn').style.display='inline-block';
        document.getElementById('faceStatus').textContent='✅ Face captured!';
        faceCaptured=true;
    }
    function retake(){
        capturedImg=null;faceCaptured=false;
        const v=document.getElementById('cam');v.style.display='block';
        const img=v.parentNode.querySelector('img');if(img)img.remove();
        document.getElementById('capBtn').style.display='inline-block';
        document.getElementById('retakeBtn').style.display='none';
        document.getElementById('faceStatus').textContent='';
    }
    function verifyFace(){
        const regNo=document.getElementById('regNo').value.trim();
        if(!regNo){showStatus('Enter registration number','err');return;}
        if(!faceCaptured){showStatus('Capture your face first','err');return;}
        const btn=document.getElementById('verifyFaceBtn');btn.disabled=true;btn.textContent='Verifying...';
        fetch('/api/student/verify-face',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({regNo,image:capturedImg})})
        .then(r=>r.json()).then(d=>{
            if(d.success){
                showStatus('✅ Face verified! Now verify your voice.','ok');
                document.getElementById('faceBox').style.opacity='0.55';
                document.getElementById('voiceBox').style.display='block';
                setStep(2);
            }else{showStatus('❌ '+d.message,'err');btn.disabled=false;btn.textContent='Verify Face →';}
        }).catch(()=>{showStatus('Error. Try again.','err');btn.disabled=false;btn.textContent='Verify Face →';});
    }

    async function startVoice(){
        vChunks=[];voiceB64=null;
        document.getElementById('vPlayback').style.display='none';
        let stream;
        try{stream=await navigator.mediaDevices.getUserMedia({audio:{sampleRate:16000,channelCount:1,echoCancellation:true}});}
        catch(e){showStatus('Microphone denied: '+e.message,'err');return;}
        const mime=MediaRecorder.isTypeSupported('audio/wav')?'audio/wav':
                   MediaRecorder.isTypeSupported('audio/webm;codecs=opus')?'audio/webm;codecs=opus':'audio/webm';
        vMR=new MediaRecorder(stream,{mimeType:mime});
        vMR.ondataavailable=e=>vChunks.push(e.data);
        vMR.onstop=()=>{
            const blob=new Blob(vChunks,{type:mime});
            document.getElementById('vPlayback').src=URL.createObjectURL(blob);
            document.getElementById('vPlayback').style.display='block';
            const reader=new FileReader();
            reader.onload=e=>{
                voiceB64=e.target.result.split(',')[1];
                document.getElementById('vStatus').textContent='✅ Recording done. Press Verify Voice.';
                document.getElementById('verifyVoiceBtn').style.display='block';
            };
            reader.readAsDataURL(blob);
            stream.getTracks().forEach(t=>t.stop());
        };
        vMR.start();
        document.getElementById('vRecBtn').style.display='none';
        document.getElementById('vStopBtn').style.display='inline-block';
        document.getElementById('vStatus').textContent='🔴 Recording... speak clearly';
        vAutoStop=setTimeout(()=>{if(vMR&&vMR.state==='recording')stopVoice();},8000);
    }
    function stopVoice(){
        if(vAutoStop){clearTimeout(vAutoStop);vAutoStop=null;}
        if(vMR&&vMR.state==='recording')vMR.stop();
        document.getElementById('vRecBtn').style.display='inline-block';
        document.getElementById('vStopBtn').style.display='none';
    }
    function verifyVoice(){
        const regNo=document.getElementById('regNo').value.trim();
        if(!voiceB64){showStatus('Record voice first','err');return;}
        const btn=document.getElementById('verifyVoiceBtn');btn.disabled=true;btn.textContent='Verifying voice...';
        fetch('/api/student/verify-voice',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({regNo,voice:voiceB64})})
        .then(r=>r.json()).then(d=>{
            if(d.success){
                showStatus('✅ Voice verified! Entering exam...','ok');
                setStep(3);
                setTimeout(()=>window.location.href='/exam-page?regNo='+regNo,1200);
            }else{showStatus('❌ '+d.message+' — please try again','err');btn.disabled=false;btn.textContent='Verify Voice →';}
        }).catch(()=>{showStatus('Error. Try again.','err');btn.disabled=false;btn.textContent='Verify Voice →';});
    }
    </script></body></html>
    """)


# ==================== ORIGINAL NOTEBOOK CELL 8 ====================
# ==================== CELL 8: EXAM PAGE ====================
# KEY FIXES:
# 1. FAST response: commands detected on INTERIM results, not just final
# 2. Silence timer reduced to 2.5s for snappier auto-capture
# 3. Voice guard runs SEPARATELY from speech recognition (no mic conflict)
#    - uses a dedicated MediaRecorder stream that does NOT conflict with SR
#    - guard runs every 15s (not 8s) to reduce false positives
# 4. "next" / "submit" / "repeat" now respond IMMEDIATELY
# 5. Recognition auto-restarts instantly on end

@app.route('/exam-page')
def exam_page():
    regNo = request.args.get('regNo', '')
    return render_template_string("""
    <!DOCTYPE html><html><head>
    <title>Exam — SmartScribe</title>
    <style>
        *{box-sizing:border-box;margin:0;padding:0;}
        body{background:white;font-family:'Segoe UI',sans-serif;min-height:100vh;display:flex;justify-content:center;align-items:flex-start;padding:15px;}
        .container{width:100%;max-width:860px;background:white;border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,.1);overflow:hidden;border:1px solid #f0f0f0;margin:8px auto;}
        .header{background:#D082D9;padding:14px 22px;color:white;display:flex;justify-content:space-between;align-items:center;}
        .header h1{font-size:20px;}
        .timer-box{background:white;color:#D082D9;padding:5px 14px;border-radius:20px;font-size:19px;font-weight:700;min-width:80px;text-align:center;}
        .timer-box.warn{color:#ff4757;animation:blink .8s infinite;}
        @keyframes blink{0%,100%{opacity:1;}50%{opacity:.4;}}
        .logout-btn{background:rgba(255,255,255,.22);color:white;border:2px solid rgba(255,255,255,.55);padding:6px 14px;border-radius:8px;cursor:pointer;font-size:13px;font-weight:600;}
        .logout-btn:hover{background:rgba(255,255,255,.35);}
        .content{padding:22px;}
        /* Status bubble — BIG and clear for visually impaired */
        .status-bubble{background:#f8f0fa;border:2px solid #D082D9;border-radius:16px;padding:18px 22px;font-size:20px;color:#222;line-height:1.65;min-height:68px;margin-bottom:16px;text-align:center;}
        .mic-area{display:flex;flex-direction:column;align-items:center;gap:9px;margin-bottom:16px;}
        .mic-icon{font-size:54px;}
        .mic-icon.listening{animation:pulse 1s infinite;}
        @keyframes pulse{0%,100%{transform:scale(1);}50%{transform:scale(1.18);}}
        .mic-label{font-size:15px;color:#888;font-style:italic;}
        .mic-label.active{color:#D082D9;font-weight:700;}
        .wave-anim{display:none;justify-content:center;gap:5px;height:38px;align-items:center;}
        .wave-anim.show{display:flex;}
        .wbar{width:7px;background:#D082D9;border-radius:4px;animation:wave 1s ease-in-out infinite;}
        .wbar:nth-child(1){height:14px;animation-delay:0s;}.wbar:nth-child(2){height:28px;animation-delay:.15s;}
        .wbar:nth-child(3){height:42px;animation-delay:.3s;}.wbar:nth-child(4){height:28px;animation-delay:.45s;}
        .wbar:nth-child(5){height:14px;animation-delay:.6s;}
        @keyframes wave{0%,100%{transform:scaleY(.4);}50%{transform:scaleY(1);}}
        .transcript-box{background:#f8f9fa;border:2px dashed #D082D9;border-radius:12px;padding:14px;font-size:17px;color:#333;min-height:52px;display:none;margin-bottom:13px;}
        .transcript-box.show{display:block;}
        .progress-wrap{background:#f0f0f0;border-radius:10px;height:9px;margin-bottom:7px;display:none;}
        .progress-wrap.show{display:block;}
        .progress-fill{background:#D082D9;height:100%;border-radius:10px;transition:width .4s;}
        .progress-text{text-align:center;color:#666;font-size:14px;margin-bottom:13px;display:none;}
        .progress-text.show{display:block;}
        .hint-box{background:#e8f5e9;border-radius:10px;padding:11px 16px;font-size:13px;color:#2e7d32;display:none;margin-top:13px;}
        .hint-box.show{display:block;}
        .hint-box strong{display:block;margin-bottom:5px;}
        .hint-box span{display:inline-block;background:white;padding:2px 9px;border-radius:20px;margin:2px;border:1px solid #81c784;}
        .submit-area{display:none;text-align:center;margin-top:20px;}
        .submit-area.show{display:block;}
        .submit-btn{background:#D082D9;color:white;border:none;padding:15px 48px;border-radius:50px;font-size:17px;font-weight:700;cursor:pointer;box-shadow:0 8px 20px rgba(208,130,217,.4);transition:all .2s;}
        .submit-btn:hover{background:#b86fc1;transform:translateY(-2px);}
        .submit-btn:disabled{background:#ccc;cursor:not-allowed;transform:none;box-shadow:none;}
        .voice-warn{display:none;background:#fff3cd;border:2px solid #ffc107;color:#856404;border-radius:10px;padding:10px 16px;font-size:14px;font-weight:600;margin-bottom:11px;text-align:center;}
        .voice-warn.show{display:block;}
    </style>
    </head><body>
    <div class="container">
        <div class="header">
            <h1>🎤 SmartScribe Exam</h1>
            <div style="display:flex;align-items:center;gap:11px;">
                <div class="timer-box" id="timerBox">--:--</div>
                <button class="logout-btn" onclick="confirmLogout()">🚪 Logout</button>
            </div>
        </div>
        <div class="content">
            <div class="voice-warn" id="voiceWarn">⚠️ Warning: Unrecognised voice detected — please speak yourself</div>
            <div class="status-bubble" id="statusBubble">Initializing... please wait.</div>
            <div class="progress-text" id="progressText">Question 1 of 0</div>
            <div class="progress-wrap" id="progressWrap"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
            <div class="mic-area">
                <div class="mic-icon" id="micIcon">🎙️</div>
                <div class="mic-label" id="micLabel">Waiting...</div>
                <div class="wave-anim" id="waveAnim">
                    <div class="wbar"></div><div class="wbar"></div><div class="wbar"></div>
                    <div class="wbar"></div><div class="wbar"></div>
                </div>
            </div>
            <div class="transcript-box" id="transcriptBox">
                <div style="font-size:11px;color:#D082D9;font-weight:600;margin-bottom:5px;">🗣️ I heard:</div>
                <div id="transcriptText"></div>
            </div>
            <div class="hint-box" id="hintBox">
                <strong>Voice Commands:</strong>
                <span>"next" → next question</span>
                <span>"repeat" → hear again</span>
                <span>"submit" → finish exam</span>
            </div>
            <div class="submit-area" id="submitArea">
                <p style="color:#666;margin-bottom:11px;">Or click to submit:</p>
                <button class="submit-btn" id="submitBtn" onclick="manualSubmit()">📄 Submit Exam</button>
            </div>
        </div>
    </div>

    <script>
    // ════════════════════════════════════════════════════════
    //  STATE
    // ════════════════════════════════════════════════════════
    const urlP = new URLSearchParams(window.location.search);
    const regNo = urlP.get('regNo') || '""" + regNo + """';
    if(!regNo){alert('No reg number. Please verify first.');window.location.href='/student-verification';}

    let questions=[], answers=[], currentIndex=0, examId=null, examName='', studentName='';
    let examDurSec=3600, timerLeft=0, timerIntvl=null;

    const S = {IDLE:'idle',WAIT:'waiting_start',SEL:'selecting_exam',
               READQ:'reading_q',LISTENS:'listening_ans',CONFIRM:'confirming_ans',
               SUBMIT_Q:'asking_submit',SUBMITTING:'submitting'};
    let state = S.IDLE;

    let recognition=null, finalT='', interimT='', isSpeaking=false, silTimer=null;let recognitionRunning = false;
let recognitionStarting = false;
let answerBuffer = '';
let restartTimer = null;
    let availExams=[];
     // need 3 consecutive fails before terminating

    // ════════════════════════════════════════════════════════
    //  TIMER
    // ════════════════════════════════════════════════════════
    function startTimer(s){
        timerLeft=s;
        timerIntvl=setInterval(()=>{
            timerLeft--; updateTimerUI();
            if(timerLeft<=300) document.getElementById('timerBox').classList.add('warn');
            if(timerLeft>0 && timerLeft<=60 && timerLeft%30===0) speak(timerLeft+' seconds remaining.');
            if(timerLeft<=0){clearInterval(timerIntvl); autoSubmit();}
        },1000);
    }
    function updateTimerUI(){
        const m=Math.floor(timerLeft/60), s=timerLeft%60;
        document.getElementById('timerBox').textContent=String(m).padStart(2,'0')+':'+String(s).padStart(2,'0');
    }
    function autoSubmit(){
        stopListening();
        const cur=(finalT+interimT).trim(); if(cur) saveAnswer(cur);
        setStatus('⏰ Time is up! Submitting...');
        speak('Time is up. Submitting your exam now.',()=>doSubmit());
    }

    // ════════════════════════════════════════════════════════
    //  UI HELPERS
    // ════════════════════════════════════════════════════════
    function setStatus(msg){ document.getElementById('statusBubble').textContent=msg; }
    function setMic(st){
        const ic=document.getElementById('micIcon'), lb=document.getElementById('micLabel'), wv=document.getElementById('waveAnim');
        if(st==='listening'){ic.textContent='🎙️';ic.className='mic-icon listening';lb.textContent='🔴 Listening... speak now';lb.className='mic-label active';wv.classList.add('show');}
        else if(st==='speaking'){ic.textContent='🔊';ic.className='mic-icon';lb.textContent='Speaking...';lb.className='mic-label';wv.classList.remove('show');}
        else{ic.textContent='🎙️';ic.className='mic-icon';lb.textContent='Waiting...';lb.className='mic-label';wv.classList.remove('show');}
    }
    function showT(t){document.getElementById('transcriptText').textContent=t;document.getElementById('transcriptBox').classList.add('show');}
    function hideT(){document.getElementById('transcriptBox').classList.remove('show');}
    function showProgress(){
        ['progressText','progressWrap','hintBox','submitArea'].forEach(id=>document.getElementById(id).classList.add('show'));
    }
    function updProgress(){
        const pct=((currentIndex+1)/questions.length)*100;
        document.getElementById('progressFill').style.width=pct+'%';
        document.getElementById('progressText').textContent='Question '+(currentIndex+1)+' of '+questions.length;
    }

    // ════════════════════════════════════════════════════════
    //  TTS  — fast, no artificial delay
    // ════════════════════════════════════════════════════════
    function speak(text, cb){
        isSpeaking=true; setMic('speaking');
        window.speechSynthesis.cancel();
        const u=new SpeechSynthesisUtterance(text);
        u.rate=1.05; u.pitch=1;   // slightly faster for visually impaired students
        u.onend=()=>{isSpeaking=false; if(cb)cb();};
        u.onerror=()=>{isSpeaking=false; if(cb)cb();};
        window.speechSynthesis.speak(u);
    }

    // ════════════════════════════════════════════════════════
    //  SPEECH RECOGNITION — fast command detection
    // ════════════════════════════════════════════════════════
    
    function initRec(){

    if(!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)){
        setStatus('❌ Speech recognition is not supported. Please use Google Chrome.');
        return false;
    }

    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

    recognition = new SR();

    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    recognition.onstart = function(){

        recognitionRunning = true;
        recognitionStarting = false;

        console.log('🎤 Microphone listening started');

        setMic('listening');
    };


    recognition.onresult = function(ev){

        finalT = '';
        interimT = '';

        for(let i = ev.resultIndex; i < ev.results.length; i++){

            const t = ev.results[i][0].transcript;

            if(ev.results[i].isFinal){

                finalT += t;

            }else{

                interimT += t;

            }
        }

        const currentSpeech = (finalT + interimT).trim();
        const lower = currentSpeech.toLowerCase();

        if(currentSpeech){

            showT(currentSpeech);

            console.log('🎤 Heard:', currentSpeech);
        }


        // ═══════════════════════════════
        // READY / YES
        // ═══════════════════════════════

        if(state === S.WAIT){

            if(
                lower.includes('yes') ||
                lower.includes('yeah') ||
                lower.includes('yep') ||
                lower.includes('ready') ||
                lower.includes('start')
            ){

                console.log('✅ READY detected:', currentSpeech);

                stopListening();

                startExamFlow();

                return;
            }
        }


        // ═══════════════════════════════
        // EXAM SELECTION
        // ═══════════════════════════════

        else if(state === S.SEL){

            const m = matchExam(lower);

            if(m !== null){

                stopListening();

                examId = availExams[m].id;
                examName = availExams[m].name;

                loadQuestions();

                return;
            }
        }


        // ═══════════════════════════════
        // ANSWER
        // ═══════════════════════════════

        else if(state === S.LISTENS){

            /*
             * IMPORTANT:
             * Keep everything already spoken.
             * A short pause must NOT erase the answer.
             */

            if(finalT){

                answerBuffer += ' ' + finalT.trim();

                answerBuffer = answerBuffer.trim();

            }

            const displayText =
                (answerBuffer + ' ' + interimT).trim();

            if(displayText){

                showT(displayText);

                console.log('📝 Current answer:', displayText);
            }


            resetSilTimer();


            const commandText = currentSpeech.toLowerCase();

            const words = commandText
                .trim()
                .split(/\s+/);

            const isShort = words.length <= 4;


            // NEXT

            if(
                isShort &&
                (
                    commandText.includes('next') ||
                    commandText.includes('done') ||
                    commandText.includes('move on')
                )
            ){

                stopListening();

                saveAnswer(answerBuffer);

                confirmAndProceed();

                return;
            }


            // REPEAT

            if(
                isShort &&
                (
                    commandText.includes('repeat') ||
                    commandText.includes('again') ||
                    commandText.includes('reread')
                )
            ){

                stopListening();

                readQ();

                return;
            }


            // SUBMIT

            if(
                isShort &&
                (
                    commandText.includes('submit') ||
                    commandText.includes('finish') ||
                    commandText.includes('end exam')
                )
            ){

                stopListening();

                saveAnswer(answerBuffer);

                askSubmit();

                return;
            }
        }


        // ═══════════════════════════════
        // CONFIRM ANSWER
        // ═══════════════════════════════

        else if(state === S.CONFIRM){

            if(
                lower.includes('next') ||
                lower.includes('yes') ||
                lower.includes('continue') ||
                lower.includes('ok')
            ){

                stopListening();

                goNext();

                return;

            }else if(
                lower.includes('repeat') ||
                lower.includes('again') ||
                lower.includes('no') ||
                lower.includes('redo')
            ){

                stopListening();

                readQ();

                return;

            }else if(
                lower.includes('submit') ||
                lower.includes('finish')
            ){

                stopListening();

                askSubmit();

                return;
            }
        }


        // ═══════════════════════════════
        // SUBMIT CONFIRMATION
        // ═══════════════════════════════

        else if(state === S.SUBMIT_Q){

            if(
                lower.includes('yes') ||
                lower.includes('submit') ||
                lower.includes('confirm') ||
                lower.includes('ok')
            ){

                stopListening();

                doSubmit();

                return;

            }else if(
                lower.includes('no') ||
                lower.includes('cancel') ||
                lower.includes('back')
            ){

                stopListening();

                state = S.LISTENS;

                readQ();

                return;
            }
        }
    };


    recognition.onerror = function(e){

        console.log('🎤 Speech recognition error:', e.error);

        recognitionStarting = false;

        if(e.error === 'not-allowed' ||
           e.error === 'service-not-allowed'){

            recognitionRunning = false;

            setMic('idle');

            setStatus(
                '❌ Microphone permission denied. Please allow microphone access in Chrome.'
            );

            return;
        }


        if(e.error === 'audio-capture'){

            recognitionRunning = false;

            setMic('idle');

            setStatus(
                '❌ Microphone not found. Please check your microphone.'
            );

            return;
        }


        if(e.error === 'network'){

            console.log('⚠️ Network error - recognition will retry');

            recognitionRunning = false;

            scheduleRecognitionRestart();

            return;
        }


        /*
         * no-speech and aborted are normal.
         * Automatically restart.
         */

        if(
            e.error === 'no-speech' ||
            e.error === 'aborted'
        ){

            recognitionRunning = false;

            scheduleRecognitionRestart();

            return;
        }


        recognitionRunning = false;

        scheduleRecognitionRestart();
    };


    recognition.onend = function(){

        console.log('🎤 Speech recognition ended');

        recognitionRunning = false;
        recognitionStarting = false;

        if(
            [
                S.LISTENS,
                S.WAIT,
                S.SEL,
                S.CONFIRM,
                S.SUBMIT_Q
            ].includes(state)
            && !isSpeaking
        ){

            scheduleRecognitionRestart();
        }
    };


    return true;
}
function scheduleRecognitionRestart(){

    if(restartTimer){
        clearTimeout(restartTimer);
    }

    restartTimer = setTimeout(()=>{

        restartTimer = null;

        if(
            !recognition ||
            recognitionRunning ||
            recognitionStarting
        ){
            return;
        }

        if(
            ![
                S.LISTENS,
                S.WAIT,
                S.SEL,
                S.CONFIRM,
                S.SUBMIT_Q
            ].includes(state)
        ){
            return;
        }

        try{

            recognitionStarting = true;

            recognition.start();

            console.log('🔄 Speech recognition restarted');

        }catch(e){

            recognitionStarting = false;

            console.log(
                '🔄 Recognition restart failed:',
                e.name
            );

            /*
             * Try again automatically.
             */
            scheduleRecognitionRestart();
        }

    }, 700);
}
    function startListening(){

    /*
     * Do NOT erase the answer when restarting
     * speech recognition.
     */

    finalT = '';
    interimT = '';

    hideT();

    if(!recognition){

        setStatus(
            '❌ Voice recognition is not available. Please use Google Chrome.'
        );

        setMic('idle');

        return;
    }


    setMic('listening');

    setStatus('🎤 Listening... Please speak now.');


    if(recognitionRunning || recognitionStarting){

        console.log('🎤 Recognition already active');

        return;
    }


    try{

        recognitionStarting = true;

        recognition.start();

        console.log('🎤 Speech recognition starting...');

    }catch(e){

        recognitionStarting = false;

        console.log(
            'Recognition start error:',
            e.name
        );


        /*
         * THIS is the important fix.
         *
         * Previously InvalidStateError was ignored.
         * Now we retry automatically.
         */

        if(e.name === 'InvalidStateError'){

            scheduleRecognitionRestart();

        }else{

            setMic('idle');

            setStatus(
                '⚠️ Microphone could not start. Retrying...'
            );

            scheduleRecognitionRestart();
        }
    }
}
    function stopListening(){
        clearSilTimer(); setMic('idle');
        try{ recognition.stop(); }catch(e){}
    }

    // Silence timer: 2.5 s of silence → auto-save answer
    function resetSilTimer(){

    clearSilTimer();

    /*
     * Give the student more time to pause while speaking.
     * 2.5 seconds was too aggressive.
     */

    silTimer = setTimeout(()=>{

        if(
            state === S.LISTENS &&
            !isSpeaking
        ){

            const ans = answerBuffer.trim();

            /*
             * Do NOT automatically finish the answer
             * after a short pause.
             *
             * The student can continue speaking.
             */

            if(ans.length > 1){

                console.log(
                    '⏸️ Silence detected, but answer is kept:',
                    ans
                );

                /*
                 * Keep listening.
                 * Do NOT call stopListening().
                 * Do NOT call confirmAndProceed().
                 */

                setStatus(
                    '🎤 You can continue your answer or say NEXT when finished.'
                );

                /*
                 * Start another longer silence window.
                 */

                resetSilTimer();
            }
        }

    },5000);
}
    function clearSilTimer(){ if(silTimer){clearTimeout(silTimer);silTimer=null;} }

    
    // ════════════════════════════════════════════════════════
    //  TTS FORMATTER — reads options correctly
    //  "a) Oxygen b) Nitrogen" → ". a, Oxygen. b, Nitrogen"
    // ════════════════════════════════════════════════════════
    function fmtTTS(q){
        const pat=/(?:^|\s)([a-dA-D])\)\s*/g;
        const matches=[...q.matchAll(pat)];
        if(!matches.length) return q;
        const stem=q.substring(0,matches[0].index).trim();
        let res=stem;
        matches.forEach((m,idx)=>{
            const letter=m[1].toLowerCase();
            const start=m.index+m[0].length;
            const end=idx+1<matches.length?matches[idx+1].index:q.length;
            const opt=q.substring(start,end).trim().replace(/\.$/, '');
            res+='. '+letter+', '+opt;
        });
        return res;
    }

    // ════════════════════════════════════════════════════════
    //  EXAM FLOW
    // ════════════════════════════════════════════════════════
    function init(){
        if(!initRec()) return;
        fetch('/api/student/data?regNo='+regNo).then(r=>r.json()).then(d=>{ if(d.name)studentName=d.name; });
        fetch('/api/student/available-exams').then(r=>r.json()).then(d=>{
            availExams=d.exams||[]; greet();
        });
    }

    function greet(){
        state=S.WAIT;
        const g=studentName
            ?('Hello '+studentName+'! Welcome to SmartScribe. Are you ready to begin your exam? Say YES when ready.')
            :'Welcome to SmartScribe exam. Are you ready? Say YES when ready.';
        setStatus(g); speak(g,()=>startListening());
    }
    function matchExam(text){
    text = (text || '').toLowerCase().trim();

    console.log("🎤 Exam selection heard:", text);

    const numberWords = [
        'zero',
        'one',
        'two',
        'three',
        'four',
        'five',
        'six',
        'seven',
        'eight',
        'nine',
        'ten'
    ];

    for(let i = 0; i < availExams.length; i++){

        const num = i + 1;
        const word = numberWords[num];

        // "1", "2"
        if(new RegExp(`\\b${num}\\b`).test(text)){
            return i;
        }

        // "one", "two"
        if(new RegExp(`\\b${word}\\b`).test(text)){
            return i;
        }

        // "option one", "option 1"
        if(
            text.includes('option ' + word) ||
            text.includes('option ' + num)
        ){
            return i;
        }

        // Exam name
        if(
            availExams[i].name &&
            text.includes(availExams[i].name.toLowerCase())
        ){
            return i;
        }
    }

    return null;
}

    function startExamFlow(){
        if(availExams.length===0){speak('No exams available. Contact your administrator.');return;}
        if(availExams.length===1){examId=availExams[0].id;examName=availExams[0].name;loadQuestions();return;}
        state=S.SEL;
        let msg='You have '+availExams.length+' exams available. ';
        availExams.forEach((e,i)=>msg+='Option '+(i+1)+': '+e.name+'. ');
        msg+='Say the number or name of your exam.';
        setStatus(msg); speak(msg,()=>startListening());
    }

    function loadQuestions(){
        setStatus('Loading exam...'); speak('Loading '+examName+'. Please wait.',null);
        fetch('/api/exam/questions/'+examId).then(r=>r.json()).then(d=>{
            if(d.questions&&d.questions.length>0){
                questions=d.questions; answers=new Array(questions.length).fill('');
                examDurSec=(d.duration_minutes||60)*60;
                currentIndex=0; showProgress(); updProgress();
                startTimer(examDurSec);
                const intro='Exam loaded. You have '+questions.length+' questions and '+(d.duration_minutes||60)+
                    ' minutes. After each question I will start recording your answer. '+
                    'Say NEXT after your answer, REPEAT to hear the question again, or SUBMIT when finished. '+
                    'Starting now. Question 1.';
                setStatus('✅ Exam started!'); speak(intro,()=>readQ());
            }else{speak('No questions found. Contact your administrator.');}
        }).catch(()=>speak('Error loading exam. Please try again.'));
    }

    function readQ(){
        state=S.READQ; updProgress(); clearSilTimer(); hideT();
        const raw = questions[currentIndex];

/*
 * Remove the question number from the text before TTS.
 *
 * Example:
 * "1. What is photosynthesis?"
 * becomes:
 * "What is photosynthesis?"
 */

        const cleanQuestion = raw
        .replace(/^\s*\d+\s*[\.\):\-]?\s*/, '')
        .trim();

        const spoken = fmtTTS(cleanQuestion);
        setStatus('📢 Q'+(currentIndex+1)+': '+raw);
        speak('Question '+(currentIndex+1)+'. '+spoken,()=>{

    state = S.LISTENS;

    finalT = '';
    interimT = '';

    /*
     * New question = new answer.
     */
    answerBuffer = '';

    setStatus('🎤 Listening... speak your answer.');

    startListening();

    resetSilTimer();
});
    }

    function saveAnswer(text){

    const cleaned = (text || '').trim();

    if(cleaned){

        answers[currentIndex] = cleaned;

        console.log(
            '💾 Answer saved for Q'+(currentIndex+1)+':',
            cleaned
        );
    }
}
    function confirmAndProceed(){
        state=S.CONFIRM;
        const ans=answers[currentIndex];
        if(!ans||!ans.trim()){
            const m='I did not catch your answer. Say REPEAT to hear the question again, or NEXT to skip.';
            setStatus('⚠️ '+m); speak(m,()=>startListening()); return;
        }
        const m='I recorded: '+ans+'. Say NEXT to move on, or REPEAT to redo this question.';
        setStatus('✅ Recorded: '+ans); speak(m,()=>startListening());
    }

    function goNext(){
        if(currentIndex<questions.length-1){currentIndex++;readQ();}else{askSubmit();}
    }

    function askSubmit(){
        state=S.SUBMIT_Q;
        const m='You have completed all '+questions.length+' questions. Say YES to submit and download your answer sheet, or NO to go back to the last question.';
        setStatus('📄 Ready to submit?'); speak(m,()=>startListening());
    }

    function manualSubmit(){
        if(state===S.SUBMITTING) return;
        const cur=(finalT+interimT).trim(); if(cur) saveAnswer(cur);
        stopListening();
        if(confirm('Submit your exam now? Your answers will be downloaded as a PDF.')) doSubmit();
    }

    function doSubmit(){
        if(state===S.SUBMITTING) return;
        state=S.SUBMITTING;
        clearInterval(timerIntvl); 
        const btn=document.getElementById('submitBtn');
        if(btn){btn.disabled=true;btn.textContent='⏳ Submitting...';}
        setMic('idle'); setStatus('⏳ Submitting and generating your answer PDF...');
        speak('Submitting your exam. Please wait.',null);
        const aData=questions.map((q,i)=>({question:q,answer:answers[i]||''}));
        fetch('/api/exam/submit',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({regNo,examId,examName,studentName,answers:aData})})
        .then(res=>{if(!res.ok)throw new Error('Server error '+res.status);return res.blob();})
        .then(blob=>{
            const url=window.URL.createObjectURL(blob);
            const a=document.createElement('a');a.href=url;
            const sn=(studentName||regNo).replace(/[^a-zA-Z0-9]/g,'_');
            const se=(examName||'Exam').replace(/[^a-zA-Z0-9]/g,'_');
            a.download=sn+'.'+se+'.pdf';
            document.body.appendChild(a);a.click();a.remove();
            window.URL.revokeObjectURL(url);
            setStatus('🎉 Exam submitted! Answer sheet downloaded.');
            speak('Congratulations! Your exam has been submitted and your answer sheet downloaded. Well done!',null);
            document.getElementById('timerBox').textContent='Done';
            document.getElementById('timerBox').classList.remove('warn');
        }).catch(err=>{
            console.error(err);
            setStatus('❌ Submission error. Please contact your invigilator.');
            speak('There was an error submitting. Please inform your invigilator.',null);
            if(btn){btn.disabled=false;btn.textContent='📄 Submit Exam';}
            state=S.SUBMIT_Q;
        });
    }

    function confirmLogout(){
        if(confirm('Logout? Unsaved answers will be lost.')){
            window.speechSynthesis.cancel();
            clearInterval(timerIntvl); 
            if(recognition)try{recognition.stop();}catch(e){}
            window.location.href='/';
        }
    }

    // Start after short delay to let browser TTS warm up
    window.addEventListener('load',()=>setTimeout(init,500));
    window.addEventListener('beforeunload',()=>{
        window.speechSynthesis.cancel(); 
        if(recognition)try{recognition.stop();}catch(e){}
    });
    </script></body></html>
    """, regNo=regNo)


# ==================== ORIGINAL NOTEBOOK CELL 9 ====================
# ==================== CELL 9: ALL API ROUTES ====================
# ==================== CREATE EXAM PAGE ====================

@app.route('/create-exam', methods=['GET'])
def create_exam_page():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Create Exam - SmartScribe</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: "Segoe UI", Arial, sans-serif;
            background: #f5f3f8;
            color: #222;
        }

        .topbar {
            background: linear-gradient(135deg, #1a1880, #3d3aa8);
            color: white;
            padding: 22px 35px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .topbar h1 {
            margin: 0;
            font-size: 25px;
        }

        .topbar p {
            margin: 5px 0 0;
            opacity: .85;
        }

        .back-btn {
            background: rgba(255,255,255,.15);
            color: white;
            border: 1px solid rgba(255,255,255,.3);
            padding: 10px 18px;
            border-radius: 8px;
            text-decoration: none;
            cursor: pointer;
        }

        .container {
            max-width: 850px;
            margin: 40px auto;
            padding: 0 20px;
        }

        .card {
            background: white;
            border-radius: 18px;
            padding: 32px;
            box-shadow: 0 8px 30px rgba(0,0,0,.08);
        }

        .card h2 {
            margin-top: 0;
            color: #1a1880;
        }

        .field {
            margin-bottom: 22px;
        }

        label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
        }

        input[type="text"],
        input[type="number"] {
            width: 100%;
            padding: 13px 15px;
            border: 1px solid #d7d7df;
            border-radius: 9px;
            font-size: 15px;
            outline: none;
        }

        input:focus {
            border-color: #1a1880;
        }

        .upload-box {
            border: 2px dashed #c8c5dc;
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            background: #faf9fd;
        }

        input[type="file"] {
            margin-top: 10px;
        }

        .file-name {
            margin-top: 10px;
            color: #555;
            font-size: 14px;
        }

        .btn {
            width: 100%;
            padding: 15px;
            border: none;
            border-radius: 10px;
            background: linear-gradient(135deg, #1a1880, #4b47bd);
            color: white;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
        }

        .btn:hover {
            opacity: .92;
        }

        .btn:disabled {
            opacity: .6;
            cursor: not-allowed;
        }

        .status {
            display: none;
            margin-top: 20px;
            padding: 15px;
            border-radius: 10px;
            font-weight: 500;
        }

        .success {
            display: block;
            background: #e8f7ed;
            color: #176b36;
            border: 1px solid #b7e5c5;
        }

        .error {
            display: block;
            background: #fdecec;
            color: #9b1c1c;
            border: 1px solid #f3bcbc;
        }

        .info {
            background: #f1efff;
            color: #3d3aa8;
            padding: 13px;
            border-radius: 9px;
            margin-bottom: 22px;
            font-size: 14px;
        }
    </style>
</head>

<body>

<div class="topbar">
    <div>
        <h1>📝 Create Exam</h1>
        <p>Upload a question paper and create a SmartScribe exam</p>
    </div>

    <a href="/admin-dashboard" class="back-btn">
        ← Dashboard
    </a>
</div>

<div class="container">

    <div class="card">

        <h2>Create New Exam</h2>

        <div class="info">
            📌 Upload a PDF containing numbered questions such as:
            <br><br>
            <b>1. What is photosynthesis?</b><br>
            <b>2. What is chlorophyll?</b><br>
            <b>3. What is bacteria?</b>
        </div>

        <div class="field">
            <label>Exam Name</label>
            <input
                type="text"
                id="examName"
                placeholder="Example: Biology Test"
            >
        </div>

        <div class="field">
            <label>Exam Duration (minutes)</label>
            <input
                type="number"
                id="duration"
                value="60"
                min="1"
            >
        </div>

        <div class="field">

            <label>Question Paper PDF</label>

            <div class="upload-box">

                📄 Select Question Paper

                <br>

                <input
                    type="file"
                    id="pdfFile"
                    accept="application/pdf"
                    onchange="showFileName()"
                >

                <div id="fileName" class="file-name">
                    No PDF selected
                </div>

            </div>

        </div>

        <button
            class="btn"
            id="createBtn"
            onclick="createExam()"
        >
            🚀 Create Exam
        </button>

        <div id="status" class="status"></div>

    </div>

</div>

<script>

function showFileName() {

    const fileInput = document.getElementById('pdfFile');
    const fileName = document.getElementById('fileName');

    if (fileInput.files.length > 0) {
        fileName.textContent =
            "Selected: " + fileInput.files[0].name;
    } else {
        fileName.textContent =
            "No PDF selected";
    }
}


function showStatus(message, type) {

    const status = document.getElementById('status');

    status.textContent = message;

    status.className = 'status ' + type;
}


async function createExam() {

    const name =
        document.getElementById('examName').value.trim();

    const duration =
        document.getElementById('duration').value;

    const fileInput =
        document.getElementById('pdfFile');

    const btn =
        document.getElementById('createBtn');


    // Validate exam name

    if (!name) {

        showStatus(
            '❌ Please enter an exam name.',
            'error'
        );

        return;
    }


    // Validate PDF

    if (!fileInput.files.length) {

        showStatus(
            '❌ Please select a PDF question paper.',
            'error'
        );

        return;
    }


    const file = fileInput.files[0];


    if (file.type !== 'application/pdf') {

        showStatus(
            '❌ Please select a PDF file only.',
            'error'
        );

        return;
    }


    btn.disabled = true;

    btn.textContent = '⏳ Creating Exam...';

    showStatus(
        '📄 Reading question paper...',
        'info'
    );


    try {

        const reader = new FileReader();


        reader.onload = async function(e) {

            const pdfBase64 = e.target.result;


            showStatus(
                '🤖 Extracting questions from PDF...',
                'info'
            );


            try {

                const response = await fetch(
                    '/api/admin/create-exam',
                    {
                        method: 'POST',

                        headers: {
                            'Content-Type': 'application/json'
                        },

                        body: JSON.stringify({

                            name: name,

                            duration: parseInt(duration) || 60,

                            pdf: pdfBase64

                        })
                    }
                );


                const data = await response.json();


                if (!response.ok || !data.success) {

                    throw new Error(
                        data.message ||
                        data.error ||
                        'Failed to create exam'
                    );

                }


                showStatus(
                    '✅ Exam created successfully! ' +
                    data.question_count +
                    ' questions detected.',
                    'success'
                );


                btn.textContent =
                    '✅ Exam Created';


                // Give the user time to see success

                setTimeout(function() {

                    window.location.href =
                        '/view-exams';

                }, 1800);

            }

            catch(error) {

                console.error(
                    'Create exam error:',
                    error
                );

                showStatus(
                    '❌ ' + error.message,
                    'error'
                );

                btn.disabled = false;

                btn.textContent =
                    '🚀 Create Exam';

            }

        };


        reader.onerror = function() {

            showStatus(
                '❌ Could not read the PDF file.',
                'error'
            );

            btn.disabled = false;

            btn.textContent =
                '🚀 Create Exam';

        };


        reader.readAsDataURL(file);

    }

    catch(error) {

        console.error(error);

        showStatus(
            '❌ Something went wrong. Please try again.',
            'error'
        );

        btn.disabled = false;

        btn.textContent =
            '🚀 Create Exam';

    }

}

</script>

</body>
</html>
""")
# ─── ADMIN ─────────────────────────────────────────────────
@app.route('/api/admin/register', methods=['POST'])
def api_admin_register():
    d=request.json; u=d.get('username'); p=d.get('password')
    if not u or not p: return jsonify({'success':False,'message':'Username and password required'})
    with get_db() as conn:
        c=conn.cursor()
        if c.execute("SELECT id FROM admins WHERE username=?",(u,)).fetchone():
            return jsonify({'success':False,'message':'Username already exists'})
        c.execute("INSERT INTO admins (username,password) VALUES (?,?)",(u,hashlib.sha256(p.encode()).hexdigest()))
    return jsonify({'success':True})

@app.route('/api/admin/login', methods=['POST'])
def api_admin_login():
    d=request.json
    with get_db() as conn:
        a=conn.cursor().execute("SELECT password FROM admins WHERE username=?",(d.get('username'),)).fetchone()
        if a and a['password']==hashlib.sha256(d.get('password','').encode()).hexdigest():
            return jsonify({'success':True})
    return jsonify({'success':False})

@app.route('/api/admin/create-student', methods=['POST'])
def api_create_student():
    data=request.json; regNo=data.get('regNo')
    with get_db() as conn:
        if conn.cursor().execute("SELECT reg_no FROM students WHERE reg_no=?",(regNo,)).fetchone():
            return jsonify({'success':False,'message':'Student already exists'})

    # Face
    face_enc=extract_face_features(data.get('image',''))
    if face_enc is None:
        return jsonify({'success':False,'message':'No face detected. Upload a clear, well-lit photo.'})

    # Voice — accept any audio format, convert to MFCC embedding + store WAV
    raw_voice=data.get('voice','')
    if not raw_voice:
        return jsonify({'success':False,'message':'Voice sample missing. Please record voice in Step 2.'})

    emb, wav_b64 = compute_voice_embedding(raw_voice)
    if emb is None:
        return jsonify({'success':False,'message':'Voice sample too short or silent. Please record 5+ seconds of clear speech.'})

    with get_db() as conn:
        conn.cursor().execute(
            "INSERT INTO students (reg_no,name,image,face_encoding,voice_embedding,voice_wav_b64) VALUES (?,?,?,?,?,?)",
            (regNo, data.get('name'), data.get('image'),
             json.dumps(face_enc), json.dumps(emb.tolist()), wav_b64)
        )
    print(f"Student registered: {regNo} | voice_emb_len={len(emb)}")
    return jsonify({'success':True})

@app.route('/api/admin/delete-student', methods=['POST'])
def api_delete_student():
    with get_db() as conn:
        conn.cursor().execute("DELETE FROM students WHERE reg_no=?",(request.json.get('regNo'),))
    return jsonify({'success':True})

@app.route('/api/admin/stats', methods=['GET'])
def api_stats():
    with get_db() as conn:
        c=conn.cursor()
        ts=c.execute("SELECT COUNT(*) as n FROM students").fetchone()['n']
        te=c.execute("SELECT COUNT(*) as n FROM exams").fetchone()['n']
        tsub=c.execute("SELECT COUNT(*) as n FROM exam_submissions").fetchone()['n']
    return jsonify({'totalStudents':ts,'totalExams':te,'totalSubmissions':tsub})

@app.route('/api/admin/create-exam', methods=['POST'])
def api_create_exam():

    d = request.json
    dur = int(d.get('duration', 60))

    qs = parse_questions_from_text(
        extract_text_from_pdf(d.get('pdf', ''))
    )

    print("🔥 QUESTIONS BEING SAVED:", qs)

    eid = str(uuid.uuid4())[:8]

    with get_db() as conn:
        conn.cursor().execute(
            "INSERT INTO exams (id,name,pdf,questions,duration_minutes) VALUES (?,?,?,?,?)",
            (
                eid,
                d.get('name'),
                d.get('pdf'),
                json.dumps(qs),
                dur
            )
        )

    return jsonify({
        'success': True,
        'exam_id': eid,
        'question_count': len(qs)
    })


@app.route('/api/admin/download-submission/<int:sid>', methods=['GET'])
def api_dl_submission(sid):
    with get_db() as conn:
        row=conn.cursor().execute("SELECT pdf_data,pdf_filename FROM exam_submissions WHERE id=?",(sid,)).fetchone()
    if not row or not row['pdf_data']: return 'PDF not found',404
    return send_file(io.BytesIO(row['pdf_data']),as_attachment=True,
        download_name=row['pdf_filename'] or f'submission_{sid}.pdf',mimetype='application/pdf')

# ─── STUDENT ───────────────────────────────────────────────
@app.route('/api/student/data', methods=['GET'])
def api_student_data():
    rn=request.args.get('regNo')
    with get_db() as conn:
        s=conn.cursor().execute("SELECT name FROM students WHERE reg_no=?",(rn,)).fetchone()
    return jsonify({'name':s['name'] if s else ''})

@app.route('/api/student/available-exams', methods=['GET'])
def api_available_exams():
    with get_db() as conn:
        exams=conn.cursor().execute("SELECT id,name FROM exams ORDER BY created_at DESC").fetchall()
    return jsonify({'exams':[{'id':e['id'],'name':e['name']} for e in exams]})

@app.route('/api/student/verify-face', methods=['POST'])
def api_verify_face():
    d=request.json; rn=d.get('regNo')
    with get_db() as conn:
        s=conn.cursor().execute("SELECT face_encoding FROM students WHERE reg_no=?",(rn,)).fetchone()
    if not s: return jsonify({'success':False,'message':'Registration number not found'})
    try: enc=json.loads(s['face_encoding'])
    except: return jsonify({'success':False,'message':'Face data corrupted'})
    ok,msg=verify_face(enc,d.get('image',''))
    print(f"Face verify {rn}: {ok} — {msg}")
    return jsonify({'success':ok,'message':msg})

@app.route('/api/student/verify-voice', methods=['POST'])
def api_verify_voice():
    d=request.json; rn=d.get('regNo')
    with get_db() as conn:
        s=conn.cursor().execute("SELECT voice_embedding FROM students WHERE reg_no=?",(rn,)).fetchone()
    if not s: return jsonify({'success':False,'message':'Student not found'})
    if not s['voice_embedding']:
        # No voice enrolled — block login, student must re-register with voice
        return jsonify({'success':False,'message':'No voice registered for this student. Please contact your administrator to re-register with a voice sample.','sim':0.0})
    reg_emb=json.loads(s['voice_embedding'])
    ok,msg,sim=verify_voice_embedding(reg_emb, d.get('voice',''))
    print(f"Voice verify {rn}: {ok} sim={sim:.3f}")
    return jsonify({'success':ok,'message':msg,'sim':round(sim,3)})

@app.route('/api/student/verify-voice-monitor', methods=['POST'])
def api_voice_monitor():
    """Real-time guard during exam — lenient threshold, silent on short samples."""
    d=request.json; rn=d.get('regNo')
    with get_db() as conn:
        s=conn.cursor().execute("SELECT voice_embedding FROM students WHERE reg_no=?",(rn,)).fetchone()
    if not s or not s['voice_embedding']:
        return jsonify({'ok':False,'sim':0.0})  # no enrollment → fail guard
    reg_emb=json.loads(s['voice_embedding'])
    pcm, _=audio_b64_to_pcm_numpy(d.get('voice',''))
    if pcm is None or len(pcm)<8000:
        return jsonify({'ok':True,'sim':1.0})   # too short → skip
    emb=extract_mfcc_embedding(pcm)
    if emb is None:
        return jsonify({'ok':True,'sim':1.0})
    sim=cosine_sim(reg_emb, emb.tolist())
    print(f"Voice monitor {rn}: sim={sim:.3f}")
    return jsonify({'ok': sim>=VOICE_MONITOR_THRESHOLD,'sim':round(sim,3)})

# ─── EXAM ──────────────────────────────────────────────────
@app.route('/api/exam/questions/<exam_id>', methods=['GET'])
def api_exam_questions(exam_id):
    with get_db() as conn:
        e=conn.cursor().execute("SELECT questions,duration_minutes FROM exams WHERE id=?",(exam_id,)).fetchone()
    if e:
        return jsonify({'questions':json.loads(e['questions']),'duration_minutes':e['duration_minutes'] or 60})
    return jsonify({'questions':[],'duration_minutes':60})

@app.route('/api/exam/pdf/<exam_id>', methods=['GET'])
def api_exam_pdf(exam_id):
    with get_db() as conn:
        e=conn.cursor().execute("SELECT name,questions,duration_minutes,created_at FROM exams WHERE id=?",(exam_id,)).fetchone()
    if e:
        qs=json.loads(e['questions'])
        qs_html=''.join([f'<div style="margin-bottom:14px;padding:12px;background:#f8f9fa;border-radius:8px;border-left:4px solid #D082D9;"><strong style="color:#D082D9;">Q{i}:</strong> {q}</div>' for i,q in enumerate(qs,1)])
        return f'<!DOCTYPE html><html><head><title>{e["name"]}</title><style>body{{font-family:Segoe UI,sans-serif;padding:20px;}}.hdr{{background:#D082D9;color:white;padding:18px;border-radius:10px;margin-bottom:18px;}}</style></head><body><div class="hdr"><h1>{e["name"]}</h1><p>Duration: {e["duration_minutes"]} min | Created: {e["created_at"] or "N/A"}</p></div>{qs_html}<button onclick="window.print()" style="background:#D082D9;color:white;border:none;padding:10px 20px;border-radius:5px;cursor:pointer;">🖨️ Print</button></body></html>'
    return '<h2>Exam not found</h2>',404

@app.route('/api/exam/submit', methods=['POST'])
def api_exam_submit():
    try:
        d=request.json
        rn=d.get('regNo'); eid=d.get('examId')
        ename=d.get('examName','Exam'); sname=d.get('studentName','')
        answers=d.get('answers',[])
        if not rn or not eid or not answers: return jsonify({'error':'Missing data'}),400
        if not sname:
            with get_db() as conn:
                s=conn.cursor().execute("SELECT name FROM students WHERE reg_no=?",(rn,)).fetchone()
                if s: sname=s['name']
        pdf_bytes=generate_exam_pdf(rn,sname or 'Unknown',ename,answers)
        safe_s=(sname or rn).replace(' ','_').replace('/','_')
        safe_e=(ename or 'Exam').replace(' ','_').replace('/','_')
        fname=f"{safe_s}.{safe_e}.pdf"
        with get_db() as conn:
            conn.cursor().execute(
                "INSERT INTO exam_submissions (reg_no,student_name,exam_id,exam_name,answers,pdf_data,pdf_filename) VALUES (?,?,?,?,?,?,?)",
                (rn,sname or 'Unknown',eid,ename,json.dumps(answers),pdf_bytes,fname)
            )
        return send_file(io.BytesIO(pdf_bytes),as_attachment=True,download_name=fname,mimetype='application/pdf')
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error':str(e)}),500

# ==================== RENDER / LOCAL STARTUP ====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 70)
    print("SMARTSCRIBE SERVER STARTING")
    print(f"Listening on 0.0.0.0:{port}")
    print("=" * 70)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
