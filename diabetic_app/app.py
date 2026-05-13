"""
MediTrack Pro — Complete Flask Backend
Uses only: Flask, Werkzeug, SQLite3, sklearn, ReportLab, itsdangerous, requests
"""
import os, sqlite3, pickle as pkl, json, secrets, time, io
from datetime import datetime, timedelta
from functools import wraps

from flask import (Flask, render_template, request, redirect, url_for,
                   flash, jsonify, g, session, Response)
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer
from dotenv import load_dotenv

from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                 Paragraph, Spacer)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

load_dotenv(override=True)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))
app.permanent_session_lifetime = timedelta(days=7)

# ── ML ────────────────────────────────────────────────────────────────────────
try:
    diabetes_model = pkl.load(open('diabetes_model.pkl', 'rb'))
    medical_scaler = pkl.load(open('scaler.pkl', 'rb'))
    print("✅ ML models loaded")
except Exception as e:
    diabetes_model = medical_scaler = None
    print(f"⚠️  Rule-based fallback: {e}")

LIVE_IOT = {"bpm": 75, "steps": 0, "spo2": 98.0, "temp": 36.6}
DB = "database.db"

# ── DATABASE ──────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    conn = get_db(); c = conn.cursor()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        full_name TEXT DEFAULT '',
        role TEXT DEFAULT 'patient',
        phone TEXT DEFAULT '',
        email TEXT DEFAULT '',
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        glucose REAL, bp REAL, bmi REAL, age INTEGER,
        insulin REAL DEFAULT 0, skin_thickness REAL DEFAULT 0,
        risk_score REAL DEFAULT 0,
        result TEXT, notes TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS profiles (
        user_id INTEGER PRIMARY KEY,
        age INTEGER DEFAULT 0, height REAL DEFAULT 0, weight REAL DEFAULT 0,
        gender TEXT DEFAULT 'Male', blood_group TEXT DEFAULT 'O+',
        target_weight REAL DEFAULT 70, target_glucose REAL DEFAULT 100,
        diabetes_type TEXT DEFAULT 'None',
        allergies TEXT DEFAULT '', current_medications TEXT DEFAULT '',
        emergency_contact TEXT DEFAULT '', doctor_name TEXT DEFAULT ''
    );
    CREATE TABLE IF NOT EXISTS medications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT, dosage TEXT, frequency TEXT,
        time_of_day TEXT, start_date TEXT, end_date TEXT,
        is_active INTEGER DEFAULT 1,
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        doctor_name TEXT, specialty TEXT,
        appt_date TEXT, appt_time TEXT,
        hospital TEXT, notes TEXT,
        status TEXT DEFAULT 'Scheduled',
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS bookings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        lab_name TEXT, test_type TEXT,
        test_date TEXT, time_slot TEXT,
        cost REAL DEFAULT 0, status TEXT DEFAULT 'Pending',
        payment_id TEXT DEFAULT '',
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS vitals_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        bpm INTEGER DEFAULT 0, steps INTEGER DEFAULT 0,
        spo2 REAL DEFAULT 98, temp REAL DEFAULT 36.6,
        logged_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        role TEXT, message TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    );
    """)
    conn.commit(); conn.close()

def sn(val, t=float, d=0.0):
    try: return t(val) if val not in (None, '', 'None') else d
    except: return d

# ── AUTH DECORATOR ────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        uid = session.get('user_id')
        if not uid:
            flash("Please log in.", "warning")
            return redirect(url_for('login'))
        conn = get_db()
        user = conn.execute(
            "SELECT id,username,role,full_name FROM users WHERE id=?", (uid,)).fetchone()
        conn.close()
        if not user:
            session.clear(); return redirect(url_for('login'))
        g.user = user
        return f(*args, **kwargs)
    return decorated

# ── RISK ENGINE ───────────────────────────────────────────────────────────────
def compute_risk(glucose, bp, bmi, age, insulin=0, skin=0):
    s = 0
    if glucose >= 126: s += 35
    elif glucose >= 100: s += 20
    if bmi >= 30: s += 22
    elif bmi >= 25: s += 11
    if bp >= 90: s += 13
    elif bp >= 80: s += 6
    if age >= 50: s += 15
    elif age >= 40: s += 8
    elif age >= 30: s += 3
    if insulin > 100: s += 10
    if skin > 30: s += 5
    return min(int(s), 100)

def risk_label(score):
    if score >= 65: return "High Risk"
    if score >= 40: return "Moderate Risk"
    return "Low Risk"

# ── SMART AI (no external API) ────────────────────────────────────────────────
def smart_chat(msg_raw: str, ud: dict) -> str:
    msg = msg_raw.lower()
    bpm, steps, spo2, temp = ud.get('bpm',75), ud.get('steps',0), ud.get('spo2',98), ud.get('temp',36.6)
    avg_g = ud.get('avg_glucose', 0)
    name = ud.get('name', 'there')

    if any(w in msg for w in ['hello','hi','hey','good morning','good afternoon']):
        hr_s = 'normal ✅' if 60<=bpm<=100 else 'elevated ⚠️'
        return (f"Hello {name}! 👋 Your heart rate is {bpm} BPM ({hr_s}). "
                f"You've taken {steps:,} steps today. How can I help you?")

    if any(w in msg for w in ['heart','bpm','pulse','rate','tachycardia','bradycardia']):
        if bpm > 100: return f"Your heart rate is **{bpm} BPM** — that's elevated (tachycardia). Rest, stay hydrated, and consult your doctor if it persists above 100 for 30+ min."
        if bpm < 60:  return f"Your heart rate is **{bpm} BPM** — slightly low (bradycardia). If you feel dizzy or fatigued, please consult a physician."
        return f"Your heart rate is **{bpm} BPM** — perfectly normal. Keep it up! 💪"

    if any(w in msg for w in ['step','walk','exercise','activ','cardio','run']):
        pct = min(int(steps/100), 100)
        tip = "🎉 Excellent! Over 7,000 steps significantly lowers blood sugar." if steps>7000 else \
              "💡 Aim for 10,000 steps daily — even a 20-minute walk after meals helps." if steps>3000 else \
              "🚶 Start with a 10-minute walk after each meal — it's one of the best things for diabetes management."
        return f"You've walked **{steps:,} steps** today ({pct}% of 10K goal). {tip}"

    if any(w in msg for w in ['glucose','sugar','blood sugar','hba1c','fasting']):
        if avg_g == 0: return "No glucose readings logged yet. Use the AI Risk Assessment to log your first reading and I'll analyse it for you!"
        s = '🟢 normal' if avg_g<100 else ('🟡 pre-diabetic range' if avg_g<126 else '🔴 diabetic range')
        return (f"Your average glucose is **{avg_g:.0f} mg/dL** — {s}. "
                f"Normal fasting: 70–99. Pre-diabetic: 100–125. Diabetic: ≥126. "
                f"Always confirm with a certified lab HbA1c test.")

    if any(w in msg for w in ['oxygen','spo2','saturation','breathe','breath']):
        s = '✅ excellent' if spo2>=97 else ('⚠️ borderline' if spo2>=94 else '🚨 low — seek medical care')
        return f"Your SpO₂ is **{spo2}%** — {s}. Normal range is 95–100%."

    if any(w in msg for w in ['temp','fever','hot','temperature','chills']):
        s = '✅ normal' if 36<=temp<=37.5 else ('⚠️ slightly elevated' if temp<=38.5 else '🚨 high fever')
        return f"Your body temperature is **{temp}°C** — {s}. Normal: 36–37.5°C. If above 38.5°C, consult a doctor."

    if any(w in msg for w in ['diet','eat','food','meal','calorie','carb','protein','nutrition']):
        return ("🥗 **Diabetic Diet Tips:**\n"
                "• Choose low-GI foods: oats, lentils, vegetables, whole grains\n"
                "• Avoid refined carbs, sugary drinks, white rice/bread\n"
                "• Eat every 3–4 hours to avoid blood sugar spikes\n"
                "• Include 25–30g protein per meal\n"
                "• Visit the **AI Diet Planner** for your personalised daily meal plan!")

    if any(w in msg for w in ['medicine','medication','metformin','tablet','insulin','pill','drug']):
        return ("💊 **Medication Guidance:**\n"
                "Never adjust your medication dose without consulting your doctor.\n"
                "Common diabetes medications include Metformin, SGLT-2 inhibitors, and GLP-1 agonists.\n"
                "Use the **Medication Tracker** to log and manage your medicines with reminders.")

    if any(w in msg for w in ['bmi','weight','fat','obese','overweight']):
        return ("⚖️ **BMI & Weight Management:**\n"
                "• Healthy BMI: 18.5–24.9  |  Overweight: 25–29.9  |  Obese: 30+\n"
                "• Even a 5–10% weight loss dramatically improves insulin sensitivity\n"
                "• Combine diet + 150 min/week of moderate exercise for best results\n"
                "• Track your progress in the **Health Profile** section.")

    if any(w in msg for w in ['bp','blood pressure','hypertension','systolic','diastolic']):
        return ("🩺 **Blood Pressure & Diabetes:**\n"
                "• Target BP for diabetic patients: below 130/80 mmHg\n"
                "• High BP + diabetes = high cardiovascular risk\n"
                "• Reduce sodium intake, avoid alcohol, exercise daily\n"
                "• Log your BP readings in the AI Risk Assessment for tracking.")

    if any(w in msg for w in ['tip','advice','recommend','suggest','help me']):
        return (f"🌟 **Top 5 Diabetes Management Tips for you, {name}:**\n"
                f"1️⃣ Log glucose readings regularly (you have {ud.get('total_records',0)} records so far)\n"
                f"2️⃣ Walk daily — you've done {steps:,} steps today!\n"
                "3️⃣ Eat small, frequent, low-GI meals\n"
                "4️⃣ Sleep 7–8 hours — poor sleep raises blood sugar\n"
                "5️⃣ Get HbA1c tested every 3 months")

    return (f"I'm MediBot 🤖 — your AI health assistant.\n"
            f"Live vitals: ❤️ {bpm} BPM · 🚶 {steps:,} steps · 🩸 {spo2}% SpO₂ · 🌡️ {temp}°C\n"
            "Ask me about glucose, blood pressure, diet, medications, exercise, or any health topic!")

def smart_diet(profile: dict, steps: int, bpm: int) -> dict:
    age, weight = profile.get('age', 40), profile.get('weight', 70)
    target_w = profile.get('target_weight', 65)
    d_type = profile.get('diabetes_type', 'None')
    lose = weight > target_w
    idx = (int(age) + int(weight)) % 3

    breakfasts = [
        ("Moong Dal Chilla + Mint Chutney", "Protein-rich, low GI — perfect for morning glucose control • ~240 kcal"),
        ("Oats Porridge with Almonds & Berries", "High-fibre, slow-release energy • ~290 kcal"),
        ("Vegetable Poha with Groundnuts", "Light, balanced, diabetic-friendly Indian breakfast • ~260 kcal"),
    ]
    lunches = [
        ("Grilled Chicken Salad with Brown Rice", "Lean protein + complex carbs — keeps glucose stable • ~430 kcal"),
        ("Rajma Chawal (small portion) + Raita", "High-fibre legumes, excellent for insulin response • ~410 kcal"),
        ("Paneer + Palak with 2 Multigrain Rotis", "Iron, calcium, protein — ideal Type 2 lunch • ~400 kcal"),
    ]
    dinners = [
        ("Baked Fish + Stir-fried Vegetables", "Omega-3, anti-inflammatory — best evening meal • ~360 kcal"),
        ("Dal Palak Soup + 1 Roti", "Light, high protein, low-carb dinner • ~300 kcal"),
        ("Tofu Bhurji + Sautéed Greens", "Plant protein, low GI — gentle on overnight glucose • ~330 kcal"),
    ]
    snacks = [
        "10–12 almonds + 1 cup green tea (no sugar)",
        "1 small apple with 1 tsp peanut butter",
        "Roasted makhana (fox nuts) — 1 small bowl",
        "Carrot & cucumber sticks with hummus (2 tbsp)",
    ]

    b = breakfasts[idx % 3]; l = lunches[(idx+1) % 3]; d = dinners[(idx+2) % 3]
    wt_msg = f"You're {abs(weight - target_w):.1f} kg {'above' if lose else 'below'} your goal. "
    step_msg = (f"Great activity today — {steps:,} steps! 🔥" if steps>7000
                else f"{steps:,} steps logged — aim for 10,000 daily.")

    return {
        "breakfast":      f"{b[0]} — {b[1]}",
        "morning_snack":  snacks[idx % 4],
        "lunch":          f"{l[0]} — {l[1]}",
        "evening_snack":  snacks[(idx+2) % 4],
        "dinner":         f"{d[0]} — {d[1]}",
        "hydration":      "8–10 glasses of water. Add lemon or cucumber. Avoid fruit juice & soda.",
        "message":        (f"{wt_msg}{step_msg} "
                           f"{'Reduce evening carbs and avoid eating after 8 PM.' if lose else 'Maintain your portion discipline!'} "
                           f"Personalised for {'Type '+d_type[-1] if 'Type' in str(d_type) else 'blood sugar management'}."),
    }

def smart_trend(records: list) -> str:
    if not records:
        return "No records yet — run your first AI Risk Assessment to start tracking your health trends!"
    gs = [r['glucose'] for r in records if r.get('glucose')]
    rs = [r['risk_score'] for r in records if r.get('risk_score')]
    if len(gs) < 2:
        return f"You have {len(gs)} reading(s). Log a few more to unlock trend analysis with actionable insights."
    g_trend = gs[0] - gs[-1]
    g_avg = sum(gs)/len(gs)
    r_avg = sum(rs)/len(rs) if rs else 0
    trend = "📉 improving" if g_trend < -5 else ("📈 worsening" if g_trend > 5 else "➡️ stable")
    concern = "glucose levels" if g_avg>120 else ("BMI" if any(r.get('bmi',0)>27 for r in records) else "risk score")
    rec = ("Cut refined carbs and walk 30 min post-meals to reduce glucose." if g_avg>120
           else "Keep up the healthy habits — your metrics are within range." if r_avg<35
           else "Book a follow-up with your endocrinologist to optimise your treatment plan.")
    return (f"📊 **Trend Report** ({len(gs)} readings) — Glucose trend: {trend}. "
            f"Average glucose: **{g_avg:.0f} mg/dL**, average risk: **{r_avg:.0f}/100**. "
            f"Key metric to watch: **{concern}**. 💡 {rec}")

# ── PDF GENERATOR ──────────────────────────────────────────────────────────────
def generate_pdf(user, records, profile, medications, appointments) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=1.5*cm, bottomMargin=1.5*cm,
                            leftMargin=1.8*cm, rightMargin=1.8*cm)
    styles = getSampleStyleSheet()
    story  = []

    T  = lambda txt, s: Paragraph(txt, s)
    SP = lambda n=6: Spacer(1, n)

    title_s = ParagraphStyle('T', fontSize=22, fontName='Helvetica-Bold',
                              textColor=rl_colors.HexColor('#003399'), spaceAfter=2)
    sub_s   = ParagraphStyle('S', fontSize=10, fontName='Helvetica',
                              textColor=rl_colors.grey, spaceAfter=10)
    h2_s    = ParagraphStyle('H2', fontSize=13, fontName='Helvetica-Bold',
                              textColor=rl_colors.HexColor('#003399'),
                              spaceBefore=12, spaceAfter=5)
    body_s  = ParagraphStyle('B', fontSize=10, fontName='Helvetica',
                              textColor=rl_colors.HexColor('#222222'), spaceAfter=3)
    foot_s  = ParagraphStyle('F', fontSize=8, fontName='Helvetica',
                              textColor=rl_colors.grey, alignment=1)

    story += [
        T("MediTrack Pro", title_s),
        T("Official Clinical Health Report", sub_s),
        T(f"Patient: <b>{user['full_name'] or user['username']}</b> &nbsp;&nbsp; "
          f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", body_s),
        Table([['']], colWidths=[17*cm],
              style=TableStyle([('LINEBELOW',(0,0),(-1,-1),1.5,
                                  rl_colors.HexColor('#003399'))])),
        SP(10),
    ]

    if profile:
        story.append(T("Patient Profile", h2_s))
        pd = [
            ['Age', f"{profile['age']} yrs", 'Gender', profile['gender'] or '—'],
            ['Blood Group', profile['blood_group'] or '—', 'Diabetes Type', profile['diabetes_type'] or '—'],
            ['Height', f"{profile['height']} cm", 'Weight', f"{profile['weight']} kg"],
            ['Target Weight', f"{profile['target_weight']} kg", 'Target Glucose', f"{profile['target_glucose']} mg/dL"],
            ['Doctor', profile['doctor_name'] or '—', 'Emergency', profile['emergency_contact'] or '—'],
            ['Allergies', profile['allergies'] or 'None', 'Medications', (profile['current_medications'] or '—')[:40]],
        ]
        t = Table(pd, colWidths=[3.5*cm,4.5*cm,3.5*cm,5*cm])
        t.setStyle(TableStyle([
            ('FONTNAME',(0,0),(-1,-1),'Helvetica'),
            ('FONTSIZE',(0,0),(-1,-1),9),
            ('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),
            ('FONTNAME',(2,0),(2,-1),'Helvetica-Bold'),
            ('TEXTCOLOR',(0,0),(0,-1),rl_colors.HexColor('#003399')),
            ('TEXTCOLOR',(2,0),(2,-1),rl_colors.HexColor('#003399')),
            ('ROWBACKGROUNDS',(0,0),(-1,-1),[rl_colors.HexColor('#f0f5ff'),rl_colors.white]),
            ('GRID',(0,0),(-1,-1),0.4,rl_colors.HexColor('#ccddff')),
            ('PADDING',(0,0),(-1,-1),5),
        ]))
        story += [t, SP()]

    if medications:
        story.append(T("Active Medications", h2_s))
        md = [['Medication','Dosage','Frequency','Time','Started']]
        for m in medications:
            md.append([m['name'],m['dosage'],m['frequency'],m['time_of_day'],m['start_date'] or '—'])
        mt = Table(md, colWidths=[4.5*cm,3*cm,3.5*cm,3*cm,3*cm])
        mt.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),rl_colors.HexColor('#003399')),
            ('TEXTCOLOR',(0,0),(-1,0),rl_colors.white),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,-1),9),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[rl_colors.HexColor('#f9f9f9'),rl_colors.white]),
            ('GRID',(0,0),(-1,-1),0.4,rl_colors.lightgrey),
            ('PADDING',(0,0),(-1,-1),5),
        ]))
        story += [mt, SP()]

    if appointments:
        story.append(T("Scheduled Appointments", h2_s))
        ad = [['Doctor','Specialty','Date','Time','Hospital']]
        for a in appointments:
            ad.append([f"Dr. {a['doctor_name']}",a['specialty'],
                       a['appt_date'],a['appt_time'],a['hospital'] or '—'])
        at = Table(ad, colWidths=[4*cm,3.5*cm,2.8*cm,2.5*cm,4.2*cm])
        at.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),rl_colors.HexColor('#006633')),
            ('TEXTCOLOR',(0,0),(-1,0),rl_colors.white),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,-1),9),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[rl_colors.HexColor('#f0fff5'),rl_colors.white]),
            ('GRID',(0,0),(-1,-1),0.4,rl_colors.lightgrey),
            ('PADDING',(0,0),(-1,-1),5),
        ]))
        story += [at, SP()]

    story.append(T("Health Assessment History", h2_s))
    if records:
        rd = [['Date','Glucose\n(mg/dL)','BP\n(mmHg)','BMI','Age','Risk','Result']]
        for r in records:
            rd.append([str(r['created_at'])[:10],str(r['glucose']),str(r['bp']),
                       str(r['bmi']),str(r['age']),f"{r['risk_score']}/100",r['result'] or '—'])
        rstyles = [
            ('BACKGROUND',(0,0),(-1,0),rl_colors.HexColor('#003399')),
            ('TEXTCOLOR',(0,0),(-1,0),rl_colors.white),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,-1),8.5),
            ('GRID',(0,0),(-1,-1),0.4,rl_colors.lightgrey),
            ('PADDING',(0,0),(-1,-1),4),
        ]
        for i, r in enumerate(records, 1):
            if 'High' in (r['result'] or ''):
                rstyles.append(('BACKGROUND',(0,i),(-1,i),rl_colors.HexColor('#fff0f0')))
            elif 'Moderate' in (r['result'] or ''):
                rstyles.append(('BACKGROUND',(0,i),(-1,i),rl_colors.HexColor('#fffbe6')))
            else:
                rstyles.append(('BACKGROUND',(0,i),(-1,i),
                    rl_colors.HexColor('#f5fff8') if i%2==0 else rl_colors.white))
        rt = Table(rd, colWidths=[2.8*cm,2.4*cm,2.2*cm,2*cm,1.6*cm,2.2*cm,3.8*cm])
        rt.setStyle(TableStyle(rstyles))
        story.append(rt)
    else:
        story.append(T("No health records found.", body_s))

    story += [SP(16), T(
        "This report is generated by MediTrack Pro. For informational purposes only. "
        "Always consult a qualified physician before making any medical decisions.",
        foot_s)]
    doc.build(story)
    return buf.getvalue()

# ── AUTH ROUTES ───────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET','POST'])
def login():
    if session.get('user_id'): return redirect(url_for('home'))
    if request.method == 'POST':
        u = request.form.get('username','').strip()
        p = request.form.get('password','')
        conn = get_db()
        user = conn.execute(
            "SELECT id,username,password,role,full_name FROM users WHERE username=?", (u,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], p):
            session.permanent = True
            session.update({'user_id': user['id'], 'username': user['username'],
                            'full_name': user['full_name'] or user['username'],
                            'role': user['role']})
            return redirect(url_for('home'))
        flash("Invalid credentials.", "danger")
    return render_template('login.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username  = request.form.get('username','').strip()
        password  = request.form.get('password','')
        full_name = request.form.get('full_name', username).strip()
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "warning")
            return render_template('register.html')
        hashed = generate_password_hash(password, method='pbkdf2:sha256')
        try:
            conn = get_db()
            conn.execute("INSERT INTO users (username,password,full_name) VALUES (?,?,?)",
                         (username, hashed, full_name))
            conn.commit(); conn.close()
            flash("Account created! Please sign in.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already taken. Try another.", "danger")
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
@app.route('/')
@login_required
def home():
    uid = g.user['id']
    conn = get_db()
    graph   = conn.execute("SELECT created_at,glucose,bp,bmi,risk_score FROM predictions "
                           "WHERE user_id=? ORDER BY created_at ASC LIMIT 14", (uid,)).fetchall()
    records = conn.execute("SELECT id,created_at,glucose,bp,bmi,age,result,risk_score,notes "
                           "FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 25", (uid,)).fetchall()
    profile = conn.execute("SELECT * FROM profiles WHERE user_id=?", (uid,)).fetchone()
    meds    = conn.execute("SELECT * FROM medications WHERE user_id=? AND is_active=1 "
                           "ORDER BY created_at DESC", (uid,)).fetchall()
    appts   = conn.execute("SELECT * FROM appointments WHERE user_id=? AND status='Scheduled' "
                           "ORDER BY appt_date ASC LIMIT 6", (uid,)).fetchall()
    bkgs    = conn.execute("SELECT * FROM bookings WHERE user_id=? "
                           "ORDER BY created_at DESC LIMIT 10", (uid,)).fetchall()
    stats   = conn.execute("SELECT COUNT(*) as total, AVG(glucose) as avg_glucose, "
                           "AVG(bmi) as avg_bmi, AVG(risk_score) as avg_risk "
                           "FROM predictions WHERE user_id=?", (uid,)).fetchone()
    lv      = conn.execute("SELECT bpm,steps,spo2,temp FROM vitals_log "
                           "WHERE user_id=? ORDER BY logged_at DESC LIMIT 1", (uid,)).fetchone()
    conn.close()

    return render_template('index.html',
        user=g.user, full_name=session.get('full_name',''), role=session.get('role','patient'),
        dates=[r['created_at'][:10] for r in graph],
        glucose_levels=[r['glucose'] for r in graph],
        bp_levels=[r['bp'] for r in graph],
        bmi_levels=[r['bmi'] for r in graph],
        risk_scores=[r['risk_score'] for r in graph],
        all_records=records, profile=profile,
        medications=meds, appointments=appts,
        all_bookings=bkgs, stats=stats,
        latest_vitals=lv, live_iot=LIVE_IOT,
        razorpay_key=os.getenv('RAZORPAY_KEY_ID',''),
    )

# ── PREDICTION ────────────────────────────────────────────────────────────────
@app.route('/predict_diabetes', methods=['POST'])
@login_required
def predict_diabetes():
    try:
        glucose = sn(request.form.get('glucose'))
        bp      = sn(request.form.get('bp'))
        bmi     = sn(request.form.get('bmi'))
        age     = sn(request.form.get('age'), int)
        insulin = sn(request.form.get('insulin', 0))
        skin    = sn(request.form.get('skin_thickness', 0))
        notes   = request.form.get('notes','').strip()
        risk    = compute_risk(glucose, bp, bmi, age, insulin, skin)
        if diabetes_model and medical_scaler:
            import pandas as pd
            feat  = medical_scaler.transform(
                pd.DataFrame([[glucose, bp, bmi, age]],
                             columns=['Glucose','BloodPressure','BMI','Age']))
            prob  = diabetes_model.predict_proba(feat)[0][1] * 100
            result = "High Risk" if prob>65 else ("Moderate Risk" if prob>40 else "Low Risk")
        else:
            result = risk_label(risk)
        conn = get_db()
        conn.execute("INSERT INTO predictions "
                     "(user_id,glucose,bp,bmi,age,insulin,skin_thickness,risk_score,result,notes) "
                     "VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (g.user['id'],glucose,bp,bmi,age,insulin,skin,risk,result,notes))
        conn.commit(); conn.close()
        flash(f"Assessment Complete — {result}  |  Risk Score: {risk}/100", "info")
    except Exception as e:
        print(f"Predict error: {e}")
        flash("Error processing. Check your inputs.", "danger")
    return redirect(url_for('home'))

# ── PROFILE ───────────────────────────────────────────────────────────────────
@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    f = request.form
    try:
        conn = get_db()
        conn.execute("""REPLACE INTO profiles
            (user_id,age,height,weight,gender,blood_group,target_weight,
             target_glucose,diabetes_type,allergies,current_medications,
             emergency_contact,doctor_name)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            g.user['id'],
            sn(f.get('age'),int), sn(f.get('height')), sn(f.get('weight')),
            f.get('gender','Male'), f.get('blood_group','O+'),
            sn(f.get('target_weight')), sn(f.get('target_glucose')),
            f.get('diabetes_type','None'),
            f.get('allergies',''), f.get('current_medications',''),
            f.get('emergency_contact',''), f.get('doctor_name',''),
        ))
        conn.commit(); conn.close()
        flash("Profile saved! ✅", "success")
    except Exception as e:
        print(e); flash("Failed to save profile.", "danger")
    return redirect(url_for('home'))

# ── MEDICATIONS ───────────────────────────────────────────────────────────────
@app.route('/add_medication', methods=['POST'])
@login_required
def add_medication():
    f = request.form
    try:
        conn = get_db()
        conn.execute("INSERT INTO medications "
                     "(user_id,name,dosage,frequency,time_of_day,start_date,end_date) "
                     "VALUES (?,?,?,?,?,?,?)",
                     (g.user['id'], f.get('med_name'), f.get('dosage'),
                      f.get('frequency'), f.get('time_of_day'),
                      f.get('start_date'), f.get('end_date','')))
        conn.commit(); conn.close()
        flash("Medication added! 💊", "success")
    except Exception as e:
        print(e); flash("Failed to add medication.", "danger")
    return redirect(url_for('home'))

@app.route('/delete_medication/<int:mid>', methods=['POST'])
@login_required
def delete_medication(mid):
    conn = get_db()
    conn.execute("UPDATE medications SET is_active=0 WHERE id=? AND user_id=?",
                 (mid, g.user['id']))
    conn.commit(); conn.close()
    return jsonify({"ok": True})

# ── APPOINTMENTS ──────────────────────────────────────────────────────────────
@app.route('/add_appointment', methods=['POST'])
@login_required
def add_appointment():
    f = request.form
    try:
        conn = get_db()
        conn.execute("INSERT INTO appointments "
                     "(user_id,doctor_name,specialty,appt_date,appt_time,hospital,notes) "
                     "VALUES (?,?,?,?,?,?,?)",
                     (g.user['id'], f.get('doctor_name'), f.get('specialty'),
                      f.get('appointment_date'), f.get('appointment_time'),
                      f.get('hospital'), f.get('appt_notes','')))
        conn.commit(); conn.close()
        flash("Appointment scheduled! 📅", "success")
    except Exception as e:
        print(e); flash("Failed to schedule.", "danger")
    return redirect(url_for('home'))

# ── LAB BOOKING ───────────────────────────────────────────────────────────────
@app.route('/book_lab', methods=['POST'])
@login_required
def book_lab():
    f = request.form
    try:
        conn = get_db()
        conn.execute("INSERT INTO bookings "
                     "(user_id,lab_name,test_type,test_date,time_slot,cost,status,payment_id) "
                     "VALUES (?,?,?,?,?,?,?,?)",
                     (g.user['id'], f.get('lab_name'), f.get('test_type'),
                      f.get('test_date'), f.get('time_slot'), sn(f.get('cost',800)),
                      f"Paid (ID: {f.get('razorpay_payment_id','DEMO')})",
                      f.get('razorpay_payment_id','DEMO')))
        conn.commit(); conn.close()
        flash("Lab test booked! 🧪", "success")
    except Exception as e:
        print(e); flash("Booking failed.", "danger")
    return redirect(url_for('home'))

@app.route('/create_razorpay_order', methods=['POST'])
@login_required
def create_razorpay_order():
    try:
        import razorpay
        d = request.json or {}
        amt = int(float(d.get('amount', 0)))
        client = razorpay.Client(
            auth=(os.getenv('RAZORPAY_KEY_ID'), os.getenv('RAZORPAY_KEY_SECRET')))
        order = client.order.create({
            "amount": amt*100, "currency": "INR",
            "receipt": f"mtp_{g.user['id']}_{int(time.time())}"
        })
        return jsonify({"order_id": order["id"], "key": os.getenv('RAZORPAY_KEY_ID')})
    except Exception as e:
        return jsonify({"error": str(e), "fallback": True}), 200

# ── PDF ────────────────────────────────────────────────────────────────────────
@app.route('/download_records')
@login_required
def download_records():
    uid = g.user['id']
    conn = get_db()
    records = conn.execute(
        "SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC", (uid,)).fetchall()
    profile = conn.execute("SELECT * FROM profiles WHERE user_id=?", (uid,)).fetchone()
    meds    = conn.execute(
        "SELECT * FROM medications WHERE user_id=? AND is_active=1", (uid,)).fetchall()
    appts   = conn.execute(
        "SELECT * FROM appointments WHERE user_id=? AND status='Scheduled' "
        "ORDER BY appt_date ASC", (uid,)).fetchall()
    conn.close()
    pdf = generate_pdf(g.user, records, profile, meds, appts)
    return Response(pdf, mimetype='application/pdf',
                    headers={'Content-Disposition':
                             f'attachment; filename=MediTrack_{g.user["username"]}_Report.pdf'})

# ── IOT ────────────────────────────────────────────────────────────────────────
@app.route('/api/vitals', methods=['POST'])
def receive_vitals():
    global LIVE_IOT
    d = request.json or {}
    LIVE_IOT.update({k: d.get(k, LIVE_IOT[k]) for k in LIVE_IOT})
    uid = request.headers.get('X-User-Id')
    if uid:
        try:
            conn = get_db()
            conn.execute("INSERT INTO vitals_log (user_id,bpm,steps,spo2,temp) VALUES (?,?,?,?,?)",
                         (uid, LIVE_IOT['bpm'], LIVE_IOT['steps'], LIVE_IOT['spo2'], LIVE_IOT['temp']))
            conn.commit(); conn.close()
        except: pass
    return jsonify({"status": "ok", "data": LIVE_IOT})

@app.route('/api/vitals/log', methods=['POST'])
@login_required
def log_vitals():
    global LIVE_IOT
    d = request.json or {}
    bpm, steps = sn(d.get('bpm',75),int), sn(d.get('steps',0),int)
    spo2, temp = sn(d.get('spo2',98)), sn(d.get('temp',36.6))
    LIVE_IOT.update({'bpm':bpm,'steps':steps,'spo2':spo2,'temp':temp})
    conn = get_db()
    conn.execute("INSERT INTO vitals_log (user_id,bpm,steps,spo2,temp) VALUES (?,?,?,?,?)",
                 (g.user['id'], bpm, steps, spo2, temp))
    conn.commit(); conn.close()
    return jsonify({"status":"logged","data":LIVE_IOT})

@app.route('/api/vitals/live')
@login_required
def get_live():
    return jsonify(LIVE_IOT)

# ── AI ENDPOINTS ──────────────────────────────────────────────────────────────
@app.route('/chat', methods=['POST'])
@login_required
def chat():
    d = request.get_json() or {}
    msg = d.get('message','').strip()
    if not msg: return jsonify({'reply': 'Please type a message!'})
    uid = g.user['id']
    conn = get_db()
    profile = conn.execute("SELECT * FROM profiles WHERE user_id=?", (uid,)).fetchone()
    recent  = conn.execute("SELECT AVG(glucose) as ag, COUNT(*) as cnt "
                           "FROM predictions WHERE user_id=?", (uid,)).fetchone()
    conn.execute("INSERT INTO chat_history (user_id,role,message) VALUES (?,?,?)",
                 (uid,'user',msg))
    conn.commit(); conn.close()

    ud = {
        'name': session.get('full_name','Patient'),
        'bpm': LIVE_IOT['bpm'], 'steps': LIVE_IOT['steps'],
        'spo2': LIVE_IOT['spo2'], 'temp': LIVE_IOT['temp'],
        'avg_glucose': round(recent['ag'] or 0, 1) if recent else 0,
        'total_records': recent['cnt'] if recent else 0,
    }
    reply = smart_chat(msg, ud)
    conn = get_db()
    conn.execute("INSERT INTO chat_history (user_id,role,message) VALUES (?,?,?)",
                 (uid,'assistant',reply))
    conn.commit(); conn.close()
    return jsonify({'reply': reply})

@app.route('/api/generate_diet')
@login_required
def generate_diet():
    conn = get_db()
    profile = conn.execute("SELECT * FROM profiles WHERE user_id=?", (g.user['id'],)).fetchone()
    conn.close()
    if not profile:
        return jsonify({"error": "Please complete your health profile first."}), 400
    return jsonify(smart_diet(dict(profile), LIVE_IOT['steps'], LIVE_IOT['bpm']))

@app.route('/api/analyze_trend')
@login_required
def analyze_trend():
    conn = get_db()
    records = conn.execute(
        "SELECT glucose,bmi,risk_score,result FROM predictions "
        "WHERE user_id=? ORDER BY created_at DESC LIMIT 10", (g.user['id'],)).fetchall()
    conn.close()
    return jsonify({"analysis": smart_trend([dict(r) for r in records])})

@app.route('/api/analytics')
@login_required
def analytics():
    uid = g.user['id']
    conn = get_db()
    monthly = conn.execute("""
        SELECT strftime('%Y-%m',created_at) as month,
               ROUND(AVG(glucose),1) as avg_glucose,
               ROUND(AVG(bmi),1) as avg_bmi,
               ROUND(AVG(risk_score),1) as avg_risk,
               COUNT(*) as cnt
        FROM predictions WHERE user_id=?
        GROUP BY month ORDER BY month DESC LIMIT 6""", (uid,)).fetchall()
    rdist = conn.execute(
        "SELECT result, COUNT(*) as cnt FROM predictions "
        "WHERE user_id=? GROUP BY result", (uid,)).fetchall()
    vhist = conn.execute("""
        SELECT date(logged_at) as day,
               ROUND(AVG(bpm),0) as avg_bpm,
               SUM(steps) as total_steps
        FROM vitals_log WHERE user_id=?
        GROUP BY day ORDER BY day DESC LIMIT 14""", (uid,)).fetchall()
    conn.close()
    return jsonify({
        "monthly": [dict(r) for r in monthly],
        "risk_distribution": [dict(r) for r in rdist],
        "vitals_history": [dict(r) for r in vhist],
    })

# ── CONTEXT ───────────────────────────────────────────────────────────────────
@app.context_processor
def inject_now():
    return {'now': datetime.now()}
# ── MAIN ──────────────────────────────────────────────────────────────────────
init_db()

if __name__ == '__main__':
    print("🚀 MediTrack Pro → http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=True)
