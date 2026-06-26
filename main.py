"""
ADMEXO — Automated Lead Management & Email Tracking System
Backend: FastAPI + SQLite
Serves: index.html (form) and dashboard.html (admin) as static files
"""

import os
import sqlite3
import datetime
import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from groq import Groq
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
SENDER_EMAIL     = os.getenv("SENDER_EMAIL", "")
PUBLIC_URL       = os.getenv("PUBLIC_URL", "http://localhost:8000")
DB_FILE          = "leads.db"
REDIRECT_URL     = "https://admexo.com"

app = FastAPI(title="ADMEXO Lead System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── SQLite Setup ──────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id            TEXT PRIMARY KEY,
            timestamp     TEXT NOT NULL,
            name          TEXT NOT NULL,
            email         TEXT NOT NULL,
            phone         TEXT,
            company       TEXT,
            requirement   TEXT NOT NULL,
            ai_response   TEXT,
            ai_category   TEXT,
            ai_priority   TEXT,
            email_sent    INTEGER DEFAULT 0,
            email_opened  INTEGER DEFAULT 0,
            link_clicked  INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

def save_lead(lead_id, name, email, phone, company, requirement,
              ai_response, ai_category, ai_priority, email_sent):
    conn = get_db()
    conn.execute("""
        INSERT INTO leads
          (id, timestamp, name, email, phone, company, requirement,
           ai_response, ai_category, ai_priority, email_sent)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        lead_id,
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        name, email, phone or "", company or "",
        requirement, ai_response, ai_category, ai_priority,
        1 if email_sent else 0,
    ))
    conn.commit()
    conn.close()

def mark_opened(lead_id):
    conn = get_db()
    conn.execute("UPDATE leads SET email_opened = 1 WHERE id = ?", (lead_id,))
    conn.commit()
    conn.close()

def mark_clicked(lead_id):
    conn = get_db()
    conn.execute("UPDATE leads SET link_clicked = 1 WHERE id = ?", (lead_id,))
    conn.commit()
    conn.close()

def fetch_all_leads():
    conn = get_db()
    rows = conn.execute("SELECT * FROM leads ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── AI Helpers ────────────────────────────────────────────────────────────────

def generate_ai_reply(name: str, requirement: str, lead_id: str) -> str:
    if not GROQ_API_KEY:
        return "Thank you for reaching out. Our team will contact you shortly."

    click_url = f"{PUBLIC_URL}/api/track/click/{lead_id}"
    client    = Groq(api_key=GROQ_API_KEY)
    prompt    = f"""You are a professional business development rep at ADMEXO, an AI-powered growth engineering company.

A lead submitted this form:
Name: {name}
Requirement: {requirement}

Write a short, warm, professional email reply:
- Start with: Hi {name.split()[0]},
- Acknowledge their specific requirement in one line
- Mention how ADMEXO can help with performance marketing, SEO, or AI automation (pick most relevant)
- Include this exact line: Learn more about our solutions here: TRACKABLE_LINK
- Suggest a 20-min discovery call as next step
- Keep it under 130 words total
- Sign off as: Team ADMEXO
- Plain text only, no markdown"""

    res  = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=300,
    )
    body = res.choices[0].message.content.strip()
    body = body.replace("TRACKABLE_LINK", click_url)
    return body


def classify_lead(requirement: str) -> tuple[str, str]:
    if not GROQ_API_KEY:
        return "General", "Medium"

    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""Analyze this business requirement and classify it.

Requirement: "{requirement}"

Respond in EXACTLY this format (2 lines only, no extra text):
CATEGORY: <one of: Performance Marketing | SEO | AI Automation | Content Marketing | CRO | Analytics | General>
PRIORITY: <one of: High | Medium | Low>

Rules:
- High priority: urgent language, enterprise, large budget signals, specific technical needs
- Medium priority: clear business need but no urgency
- Low priority: vague, exploratory, or unclear requirements"""

    res    = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=50,
    )
    output   = res.choices[0].message.content.strip()
    category = "General"
    priority = "Medium"
    for line in output.split("\n"):
        if line.startswith("CATEGORY:"):
            category = line.replace("CATEGORY:", "").strip()
        elif line.startswith("PRIORITY:"):
            priority = line.replace("PRIORITY:", "").strip()
    return category, priority


# ── Email Helper ──────────────────────────────────────────────────────────────

def send_email(to_email: str, to_name: str, body: str, lead_id: str) -> bool:
    if not SENDGRID_API_KEY or not SENDER_EMAIL:
        print("SendGrid credentials missing")
        return False

    open_pixel = f"{PUBLIC_URL}/api/track/open/{lead_id}"
    click_url  = f"{PUBLIC_URL}/api/track/click/{lead_id}"

    html_body = body.replace(
        click_url,
        f'<a href="{click_url}" style="background:#6C63FF;color:white;padding:8px 20px;'
        f'border-radius:6px;text-decoration:none;font-weight:600;">Learn More →</a>'
    ).replace("\n", "<br>")

    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:auto;">
      <div style="background:linear-gradient(90deg,#6C63FF,#8B5CF6);padding:20px 28px;border-radius:8px 8px 0 0;">
        <h2 style="color:white;margin:0;">⚡ ADMEXO</h2>
        <p style="color:rgba(255,255,255,0.8);margin:4px 0 0;font-size:0.85rem;">AI-Powered Growth Engineering</p>
      </div>
      <div style="background:#fff;padding:28px;border:1px solid #e5e7eb;border-top:none;border-radius:0 0 8px 8px;">
        <p style="line-height:1.8;color:#333;">{html_body}</p>
        <hr style="border:none;border-top:1px solid #e5e7eb;margin:20px 0;">
        <p style="font-size:0.75rem;color:#9ca3af;">ADMEXO · AI-Powered Growth Engineering<br>
        This email was sent in response to your enquiry.</p>
      </div>
      <img src="{open_pixel}" width="1" height="1" style="display:none;" alt="">
    </div>"""

    try:
        sg  = SendGridAPIClient(SENDGRID_API_KEY)
        msg = Mail(
            from_email=SENDER_EMAIL,
            to_emails=to_email,
            subject=f"Re: Your enquiry to ADMEXO — {to_name}",
            html_content=html,
        )
        r = sg.send(msg)
        print(f"SendGrid status: {r.status_code}")
        return True
    except Exception as e:
        print(f"SendGrid error: {e}")
        return False


# ── HTML Page Routes ──────────────────────────────────────────────────────────

@app.get("/")
def serve_form():
    """Serve the lead capture form."""
    return FileResponse("index.html")

@app.get("/dashboard")
def serve_dashboard():
    """Serve the admin dashboard."""
    return FileResponse("dashboard.html")


# ── API Routes ────────────────────────────────────────────────────────────────

class LeadRequest(BaseModel):
    name:        str
    email:       str
    phone:       str = ""
    company:     str = ""
    requirement: str

@app.on_event("startup")
def startup():
    init_db()
    print(f"✅ DB ready | Public URL: {PUBLIC_URL}")

@app.post("/api/submit")
def submit_lead(lead: LeadRequest):
    lead_id = str(uuid.uuid4())

    ai_category, ai_priority = classify_lead(lead.requirement)
    ai_response  = generate_ai_reply(lead.name, lead.requirement, lead_id)
    email_sent   = send_email(lead.email, lead.name, ai_response, lead_id)

    save_lead(
        lead_id=lead_id,
        name=lead.name, email=lead.email,
        phone=lead.phone, company=lead.company,
        requirement=lead.requirement,
        ai_response=ai_response,
        ai_category=ai_category,
        ai_priority=ai_priority,
        email_sent=email_sent,
    )

    return {
        "success":     True,
        "lead_id":     lead_id,
        "ai_response": ai_response,
        "ai_category": ai_category,
        "ai_priority": ai_priority,
        "email_sent":  email_sent,
    }

@app.get("/api/track/open/{lead_id}")
def track_open(lead_id: str):
    mark_opened(lead_id)
    print(f"📧 Email opened: {lead_id}")
    pixel = b"\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00\x21\xf9\x04\x00\x00\x00\x00\x00\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00\x3b"
    return Response(content=pixel, media_type="image/gif")

@app.get("/api/track/click/{lead_id}")
def track_click(lead_id: str):
    mark_clicked(lead_id)
    print(f"🔗 Link clicked: {lead_id}")
    return RedirectResponse(url=REDIRECT_URL, status_code=302)

@app.get("/api/leads")
def get_leads():
    leads      = fetch_all_leads()
    total      = len(leads)
    sent       = sum(1 for l in leads if l["email_sent"])
    opened     = sum(1 for l in leads if l["email_opened"])
    clicked    = sum(1 for l in leads if l["link_clicked"])
    open_rate  = round((opened / sent * 100), 1) if sent > 0 else 0
    click_rate = round((clicked / sent * 100), 1) if sent > 0 else 0
    return {
        "leads": leads,
        "stats": {
            "total": total, "sent": sent,
            "opened": opened, "clicked": clicked,
            "open_rate": open_rate, "click_rate": click_rate,
        }
    }

@app.delete("/api/clear")
def clear_leads():
    conn = get_db()
    conn.execute("DELETE FROM leads")
    conn.commit()
    conn.close()
    return {"success": True}