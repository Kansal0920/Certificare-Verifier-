# main.py
# AAGNI-CertiFier Backend (Definitive Intelligence Edition V3 - OpenCV)
# Final, stable version with OpenCV QR Code Support and Flexible OCR.

import pytesseract
import cv2
import numpy as np
import re
import sqlite3
import pandas as pd
from io import BytesIO, StringIO
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import qrcode
import base64
import hashlib
from datetime import datetime

# --- SETUP ---
DB_FILE = "database.db"
LOGO_DIR = "uploads/logos"
os.makedirs(LOGO_DIR, exist_ok=True)

# --- ADMIN CREDS ---
ADMIN_EMAIL = "admin@gmail.com"
ADMIN_PASSWORD = "P@ssword123"

# --- FastAPI Init ---
app = FastAPI(title="CertiFier API (Intelligence V3 - OpenCV)", version="V-Final-OpenCV")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DB INIT (UPGRADED SCHEMA) ---
@app.on_event("startup")
async def startup_event():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS institutions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, logo_path TEXT
        )""")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS certificates (
            id INTEGER PRIMARY KEY AUTOINCREMENT, institution_id INTEGER, registration_no TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL, course TEXT, blockchain_hash TEXT,
            FOREIGN KEY (institution_id) REFERENCES institutions (id)
        )""")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS verifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, institution_id INTEGER,
            provided_reg_no TEXT, status TEXT, is_blacklisted INTEGER DEFAULT 0, verification_method TEXT,
            FOREIGN KEY (institution_id) REFERENCES institutions (id)
        )""")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS blacklist (
            id INTEGER PRIMARY KEY AUTOINCREMENT, registration_no TEXT NOT NULL UNIQUE,
            institution_id INTEGER, reason TEXT, timestamp TEXT,
            FOREIGN KEY (institution_id) REFERENCES institutions (id)
        )""")
    conn.commit()
    conn.close()

# --- HELPERS ---
def generate_blockchain_hash(cert_data: dict) -> str:
    raw_string = f"{cert_data['registration_no']}-{cert_data['name']}-{datetime.utcnow()}"
    return hashlib.sha256(raw_string.encode()).hexdigest()

def generate_qr_code_base64(cert_data: dict):
    qr_data_string = (f"Verified by CertiFier\nName: {cert_data['name']}\nReg No: {cert_data.get('registration_no', 'N/A')}\nHash: {cert_data.get('blockchain_hash', 'N/A')[:16]}...")
    qr = qrcode.QRCode(version=1, box_size=10, border=4); qr.add_data(qr_data_string); qr.make(fit=True)
    img = qr.make_image(fill_color='black', back_color='white'); buffered = BytesIO(); img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- API ENDPOINTS ---
@app.post("/admin/login")
async def admin_login(email: str = Form(...), password: str = Form(...)):
    if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
        return {"status": "success", "message": "Authentication successful."}
    return {"status": "error", "message": "Invalid credentials."}

@app.post("/admin/blacklist")
async def blacklist_certificate(reg_no: str = Form(...), inst_id: int = Form(...), reason: str = Form(...)):
    conn = sqlite3.connect(DB_FILE); cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO blacklist (registration_no, institution_id, reason, timestamp) VALUES (?, ?, ?, ?)",
                       (reg_no, inst_id, reason, datetime.now().isoformat()))
        conn.commit()
        return {"status": "success", "message": f"'{reg_no}' has been blacklisted."}
    except sqlite3.IntegrityError:
        return {"status": "error", "message": "This registration number is already blacklisted."}
    finally:
        conn.close()

@app.post("/institution/register")
async def register_institution(name: str = Form(...), logo: UploadFile = File(...)):
    logo_path = os.path.join(LOGO_DIR, logo.filename)
    with open(logo_path, "wb") as buffer: buffer.write(await logo.read())
    conn = sqlite3.connect(DB_FILE); cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO institutions (name, logo_path) VALUES (?, ?)", (name, logo_path))
        conn.commit()
        return {"status": "success", "message": f"Institution '{name}' registered."}
    except sqlite3.IntegrityError:
        return {"status": "error", "message": "Institution already exists."}
    finally: conn.close()

@app.post("/institution/bulk_upload_csv")
async def bulk_upload_csv(institution_name: str = Form(...), csv_file: UploadFile = File(...)):
    conn = sqlite3.connect(DB_FILE); cursor = conn.cursor()
    cursor.execute("SELECT id FROM institutions WHERE name = ?", (institution_name,)); result = cursor.fetchone()
    if not result: conn.close(); return {"status": "error", "message": "Institution not registered."}
    institution_id = result[0]
    try:
        contents = await csv_file.read(); df = pd.read_csv(StringIO(contents.decode('utf-8'))); count = 0
        for _, row in df.iterrows():
            try:
                # Handle multiple possible column names for the unique ID
                unique_id_val = str(row.get('student_id') or row.get('registration_no') or row.get('roll_no'))
                cert_data = {"registration_no": unique_id_val, "name": row['name'], "course": row['course']}
                blockchain_hash = generate_blockchain_hash(cert_data)
                cursor.execute("INSERT INTO certificates (institution_id, registration_no, name, course, blockchain_hash) VALUES (?, ?, ?, ?, ?)",
                               (institution_id, cert_data['registration_no'], cert_data['name'], cert_data['course'], blockchain_hash))
                count += 1
            except (sqlite3.IntegrityError, KeyError): continue
        conn.commit()
        return {"status": "success", "message": f"{count} new records added."}
    except Exception as e: return {"status": "error", "message": f"Failed to process CSV. Error: {e}"}
    finally: conn.close()

@app.get("/dashboard/live_stats")
async def get_live_stats():
    conn = sqlite3.connect(DB_FILE); cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM certificates"); r = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM institutions"); i = cursor.fetchone()[0]
    conn.close()
    return {"total_records_in_db": r, "registered_institutions": i}

@app.get("/dashboard/live_intelligence")
async def get_live_intelligence():
    conn = sqlite3.connect(DB_FILE); cursor = conn.cursor()
    cursor.execute("""
        SELECT i.name, COUNT(v.id) as forgery_count FROM verifications v
        JOIN institutions i ON v.institution_id = i.id WHERE v.status != 'Genuine'
        GROUP BY i.name ORDER BY forgery_count DESC """)
    forgery_trends = [{"institution_name": row[0], "forgery_count": row[1]} for row in cursor.fetchall()]
    
    cursor.execute("""
        SELECT v.timestamp, i.name, v.provided_reg_no, v.status, v.institution_id FROM verifications v
        JOIN institutions i ON v.institution_id = i.id
        WHERE v.status != 'Genuine' AND v.provided_reg_no NOT IN (SELECT registration_no FROM blacklist)
        ORDER BY v.timestamp DESC LIMIT 10 """)
    alerts = [{"timestamp": row[0], "institution_name": row[1], "provided_reg_no": row[2], "status": row[3], "institution_id": row[4]} for row in cursor.fetchall()]
    conn.close()
    return {"forgery_trends": forgery_trends, "alerts": alerts}

@app.post("/verify/certificate_live")
async def verify_certificate_live(file: UploadFile = File(...)):
    contents = await file.read()
    image_cv = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Detect institution first using a quick OCR pass
    quick_text = pytesseract.image_to_string(gray_image)
    cursor.execute("SELECT id, name, logo_path FROM institutions")
    all_institutions = cursor.fetchall()
    detected_institution = None
    for inst_id, inst_name, logo_path in all_institutions:
        if inst_name.split()[0].lower() in quick_text.lower():
            detected_institution = {"id": inst_id, "name": inst_name, "logo_path": logo_path}
            break
            
    if not detected_institution:
        conn.close()
        return {"status": "Forgery Alert", "message": "Could not identify a registered institution on this certificate."}

    inst_id = detected_institution["id"]
    extracted_id = None
    verification_method = None
    
    # ✅ FINAL VERSION: Attempt QR Code detection using OpenCV
    try:
        qr_detector = cv2.QRCodeDetector()
        qr_data, _, _ = qr_detector.detectAndDecode(image_cv)
        if qr_data:
            # Use regex to find any likely ID (e.g., long number/string) in the QR data
            id_match = re.search(r'([A-Za-z0-9]{8,})', qr_data)
            if id_match:
                extracted_id = id_match.group(1)
                verification_method = "QR Code"
    except Exception as e:
        print(f"QR decoding failed, falling back to OCR. Error: {e}")

    # STEP 2: If QR fails, use flexible OCR as a fallback
    if not extracted_id:
        verification_method = "OCR"
        id_pattern = r'(Registration No|Student ID|Roll No|Enrollment No)\.?\s*([A-Za-z0-9\-]+)'
        id_match = re.search(id_pattern, quick_text, re.IGNORECASE)
        if id_match:
            extracted_id = id_match.group(2).strip()

    # If BOTH methods fail to find an ID, exit
    if not extracted_id:
        status = "ID Not Found"
        message = "Could not find a valid ID (Reg No, Student ID, or QR Code) on the certificate."
        cursor.execute("INSERT INTO verifications (timestamp, institution_id, provided_reg_no, status, verification_method) VALUES (?, ?, ?, ?, ?)",
                       (datetime.now().isoformat(), inst_id, "UNKNOWN", status, "N/A"))
        conn.commit(); conn.close()
        return {"status": "Forgery Alert", "message": message}

    # STEP 3: Check against blacklist
    cursor.execute("SELECT reason FROM blacklist WHERE registration_no = ? AND institution_id = ?", (extracted_id, inst_id))
    if cursor.fetchone():
        status = "Blacklisted"; message = f"This certificate ID ({extracted_id}) is blacklisted."
        cursor.execute("INSERT INTO verifications (timestamp, institution_id, provided_reg_no, status, verification_method) VALUES (?, ?, ?, ?, ?)",
                       (datetime.now().isoformat(), inst_id, extracted_id, status, verification_method))
        conn.commit(); conn.close()
        return {"status": "Forgery Alert", "message": message}

    # STEP 4: Perform Structural Analysis (Logo Check)
    try:
        logo_template = cv2.imread(detected_institution["logo_path"], 0)
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(logo_template, None)
        kp2, des2 = orb.detectAndCompute(gray_image, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
        if len(matches) > 10:
             good_matches = matches[:10] # Use top 10 matches
             if good_matches[0].distance > 75: # Check distance of the best match
                 raise Exception("Poor logo match quality.")
        else:
             raise Exception("Not enough logo matches found.")
    except Exception as e:
        status = "Structural Anomaly"; message = f"Structural anomaly detected for {detected_institution['name']} (logo mismatch). Details: {e}"
        cursor.execute("INSERT INTO verifications (timestamp, institution_id, provided_reg_no, status, verification_method) VALUES (?, ?, ?, ?, ?)",
                       (datetime.now().isoformat(), inst_id, extracted_id, status, verification_method))
        conn.commit(); conn.close()
        return {"status": "Forgery Alert", "message": message}

    # STEP 5: Final Data Verification against DB
    cursor.execute("SELECT name, course, blockchain_hash FROM certificates WHERE registration_no = ? AND institution_id = ?",
                   (extracted_id, inst_id))
    cert_result = cursor.fetchone()
    
    if cert_result:
        status = "Genuine"
        name, course, blockchain_hash = cert_result
        verified_data = {"name": name, "course": course, "registration_no": extracted_id, "blockchain_hash": blockchain_hash}
        qr_code_b64 = generate_qr_code_base64(verified_data)
        response = {"status": "Genuine", "message": "Certificate is genuine and verified.", "details": verified_data, "qr_code_base64": qr_code_b64}
    else:
        status = "Data Mismatch"
        message = f"Data mismatch: ID {extracted_id} not found for {detected_institution['name']}."
        response = {"status": "Forgery Alert", "message": message}

    cursor.execute("INSERT INTO verifications (timestamp, institution_id, provided_reg_no, status, verification_method) VALUES (?, ?, ?, ?, ?)",
                   (datetime.now().isoformat(), inst_id, extracted_id, status, verification_method))
    conn.commit()
    conn.close()
    return response
if __name__ == "__main__":
    import uvicorn
    # This line tells the script to run the FastAPI app using the uvicorn server
    uvicorn.run(app, host="127.0.0.1", port=8000)