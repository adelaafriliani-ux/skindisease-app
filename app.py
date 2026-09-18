import streamlit as st
import tensorflow as tf  # dipakai untuk tf.lite.Interpreter (model .tflite)
from PIL import Image
import numpy as np
import json
import time
import hashlib
import os
import base64
import io
import re
import uuid
import csv
from datetime import datetime
from supabase import create_client

try:
    from fpdf import FPDF, XPos, YPos
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Klasifikasi Penyakit Kulit",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")
INFO_IMAGE_DIR = os.path.join(BASE_DIR, "assets", "informasi")


def md(html_string):
    """Render an HTML/markdown block safely.

    Streamlit's markdown renderer treats a physical line as an indented code
    block unless the tag starts at (almost) column zero. Plain textwrap.dedent()
    only removes the *common* leading whitespace across lines and still leaves
    a blank-line inside the block able to reset the parser back into "normal
    text" mode, after which the next line can be re-classified as code again --
    which is exactly what kept happening on the kwitansi/surat block once it
    picked up several blank lines between sections. The reliable fix is to
    collapse the entire snippet onto a single physical line: with every
    run of whitespace (including newlines) squashed to one space, there is no
    longer any leading indentation or blank line for the parser to trip on.
    """
    flattened = re.sub(r"\s+", " ", html_string.strip())
    st.markdown(flattened, unsafe_allow_html=True)


@st.cache_data
def get_logo_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

@st.cache_data
def get_image_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ============================================================
# CUSTOM CSS (dipadatkan supaya tidak makan banyak tempat)
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background-color: #EAF4FB;
    }

    /* Header utama -- dipadatkan */
    .main-header {
        background: linear-gradient(135deg, #6FB6E0 0%, #54A6D8 100%);
        padding: 0.9rem 1.4rem;
        border-radius: 12px;
        margin-bottom: 0.7rem;
        border: none;
        box-shadow: 0 3px 10px rgba(84, 166, 216, 0.25);
    }
    .main-header h1 {
        color: #FFFFFF;
        font-weight: 700;
        margin: 0;
        font-size: 1.25rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.92);
        margin: 0.15rem 0 0 0;
        font-size: 0.8rem;
    }

    /* Card umum */
    .info-card {
        background-color: #FFFFFF;
        border: 1px solid #DCE6EE;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }

    .result-card {
        background: #FFFFFF;
        border: 1px solid #D6E8FA;
        border-radius: 10px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 3px 10px rgba(27, 73, 101, 0.07);
    }
    .result-label {
        font-size: 0.75rem;
        color: #8492A6;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .result-class {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1B4965;
        margin: 0.2rem 0;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    .conf-neutral { background-color: #D6ECF9; color: #17445E; }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #54A6D8 0%, #8FC3E6 100%) !important;
    }
    .stProgress > div > div > div {
        background-color: #D6ECF9 !important;
    }

    /* ==================== KARTU PENANGANAN & RUJUKAN ==================== */
    .action-card {
        background-color: #FFFFFF;
        border: 1px solid #D3E9F6;
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-top: 0.8rem;
    }
    .urgensi-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.78rem;
        margin-bottom: 0.4rem;
    }
    .action-section-title {
        font-weight: 700;
        color: #204C64;
        font-size: 0.88rem;
        margin: 0.6rem 0 0.3rem 0;
    }
    .action-list {
        margin: 0;
        padding-left: 1.1rem;
        color: #39657C;
        font-size: 0.85rem;
        line-height: 1.5;
    }
    .rujukan-box {
        background-color: #EAF4FB;
        border: 1px solid #8FC3E6;
        border-left: 4px solid #54A6D8;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin-top: 0.7rem;
        font-size: 0.85rem;
        color: #204C64;
        line-height: 1.5;
    }

    /* ==================== SURAT RUJUKAN -- GAYA KWITANSI (RINGKAS) ==================== */
    .kwitansi-box {
        background-color: #FFFFFF;
        border: 1px solid #D3E9F6;
        border-radius: 10px;
        box-shadow: 0 4px 14px rgba(84, 166, 216, 0.12);
        padding: 1rem 1.2rem;
        max-width: 380px;
        margin: 0 auto;
        font-family: 'Poppins', sans-serif;
        color: #17324A;
        font-size: 0.8rem;
    }
    .kwitansi-header {
        text-align: center;
        padding-bottom: 0.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px dashed #8FC3E6;
    }
    .kwitansi-header .kwi-title {
        font-weight: 700;
        font-size: 0.92rem;
        color: #1B4965;
    }
    .kwitansi-header .kwi-sub {
        font-size: 0.72rem;
        color: #7C93A6;
        margin-top: 0.1rem;
    }
    .kwi-meta {
        display: flex;
        justify-content: space-between;
        font-size: 0.72rem;
        color: #5C7C8F;
        margin-bottom: 0.5rem;
    }
    .kwi-divider {
        border-top: 1px dashed #B9D9EF;
        margin: 0.5rem 0;
    }
    .kwi-section-title {
        font-weight: 700;
        font-size: 0.78rem;
        color: #204C64;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-bottom: 0.25rem;
    }
    .kwi-row {
        display: flex;
        justify-content: space-between;
        gap: 0.5rem;
        font-size: 0.8rem;
        margin-bottom: 0.15rem;
    }
    .kwi-row .kwi-label { color: #7C93A6; }
    .kwi-row .kwi-value { text-align: right; font-weight: 500; color: #17324A; }
    .kwi-list {
        margin: 0.1rem 0 0.3rem 0;
        padding-left: 1.05rem;
        font-size: 0.78rem;
        line-height: 1.45;
    }
    .kwi-note {
        font-size: 0.76rem;
        line-height: 1.45;
        background: #F7FBFD;
        border-left: 3px solid #54A6D8;
        padding: 0.45rem 0.65rem;
        margin: 0.3rem 0;
    }
    .kwi-disclaimer {
        font-size: 0.68rem;
        color: #97A3AE;
        font-style: italic;
        text-align: center;
        margin-top: 0.6rem;
        border-top: 1px dashed #B9D9EF;
        padding-top: 0.5rem;
    }

    .disclaimer {
        background-color: #FFF8E6;
        border-left: 4px solid #F4B400;
        padding: 0.7rem 1rem;
        border-radius: 6px;
        font-size: 0.8rem;
        color: #6B5B00;
        margin-top: 0.8rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}

    /* ================================================================
       TOOLBAR / MENU BERTINGKAT -- dipadatkan
    ================================================================ */
    .st-key-nav_level1 {
        background-color: #FFFFFF;
        border: 1px solid #C6E4F5;
        border-radius: 12px;
        padding: 4px;
        margin-bottom: 3px;
    }
    .st-key-nav_level1 div[data-testid="stHorizontalBlock"] { gap: 4px; }
    .st-key-nav_level1 button {
        border-radius: 9px !important;
        border: 1px solid transparent !important;
        background-color: transparent !important;
        color: #39657C !important;
        font-weight: 600 !important;
        padding: 0.45rem 0.4rem !important;
        font-size: 0.85rem !important;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    .st-key-nav_level1 button:hover {
        background-color: #EAF4FB !important;
        border-color: #8FC3E6 !important;
        color: #204C64 !important;
    }
    .st-key-nav_level1 button[kind="primary"] {
        background: linear-gradient(135deg, #6FB6E0 0%, #54A6D8 100%) !important;
        border-color: #54A6D8 !important;
        color: #FFFFFF !important;
    }

    .st-key-nav_level2 {
        background-color: #F3F9FC;
        border: 1px solid #D3E9F6;
        border-radius: 11px;
        padding: 4px;
        margin-bottom: 3px;
    }
    .st-key-nav_level2 div[data-testid="stHorizontalBlock"] { gap: 4px; }
    .st-key-nav_level2 button {
        border-radius: 999px !important;
        border: 1px solid transparent !important;
        background-color: transparent !important;
        color: #436F87 !important;
        font-weight: 500 !important;
        padding: 0.4rem 0.5rem !important;
        font-size: 0.82rem !important;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    .st-key-nav_level2 button:hover {
        background-color: #D6ECF9 !important;
        border-color: #8FC3E6 !important;
        color: #204C64 !important;
    }
    .st-key-nav_level2 button[kind="primary"] {
        background-color: #7CB9E0 !important;
        border-color: #7CB9E0 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    .st-key-nav_level3 {
        background-color: #F7FBFD;
        border: 1px solid #D6E9F6;
        border-radius: 10px;
        padding: 4px;
        margin-bottom: 0.7rem;
    }
    .st-key-nav_level3 div[data-testid="stHorizontalBlock"] { gap: 4px; }
    .st-key-nav_level3 button {
        border-radius: 999px !important;
        border: 1px solid transparent !important;
        background-color: transparent !important;
        color: #5C7C8F !important;
        font-weight: 500 !important;
        padding: 0.35rem 0.4rem !important;
        font-size: 0.8rem !important;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    .st-key-nav_level3 button:hover {
        background-color: #EAF4FB !important;
        border-color: #8FC3E6 !important;
        color: #39657C !important;
    }
    .st-key-nav_level3 button[kind="primary"] {
        background-color: #A9D4EE !important;
        border-color: #8FC3E6 !important;
        color: #17445E !important;
        font-weight: 600 !important;
    }

    /* ==================== KARTU INFORMASI PENYAKIT ==================== */
    .disease-photo-box {
        width: 100%;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #DCE6EE;
        margin-bottom: 0.8rem;
    }
    .disease-photo-placeholder {
        background-color: #F5F9FC;
        border: 1.5px dashed #B9CEDE;
        border-radius: 8px;
        padding: 2rem 1rem;
        text-align: center;
        color: #7C8DA0;
        margin-bottom: 0.8rem;
    }
    .section-title {
        font-weight: 700;
        color: #1B4965;
        font-size: 0.95rem;
        margin: 0.8rem 0 0.3rem 0;
    }

    /* ==================== LOGIN PAGE ==================== */
    div[data-testid="stVerticalBlock"]:has(> div.st-key-login_card) {
        display: flex;
        justify-content: center;
    }
    .st-key-login_card {
        background: #ffffff;
        border-radius: 18px;
        padding: 2rem 2.2rem 1.8rem 2.2rem;
        max-width: 380px;
        width: 100%;
        box-shadow: 0 10px 32px rgba(127, 190, 224, 0.2);
        border: 1px solid #D3E9F6;
        text-align: center;
        margin: 2.2rem auto 0 auto;
    }
    .login-icon-badge {
        width: 150px;
        height: 66px;
        border-radius: 14px;
        background: linear-gradient(135deg, #54A6D8 0%, #8FC3E6 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin: 0 auto 0.8rem auto;
        box-shadow: 0 5px 14px rgba(127, 190, 224, 0.28);
        padding: 0.6rem 1rem;
    }
    .login-icon-badge img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }
    .login-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1B4965;
        margin: 0 0 0.2rem 0;
    }
    .login-subtitle {
        font-size: 0.8rem;
        color: #8492A6;
        margin: 0 0 1.2rem 0;
    }
    .st-key-login_card input {
        border-radius: 9px;
        border: 1.5px solid #E0E6ED;
        padding: 0.5rem 0.8rem;
    }
    .st-key-login_card input:focus {
        border-color: #54A6D8;
        box-shadow: 0 0 0 3px rgba(127, 190, 224, 0.18);
    }
    .st-key-login_card .stButton button {
        background: linear-gradient(135deg, #54A6D8 0%, #8FC3E6 100%);
        color: white;
        border: none;
        border-radius: 9px;
        padding: 0.45rem 0;
        font-weight: 600;
        margin-top: 0.3rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .st-key-login_card .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 5px 14px rgba(127, 190, 224, 0.28);
    }
    .login-footer-note {
        font-size: 0.7rem;
        color: #A9B4C2;
        margin-top: 1.1rem;
    }
    .st-key-login_card .stButton:has(button[kind="secondary"]) button {
        background: transparent;
        color: #39657C;
        border: 1.5px solid #C6E4F5;
        box-shadow: none;
        font-weight: 500;
        margin-top: 0.4rem;
    }
    .st-key-login_card .stButton:has(button[kind="secondary"]) button:hover {
        background: #EAF4FB;
        border-color: #8FC3E6;
        color: #204C64;
        box-shadow: none;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# KONEKSI SUPABASE (database permanen, gantiin users.json & riwayat_pasien.json)
# ============================================================
@st.cache_resource
def get_supabase_client():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = get_supabase_client()

# Username yang berperan sebagai "pemilik" akses Administrator. Akun dengan
# username ini otomatis jadi admin tanpa perlu persetujuan (karena dialah
# yang menyetujui admin lain), dan hanya dia yang bisa menyetujui/menolak
# permintaan status Administrator dari akun lain.
SUPERADMIN_USERNAME = "puskesmastambun"

def _normalize_user_record(uname, record):
    """Data user lama (format lawas) dikonversi otomatis di sini setiap kali
    dibaca, supaya tidak perlu migrasi manual dan akun lama tetap bisa login
    seperti biasa.

    Role yang dikenali: "admin", "petugas", "pending_petugas", "user".
    Default untuk akun baru/lama yang tidak punya role tersimpan adalah
    "user" (pengguna biasa, bukan petugas)."""
    if isinstance(record, str):
        role = "admin" if uname.strip().lower() == SUPERADMIN_USERNAME else "user"
        record = {"password": record, "role": role}
    role = record.get("role", "user")
    if role not in ("admin", "petugas", "pending_petugas", "user"):
        role = "user"
    # Akun dengan username SUPERADMIN_USERNAME selalu berstatus admin,
    # tidak peduli apa yang tersimpan di database (mencegah salah konfigurasi).
    if uname.strip().lower() == SUPERADMIN_USERNAME:
        role = "admin"
    return {
        "password": record.get("password", ""),
        "role": role,
        "nama_lengkap": record.get("nama_lengkap", ""),
        "tanggal_lahir": record.get("tanggal_lahir", ""),  # disimpan format "YYYY-MM-DD"
        "email": record.get("email", ""),
    }

def load_users():
    try:
        res = supabase.table("users").select("*").execute()
        raw = {row["username"]: row for row in res.data}
    except Exception:
        return {}
    return {uname: _normalize_user_record(uname, rec) for uname, rec in raw.items()}

def save_users(users):
    """Menimpa seluruh isi tabel users dengan isi dict 'users' -- perilakunya
    sama seperti versi lama yang menimpa seluruh users.json."""
    try:
        supabase.table("users").delete().neq("username", "").execute()
        if users:
            rows = [{"username": uname, **rec} for uname, rec in users.items()]
            supabase.table("users").insert(rows).execute()
    except Exception:
        pass

def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def is_valid_email(email):
    """Validasi format email sederhana -- cukup untuk mencegah salah ketik
    yang jelas-jelas bukan email, bukan verifikasi email sungguhan."""
    if not email:
        return True  # email opsional, boleh dikosongkan
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()) is not None

def get_user_role(username, users=None):
    """'admin', 'petugas', 'pending_petugas', atau 'user'. Mengambil ulang
    dari database kalau 'users' tidak diberikan, supaya selalu dapat status
    terbaru."""
    if users is None:
        users = load_users()
    rec = users.get(username)
    return rec["role"] if rec else "user"

# ============================================================
# RIWAYAT PASIEN (tersimpan permanen di tabel riwayat_pasien Supabase)
# ============================================================
def load_riwayat():
    try:
        res = supabase.table("riwayat_pasien").select("*").order("id").execute()
        return res.data
    except Exception:
        return []

def save_riwayat_entry(entry):
    """Tambahkan satu record riwayat baru ke tabel, tanpa menimpa yang lama."""
    try:
        supabase.table("riwayat_pasien").insert(entry).execute()
    except Exception:
        pass

def update_riwayat_entry_by_index(row_index, updates):
    """Perbarui satu record riwayat yang sudah ada, berdasarkan posisinya di
    hasil load_riwayat() (bukan berdasarkan nomor_tiket) -- supaya tetap benar
    sekalipun ada dua record dengan nomor_tiket yang kebetulan sama (data lama
    peninggalan bug nomor tiket sebelum diperbaiki)."""
    data = load_riwayat()
    if 0 <= row_index < len(data):
        record_id = data[row_index]["id"]
        try:
            supabase.table("riwayat_pasien").update(updates).eq("id", record_id).execute()
            return True
        except Exception:
            return False
    return False

BULAN_INDO = ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
              "Juli", "Agustus", "September", "Oktober", "November", "Desember"]

def parse_waktu_riwayat(waktu_str):
    """Ubah teks 'waktu' yang tersimpan (format '17 Sep 2026, 14:35') jadi
    objek datetime, supaya bisa difilter per bulan. Mengembalikan None kalau
    formatnya tidak dikenali (misalnya data lama yang rusak)."""
    try:
        return datetime.strptime(waktu_str, "%d %b %Y, %H:%M")
    except Exception:
        return None

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

# --- Auto-login dari parameter URL, supaya refresh browser tidak logout ---
if not st.session_state.authenticated:
    try:
        qp_user = st.query_params.get("u")
    except Exception:
        qp_user = None
    if qp_user:
        _users_check = load_users()
        if qp_user in _users_check:
            st.session_state.authenticated = True
            st.session_state.username = qp_user

if not st.session_state.authenticated:
    st.markdown("""
    <style>
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], body {
            background: linear-gradient(135deg, #6FB6E0 0%, #54A6D8 100%) !important;
        }
    </style>
    """, unsafe_allow_html=True)

    users = load_users()
    logo_b64 = get_logo_base64(LOGO_PATH)
    if not logo_b64:
        st.warning(f"File logo '{LOGO_PATH}' tidak ditemukan. Pastikan file tersebut ada di folder yang sama dengan app.py.")

    badge_content = f'<img src="data:image/png;base64,{logo_b64}">' if logo_b64 else "🩺"

    with st.container(key="login_card"):
        if st.session_state.auth_mode == "login":
            md(f"""
            <div class="login-icon-badge">{badge_content}</div>
            <div class="login-title">Klasifikasi Penyakit Kulit</div>
            <div class="login-subtitle">Masuk untuk mengakses aplikasi</div>
            """)

            username_input = st.text_input("Username", key="login_username", placeholder="Username")
            password_input = st.text_input("Password", type="password", key="login_password", placeholder="Password")
            login_clicked = st.button("Masuk", use_container_width=True)

            if login_clicked:
                uname = username_input.strip()
                if uname in users and users[uname]["password"] == hash_password(password_input):
                    st.session_state.authenticated = True
                    st.session_state.username = uname
                    try:
                        st.query_params["u"] = uname
                    except Exception:
                        pass
                    st.rerun()
                else:
                    st.error("Username atau password salah.")

            md('<div style="text-align:center;">Belum punya akun?</div>')
            if st.button("Daftar Akun Baru", use_container_width=True, type="secondary", key="switch_to_register"):
                st.session_state.auth_mode = "register"
                st.rerun()

        else:
            md(f"""
            <div class="login-icon-badge">{badge_content}</div>
            <div class="login-title">Daftar Akun</div>
            <div class="login-subtitle">Buat akun baru untuk aplikasi ini</div>
            """)

            new_username = st.text_input("Username", key="register_username", placeholder="Username baru")
            new_password = st.text_input("Password", type="password", key="register_password", placeholder="Password")
            confirm_password = st.text_input("Konfirmasi Password", type="password", key="register_confirm", placeholder="Ulangi password")
            daftar_sebagai_petugas = st.checkbox(
                "Daftar sebagai Petugas",
                key="register_as_petugas",
                help="Akun akan tetap bisa langsung dipakai seperti biasa sebagai "
                     "User, tapi akses Petugas (memverifikasi & mengedit semua "
                     "riwayat rujukan yang masuk) baru aktif setelah disetujui "
                     "oleh Administrator.",
            )
            register_clicked = st.button("Daftar", use_container_width=True)

            if register_clicked:
                uname = new_username.strip()
                if not uname or not new_password:
                    st.error("Username dan password wajib diisi.")
                elif uname in users:
                    st.error("Username sudah dipakai. Pilih username lain.")
                elif new_password != confirm_password:
                    st.error("Konfirmasi password tidak cocok.")
                else:
                    if uname.strip().lower() == SUPERADMIN_USERNAME:
                        # Akun pemilik akses admin utama -- langsung admin,
                        # tidak perlu (dan tidak mungkin) menyetujui dirinya sendiri.
                        role = "admin"
                    elif daftar_sebagai_petugas:
                        role = "pending_petugas"
                    else:
                        role = "user"
                    users[uname] = {"password": hash_password(new_password), "role": role}
                    save_users(users)
                    if role == "pending_petugas":
                        st.success(
                            "Akun berhasil dibuat! Kamu bisa langsung login sebagai "
                            "User biasa. Akses Petugas akan aktif setelah disetujui "
                            "oleh Administrator."
                        )
                    else:
                        st.success("Akun berhasil dibuat! Silakan masuk.")
                    st.session_state.auth_mode = "login"
                    st.rerun()

            md('<div style="text-align:center;">Sudah punya akun?</div>')
            if st.button("Kembali ke Login", use_container_width=True, type="secondary", key="switch_to_login"):
                st.session_state.auth_mode = "login"
                st.rerun()

        md('<div class="login-footer-note">Skripsi · Klasifikasi Penyakit Kulit · MobileNetV2</div>')

    st.stop()

# ============================================================
# HELPER: TEST-TIME AUGMENTATION (TTA)
# ============================================================
def predict_with_tta(interpreter, base_img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    variations = [
        base_img_array,
        np.fliplr(base_img_array),
        np.flipud(base_img_array),
        np.rot90(base_img_array, k=1),
        np.rot90(base_img_array, k=-1),
    ]

    preds = []
    for v in variations:
        input_data = np.expand_dims(v, axis=0).astype(input_details[0]["dtype"])
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])
        preds.append(output_data[0])

    return np.mean(preds, axis=0)

@st.cache_resource
def load_classifier():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    with open("class_names.json") as f:
        class_names = json.load(f)
    return interpreter, class_names

# ============================================================
# HELPER: SURAT RUJUKAN / SKRINING (PDF, gaya resi antrian)
# ============================================================
def _pdf_safe(text):
    """Ganti karakter khusus (—, emoji, dsb) yang tidak didukung font PDF dasar."""
    if text is None:
        return ""
    replacements = {
        "—": "-", "–": "-", "'": "'", "'": "'", '"': '"', '"': '"', "…": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "ignore").decode("latin-1")


def generate_referral_pdf(nomor_tiket, nama_pasien, usia, no_rm, keluhan,
                           predicted_class, confidence, detail, urgensi_info,
                           no_hp=""):
    """
    Catatan perbaikan: versi sebelumnya memakai parameter `ln=True` yang
    sudah deprecated. Untuk multi_cell(), tanpa new_x/new_y eksplisit,
    fpdf2 meninggalkan kursor di TEPI KANAN sel (bukan kembali ke margin
    kiri). Efeknya baris berikutnya cuma tersisa ruang ~0mm, sehingga
    fpdf2 melempar "Not enough horizontal space to render a single
    character" begitu ada 2 butir gejala/penanganan berturut-turut --
    itulah sumber pop-up merah yang muncul. Sekarang setiap cell/multi_cell
    diberi new_x=XPos.LMARGIN, new_y=YPos.NEXT secara eksplisit.
    """
    pdf = FPDF(format="A4", unit="mm")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    NEXT = {"new_x": XPos.LMARGIN, "new_y": YPos.NEXT}

    # ---------- Kop surat ----------
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, _pdf_safe("PUSKESMAS TAMBUN - SKRINING PENYAKIT KULIT"), align="C", **NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, _pdf_safe("Surat Keterangan Hasil Skrining Awal & Rujukan"), align="C", **NEXT)
    pdf.ln(2)
    pdf.set_draw_color(120, 170, 210)
    pdf.set_line_width(0.6)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(5)

    # ---------- Nomor tiket & tanggal ----------
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(95, 6, _pdf_safe(f"No. Tiket: {nomor_tiket}"))
    pdf.cell(95, 6, _pdf_safe(f"Tanggal: {datetime.now().strftime('%d %B %Y, %H:%M')}"), align="R", **NEXT)
    pdf.ln(2)

    # ---------- Data pasien ----------
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Data Pasien", **NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(40, 6, "Nama")
    pdf.cell(0, 6, _pdf_safe(f": {nama_pasien or '-'}"), **NEXT)
    pdf.cell(40, 6, "Usia")
    pdf.cell(0, 6, _pdf_safe(f": {usia or '-'}"), **NEXT)
    pdf.cell(40, 6, "No. RM / NIK")
    pdf.cell(0, 6, _pdf_safe(f": {no_rm or '-'}"), **NEXT)
    pdf.cell(40, 6, "No. HP")
    pdf.cell(0, 6, _pdf_safe(f": {no_hp or '-'}"), **NEXT)
    pdf.cell(40, 6, "Keluhan")
    pdf.multi_cell(0, 6, _pdf_safe(f": {keluhan or '-'}"), **NEXT)
    pdf.ln(1)

    # ---------- Hasil skrining ----------
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Hasil Skrining AI", **NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(40, 6, "Kondisi terdeteksi")
    pdf.cell(0, 6, _pdf_safe(f": {predicted_class}"), **NEXT)
    pdf.cell(40, 6, "Confidence score")
    pdf.cell(0, 6, _pdf_safe(f": {confidence:.1f}%"), **NEXT)
    pdf.cell(40, 6, "Kategori")
    pdf.cell(0, 6, _pdf_safe(f": {detail.get('kategori', '-')}"), **NEXT)
    pdf.ln(1)

    # ---------- Catatan rujukan ----------
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Catatan Rujukan", **NEXT)
    pdf.set_font("Helvetica", "", 10)
    if detail.get("urgensi") == "tinggi":
        catatan = ("Pasien disarankan SEGERA mendapat "
                   "pemeriksaan lanjutan / rujukan ke dokter spesialis kulit.")
    elif detail.get("urgensi") == "sedang":
        catatan = ("Pasien disarankan periksa ke Puskesmas dalam waktu dekat. Rujukan ke dokter "
                   "diberikan apabila kondisi tidak membaik dalam beberapa hari.")
    else:
        catatan = ("Pasien tetap disarankan periksa ke Puskesmas untuk "
                   "memastikan diagnosis, dan dapat meminta rujukan ke dokter kulit bila diperlukan.")
    pdf.multi_cell(0, 5.5, _pdf_safe(catatan), **NEXT)
    pdf.ln(3)

    # ---------- Disclaimer ----------
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4.5, _pdf_safe(
        "Catatan: hasil ini adalah luaran model machine learning (skrining awal) dan bukan "
        "diagnosis medis final. Diagnosis dan penanganan pasti tetap harus melalui pemeriksaan "
        "langsung oleh tenaga medis profesional."
    ), **NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(8)

    # ---------- Tanda tangan ----------
    y_before = pdf.get_y()
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(120, y_before)
    pdf.cell(70, 6, _pdf_safe(f"..................., {datetime.now().strftime('%d-%m-%Y')}"), align="C", **NEXT)
    pdf.set_xy(120, pdf.get_y())
    pdf.cell(70, 6, "Petugas Pemeriksa", align="C", **NEXT)
    pdf.set_xy(120, pdf.get_y() + 20)
    pdf.cell(70, 6, "(...........................)", align="C", **NEXT)

    return bytes(pdf.output())

def generate_referral_pdf_manual(nomor_tiket, nama_pasien, usia, no_rm, keluhan,
                                  kondisi_awal, urgensi_info, catatan_tambahan, petugas,
                                  no_hp=""):
    """Surat rujukan  -- dibuat TANPA melalui
    klasifikasi foto AI, misalnya untuk pasien yang menolak difoto atau
    butuh rujukan segera. Strukturnya sengaja dibuat mirip surat hasil
    skrining AI supaya format dokumennya konsisten, tapi bagian "Hasil
    Skrining AI" (kelas terdeteksi & confidence score) diganti jadi catatan
    klinis manual dari petugas."""
    pdf = FPDF(format="A4", unit="mm")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    NEXT = {"new_x": XPos.LMARGIN, "new_y": YPos.NEXT}

    # ---------- Kop surat ----------
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, _pdf_safe("PUSKESMAS TAMBUN - SKRINING PENYAKIT KULIT"), align="C", **NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, _pdf_safe("Surat Rujukan"), align="C", **NEXT)
    pdf.ln(2)
    pdf.set_draw_color(120, 170, 210)
    pdf.set_line_width(0.6)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(5)

    # ---------- Nomor tiket & tanggal ----------
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(95, 6, _pdf_safe(f"No. Tiket: {nomor_tiket}"))
    pdf.cell(95, 6, _pdf_safe(f"Tanggal: {datetime.now().strftime('%d %B %Y, %H:%M')}"), align="R", **NEXT)
    pdf.ln(2)

    # ---------- Data pasien ----------
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Data Pasien", **NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(40, 6, "Nama")
    pdf.cell(0, 6, _pdf_safe(f": {nama_pasien or '-'}"), **NEXT)
    pdf.cell(40, 6, "Usia")
    pdf.cell(0, 6, _pdf_safe(f": {usia or '-'}"), **NEXT)
    pdf.cell(40, 6, "No. RM / NIK")
    pdf.cell(0, 6, _pdf_safe(f": {no_rm or '-'}"), **NEXT)
    pdf.cell(40, 6, "No. HP")
    pdf.cell(0, 6, _pdf_safe(f": {no_hp or '-'}"), **NEXT)
    pdf.cell(40, 6, "Keluhan")
    pdf.multi_cell(0, 6, _pdf_safe(f": {keluhan or '-'}"), **NEXT)
    pdf.ln(1)

    # ---------- Kondisi klinis (manual, tanpa AI) ----------
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Kondisi Klinis (Penilaian Manual Petugas)", **NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(40, 6, "Dugaan kondisi")
    pdf.cell(0, 6, _pdf_safe(f": {kondisi_awal or 'Belum ditentukan / menunggu pemeriksaan lanjutan'}"), **NEXT)
    pdf.cell(40, 6, "Diinput oleh")
    pdf.cell(0, 6, _pdf_safe(f": {petugas}"), **NEXT)
    pdf.ln(1)

    # ---------- Catatan rujukan ----------
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Catatan Rujukan", **NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, _pdf_safe(
        catatan_tambahan or "Pasien dirujuk melalui jalur cepat tanpa foto klasifikasi AI, "
        "berdasarkan penilaian klinis langsung oleh petugas."
    ), **NEXT)
    pdf.ln(3)

    # ---------- Disclaimer ----------
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4.5, _pdf_safe(
        "Catatan: surat ini dibuat melalui jalur cepat (tanpa klasifikasi AI) berdasarkan "
        "penilaian klinis petugas. Diagnosis dan penanganan pasti tetap harus melalui "
        "pemeriksaan langsung oleh tenaga medis profesional."
    ), **NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(8)

    # ---------- Tanda tangan ----------
    y_before = pdf.get_y()
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(120, y_before)
    pdf.cell(70, 6, _pdf_safe(f"..................., {datetime.now().strftime('%d-%m-%Y')}"), align="C", **NEXT)
    pdf.set_xy(120, pdf.get_y())
    pdf.cell(70, 6, "Petugas Pemeriksa", align="C", **NEXT)
    pdf.set_xy(120, pdf.get_y() + 20)
    pdf.cell(70, 6, "(...........................)", align="C", **NEXT)

    return bytes(pdf.output())

CLASS_INFO = {
    "Herpes": "Infeksi virus yang menimbulkan lepuhan berkelompok pada kulit.",
    "Pityriasis Versicolor": "Infeksi jamur permukaan kulit yang menimbulkan bercak berubah warna.",
    "Tinea Corporis": "Infeksi jamur berbentuk cincin dengan tepi kemerahan yang meninggi.",
    "Scabies": "Infestasi tungau yang menyebabkan rasa gatal hebat, terutama malam hari.",
    "Seboroik Keratosis": "Pertumbuhan kulit jinak berwarna coklat hingga hitam dengan tekstur berlilin seperti tempelan, umum muncul seiring bertambahnya usia.",
}

CLASS_NAME_ALIASES = {
    "herpes": "Herpes",
    "pityriasis versicolor": "Pityriasis Versicolor",
    "pityriasis": "Pityriasis Versicolor",  # label mentah dari model cuma "pityriasis",
                                             # tanpa "Versicolor" -- tanpa alias ini,
                                             # kelas dianggap "belum terdaftar".
    "tinea corporis": "Tinea Corporis",
    "tinea": "Tinea Corporis",
    "scabies": "Scabies",
    "seborrheic keratosis": "Seboroik Keratosis",
    "seboroik keratosis": "Seboroik Keratosis",
    "seborrheic": "Seboroik Keratosis",
}

def resolve_disease_name(raw):
    norm = raw.lower().replace("_", " ").replace("-", " ").strip()
    if norm in CLASS_NAME_ALIASES:
        return CLASS_NAME_ALIASES[norm]
    for k in DISEASE_DETAILS:
        if k.lower().replace("_", " ").replace("-", " ").strip() == norm:
            return k
    # Fallback jaga-jaga: label mentah dari model kadang lebih pendek/panjang
    # dari nama resmi di DISEASE_DETAILS (mis. "pityriasis" vs "Pityriasis
    # Versicolor"). Kalau salah satu adalah awalan dari yang lain, tetap
    # anggap cocok, supaya kelas baru yang labelnya sedikit berbeda tidak
    # langsung dianggap "belum terdaftar".
    for k in DISEASE_DETAILS:
        k_norm = k.lower().replace("_", " ").replace("-", " ").strip()
        if norm and (k_norm.startswith(norm) or norm.startswith(k_norm)):
            return k
    return raw

DISEASE_DETAILS = {
    "Herpes": {
        "kategori": "Infeksi Virus",
        "image_file": "herpes.jpg",
        "deskripsi": (
            "Herpes kulit adalah infeksi yang disebabkan oleh virus Herpes Simplex (HSV), "
            "ditandai dengan munculnya kelompok lepuhan kecil berisi cairan di atas dasar "
            "kulit yang kemerahan. Lepuhan ini dapat terasa nyeri, gatal, atau seperti "
            "sensasi terbakar sebelum muncul."
        ),
        "gejala": [
            "Muncul kelompok lepuhan kecil berisi cairan bening",
            "Kulit di sekitar lepuhan tampak kemerahan",
            "Rasa nyeri, gatal, atau perih sebelum lepuhan muncul",
            "Dapat disertai demam ringan atau badan terasa tidak enak",
        ],
        "penanganan": [
            "Segera periksakan ke Puskesmas/dokter untuk konfirmasi diagnosis dan terapi antivirus bila diperlukan",
            "Jaga area yang terkena tetap bersih dan kering, hindari memecahkan lepuhan",
            "Hindari menyentuh area lain (mata, mulut) setelah menyentuh lesi untuk mencegah penyebaran",
            "Gunakan pakaian/handuk terpisah selama masa aktif untuk mencegah penularan ke orang lain",
            "Segera ke fasilitas kesehatan bila lepuhan menyebar luas, disertai demam tinggi, atau muncul di area mata",
        ],
        "urgensi": "sedang",
    },
    "Pityriasis Versicolor": {
        "kategori": "Infeksi Jamur",
        "image_file": "pityriasis_versicolor.jpg",
        "deskripsi": (
            "Pityriasis versicolor (panu) adalah infeksi jamur permukaan kulit yang "
            "disebabkan oleh jamur Malassezia. Ditandai dengan bercak kulit yang berubah "
            "warna (lebih terang atau lebih gelap dari kulit sekitar) disertai sedikit sisik halus."
        ),
        "gejala": [
            "Bercak kulit berubah warna, bisa lebih terang atau lebih gelap",
            "Bercak sering muncul di punggung, dada, leher, atau lengan atas",
            "Permukaan bercak bersisik halus",
            "Gatal ringan, terutama saat berkeringat",
        ],
        "penanganan": [
            "Periksakan ke Puskesmas untuk mendapat obat antijamur topikal yang sesuai",
            "Jaga kulit tetap kering, segera ganti pakaian setelah berkeringat",
            "Hindari memakai pakaian ketat dan bahan yang tidak menyerap keringat",
            "Gunakan sabun/sampo antijamur sesuai anjuran tenaga medis secara rutin",
            "Perubahan warna kulit dapat menetap sementara meski infeksi jamurnya sudah sembuh",
        ],
        "urgensi": "rendah",
    },
    "Tinea Corporis": {
        "kategori": "Infeksi Jamur",
        "image_file": "tinea_corporis.jpg",
        "deskripsi": (
            "Tinea corporis (kurap badan) adalah infeksi jamur dermatofita pada kulit "
            "yang khas berbentuk cincin (ring-shaped) dengan tepi meninggi kemerahan dan "
            "bagian tengah tampak lebih bersih/pucat."
        ),
        "gejala": [
            "Ruam berbentuk cincin dengan tepi meninggi dan kemerahan",
            "Bagian tengah ruam cenderung lebih pucat/normal",
            "Gatal, terkadang disertai rasa perih",
            "Dapat menyebar dan membentuk beberapa cincin baru",
        ],
        "penanganan": [
            "Periksakan ke Puskesmas untuk mendapat obat antijamur (topikal atau oral sesuai tingkat keparahan)",
            "Jaga area terkena tetap kering dan bersih",
            "Hindari berbagi pakaian, handuk, atau alat mandi dengan orang lain",
            "Cuci pakaian dan sprei dengan air hangat untuk membantu mengurangi penyebaran jamur",
            "Jika kontak dengan hewan peliharaan, periksakan juga hewan tersebut karena kurap bisa menular dari hewan",
        ],
        "urgensi": "rendah",
    },
    "Scabies": {
        "kategori": "Infestasi Parasit",
        "image_file": "scabies.jpg",
        "deskripsi": (
            "Skabies adalah penyakit kulit menular yang disebabkan oleh tungau Sarcoptes "
            "scabiei. Tungau ini membuat terowongan kecil di bawah kulit dan menimbulkan "
            "rasa gatal yang sangat hebat, terutama pada malam hari."
        ),
        "gejala": [
            "Gatal hebat yang memburuk pada malam hari",
            "Muncul garis-garis kecil (terowongan tungau) di sela jari, pergelangan tangan, atau lipatan kulit",
            "Bintil-bintil kecil kemerahan yang dapat meluas akibat garukan",
            "Sering terjadi pada beberapa anggota keluarga/asrama secara bersamaan",
        ],
        "penanganan": [
            "Segera periksakan ke Puskesmas — skabies memerlukan obat skabisida yang diresepkan tenaga medis",
            "Obati SEMUA anggota keluarga/orang serumah secara bersamaan meski belum bergejala",
            "Cuci seluruh pakaian, sprei, dan handuk dengan air panas, lalu jemur/setrika",
            "Hindari kontak kulit langsung dan pemakaian barang bersama selama masa pengobatan",
            "Jangan menggaruk berlebihan karena dapat menimbulkan infeksi kulit sekunder",
        ],
        "urgensi": "tinggi",
    },
    "Seboroik Keratosis": {
        "kategori": "Kelainan Kulit Jinak",
        "image_file": "seboroik_keratosis.jpg",
        "deskripsi": (
            "Keratosis seboroik adalah pertumbuhan kulit jinak (bukan kanker) berwarna "
            "coklat hingga kehitaman dengan permukaan berlilin/kasar seperti tempelan. "
            "Umumnya muncul seiring bertambahnya usia dan tidak menular."
        ),
        "gejala": [
            "Bercak/benjolan berwarna coklat hingga hitam",
            "Permukaan tampak berlilin, kasar, atau seperti tertempel di kulit",
            "Umumnya tidak nyeri, kadang terasa gatal ringan",
            "Ukuran dapat membesar perlahan dalam waktu lama",
        ],
        "penanganan": [
            "Umumnya tidak berbahaya, namun tetap periksakan ke Puskesmas untuk memastikan bukan kondisi lain",
            "Konsultasikan ke dokter/dermatolog bila lesi berubah bentuk, warna, cepat membesar, mudah berdarah, atau terasa nyeri",
            "Bila mengganggu penampilan atau sering teriritasi pakaian, diskusikan opsi pengangkatan dengan dokter kulit",
            "Hindari menggaruk atau mengelupas sendiri karena berisiko infeksi atau perdarahan",
        ],
        "urgensi": "rendah",
    },
}

URGENSI_STYLE = {
    "tinggi": {"color": "#C0392B", "bg": "#FDEDEC", "label": "Prioritas Tinggi — segera ke faskes"},
    "sedang": {"color": "#B9770E", "bg": "#FEF5E7", "label": "Prioritas Sedang — periksa dalam waktu dekat"},
    "rendah": {"color": "#1E8449", "bg": "#EAFAF1", "label": "Prioritas Rendah — tetap disarankan periksa"},
}

CATEGORY_MAP = {}
for _name, _d in DISEASE_DETAILS.items():
    CATEGORY_MAP.setdefault(_d["kategori"], []).append(_name)

try:
    with st.spinner("Memuat model AI (hanya di percobaan pertama)..."):
        model, class_names = load_classifier()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    LOAD_ERROR = str(e)

# ============================================================
# STATE NAVIGASI
# ============================================================
LEVEL1_ITEMS = ["🔬 Deteksi", "📖 Informasi Penyakit", "ℹ️ Tentang Aplikasi", "👤 Akun"]

if "nav_l1" not in st.session_state:
    st.session_state.nav_l1 = LEVEL1_ITEMS[0]
if "nav_l2_informasi" not in st.session_state:
    st.session_state.nav_l2_informasi = list(CATEGORY_MAP.keys())[0]
if "nav_l3_informasi" not in st.session_state:
    st.session_state.nav_l3_informasi = CATEGORY_MAP[st.session_state.nav_l2_informasi][0]
if "nav_l2_tentang" not in st.session_state:
    st.session_state.nav_l2_tentang = "Tentang Aplikasi"
if "nav_l2_akun" not in st.session_state:
    st.session_state.nav_l2_akun = "Profil Saya"
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = False

# ============================================================
# HEADER
# ============================================================
md("""
<div class="main-header">
    <h1>🩺 Klasifikasi Penyakit Kulit</h1>
    <p>Aplikasi bantu skrining awal penyakit kulit untuk kebutuhan layanan Puskesmas</p>
</div>
""")

# ---------- Tingkat 1 ----------
with st.container(key="nav_level1"):
    cols = st.columns(len(LEVEL1_ITEMS))
    for col, item in zip(cols, LEVEL1_ITEMS):
        with col:
            active = st.session_state.nav_l1 == item
            if st.button(item, key=f"l1_{item}", type="primary" if active else "secondary", use_container_width=True):
                st.session_state.nav_l1 = item
                st.rerun()

current_l1 = st.session_state.nav_l1

# ---------- Tingkat 2 & 3 ----------
if current_l1 == "📖 Informasi Penyakit":
    kategori_list = list(CATEGORY_MAP.keys())
    with st.container(key="nav_level2"):
        cols = st.columns(len(kategori_list))
        for col, kategori in zip(cols, kategori_list):
            with col:
                active = st.session_state.nav_l2_informasi == kategori
                if st.button(kategori, key=f"l2_info_{kategori}", type="primary" if active else "secondary", use_container_width=True):
                    st.session_state.nav_l2_informasi = kategori
                    st.session_state.nav_l3_informasi = CATEGORY_MAP[kategori][0]
                    st.rerun()

    penyakit_list = CATEGORY_MAP[st.session_state.nav_l2_informasi]
    with st.container(key="nav_level3"):
        cols = st.columns(max(len(penyakit_list), 1))
        for col, penyakit in zip(cols, penyakit_list):
            with col:
                active = st.session_state.nav_l3_informasi == penyakit
                if st.button(penyakit, key=f"l3_info_{penyakit}", type="primary" if active else "secondary", use_container_width=True):
                    st.session_state.nav_l3_informasi = penyakit
                    st.rerun()

elif current_l1 == "ℹ️ Tentang Aplikasi":
    tentang_items = ["Tentang Aplikasi", "Kelas yang Dikenali", "Catatan Penting"]
    with st.container(key="nav_level2"):
        cols = st.columns(len(tentang_items))
        for col, item in zip(cols, tentang_items):
            with col:
                active = st.session_state.nav_l2_tentang == item
                if st.button(item, key=f"l2_tentang_{item}", type="primary" if active else "secondary", use_container_width=True):
                    st.session_state.nav_l2_tentang = item
                    st.rerun()

elif current_l1 == "👤 Akun":
    _role_menu_akun = get_user_role(st.session_state.username)
    akun_items = ["Profil Saya", "Riwayat Pasien"]
    if _role_menu_akun in ("admin", "petugas"):
        akun_items.append("Evaluasi Bulanan")
    if _role_menu_akun == "admin":
        akun_items.append("Daftar Petugas")
        akun_items.append("Permintaan Petugas")
    akun_items += ["Keluar", "Hapus Akun"]
    with st.container(key="nav_level2"):
        cols = st.columns(len(akun_items))
        for col, item in zip(cols, akun_items):
            with col:
                if st.button(item, key=f"l2_akun_{item}", type="primary" if st.session_state.nav_l2_akun == item else "secondary", use_container_width=True):
                    if item == "Keluar":
                        st.session_state.authenticated = False
                        st.session_state.username = None
                        try:
                            st.query_params.clear()
                        except Exception:
                            pass
                        st.rerun()
                    else:
                        st.session_state.nav_l2_akun = item
                        st.rerun()

try:
    # ============================================================
    # KONTEN: DETEKSI
    # ============================================================
    if current_l1 == "🔬 Deteksi":
        mode_deteksi = st.radio(
            "Metode",
            ["🖼️ Deteksi dari Foto (AI)", "⚡ Buat Surat Rujukan"],
            horizontal=True,
            key="mode_deteksi",
            label_visibility="collapsed",
        )
        st.markdown("---")

        if mode_deteksi.startswith("⚡"):
            # ============================================================
            # JALUR CEPAT: SURAT RUJUKAN TANPA FOTO / TANPA KLASIFIKASI AI
            # Dipakai untuk pasien yang menolak difoto atau butuh rujukan
            # segera tanpa menunggu proses deteksi.
            # ============================================================
            st.markdown("#### ⚡ Membuat Surat Rujukan")
            st.caption(
                "Untuk pasien yang butuh rujukan segera atau tidak ingin difoto -- "
                "surat dibuat langsung dari pengguna, tanpa klasifikasi AI."
            )

            with st.form(key="form_rujukan_manual"):
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    m_nama = st.text_input("Nama Pasien", key="m_nama")
                    m_usia = st.text_input("Usia", key="m_usia")
                with col_m2:
                    m_rm = st.text_input("No. RM / NIK", key="m_rm")
                    m_hp = st.text_input("No. HP", key="m_hp")
                m_keluhan = st.text_area("Keluhan", key="m_keluhan")
                m_catatan = st.text_area(
                    "Catatan Tambahan (opsional)", key="m_catatan",
                    placeholder="Isi catatan tambahan",
                )
                st.caption(
                    "Ketika sudah berhasil mendapatkan surat rujukan -- bisa "
                    "diberikan ke petugas input Puskesmas Tambun."
                )
                m_submit = st.form_submit_button("🖨️ Buat & Simpan Surat Rujukan", use_container_width=True)

            if m_submit:
                if not m_nama.strip() and not m_keluhan.strip():
                    st.error("Isi minimal Nama Pasien atau Keluhan sebelum membuat surat rujukan.")
                else:
                    m_tiket = f"RJK-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
                    m_waktu = datetime.now().strftime("%d %b %Y, %H:%M")
                    # Urgensi sengaja TIDAK dipilih oleh petugas -- itu wewenang
                    # tenaga medis setelah pasien diperiksa langsung, bukan sesuatu
                    # yang bisa ditentukan dari data yang diinput saat pendaftaran.
                    m_urgensi_info = {"label": "Menunggu penilaian tenaga medis"}

                    m_pdf_b64 = None
                    if FPDF_AVAILABLE:
                        try:
                            m_pdf_bytes = generate_referral_pdf_manual(
                                nomor_tiket=m_tiket,
                                nama_pasien=m_nama.strip() or "-",
                                usia=m_usia.strip() or "-",
                                no_rm=m_rm.strip() or "-",
                                no_hp=m_hp.strip() or "-",
                                keluhan=m_keluhan.strip() or "-",
                                kondisi_awal="",
                                urgensi_info=m_urgensi_info,
                                catatan_tambahan=m_catatan.strip(),
                                petugas=st.session_state.username,
                            )
                            m_pdf_b64 = base64.b64encode(m_pdf_bytes).decode("utf-8")
                        except Exception as _e:
                            st.warning(f"Data pasien tetap disimpan, tapi PDF gagal dibuat: {_e}")

                    save_riwayat_entry({
                        "nomor_tiket": m_tiket,
                        "waktu": m_waktu,
                        "username": st.session_state.username,
                        "nama_pasien": m_nama.strip() or "-",
                        "usia": m_usia.strip() or "-",
                        "no_rm": m_rm.strip() or "-",
                        "no_hp": m_hp.strip() or "-",
                        "keluhan": m_keluhan.strip() or "-",
                        "predicted_class": "Rujukan",
                        "confidence": "-",
                        "kategori": "Rujukan Manual",
                        "urgensi_label": m_urgensi_info["label"],
                        "pdf_base64": m_pdf_b64,
                    })
                    st.success(f"✅ Surat rujukan manual berhasil dibuat & disimpan. No. Tiket: {m_tiket}")
                    if m_pdf_b64:
                        st.download_button(
                            "⬇️ Unduh Surat Rujukan (PDF)",
                            data=base64.b64decode(m_pdf_b64),
                            file_name=f"surat_rujukan_{m_tiket}.pdf",
                            mime="application/pdf",
                            key=f"dl_manual_{m_tiket}",
                            use_container_width=True,
                        )
            st.stop()

        if not MODEL_LOADED:
            st.error(f"Model belum bisa dimuat. Pastikan file 'model.tflite' dan 'class_names.json' berada di folder yang sama dengan app.py.\n\nDetail error: {LOAD_ERROR}")
            st.stop()

        col_upload, col_result = st.columns([1, 1.2], gap="large")

        with col_upload:
            st.markdown("#### 📤 Unggah Gambar")
            uploaded_file = st.file_uploader(
                "Pilih gambar kondisi kulit (JPG/PNG)",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
            if uploaded_file is not None:
                st.image(uploaded_file, caption="Gambar yang diunggah", use_container_width=True)
            else:
                md("""
                <div class="info-card" style="text-align:center; color:#8492A6;">
                    <div style="font-size:2rem;">🖼️</div>
                    Belum ada gambar diunggah.<br>
                    Format yang didukung: JPG, JPEG, PNG
                </div>
                """)

        with col_result:
            st.markdown("#### 🔍 Hasil Prediksi")
            if uploaded_file is not None:
                with st.spinner("Menganalisis gambar..."):
                    img = Image.open(uploaded_file).convert("RGB")
                    img = img.resize((224, 224), Image.NEAREST)
                    img_array = np.array(img) / 255.0
                    prediction = predict_with_tta(model, img_array)
                    predicted_idx = int(np.argmax(prediction))
                    predicted_class = resolve_disease_name(class_names[str(predicted_idx)])
                    confidence = float(np.max(prediction)) * 100
                    time.sleep(0.3)

                md(f"""
                <div class="result-card">
                    <div class="result-label">Prediksi Kondisi Kulit</div>
                    <div class="result-class">{predicted_class}</div>
                    <span class="confidence-badge conf-neutral">Confidence Score: {confidence:.1f}%</span>
                </div>
                """)

                md(f"<p style='color:#6B7686; margin-top:0.6rem; font-size:0.88rem;'>{CLASS_INFO.get(predicted_class, '')}</p>")

                st.markdown("##### Distribusi Probabilitas Seluruh Kelas")
                sorted_idx = np.argsort(prediction)[::-1]
                for idx in sorted_idx:
                    cname = resolve_disease_name(class_names[str(idx)])
                    prob = float(prediction[idx]) * 100
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.progress(min(int(prob), 100), text=cname)
                    with c2:
                        md(f"<div style='text-align:right; padding-top:6px; font-size:0.88rem;'>{prob:.1f}%</div>")

                def _norm_nama(s):
                    return s.lower().replace("_", " ").replace("-", " ").strip()

                matched_key = next(
                    (k for k in DISEASE_DETAILS if _norm_nama(k) == _norm_nama(predicted_class)),
                    None
                )
                if matched_key:
                    detail = DISEASE_DETAILS[matched_key]
                    urgensi = URGENSI_STYLE.get(detail["urgensi"], URGENSI_STYLE["rendah"])

                    # Nomor tiket dibuat sekali per sesi pemeriksaan dan disimpan di
                    # session_state -- bukan dihitung ulang dari hash nama+ukuran file
                    # setiap rerun. Sebelumnya nomor tiket murni hasil hash file+kelas,
                    # sehingga gambar yang sama diproses di sesi/browser berbeda selalu
                    # menghasilkan nomor tiket identik dan menabrak key widget unduh PDF
                    # di halaman Riwayat Pasien ("multiple elements with the same key").
                    _file_identity = f"{uploaded_file.name}-{getattr(uploaded_file, 'size', 0)}-{matched_key}"
                    if st.session_state.get("_tiket_file_identity") != _file_identity:
                        st.session_state["_tiket_file_identity"] = _file_identity
                        st.session_state["_tiket_hash"] = uuid.uuid4().hex[:6].upper()
                    _tiket_hash = st.session_state["_tiket_hash"]
                    nomor_tiket = f"SKR-{datetime.now().strftime('%Y%m%d')}-{_tiket_hash}"
                    waktu_periksa = datetime.now().strftime("%d %b %Y, %H:%M")
                    _riwayat_key = f"riwayat_tercatat_{nomor_tiket}"
                    _sudah_tersimpan = st.session_state.get(_riwayat_key, False)

                    with st.expander("📝 Data Pasien (isi jika ingin mencetak surat rujukan)", expanded=not _sudah_tersimpan):
                        c_nama, c_usia, c_rm = st.columns([2, 1, 1.3])
                        with c_nama:
                            st.text_input("Nama Pasien", key="pasien_nama", placeholder="Nama lengkap")
                        with c_usia:
                            st.text_input("Usia", key="pasien_usia", placeholder="cth. 34 th")
                        with c_rm:
                            st.text_input("No. RM / NIK", key="pasien_rm", placeholder="Masukan NIK")
                        st.text_input("No. HP", key="pasien_hp", placeholder="cth. 0812xxxxxxx")
                        st.text_input("Keluhan singkat", key="pasien_keluhan", placeholder="cth. gatal sejak 3 hari")

                        # ---- Tombol simpan dipindah ke sini (menyatu dengan form data
                        # pasien) supaya fokus user tetap di satu tempat: isi data lalu
                        # langsung simpan. Sebelumnya tombol ini ada jauh di bawah, setelah
                        # pratinjau surat rujukan -- petugas yang cuma baca/unduh dokumen
                        # lalu langsung pindah ke foto berikutnya sering lupa menekannya,
                        # sehingga datanya tidak tercatat di riwayat.
                        if FPDF_AVAILABLE:
                            try:
                                pdf_bytes = generate_referral_pdf(
                                    nomor_tiket=nomor_tiket,
                                    nama_pasien=st.session_state.get("pasien_nama", ""),
                                    usia=st.session_state.get("pasien_usia", ""),
                                    no_rm=st.session_state.get("pasien_rm", ""),
                                    no_hp=st.session_state.get("pasien_hp", ""),
                                    keluhan=st.session_state.get("pasien_keluhan", ""),
                                    predicted_class=matched_key,
                                    confidence=confidence,
                                    detail=detail,
                                    urgensi_info=urgensi,
                                )
                                pdf_error = None
                            except Exception as pdf_exc:
                                pdf_bytes = None
                                pdf_error = str(pdf_exc)
                        else:
                            pdf_bytes = None
                            pdf_error = None

                        if not _sudah_tersimpan:
                            if st.button("💾 Simpan Data ke Riwayat Pasien", key="simpan_riwayat", use_container_width=True):
                                nama_final = st.session_state.get("pasien_nama", "").strip() or "-"
                                usia_final = st.session_state.get("pasien_usia", "").strip() or "-"
                                rm_final = st.session_state.get("pasien_rm", "").strip() or "-"
                                hp_final = st.session_state.get("pasien_hp", "").strip() or "-"
                                keluhan_final = st.session_state.get("pasien_keluhan", "").strip() or "-"
                                _pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8") if pdf_bytes else None
                                save_riwayat_entry({
                                    "nomor_tiket": nomor_tiket,
                                    "waktu": waktu_periksa,
                                    "username": st.session_state.username,
                                    "nama_pasien": nama_final,
                                    "usia": usia_final,
                                    "no_rm": rm_final,
                                    "no_hp": hp_final,
                                    "keluhan": keluhan_final,
                                    "predicted_class": matched_key,
                                    "confidence": round(confidence, 1),
                                    "kategori": detail["kategori"],
                                    "urgensi_label": urgensi["label"],
                                    "pdf_base64": _pdf_b64,
                                })
                                st.session_state[_riwayat_key] = True
                                st.rerun()

                        if not FPDF_AVAILABLE:
                            st.caption("💡 Unduh PDF perlu library 'fpdf2'. Jalankan: pip install fpdf2")
                        elif pdf_error:
                            st.warning(f"PDF belum bisa dibuat otomatis ({pdf_error}). Pratinjau di bawah tetap bisa dipakai.")

                    if _sudah_tersimpan:
                        st.success(f"✅ Data pasien untuk tiket {nomor_tiket} sudah tersimpan di riwayat.")

                    gejala_html = "".join(f"<li>{g}</li>" for g in detail["gejala"])
                    penanganan_html = "".join(f"<li>{p}</li>" for p in detail["penanganan"])

                    # Dipindah ke kolom kiri (di bawah gambar) supaya ruang kosong
                    # di sana terpakai, dan kolom kanan tidak terlalu panjang.
                    with col_upload:
                        md(f"""
                        <div class="action-card">
                            <div class="action-section-title">🔎 Gejala umum yang biasa menyertai</div>
                            <ul class="action-list">{gejala_html}</ul>
                            <div class="action-section-title">✅ Tindakan awal &amp; penanganan</div>
                            <ul class="action-list">{penanganan_html}</ul>
                        </div>
                        """)

                    if detail["urgensi"] == "tinggi":
                        rujukan_text = (
                            "Kondisi ini termasuk <strong>prioritas tinggi</strong> — sebaiknya segera "
                            "periksakan diri ke Puskesmas atau dokter agar bisa cepat mendapat penanganan "
                            "dan, bila perlu, <strong>rujukan lebih lanjut</strong> ke dokter spesialis kulit."
                        )
                    elif detail["urgensi"] == "sedang":
                        rujukan_text = (
                            "Disarankan periksa ke Puskesmas dalam waktu dekat untuk memastikan diagnosis "
                            "dan mendapat <strong>rujukan</strong> ke dokter bila kondisi tidak membaik."
                        )
                    else:
                        rujukan_text = (
                            "Tetap disarankan periksa ke Puskesmas "
                            "untuk memastikan diagnosis. Minta <strong>surat rujukan</strong> ke dokter kulit "
                            "bila keluhan menetap atau memburuk."
                        )

                    md(f"""
                    <div class="rujukan-box">
                        🏥 <strong>Langkah selanjutnya:</strong> {rujukan_text}
                    </div>
                    """)

                    nama_p = st.session_state.get("pasien_nama", "").strip() or "-"
                    usia_p = st.session_state.get("pasien_usia", "").strip() or "-"
                    rm_p = st.session_state.get("pasien_rm", "").strip() or "-"
                    keluhan_p = st.session_state.get("pasien_keluhan", "").strip() or "-"

                    # ---- Pratinjau ringkas gaya kwitansi (bukan surat 1 halaman lagi) ----
                    with st.expander("🧾 Lihat Pratinjau Surat Rujukan", expanded=False):
                        md(f"""
                        <div class="kwitansi-box">
                            <div class="kwitansi-header">
                                <div class="kwi-title">PUSKESMAS TAMBUN — SKRINING KULIT</div>
                                <div class="kwi-sub">Keterangan Hasil Skrining &amp; Rujukan</div>
                            </div>
                            <div class="kwi-meta">
                                <span>No. {nomor_tiket}</span>
                                <span>{waktu_periksa}</span>
                            </div>

                            <div class="kwi-row"><span class="kwi-label">Nama</span><span class="kwi-value">{nama_p}</span></div>
                            <div class="kwi-row"><span class="kwi-label">Usia</span><span class="kwi-value">{usia_p}</span></div>
                            <div class="kwi-row"><span class="kwi-label">No. RM/NIK</span><span class="kwi-value">{rm_p}</span></div>
                            <div class="kwi-row"><span class="kwi-label">Keluhan</span><span class="kwi-value">{keluhan_p}</span></div>

                            <div class="kwi-divider"></div>

                            <div class="kwi-row"><span class="kwi-label">Kondisi</span><span class="kwi-value">{matched_key}</span></div>
                            <div class="kwi-row"><span class="kwi-label">Confidence</span><span class="kwi-value">{confidence:.1f}%</span></div>
                            <div class="kwi-row"><span class="kwi-label">Kategori</span><span class="kwi-value">{detail['kategori']}</span></div>

                            <div class="kwi-divider"></div>

                            <div class="kwi-note">{rujukan_text}</div>

                            <div class="kwi-disclaimer">
                                Luaran model machine learning (skrining awal), bukan diagnosis medis final.
                                Diagnosis pasti tetap melalui pemeriksaan tenaga medis profesional.
                            </div>
                        </div>
                        """)

                    col_btn1, col_btn2, col_btn3 = st.columns([1.2, 1, 1])
                    with col_btn1:
                        if st.button(f"📖 Lihat halaman lengkap {matched_key}", key="jump_to_info", use_container_width=True):
                            st.session_state.nav_l1 = "📖 Informasi Penyakit"
                            st.session_state.nav_l2_informasi = detail["kategori"]
                            st.session_state.nav_l3_informasi = matched_key
                            st.rerun()

                    if "show_pdf_preview" not in st.session_state:
                        st.session_state.show_pdf_preview = False

                    with col_btn2:
                        if st.button("👁️ Tampilkan PDF", key="toggle_pdf_preview", use_container_width=True,
                                     disabled=pdf_bytes is None):
                            st.session_state.show_pdf_preview = not st.session_state.show_pdf_preview

                    with col_btn3:
                        if pdf_bytes is not None:
                            st.download_button(
                                "⬇️ Unduh PDF",
                                data=pdf_bytes,
                                file_name=f"surat_rujukan_{nomor_tiket}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                        else:
                            st.button("⬇️ Unduh PDF", disabled=True, use_container_width=True)

                    if pdf_bytes is not None and st.session_state.show_pdf_preview:
                        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
                        md(f"""
                        <div style="margin-top:0.6rem; border:1px solid #D3E9F6; border-radius:10px; overflow:hidden;">
                            <iframe src="data:application/pdf;base64,{b64_pdf}"
                                    width="100%" height="480px" style="border:none; display:block;">
                            </iframe>
                        </div>
                        """)
                        st.caption("Pratinjau ditampilkan langsung dari browser. Jika tidak muncul, gunakan tombol Unduh PDF di atas.")

                    if FPDF_AVAILABLE and not st.session_state.get("pasien_nama", "").strip():
                        st.caption("💡 Isi nama pasien di bagian \"Data Pasien\" di atas supaya nama tercetak di surat rujukan.")
                else:
                    st.caption(
                        f"ℹ️ Detail penanganan & surat rujukan belum tersedia untuk kelas '{predicted_class}' "
                        "(nama kelas belum terdaftar di data DISEASE_DETAILS)."
                    )

                md("""
                <div class="disclaimer">
                    ⚠️ <strong>Disclaimer:</strong> Hasil prediksi ini adalah luaran model machine learning
                    dan bersifat sebagai alat skrining awal, <u>bukan diagnosis medis final</u>.
                    Konsultasikan dengan tenaga medis profesional untuk pemeriksaan lebih lanjut.
                </div>
                """)
            else:
                md("""
                <div class="info-card" style="text-align:center; color:#8492A6; padding: 2rem 1.2rem;">
                    <div style="font-size:2rem;">⏳</div>
                    Hasil prediksi akan muncul di sini<br>setelah gambar diunggah.
                </div>
                """)

    # ============================================================
    # KONTEN: INFORMASI PENYAKIT
    # ============================================================
    elif current_l1 == "📖 Informasi Penyakit":
        pilihan = st.session_state.nav_l3_informasi
        detail = DISEASE_DETAILS[pilihan]
        urgensi = URGENSI_STYLE.get(detail["urgensi"], URGENSI_STYLE["rendah"])

        col_foto, col_teks = st.columns([1, 1.4], gap="large")

        with col_foto:
            img_path = os.path.join(INFO_IMAGE_DIR, detail["image_file"])
            img_b64 = get_image_base64(img_path)
            if img_b64:
                md(f'<div class="disease-photo-box"><img src="data:image/jpeg;base64,{img_b64}" style="width:100%; display:block;"></div>')
            else:
                md(f"""
                <div class="disease-photo-placeholder">
                    <div style="font-size:2rem;">🖼️</div>
                    Foto referensi belum tersedia.<br>
                    <span style="font-size:0.75rem;">
                    Simpan foto dengan nama <code>{detail['image_file']}</code><br>
                    di folder <code>{INFO_IMAGE_DIR}/</code>
                    </span>
                </div>
                """)

            st.caption(f"Kategori: {detail['kategori']}")

        with col_teks:
            st.markdown(f"### {pilihan}")
            md('<div class="section-title">📝 Deskripsi</div>')
            st.write(detail["deskripsi"])
            md('<div class="section-title">🔎 Gejala Umum</div>')
            for g in detail["gejala"]:
                st.markdown(f"- {g}")
            md('<div class="section-title">✅ Saran Penanganan</div>')
            for p in detail["penanganan"]:
                st.markdown(f"- {p}")

        md("""
        <div class="disclaimer">
            ⚠️ <strong>Disclaimer:</strong> Informasi di halaman ini bersifat edukasi kesehatan umum
            dan <u>bukan pengganti pemeriksaan serta resep dari tenaga medis</u>. Untuk diagnosis
            dan pengobatan yang pasti, silakan periksakan diri ke Puskesmas atau fasilitas kesehatan terdekat.
        </div>
        """)

    # ============================================================
    # KONTEN: TENTANG APLIKASI
    # ============================================================
    elif current_l1 == "ℹ️ Tentang Aplikasi":
        sub = st.session_state.nav_l2_tentang

        if sub == "Tentang Aplikasi":
            st.markdown("#### 🩺 Tentang Aplikasi")
            st.markdown("""
            Aplikasi ini membantu klasifikasi awal jenis penyakit kulit dari citra digital
            menggunakan model **MobileNetV2** dengan pendekatan *transfer learning*.
            Dikembangkan sebagai skrining awal dan membuat surat rujukan ke bagian Administrasi Puskesmas Tambun untuk ditindak lebih lanjut.
            """)

        elif sub == "Kelas yang Dikenali":
            st.markdown("#### 📋 Kelas yang Dikenali")
            for cname, desc in CLASS_INFO.items():
                st.markdown(f"**{cname}**")
                st.caption(desc)

        elif sub == "Catatan Penting":
            st.markdown("#### ⚠️ Catatan Penting")
            md("""
            <div class="disclaimer">
                Hasil prediksi bersifat pendukung keputusan awal dan tidak menggantikan
                diagnosis tenaga medis profesional. Untuk kepastian diagnosis dan
                penanganan, pasien tetap perlu diperiksa langsung oleh dokter atau
                petugas kesehatan di Puskesmas / fasilitas kesehatan terdekat.
            </div>
            """)

    # ============================================================
    # KONTEN: AKUN
    # ============================================================
    elif current_l1 == "👤 Akun":
        sub = st.session_state.nav_l2_akun

        if sub == "Profil Saya":
            st.markdown("#### 👤 Profil Saya")
            users_saya = load_users()
            data_saya = users_saya.get(st.session_state.username, {})

            _role_saya = get_user_role(st.session_state.username, users_saya)
            if _role_saya == "admin":
                _status_html = '<span style="background-color:#EAF4FB; color:#204C64; padding:0.2rem 0.6rem; border-radius:6px; font-size:0.78rem; font-weight:600;">🛡️ Administrator</span>'
            elif _role_saya == "petugas":
                _status_html = '<span style="background-color:#EAF4FB; color:#204C64; padding:0.2rem 0.6rem; border-radius:6px; font-size:0.78rem; font-weight:600;">🩺 Petugas</span>'
            elif _role_saya == "pending_petugas":
                _status_html = '<span style="background-color:#FFF4E0; color:#8A5A00; padding:0.2rem 0.6rem; border-radius:6px; font-size:0.78rem; font-weight:600;">⏳ Menunggu Persetujuan Petugas</span>'
            else:
                _status_html = '<span style="background-color:#EAF4FB; color:#204C64; padding:0.2rem 0.6rem; border-radius:6px; font-size:0.78rem; font-weight:600;">👤 User</span>'

            _nama_tampil = data_saya.get("nama_lengkap") or "-"
            _tgl_tampil = data_saya.get("tanggal_lahir") or "-"
            _email_tampil = data_saya.get("email") or "-"

            md(f"""
            <div class="info-card">
                Masuk sebagai akun: <strong>{st.session_state.username}</strong><br><br>
                {_status_html}<br><br>
                <strong>Nama Lengkap:</strong> {_nama_tampil}<br>
                <strong>Tanggal Lahir:</strong> {_tgl_tampil}<br>
                <strong>Email:</strong> {_email_tampil}
            </div>
            """)

            if "edit_profil_saya" not in st.session_state:
                st.session_state.edit_profil_saya = False

            if st.button("✏️ Edit Profil", key="toggle_edit_profil", use_container_width=True):
                st.session_state.edit_profil_saya = not st.session_state.edit_profil_saya
                st.rerun()

            if st.session_state.edit_profil_saya:
                _tgl_lahir_awal = None
                if data_saya.get("tanggal_lahir"):
                    try:
                        _tgl_lahir_awal = datetime.strptime(data_saya["tanggal_lahir"], "%Y-%m-%d").date()
                    except Exception:
                        _tgl_lahir_awal = None

                with st.form(key="form_edit_profil"):
                    st.caption("Semua kolom di bawah ini opsional -- boleh dikosongkan.")
                    p_nama = st.text_input("Nama Lengkap", value=data_saya.get("nama_lengkap", ""), key="p_nama_lengkap")
                    p_tgl = st.date_input(
                        "Tanggal Lahir", value=_tgl_lahir_awal,
                        min_value=datetime(1940, 1, 1).date(), max_value=datetime.now().date(),
                        key="p_tanggal_lahir",
                    )
                    p_email = st.text_input("Email", value=data_saya.get("email", ""), key="p_email", placeholder="nama@contoh.com")
                    simpan_profil = st.form_submit_button("💾 Simpan Profil", use_container_width=True)

                if simpan_profil:
                    if p_email.strip() and not is_valid_email(p_email):
                        st.error("Format email tidak valid. Contoh yang benar: nama@contoh.com")
                    else:
                        users_now = load_users()
                        if st.session_state.username in users_now:
                            users_now[st.session_state.username]["nama_lengkap"] = p_nama.strip()
                            users_now[st.session_state.username]["tanggal_lahir"] = p_tgl.strftime("%Y-%m-%d") if p_tgl else ""
                            users_now[st.session_state.username]["email"] = p_email.strip()
                            save_users(users_now)
                            st.session_state.edit_profil_saya = False
                            st.success("✅ Profil berhasil diperbarui.")
                            st.rerun()

        elif sub == "Permintaan Petugas":
            if get_user_role(st.session_state.username) != "admin":
                st.error("Halaman ini hanya untuk akun Administrator.")
                st.stop()

            st.markdown("#### 🩺 Permintaan Status Petugas")
            st.caption("Hanya akun Administrator yang bisa menyetujui atau menolak permintaan ini.")

            users_all = load_users()
            pending_list = [u for u, rec in users_all.items() if rec["role"] == "pending_petugas"]

            if not pending_list:
                st.info("Tidak ada permintaan Petugas yang menunggu persetujuan.")
            else:
                for _uname in pending_list:
                    with st.container(border=True):
                        st.markdown(f"**{_uname}**")
                        st.caption("Meminta akses Petugas (verifikasi & edit semua riwayat rujukan yang masuk).")
                        c_setuju, c_tolak = st.columns(2)
                        with c_setuju:
                            if st.button("✅ Setujui", key=f"approve_{_uname}", use_container_width=True):
                                users_now = load_users()
                                if _uname in users_now:
                                    users_now[_uname]["role"] = "petugas"
                                    save_users(users_now)
                                    st.success(f"{_uname} sekarang menjadi Petugas.")
                                    st.rerun()
                        with c_tolak:
                            if st.button("❌ Tolak", key=f"reject_{_uname}", use_container_width=True):
                                users_now = load_users()
                                if _uname in users_now:
                                    users_now[_uname]["role"] = "user"
                                    save_users(users_now)
                                    st.info(f"Permintaan dari {_uname} ditolak. Akun tetap aktif sebagai User biasa.")
                                    st.rerun()

        elif sub == "Daftar Petugas":
            if get_user_role(st.session_state.username) != "admin":
                st.error("Halaman ini hanya untuk akun Administrator.")
                st.stop()

            st.markdown("#### 📋 Daftar Petugas")
            st.caption("Hanya menampilkan nama/username dan email -- bukan data diri lain seperti tanggal lahir.")

            users_all = load_users()
            daftar_petugas = sorted(
                [(u, rec) for u, rec in users_all.items() if rec.get("role") == "petugas"],
                key=lambda pair: pair[0].lower(),
            )

            if not daftar_petugas:
                st.info("Belum ada akun dengan status Petugas.")
            else:
                for _uname, _rec in daftar_petugas:
                    _nama_p = _rec.get("nama_lengkap") or "-"
                    _email_p = _rec.get("email") or "-"
                    with st.container(border=True):
                        st.markdown(f"**{_uname}**" + (f" · {_nama_p}" if _nama_p != "-" else ""))
                        st.caption(f"✉️ {_email_p}")

        elif sub == "Riwayat Pasien":
            _role_riwayat = get_user_role(st.session_state.username)
            # Admin & Petugas jadi verifikator: melihat & mengedit SEMUA riwayat
            # rujukan yang masuk dari semua User. User biasa hanya melihat
            # aktivitas rujukan miliknya sendiri, tanpa bisa mengedit.
            can_see_all = _role_riwayat in ("admin", "petugas")
            can_edit = _role_riwayat in ("admin", "petugas")
            # Hanya Administrator yang boleh tahu siapa (petugas/user) di balik
            # tiap rujukan -- Petugas mengedit tanpa melihat identitas pengirim.
            can_see_submitter = _role_riwayat == "admin"

            semua_riwayat = load_riwayat()
            # Tandai posisi asli tiap record di file (dipakai untuk edit yang
            # akurat, karena nomor_tiket saja tidak cukup unik untuk data lama).
            for _pos, _r in enumerate(semua_riwayat):
                _r["_row_index"] = _pos
            if can_see_all:
                st.markdown("#### 🗂️ Riwayat Rujukan (Semua)")
                if _role_riwayat == "admin":
                    md("""<span style="background-color:#EAF4FB; color:#204C64; padding:0.2rem 0.6rem; border-radius:6px; font-size:0.78rem; font-weight:600;">🛡️ Mode Administrator</span>""")
                else:
                    md("""<span style="background-color:#EAF4FB; color:#204C64; padding:0.2rem 0.6rem; border-radius:6px; font-size:0.78rem; font-weight:600;">🩺 Mode Petugas · Verifikator</span>""")
                riwayat_saya = list(semua_riwayat)
            else:
                st.markdown("#### 🗂️ Riwayat Rujukan Saya")
                st.caption("Rujukan yang sudah kamu buat/unggah. Hubungi Petugas untuk mengubah data yang sudah tersimpan.")
                riwayat_saya = [r for r in semua_riwayat if r.get("username") == st.session_state.username]

            riwayat_saya = sorted(riwayat_saya, key=lambda r: r.get("nomor_tiket", ""), reverse=True)

            if not riwayat_saya:
                st.info("Belum ada riwayat pemeriksaan pasien.")
            else:
                cari = st.text_input("🔎 Cari nama pasien", key="cari_riwayat", placeholder="Ketik nama pasien...")
                if cari.strip():
                    riwayat_saya = [r for r in riwayat_saya if cari.strip().lower() in r.get("nama_pasien", "").lower()]

                if can_see_submitter:
                    petugas_list = sorted(set(r.get("username", "-") for r in riwayat_saya))
                    filter_petugas = st.selectbox("Filter pengirim", ["Semua"] + petugas_list, key="filter_petugas_riwayat")
                    if filter_petugas != "Semua":
                        riwayat_saya = [r for r in riwayat_saya if r.get("username") == filter_petugas]

                st.caption(f"Menampilkan {len(riwayat_saya)} riwayat")

                for _idx, r in enumerate(riwayat_saya):
                    label = f"{r['waktu']} — {r['nama_pasien']} — {r['predicted_class']} ({r['confidence']}%)"
                    if can_see_submitter:
                        label += f" — 👤 {r.get('username', '-')}"
                    with st.expander(label, expanded=False):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**No. Tiket:** {r['nomor_tiket']}")
                            st.markdown(f"**Nama:** {r['nama_pasien']}")
                            st.markdown(f"**Usia:** {r['usia']}")
                            st.markdown(f"**No. RM/NIK:** {r['no_rm']}")
                            st.markdown(f"**No. HP:** {r.get('no_hp', '-')}")
                        with c2:
                            st.markdown(f"**Kondisi:** {r['predicted_class']}")
                            st.markdown(f"**Kategori:** {r['kategori']}")
                            st.markdown(f"**Confidence:** {r['confidence']}%")
                        st.markdown(f"**Keluhan:** {r['keluhan']}")

                        if r.get("pdf_base64"):
                            pdf_bytes_r = base64.b64decode(r["pdf_base64"])
                            st.download_button(
                                "⬇️ Unduh Ulang PDF",
                                data=pdf_bytes_r,
                                file_name=f"surat_rujukan_{r['nomor_tiket']}.pdf",
                                mime="application/pdf",
                                # Disertai indeks baris supaya tetap unik walau ada
                                # data lama dengan nomor tiket yang kebetulan sama
                                # (peninggalan bug nomor tiket sebelum diperbaiki).
                                key=f"dl_{_idx}_{r['nomor_tiket']}",
                                use_container_width=True,
                            )
                        else:
                            st.caption("PDF tidak tersedia untuk riwayat ini.")

                        # ---- Edit data pasien pada record yang sudah tersimpan ----
                        # Hanya Admin & Petugas (verifikator) yang boleh mengedit;
                        # User hanya boleh melihat & mengunduh punya sendiri.
                        if not can_edit:
                            continue

                        _edit_flag_key = f"edit_mode_{r['_row_index']}"
                        if _edit_flag_key not in st.session_state:
                            st.session_state[_edit_flag_key] = False

                        if st.button("✏️ Edit Data Pasien", key=f"toggle_edit_{r['_row_index']}", use_container_width=True):
                            st.session_state[_edit_flag_key] = not st.session_state[_edit_flag_key]
                            st.rerun()

                        if st.session_state[_edit_flag_key]:
                            with st.form(key=f"form_edit_{r['_row_index']}"):
                                st.caption("Ubah data di bawah ini, lalu simpan.")
                                e_nama = st.text_input("Nama Pasien", value=r.get("nama_pasien", "-"), key=f"e_nama_{r['_row_index']}")
                                e_usia = st.text_input("Usia", value=r.get("usia", "-"), key=f"e_usia_{r['_row_index']}")
                                e_rm = st.text_input("No. RM / NIK", value=r.get("no_rm", "-"), key=f"e_rm_{r['_row_index']}")
                                e_hp = st.text_input("No. HP", value=r.get("no_hp", "-"), key=f"e_hp_{r['_row_index']}")
                                e_keluhan = st.text_area("Keluhan", value=r.get("keluhan", "-"), key=f"e_keluhan_{r['_row_index']}")
                                simpan_edit = st.form_submit_button("💾 Simpan Perubahan", use_container_width=True)

                            if simpan_edit:
                                updates = {
                                    "nama_pasien": e_nama.strip() or "-",
                                    "usia": e_usia.strip() or "-",
                                    "no_rm": e_rm.strip() or "-",
                                    "no_hp": e_hp.strip() or "-",
                                    "keluhan": e_keluhan.strip() or "-",
                                }

                                # Buat ulang PDF-nya juga supaya suratnya ikut
                                # mencerminkan data terbaru, bukan cuma data di
                                # halaman riwayat. Record hasil rujukan manual
                                # (tanpa foto) dan hasil deteksi AI dibuat ulang
                                # lewat fungsi PDF yang berbeda.
                                if FPDF_AVAILABLE:
                                    if r.get("kategori") == "Rujukan Manual":
                                        try:
                                            pdf_bytes_baru = generate_referral_pdf_manual(
                                                nomor_tiket=r["nomor_tiket"],
                                                nama_pasien=updates["nama_pasien"],
                                                usia=updates["usia"],
                                                no_rm=updates["no_rm"],
                                                no_hp=updates["no_hp"],
                                                keluhan=updates["keluhan"],
                                                kondisi_awal=r.get("predicted_class", ""),
                                                urgensi_info={"label": r.get("urgensi_label", "Menunggu penilaian tenaga medis")},
                                                catatan_tambahan="",
                                                petugas=r.get("username", "-"),
                                            )
                                            updates["pdf_base64"] = base64.b64encode(pdf_bytes_baru).decode("utf-8")
                                        except Exception:
                                            pass  # data pasien tetap diperbarui walau PDF gagal dibuat ulang
                                    else:
                                        detail_r = DISEASE_DETAILS.get(r["predicted_class"])
                                        if detail_r:
                                            urgensi_r = URGENSI_STYLE.get(detail_r["urgensi"], URGENSI_STYLE["rendah"])
                                            try:
                                                pdf_bytes_baru = generate_referral_pdf(
                                                    nomor_tiket=r["nomor_tiket"],
                                                    nama_pasien=updates["nama_pasien"],
                                                    usia=updates["usia"],
                                                    no_rm=updates["no_rm"],
                                                    no_hp=updates["no_hp"],
                                                    keluhan=updates["keluhan"],
                                                    predicted_class=r["predicted_class"],
                                                    confidence=r["confidence"],
                                                    detail=detail_r,
                                                    urgensi_info=urgensi_r,
                                                )
                                                updates["pdf_base64"] = base64.b64encode(pdf_bytes_baru).decode("utf-8")
                                            except Exception:
                                                pass  # data pasien tetap diperbarui walau PDF gagal dibuat ulang

                                if update_riwayat_entry_by_index(r["_row_index"], updates):
                                    st.session_state[_edit_flag_key] = False
                                    st.success("✅ Data pasien berhasil diperbarui.")
                                    st.rerun()
                                else:
                                    st.error("Gagal menyimpan perubahan. Coba lagi.")

        elif sub == "Evaluasi Bulanan":
            if get_user_role(st.session_state.username) not in ("admin", "petugas"):
                st.error("Halaman ini hanya untuk akun Administrator dan Petugas.")
                st.stop()

            st.markdown("#### 📊 Evaluasi Bulanan")
            st.caption(
                "Rekap statistik skrining per bulan. Bersifat anonim/agregat -- tidak "
                "menampilkan nama pasien maupun identitas petugas, demi privasi pasien "
                "yang datanya diinput."
            )

            semua_riwayat_eval = load_riwayat()
            # Pasangkan tiap record dengan tanggal hasil parsing, dan buang
            # yang formatnya tidak terbaca (data lama yang rusak/kosong).
            berpasangan = []
            for r in semua_riwayat_eval:
                tgl = parse_waktu_riwayat(r.get("waktu", ""))
                if tgl is not None:
                    berpasangan.append((tgl, r))

            # Dropdown Bulan & Tahun SELALU tampil, tidak peduli riwayat masih
            # kosong atau tidak -- supaya tetap bisa mengecek bulan mana pun dan
            # selalu dapat jawaban yang jelas (bukan malah kontrolnya hilang
            # begitu saja saat data belum ada). Tahun berjalan selalu ikut jadi
            # pilihan walau belum ada satu pun data di tahun itu.
            tahun_sekarang = datetime.now().year
            # Selain tahun yang benar-benar ada datanya, sertakan juga beberapa
            # tahun ke belakang & tahun depan supaya dropdown tidak cuma
            # menampilkan tahun berjalan (bisa dipakai cek bulan di tahun lain
            # walau datanya belum ada sama sekali).
            tahun_range = set(range(tahun_sekarang - 5, tahun_sekarang + 2))
            tahun_tersedia = sorted({tgl.year for tgl, _ in berpasangan} | {tahun_sekarang} | tahun_range, reverse=True)

            col_bulan, col_tahun = st.columns(2)
            with col_bulan:
                bulan_pilih = st.selectbox("Bulan", BULAN_INDO, index=datetime.now().month - 1, key="eval_bulan")
            with col_tahun:
                tahun_pilih = st.selectbox("Tahun", tahun_tersedia, key="eval_tahun")
            bulan_idx = BULAN_INDO.index(bulan_pilih) + 1

            pairs_bulan_ini = [(tgl, r) for tgl, r in berpasangan if tgl.year == tahun_pilih and tgl.month == bulan_idx]
            data_bulan_ini = [r for _tgl, r in pairs_bulan_ini]

            st.markdown(f"##### Rekap {bulan_pilih} {tahun_pilih}")

            if not data_bulan_ini:
                st.warning(f"Tidak ada data pemeriksaan pada {bulan_pilih} {tahun_pilih}.")
            else:
                # Hanya dihitung agregat per kelas penyakit -- sengaja TIDAK
                # dipecah per petugas maupun per pasien/record supaya rekap ini
                # tetap anonim. Urgensi juga tidak direkap di sini karena itu
                # wewenang penilaian tenaga medis, bukan statistik aplikasi.
                per_kelas = {}
                for r in data_bulan_ini:
                    _k = r.get("predicted_class", "-")
                    per_kelas[_k] = per_kelas.get(_k, 0) + 1

                c1, c2 = st.columns(2)
                c1.metric("Total Pemeriksaan", len(data_bulan_ini))
                c2.metric("Jenis Kelas Terdeteksi", len(per_kelas))

                st.markdown("**Jumlah per Kelas Penyakit**")
                for _k, _n in sorted(per_kelas.items(), key=lambda x: -x[1]):
                    st.markdown(f"- {_k} — {_n} kasus")

                # ---- Sebaran per tanggal (tetap anonim: hanya tanggal + kelas +
                # jumlah, tidak ada nama pasien atau username petugas) ----
                per_tanggal = {}
                for _tgl, r in pairs_bulan_ini:
                    _hari = _tgl.day
                    _k = r.get("predicted_class", "-")
                    per_tanggal.setdefault(_hari, {})
                    per_tanggal[_hari][_k] = per_tanggal[_hari].get(_k, 0) + 1

                st.markdown("---")
                st.markdown("**Sebaran Kasus per Tanggal**")

                total_per_hari = {
                    f"{_hari:02d}": sum(_kelas_hari.values())
                    for _hari, _kelas_hari in sorted(per_tanggal.items())
                }
                st.bar_chart(total_per_hari)

                for _hari in sorted(per_tanggal.keys()):
                    _kelas_hari = per_tanggal[_hari]
                    _total_hari = sum(_kelas_hari.values())
                    _rincian = ", ".join(
                        f"{k} ({n})" for k, n in sorted(_kelas_hari.items(), key=lambda x: -x[1])
                    )
                    st.markdown(f"- **{_hari} {bulan_pilih} {tahun_pilih}** — {_total_hari} kasus: {_rincian}")

                # ---- Unduh rekap statistik (agregat, anonim) sebagai CSV ----
                _buf = io.StringIO()
                _writer = csv.writer(_buf)
                _writer.writerow(["Bulan", "Tahun", "Kelas Terdeteksi", "Jumlah Kasus"])
                for _k, _n in sorted(per_kelas.items(), key=lambda x: -x[1]):
                    _writer.writerow([bulan_pilih, tahun_pilih, _k, _n])
                st.download_button(
                    "⬇️ Unduh Rekap Statistik Bulan Ini (CSV)",
                    data=_buf.getvalue(),
                    file_name=f"rekap_statistik_{bulan_pilih.lower()}_{tahun_pilih}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                # ---- Unduh sebaran per tanggal (agregat, anonim) sebagai CSV ----
                _buf_tgl = io.StringIO()
                _writer_tgl = csv.writer(_buf_tgl)
                _writer_tgl.writerow(["Tanggal", "Bulan", "Tahun", "Kelas Terdeteksi", "Jumlah Kasus"])
                for _hari in sorted(per_tanggal.keys()):
                    for _k, _n in sorted(per_tanggal[_hari].items(), key=lambda x: -x[1]):
                        _writer_tgl.writerow([_hari, bulan_pilih, tahun_pilih, _k, _n])
                st.download_button(
                    "⬇️ Unduh Sebaran per Tanggal (CSV)",
                    data=_buf_tgl.getvalue(),
                    file_name=f"rekap_per_tanggal_{bulan_pilih.lower()}_{tahun_pilih}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        elif sub == "Hapus Akun":
            st.markdown("#### 🗑️ Hapus Akun")
            if not st.session_state.confirm_delete:
                st.warning("Tindakan ini akan menghapus akunmu secara permanen dari aplikasi.")
                if st.button("Hapus Akun Saya", key="ask_delete"):
                    st.session_state.confirm_delete = True
                    st.rerun()
            else:
                st.error(f"Yakin ingin menghapus akun **{st.session_state.username}**? Tindakan ini tidak bisa dibatalkan.")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("Ya, Hapus Permanen", key="confirm_delete_yes", use_container_width=True):
                        users_now = load_users()
                        users_now.pop(st.session_state.username, None)
                        save_users(users_now)
                        st.session_state.authenticated = False
                        st.session_state.username = None
                        st.session_state.confirm_delete = False
                        try:
                            st.query_params.clear()
                        except Exception:
                            pass
                        st.rerun()
                with col_no:
                    if st.button("Batal", key="confirm_delete_no", use_container_width=True):
                        st.session_state.confirm_delete = False
                        st.rerun()

except Exception as e:
    st.error(f"Terjadi kesalahan saat menampilkan halaman ini: {e}")
    st.caption("Coba klik menu lain, lalu kembali lagi ke halaman ini. Jika masih error, cek terminal untuk detail lengkap.")
# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Aplikasi Skrining Awal Penyakit Kulit | Puskesmas Tambun")
