import os as _os

# ── Crash prevention — MUST be before ANY other imports ──
# On Windows under uvicorn, the `multiprocess` package (used by
# HuggingFace datasets/sentence-transformers) starts a ResourceTracker
# subprocess.  Its __del__ destructor calls _stop() during GC, which
# kills the parent uvicorn process with a silent exit(1).
# Monkey-patching __del__ to a no-op prevents this.
_os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_os.environ["OMP_NUM_THREADS"] = "1"
_os.environ["MKL_NUM_THREADS"] = "1"
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
_os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import multiprocess.resource_tracker as _mrt
    _mrt.ResourceTracker.__del__ = lambda self: None
except Exception:
    pass

# Force torch to grab OpenMP first, with 1 thread to avoid contention.
import torch as _torch
_torch.set_num_threads(1)

# Pre-import sentence_transformers BEFORE other libraries (SQLAlchemy,
# socketio, etc.) to avoid a silent exit(1) crash on Windows caused by
# import-order-dependent DLL conflicts.
import sentence_transformers as _st  # noqa: F401

import sys
import mimetypes

# Fix MIME type for .js files on Windows
mimetypes.init()
mimetypes.add_type('application/javascript', '.js', True)
mimetypes.add_type('application/javascript', '.mjs', True)

# Ensure this directory is on the path for sibling module imports
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
import socketio
import os
import uvicorn
from email_validator import EmailNotValidError, validate_email
from threading import Lock
import time
import re

import base64
import requests

# ── Provider abstraction (LLM / TTS / STT) ────────────────────
# Concrete vendor calls live behind the providers package. Gemini is the
# default for all three; offline fallbacks (Ollama / pyttsx3 / Whisper) kick
# in automatically when a cloud key is missing or a call fails. See
# providers/__init__.py and .env.example.
from providers import (
    ProviderError,
    get_llm_provider,
    get_tts_provider,
    get_stt_provider,
)


def call_gemini_llm(prompt: str, temperature: float = 0.0) -> str:
    """Backwards-compatible name: now routes through the configured LLM provider chain."""
    try:
        return get_llm_provider().generate(prompt, temperature)
    except ProviderError as e:
        print(f"[LLM Error] {e}")
        return ""


def call_gemini_tts(text: str, language: str = "arabic") -> bytes:
    """Backwards-compatible name: routes through the configured TTS provider chain."""
    return get_tts_provider().synthesize(text, language)


def call_gemini_stt(audio_bytes: bytes, mime_type: str = "audio/wav", language: str = "arabic") -> str:
    """Backwards-compatible name: routes through the configured STT provider chain."""
    try:
        return get_stt_provider().transcribe(audio_bytes, mime_type, language)
    except ProviderError as e:
        print(f"[STT Error] {e}")
        return ""


SLOW_REQUEST_MS = float(os.environ.get("SLOW_REQUEST_MS", "5000"))

# ── Local modules ─────────────────────────────────────────────
from database import engine, Base, User, get_db, init_db
from auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    get_password_hash,
    needs_rehash,
    require_auth_rate_limit,
    rotate_refresh_token,
    verify_password,
)
from schemas import UserCreate, UserResponse, Token, TokenRefresh
from asl_service import ASLService, SignDB, ArabicSignDB

# Offload blocking ML/LLM/network calls off the event loop so concurrent
# requests aren't serialized behind one inference (see Phase 5 / LATENCY.md).
from starlette.concurrency import run_in_threadpool
import profiling
from profiling import Timer

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(title="Together — Sign Language AI Platform", version="2.0.0")


@app.on_event("startup")
def _ensure_database_ready():
    """Idempotently ensure the pgvector extension + tables exist.

    Production deployments run `alembic upgrade head` first (the start scripts /
    docker-compose do this); this is a harmless no-op when the schema already
    exists, and keeps a fresh dev database bootable. Failure here must not crash
    boot — the lazy model loaders surface DB problems via /api/health.
    """
    try:
        init_db()
    except Exception as e:
        print(f"[startup] init_db skipped: {e}")


def normalize_and_validate_email(raw_email: str, *, check_deliverability: bool = False) -> str:
    email = validate_email(raw_email, check_deliverability=check_deliverability).normalized
    return email

# ── Global error handler (always returns JSON, never raw HTML) ─
from fastapi import Request as _Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(Exception)
async def generic_exception_handler(request: _Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: _Request, exc: RequestValidationError):
    errors = exc.errors()
    detail = "; ".join(f"{e['loc'][-1]}: {e['msg']}" for e in errors)
    return JSONResponse(status_code=422, content={"detail": detail})

# ── CORS ──────────────────────────────────────────────────────
# allow_origins=["*"] is spec-invalid with allow_credentials=True.
# Default to localhost origins; override via ALLOWED_ORIGINS env var
# (comma-separated list, e.g. "https://example.com,https://www.example.com").
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
_cors_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compress non-trivial responses to reduce transfer time on slower networks.
app.add_middleware(GZipMiddleware, minimum_size=1024)

# ── MIME Fix (Windows) ──────────────────────────────────────────
import mimetypes
mimetypes.init()
mimetypes.add_type('application/javascript', '.js', True)

class ForceJSMimeMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # We only want to process HTTP requests. If it's a websocket (e.g. Socket.IO upgrade), pass through directly.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "").lower()
        started = time.perf_counter()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])

                # Check for X-Process-Time-ms (calculate when response starts)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                headers.append((b"x-process-time-ms", f"{elapsed_ms:.1f}".encode("latin1")))

                # Security headers
                headers.append((b"x-frame-options", b"DENY"))
                headers.append((b"x-content-type-options", b"nosniff"))
                headers.append((b"referrer-policy", b"strict-origin-when-cross-origin"))
                headers.append((b"permissions-policy", b"camera=(), microphone=(), geolocation=()"))

                # Check if it is a .js file
                if path.endswith(".js"):
                    # Remove any existing Content-Type header
                    headers[:] = [h for h in headers if h[0].lower() != b"content-type"]
                    headers.append((b"content-type", b"application/javascript"))

                # Cache versioned/static assets aggressively for faster repeat page loads.
                if path.startswith("/static/"):
                    headers[:] = [h for h in headers if h[0].lower() != b"cache-control"]
                    headers.append((b"cache-control", b"public, max-age=86400"))
            
            await send(message)

        await self.app(scope, receive, send_wrapper)

app.add_middleware(ForceJSMimeMiddleware)

# ── Static files & Templates ──────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ── Socket.IO ─────────────────────────────────────────────────
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins=_cors_origins)
socket_app = socketio.ASGIApp(sio, app)

# ── Deferred model loading ────────────────────────────────────
# All heavy models (SBERT, TFLite, PyTorch) are loaded lazily on
# first use instead of at module-import time.  This avoids the
# silent OpenMP / libiomp5md.dll double-init crash that happens
# on Windows when PyTorch + SBERT are loaded during uvicorn's
# module-import phase.
# ──────────────────────────────────────────────────────────────

_sign_db = None
_sign_db_error = None
_sign_db_last_error_time = 0
_sign_db_lock = Lock()

def get_sign_db():
    global _sign_db, _sign_db_error, _sign_db_last_error_time
    if _sign_db is not None:
        return _sign_db
    
    now = time.time()
    if _sign_db_error is not None:
        if now - _sign_db_last_error_time > 5:
            print("[SignDB] Clearing previous init error to retry...")
            _sign_db_error = None
        else:
            return None

    with _sign_db_lock:
        if _sign_db is not None:
            return _sign_db
        if _sign_db_error is not None:
            return None
        try:
            print("[Lazy] Initializing SignDB...")
            _sign_db = SignDB()
            return _sign_db
        except Exception as e:
            _sign_db_error = str(e)
            _sign_db_last_error_time = now
            print(f"[ERROR] SignDB init failed: {e}")
            return None

# ── Lazy ASL inference engine (for Sign-to-Text & Sign-to-Speech) ──
_asl_engine = None
_asl_engine_error = None
_asl_engine_retry_count = 0
_asl_engine_last_error_time = None
_asl_engine_lock = Lock()

def get_asl_engine():
    """
    Lazy-load ASL engine with retry logic.
    Retries on failure with exponential backoff to handle transient issues.
    """
    global _asl_engine, _asl_engine_error, _asl_engine_retry_count, _asl_engine_last_error_time
    import time as _time_module
    
    # If engine is already loaded, return it
    if _asl_engine is not None:
        return _asl_engine
    
    # If error occurred, check if we should retry (exponential backoff)
    if _asl_engine_error is not None:
        now = _time_module.time()
        retry_delay = min(2 ** _asl_engine_retry_count, 300)  # Cap at 5 minutes
        
        # Don't retry too frequently
        if _asl_engine_last_error_time is not None and (now - _asl_engine_last_error_time) < retry_delay:
            return None
        
        # Time to retry - reset and try again
        print(f"[ASLService] Retrying initialization (attempt {_asl_engine_retry_count + 1})...")
        _asl_engine_error = None
        _asl_engine_retry_count += 1
    
    # Try to initialize (guarded so concurrent requests don't trigger duplicate heavy loads)
    with _asl_engine_lock:
        if _asl_engine is not None:
            return _asl_engine
        if _asl_engine_error is not None:
            return None

        try:
            sign_db = get_sign_db()
            _asl_engine = ASLService(sign_db=sign_db)
            _asl_engine_retry_count = 0  # Reset retry count on success
            print("[ASLService] Successfully initialized.")
            return _asl_engine
        except Exception as e:
            _asl_engine_error = str(e)
            _asl_engine_last_error_time = _time_module.time()
            print(f"[ERROR] ASLService init failed (attempt {_asl_engine_retry_count}): {e}")
            return None

# ── Lazy Arabic Sign DB (separate from English signs.db) ──
_arabic_sign_db = None
_arabic_sign_db_error = None
_arabic_sign_db_last_error_time = 0
_arabic_sign_db_lock = Lock()

def get_arabic_sign_db():
    global _arabic_sign_db, _arabic_sign_db_error, _arabic_sign_db_last_error_time
    if _arabic_sign_db is not None:
        return _arabic_sign_db
    
    now = time.time()
    if _arabic_sign_db_error is not None:
        if now - _arabic_sign_db_last_error_time > 5:
            print("[ArabicSignDB] Clearing previous init error to retry...")
            _arabic_sign_db_error = None
        else:
            return None

    with _arabic_sign_db_lock:
        if _arabic_sign_db is not None:
            return _arabic_sign_db
        if _arabic_sign_db_error is not None:
            return None
        try:
            print("[Lazy] Initializing ArabicSignDB...")
            # Reuse SBERT from SignDB if available
            sign_db = get_sign_db()
            if sign_db is not None:
                _arabic_sign_db = ArabicSignDB(sbert_model=sign_db.model, sbert_util=sign_db.util)
            else:
                _arabic_sign_db = ArabicSignDB()
            return _arabic_sign_db
        except Exception as e:
            _arabic_sign_db_error = str(e)
            _arabic_sign_db_last_error_time = now
            print(f"[ERROR] ArabicSignDB init failed: {e}")
            return None

# ── Lazy Arabic inference engine (PyTorch) ──
_arabic_engine = None
_arabic_engine_error = None
_arabic_engine_last_error_time = 0
_arabic_engine_lock = Lock()

def get_arabic_engine():
    global _arabic_engine, _arabic_engine_error, _arabic_engine_last_error_time
    if _arabic_engine is not None:
        return _arabic_engine
    
    now = time.time()
    if _arabic_engine_error is not None:
        if now - _arabic_engine_last_error_time > 5:
            print("[ArabicPredictor] Clearing previous init error to retry...")
            _arabic_engine_error = None
        else:
            return None

    with _arabic_engine_lock:
        if _arabic_engine is not None:
            return _arabic_engine
        if _arabic_engine_error is not None:
            return None
        try:
            print("[Lazy] Initializing Arabic model...")
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if root_dir not in sys.path:
                sys.path.insert(0, root_dir)
            model_path = os.path.join(root_dir, "models", "best_model.pth")
            class_mapping_path = os.path.join(root_dir, "models", "class_mapping.json")
            from models.sign_predictor import SignLanguagePredictor
            _arabic_engine = SignLanguagePredictor(
                model_path=model_path,
                class_mapping_path=class_mapping_path
            )
            print("[ArabicPredictor] Successfully initialized Arabic model.")
            return _arabic_engine
        except Exception as e:
            _arabic_engine_error = str(e)
            _arabic_engine_last_error_time = now
            print(f"[ERROR] ArabicPredictor init failed: {e}")
            return None

# NOTE: Offline pyttsx3 Arabic TTS now lives in providers/local.py (Pyttsx3TTS)
# and is reached automatically via the TTS provider fallback chain.


@app.get("/api/health")
async def health(warm: bool = False):
    """Quick status check.

    By default this endpoint is non-blocking and does not initialize heavy AI models.
    Pass ?warm=1 to trigger lazy initialization in the background request path.
    """
    global _sign_db, _sign_db_error, _asl_engine, _asl_engine_error, _arabic_engine, _arabic_engine_error

    if warm:
        get_sign_db()
        get_asl_engine()
        get_arabic_engine()

    db_status = "ok" if _sign_db is not None else ("error" if _sign_db_error else "not_loaded")
    engine_status = "ok" if _asl_engine is not None else ("error" if _asl_engine_error else "not_loaded")
    arabic_status = "ok" if _arabic_engine is not None else ("error" if _arabic_engine_error else "not_loaded")
    has_error = bool(_sign_db_error or _asl_engine_error or _arabic_engine_error)
    models_ready = db_status == "ok" and engine_status == "ok" and arabic_status == "ok"

    return {
        # Keep /api/health fast by default: lazy, not-yet-loaded models are healthy but not warmed.
        "status": "degraded" if has_error else "ok",
        "models_ready": models_ready,
        "db_status": db_status,
        "engine_status": engine_status,
        "arabic_status": arabic_status,
        "signs_loaded": len(_sign_db.words) if _sign_db is not None else 0,
        "model_signs_loaded": len(_asl_engine.words) if _asl_engine is not None else 0,
        "arabic_signs_loaded": len(_arabic_engine.labels) if _arabic_engine is not None else 0,
        "error": _asl_engine_error or _sign_db_error or _arabic_engine_error or None,
        # Configured provider chains (no network probe here — just the resolved order).
        "providers": {
            "llm": get_llm_provider().chain_names,
            "tts": get_tts_provider().chain_names,
            "stt": get_stt_provider().chain_names,
        },
    }


# ═══════════════════════════════════════════════════════════════
# PAGE ROUTES
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "landing.html")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request, lang: str = "en"):
    if lang == "ar":
        return templates.TemplateResponse(request, "index_ar.html")
    return templates.TemplateResponse(request, "index.html")

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html")

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse(request, "signup.html")

@app.get("/products/handscript", response_class=HTMLResponse)
async def page_handscript(request: Request):
    return templates.TemplateResponse(request, "products/handscript.html")

@app.get("/products/voicebridge", response_class=HTMLResponse)
async def page_voicebridge(request: Request):
    return templates.TemplateResponse(request, "products/voicebridge.html")

@app.get("/products/signtype", response_class=HTMLResponse)
async def page_signtype(request: Request):
    return templates.TemplateResponse(request, "products/signtype.html")

@app.get("/products/talkside", response_class=HTMLResponse)
async def page_talkside(request: Request):
    return templates.TemplateResponse(request, "products/talkside.html")

@app.get("/products/signlens", response_class=HTMLResponse)
async def page_signlens(request: Request):
    return templates.TemplateResponse(request, "products/signlens.html")

@app.get("/products/signbridge", response_class=HTMLResponse)
async def page_signbridge(request: Request):
    return templates.TemplateResponse(request, "products/signbridge.html")

@app.get("/products/openhands", response_class=HTMLResponse)
async def page_openhands(request: Request):
    return templates.TemplateResponse(request, "products/openhands.html")

@app.get("/languages/asl", response_class=HTMLResponse)
async def page_asl(request: Request):
    return templates.TemplateResponse(request, "languages/asl.html")

@app.get("/languages/arsl", response_class=HTMLResponse)
async def page_arsl(request: Request):
    return templates.TemplateResponse(request, "languages/arsl.html")

@app.get("/resources", response_class=HTMLResponse)
async def page_resources(request: Request):
    return templates.TemplateResponse(request, "resources.html")

@app.get("/about", response_class=HTMLResponse)
async def page_about(request: Request):
    return templates.TemplateResponse(request, "about.html")

@app.get("/partnership", response_class=HTMLResponse)
async def page_partnership(request: Request):
    return templates.TemplateResponse(request, "partnership.html")

@app.get("/contact", response_class=HTMLResponse)
async def page_contact(request: Request):
    return templates.TemplateResponse(request, "contact.html")

@app.get("/demo", response_class=HTMLResponse)
async def page_demo(request: Request):
    return templates.TemplateResponse(request, "demo.html")


# ═══════════════════════════════════════════════════════════════
# AUTH API
# ═══════════════════════════════════════════════════════════════

@app.post("/api/auth/signup", response_model=UserResponse, status_code=201)
async def signup(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    _: None = Depends(require_auth_rate_limit),
):
    raw_email = str(user_data.email).strip().lower()
    password = (user_data.password or "").strip()
    full_name = (user_data.full_name or "").strip() or None
    role = (user_data.role or "Speaker").strip() or "Speaker"

    try:
        email = normalize_and_validate_email(raw_email, check_deliverability=True)
    except EmailNotValidError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please enter a real email address that can receive mail."
        )

    if len(password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long."
        )

    # Check for duplicate email
    existing = db.query(User).filter(func.lower(User.email) == email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists."
        )


    hashed = get_password_hash(password)
    new_user = User(
        full_name=full_name,
        email=email,
        hashed_password=hashed,
        role=role,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.post("/api/auth/login", response_model=Token)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
    _: None = Depends(require_auth_rate_limit),
):
    from fastapi.responses import JSONResponse as _JSONResponse
    raw_email = (form_data.username or "").strip().lower()
    password = (form_data.password or "")

    try:
        email = normalize_and_validate_email(raw_email, check_deliverability=False)
    except EmailNotValidError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please enter a valid email address.",
        )

    user = db.query(User).filter(func.lower(User.email) == email).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Transparently upgrade legacy bcrypt hashes to Argon2id on successful login.
    if needs_rehash(user.hashed_password):
        user.hashed_password = get_password_hash(password)

    access_token = create_access_token(data={
        "sub":       user.email,
        "full_name": user.full_name or "",
        "role":      user.role,
        "user_id":   user.id,
    })

    _, refresh_signed = create_refresh_token(user.id, db)
    db.commit()

    # Deliver the access token in the JSON body (browser stores in localStorage).
    # The refresh token goes in a httpOnly, SameSite=Lax cookie so JS cannot read it,
    # which prevents XSS from stealing the refresh credential.
    is_secure = request.url.scheme == "https"
    response = _JSONResponse(content={"access_token": access_token, "token_type": "bearer"})
    response.set_cookie(
        key="refresh_token",
        value=refresh_signed,
        httponly=True,
        secure=is_secure,
        samesite="lax",
        max_age=REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        path="/api/auth/refresh",
    )
    return response


@app.post("/api/auth/refresh", response_model=TokenRefresh)
async def refresh_token_endpoint(
    request: Request,
    db: Session = Depends(get_db),
):
    """Exchange a valid refresh-token cookie for a fresh access token.

    The old refresh token is revoked on every use (rotation), limiting the
    blast radius of a stolen token to a single request window.
    """
    from fastapi.responses import JSONResponse as _JSONResponse
    raw_rt = request.cookies.get("refresh_token")
    if not raw_rt:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No refresh token cookie.")

    user_id, new_refresh_signed = rotate_refresh_token(raw_rt, db)

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")

    db.commit()

    new_access = create_access_token(data={
        "sub":       user.email,
        "full_name": user.full_name or "",
        "role":      user.role,
        "user_id":   user.id,
    })

    is_secure = request.url.scheme == "https"
    response = _JSONResponse(content={
        "access_token": new_access,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    })
    response.set_cookie(
        key="refresh_token",
        value=new_refresh_signed,
        httponly=True,
        secure=is_secure,
        samesite="lax",
        max_age=REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        path="/api/auth/refresh",
    )
    return response


@app.post("/api/auth/logout")
async def logout(
    request: Request,
    db: Session = Depends(get_db),
):
    """Revoke the refresh token and clear the cookie."""
    from fastapi.responses import JSONResponse as _JSONResponse
    raw_rt = request.cookies.get("refresh_token")
    if raw_rt:
        try:
            from jose import jwt as _jose_jwt
            from auth import SECRET_KEY as _SK, ALGORITHM as _ALG
            payload = _jose_jwt.decode(raw_rt, _SK, algorithms=[_ALG])
            jti = payload.get("jti")
            if jti:
                from db.repository import RefreshTokenRepository
                RefreshTokenRepository(db).revoke(jti)
                db.commit()
        except Exception:
            pass  # Expired / malformed tokens are fine — cookie will be cleared anyway

    response = _JSONResponse(content={"detail": "Logged out."})
    response.delete_cookie(key="refresh_token", path="/api/auth/refresh")
    return response


@app.get("/api/auth/me", response_model=UserResponse)
async def get_me(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Returns the authenticated user's profile."""
    user = db.query(User).filter(User.email == current_user["sub"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return user


# ═══════════════════════════════════════════════════════════════
# ASL ENGINE API
# ═══════════════════════════════════════════════════════════════

from pydantic import BaseModel as _BaseModel
from typing import List as _List, Optional as _Optional

class TranslateRequest(_BaseModel):
    # Rolling window sent from the browser: [N_frames, 543, 3]
    # N can be 1..60; the backend pads/trims to 60 automatically.
    # We use Optional[float] because missing landmarks are sent as null.
    frames: _List[_List[_List[_Optional[float]]]]
    language: _Optional[str] = "english"
    w: _Optional[int] = 640
    h: _Optional[int] = 480

class SentenceRequest(_BaseModel):
    gloss: _List[str]  # list of sign words e.g. ["hello", "world"]
    language: _Optional[str] = "english"

class BatchSignRequest(_BaseModel):
    words: _List[str]

class GlossRequest(_BaseModel):
    sentence: str
    language: _Optional[str] = "english"


def normalize_sign_words(raw_words: _List[str]) -> _List[str]:
    """Normalize input tokens and expand common contractions for sign lookup."""
    normalized = []
    for raw in raw_words:
        if raw is None:
            continue
        cleaned = re.sub(r"[^a-z0-9']", "", str(raw).lower().strip())
        if not cleaned:
            continue

        cleaned_no_apostrophe = cleaned.replace("'", "")
        if cleaned_no_apostrophe == "im":
            normalized.extend(["i", "am"])
            continue

        normalized.append(cleaned_no_apostrophe)
    return normalized


ARABIC_TRANSLATIONS = {
    "baby": "طفل",
    "eat": "أكل",
    "father": "أب",
    "finish": "انتهى",
    "good": "جيد",
    "happy": "سعيد",
    "hear": "يسمع",
    "house": "منزل",
    "important": "مهم",
    "love": "حب",
    "mall": "مول",
    "me": "أنا",
    "mosque": "مسجد",
    "mother": "أم",
    "normal": "طبيعي",
    "sad": "حزين",
    "stop": "توقف",
    "thanks": "شكراً",
    "thinking": "يفكر",
    "worry": "قلق"
}

ARABIC_TO_ENGLISH = {
    "طفل": "baby", "أطفال": "baby", "اطفال": "baby",
    "أكل": "eat", "اكل": "eat", "طعام": "eat", "ياكل": "eat", "يأكل": "eat",
    "أب": "father", "اب": "father", "والد": "father", "بابا": "father",
    "انتهى": "finish", "خلص": "finish", "انتهيت": "finish",
    "جيد": "good", "جميل": "good", "بخير": "good",
    "سعيد": "happy", "فرحان": "happy", "مبسوط": "happy",
    "يسمع": "hear", "سمع": "hear", "يصغي": "hear",
    "منزل": "house", "بيت": "house", "دار": "house",
    "مهم": "important", "ضروري": "important",
    "حب": "love", "يحب": "love", "أحب": "love",
    "مول": "mall", "سوق": "mall", "متجر": "mall",
    "أنا": "me", "انا": "me", "نفسي": "me",
    "مسجد": "mosque", "جامع": "mosque",
    "أم": "mother", "ام": "mother", "والدة": "mother", "ماما": "mother",
    "طبيعي": "normal", "عادي": "normal",
    "حزين": "sad", "زعلان": "sad",
    "توقف": "stop", "قف": "stop", "بس": "stop",
    "شكرا": "thanks", "شكراً": "thanks",
    "يفكر": "thinking", "فكر": "thinking", "تفكير": "thinking",
    "قلق": "worry", "قلقان": "worry", "خوف": "worry"
}

def translate_arabic_to_english(word: str) -> str:
    word_str = str(word).strip()
    if not any('\u0600' <= char <= '\u06FF' for char in word_str):
        return word_str
    
    clean_w = re.sub(r"[.,\/#!$%\^&\*;:{}=\-_`~()]", "", word_str).strip()
    if clean_w in ARABIC_TO_ENGLISH:
        return ARABIC_TO_ENGLISH[clean_w]
        
    try:
        prompt = (
            f"Translate the following Arabic sign language word or short phrase directly to its closest single-word or short-phrase English equivalent. "
            f"Output ONLY the translated English words, without explanation, without quotes, and without punctuation.\n\n"
            f"Arabic: {clean_w}\n"
            f"English:"
        )
        translated = call_gemini_llm(prompt)
        translated = re.sub(r'["\']', '', translated).strip().lower()
        if translated:
            try:
                print(f"[Translate API] Translated Arabic '{clean_w}' to English '{translated}' using Gemini")
            except UnicodeEncodeError:
                print(f"[Translate API] Translated Arabic word to English '{translated}' using Gemini")
            return translated
    except Exception as e:
        try:
            print(f"[Translate API Error] {e}")
        except UnicodeEncodeError:
            print("[Translate API Error] UnicodeEncodeError (Arabic chars in error message)")
    
    return word_str



@app.post("/api/translate")
async def translate_sign(
    body: TranslateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Accepts a rolling window of landmark frames, runs TFLite (ASL) or PyTorch (Arabic) inference."""
    lang = (body.language or "english").lower().strip()
    if lang in ["arabic", "ar", "egyptian", "eg"]:
        engine = get_arabic_engine()
        if engine is None:
            # Try to warm it up; return pending if still loading
            return {"text": "", "confidence": 0.0, "status": "engine_loading"}
        w = body.w if body.w is not None else 640
        h = body.h if body.h is not None else 480
        # Run blocking PyTorch inference in a worker thread to keep the event loop free.
        with Timer("infer.arabic"):
            prediction, confidence = await run_in_threadpool(
                engine.predict_sign_from_landmarks, body.frames, w, h
            )
        if prediction:
            arabic_prediction = ARABIC_TRANSLATIONS.get(prediction.lower(), prediction)
            try:
                print(f"[/api/translate] Detected Arabic: {arabic_prediction} ({confidence:.2f})")
            except UnicodeEncodeError:
                print(f"[/api/translate] Detected Arabic: (safe/transliterated) {prediction} ({confidence:.2f})")
            return {"text": arabic_prediction, "confidence": float(confidence)}
        return {"text": "", "confidence": 0.0}
    else:
        engine = get_asl_engine()
        if engine is None:
            # Try to warm it up; return pending if still loading
            return {"text": "", "confidence": 0.0, "status": "engine_loading"}
        # Run blocking TFLite inference in a worker thread.
        with Timer("infer.asl"):
            prediction, confidence = await run_in_threadpool(engine.predict_sign, body.frames)
        if prediction:
            print(f"[/api/translate] Detected ASL: {prediction} ({confidence:.2f})")
            return {"text": prediction, "confidence": float(confidence)}
        return {"text": "", "confidence": 0.0}


@app.post("/api/translate/sentence")
async def translate_sentence_api(
    body: SentenceRequest,
    current_user: dict = Depends(get_current_user)
):
    """Convert recognized ASL gloss (Topic-Comment) into a natural SVO sentence.

    Uses few-shot gloss→sentence prompting (see gloss.py) through the configured
    LLM provider, with a graceful fallback to the raw gloss when no LLM is
    reachable.
    """
    import gloss as gloss_mod
    lang = (body.language or "english").lower().strip()
    raw_gloss = " ".join(body.gloss)

    # gloss_to_sentence may hit the network LLM — offload it (caching inside it
    # short-circuits repeat gloss, the common case during a conversation).
    with Timer("llm.gloss_to_sentence"):
        sentence = await run_in_threadpool(
            gloss_mod.gloss_to_sentence, body.gloss, lang
        )
    if sentence and sentence != raw_gloss:
        return {"sentence": sentence, "gloss": raw_gloss, "source": "llm"}
    # LLM unavailable / empty: degrade gracefully.
    if lang in ["arabic", "ar", "egyptian", "eg"]:
        return {"sentence": sentence or "", "gloss": raw_gloss, "source": "fallback"}
    return {"sentence": sentence or raw_gloss, "gloss": raw_gloss, "source": "fallback"}


@app.post("/api/gloss")
async def sentence_to_gloss_api(
    body: GlossRequest,
    current_user: dict = Depends(get_current_user)
):
    """Convert a natural SVO sentence into ASL/ArSL gloss (Topic-Comment) + NMM.

    This is the front of the text-to-sign pipeline: the returned gloss tokens
    feed /api/signs/batch for landmark lookup, and `nmm` carries the non-manual
    marker annotations (facial grammar) the avatar layer can render.
    """
    import gloss as gloss_mod
    lang = (body.language or "english").lower().strip()
    with Timer("llm.english_to_gloss"):
        tokens = await run_in_threadpool(gloss_mod.english_to_gloss, body.sentence, lang)
    nmm = gloss_mod.annotate_nmm(tokens, body.sentence)
    return {"gloss": tokens, "nmm": nmm, "sentence": body.sentence}


from fastapi.responses import StreamingResponse
import io

@app.get("/api/tts")
def text_to_speech_endpoint(text: str, language: str = "english"):
    lang = language.lower().strip()
    if lang in ["arabic", "ar", "egyptian", "eg", "english", "en"]:
        try:
            audio_bytes = call_gemini_tts(text, lang)
            return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail=f"TTS language {language} is not supported.")


from fastapi import UploadFile as _UploadFile, File as _File, Form as _Form

@app.post("/api/stt")
async def speech_to_text_endpoint(
    file: _UploadFile = _File(...),
    language: str = _Form("english")
):
    try:
        audio_bytes = await file.read()
        mime_type = file.content_type or "audio/wav"
        # Blocking network transcription -> worker thread.
        with Timer("stt.transcribe"):
            transcription = await run_in_threadpool(
                call_gemini_stt, audio_bytes, mime_type, language
            )
        return {"text": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT transcription failed: {str(e)}")


@app.get("/api/metrics")
async def metrics(current_user: dict = Depends(get_current_user)):
    """Per-stage latency aggregates (count, avg, p50, p95, max, last) in ms.

    Populated by the Timer instrumentation across the inference / LLM / STT
    pipeline. Useful for the before/after profiling in docs/LATENCY.md.
    """
    return {"stages": profiling.snapshot()}


from fastapi.responses import FileResponse
import sqlite3

def get_video_filename(word: str) -> str:
    """Return the stored video filename for an English sign word (from Postgres)."""
    try:
        from db.base import SessionLocal
        from db.repository import SignRepository
        db = SessionLocal()
        try:
            sign = SignRepository(db).get_exact(word, "en")
            if sign and sign.video_filename:
                return os.path.basename(sign.video_filename)
        finally:
            db.close()
    except Exception as e:
        print(f"[ERROR] get_video_filename query failed: {e}")
    return ""

@app.get("/api/videos/{filename}")
def serve_sign_video(filename: str):
    """Serve a sign MP4 video file from the data/signs_videos or root signs_videos directories."""
    safe_name = os.path.basename(filename)
    # 1. Check inside data/signs_videos
    path_data = os.path.join(BASE_DIR, "data", "signs_videos", safe_name)
    if os.path.exists(path_data):
        return FileResponse(path_data, media_type="video/mp4")
    # 2. Check inside root signs_videos
    path_root = os.path.join(BASE_DIR, "signs_videos", safe_name)
    if os.path.exists(path_root):
        return FileResponse(path_root, media_type="video/mp4")
    raise HTTPException(status_code=404, detail=f"Video '{safe_name}' not found.")


# NOTE: /api/signs/lookup MUST be registered before /api/signs/{word}
# otherwise FastAPI's wildcard {word} will capture the literal string "lookup".
@app.get("/api/signs/lookup")
async def lookup_signs(
    words: str,
    current_user: dict = Depends(get_current_user)
):
    """GET /api/signs/lookup?words=hello+world — multi-word lookup for avatar animation."""
    db = get_sign_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Sign database not ready.")

    def _sync():
        # SBERT encode + DB query per word — blocking; runs off the event loop.
        word_list = normalize_sign_words(words.split())
        found, missing = [], []
        for w in word_list:
            lms = db.get_landmarks(w)
            if lms:
                found.append({"word": w, "landmarks": lms, "frame_count": len(lms)})
            else:
                missing.append(w)
        return {"found": found, "missing": missing}

    with Timer("signs.lookup"):
        return await run_in_threadpool(_sync)


@app.get("/api/signs/{word}")
async def get_sign_landmarks(
    word: str,
    current_user: dict = Depends(get_current_user)
):
    """Lookup landmark sequence for a single word (semantic match) and include video_url if available."""
    db = get_sign_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Sign database not ready.")

    def _sync():
        # translate_arabic_to_english may hit the LLM; get_landmarks runs SBERT —
        # both blocking, so the whole lookup runs in a worker thread.
        translated_word = translate_arabic_to_english(word)
        normalized_words = normalize_sign_words([translated_word])
        if not normalized_words:
            raise HTTPException(status_code=404, detail=f"Sign not found for '{word}'.")

        if len(normalized_words) > 1:
            full_phrase = " ".join(normalized_words)
            lm = db.phrase_landmarks(full_phrase)
            if lm is not None:
                video_fn = get_video_filename(full_phrase)
                video_url = f"/api/videos/{video_fn}" if video_fn else None
                return {"word": full_phrase, "landmarks": lm, "frame_count": len(lm), "video_url": video_url}

        lookup_word = normalized_words[0]
        lms = db.get_landmarks(lookup_word)
        if lms:
            matched_sign = db.match_word(lookup_word)
            video_fn = get_video_filename(matched_sign or lookup_word)
            video_url = f"/api/videos/{video_fn}" if video_fn else None
            return {"word": lookup_word, "landmarks": lms, "frame_count": len(lms), "video_url": video_url}
        raise HTTPException(status_code=404, detail=f"Sign not found for '{word}'.")

    with Timer("signs.single"):
        return await run_in_threadpool(_sync)


@app.post("/api/signs/batch")
async def get_batch_sign_landmarks(
    body: BatchSignRequest,
    current_user: dict = Depends(get_current_user)
):
    """Lookup landmarks for multiple words at once (POST version).
    
    Strategy:
    1. First try to match the complete phrase (e.g., "i am fine" as one unit)
    2. If not found, fall back to individual word matching
    3. Deduplicates consecutive identical matched signs
    """
    db = get_sign_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Sign database not ready.")

    def _sync():
        translated_words = [translate_arabic_to_english(w) for w in body.words]
        normalized_words = normalize_sign_words(translated_words)

        # Try matching the complete phrase first
        if normalized_words and len(normalized_words) > 1:
            full_phrase = ' '.join(normalized_words)
            lm = db.phrase_landmarks(full_phrase)
            if lm is not None:
                video_fn = get_video_filename(full_phrase)
                video_url = f"/api/videos/{video_fn}" if video_fn else None
                return {
                    "found": [{"word": full_phrase, "landmarks": lm, "frame_count": len(lm), "video_url": video_url}],
                    "missing": []
                }

        # Fall back to individual word matching
        found = []
        missing = []
        prev_matched_sign = None

        for w in normalized_words:
            lms = db.get_landmarks(w)
            if lms:
                # Get the actual matched word (not the input word)
                matched_sign = db.match_word(w)

                # Skip if this is a duplicate of the previous sign
                if matched_sign == prev_matched_sign:
                    continue

                video_fn = get_video_filename(matched_sign or w)
                video_url = f"/api/videos/{video_fn}" if video_fn else None
                found.append({"word": w, "landmarks": lms, "frame_count": len(lms), "video_url": video_url})
                prev_matched_sign = matched_sign
            else:
                # Out-of-vocabulary: try Gloss & Stitch (fingerspell + smooth) before
                # declaring the word unrenderable.
                try:
                    import stitch as stitch_mod
                    stitched = stitch_mod.gloss_and_stitch(w, db)
                except Exception as e:
                    print(f"[Gloss&Stitch] failed for '{w}': {e}")
                    stitched = None
                if stitched:
                    found.append({
                        "word": w, "landmarks": stitched,
                        "frame_count": len(stitched), "video_url": None,
                        "source": "stitch",
                    })
                    prev_matched_sign = None
                else:
                    missing.append(w)
        return {"found": found, "missing": missing}

    with Timer("signs.batch"):
        return await run_in_threadpool(_sync)


# ═══════════════════════════════════════════════════════════════
# ARABIC SIGN DB API (completely separate from English)
# ═══════════════════════════════════════════════════════════════

@app.post("/api/signs_ar/batch")
async def get_batch_arabic_sign_landmarks(
    body: BatchSignRequest,
    current_user: dict = Depends(get_current_user)
):
    """Lookup landmarks for Arabic text-to-sign.
    
    Flow: Arabic input → translate to English → match against 20 Arabic vocab words
    → return landmarks from signs_ar.db (NOT signs.db).
    """
    ar_db = get_arabic_sign_db()
    if ar_db is None:
        raise HTTPException(status_code=503, detail="Arabic sign database not ready.")

    # Translate each word from Arabic to English
    translated_words = [translate_arabic_to_english(w) for w in body.words]
    
    # Flatten multi-word translations
    all_english_words = []
    for tw in translated_words:
        for part in tw.split():
            cleaned = re.sub(r"[^a-z0-9]", "", part.lower().strip())
            if cleaned:
                all_english_words.append(cleaned)

    found = []
    missing = []
    prev_matched = None

    for w in all_english_words:
        lms = ar_db.get_landmarks(w)
        if lms:
            matched = ar_db.match_word(w)
            # Skip consecutive duplicates
            if matched == prev_matched:
                continue
            found.append({"word": w, "landmarks": lms, "frame_count": len(lms)})
            prev_matched = matched
        else:
            missing.append(w)

    return {"found": found, "missing": missing}


# ═══════════════════════════════════════════════════════════════
# SOCKET.IO — WebRTC signaling & meeting relay
# ═══════════════════════════════════════════════════════════════

rooms: dict = {}  # Map sid -> room_id

@sio.on("connect")
async def on_connect(sid, environ, auth=None):
    token = (auth or {}).get("token") if isinstance(auth, dict) else None
    if token:
        try:
            decode_token(token)
        except HTTPException:
            print(f"[WS] Rejected {sid}: invalid auth token")
            raise ConnectionRefusedError("Unauthorized")
    print(f"[WS] Client connected: {sid}")

@sio.on("disconnect")
async def on_disconnect(sid):
    print(f"[WS] Client disconnected: {sid}")
    room_id = rooms.pop(sid, None)
    if room_id:
        await sio.emit("peer_left", {"sid": sid}, room=room_id)
        print(f"[WS] {sid} disconnected and left room '{room_id}'")

@sio.on("join_room")
async def join_room(sid, data):
    room_id = data.get("room", "general")
    
    # Limit to 1-1 (max 2 participants)
    participants = [s for s, r in rooms.items() if r == room_id]
    if len(participants) >= 2:
        await sio.emit("room_full", {"room": room_id}, to=sid)
        print(f"[WS] Rejecting {sid} from room '{room_id}' (room full)")
        return

    await sio.enter_room(sid, room_id)
    rooms[sid] = room_id
    await sio.emit("new_peer", {"sid": sid}, room=room_id, skip_sid=sid)
    print(f"[WS] {sid} joined room '{room_id}'")

@sio.on("leave_room")
async def leave_room(sid, data):
    room_id = data.get("room", "general")
    await sio.leave_room(sid, room_id)
    rooms.pop(sid, None)
    await sio.emit("peer_left", {"sid": sid}, room=room_id, skip_sid=sid)
    print(f"[WS] {sid} left room '{room_id}'")

@sio.on("announce_presence")
async def announce_presence(sid, data):
    room = data.get("room", "general")
    payload = dict(data) if isinstance(data, dict) else {}
    payload["sender_sid"] = sid
    await sio.emit("presence_announced", payload, room=room, skip_sid=sid)

@sio.on("announce_presence_reply")
async def announce_presence_reply(sid, data):
    room = data.get("room", "general")
    payload = dict(data) if isinstance(data, dict) else {}
    payload["sender_sid"] = sid
    await sio.emit("presence_reply", payload, room=room, skip_sid=sid)

@sio.on("webrtc_offer")
async def webrtc_offer(sid, data):
    room = data.get("room", "general")
    await sio.emit("webrtc_offer", {"sdp": data["sdp"], "sender_sid": sid}, room=room, skip_sid=sid)

@sio.on("webrtc_answer")
async def webrtc_answer(sid, data):
    room = data.get("room", "general")
    await sio.emit("webrtc_answer", {"sdp": data["sdp"], "sender_sid": sid}, room=room, skip_sid=sid)

@sio.on("webrtc_ice_candidate")
async def webrtc_ice_candidate(sid, data):
    room = data.get("room", "general")
    await sio.emit("webrtc_ice_candidate", {"candidate": data["candidate"], "sender_sid": sid}, room=room, skip_sid=sid)

@sio.on("translate_sentence")
async def translate_sentence(sid, data):
    room = data.get("room", "general")
    await sio.emit("remote_sentence", data, room=room, skip_sid=sid)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(socket_app, host="0.0.0.0", port=port, reload=False)

