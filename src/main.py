import logging
import os
import sys
import time
import traceback

import google.generativeai as genai
from backend.app.api.routes import router
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

# --- UTF-8 bootstrap for Windows consoles with emojis in logs ---
try:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    # Never crash just because the terminal can't change encoding
    pass
# ----------------------------------------------------------------

# Set up logging (once)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (once)
load_dotenv()

# Configure Gemini API (once)
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    logger.info("[OK] Gemini API configured successfully")
else:
    logger.warning("[WARN] GEMINI_API_KEY not found in environment variables")

app = FastAPI(
    title="Terraform AI Agent",
    description="AI-powered Terraform code generator using Strands Agents framework",
    version="2.0.1",
    docs_url="/docs",
    redoc_url="/redoc",
)

# FIXED: Simplified CORS configuration - removed conflicting middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# REMOVED: Duplicate CORS middleware that was causing conflicts


def _add_bearer_auth(app):
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=getattr(app, "version", "0.1.0"),
            description=getattr(app, "description", None),
            routes=app.routes,
        )
        # ensure sections exist
        schema.setdefault("components", {}).setdefault("securitySchemes", {})
        # declare HTTP Bearer (JWT)
        schema["components"]["securitySchemes"]["BearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
        # make it global so the top-right "Authorize" appears
        schema["security"] = [{"BearerAuth": []}]
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi


_add_bearer_auth(app)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"[ERROR] Unhandled exception in {request.method} {request.url}")
    logger.error(f"Exception type: {type(exc).__name__}")
    logger.error(f"Exception message: {str(exc)}")
    logger.error("Full traceback:")
    logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "type": "server_error",
            "error": f"Internal server error: {str(exc)}",
            "exception_type": type(exc).__name__,
        },
    )


@app.get("/")
def read_root():
    """Root endpoint with service information"""
    return {
        "message": "Terraform AI Agent backend is running with Strands Agents framework",
        "version": "2.0.1",
        "framework": "Strands Agents",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "test": "/test",
            "chat": "/chat",
            "streaming": "/chat/stream",
            "docs": "/docs",
        },
    }


# Removed @app.get("/health") to avoid conflict with routes.py version


@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("[START] Starting Terraform AI Agent...")
    logger.info(f"[DIR] Working directory: {os.getcwd()}")
    logger.info(f"[KEY] API Key configured: {bool(os.getenv('GEMINI_API_KEY'))}")
    logger.info("[OK] Terraform AI Agent started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("[STOP] Shutting down Terraform AI Agent...")


# Include the main router
app.include_router(router, prefix="", tags=["main"])


# Additional middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    start_time = time.time()

    logger.info(f"[IN] {request.method} {request.url.path}")

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"[OUT] {request.method} {request.url.path} - {response.status_code} ({process_time:.2f}s)"
    )

    return response
