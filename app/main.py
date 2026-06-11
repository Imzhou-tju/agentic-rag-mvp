from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import get_settings


settings = get_settings()
app = FastAPI(title=settings.app_name, version='0.1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(router, prefix='/api')

static_dir = Path('frontend')
if static_dir.exists():
    app.mount('/frontend', StaticFiles(directory='frontend'), name='frontend')


@app.get('/')
def root() -> dict:
    return {
        'message': settings.app_name,
        'docs': '/docs',
        'frontend_hint': 'Run `streamlit run frontend/app.py` for the UI.',
    }
