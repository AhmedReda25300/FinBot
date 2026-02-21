import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.config import get_settings
from app.database import connect_to_mongo, close_mongo_connection
from app.api.routes import embed, search, admin, chat

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Vector Store API with MongoDB and Google Gemini Embeddings"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(embed.router)
app.include_router(search.router)
app.include_router(chat.router)
app.include_router(admin.router)

# --- Static files ---
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")

@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve chat_ui.html at root"""
    return FileResponse(os.path.join(STATIC_FOLDER, "chat_ui.html"))

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()
    print(f"{settings.APP_NAME} v{settings.APP_VERSION} started")

@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()
    print(f"{settings.APP_NAME} shutdown")
