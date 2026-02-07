"""
FastAPI application entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.database import connect_to_mongo, close_mongo_connection
from app.api.routes import embed, search, admin, chat

settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Vector Store API with MongoDB and Google Gemini Embeddings"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(embed.router)
app.include_router(search.router)
app.include_router(chat.router)
app.include_router(admin.router)


@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    await connect_to_mongo()
    print(f"{settings.APP_NAME} v{settings.APP_VERSION} started")


@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown"""
    await close_mongo_connection()
    print(f"{settings.APP_NAME} shutdown")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }