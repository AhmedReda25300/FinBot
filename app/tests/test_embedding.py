"""
Tests for embedding endpoints
"""
import pytest
from httpx import AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_embed_guideline():
    """Test embedding a guideline document"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        guideline = {
            "document_title": "Test Guide",
            "chunk_title": "Introduction",
            "content": "This is test content for embedding.",
            "category": "TEST",
            "source_type": "Guidelines"
        }
        
        response = await client.post("/embed", json=[guideline])
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["chunks_count"] > 0


@pytest.mark.asyncio
async def test_embed_decision():
    """Test embedding a decision document"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        decision = {
            "تفصيل_القرار": {
                "رقم_القرار_النهائي": "TEST-001",
                "البنود_محل_الدعوى": [
                    {
                        "اسم_البند": "Test Item",
                        "نبذة_مختصرة_عن_الاعتراض": "Brief description",
                        "وجهة_نظر_المكلف_بالتفصيل": "Taxpayer view",
                        "وجهة_نظر_الهيئة_بالتفصيل": "Authority view",
                        "الرأي_النهائي_لجنة_الاستئناف_ومبرراته": "Final decision"
                    }
                ]
            },
            "Source_Filename": "test.pdf",
            "category": "TEST",
            "source_type": "decisions-TEST"
        }
        
        response = await client.post("/embed", json=[decision])
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["chunks_count"] > 0