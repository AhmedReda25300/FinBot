"""
Admin endpoints for management operations
"""
from typing import Dict
from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorCollection
from app.models.schemas import DeleteResponse, StatsResponse
from app.api.dependencies import get_db_collection

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.delete("/delete_all", response_model=DeleteResponse)
async def delete_all_documents(
    collection: AsyncIOMotorCollection = Depends(get_db_collection)
):
    """
    Delete all documents from the collection.
    
    Returns:
        DeleteResponse with status and count
    """
    try:
        result = await collection.delete_many({})
        return DeleteResponse(
            status="success",
            message=f"Deleted {result.deleted_count} documents",
            deleted_count=result.deleted_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    collection: AsyncIOMotorCollection = Depends(get_db_collection)
):
    """
    Get statistics about the collection.
    
    Returns:
        StatsResponse with counts by category and source_type
    """
    try:
        total_count = await collection.count_documents({})
        
        # Count by category
        category_pipeline = [
            {"$group": {"_id": "$metadata.category", "count": {"$sum": 1}}}
        ]
        category_stats = await collection.aggregate(category_pipeline).to_list(length=None)
        
        # Count by source_type
        source_pipeline = [
            {"$group": {"_id": "$metadata.source_type", "count": {"$sum": 1}}}
        ]
        source_stats = await collection.aggregate(source_pipeline).to_list(length=None)
        
        return StatsResponse(
            total_documents=total_count,
            by_category={item['_id']: item['count'] for item in category_stats if item['_id']},
            by_source_type={item['_id']: item['count'] for item in source_stats if item['_id']}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(
    collection: AsyncIOMotorCollection = Depends(get_db_collection)
):
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    try:
        # Try to count documents to verify DB connection
        await collection.count_documents({})
        return {
            "status": "healthy",
            "database": "connected"
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable: {str(e)}"
        )