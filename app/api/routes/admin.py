"""
Admin endpoints for management operations
"""
from typing import Optional
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


@router.delete("/delete", response_model=DeleteResponse)
async def delete_documents_by_filters(
    category: Optional[str] = None,
    source_type: Optional[str] = None,
    collection: AsyncIOMotorCollection = Depends(get_db_collection)
):
    """
    Delete documents filtered by category and/or source type.

    At least one filter must be provided.

    Returns:
        DeleteResponse with status and count
    """
    category = category.strip() if category else None
    source_type = source_type.strip() if source_type else None

    if not category and not source_type:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one filter: category or source_type"
        )

    filter_query = {}
    if category:
        filter_query["metadata.category"] = category
    if source_type:
        filter_query["metadata.source_type"] = source_type

    try:
        result = await collection.delete_many(filter_query)
        filters_used_parts = []
        if category:
            filters_used_parts.append(f"category='{category}'")
        if source_type:
            filters_used_parts.append(f"source_type='{source_type}'")
        filters_used = ", ".join(filters_used_parts)
        return DeleteResponse(
            status="success",
            message=f"Deleted {result.deleted_count} documents for {filters_used}",
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