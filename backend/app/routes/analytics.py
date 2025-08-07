"""
Analytics API Routes
Comprehensive cost tracking, performance analytics, and system metrics
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, desc, func, and_, or_
from pydantic import BaseModel, Field

from app.database.database import get_db
from app.database.models import (
    User, AIRequest, CostTracking, GeneratedContent, PerformanceMetrics, 
    SystemAlert, UserBudget, AuditLog, ModelProvider, ContentType, 
    BudgetPeriod, RequestStatus, AlertType
)
from app.middleware.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

# Response Models
class CostSummary(BaseModel):
    total_cost: float
    total_requests: int
    total_tokens: int
    average_cost_per_request: float
    period_start: datetime
    period_end: datetime

class ModelUsage(BaseModel):
    model_provider: str
    model_name: str
    requests: int
    tokens: int
    cost: float
    success_rate: float
    avg_response_time: float

class ContentStats(BaseModel):
    total_generated: int
    by_type: Dict[str, int]
    total_size_mb: float
    public_content: int
    templates: int

class PerformanceOverview(BaseModel):
    avg_response_time: float
    p95_response_time: float
    success_rate: float
    total_requests: int
    error_rate: float
    uptime_percent: float

class BudgetStatus(BaseModel):
    budget_id: int
    budget_name: str
    budget_amount: float
    current_spend: float
    remaining_budget: float
    utilization_percent: float
    days_remaining: int
    is_over_budget: bool

class AlertSummary(BaseModel):
    id: int
    alert_type: str
    severity: str
    title: str
    message: str
    is_read: bool
    created_at: datetime

# Cost Analytics Endpoints
@router.get("/costs", response_model=CostSummary)
async def get_cost_summary(
    period_days: int = Query(30, ge=1, le=365),
    model_provider: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get cost summary for specified period"""
    try:
        period_start = datetime.utcnow() - timedelta(days=period_days)
        period_end = datetime.utcnow()
        
        # Build query
        query = select(
            func.sum(AIRequest.total_cost_usd).label('total_cost'),
            func.count(AIRequest.id).label('total_requests'),
            func.sum(AIRequest.total_tokens).label('total_tokens'),
            func.avg(AIRequest.total_cost_usd).label('avg_cost')
        ).where(
            and_(
                AIRequest.user_id == current_user.id,
                AIRequest.created_at >= period_start,
                AIRequest.created_at <= period_end
            )
        )
        
        if model_provider:
            query = query.where(AIRequest.model_provider == model_provider)
        
        result = await db.execute(query)
        row = result.first()
        
        return CostSummary(
            total_cost=row.total_cost or 0.0,
            total_requests=row.total_requests or 0,
            total_tokens=row.total_tokens or 0,
            average_cost_per_request=row.avg_cost or 0.0,
            period_start=period_start,
            period_end=period_end
        )
        
    except Exception as e:
        logger.error(f"Error getting cost summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch cost summary")

@router.get("/costs/by-model", response_model=List[ModelUsage])
async def get_costs_by_model(
    period_days: int = Query(30, ge=1, le=365),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get cost breakdown by model"""
    try:
        period_start = datetime.utcnow() - timedelta(days=period_days)
        
        query = select(
            AIRequest.model_provider,
            AIRequest.model_name,
            func.count(AIRequest.id).label('requests'),
            func.sum(AIRequest.total_tokens).label('tokens'),
            func.sum(AIRequest.total_cost_usd).label('cost'),
            func.avg(func.case((AIRequest.success == True, 1.0), else_=0.0)).label('success_rate'),
            func.avg(AIRequest.response_time_ms).label('avg_response_time')
        ).where(
            and_(
                AIRequest.user_id == current_user.id,
                AIRequest.created_at >= period_start
            )
        ).group_by(
            AIRequest.model_provider,
            AIRequest.model_name
        ).order_by(
            desc('cost')
        ).limit(limit)
        
        result = await db.execute(query)
        rows = result.all()
        
        return [
            ModelUsage(
                model_provider=row.model_provider,
                model_name=row.model_name,
                requests=row.requests,
                tokens=row.tokens or 0,
                cost=row.cost or 0.0,
                success_rate=row.success_rate or 0.0,
                avg_response_time=row.avg_response_time or 0.0
            )
            for row in rows
        ]
        
    except Exception as e:
        logger.error(f"Error getting costs by model: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch model costs")

@router.get("/costs/daily")
async def get_daily_costs(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get daily cost breakdown"""
    try:
        period_start = datetime.utcnow() - timedelta(days=days)
        
        query = select(
            func.date(AIRequest.created_at).label('date'),
            func.sum(AIRequest.total_cost_usd).label('cost'),
            func.count(AIRequest.id).label('requests'),
            func.sum(AIRequest.total_tokens).label('tokens')
        ).where(
            and_(
                AIRequest.user_id == current_user.id,
                AIRequest.created_at >= period_start
            )
        ).group_by(
            func.date(AIRequest.created_at)
        ).order_by('date')
        
        result = await db.execute(query)
        rows = result.all()
        
        return [
            {
                "date": row.date.isoformat(),
                "cost": row.cost or 0.0,
                "requests": row.requests or 0,
                "tokens": row.tokens or 0
            }
            for row in rows
        ]
        
    except Exception as e:
        logger.error(f"Error getting daily costs: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch daily costs")

# Content Analytics
@router.get("/content/stats", response_model=ContentStats)
async def get_content_stats(
    period_days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get generated content statistics"""
    try:
        period_start = datetime.utcnow() - timedelta(days=period_days)
        
        # Total content generated
        total_query = select(func.count(GeneratedContent.id)).where(
            and_(
                GeneratedContent.user_id == current_user.id,
                GeneratedContent.created_at >= period_start
            )
        )
        total_result = await db.execute(total_query)
        total_generated = total_result.scalar() or 0
        
        # By content type
        type_query = select(
            GeneratedContent.content_type,
            func.count(GeneratedContent.id).label('count')
        ).where(
            and_(
                GeneratedContent.user_id == current_user.id,
                GeneratedContent.created_at >= period_start
            )
        ).group_by(GeneratedContent.content_type)
        
        type_result = await db.execute(type_query)
        by_type = {row.content_type: row.count for row in type_result.all()}
        
        # Size and other stats
        stats_query = select(
            func.sum(GeneratedContent.file_size_bytes).label('total_size'),
            func.count(func.case((GeneratedContent.is_public == True, 1))).label('public_count'),
            func.count(func.case((GeneratedContent.is_template == True, 1))).label('template_count')
        ).where(
            and_(
                GeneratedContent.user_id == current_user.id,
                GeneratedContent.created_at >= period_start
            )
        )
        
        stats_result = await db.execute(stats_query)
        stats_row = stats_result.first()
        
        return ContentStats(
            total_generated=total_generated,
            by_type=by_type,
            total_size_mb=(stats_row.total_size or 0) / (1024 * 1024),
            public_content=stats_row.public_count or 0,
            templates=stats_row.template_count or 0
        )
        
    except Exception as e:
        logger.error(f"Error getting content stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch content stats")

@router.get("/content/recent")
async def get_recent_content(
    limit: int = Query(20, ge=1, le=100),
    content_type: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get recently generated content"""
    try:
        query = select(GeneratedContent).where(
            GeneratedContent.user_id == current_user.id
        )
        
        if content_type:
            query = query.where(GeneratedContent.content_type == content_type)
        
        query = query.order_by(desc(GeneratedContent.created_at)).limit(limit)
        
        result = await db.execute(query)
        content = result.scalars().all()
        
        return [
            {
                "id": item.id,
                "uuid": item.uuid,
                "title": item.title,
                "content_type": item.content_type,
                "file_size_bytes": item.file_size_bytes,
                "language": item.language,
                "is_public": item.is_public,
                "created_at": item.created_at.isoformat()
            }
            for item in content
        ]
        
    except Exception as e:
        logger.error(f"Error getting recent content: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch recent content")

# Performance Analytics
@router.get("/performance", response_model=PerformanceOverview)
async def get_performance_overview(
    period_days: int = Query(7, ge=1, le=30),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get system performance overview"""
    try:
        period_start = datetime.utcnow() - timedelta(days=period_days)
        
        query = select(
            func.avg(AIRequest.response_time_ms).label('avg_response_time'),
            func.percentile_cont(0.95).within_group(AIRequest.response_time_ms).label('p95_response_time'),
            func.avg(func.case((AIRequest.success == True, 1.0), else_=0.0)).label('success_rate'),
            func.count(AIRequest.id).label('total_requests'),
            func.avg(func.case((AIRequest.success == False, 1.0), else_=0.0)).label('error_rate')
        ).where(
            and_(
                AIRequest.user_id == current_user.id,
                AIRequest.created_at >= period_start
            )
        )
        
        result = await db.execute(query)
        row = result.first()
        
        return PerformanceOverview(
            avg_response_time=row.avg_response_time or 0.0,
            p95_response_time=row.p95_response_time or 0.0,
            success_rate=row.success_rate or 0.0,
            total_requests=row.total_requests or 0,
            error_rate=row.error_rate or 0.0,
            uptime_percent=99.9  # This would come from system metrics in production
        )
        
    except Exception as e:
        logger.error(f"Error getting performance overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch performance data")

# Budget Management
@router.get("/budgets", response_model=List[BudgetStatus])
async def get_user_budgets(
    active_only: bool = Query(True),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get user budget information"""
    try:
        query = select(UserBudget).where(UserBudget.user_id == current_user.id)
        
        if active_only:
            query = query.where(UserBudget.is_active == True)
        
        query = query.order_by(desc(UserBudget.created_at))
        
        result = await db.execute(query)
        budgets = result.scalars().all()
        
        budget_statuses = []
        for budget in budgets:
            days_remaining = (budget.period_end - datetime.utcnow()).days
            days_remaining = max(0, days_remaining)
            
            budget_statuses.append(BudgetStatus(
                budget_id=budget.id,
                budget_name=budget.budget_name,
                budget_amount=budget.budget_amount,
                current_spend=budget.current_spend,
                remaining_budget=budget.remaining_budget,
                utilization_percent=budget.utilization_percent,
                days_remaining=days_remaining,
                is_over_budget=budget.budget_exceeded
            ))
        
        return budget_statuses
        
    except Exception as e:
        logger.error(f"Error getting user budgets: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch budget data")

# System Alerts
@router.get("/alerts", response_model=List[AlertSummary])
async def get_user_alerts(
    unread_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get user alerts and notifications"""
    try:
        query = select(SystemAlert).where(
            or_(
                SystemAlert.user_id == current_user.id,
                SystemAlert.user_id.is_(None)  # System-wide alerts
            )
        )
        
        if unread_only:
            query = query.where(SystemAlert.is_read == False)
        
        query = query.order_by(desc(SystemAlert.created_at)).limit(limit)
        
        result = await db.execute(query)
        alerts = result.scalars().all()
        
        return [
            AlertSummary(
                id=alert.id,
                alert_type=alert.alert_type,
                severity=alert.severity,
                title=alert.title,
                message=alert.message,
                is_read=alert.is_read,
                created_at=alert.created_at
            )
            for alert in alerts
        ]
        
    except Exception as e:
        logger.error(f"Error getting user alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch alerts")

@router.put("/alerts/{alert_id}/read")
async def mark_alert_as_read(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Mark alert as read"""
    try:
        query = select(SystemAlert).where(
            and_(
                SystemAlert.id == alert_id,
                or_(
                    SystemAlert.user_id == current_user.id,
                    SystemAlert.user_id.is_(None)
                )
            )
        )
        
        result = await db.execute(query)
        alert = result.scalar_one_or_none()
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert.is_read = True
        alert.updated_at = datetime.utcnow()
        
        await db.commit()
        
        return {"message": "Alert marked as read"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking alert as read: {e}")
        raise HTTPException(status_code=500, detail="Failed to update alert")

# Dashboard Summary
@router.get("/dashboard")
async def get_dashboard_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive dashboard summary"""
    try:
        # Get current month data
        month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Cost summary
        cost_query = select(
            func.sum(AIRequest.total_cost_usd).label('monthly_cost'),
            func.count(AIRequest.id).label('monthly_requests'),
            func.sum(AIRequest.total_tokens).label('monthly_tokens')
        ).where(
            and_(
                AIRequest.user_id == current_user.id,
                AIRequest.created_at >= month_start
            )
        )
        
        cost_result = await db.execute(cost_query)
        cost_row = cost_result.first()
        
        # Content summary
        content_query = select(
            func.count(GeneratedContent.id).label('monthly_content'),
            func.sum(GeneratedContent.file_size_bytes).label('total_size')
        ).where(
            and_(
                GeneratedContent.user_id == current_user.id,
                GeneratedContent.created_at >= month_start
            )
        )
        
        content_result = await db.execute(content_query)
        content_row = content_result.first()
        
        # Unread alerts
        alerts_query = select(func.count(SystemAlert.id)).where(
            and_(
                or_(
                    SystemAlert.user_id == current_user.id,
                    SystemAlert.user_id.is_(None)
                ),
                SystemAlert.is_read == False
            )
        )
        
        alerts_result = await db.execute(alerts_query)
        unread_alerts = alerts_result.scalar() or 0
        
        return {
            "monthly_cost": cost_row.monthly_cost or 0.0,
            "monthly_requests": cost_row.monthly_requests or 0,
            "monthly_tokens": cost_row.monthly_tokens or 0,
            "monthly_content": content_row.monthly_content or 0,
            "total_content_size_mb": (content_row.total_size or 0) / (1024 * 1024),
            "unread_alerts": unread_alerts,
            "budget_limit": current_user.monthly_budget_limit,
            "budget_remaining": current_user.monthly_budget_limit - current_user.current_month_spend,
            "budget_utilization": (current_user.current_month_spend / current_user.monthly_budget_limit) * 100 if current_user.monthly_budget_limit > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch dashboard data")

# Export Data
@router.get("/export/costs")
async def export_cost_data(
    format: str = Query("csv", regex="^(csv|json)$"),
    period_days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Export cost data in CSV or JSON format"""
    try:
        period_start = datetime.utcnow() - timedelta(days=period_days)
        
        query = select(AIRequest).where(
            and_(
                AIRequest.user_id == current_user.id,
                AIRequest.created_at >= period_start
            )
        ).order_by(AIRequest.created_at)
        
        result = await db.execute(query)
        requests = result.scalars().all()
        
        data = [
            {
                "date": req.created_at.isoformat(),
                "model_provider": req.model_provider,
                "model_name": req.model_name,
                "input_tokens": req.input_tokens,
                "output_tokens": req.output_tokens,
                "total_tokens": req.total_tokens,
                "cost_usd": req.total_cost_usd,
                "response_time_ms": req.response_time_ms,
                "success": req.success
            }
            for req in requests
        ]
        
        if format == "json":
            from fastapi.responses import JSONResponse
            return JSONResponse(content=data)
        else:
            # CSV format
            import csv
            import io
            from fastapi.responses import StreamingResponse
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=data[0].keys() if data else [])
            writer.writeheader()
            writer.writerows(data)
            
            response = StreamingResponse(
                io.BytesIO(output.getvalue().encode('utf-8')),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=cost_data_{period_days}days.csv"}
            )
            return response
        
    except Exception as e:
        logger.error(f"Error exporting cost data: {e}")
        raise HTTPException(status_code=500, detail="Failed to export data")