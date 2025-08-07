from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_
from pydantic import BaseModel, Field

from app.database.database import get_db
from app.database.models import User, Conversation, Message, UsageLog, AIModel, SystemMetrics, UserRole, ModelProvider, PerformanceMetrics, CostTracking
from app.middleware.auth import RequireRole
from app.services.cost_optimizer import CostOptimizer
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Pydantic models for responses
class UserStats(BaseModel):
    total_users: int
    active_users: int
    new_users_today: int
    new_users_this_week: int

class ConversationStats(BaseModel):
    total_conversations: int
    active_conversations: int
    conversations_today: int
    avg_messages_per_conversation: float

class UsageStats(BaseModel):
    total_requests: int
    requests_today: int
    total_tokens: int
    total_cost: float
    avg_cost_per_request: float

class ModelUsageStats(BaseModel):
    provider: str
    model_name: str
    request_count: int
    total_tokens: int
    total_cost: float
    avg_response_time: float
    success_rate: float

class SystemOverview(BaseModel):
    user_stats: UserStats
    conversation_stats: ConversationStats
    usage_stats: UsageStats
    top_models: List[ModelUsageStats]

class OptimizationResult(BaseModel):
    optimizations_applied: int
    cost_savings_estimated: float
    performance_improvements: List[str]
    recommendations: List[str]

class DetailedUserInfo(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime]
    conversation_count: int
    message_count: int
    total_tokens_used: int
    total_cost: float

@router.get("/overview", response_model=SystemOverview)
@RequireRole("admin")
async def get_system_overview(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Get comprehensive system overview and statistics"""
    
    try:
        # Get current time for date calculations
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=7)
        
        # User statistics
        total_users_result = await db.execute(select(func.count(User.id)))
        total_users = total_users_result.scalar()
        
        active_users_result = await db.execute(
            select(func.count(User.id)).where(User.is_active == True)
        )
        active_users = active_users_result.scalar()
        
        new_users_today_result = await db.execute(
            select(func.count(User.id)).where(User.created_at >= today_start)
        )
        new_users_today = new_users_today_result.scalar()
        
        new_users_week_result = await db.execute(
            select(func.count(User.id)).where(User.created_at >= week_start)
        )
        new_users_week = new_users_week_result.scalar()
        
        user_stats = UserStats(
            total_users=total_users or 0,
            active_users=active_users or 0,
            new_users_today=new_users_today or 0,
            new_users_this_week=new_users_week or 0
        )
        
        # Conversation statistics
        total_convs_result = await db.execute(select(func.count(Conversation.id)))
        total_conversations = total_convs_result.scalar()
        
        active_convs_result = await db.execute(
            select(func.count(Conversation.id)).where(Conversation.status == "active")
        )
        active_conversations = active_convs_result.scalar()
        
        convs_today_result = await db.execute(
            select(func.count(Conversation.id)).where(Conversation.created_at >= today_start)
        )
        conversations_today = convs_today_result.scalar()
        
        # Average messages per conversation
        avg_messages_result = await db.execute(
            select(func.avg(func.count(Message.id)))
            .select_from(Message)
            .group_by(Message.conversation_id)
        )
        avg_messages = avg_messages_result.scalar() or 0
        
        conversation_stats = ConversationStats(
            total_conversations=total_conversations or 0,
            active_conversations=active_conversations or 0,
            conversations_today=conversations_today or 0,
            avg_messages_per_conversation=float(avg_messages)
        )
        
        # Usage statistics
        total_usage_result = await db.execute(select(func.count(UsageLog.id)))
        total_requests = total_usage_result.scalar()
        
        usage_today_result = await db.execute(
            select(func.count(UsageLog.id)).where(UsageLog.created_at >= today_start)
        )
        requests_today = usage_today_result.scalar()
        
        total_tokens_result = await db.execute(select(func.sum(UsageLog.total_tokens)))
        total_tokens = total_tokens_result.scalar()
        
        total_cost_result = await db.execute(select(func.sum(UsageLog.cost)))
        total_cost = total_cost_result.scalar()
        
        avg_cost = (total_cost / total_requests) if total_requests and total_cost else 0
        
        usage_stats = UsageStats(
            total_requests=total_requests or 0,
            requests_today=requests_today or 0,
            total_tokens=int(total_tokens or 0),
            total_cost=float(total_cost or 0),
            avg_cost_per_request=float(avg_cost)
        )
        
        # Top models by usage
        top_models_result = await db.execute(
            select(
                UsageLog.model_provider,
                UsageLog.model_name,
                func.count(UsageLog.id).label('request_count'),
                func.sum(UsageLog.total_tokens).label('total_tokens'),
                func.sum(UsageLog.cost).label('total_cost'),
                func.avg(UsageLog.response_time).label('avg_response_time'),
                func.avg(func.cast(UsageLog.success, func.Float)).label('success_rate')
            )
            .group_by(UsageLog.model_provider, UsageLog.model_name)
            .order_by(desc('request_count'))
            .limit(5)
        )
        
        top_models = [
            ModelUsageStats(
                provider=row.model_provider.value,
                model_name=row.model_name,
                request_count=row.request_count,
                total_tokens=int(row.total_tokens or 0),
                total_cost=float(row.total_cost or 0),
                avg_response_time=float(row.avg_response_time or 0),
                success_rate=float(row.success_rate or 0)
            )
            for row in top_models_result.all()
        ]
        
        return SystemOverview(
            user_stats=user_stats,
            conversation_stats=conversation_stats,
            usage_stats=usage_stats,
            top_models=top_models
        )
        
    except Exception as e:
        logger.error(f"Failed to get system overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system overview")

@router.get("/users", response_model=List[DetailedUserInfo])
@RequireRole("admin")
async def get_users(
    request: Request,
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50, le=1000),
    offset: int = Query(0, ge=0),
    search: Optional[str] = None,
    role: Optional[str] = None,
    active_only: bool = False
):
    """Get detailed user information with statistics"""
    
    try:
        # Build base query
        stmt = select(
            User,
            func.count(func.distinct(Conversation.id)).label('conversation_count'),
            func.count(func.distinct(Message.id)).label('message_count'),
            func.sum(UsageLog.total_tokens).label('total_tokens'),
            func.sum(UsageLog.cost).label('total_cost')
        ).outerjoin(Conversation).outerjoin(Message).outerjoin(UsageLog)
        
        # Apply filters
        if search:
            stmt = stmt.where(
                func.or_(
                    User.username.ilike(f"%{search}%"),
                    User.email.ilike(f"%{search}%"),
                    User.full_name.ilike(f"%{search}%")
                )
            )
        
        if role:
            try:
                user_role = UserRole(role)
                stmt = stmt.where(User.role == user_role)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid role")
        
        if active_only:
            stmt = stmt.where(User.is_active == True)
        
        # Group and order
        stmt = stmt.group_by(User.id).order_by(desc(User.created_at)).limit(limit).offset(offset)
        
        result = await db.execute(stmt)
        users_data = result.all()
        
        return [
            DetailedUserInfo(
                id=user.id,
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                role=user.role.value,
                is_active=user.is_active,
                is_verified=user.is_verified,
                created_at=user.created_at,
                last_login=user.last_login,
                conversation_count=conversation_count or 0,
                message_count=message_count or 0,
                total_tokens_used=int(total_tokens or 0),
                total_cost=float(total_cost or 0)
            )
            for user, conversation_count, message_count, total_tokens, total_cost in users_data
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get users: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve users")

@router.put("/users/{user_id}/toggle-active")
@RequireRole("admin")
async def toggle_user_active(
    user_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Toggle user active status"""
    
    try:
        stmt = select(User).where(User.id == user_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Don't allow deactivating the last admin
        if user.role == UserRole.ADMIN and user.is_active:
            admin_count_result = await db.execute(
                select(func.count(User.id)).where(
                    and_(User.role == UserRole.ADMIN, User.is_active == True)
                )
            )
            admin_count = admin_count_result.scalar()
            
            if admin_count <= 1:
                raise HTTPException(
                    status_code=400, 
                    detail="Cannot deactivate the last active admin"
                )
        
        user.is_active = not user.is_active
        user.updated_at = datetime.utcnow()
        
        await db.commit()
        
        logger.info(f"User {user.username} active status changed to {user.is_active}")
        
        return {
            "message": f"User {'activated' if user.is_active else 'deactivated'} successfully",
            "user_id": user_id,
            "is_active": user.is_active
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to toggle user active status: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user status")

@router.get("/usage-analytics")
@RequireRole("admin")
async def get_usage_analytics(
    request: Request,
    db: AsyncSession = Depends(get_db),
    days: int = Query(30, ge=1, le=365),
    provider: Optional[str] = None
):
    """Get detailed usage analytics"""
    
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Base query
        stmt = select(
            func.date(UsageLog.created_at).label('date'),
            func.count(UsageLog.id).label('requests'),
            func.sum(UsageLog.total_tokens).label('tokens'),
            func.sum(UsageLog.cost).label('cost'),
            func.avg(UsageLog.response_time).label('avg_response_time')
        ).where(UsageLog.created_at >= start_date)
        
        if provider:
            try:
                model_provider = ModelProvider(provider)
                stmt = stmt.where(UsageLog.model_provider == model_provider)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid provider")
        
        stmt = stmt.group_by(func.date(UsageLog.created_at)).order_by('date')
        
        result = await db.execute(stmt)
        daily_stats = result.all()
        
        # Model breakdown
        model_stats_stmt = select(
            UsageLog.model_provider,
            UsageLog.model_name,
            func.count(UsageLog.id).label('requests'),
            func.sum(UsageLog.total_tokens).label('tokens'),
            func.sum(UsageLog.cost).label('cost'),
            func.avg(UsageLog.response_time).label('avg_response_time'),
            func.sum(func.cast(UsageLog.success, func.Integer)).label('successful_requests')
        ).where(UsageLog.created_at >= start_date)
        
        if provider:
            model_stats_stmt = model_stats_stmt.where(UsageLog.model_provider == model_provider)
        
        model_stats_stmt = model_stats_stmt.group_by(
            UsageLog.model_provider, UsageLog.model_name
        ).order_by(desc('requests'))
        
        model_result = await db.execute(model_stats_stmt)
        model_stats = model_result.all()
        
        return {
            "period_days": days,
            "provider_filter": provider,
            "daily_usage": [
                {
                    "date": stat.date.isoformat(),
                    "requests": stat.requests,
                    "tokens": int(stat.tokens or 0),
                    "cost": float(stat.cost or 0),
                    "avg_response_time": float(stat.avg_response_time or 0)
                }
                for stat in daily_stats
            ],
            "model_breakdown": [
                {
                    "provider": stat.model_provider.value,
                    "model": stat.model_name,
                    "requests": stat.requests,
                    "tokens": int(stat.tokens or 0),
                    "cost": float(stat.cost or 0),
                    "avg_response_time": float(stat.avg_response_time or 0),
                    "success_rate": float(stat.successful_requests / stat.requests) if stat.requests > 0 else 0
                }
                for stat in model_stats
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get usage analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve usage analytics")

@router.get("/cost-optimization")
@RequireRole("admin")
async def get_cost_optimization_data(request: Request):
    """Get cost optimization insights and recommendations"""
    
    try:
        # Get cost optimizer from app state
        from main import app
        if hasattr(app.state, 'ai_orchestrator'):
            orchestrator = app.state.ai_orchestrator
            cost_optimizer = orchestrator.cost_optimizer
            
            # Get analytics
            analytics = await cost_optimizer.get_cost_analytics()
            predictions = await cost_optimizer.predict_monthly_cost()
            
            # Get model recommendations for common capabilities
            common_capabilities = ["text", "reasoning", "code"]
            recommendations = cost_optimizer.get_model_recommendations(common_capabilities)
            
            return {
                "current_analytics": analytics,
                "cost_predictions": predictions,
                "model_recommendations": recommendations[:10],  # Top 10
                "optimization_tips": [
                    "Consider using faster models for simple queries",
                    "Cache frequently requested responses",
                    "Set appropriate token limits for different use cases",
                    "Monitor and adjust temperature settings",
                    "Use streaming for long responses to improve perceived performance"
                ]
            }
        else:
            raise HTTPException(status_code=503, detail="AI orchestrator not available")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cost optimization data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cost optimization data")

@router.post("/system/cleanup")
@RequireRole("admin")
async def cleanup_system_data(
    request: Request,
    db: AsyncSession = Depends(get_db),
    days_old: int = Query(90, ge=30, le=365)
):
    """Clean up old system data"""
    
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        # Clean up old usage logs
        old_logs_stmt = select(func.count(UsageLog.id)).where(UsageLog.created_at < cutoff_date)
        old_logs_result = await db.execute(old_logs_stmt)
        old_logs_count = old_logs_result.scalar()
        
        if old_logs_count > 0:
            delete_logs_stmt = UsageLog.__table__.delete().where(UsageLog.created_at < cutoff_date)
            await db.execute(delete_logs_stmt)
        
        # Clean up old system metrics
        old_metrics_stmt = select(func.count(SystemMetrics.id)).where(SystemMetrics.created_at < cutoff_date)
        old_metrics_result = await db.execute(old_metrics_stmt)
        old_metrics_count = old_metrics_result.scalar()
        
        if old_metrics_count > 0:
            delete_metrics_stmt = SystemMetrics.__table__.delete().where(SystemMetrics.created_at < cutoff_date)
            await db.execute(delete_metrics_stmt)
        
        # Clean up deleted conversations
        deleted_convs_stmt = select(func.count(Conversation.id)).where(
            and_(Conversation.status == "deleted", Conversation.updated_at < cutoff_date)
        )
        deleted_convs_result = await db.execute(deleted_convs_stmt)
        deleted_convs_count = deleted_convs_result.scalar()
        
        if deleted_convs_count > 0:
            # First delete associated messages
            delete_messages_stmt = Message.__table__.delete().where(
                Message.conversation_id.in_(
                    select(Conversation.id).where(
                        and_(Conversation.status == "deleted", Conversation.updated_at < cutoff_date)
                    )
                )
            )
            await db.execute(delete_messages_stmt)
            
            # Then delete conversations
            delete_convs_stmt = Conversation.__table__.delete().where(
                and_(Conversation.status == "deleted", Conversation.updated_at < cutoff_date)
            )
            await db.execute(delete_convs_stmt)
        
        await db.commit()
        
        logger.info(
            f"System cleanup completed: {old_logs_count} usage logs, "
            f"{old_metrics_count} metrics, {deleted_convs_count} conversations removed"
        )
        
        return {
            "message": "System cleanup completed successfully",
            "removed_counts": {
                "usage_logs": old_logs_count,
                "system_metrics": old_metrics_count,
                "deleted_conversations": deleted_convs_count
            },
            "cutoff_date": cutoff_date.isoformat()
        }
        
    except Exception as e:
        await db.rollback()
        logger.error(f"System cleanup failed: {e}")
        raise HTTPException(status_code=500, detail="System cleanup failed")

@router.get("/system/health")
@RequireRole("admin")
async def get_detailed_system_health(request: Request):
    """Get detailed system health information"""
    
    try:
        from main import app
        
        health_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "services": {},
            "performance": {},
            "resources": {}
        }
        
        # Check AI orchestrator
        if hasattr(app.state, 'ai_orchestrator'):
            ai_health = await app.state.ai_orchestrator.health_check()
            health_data["services"]["ai_models"] = ai_health
        
        # Check cache
        if hasattr(app.state, 'cache'):
            cache_healthy = await app.state.cache.ping()
            health_data["services"]["redis_cache"] = {
                "status": "healthy" if cache_healthy else "unhealthy"
            }
        
        # Add rate limiting stats
        from main import app
        for middleware in app.user_middleware:
            if hasattr(middleware, 'cls') and middleware.cls.__name__ == 'RateLimitMiddleware':
                # This would need to be implemented to access middleware stats
                pass
        
        # System resources (basic)
        import psutil
        health_data["resources"] = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
        
        return health_data
        
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system health")

@router.post("/optimize", response_model=OptimizationResult)
@RequireRole("admin")
async def trigger_system_optimization(
    request: Request,
    db: AsyncSession = Depends(get_db),
    optimize_costs: bool = Query(True),
    optimize_performance: bool = Query(True),
    cleanup_data: bool = Query(False)
):
    """Trigger comprehensive system optimization"""
    
    try:
        optimizations_applied = 0
        cost_savings = 0.0
        improvements = []
        recommendations = []
        
        # Cost Optimization
        if optimize_costs:
            try:
                from main import app
                if hasattr(app.state, 'ai_orchestrator'):
                    orchestrator = app.state.ai_orchestrator
                    cost_optimizer = orchestrator.cost_optimizer
                    
                    # Update model pricing
                    await cost_optimizer.update_model_pricing()
                    optimizations_applied += 1
                    improvements.append("Updated model pricing data")
                    
                    # Analyze cost patterns
                    analytics = await cost_optimizer.get_cost_analytics()
                    if analytics.get("potential_savings", 0) > 0:
                        cost_savings += analytics["potential_savings"]
                        recommendations.append(
                            f"Switch to more cost-effective models could save ${analytics['potential_savings']:.2f}/month"
                        )
                    
            except Exception as e:
                logger.warning(f"Cost optimization failed: {e}")
        
        # Performance Optimization
        if optimize_performance:
            try:
                # Clean up old performance metrics
                week_ago = datetime.utcnow() - timedelta(days=7)
                old_metrics_count = await db.execute(
                    select(func.count(PerformanceMetrics.id)).where(
                        PerformanceMetrics.created_at < week_ago
                    )
                )
                count = old_metrics_count.scalar()
                
                if count > 1000:  # Only clean if we have many old metrics
                    await db.execute(
                        PerformanceMetrics.__table__.delete().where(
                            PerformanceMetrics.created_at < week_ago
                        )
                    )
                    optimizations_applied += 1
                    improvements.append(f"Cleaned up {count} old performance metrics")
                
                # Optimize database queries (placeholder)
                recommendations.append("Consider adding database indexes for frequently queried fields")
                
            except Exception as e:
                logger.warning(f"Performance optimization failed: {e}")
        
        # Data Cleanup
        if cleanup_data:
            try:
                # Clean up old cost tracking data (keep last 3 months)
                three_months_ago = datetime.utcnow() - timedelta(days=90)
                old_tracking_count = await db.execute(
                    select(func.count(CostTracking.id)).where(
                        CostTracking.created_at < three_months_ago
                    )
                )
                count = old_tracking_count.scalar()
                
                if count > 0:
                    await db.execute(
                        CostTracking.__table__.delete().where(
                            CostTracking.created_at < three_months_ago
                        )
                    )
                    optimizations_applied += 1
                    improvements.append(f"Cleaned up {count} old cost tracking records")
                
            except Exception as e:
                logger.warning(f"Data cleanup failed: {e}")
        
        # General Recommendations
        recommendations.extend([
            "Monitor high-frequency API endpoints for optimization opportunities",
            "Implement response caching for frequently requested content",
            "Consider using model routing based on query complexity",
            "Set up automated cost alerts for budget management",
            "Review and optimize database connection pooling"
        ])
        
        await db.commit()
        
        logger.info(f"System optimization completed: {optimizations_applied} optimizations applied")
        
        return OptimizationResult(
            optimizations_applied=optimizations_applied,
            cost_savings_estimated=cost_savings,
            performance_improvements=improvements,
            recommendations=recommendations
        )
        
    except Exception as e:
        await db.rollback()
        logger.error(f"System optimization failed: {e}")
        raise HTTPException(status_code=500, detail="System optimization failed")