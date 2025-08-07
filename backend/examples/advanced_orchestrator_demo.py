#!/usr/bin/env python3
"""
Advanced AI Model Orchestrator Demo

This script demonstrates the capabilities of the Advanced AI Model Orchestrator,
including intelligent routing, cost optimization, performance monitoring,
user preferences, and batch processing.

Run this script to see the orchestrator in action with various scenarios.
"""

import asyncio
import json
import time
from typing import Dict, Any, List
from datetime import datetime

# Import the orchestrator components
from app.models.advanced_ai_orchestrator import (
    AdvancedAIOrchestrator,
    RoutingStrategy,
    TaskComplexity,
    RoutingDecision
)
from app.services.advanced_orchestrator_service import (
    ChatOrchestratorService,
    BatchOrchestratorService
)
from app.core.config import settings
from app.utils.cache import CacheManager


class OrchestratorDemo:
    """Demo class showcasing the Advanced AI Orchestrator capabilities"""
    
    def __init__(self):
        self.orchestrator: AdvancedAIOrchestrator = None
        self.chat_service: ChatOrchestratorService = None
        self.batch_service: BatchOrchestratorService = None
        self.demo_user_id = 1
    
    async def initialize(self):
        """Initialize the orchestrator for demo"""
        print("🚀 Initializing Advanced AI Orchestrator Demo...")
        
        # Initialize cache
        cache = CacheManager(settings.REDIS_URL)
        await cache.connect()
        
        # Initialize orchestrator
        self.orchestrator = AdvancedAIOrchestrator(cache)
        await self.orchestrator.initialize()
        
        # Initialize services
        self.chat_service = ChatOrchestratorService()
        self.batch_service = BatchOrchestratorService()
        
        print("✅ Orchestrator initialized successfully!")
        print(f"📊 Available providers: {list(self.orchestrator.model_clients.keys())}")
        print()
    
    async def demo_task_complexity_analysis(self):
        """Demo 1: Task Complexity Analysis"""
        print("=" * 60)
        print("📋 DEMO 1: Task Complexity Analysis")
        print("=" * 60)
        
        test_tasks = [
            "Hello, how are you today?",
            "Explain the concept of machine learning and its applications in healthcare",
            "Write a Python function to implement a binary search algorithm with error handling",
            "Conduct a comprehensive analysis of quantum computing's potential impact on cryptography, including mathematical proofs and implementation considerations"
        ]
        
        for i, task in enumerate(test_tasks, 1):
            print(f"\n📝 Task {i}: {task[:50]}{'...' if len(task) > 50 else ''}")
            
            analysis = await self.orchestrator._analyze_task_complexity(task)
            
            print(f"   🎯 Complexity: {analysis.complexity.value}")
            print(f"   📏 Estimated tokens: {analysis.estimated_tokens}")
            print(f"   💻 Requires code: {analysis.requires_code}")
            print(f"   🧠 Requires reasoning: {analysis.requires_reasoning}")
            print(f"   📊 Confidence: {analysis.confidence_score:.2f}")
    
    async def demo_routing_strategies(self):
        """Demo 2: Different Routing Strategies"""
        print("\n" + "=" * 60)
        print("🎯 DEMO 2: Routing Strategies Comparison")
        print("=" * 60)
        
        test_message = "Write a Python function to calculate fibonacci numbers using dynamic programming"
        
        strategies = [
            RoutingStrategy.COST_OPTIMAL,
            RoutingStrategy.PERFORMANCE_OPTIMAL,
            RoutingStrategy.QUALITY_OPTIMAL,
            RoutingStrategy.BALANCED
        ]
        
        print(f"📝 Test message: {test_message}")
        print()
        
        for strategy in strategies:
            print(f"🎯 Strategy: {strategy.value}")
            
            try:
                decision = await self.orchestrator.route_request(
                    task=test_message,
                    user_id=self.demo_user_id,
                    strategy=strategy
                )
                
                print(f"   🤖 Selected: {decision.chosen_provider.value}:{decision.chosen_model}")
                print(f"   💰 Estimated cost: ${decision.estimated_cost:.4f}")
                print(f"   ⏱️  Estimated time: {decision.estimated_time:.1f}s")
                print(f"   📋 Reasoning: {decision.reasoning}")
                print(f"   🎯 Confidence: {decision.confidence:.2f}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            print()
    
    async def demo_execution_with_fallback(self):
        """Demo 3: Request Execution with Fallback"""
        print("=" * 60)
        print("⚡ DEMO 3: Request Execution with Fallback")
        print("=" * 60)
        
        test_message = "Explain the difference between async and sync programming in Python"
        
        print(f"📝 Executing: {test_message}")
        print()
        
        start_time = time.time()
        
        try:
            result = await self.orchestrator.execute_request(
                task=test_message,
                user_id=self.demo_user_id
            )
            
            execution_time = time.time() - start_time
            
            print("✅ Execution successful!")
            print(f"🤖 Model used: {result['model_used']}")
            print(f"🔄 Attempts: {result['attempts']}")
            print(f"⏱️  Execution time: {execution_time:.2f}s")
            print(f"📝 Response preview: {result['response'].content[:200]}...")
            print(f"💰 Cost: ${result['response'].cost:.4f}")
            print(f"📊 Tokens used: {result['response'].tokens_used}")
            
        except Exception as e:
            print(f"❌ Execution failed: {e}")
    
    async def demo_streaming(self):
        """Demo 4: Streaming Response"""
        print("\n" + "=" * 60)
        print("🌊 DEMO 4: Streaming Response")
        print("=" * 60)
        
        test_message = "Write a short story about a robot learning to paint"
        
        print(f"📝 Streaming: {test_message}")
        print("🌊 Response stream:")
        print("-" * 40)
        
        try:
            chunks_received = 0
            total_content = ""
            
            async for chunk in self.orchestrator.stream_request(
                task=test_message,
                user_id=self.demo_user_id
            ):
                print(chunk, end="", flush=True)
                total_content += chunk
                chunks_received += 1
                
                # Limit for demo
                if chunks_received > 50:
                    print("\n\n[... truncated for demo ...]")
                    break
            
            print(f"\n\n✅ Streaming completed!")
            print(f"📊 Chunks received: {chunks_received}")
            print(f"📏 Total length: {len(total_content)} characters")
            
        except Exception as e:
            print(f"❌ Streaming failed: {e}")
    
    async def demo_batch_processing(self):
        """Demo 5: Batch Processing"""
        print("\n" + "=" * 60)
        print("📦 DEMO 5: Batch Processing")
        print("=" * 60)
        
        batch_requests = [
            {"task": "What is machine learning?"},
            {"task": "Explain quantum computing"},
            {"task": "How does blockchain work?"},
            {"task": "What are neural networks?"},
            {"task": "Define artificial intelligence"}
        ]
        
        print(f"📦 Submitting batch with {len(batch_requests)} requests...")
        
        try:
            batch_id = await self.orchestrator.batch_process(
                requests=batch_requests,
                user_id=self.demo_user_id,
                priority=1
            )
            
            print(f"✅ Batch submitted with ID: {batch_id}")
            
            # Wait a bit for processing
            print("⏳ Waiting for batch processing...")
            await asyncio.sleep(5)
            
            # Check status (this is a simplified demo - in reality you'd check periodically)
            print("📊 Batch processing status: In progress...")
            print("💡 In a real application, you would poll the status endpoint periodically")
            
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
    
    async def demo_user_analytics(self):
        """Demo 6: User Analytics and Preferences"""
        print("\n" + "=" * 60)
        print("📈 DEMO 6: User Analytics and Preferences")
        print("=" * 60)
        
        try:
            # Get user preferences
            user_prefs = await self.orchestrator._get_user_preferences(self.demo_user_id)
            
            print("👤 User Preferences:")
            print(f"   💰 Cost sensitivity: {user_prefs.cost_sensitivity:.2f}")
            print(f"   ⚡ Speed preference: {user_prefs.speed_preference:.2f}")
            print(f"   🏆 Quality preference: {user_prefs.quality_preference:.2f}")
            print(f"   💳 Monthly budget: ${user_prefs.monthly_budget:.2f}")
            print(f"   💸 Current spend: ${user_prefs.current_spend:.2f}")
            print()
            
            print("🤖 Preferred models by complexity:")
            for complexity, models in user_prefs.preferred_models.items():
                print(f"   {complexity.value}: {', '.join(models)}")
            
        except Exception as e:
            print(f"❌ Analytics failed: {e}")
    
    async def demo_performance_monitoring(self):
        """Demo 7: Performance Monitoring"""
        print("\n" + "=" * 60)
        print("📊 DEMO 7: Performance Monitoring")
        print("=" * 60)
        
        try:
            analytics = await self.orchestrator.get_analytics()
            
            print("📈 System Performance:")
            print(f"   🔄 Active streams: {analytics['active_streams']}")
            print(f"   📦 Queued batches: {analytics['queued_batches']}")
            print()
            
            if analytics['models']:
                print("🤖 Model Performance:")
                for model_name, metrics in analytics['models'].items():
                    if metrics.get('total_requests', 0) > 0:
                        print(f"   {model_name}:")
                        print(f"      📊 Requests: {metrics['total_requests']}")
                        print(f"      ✅ Success rate: {metrics.get('success_rate', 0):.2%}")
                        print(f"      ⏱️  Avg response time: {metrics.get('avg_response_time', 0):.2f}s")
                        print(f"      💰 Total cost: ${metrics['total_cost']:.4f}")
            else:
                print("📊 No performance data available yet (run some requests first)")
            
        except Exception as e:
            print(f"❌ Performance monitoring failed: {e}")
    
    async def demo_health_check(self):
        """Demo 8: Health Check"""
        print("\n" + "=" * 60)
        print("🏥 DEMO 8: System Health Check")
        print("=" * 60)
        
        try:
            health = await self.orchestrator.health_check()
            
            print(f"🏥 Overall status: {health['status']}")
            print(f"🚀 Initialized: {health['initialized']}")
            print(f"🔗 Cache connected: {health['cache_connected']}")
            print()
            
            print("🤖 Model Clients:")
            for provider, status in health['model_clients'].items():
                status_emoji = "✅" if status.get('status') == 'healthy' else "❌"
                print(f"   {status_emoji} {provider}: {status.get('status', 'unknown')}")
                if 'error' in status:
                    print(f"      Error: {status['error']}")
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
    
    async def demo_cost_estimation(self):
        """Demo 9: Cost Estimation"""
        print("\n" + "=" * 60)
        print("💰 DEMO 9: Cost Estimation")
        print("=" * 60)
        
        test_scenarios = [
            {"task": "Hello", "tokens": 100},
            {"task": "Write a detailed analysis of climate change", "tokens": 2000},
            {"task": "Create a complex machine learning model in Python", "tokens": 4000}
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"📊 Scenario {i}: {scenario['task'][:50]}...")
            
            # Estimate costs for different models
            for provider_models in self.orchestrator.model_capabilities.values():
                for model_name, capabilities in provider_models.items():
                    if capabilities.provider.value in self.orchestrator.model_clients:
                        cost = self.orchestrator._estimate_request_cost(
                            capabilities, scenario['tokens']
                        )
                        print(f"   {capabilities.provider.value}:{model_name}: ${cost:.4f}")
            print()
    
    async def run_all_demos(self):
        """Run all demo scenarios"""
        print("🎭 Advanced AI Model Orchestrator - Comprehensive Demo")
        print("=" * 60)
        print("This demo showcases the intelligent routing, cost optimization,")
        print("performance monitoring, and advanced features of the orchestrator.")
        print("=" * 60)
        
        demos = [
            self.demo_task_complexity_analysis,
            self.demo_routing_strategies,
            self.demo_execution_with_fallback,
            self.demo_streaming,
            self.demo_batch_processing,
            self.demo_user_analytics,
            self.demo_performance_monitoring,
            self.demo_health_check,
            self.demo_cost_estimation
        ]
        
        for demo in demos:
            try:
                await demo()
                await asyncio.sleep(1)  # Brief pause between demos
            except Exception as e:
                print(f"❌ Demo failed: {e}")
                continue
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed! The Advanced AI Orchestrator provides:")
        print("   🎯 Intelligent model routing based on task complexity")
        print("   💰 Cost optimization and budget management")
        print("   📊 Real-time performance monitoring")
        print("   👤 User preference learning and adaptation")
        print("   🔄 Automatic fallback and retry logic")
        print("   🌊 Streaming response support")
        print("   📦 Efficient batch processing")
        print("   🏥 Comprehensive health monitoring")
        print("=" * 60)
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.orchestrator:
            await self.orchestrator.cleanup()


async def main():
    """Main demo function"""
    demo = OrchestratorDemo()
    
    try:
        await demo.initialize()
        await demo.run_all_demos()
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
    finally:
        await demo.cleanup()
        print("\n👋 Demo cleanup completed")


if __name__ == "__main__":
    # Example usage patterns that can be used in your application
    
    print("Advanced AI Model Orchestrator - Usage Examples")
    print("=" * 50)
    
    # Example 1: Simple routing and execution
    example_1 = """
    # Example 1: Simple routing and execution
    from app.services.advanced_orchestrator_service import ChatOrchestratorService
    
    chat_service = ChatOrchestratorService()
    
    # Route a message to optimal model
    decision = await chat_service.route_chat_message(
        message="Explain machine learning",
        user_id=123,
        strategy=RoutingStrategy.BALANCED
    )
    
    # Execute with automatic fallback
    result = await chat_service.execute_chat_request(
        message="Explain machine learning", 
        user_id=123,
        budget_limit=0.50  # Max $0.50 for this request
    )
    """
    
    # Example 2: Streaming with custom preferences
    example_2 = """
    # Example 2: Streaming with custom preferences
    
    # Stream response with cost optimization
    async for chunk in chat_service.stream_chat_response(
        message="Write a story about AI",
        user_id=123,
        strategy=RoutingStrategy.COST_OPTIMAL
    ):
        print(chunk, end="")
    """
    
    # Example 3: Batch processing
    example_3 = """
    # Example 3: Batch processing
    from app.services.advanced_orchestrator_service import BatchOrchestratorService
    
    batch_service = BatchOrchestratorService()
    
    # Submit batch
    batch_id = await batch_service.submit_batch(
        requests=[
            {"task": "Summarize this article: ..."},
            {"task": "Translate to Spanish: ..."},
            {"task": "Generate code for: ..."}
        ],
        user_id=123,
        priority=5
    )
    
    # Check status
    status = await batch_service.get_batch_status(batch_id)
    """
    
    print(example_1)
    print(example_2)
    print(example_3)
    print("\n" + "=" * 50)
    print("Running comprehensive demo...")
    
    # Run the actual demo
    asyncio.run(main())