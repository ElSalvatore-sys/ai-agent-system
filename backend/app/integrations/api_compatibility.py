"""
API Compatibility Integrations
Comprehensive API compatibility layers for various protocols and standards
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import uuid
import logging

import aiohttp
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import strawberry
from strawberry.fastapi import GraphQLRouter
import grpc
from concurrent import futures
import websockets

from app.core.integration_framework import (
    BaseIntegration, IntegrationConfig, IntegrationType, IntegrationStatus
)
from app.core.logger import LoggerMixin


class OpenAIAPIIntegration(BaseIntegration):
    """OpenAI API compatibility layer for seamless integration with existing tools"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.api_prefix = config.settings.get("api_prefix", "/v1")
        self.supported_models = config.settings.get("supported_models", [])
        self.enable_streaming = config.settings.get("enable_streaming", True)
        self.max_tokens_default = config.settings.get("max_tokens_default", 2048)
        
        self.app: Optional[FastAPI] = None
        self._active_requests: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> bool:
        """Initialize OpenAI API compatibility layer"""
        try:
            # Create FastAPI app for compatibility endpoints
            self._setup_openai_endpoints()
            
            self.logger.info("OpenAI API compatibility layer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI API integration: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check OpenAI API compatibility health"""
        return True  # Always healthy if initialized
    
    async def shutdown(self) -> None:
        """Shutdown OpenAI API compatibility layer"""
        # Cancel any active requests
        for request_id in list(self._active_requests.keys()):
            await self._cancel_request(request_id)
    
    def _setup_openai_endpoints(self) -> None:
        """Setup OpenAI-compatible API endpoints"""
        from fastapi import Request
        from pydantic import BaseModel
        
        class ChatCompletionRequest(BaseModel):
            model: str
            messages: List[Dict[str, str]]
            max_tokens: Optional[int] = None
            temperature: Optional[float] = 0.7
            top_p: Optional[float] = 1.0
            n: Optional[int] = 1
            stream: Optional[bool] = False
            stop: Optional[Union[str, List[str]]] = None
            presence_penalty: Optional[float] = 0.0
            frequency_penalty: Optional[float] = 0.0
            user: Optional[str] = None
        
        class CompletionRequest(BaseModel):
            model: str
            prompt: Union[str, List[str]]
            max_tokens: Optional[int] = None
            temperature: Optional[float] = 0.7
            top_p: Optional[float] = 1.0
            n: Optional[int] = 1
            stream: Optional[bool] = False
            stop: Optional[Union[str, List[str]]] = None
            presence_penalty: Optional[float] = 0.0
            frequency_penalty: Optional[float] = 0.0
            user: Optional[str] = None
        
        # Store endpoint handlers
        self._openai_handlers = {
            "chat_completions": self._handle_chat_completions,
            "completions": self._handle_completions,
            "models": self._handle_models,
            "embeddings": self._handle_embeddings
        }
    
    async def _handle_chat_completions(self, request_data: Dict[str, Any]) -> Union[Dict[str, Any], AsyncGenerator]:
        """Handle OpenAI chat completions endpoint"""
        try:
            request_id = str(uuid.uuid4())
            self._active_requests[request_id] = {
                "type": "chat_completion",
                "data": request_data,
                "created_at": datetime.utcnow()
            }
            
            # Extract request parameters
            model = request_data.get("model")
            messages = request_data.get("messages", [])
            max_tokens = request_data.get("max_tokens", self.max_tokens_default)
            temperature = request_data.get("temperature", 0.7)
            stream = request_data.get("stream", False)
            
            # Convert messages to prompt format
            prompt = self._convert_messages_to_prompt(messages)
            
            # Generate response using internal LLM system
            if stream:
                return self._stream_chat_completion(request_id, model, prompt, max_tokens, temperature)
            else:
                response_text = await self._generate_completion(model, prompt, max_tokens, temperature)
                
                return {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion",
                    "created": int(datetime.utcnow().timestamp()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_text
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(response_text.split()),
                        "total_tokens": len(prompt.split()) + len(response_text.split())
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Error in chat completions: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if request_id in self._active_requests:
                del self._active_requests[request_id]
    
    async def _handle_completions(self, request_data: Dict[str, Any]) -> Union[Dict[str, Any], AsyncGenerator]:
        """Handle OpenAI completions endpoint"""
        try:
            request_id = str(uuid.uuid4())
            self._active_requests[request_id] = {
                "type": "completion",
                "data": request_data,
                "created_at": datetime.utcnow()
            }
            
            # Extract request parameters
            model = request_data.get("model")
            prompt = request_data.get("prompt", "")
            max_tokens = request_data.get("max_tokens", self.max_tokens_default)
            temperature = request_data.get("temperature", 0.7)
            stream = request_data.get("stream", False)
            
            if isinstance(prompt, list):
                prompt = " ".join(prompt)
            
            # Generate response
            if stream:
                return self._stream_completion(request_id, model, prompt, max_tokens, temperature)
            else:
                response_text = await self._generate_completion(model, prompt, max_tokens, temperature)
                
                return {
                    "id": f"cmpl-{request_id}",
                    "object": "text_completion",
                    "created": int(datetime.utcnow().timestamp()),
                    "model": model,
                    "choices": [
                        {
                            "text": response_text,
                            "index": 0,
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(response_text.split()),
                        "total_tokens": len(prompt.split()) + len(response_text.split())
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Error in completions: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if request_id in self._active_requests:
                del self._active_requests[request_id]
    
    async def _handle_models(self) -> Dict[str, Any]:
        """Handle OpenAI models endpoint"""
        models = []
        
        # Get available models from the system
        for model_name in self.supported_models:
            models.append({
                "id": model_name,
                "object": "model",
                "created": int(datetime.utcnow().timestamp()),
                "owned_by": "local",
                "permission": [],
                "root": model_name,
                "parent": None
            })
        
        return {
            "object": "list",
            "data": models
        }
    
    async def _handle_embeddings(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle OpenAI embeddings endpoint"""
        try:
            model = request_data.get("model")
            input_text = request_data.get("input", "")
            
            if isinstance(input_text, list):
                input_text = " ".join(input_text)
            
            # Generate embeddings (placeholder implementation)
            # In production, this would use actual embedding models
            embedding = [0.1] * 1536  # OpenAI ada-002 dimension
            
            return {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": embedding,
                        "index": 0
                    }
                ],
                "model": model,
                "usage": {
                    "prompt_tokens": len(input_text.split()),
                    "total_tokens": len(input_text.split())
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in embeddings: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI messages format to simple prompt"""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    async def _generate_completion(self, model: str, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate completion using internal LLM system"""
        # This would integrate with the existing LLM orchestrator
        # For now, return a placeholder response
        return f"This is a response from {model} for prompt: {prompt[:50]}..."
    
    async def _stream_completion(self, request_id: str, model: str, prompt: str, max_tokens: int, temperature: float) -> AsyncGenerator:
        """Stream completion response"""
        response_text = await self._generate_completion(model, prompt, max_tokens, temperature)
        
        # Split response into chunks for streaming
        words = response_text.split()
        
        for i, word in enumerate(words):
            chunk = {
                "id": f"cmpl-{request_id}",
                "object": "text_completion",
                "created": int(datetime.utcnow().timestamp()),
                "model": model,
                "choices": [
                    {
                        "text": word + " ",
                        "index": 0,
                        "finish_reason": None if i < len(words) - 1 else "stop"
                    }
                ]
            }
            
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.1)  # Simulate streaming delay
        
        yield "data: [DONE]\n\n"
    
    async def _stream_chat_completion(self, request_id: str, model: str, prompt: str, max_tokens: int, temperature: float) -> AsyncGenerator:
        """Stream chat completion response"""
        response_text = await self._generate_completion(model, prompt, max_tokens, temperature)
        
        # Split response into chunks for streaming
        words = response_text.split()
        
        for i, word in enumerate(words):
            chunk = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion.chunk",
                "created": int(datetime.utcnow().timestamp()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": word + " "
                        },
                        "finish_reason": None if i < len(words) - 1 else "stop"
                    }
                ]
            }
            
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.1)
        
        yield "data: [DONE]\n\n"
    
    async def _cancel_request(self, request_id: str) -> None:
        """Cancel an active request"""
        if request_id in self._active_requests:
            del self._active_requests[request_id]
            self.logger.info(f"Cancelled request: {request_id}")


class GraphQLIntegration(BaseIntegration):
    """GraphQL API integration for complex queries and real-time subscriptions"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.endpoint_path = config.settings.get("endpoint_path", "/graphql")
        self.enable_playground = config.settings.get("enable_playground", True)
        self.enable_subscriptions = config.settings.get("enable_subscriptions", True)
        
        self.schema = None
        self.router = None
    
    async def initialize(self) -> bool:
        """Initialize GraphQL integration"""
        try:
            # Define GraphQL schema
            self._create_graphql_schema()
            
            # Create GraphQL router
            self._create_graphql_router()
            
            self.logger.info("GraphQL integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GraphQL integration: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check GraphQL health"""
        return self.schema is not None and self.router is not None
    
    async def shutdown(self) -> None:
        """Shutdown GraphQL integration"""
        pass  # No cleanup needed
    
    def _create_graphql_schema(self) -> None:
        """Create GraphQL schema with types and resolvers"""
        
        @strawberry.type
        class Model:
            id: str
            name: str
            provider: str
            status: str
            capabilities: List[str]
            
        @strawberry.type
        class GenerationRequest:
            id: str
            model: str
            prompt: str
            status: str
            created_at: str
            
        @strawberry.type
        class GenerationResponse:
            id: str
            request_id: str
            text: str
            tokens: int
            cost: float
            
        @strawberry.type
        class Metric:
            name: str
            value: float
            timestamp: str
            labels: Dict[str, str]
        
        @strawberry.type
        class Query:
            @strawberry.field
            async def models(self) -> List[Model]:
                """Get available models"""
                # This would integrate with the model discovery system
                return [
                    Model(
                        id="llama2-7b",
                        name="Llama 2 7B",
                        provider="ollama",
                        status="active",
                        capabilities=["text_generation", "chat"]
                    )
                ]
            
            @strawberry.field
            async def generation_requests(self, limit: int = 10) -> List[GenerationRequest]:
                """Get recent generation requests"""
                # This would query the database
                return []
            
            @strawberry.field
            async def metrics(self, metric_name: str, hours: int = 1) -> List[Metric]:
                """Get metrics for the specified time range"""
                # This would query the metrics system
                return []
        
        @strawberry.type
        class Mutation:
            @strawberry.field
            async def generate_text(self, model: str, prompt: str, max_tokens: int = 100) -> GenerationResponse:
                """Generate text using specified model"""
                # This would integrate with the generation system
                request_id = str(uuid.uuid4())
                
                return GenerationResponse(
                    id=str(uuid.uuid4()),
                    request_id=request_id,
                    text=f"Generated response for: {prompt[:50]}...",
                    tokens=max_tokens,
                    cost=0.01
                )
            
            @strawberry.field
            async def load_model(self, model_id: str) -> Model:
                """Load a model"""
                # This would integrate with the model loading system
                return Model(
                    id=model_id,
                    name=model_id,
                    provider="local",
                    status="loading",
                    capabilities=["text_generation"]
                )
        
        @strawberry.type
        class Subscription:
            @strawberry.subscription
            async def model_status_updates(self) -> AsyncGenerator[Model, None]:
                """Subscribe to model status updates"""
                # This would listen to model status changes
                while True:
                    yield Model(
                        id="test-model",
                        name="Test Model",
                        provider="local",
                        status="active",
                        capabilities=["text_generation"]
                    )
                    await asyncio.sleep(30)
            
            @strawberry.subscription
            async def metrics_stream(self, metric_name: str) -> AsyncGenerator[Metric, None]:
                """Subscribe to real-time metrics"""
                while True:
                    yield Metric(
                        name=metric_name,
                        value=1.0,
                        timestamp=datetime.utcnow().isoformat(),
                        labels={"service": "llm-platform"}
                    )
                    await asyncio.sleep(10)
        
        # Create schema
        if self.enable_subscriptions:
            self.schema = strawberry.Schema(
                query=Query,
                mutation=Mutation,
                subscription=Subscription
            )
        else:
            self.schema = strawberry.Schema(
                query=Query,
                mutation=Mutation
            )
    
    def _create_graphql_router(self) -> None:
        """Create GraphQL router"""
        if self.schema:
            self.router = GraphQLRouter(
                self.schema,
                graphiql=self.enable_playground
            )


class GRPCIntegration(BaseIntegration):
    """gRPC integration for high-performance inter-service communication"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.port = config.settings.get("port", 50051)
        self.host = config.settings.get("host", "localhost")
        self.max_workers = config.settings.get("max_workers", 10)
        self.enable_reflection = config.settings.get("enable_reflection", True)
        
        self.server = None
        self._server_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> bool:
        """Initialize gRPC integration"""
        try:
            # Create gRPC server
            self._create_grpc_server()
            
            # Start gRPC server
            await self._start_grpc_server()
            
            self.logger.info(f"gRPC integration initialized on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize gRPC integration: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check gRPC server health"""
        try:
            # Try to connect to the gRPC server
            return self.server is not None
        except Exception as e:
            self.logger.error(f"gRPC health check failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown gRPC integration"""
        if self.server:
            await self.server.stop(grace=5)
        
        if self._server_task:
            self._server_task.cancel()
    
    def _create_grpc_server(self) -> None:
        """Create gRPC server with services"""
        # Note: This is a simplified example
        # In production, you would define proper protobuf services
        
        self.server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers)
        )
        
        # Add services to server
        # self._add_llm_service()
        # self._add_model_service()
        
        # Add reflection service if enabled
        if self.enable_reflection:
            from grpc_reflection.v1alpha import reflection
            reflection.enable_server_reflection((), self.server)
        
        # Bind server to port
        listen_addr = f"{self.host}:{self.port}"
        self.server.add_insecure_port(listen_addr)
    
    async def _start_grpc_server(self) -> None:
        """Start gRPC server"""
        if self.server:
            await self.server.start()
            self._server_task = asyncio.create_task(self.server.wait_for_termination())


class WebSocketIntegration(BaseIntegration):
    """WebSocket integration for real-time communication"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.endpoint_path = config.settings.get("endpoint_path", "/ws")
        self.max_connections = config.settings.get("max_connections", 1000)
        self.heartbeat_interval = config.settings.get("heartbeat_interval", 30)
        
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_stats: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> bool:
        """Initialize WebSocket integration"""
        try:
            # Setup WebSocket endpoints
            self._setup_websocket_endpoints()
            
            # Start heartbeat task
            asyncio.create_task(self._heartbeat_loop())
            
            self.logger.info("WebSocket integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize WebSocket integration: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check WebSocket health"""
        return len(self.active_connections) <= self.max_connections
    
    async def shutdown(self) -> None:
        """Shutdown WebSocket integration"""
        # Close all active connections
        for connection_id, websocket in list(self.active_connections.items()):
            try:
                await websocket.close()
            except Exception as e:
                self.logger.error(f"Error closing WebSocket {connection_id}: {e}")
        
        self.active_connections.clear()
        self.connection_stats.clear()
    
    def _setup_websocket_endpoints(self) -> None:
        """Setup WebSocket endpoint handlers"""
        # These would be registered with the FastAPI app
        self._websocket_handlers = {
            "chat": self._handle_chat_websocket,
            "models": self._handle_models_websocket,
            "metrics": self._handle_metrics_websocket
        }
    
    async def _handle_chat_websocket(self, websocket: WebSocket) -> None:
        """Handle chat WebSocket connections"""
        connection_id = str(uuid.uuid4())
        
        try:
            await websocket.accept()
            self.active_connections[connection_id] = websocket
            self.connection_stats[connection_id] = {
                "type": "chat",
                "connected_at": datetime.utcnow(),
                "messages_sent": 0,
                "messages_received": 0
            }
            
            self.logger.info(f"Chat WebSocket connected: {connection_id}")
            
            while True:
                # Receive message from client
                data = await websocket.receive_json()
                self.connection_stats[connection_id]["messages_received"] += 1
                
                # Process chat message
                response = await self._process_chat_message(data)
                
                # Send response
                await websocket.send_json(response)
                self.connection_stats[connection_id]["messages_sent"] += 1
                
        except WebSocketDisconnect:
            self.logger.info(f"Chat WebSocket disconnected: {connection_id}")
        except Exception as e:
            self.logger.error(f"Error in chat WebSocket {connection_id}: {e}")
        finally:
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
            if connection_id in self.connection_stats:
                del self.connection_stats[connection_id]
    
    async def _handle_models_websocket(self, websocket: WebSocket) -> None:
        """Handle model status WebSocket connections"""
        connection_id = str(uuid.uuid4())
        
        try:
            await websocket.accept()
            self.active_connections[connection_id] = websocket
            self.connection_stats[connection_id] = {
                "type": "models",
                "connected_at": datetime.utcnow(),
                "updates_sent": 0
            }
            
            self.logger.info(f"Models WebSocket connected: {connection_id}")
            
            # Send periodic model status updates
            while True:
                model_status = await self._get_model_status()
                await websocket.send_json({
                    "type": "model_status",
                    "data": model_status,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                self.connection_stats[connection_id]["updates_sent"] += 1
                await asyncio.sleep(5)  # Send updates every 5 seconds
                
        except WebSocketDisconnect:
            self.logger.info(f"Models WebSocket disconnected: {connection_id}")
        except Exception as e:
            self.logger.error(f"Error in models WebSocket {connection_id}: {e}")
        finally:
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
            if connection_id in self.connection_stats:
                del self.connection_stats[connection_id]
    
    async def _handle_metrics_websocket(self, websocket: WebSocket) -> None:
        """Handle metrics WebSocket connections"""
        connection_id = str(uuid.uuid4())
        
        try:
            await websocket.accept()
            self.active_connections[connection_id] = websocket
            self.connection_stats[connection_id] = {
                "type": "metrics",
                "connected_at": datetime.utcnow(),
                "metrics_sent": 0
            }
            
            self.logger.info(f"Metrics WebSocket connected: {connection_id}")
            
            # Send periodic metrics updates
            while True:
                metrics = await self._get_current_metrics()
                await websocket.send_json({
                    "type": "metrics",
                    "data": metrics,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                self.connection_stats[connection_id]["metrics_sent"] += 1
                await asyncio.sleep(10)  # Send metrics every 10 seconds
                
        except WebSocketDisconnect:
            self.logger.info(f"Metrics WebSocket disconnected: {connection_id}")
        except Exception as e:
            self.logger.error(f"Error in metrics WebSocket {connection_id}: {e}")
        finally:
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
            if connection_id in self.connection_stats:
                del self.connection_stats[connection_id]
    
    async def _process_chat_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming chat message"""
        message = data.get("message", "")
        model = data.get("model", "default")
        
        # This would integrate with the LLM system
        response_text = f"Echo: {message}"
        
        return {
            "type": "response",
            "message": response_text,
            "model": model,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _get_model_status(self) -> List[Dict[str, Any]]:
        """Get current model status"""
        # This would integrate with the model management system
        return [
            {
                "id": "llama2-7b",
                "name": "Llama 2 7B",
                "status": "active",
                "load_time": "2.3s",
                "memory_usage": "4.2GB"
            }
        ]
    
    async def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        # This would integrate with the metrics system
        return {
            "active_models": 2,
            "requests_per_minute": 45,
            "average_response_time": 1.2,
            "gpu_memory_usage": 75.5,
            "cpu_usage": 32.1
        }
    
    async def _heartbeat_loop(self) -> None:
        """Send heartbeat to all connections"""
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            
            disconnected = []
            for connection_id, websocket in self.active_connections.items():
                try:
                    await websocket.ping()
                except Exception:
                    disconnected.append(connection_id)
            
            # Remove disconnected connections
            for connection_id in disconnected:
                if connection_id in self.active_connections:
                    del self.active_connections[connection_id]
                if connection_id in self.connection_stats:
                    del self.connection_stats[connection_id]
            
            if disconnected:
                self.logger.info(f"Cleaned up {len(disconnected)} disconnected WebSocket connections")
    
    async def broadcast_message(self, message: Dict[str, Any], connection_type: Optional[str] = None) -> None:
        """Broadcast message to all connections of a specific type"""
        disconnected = []
        
        for connection_id, websocket in self.active_connections.items():
            if connection_type and self.connection_stats.get(connection_id, {}).get("type") != connection_type:
                continue
            
            try:
                await websocket.send_json(message)
            except Exception as e:
                self.logger.error(f"Error broadcasting to {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
            if connection_id in self.connection_stats:
                del self.connection_stats[connection_id]


# Default configurations for API compatibility integrations
DEFAULT_OPENAIINTEGRATION_CONFIG = IntegrationConfig(
    name="openai_api",
    type=IntegrationType.API_COMPATIBILITY,
    enabled=True,
    priority=95,
    settings={
        "api_prefix": "/v1",
        "supported_models": ["llama2-7b", "codellama-13b", "mistral-7b"],
        "enable_streaming": True,
        "max_tokens_default": 2048
    },
    health_check_interval=60
)

DEFAULT_GRAPHQLINTEGRATION_CONFIG = IntegrationConfig(
    name="graphql",
    type=IntegrationType.API_COMPATIBILITY,
    enabled=True,
    priority=85,
    settings={
        "endpoint_path": "/graphql",
        "enable_playground": True,
        "enable_subscriptions": True
    },
    health_check_interval=60
)

DEFAULT_GRPCINTEGRATION_CONFIG = IntegrationConfig(
    name="grpc",
    type=IntegrationType.API_COMPATIBILITY,
    enabled=True,
    priority=80,
    settings={
        "port": 50051,
        "host": "localhost",
        "max_workers": 10,
        "enable_reflection": True
    },
    health_check_interval=60
)

DEFAULT_WEBSOCKETINTEGRATION_CONFIG = IntegrationConfig(
    name="websocket",
    type=IntegrationType.API_COMPATIBILITY,
    enabled=True,
    priority=90,
    settings={
        "endpoint_path": "/ws",
        "max_connections": 1000,
        "heartbeat_interval": 30
    },
    health_check_interval=60
)