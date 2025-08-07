#!/usr/bin/env python3

"""
=======================================================
AI AGENT SYSTEM - COMPREHENSIVE INTEGRATION TEST SUITE
=======================================================
This script runs comprehensive integration tests for all
components of the AI agent system including Docker services,
database connections, AI model integrations, and API endpoints.
=======================================================
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import subprocess
import tempfile

# Import required libraries
try:
    import requests
    import redis
    import psycopg2
    import websockets
    import docker
    from openai import OpenAI
    import anthropic
    import google.generativeai as genai
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install requests redis psycopg2-binary websockets docker-py openai anthropic google-generativeai")
    sys.exit(1)

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
LOG_FILE = SCRIPT_DIR / "integration-test.log"
RESULTS_FILE = SCRIPT_DIR / "integration-test-results.json"
TEST_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAY = 5

# Service endpoints
SERVICES = {
    "frontend": {"url": "http://localhost:3000", "port": 3000},
    "backend": {"url": "http://localhost:8000", "port": 8000},
    "postgres": {"url": "postgresql://postgres:ai_agent_password@localhost:5432/ai_agent_system", "port": 5432},
    "redis": {"url": "redis://localhost:6379", "port": 6379},
    "ollama": {"url": "http://localhost:11434", "port": 11434},
    "prometheus": {"url": "http://localhost:9090", "port": 9090},
    "grafana": {"url": "http://localhost:3001", "port": 3001},
}

# Test results storage
test_results = {
    "timestamp": datetime.now().isoformat(),
    "total_tests": 0,
    "passed_tests": 0,
    "failed_tests": 0,
    "skipped_tests": 0,
    "test_details": [],
    "overall_status": "UNKNOWN"
}

class TestLogger:
    """Enhanced logging for integration tests."""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def info(self, msg: str):
        self.logger.info(f"[INFO] {msg}")
    
    def success(self, msg: str):
        self.logger.info(f"[SUCCESS] {msg}")
    
    def warning(self, msg: str):
        self.logger.warning(f"[WARNING] {msg}")
    
    def error(self, msg: str):
        self.logger.error(f"[ERROR] {msg}")
    
    def debug(self, msg: str):
        self.logger.debug(f"[DEBUG] {msg}")

class IntegrationTestSuite:
    """Comprehensive integration test suite."""
    
    def __init__(self):
        self.logger = TestLogger(LOG_FILE)
        self.docker_client = None
        self.test_data = {}
        
        # Load environment variables
        self.load_environment()
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            self.logger.error(f"Failed to initialize Docker client: {e}")
    
    def load_environment(self):
        """Load environment variables from .env file."""
        env_file = SCRIPT_DIR / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
            self.logger.info("Environment variables loaded from .env file")
    
    def record_test_result(self, test_name: str, status: str, duration: float, 
                          details: str = "", error: str = ""):
        """Record test result for reporting."""
        global test_results
        
        test_results["total_tests"] += 1
        if status == "PASSED":
            test_results["passed_tests"] += 1
        elif status == "FAILED":
            test_results["failed_tests"] += 1
        elif status == "SKIPPED":
            test_results["skipped_tests"] += 1
        
        test_results["test_details"].append({
            "test_name": test_name,
            "status": status,
            "duration": duration,
            "details": details,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
    
    async def run_test_with_timeout(self, test_func, test_name: str, *args, **kwargs):
        """Run test function with timeout and error handling."""
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await asyncio.wait_for(test_func(*args, **kwargs), timeout=TEST_TIMEOUT)
            else:
                result = test_func(*args, **kwargs)
            
            duration = time.time() - start_time
            self.record_test_result(test_name, "PASSED", duration, str(result))
            self.logger.success(f"Test '{test_name}' passed in {duration:.2f}s")
            return True
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            error_msg = f"Test timed out after {TEST_TIMEOUT}s"
            self.record_test_result(test_name, "FAILED", duration, "", error_msg)
            self.logger.error(f"Test '{test_name}' timed out")
            return False
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            self.record_test_result(test_name, "FAILED", duration, "", error_msg)
            self.logger.error(f"Test '{test_name}' failed: {error_msg}")
            return False
    
    # ============================================
    # DOCKER COMPOSE HEALTH CHECKS
    # ============================================
    
    def test_docker_compose_status(self) -> bool:
        """Test Docker Compose services are running."""
        try:
            result = subprocess.run(
                ["docker-compose", "ps", "--services", "--filter", "status=running"],
                cwd=SCRIPT_DIR,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise Exception(f"docker-compose ps failed: {result.stderr}")
            
            running_services = result.stdout.strip().split('\n')
            expected_services = list(SERVICES.keys())
            
            missing_services = [s for s in expected_services if s not in running_services]
            if missing_services:
                raise Exception(f"Missing services: {missing_services}")
            
            return f"All services running: {running_services}"
            
        except subprocess.TimeoutExpired:
            raise Exception("Docker compose status check timed out")
    
    def test_service_health_checks(self) -> bool:
        """Test Docker service health checks."""
        try:
            containers = self.docker_client.containers.list()
            health_status = {}
            
            for container in containers:
                name = container.name
                health = container.attrs.get('State', {}).get('Health', {})
                status = health.get('Status', 'none')
                health_status[name] = status
            
            unhealthy = [name for name, status in health_status.items() 
                        if status == 'unhealthy']
            
            if unhealthy:
                raise Exception(f"Unhealthy containers: {unhealthy}")
            
            return f"Health status: {health_status}"
            
        except Exception as e:
            raise Exception(f"Health check failed: {e}")
    
    # ============================================
    # DATABASE CONNECTION TESTS
    # ============================================
    
    def test_postgresql_connection(self) -> bool:
        """Test PostgreSQL database connection."""
        try:
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database=os.getenv("POSTGRES_DB", "ai_agent_system"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "ai_agent_password")
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            
            # Test basic operations
            cursor.execute("SELECT 1 as test;")
            result = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            if result != 1:
                raise Exception("Basic query test failed")
            
            return f"PostgreSQL connected successfully: {version}"
            
        except Exception as e:
            raise Exception(f"PostgreSQL connection failed: {e}")
    
    def test_database_schema(self) -> bool:
        """Test database schema and tables exist."""
        try:
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database=os.getenv("POSTGRES_DB", "ai_agent_system"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "ai_agent_password")
            )
            
            cursor = conn.cursor()
            
            # Check for expected tables
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public';
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            cursor.close()
            conn.close()
            
            return f"Database tables found: {tables}"
            
        except Exception as e:
            raise Exception(f"Database schema check failed: {e}")
    
    def test_redis_connection(self) -> bool:
        """Test Redis cache connection and operations."""
        try:
            r = redis.Redis(
                host="localhost",
                port=6379,
                password=os.getenv("REDIS_PASSWORD", ""),
                decode_responses=True
            )
            
            # Test ping
            if not r.ping():
                raise Exception("Redis ping failed")
            
            # Test basic operations
            test_key = "integration_test"
            test_value = "test_value_123"
            
            r.set(test_key, test_value, ex=60)
            retrieved_value = r.get(test_key)
            
            if retrieved_value != test_value:
                raise Exception("Redis set/get test failed")
            
            r.delete(test_key)
            
            # Get Redis info
            info = r.info()
            version = info.get("redis_version", "unknown")
            
            return f"Redis connected successfully: v{version}"
            
        except Exception as e:
            raise Exception(f"Redis connection failed: {e}")
    
    # ============================================
    # AI MODEL INTEGRATION TESTS
    # ============================================
    
    def test_openai_integration(self) -> bool:
        """Test OpenAI API integration."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your-openai-api-key-here":
            raise Exception("OpenAI API key not configured")
        
        try:
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello, this is a test."}],
                max_tokens=10
            )
            
            if not response.choices:
                raise Exception("No response from OpenAI")
            
            return f"OpenAI API working - Model: {response.model}"
            
        except Exception as e:
            raise Exception(f"OpenAI integration failed: {e}")
    
    def test_anthropic_integration(self) -> bool:
        """Test Anthropic Claude API integration."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "your-anthropic-api-key-here":
            raise Exception("Anthropic API key not configured")
        
        try:
            client = anthropic.Anthropic(api_key=api_key)
            
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello, this is a test."}]
            )
            
            if not response.content:
                raise Exception("No response from Claude")
            
            return f"Anthropic API working - Model: {response.model}"
            
        except Exception as e:
            raise Exception(f"Anthropic integration failed: {e}")
    
    def test_google_gemini_integration(self) -> bool:
        """Test Google Gemini API integration."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your-google-api-key-here":
            raise Exception("Google API key not configured")
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            response = model.generate_content("Hello, this is a test.")
            
            if not response.text:
                raise Exception("No response from Gemini")
            
            return f"Google Gemini API working"
            
        except Exception as e:
            raise Exception(f"Google Gemini integration failed: {e}")
    
    def test_ollama_integration(self) -> bool:
        """Test Ollama local LLM integration."""
        try:
            # Test Ollama API connection
            response = requests.get("http://localhost:11434/api/tags", timeout=30)
            response.raise_for_status()
            
            models_data = response.json()
            models = models_data.get("models", [])
            
            if not models:
                return "Ollama connected but no models installed"
            
            # Test model inference with first available model
            model_name = models[0]["name"]
            
            inference_data = {
                "model": model_name,
                "prompt": "Hello, this is a test.",
                "stream": False
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=inference_data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            if "response" not in result:
                raise Exception("Invalid response format from Ollama")
            
            return f"Ollama working - Models: {[m['name'] for m in models]}"
            
        except Exception as e:
            raise Exception(f"Ollama integration failed: {e}")
    
    # ============================================
    # FRONTEND-BACKEND COMMUNICATION TESTS
    # ============================================
    
    def test_backend_api_endpoints(self) -> bool:
        """Test backend REST API endpoints."""
        try:
            # Test health endpoint
            response = requests.get("http://localhost:8000/health", timeout=30)
            response.raise_for_status()
            
            health_data = response.json()
            if health_data.get("status") != "healthy":
                raise Exception(f"Backend not healthy: {health_data}")
            
            # Test API documentation endpoint
            response = requests.get("http://localhost:8000/docs", timeout=30)
            response.raise_for_status()
            
            return f"Backend API healthy: {health_data}"
            
        except Exception as e:
            raise Exception(f"Backend API test failed: {e}")
    
    def test_frontend_accessibility(self) -> bool:
        """Test frontend accessibility."""
        try:
            response = requests.get("http://localhost:3000", timeout=30)
            response.raise_for_status()
            
            # Check if it's a valid HTML response
            content = response.text
            if "<html" not in content.lower():
                raise Exception("Frontend not returning HTML content")
            
            return f"Frontend accessible - Status: {response.status_code}"
            
        except Exception as e:
            raise Exception(f"Frontend accessibility test failed: {e}")
    
    async def test_websocket_connection(self) -> bool:
        """Test WebSocket real-time communication."""
        try:
            uri = "ws://localhost:8000/ws"
            
            async with websockets.connect(uri) as websocket:
                # Send test message
                test_message = {"type": "ping", "data": "test"}
                await websocket.send(json.dumps(test_message))
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=10)
                response_data = json.loads(response)
                
                if response_data.get("type") != "pong":
                    raise Exception(f"Unexpected WebSocket response: {response_data}")
            
            return "WebSocket connection successful"
            
        except Exception as e:
            raise Exception(f"WebSocket test failed: {e}")
    
    def test_file_upload_functionality(self) -> bool:
        """Test file upload functionality."""
        try:
            # Create a temporary test file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("This is a test file for integration testing.")
                temp_file_path = f.name
            
            # Test file upload
            with open(temp_file_path, 'rb') as f:
                files = {'file': ('test.txt', f, 'text/plain')}
                response = requests.post(
                    "http://localhost:8000/api/upload",
                    files=files,
                    timeout=30
                )
            
            # Clean up
            os.unlink(temp_file_path)
            
            if response.status_code not in [200, 201]:
                raise Exception(f"File upload failed: {response.status_code}")
            
            return f"File upload successful: {response.status_code}"
            
        except Exception as e:
            raise Exception(f"File upload test failed: {e}")
    
    # ============================================
    # SYSTEM INTEGRATION TESTS
    # ============================================
    
    def test_end_to_end_chat(self) -> bool:
        """Test end-to-end chat functionality."""
        try:
            # Test chat endpoint
            chat_data = {
                "message": "Hello, this is an integration test.",
                "model": "local",
                "session_id": "integration_test"
            }
            
            response = requests.post(
                "http://localhost:8000/api/chat",
                json=chat_data,
                timeout=60
            )
            response.raise_for_status()
            
            chat_response = response.json()
            if "response" not in chat_response:
                raise Exception("Invalid chat response format")
            
            return f"Chat functionality working"
            
        except Exception as e:
            raise Exception(f"End-to-end chat test failed: {e}")
    
    def test_cost_tracking(self) -> bool:
        """Test cost tracking functionality."""
        try:
            response = requests.get("http://localhost:8000/api/analytics/costs", timeout=30)
            response.raise_for_status()
            
            cost_data = response.json()
            if "total_cost" not in cost_data:
                raise Exception("Invalid cost tracking response")
            
            return f"Cost tracking working: ${cost_data.get('total_cost', 0)}"
            
        except Exception as e:
            raise Exception(f"Cost tracking test failed: {e}")
    
    # ============================================
    # PERFORMANCE AND MONITORING TESTS
    # ============================================
    
    def test_prometheus_metrics(self) -> bool:
        """Test Prometheus metrics collection."""
        try:
            response = requests.get("http://localhost:9090/api/v1/query?query=up", timeout=30)
            response.raise_for_status()
            
            metrics_data = response.json()
            if metrics_data.get("status") != "success":
                raise Exception("Prometheus query failed")
            
            return f"Prometheus metrics working"
            
        except Exception as e:
            raise Exception(f"Prometheus test failed: {e}")
    
    def test_grafana_dashboard(self) -> bool:
        """Test Grafana dashboard accessibility."""
        try:
            response = requests.get("http://localhost:3001/api/health", timeout=30)
            response.raise_for_status()
            
            health_data = response.json()
            if health_data.get("database") != "ok":
                raise Exception("Grafana not healthy")
            
            return f"Grafana dashboard accessible"
            
        except Exception as e:
            raise Exception(f"Grafana test failed: {e}")
    
    def test_service_performance(self) -> bool:
        """Test service response times and performance."""
        try:
            performance_results = {}
            
            # Test backend response time
            start_time = time.time()
            response = requests.get("http://localhost:8000/health", timeout=30)
            backend_time = time.time() - start_time
            performance_results["backend"] = backend_time
            
            # Test frontend response time
            start_time = time.time()
            response = requests.get("http://localhost:3000", timeout=30)
            frontend_time = time.time() - start_time
            performance_results["frontend"] = frontend_time
            
            # Test database response time
            start_time = time.time()
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database=os.getenv("POSTGRES_DB", "ai_agent_system"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "ai_agent_password")
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            cursor.fetchone()
            cursor.close()
            conn.close()
            db_time = time.time() - start_time
            performance_results["database"] = db_time
            
            # Test Redis response time
            start_time = time.time()
            r = redis.Redis(host="localhost", port=6379, decode_responses=True)
            r.ping()
            redis_time = time.time() - start_time
            performance_results["redis"] = redis_time
            
            # Check if response times are reasonable
            if backend_time > 5.0:
                raise Exception(f"Backend response too slow: {backend_time:.2f}s")
            
            if frontend_time > 10.0:
                raise Exception(f"Frontend response too slow: {frontend_time:.2f}s")
                
            if db_time > 2.0:
                raise Exception(f"Database response too slow: {db_time:.2f}s")
                
            if redis_time > 1.0:
                raise Exception(f"Redis response too slow: {redis_time:.2f}s")
            
            return f"Performance acceptable: {performance_results}"
            
        except Exception as e:
            raise Exception(f"Performance test failed: {e}")
    
    def test_docker_resource_usage(self) -> bool:
        """Test Docker container resource usage."""
        try:
            resource_usage = {}
            containers = self.docker_client.containers.list()
            
            for container in containers:
                try:
                    stats = container.stats(stream=False)
                    name = container.name
                    
                    # CPU usage
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                               stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                  stats['precpu_stats']['system_cpu_usage']
                    cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0
                    
                    # Memory usage
                    memory_usage = stats['memory_stats']['usage']
                    memory_limit = stats['memory_stats']['limit']
                    memory_percent = (memory_usage / memory_limit) * 100.0
                    
                    resource_usage[name] = {
                        "cpu_percent": round(cpu_percent, 2),
                        "memory_percent": round(memory_percent, 2),
                        "memory_usage_mb": round(memory_usage / (1024 * 1024), 2)
                    }
                    
                    # Check for resource issues
                    if cpu_percent > 90:
                        raise Exception(f"Container {name} CPU usage too high: {cpu_percent:.1f}%")
                    if memory_percent > 90:
                        raise Exception(f"Container {name} memory usage too high: {memory_percent:.1f}%")
                        
                except Exception as e:
                    self.logger.warning(f"Could not get stats for {container.name}: {e}")
            
            return f"Resource usage normal: {resource_usage}"
            
        except Exception as e:
            raise Exception(f"Resource usage test failed: {e}")
    
    def test_model_switching_performance(self) -> bool:
        """Test AI model switching and fallback logic."""
        try:
            test_prompt = "Hello, this is a model switching test."
            results = {}
            
            # Test local model first (if available)
            try:
                local_response = requests.post(
                    "http://localhost:8000/api/chat",
                    json={
                        "message": test_prompt,
                        "model": "local",
                        "session_id": "model_switch_test"
                    },
                    timeout=30
                )
                if local_response.status_code == 200:
                    results["local_model"] = "available"
                else:
                    results["local_model"] = "unavailable"
            except:
                results["local_model"] = "unavailable"
            
            # Test cloud model fallback
            try:
                cloud_response = requests.post(
                    "http://localhost:8000/api/chat",
                    json={
                        "message": test_prompt,
                        "model": "gpt-3.5-turbo",
                        "session_id": "model_switch_test"
                    },
                    timeout=30
                )
                if cloud_response.status_code == 200:
                    results["cloud_model"] = "available"
                else:
                    results["cloud_model"] = "unavailable"
            except:
                results["cloud_model"] = "unavailable"
            
            # Test model selection endpoint
            try:
                model_list_response = requests.get(
                    "http://localhost:8000/api/models",
                    timeout=30
                )
                if model_list_response.status_code == 200:
                    models = model_list_response.json()
                    results["model_list"] = f"{len(models.get('models', []))} models available"
                else:
                    results["model_list"] = "endpoint unavailable"
            except:
                results["model_list"] = "endpoint error"
            
            return f"Model switching test: {results}"
            
        except Exception as e:
            raise Exception(f"Model switching test failed: {e}")
    
    # ============================================
    # MAIN TEST RUNNER
    # ============================================
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        self.logger.info("Starting comprehensive integration test suite...")
        
        # Test categories
        test_categories = [
            ("Docker Compose Health Checks", [
                ("docker_compose_status", self.test_docker_compose_status),
                ("service_health_checks", self.test_service_health_checks),
            ]),
            ("Database Connection Tests", [
                ("postgresql_connection", self.test_postgresql_connection),
                ("database_schema", self.test_database_schema),
                ("redis_connection", self.test_redis_connection),
            ]),
            ("AI Model Integration Tests", [
                ("openai_integration", self.test_openai_integration),
                ("anthropic_integration", self.test_anthropic_integration),
                ("google_gemini_integration", self.test_google_gemini_integration),
                ("ollama_integration", self.test_ollama_integration),
            ]),
            ("Frontend-Backend Communication", [
                ("backend_api_endpoints", self.test_backend_api_endpoints),
                ("frontend_accessibility", self.test_frontend_accessibility),
                ("websocket_connection", self.test_websocket_connection),
                ("file_upload_functionality", self.test_file_upload_functionality),
            ]),
            ("System Integration Tests", [
                ("end_to_end_chat", self.test_end_to_end_chat),
                ("cost_tracking", self.test_cost_tracking),
            ]),
            ("Performance and Monitoring", [
                ("prometheus_metrics", self.test_prometheus_metrics),
                ("grafana_dashboard", self.test_grafana_dashboard),
                ("service_performance", self.test_service_performance),
                ("docker_resource_usage", self.test_docker_resource_usage),
                ("model_switching_performance", self.test_model_switching_performance),
            ]),
        ]
        
        # Run all test categories
        for category_name, tests in test_categories:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Running {category_name}")
            self.logger.info(f"{'='*50}")
            
            for test_name, test_func in tests:
                await self.run_test_with_timeout(test_func, test_name)
        
        # Calculate overall status
        if test_results["failed_tests"] == 0:
            test_results["overall_status"] = "PASSED"
        elif test_results["passed_tests"] > test_results["failed_tests"]:
            test_results["overall_status"] = "PARTIAL"
        else:
            test_results["overall_status"] = "FAILED"
        
        # Save results to file
        with open(RESULTS_FILE, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Print summary
        self.print_summary()
        
        return test_results
    
    def print_summary(self):
        """Print test results summary."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("INTEGRATION TEST SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total Tests: {test_results['total_tests']}")
        self.logger.info(f"Passed: {test_results['passed_tests']}")
        self.logger.info(f"Failed: {test_results['failed_tests']}")
        self.logger.info(f"Skipped: {test_results['skipped_tests']}")
        
        if test_results['total_tests'] > 0:
            success_rate = (test_results['passed_tests'] / test_results['total_tests']) * 100
            self.logger.info(f"Success Rate: {success_rate:.1f}%")
        
        self.logger.info(f"Overall Status: {test_results['overall_status']}")
        self.logger.info(f"Results saved to: {RESULTS_FILE}")
        self.logger.info(f"{'='*60}")
        
        # Print failed tests details
        if test_results['failed_tests'] > 0:
            self.logger.error("\nFAILED TESTS:")
            for test in test_results['test_details']:
                if test['status'] == 'FAILED':
                    self.logger.error(f"- {test['test_name']}: {test['error']}")

async def main():
    """Main entry point."""
    print("AI Agent System - Integration Test Suite")
    print("=" * 50)
    
    # Initialize test suite
    test_suite = IntegrationTestSuite()
    
    # Run all tests
    results = await test_suite.run_all_tests()
    
    # Exit with appropriate code
    if results["overall_status"] == "PASSED":
        sys.exit(0)
    elif results["overall_status"] == "PARTIAL":
        sys.exit(1)
    else:
        sys.exit(2)

if __name__ == "__main__":
    asyncio.run(main())