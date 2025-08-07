"""GPU/CPU Optimization System.

Enterprise-grade GPU/CPU optimization with CUDA memory management,
multi-GPU model sharding, CPU fallback, and dynamic resource allocation.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import psutil

from app.core.logger import LoggerMixin
from app.core.config import settings
from app.database.models import ModelProvider


class DeviceType(str, Enum):
    """Device types for computation"""
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"  # Apple Metal Performance Shaders
    XPU = "xpu"  # Intel XPU


class MemoryStrategy(str, Enum):
    """Memory management strategies"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    STREAMING = "streaming"


class ShardingStrategy(str, Enum):
    """Model sharding strategies"""
    LAYER_WISE = "layer_wise"
    TENSOR_PARALLEL = "tensor_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID = "hybrid"


@dataclass
class GPUResource:
    """GPU resource information with optimization metadata"""
    device_id: str
    name: str
    compute_capability: Tuple[int, int]
    memory_total_mb: float
    memory_free_mb: float
    memory_used_mb: float
    utilization_percent: float
    temperature: float
    power_usage_watts: float
    memory_clock_mhz: int
    graphics_clock_mhz: int
    allocated_models: List[str] = field(default_factory=list)
    memory_pools: Dict[str, float] = field(default_factory=dict)
    is_available: bool = True
    optimization_flags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CPUResource:
    """CPU resource information"""
    cpu_count: int
    cpu_count_physical: int
    cpu_percent: float
    memory_total_mb: float
    memory_available_mb: float
    memory_percent: float
    load_average: Tuple[float, float, float]
    allocated_threads: int = 0
    optimization_flags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelAllocation:
    """Model resource allocation information"""
    model_key: str
    device_assignments: Dict[str, List[str]]  # device_type -> device_ids
    memory_allocations: Dict[str, float]  # device_id -> memory_mb
    thread_allocations: Dict[str, int]  # device_id -> thread_count
    sharding_config: Optional[Dict[str, Any]] = None
    fallback_devices: List[str] = field(default_factory=list)
    optimization_metadata: Dict[str, Any] = field(default_factory=dict)


class GPUCPUOptimizer(LoggerMixin):
    """Enterprise GPU/CPU optimization system"""
    
    def __init__(self):
        super().__init__()
        self.gpu_resources: Dict[str, GPUResource] = {}
        self.cpu_resource: Optional[CPUResource] = None
        self.model_allocations: Dict[str, ModelAllocation] = {}
        
        # Optimization settings
        self.memory_safety_margin = 0.1  # 10% safety margin
        self.gpu_utilization_threshold = 0.8  # 80% utilization threshold
        self.temperature_threshold = 85.0  # 85°C temperature threshold
        self.memory_defragmentation_interval = 300  # 5 minutes
        
        # CUDA optimization settings
        self.cuda_cache_config = "max_split_size_mb:512"
        self.cuda_memory_fraction = 0.9
        self.enable_tf32 = True
        self.enable_cudnn_benchmark = True
        
        # Multi-GPU settings
        self.enable_nccl = True
        self.nccl_timeout_seconds = 600
        self.tensor_parallel_size = 1
        self.pipeline_parallel_size = 1
        
        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._optimization_task: Optional[asyncio.Task] = None
        self._defrag_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Statistics
        self.stats = {
            "memory_optimizations": 0,
            "device_migrations": 0,
            "fallback_activations": 0,
            "oom_recoveries": 0,
            "sharding_operations": 0,
            "defragmentations": 0
        }
        
        # Thread locks
        self._allocation_lock = threading.Lock()
        self._optimization_lock = threading.Lock()
    
    async def initialize(self):
        """Initialize the GPU/CPU optimization system"""
        try:
            # Initialize CUDA if available
            await self._initialize_cuda()
            
            # Discover GPU resources
            await self._discover_gpu_resources()
            
            # Initialize CPU resources
            await self._initialize_cpu_resources()
            
            # Setup optimization settings
            await self._setup_optimization_settings()
            
            # Start background tasks
            self._monitor_task = asyncio.create_task(self._resource_monitor_loop())
            self._optimization_task = asyncio.create_task(self._optimization_loop())
            self._defrag_task = asyncio.create_task(self._memory_defrag_loop())
            
            self.logger.info(f"GPU/CPU optimizer initialized with {len(self.gpu_resources)} GPUs")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GPU/CPU optimizer: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the optimization system"""
        self.logger.info("Shutting down GPU/CPU optimizer")
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in [self._monitor_task, self._optimization_task, self._defrag_task]:
            if task:
                task.cancel()
        
        # Cleanup all allocations
        await self._cleanup_all_allocations()
    
    async def _initialize_cuda(self):
        """Initialize CUDA settings"""
        try:
            import torch
            
            if torch.cuda.is_available():
                # Set CUDA optimization flags
                torch.backends.cuda.matmul.allow_tf32 = self.enable_tf32
                torch.backends.cudnn.benchmark = self.enable_cudnn_benchmark
                torch.backends.cudnn.deterministic = False
                
                # Set memory management
                torch.cuda.set_per_process_memory_fraction(self.cuda_memory_fraction)
                
                # Configure memory allocator
                import os
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = self.cuda_cache_config
                
                self.logger.info("CUDA optimization settings configured")
            else:
                self.logger.warning("CUDA not available")
                
        except ImportError:
            self.logger.warning("PyTorch not available for CUDA optimization")
        except Exception as e:
            self.logger.error(f"CUDA initialization failed: {e}")
    
    async def _discover_gpu_resources(self):
        """Discover and analyze GPU resources"""
        try:
            # Try NVIDIA GPUs first
            await self._discover_nvidia_gpus()
            
            # Try AMD GPUs
            await self._discover_amd_gpus()
            
            # Try Intel GPUs
            await self._discover_intel_gpus()
            
            self.logger.info(f"Discovered {len(self.gpu_resources)} GPU devices")
            
        except Exception as e:
            self.logger.error(f"GPU discovery failed: {e}")
    
    async def _discover_nvidia_gpus(self):
        """Discover NVIDIA GPUs"""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get device info
                name = pynvml.nvmlDeviceGetName(handle).decode()
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                
                # Get compute capability
                major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
                minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
                
                # Get clock speeds
                memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                
                gpu_resource = GPUResource(
                    device_id=f"cuda:{i}",
                    name=name,
                    compute_capability=(major, minor),
                    memory_total_mb=memory_info.total / 1024 / 1024,
                    memory_free_mb=memory_info.free / 1024 / 1024,
                    memory_used_mb=memory_info.used / 1024 / 1024,
                    utilization_percent=utilization.gpu,
                    temperature=temperature,
                    power_usage_watts=power_usage,
                    memory_clock_mhz=memory_clock,
                    graphics_clock_mhz=graphics_clock
                )
                
                self.gpu_resources[f"cuda:{i}"] = gpu_resource
                
        except ImportError:
            self.logger.warning("pynvml not available for NVIDIA GPU monitoring")
        except Exception as e:
            self.logger.error(f"NVIDIA GPU discovery failed: {e}")
    
    async def _discover_amd_gpus(self):
        """Discover AMD GPUs"""
        try:
            # AMD GPU discovery would go here
            # This is a placeholder for AMD-specific implementation
            self.logger.info("AMD GPU discovery not implemented yet")
            
        except Exception as e:
            self.logger.error(f"AMD GPU discovery failed: {e}")
    
    async def _discover_intel_gpus(self):
        """Discover Intel GPUs"""
        try:
            # Intel GPU discovery would go here
            # This is a placeholder for Intel-specific implementation
            self.logger.info("Intel GPU discovery not implemented yet")
            
        except Exception as e:
            self.logger.error(f"Intel GPU discovery failed: {e}")
    
    async def _initialize_cpu_resources(self):
        """Initialize CPU resource monitoring"""
        try:
            self.cpu_resource = CPUResource(
                cpu_count=psutil.cpu_count(),
                cpu_count_physical=psutil.cpu_count(logical=False),
                cpu_percent=psutil.cpu_percent(),
                memory_total_mb=psutil.virtual_memory().total / 1024 / 1024,
                memory_available_mb=psutil.virtual_memory().available / 1024 / 1024,
                memory_percent=psutil.virtual_memory().percent,
                load_average=psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0.0, 0.0, 0.0)
            )
            
            self.logger.info(f"CPU resources initialized: {self.cpu_resource.cpu_count} cores, "
                           f"{self.cpu_resource.memory_total_mb:.0f}MB memory")
            
        except Exception as e:
            self.logger.error(f"CPU resource initialization failed: {e}")
    
    async def _setup_optimization_settings(self):
        """Setup optimization settings based on available hardware"""
        try:
            # Configure multi-GPU settings
            if len(self.gpu_resources) > 1:
                self.tensor_parallel_size = min(len(self.gpu_resources), 4)
                
                # Setup NCCL for multi-GPU communication
                if self.enable_nccl:
                    import os
                    os.environ["NCCL_TIMEOUT"] = str(self.nccl_timeout_seconds)
                    os.environ["NCCL_BLOCKING_WAIT"] = "1"
            
            # Set CPU optimization flags
            if self.cpu_resource:
                import torch
                if torch.get_num_threads() != self.cpu_resource.cpu_count:
                    torch.set_num_threads(self.cpu_resource.cpu_count)
            
            self.logger.info("Optimization settings configured")
            
        except Exception as e:
            self.logger.error(f"Optimization setup failed: {e}")
    
    async def allocate_resources(
        self,
        model_key: str,
        memory_requirement_mb: float,
        compute_requirement: str = "auto",
        sharding_strategy: ShardingStrategy = ShardingStrategy.LAYER_WISE,
        enable_fallback: bool = True
    ) -> ModelAllocation:
        """Allocate optimal resources for a model"""
        
        with self._allocation_lock:
            try:
                # Find optimal device allocation
                allocation = await self._find_optimal_allocation(
                    model_key, memory_requirement_mb, compute_requirement, sharding_strategy
                )
                
                if not allocation and enable_fallback:
                    # Try CPU fallback
                    allocation = await self._create_cpu_fallback_allocation(
                        model_key, memory_requirement_mb
                    )
                    self.stats["fallback_activations"] += 1
                
                if not allocation:
                    raise RuntimeError(f"Cannot allocate resources for model {model_key}")
                
                # Reserve resources
                await self._reserve_resources(allocation)
                
                self.model_allocations[model_key] = allocation
                
                self.logger.info(f"Resources allocated for {model_key}: {allocation.device_assignments}")
                return allocation
                
            except Exception as e:
                self.logger.error(f"Resource allocation failed for {model_key}: {e}")
                raise
    
    async def _find_optimal_allocation(
        self,
        model_key: str,
        memory_requirement_mb: float,
        compute_requirement: str,
        sharding_strategy: ShardingStrategy
    ) -> Optional[ModelAllocation]:
        """Find optimal resource allocation"""
        
        # Try single GPU allocation first
        single_gpu_allocation = await self._try_single_gpu_allocation(
            model_key, memory_requirement_mb
        )
        
        if single_gpu_allocation:
            return single_gpu_allocation
        
        # Try multi-GPU allocation with sharding
        if len(self.gpu_resources) > 1:
            multi_gpu_allocation = await self._try_multi_gpu_allocation(
                model_key, memory_requirement_mb, sharding_strategy
            )
            
            if multi_gpu_allocation:
                self.stats["sharding_operations"] += 1
                return multi_gpu_allocation
        
        return None
    
    async def _try_single_gpu_allocation(
        self,
        model_key: str,
        memory_requirement_mb: float
    ) -> Optional[ModelAllocation]:
        """Try to allocate model to single GPU"""
        
        # Sort GPUs by available memory and performance
        available_gpus = []
        for device_id, gpu in self.gpu_resources.items():
            if not gpu.is_available:
                continue
                
            available_memory = gpu.memory_free_mb * (1 - self.memory_safety_margin)
            
            if available_memory >= memory_requirement_mb:
                # Score based on available memory and compute capability
                score = available_memory + (gpu.compute_capability[0] * 1000)
                available_gpus.append((device_id, gpu, score))
        
        if not available_gpus:
            return None
        
        # Select best GPU
        best_device_id, best_gpu, _ = max(available_gpus, key=lambda x: x[2])
        
        allocation = ModelAllocation(
            model_key=model_key,
            device_assignments={"cuda": [best_device_id]},
            memory_allocations={best_device_id: memory_requirement_mb},
            thread_allocations={},
            optimization_metadata={
                "single_gpu": True,
                "compute_capability": best_gpu.compute_capability
            }
        )
        
        return allocation
    
    async def _try_multi_gpu_allocation(
        self,
        model_key: str,
        memory_requirement_mb: float,
        sharding_strategy: ShardingStrategy
    ) -> Optional[ModelAllocation]:
        """Try to allocate model across multiple GPUs"""
        
        available_gpus = [
            (device_id, gpu) for device_id, gpu in self.gpu_resources.items()
            if gpu.is_available and gpu.memory_free_mb > 0
        ]
        
        if len(available_gpus) < 2:
            return None
        
        # Calculate sharding configuration
        sharding_config = await self._calculate_sharding_config(
            available_gpus, memory_requirement_mb, sharding_strategy
        )
        
        if not sharding_config:
            return None
        
        device_assignments = {"cuda": list(sharding_config.keys())}
        memory_allocations = sharding_config
        
        allocation = ModelAllocation(
            model_key=model_key,
            device_assignments=device_assignments,
            memory_allocations=memory_allocations,
            thread_allocations={},
            sharding_config={
                "strategy": sharding_strategy,
                "num_shards": len(sharding_config)
            },
            optimization_metadata={
                "multi_gpu": True,
                "sharding_strategy": sharding_strategy.value
            }
        )
        
        return allocation
    
    async def _calculate_sharding_config(
        self,
        available_gpus: List[Tuple[str, GPUResource]],
        memory_requirement_mb: float,
        sharding_strategy: ShardingStrategy
    ) -> Optional[Dict[str, float]]:
        """Calculate optimal sharding configuration"""
        
        if sharding_strategy == ShardingStrategy.LAYER_WISE:
            return await self._calculate_layer_wise_sharding(available_gpus, memory_requirement_mb)
        elif sharding_strategy == ShardingStrategy.TENSOR_PARALLEL:
            return await self._calculate_tensor_parallel_sharding(available_gpus, memory_requirement_mb)
        elif sharding_strategy == ShardingStrategy.PIPELINE_PARALLEL:
            return await self._calculate_pipeline_parallel_sharding(available_gpus, memory_requirement_mb)
        else:
            return await self._calculate_hybrid_sharding(available_gpus, memory_requirement_mb)
    
    async def _calculate_layer_wise_sharding(
        self,
        available_gpus: List[Tuple[str, GPUResource]],
        memory_requirement_mb: float
    ) -> Optional[Dict[str, float]]:
        """Calculate layer-wise sharding configuration"""
        
        # Sort GPUs by available memory
        sorted_gpus = sorted(available_gpus, key=lambda x: x[1].memory_free_mb, reverse=True)
        
        total_available_memory = sum(
            gpu.memory_free_mb * (1 - self.memory_safety_margin)
            for _, gpu in sorted_gpus
        )
        
        if total_available_memory < memory_requirement_mb:
            return None
        
        # Distribute memory proportionally
        sharding_config = {}
        remaining_memory = memory_requirement_mb
        
        for device_id, gpu in sorted_gpus:
            if remaining_memory <= 0:
                break
                
            available_memory = gpu.memory_free_mb * (1 - self.memory_safety_margin)
            allocated_memory = min(remaining_memory, available_memory)
            
            if allocated_memory > 0:
                sharding_config[device_id] = allocated_memory
                remaining_memory -= allocated_memory
        
        return sharding_config if remaining_memory <= 0 else None
    
    async def _calculate_tensor_parallel_sharding(
        self,
        available_gpus: List[Tuple[str, GPUResource]],
        memory_requirement_mb: float
    ) -> Optional[Dict[str, float]]:
        """Calculate tensor parallel sharding configuration"""
        
        # For tensor parallelism, distribute memory equally
        num_gpus = len(available_gpus)
        memory_per_gpu = memory_requirement_mb / num_gpus
        
        sharding_config = {}
        
        for device_id, gpu in available_gpus:
            available_memory = gpu.memory_free_mb * (1 - self.memory_safety_margin)
            
            if available_memory >= memory_per_gpu:
                sharding_config[device_id] = memory_per_gpu
            else:
                return None  # Not enough memory on this GPU
        
        return sharding_config
    
    async def _calculate_pipeline_parallel_sharding(
        self,
        available_gpus: List[Tuple[str, GPUResource]],
        memory_requirement_mb: float
    ) -> Optional[Dict[str, float]]:
        """Calculate pipeline parallel sharding configuration"""
        
        # For pipeline parallelism, each GPU gets a portion of the model
        # Memory requirements vary by stage
        
        num_stages = len(available_gpus)
        base_memory_per_stage = memory_requirement_mb / num_stages
        
        # First and last stages typically need more memory
        stage_multipliers = [1.2] + [0.9] * (num_stages - 2) + [1.2] if num_stages > 2 else [1.0, 1.0]
        
        sharding_config = {}
        
        for i, (device_id, gpu) in enumerate(available_gpus):
            required_memory = base_memory_per_stage * stage_multipliers[min(i, len(stage_multipliers) - 1)]
            available_memory = gpu.memory_free_mb * (1 - self.memory_safety_margin)
            
            if available_memory >= required_memory:
                sharding_config[device_id] = required_memory
            else:
                return None
        
        return sharding_config
    
    async def _calculate_hybrid_sharding(
        self,
        available_gpus: List[Tuple[str, GPUResource]],
        memory_requirement_mb: float
    ) -> Optional[Dict[str, float]]:
        """Calculate hybrid sharding configuration"""
        
        # Try tensor parallel first, then layer-wise fallback
        tensor_config = await self._calculate_tensor_parallel_sharding(available_gpus, memory_requirement_mb)
        
        if tensor_config:
            return tensor_config
        
        return await self._calculate_layer_wise_sharding(available_gpus, memory_requirement_mb)
    
    async def _create_cpu_fallback_allocation(
        self,
        model_key: str,
        memory_requirement_mb: float
    ) -> Optional[ModelAllocation]:
        """Create CPU fallback allocation"""
        
        if not self.cpu_resource:
            return None
        
        if self.cpu_resource.memory_available_mb < memory_requirement_mb:
            return None
        
        # Calculate optimal CPU thread allocation
        optimal_threads = min(
            self.cpu_resource.cpu_count,
            max(1, self.cpu_resource.cpu_count // 2)  # Use half the cores for one model
        )
        
        allocation = ModelAllocation(
            model_key=model_key,
            device_assignments={"cpu": ["cpu:0"]},
            memory_allocations={"cpu:0": memory_requirement_mb},
            thread_allocations={"cpu:0": optimal_threads},
            optimization_metadata={
                "cpu_fallback": True,
                "cpu_threads": optimal_threads
            }
        )
        
        return allocation
    
    async def _reserve_resources(self, allocation: ModelAllocation):
        """Reserve resources for allocation"""
        
        # Reserve GPU memory
        for device_id, memory_mb in allocation.memory_allocations.items():
            if device_id.startswith("cuda:"):
                gpu = self.gpu_resources.get(device_id)
                if gpu:
                    gpu.memory_free_mb -= memory_mb
                    gpu.memory_used_mb += memory_mb
                    gpu.allocated_models.append(allocation.model_key)
            
        # Reserve CPU threads
        for device_id, threads in allocation.thread_allocations.items():
            if device_id.startswith("cpu:"):
                if self.cpu_resource:
                    self.cpu_resource.allocated_threads += threads
    
    async def deallocate_resources(self, model_key: str):
        """Deallocate resources for a model"""
        
        with self._allocation_lock:
            try:
                allocation = self.model_allocations.get(model_key)
                if not allocation:
                    return
                
                # Free GPU memory
                for device_id, memory_mb in allocation.memory_allocations.items():
                    if device_id.startswith("cuda:"):
                        gpu = self.gpu_resources.get(device_id)
                        if gpu:
                            gpu.memory_free_mb += memory_mb
                            gpu.memory_used_mb -= memory_mb
                            if allocation.model_key in gpu.allocated_models:
                                gpu.allocated_models.remove(allocation.model_key)
                
                # Free CPU threads
                for device_id, threads in allocation.thread_allocations.items():
                    if device_id.startswith("cpu:"):
                        if self.cpu_resource:
                            self.cpu_resource.allocated_threads -= threads
                
                del self.model_allocations[model_key]
                
                self.logger.info(f"Resources deallocated for {model_key}")
                
            except Exception as e:
                self.logger.error(f"Resource deallocation failed for {model_key}: {e}")
    
    async def optimize_memory_usage(self, model_key: str) -> bool:
        """Optimize memory usage for a specific model"""
        
        with self._optimization_lock:
            try:
                allocation = self.model_allocations.get(model_key)
                if not allocation:
                    return False
                
                # Try memory defragmentation
                defrag_success = await self._defragment_gpu_memory(allocation)
                
                # Try memory compaction
                compact_success = await self._compact_model_memory(allocation)
                
                # Try gradient checkpointing if applicable
                checkpoint_success = await self._enable_gradient_checkpointing(allocation)
                
                if any([defrag_success, compact_success, checkpoint_success]):
                    self.stats["memory_optimizations"] += 1
                    return True
                
                return False
                
            except Exception as e:
                self.logger.error(f"Memory optimization failed for {model_key}: {e}")
                return False
    
    async def _defragment_gpu_memory(self, allocation: ModelAllocation) -> bool:
        """Defragment GPU memory for allocation"""
        try:
            import torch
            
            for device_id in allocation.device_assignments.get("cuda", []):
                if device_id.startswith("cuda:"):
                    device_num = int(device_id.split(":")[1])
                    
                    # Clear cache and defragment
                    torch.cuda.empty_cache()
                    
                    if hasattr(torch.cuda, 'memory_reserved'):
                        # PyTorch 1.4+
                        reserved_before = torch.cuda.memory_reserved(device_num)
                        torch.cuda.synchronize(device_num)
                        reserved_after = torch.cuda.memory_reserved(device_num)
                        
                        if reserved_after < reserved_before:
                            self.logger.info(f"Defragmented {reserved_before - reserved_after} bytes on {device_id}")
                            return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"GPU memory defragmentation failed: {e}")
            return False
    
    async def _compact_model_memory(self, allocation: ModelAllocation) -> bool:
        """Compact model memory usage"""
        try:
            # This would implement model-specific memory compaction
            # Placeholder for actual implementation
            return False
            
        except Exception as e:
            self.logger.error(f"Model memory compaction failed: {e}")
            return False
    
    async def _enable_gradient_checkpointing(self, allocation: ModelAllocation) -> bool:
        """Enable gradient checkpointing to reduce memory usage"""
        try:
            # This would enable gradient checkpointing for the model
            # Placeholder for actual implementation
            return False
            
        except Exception as e:
            self.logger.error(f"Gradient checkpointing failed: {e}")
            return False
    
    async def handle_oom_error(self, model_key: str) -> bool:
        """Handle out-of-memory error"""
        try:
            self.logger.warning(f"Handling OOM error for {model_key}")
            
            # Try memory optimization first
            if await self.optimize_memory_usage(model_key):
                self.stats["oom_recoveries"] += 1
                return True
            
            # Try moving to CPU fallback
            allocation = self.model_allocations.get(model_key)
            if allocation and "cuda" in allocation.device_assignments:
                
                # Calculate memory requirement
                total_memory = sum(allocation.memory_allocations.values())
                
                # Deallocate current resources
                await self.deallocate_resources(model_key)
                
                # Try CPU fallback
                cpu_allocation = await self._create_cpu_fallback_allocation(model_key, total_memory)
                
                if cpu_allocation:
                    await self._reserve_resources(cpu_allocation)
                    self.model_allocations[model_key] = cpu_allocation
                    self.stats["fallback_activations"] += 1
                    self.stats["oom_recoveries"] += 1
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"OOM error handling failed: {e}")
            return False
    
    async def _resource_monitor_loop(self):
        """Background task for monitoring resource usage"""
        while not self._shutdown_event.is_set():
            try:
                await self._update_gpu_metrics()
                await self._update_cpu_metrics()
                await self._check_resource_health()
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Resource monitor error: {e}")
                await asyncio.sleep(10)
    
    async def _optimization_loop(self):
        """Background task for continuous optimization"""
        while not self._shutdown_event.is_set():
            try:
                await self._optimize_resource_allocation()
                await self._balance_gpu_load()
                await self._check_temperature_throttling()
                
                await asyncio.sleep(30)  # Optimize every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(30)
    
    async def _memory_defrag_loop(self):
        """Background task for memory defragmentation"""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_memory_defragmentation()
                await asyncio.sleep(self.memory_defragmentation_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Memory defrag loop error: {e}")
                await asyncio.sleep(self.memory_defragmentation_interval)
    
    async def _update_gpu_metrics(self):
        """Update GPU metrics"""
        try:
            import pynvml
            
            for device_id, gpu in self.gpu_resources.items():
                if device_id.startswith("cuda:"):
                    device_num = int(device_id.split(":")[1])
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_num)
                    
                    # Update memory info
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu.memory_free_mb = memory_info.free / 1024 / 1024
                    gpu.memory_used_mb = memory_info.used / 1024 / 1024
                    
                    # Update utilization
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu.utilization_percent = utilization.gpu
                    
                    # Update temperature
                    gpu.temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    # Update power usage
                    gpu.power_usage_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    
        except Exception as e:
            self.logger.error(f"GPU metrics update failed: {e}")
    
    async def _update_cpu_metrics(self):
        """Update CPU metrics"""
        try:
            if self.cpu_resource:
                memory = psutil.virtual_memory()
                
                self.cpu_resource.cpu_percent = psutil.cpu_percent()
                self.cpu_resource.memory_available_mb = memory.available / 1024 / 1024
                self.cpu_resource.memory_percent = memory.percent
                
                if hasattr(psutil, 'getloadavg'):
                    self.cpu_resource.load_average = psutil.getloadavg()
                    
        except Exception as e:
            self.logger.error(f"CPU metrics update failed: {e}")
    
    async def _check_resource_health(self):
        """Check resource health and availability"""
        try:
            for device_id, gpu in self.gpu_resources.items():
                # Check temperature
                if gpu.temperature > self.temperature_threshold:
                    self.logger.warning(f"GPU {device_id} temperature high: {gpu.temperature}°C")
                    gpu.is_available = False
                else:
                    gpu.is_available = True
                
                # Check utilization
                if gpu.utilization_percent > self.gpu_utilization_threshold * 100:
                    self.logger.info(f"GPU {device_id} high utilization: {gpu.utilization_percent}%")
                    
        except Exception as e:
            self.logger.error(f"Resource health check failed: {e}")
    
    async def _optimize_resource_allocation(self):
        """Optimize current resource allocations"""
        try:
            # Find underutilized GPUs and overutilized GPUs
            underutilized = []
            overutilized = []
            
            for device_id, gpu in self.gpu_resources.items():
                if gpu.utilization_percent < 30 and len(gpu.allocated_models) > 0:
                    underutilized.append((device_id, gpu))
                elif gpu.utilization_percent > 90:
                    overutilized.append((device_id, gpu))
            
            # Try to balance load
            if underutilized and overutilized:
                await self._balance_load_between_gpus(underutilized, overutilized)
                
        except Exception as e:
            self.logger.error(f"Resource allocation optimization failed: {e}")
    
    async def _balance_gpu_load(self):
        """Balance load between GPUs"""
        try:
            # Implementation for load balancing
            pass
            
        except Exception as e:
            self.logger.error(f"GPU load balancing failed: {e}")
    
    async def _check_temperature_throttling(self):
        """Check and handle temperature throttling"""
        try:
            for device_id, gpu in self.gpu_resources.items():
                if gpu.temperature > self.temperature_threshold * 0.9:  # 90% of threshold
                    self.logger.warning(f"GPU {device_id} approaching temperature limit")
                    
                    # Reduce GPU usage or migrate models
                    await self._reduce_gpu_load(device_id)
                    
        except Exception as e:
            self.logger.error(f"Temperature throttling check failed: {e}")
    
    async def _perform_memory_defragmentation(self):
        """Perform periodic memory defragmentation"""
        try:
            import torch
            
            for device_id in self.gpu_resources:
                if device_id.startswith("cuda:"):
                    torch.cuda.empty_cache()
                    
            self.stats["defragmentations"] += 1
            
        except Exception as e:
            self.logger.error(f"Memory defragmentation failed: {e}")
    
    async def _balance_load_between_gpus(self, underutilized, overutilized):
        """Balance load between underutilized and overutilized GPUs"""
        # Implementation for load balancing between GPUs
        pass
    
    async def _reduce_gpu_load(self, device_id: str):
        """Reduce load on a specific GPU"""
        # Implementation for reducing GPU load
        pass
    
    async def _cleanup_all_allocations(self):
        """Cleanup all resource allocations"""
        for model_key in list(self.model_allocations.keys()):
            await self.deallocate_resources(model_key)
    
    async def get_resource_stats(self) -> Dict[str, Any]:
        """Get comprehensive resource statistics"""
        gpu_stats = {}
        for device_id, gpu in self.gpu_resources.items():
            gpu_stats[device_id] = {
                "name": gpu.name,
                "memory_total_mb": gpu.memory_total_mb,
                "memory_used_mb": gpu.memory_used_mb,
                "memory_free_mb": gpu.memory_free_mb,
                "utilization_percent": gpu.utilization_percent,
                "temperature": gpu.temperature,
                "power_usage_watts": gpu.power_usage_watts,
                "allocated_models": gpu.allocated_models,
                "is_available": gpu.is_available
            }
        
        cpu_stats = {}
        if self.cpu_resource:
            cpu_stats = {
                "cpu_count": self.cpu_resource.cpu_count,
                "cpu_percent": self.cpu_resource.cpu_percent,
                "memory_total_mb": self.cpu_resource.memory_total_mb,
                "memory_available_mb": self.cpu_resource.memory_available_mb,
                "memory_percent": self.cpu_resource.memory_percent,
                "allocated_threads": self.cpu_resource.allocated_threads
            }
        
        return {
            "gpu_stats": gpu_stats,
            "cpu_stats": cpu_stats,
            "model_allocations": {
                k: {
                    "device_assignments": v.device_assignments,
                    "memory_allocations": v.memory_allocations,
                    "thread_allocations": v.thread_allocations
                }
                for k, v in self.model_allocations.items()
            },
            "optimization_stats": self.stats
        }


# Global instance
_optimizer: Optional[GPUCPUOptimizer] = None


async def get_gpu_cpu_optimizer() -> GPUCPUOptimizer:
    """Get the global GPU/CPU optimizer instance"""
    global _optimizer
    if _optimizer is None:
        _optimizer = GPUCPUOptimizer()
        await _optimizer.initialize()
    return _optimizer