"""
Configuration loader for GPU Server

Loads and validates configuration from YAML files.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    node_id: str = "gpu-node-1"
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1


@dataclass
class SecurityConfig:
    internal_auth_token: str = ""
    require_auth: bool = True


@dataclass
class AssetServiceConfig:
    callback_url: str = ""
    internal_auth_token: str = ""
    timeout: int = 10
    retries: int = 3
    retry_backoff: list = field(default_factory=lambda: [1, 2, 4])


@dataclass
class LoadBalancerConfig:
    url: str = ""
    internal_auth_token: str = ""


@dataclass
class ModelConfig:
    model_type: str = "qwen"
    model_version: str = "1.0.0"
    device: str = "cuda"
    default_mode: str = "fp8"
    default_steps: int = 4
    default_seed: int = 42
    default_cfg: float = 1.0
    default_resolution: str = "720p"
    enable_teacache: bool = False
    teacache_thresh: float = 0.05


@dataclass
class WorkflowPreprocessConfig:
    resize_to_resolution: bool = True
    maintain_aspect_ratio: bool = True


@dataclass
class WorkflowPostprocessConfig:
    create_comparison: bool = False
    compress_output: bool = False


@dataclass
class WorkflowAdvancedConfig:
    warmup_on_startup: bool = True
    clear_cache_between_jobs: bool = True


@dataclass
class WorkflowConfig:
    preprocess: WorkflowPreprocessConfig = field(default_factory=WorkflowPreprocessConfig)
    postprocess: WorkflowPostprocessConfig = field(default_factory=WorkflowPostprocessConfig)
    advanced: WorkflowAdvancedConfig = field(default_factory=WorkflowAdvancedConfig)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class GPUServerConfig:
    """Complete GPU Server configuration."""
    server: ServerConfig = field(default_factory=ServerConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    asset_service: AssetServiceConfig = field(default_factory=AssetServiceConfig)
    load_balancer: LoadBalancerConfig = field(default_factory=LoadBalancerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GPUServerConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GPUServerConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        # Server config
        if "server" in data:
            for key, value in data["server"].items():
                if hasattr(config.server, key):
                    setattr(config.server, key, value)
        
        # Security config
        if "security" in data:
            for key, value in data["security"].items():
                if hasattr(config.security, key):
                    setattr(config.security, key, value)
        
        # Asset service config
        if "asset_service" in data:
            for key, value in data["asset_service"].items():
                if hasattr(config.asset_service, key):
                    setattr(config.asset_service, key, value)
        
        # Load balancer config
        if "load_balancer" in data:
            for key, value in data["load_balancer"].items():
                if hasattr(config.load_balancer, key):
                    setattr(config.load_balancer, key, value)
        
        # Model config
        if "model" in data:
            for key, value in data["model"].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        # Workflow config
        if "workflow" in data:
            workflow_data = data["workflow"]
            if "preprocess" in workflow_data:
                for key, value in workflow_data["preprocess"].items():
                    if hasattr(config.workflow.preprocess, key):
                        setattr(config.workflow.preprocess, key, value)
            if "postprocess" in workflow_data:
                for key, value in workflow_data["postprocess"].items():
                    if hasattr(config.workflow.postprocess, key):
                        setattr(config.workflow.postprocess, key, value)
            if "advanced" in workflow_data:
                for key, value in workflow_data["advanced"].items():
                    if hasattr(config.workflow.advanced, key):
                        setattr(config.workflow.advanced, key, value)
        
        # Logging config
        if "logging" in data:
            for key, value in data["logging"].items():
                if hasattr(config.logging, key):
                    setattr(config.logging, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "server": {
                "node_id": self.server.node_id,
                "host": self.server.host,
                "port": self.server.port,
                "workers": self.server.workers,
            },
            "security": {
                "internal_auth_token": self.security.internal_auth_token,
                "require_auth": self.security.require_auth,
            },
            "asset_service": {
                "callback_url": self.asset_service.callback_url,
                "internal_auth_token": self.asset_service.internal_auth_token,
                "timeout": self.asset_service.timeout,
                "retries": self.asset_service.retries,
                "retry_backoff": self.asset_service.retry_backoff,
            },
            "load_balancer": {
                "url": self.load_balancer.url,
                "internal_auth_token": self.load_balancer.internal_auth_token,
            },
            "model": {
                "model_type": self.model.model_type,
                "model_version": self.model.model_version,
                "device": self.model.device,
                "default_mode": self.model.default_mode,
                "default_steps": self.model.default_steps,
                "default_seed": self.model.default_seed,
                "default_cfg": self.model.default_cfg,
                "default_resolution": self.model.default_resolution,
                "enable_teacache": self.model.enable_teacache,
                "teacache_thresh": self.model.teacache_thresh,
            },
            "workflow": {
                "preprocess": {
                    "resize_to_resolution": self.workflow.preprocess.resize_to_resolution,
                    "maintain_aspect_ratio": self.workflow.preprocess.maintain_aspect_ratio,
                },
                "postprocess": {
                    "create_comparison": self.workflow.postprocess.create_comparison,
                    "compress_output": self.workflow.postprocess.compress_output,
                },
                "advanced": {
                    "warmup_on_startup": self.workflow.advanced.warmup_on_startup,
                    "clear_cache_between_jobs": self.workflow.advanced.clear_cache_between_jobs,
                },
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
            },
        }


def load_config(config_path: Optional[str] = None) -> GPUServerConfig:
    """
    Load configuration from file or use defaults.
    
    Priority:
    1. Provided config_path
    2. CONFIG_PATH environment variable
    3. Default path: configs/config.yaml
    """
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "configs/config.yaml")
    
    config_file = Path(config_path)
    
    if config_file.exists():
        print(f"📁 Loading config from: {config_file}")
        return GPUServerConfig.from_yaml(str(config_file))
    else:
        print(f"⚠️ Config file not found: {config_file}, using defaults")
        return GPUServerConfig()


# Singleton config instance
_config: Optional[GPUServerConfig] = None


def get_config() -> GPUServerConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: GPUServerConfig):
    """Set the global configuration instance."""
    global _config
    _config = config
