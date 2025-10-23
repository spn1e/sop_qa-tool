"""
Configuration management system with environment variables and settings validation.
Supports both AWS and local mode operation.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Optional, List
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class OperationMode(str, Enum):
    """System operation modes"""
    AWS = "aws"
    LOCAL = "local"


class Settings(BaseSettings):
    """Application settings with validation"""
    
    # Mode Selection
    mode: OperationMode = Field(default=OperationMode.LOCAL, env="MODE")
    
    # AWS Configuration
    aws_profile: str = Field(default="default", env="AWS_PROFILE")
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-sonnet-20240229-v1:0", 
        env="BEDROCK_MODEL_ID"
    )
    titan_embeddings_id: str = Field(
        default="amazon.titan-embed-text-v2:0", 
        env="TITAN_EMBEDDINGS_ID"
    )
    opensearch_endpoint: Optional[str] = Field(default=None, env="OPENSEARCH_ENDPOINT")
    s3_raw_bucket: Optional[str] = Field(default=None, env="S3_RAW_BUCKET")
    s3_chunks_bucket: Optional[str] = Field(default=None, env="S3_CHUNKS_BUCKET")
    
    # Local Configuration
    local_data_path: Path = Field(default=Path("./data"), env="LOCAL_DATA_PATH")
    faiss_index_path: Path = Field(default=Path("./data/faiss_index"), env="FAISS_INDEX_PATH")
    hf_model_path: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", 
        env="HF_MODEL_PATH"
    )
    
    # Application Settings
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB", ge=1, le=500)
    chunk_size: int = Field(default=800, env="CHUNK_SIZE", ge=200, le=2000)
    chunk_overlap: int = Field(default=150, env="CHUNK_OVERLAP", ge=0, le=500)
    top_k_retrieval: int = Field(default=5, env="TOP_K_RETRIEVAL", ge=1, le=20)
    confidence_threshold: float = Field(
        default=0.35, env="CONFIDENCE_THRESHOLD", ge=0.0, le=1.0
    )
    
    # Security Settings
    allowed_file_types: str = Field(
        default="pdf,docx,html,txt", 
        env="ALLOWED_FILE_TYPES"
    )
    enable_pii_redaction: bool = Field(default=False, env="ENABLE_PII_REDACTION")
    block_localhost_urls: bool = Field(default=True, env="BLOCK_LOCALHOST_URLS")
    
    # Performance Settings
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS", ge=1, le=100)
    request_timeout_seconds: int = Field(default=30, env="REQUEST_TIMEOUT_SECONDS", ge=5, le=300)
    embedding_batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE", ge=1, le=128)
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file_path: Optional[Path] = Field(default=None, env="LOG_FILE_PATH")
    enable_structured_logging: bool = Field(default=True, env="ENABLE_STRUCTURED_LOGGING")
    
    @field_validator("local_data_path", "faiss_index_path", mode="before")
    @classmethod
    def validate_paths(cls, v):
        """Convert string paths to Path objects"""
        if isinstance(v, str):
            return Path(v)
        return v
    
    def get_allowed_file_types(self) -> List[str]:
        """Get allowed file types as a list"""
        if isinstance(self.allowed_file_types, str):
            return [ft.strip().lower() for ft in self.allowed_file_types.split(",")]
        return self.allowed_file_types
    
    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v, info):
        """Ensure chunk overlap is less than chunk size"""
        if info.data and "chunk_size" in info.data:
            chunk_size = info.data["chunk_size"]
            if v >= chunk_size:
                raise ValueError("chunk_overlap must be less than chunk_size")
        return v
    
    @model_validator(mode="after")
    def validate_aws_settings(self):
        """Validate AWS-specific settings when in AWS mode"""
        if self.mode == OperationMode.AWS:
            required_aws_fields = [
                ("opensearch_endpoint", self.opensearch_endpoint), 
                ("s3_raw_bucket", self.s3_raw_bucket), 
                ("s3_chunks_bucket", self.s3_chunks_bucket)
            ]
            for field_name, field_value in required_aws_fields:
                if not field_value:
                    raise ValueError(f"{field_name} is required when mode=aws")
        return self
    
    @model_validator(mode="after")
    def validate_local_settings(self):
        """Validate local-specific settings when in local mode"""
        if self.mode == OperationMode.LOCAL:
            if self.local_data_path and not self.local_data_path.parent.exists():
                # Create parent directory if it doesn't exist
                self.local_data_path.parent.mkdir(parents=True, exist_ok=True)
        return self
    
    def create_directories(self):
        """Create necessary directories for local mode"""
        if self.mode == OperationMode.LOCAL:
            self.local_data_path.mkdir(parents=True, exist_ok=True)
            self.faiss_index_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.local_data_path / "raw_docs").mkdir(exist_ok=True)
            (self.local_data_path / "chunks").mkdir(exist_ok=True)
            (self.local_data_path / "logs").mkdir(exist_ok=True)
    
    def get_max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes"""
        return self.max_file_size_mb * 1024 * 1024
    
    def is_aws_mode(self) -> bool:
        """Check if running in AWS mode"""
        return self.mode == OperationMode.AWS
    
    def is_local_mode(self) -> bool:
        """Check if running in local mode"""
        return self.mode == OperationMode.LOCAL
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings


def validate_settings() -> bool:
    """Validate current settings and return True if valid"""
    try:
        settings.create_directories()
        return True
    except Exception as e:
        print(f"Settings validation failed: {e}")
        return False