#Setting up pydantic settings here now

from pydantic_settings import BaseSettings
from pydantic import Field 
from functools import lru_cache

class Settings(BaseSettings):
    vercel_ai_gateway_url: str = Field(
        default='https://gateway.ai.vercel.sh/v1',
        description='Vercel AI Gateway endpoint URL'
    )
    vercel_ai_gateway_api_key: str = Field(
        ...,
        description='API key for Vercel AI Gateway'
    )

    model_name: str = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description='Sampling temprature'
    )
    model_max_tokens: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description='Maximum tokens in response'
    )
    max_context_cells: int = Field(
        default=5,
        ge=1,
        le=10,
        description='Number of previous cells for context'
    )
    max_cell_length: int = Field(
        default=2000,
        description='Max characters per cell before truncation'
    )

    class Config:
        env_file = '.env'
        env_file_encoding = "utf-8"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()