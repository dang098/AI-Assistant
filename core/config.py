from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"

    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "knowledge_base"

    supabase_url: str = ""
    supabase_key: str = ""

    top_k_retrieval: int = 10
    top_k_rerank: int = 3
    chunk_size: int = 500
    chunk_overlap: int = 50

    log_level: str = "INFO"
    app_env: str = "development"


settings = Settings()
