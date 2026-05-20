from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    LMSTUDIO_BASE_URL: str = "http://127.0.0.1:1234/v1"
    LMSTUDIO_MODEL: str = "google/gemma-4-e4b"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    DB_PATH: str = "./data/hardware.db"
    DATASHEET_DIR: str = "./data/datasheets"

    MAX_PDF_PAGES: int = 40
    RAG_CHUNK_SIZE: int = 800
    RAG_CHUNK_OVERLAP: int = 120
    RAG_TOP_K: int = 6

    @property
    def db_path_abs(self) -> Path:
        p = Path(self.DB_PATH)
        return p if p.is_absolute() else (ROOT / p).resolve()

    @property
    def datasheet_dir_abs(self) -> Path:
        p = Path(self.DATASHEET_DIR)
        return p if p.is_absolute() else (ROOT / p).resolve()


settings = Settings()
settings.db_path_abs.parent.mkdir(parents=True, exist_ok=True)
settings.datasheet_dir_abs.mkdir(parents=True, exist_ok=True)
