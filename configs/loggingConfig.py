from pydantic import BaseModel, ConfigDict, Field


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    console_level: str = Field(..., description="Console log level (e.g., INFO, DEBUG).")
    file_level: str = Field(..., description="File log level for run_dir logs.")
    components: dict[str, str] = Field(
        default_factory=dict,
        description="Optional per-component log level overrides.",
    )
