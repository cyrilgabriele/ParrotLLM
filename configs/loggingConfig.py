from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProfilerConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = Field(False, description="Enable PyTorch profiler during training.")
    wait: int = Field(1, ge=0, description="Profiler schedule wait steps.")
    warmup: int = Field(1, ge=0, description="Profiler schedule warmup steps.")
    active: int = Field(1, ge=1, description="Profiler schedule active steps to record.")
    repeat: int = Field(1, ge=1, description="Number of times to repeat the schedule block.")
    skip_first: int = Field(0, ge=0, description="Number of initial steps to skip entirely.")
    activities: list[Literal["cpu", "cuda"]] = Field(
        default_factory=lambda: ["cpu", "cuda"],
        description="Profiler activities to record.",
    )
    record_shapes: bool = Field(True)
    profile_memory: bool = Field(True)
    with_stack: bool = Field(False)
    with_modules: bool = Field(False)
    trace_subdir: str = Field("profiler", description="Subdirectory for profiler traces.")
    tensorboard_subdir: str = Field(
        "tensorboard", description="Subdir for TensorBoard traces if enabled.",
    )
    export_chrome_trace: bool = Field(True, description="Write chrome trace JSON files.")
    export_tensorboard: bool = Field(False, description="Emit tensorboard trace directory.")
    log_summary: bool = Field(True, description="Log the aggregated profiler summary table.")
    summary_sort_by: str = Field("self_cpu_time_total")
    summary_row_limit: int = Field(20, ge=1)
    run_on_all_ranks: bool = Field(
        False,
        description="If False, only rank0 process runs the profiler in distributed runs.",
    )


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    console_level: str = Field(..., description="Console log level (e.g., INFO, DEBUG).")
    file_level: str = Field(..., description="File log level for run_dir logs.")
    components: dict[str, str] = Field(
        default_factory=dict,
        description="Optional per-component log level overrides.",
    )
    profiler: ProfilerConfig | None = Field(
        default=None,
        description="Optional PyTorch profiler configuration.",
    )
