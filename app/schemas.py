from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    model_id: str | None = None
    project_id: str | None = None
    # Accepted for backward compatibility with older clients. The app now uses
    # one shared Azure connection and ignores any client-supplied endpoint.
    endpoint_id: str | None = None


class ConversationUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    project_id: str | None = None


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    instructions: str = Field(default="", max_length=20000)
    default_model_id: str | None = None
    endpoint_id: str | None = None
    default_endpoint_id: str | None = None


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    instructions: str | None = Field(default=None, max_length=20000)
    default_model_id: str | None = None
    endpoint_id: str | None = None
    default_endpoint_id: str | None = None


class ProjectFileUpdate(BaseModel):
    is_active: bool


class MessageCreate(BaseModel):
    content: str = Field(min_length=1)
    model_id: str | None = None
    endpoint_id: str | None = None
    reasoning_effort: str = "auto"
    verbosity: str = "medium"
    max_output_tokens: int = Field(default=16384, ge=256, le=128000)
    use_code_interpreter: bool = True
    use_web_search: bool = False
    research_depth: Literal["quick", "thorough"] = "thorough"
    web_allowed_domains: list[str] = Field(default_factory=list, max_length=100)
    web_blocked_domains: list[str] = Field(default_factory=list, max_length=100)
    attachment_ids: list[str] = Field(default_factory=list)


class PartialAssistantCreate(BaseModel):
    content: str = ""
    model_id: str | None = None
    endpoint_id: str | None = None
    use_web_search: bool = False
    research_depth: Literal["quick", "thorough"] = "thorough"
    reasoning_summary: str = ""
    activities: list[str] = Field(default_factory=list)
    duration_seconds: float | None = Field(default=None, ge=0)
