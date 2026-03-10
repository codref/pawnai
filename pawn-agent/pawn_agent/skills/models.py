"""Pydantic models for tool and skill YAML descriptors.

Tool YAML schema
----------------
.. code-block:: yaml

    name: download_s3
    description: Download an audio file from S3 to a local temp directory
    function: s3.download          # key registered in ExecutorRegistry
    input_schema:
      type: object
      required: [audio_path]
      properties:
        audio_path: {type: string}
    output_schema:
      type: object
      properties:
        local_path: {type: string}
        temp_dir:   {type: string}
    enabled: true

Skill YAML schema
-----------------
.. code-block:: yaml

    name: transcribe
    description: Convert audio to text with word-level timestamps
    tools:
      - step: 1
        tool: download_s3
        params:
          audio_path: "{{input.audio_path}}"
      - step: 2
        tool: run_transcription
        params:
          local_path: "{{steps.1.local_path}}"
          timestamps: "{{input.timestamps}}"
      - step: 3
        tool: cleanup_temp
        params:
          temp_dir: "{{steps.1.temp_dir}}"
    output:
      transcript:  "{{steps.2.transcript}}"
      session_id:  "{{steps.2.session_id}}"
    input_schema:
      type: object
      required: [audio_path]
      properties:
        audio_path: {type: string}
        timestamps: {type: boolean, default: true}
    enabled: true
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class ToolDefinition(BaseModel):
    """Descriptor for an atomic tool loaded from a YAML file."""

    name: str = Field(..., description="Unique tool identifier")
    description: str = Field(..., description="Natural-language description")
    function: str = Field(
        ...,
        description="Dot-path key registered in ExecutorRegistry (e.g. 's3.download')",
    )
    input_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for accepted parameters",
    )
    output_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for returned values (used for inter-step piping)",
    )
    enabled: bool = Field(True, description="Whether the tool is active")


class SkillToolStep(BaseModel):
    """A single tool invocation step within a skill pipeline."""

    step: int = Field(..., description="1-based step number (must be unique within a skill)")
    tool: str = Field(..., description="Tool name to invoke (must exist in ToolRegistry)")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Static parameters and/or template expressions "
            "(e.g. '{{input.audio_path}}', '{{steps.1.local_path}}')"
        ),
    )


class SkillDefinition(BaseModel):
    """Descriptor for a skill pipeline loaded from a YAML file.

    A skill is an ordered sequence of tool steps with data-flow declared via
    ``{{template}}`` expressions.  The ``output`` mapping defines what the
    skill returns to the caller (also expressed as templates).
    """

    name: str = Field(..., description="Unique skill identifier (used in agent plans)")
    description: str = Field(..., description="Natural-language description fed to the DSPy planner")
    tools: List[SkillToolStep] = Field(
        ...,
        description="Ordered list of tool steps that make up this skill's pipeline",
    )
    output: Dict[str, Any] = Field(
        default_factory=dict,
        description="Output mapping — keys are return values, values are template expressions",
    )
    input_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema describing the parameters this skill accepts",
    )
    enabled: bool = Field(True, description="Whether the skill is active")

    @model_validator(mode="after")
    def _unique_step_numbers(self) -> "SkillDefinition":
        numbers = [s.step for s in self.tools]
        if len(numbers) != len(set(numbers)):
            raise ValueError(f"Skill '{self.name}': duplicate step numbers in tools list")
        return self

    def describe(self) -> Dict[str, Any]:
        """Return a concise dict suitable for feeding to the DSPy planner."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
