"""Pydantic models for parsing backend responses.

Request payloads are built as plain dicts in client.py — no Pydantic models
for outbound data. Validation lives as simple if-checks in the client layer.
"""

from __future__ import annotations

import builtins
import re
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, PlainSerializer, field_validator, model_validator

from jarvislabs.regions import region_code

# Region fields serialize their internal id to a display code (IN1/IN2/EU1) in
# model dumps, while the attribute keeps the raw id. Two variants preserve each
# field's required/optional contract.
RegionCode = Annotated[str, PlainSerializer(region_code)]
OptionalRegionCode = Annotated[str | None, PlainSerializer(region_code)]

# ── Account ──────────────────────────────────────────────────────────────────


class Balance(BaseModel):
    balance: float
    grants: float


class UserInfo(BaseModel):
    user_id: str
    name: str | None = None
    address1: str | None = None
    address2: str | None = None
    city: str | None = None
    country: str | None = None
    phone_number: str | None = None
    state: str | None = None
    zip_code: str | None = None
    tax_id: str | None = None


class ResourceMetrics(BaseModel):
    running_instances: int
    paused_instances: int
    running_vms: int
    paused_vms: int
    deployments: int
    filesystems: int


class ApiKey(BaseModel):
    """A named API key on the account, without its secret.

    Attributes:
        kind: Who pays for the key's Models API requests: "subscription" (the plan) or
            "payg" (the wallet). "payg" when the response has no kind.
    """

    id: str
    name: str
    kind: Literal["payg", "subscription"] = "payg"
    created_at: datetime | None = None


class CreatedApiKey(ApiKey):
    """A key just created, with its secret, which the backend returns only this once."""

    token: str


# ── Subscription ─────────────────────────────────────────────────────────────


class SubscriptionPlan(BaseModel):
    """The plan the account is subscribed to."""

    id: int
    name: str


class SubscriptionPeriod(BaseModel):
    """The paid period's start and when it renews.

    ``renews_at`` is when the period ends instead when ``cancel_at_period_end`` is true.
    """

    start: datetime
    renews_at: datetime
    cancel_at_period_end: bool


class UsageWindow(BaseModel):
    """One usage window's share used, in percent, and its reset time.

    ``resets_at`` is None for the 5-hour window while none is running, and for the monthly
    window while the renewal payment is awaited.
    """

    percent_used: float
    resets_at: datetime | None


class UsageWindows(BaseModel):
    """The plan's three usage windows."""

    five_hour: UsageWindow
    weekly: UsageWindow
    monthly: UsageWindow


class UsageSlots(BaseModel):
    """Plan-paid requests running now, and how many the plan allows at once."""

    in_use: int
    limit: int


class SubscriptionUsage(BaseModel):
    """The account's subscription and usage, the same whichever key asks.

    ``use_balance`` is true when requests the plan doesn't cover are paid from the wallet.
    ``renewal_pending`` is true once ``renews_at`` has passed and the renewal payment hasn't arrived.
    """

    plan: SubscriptionPlan
    period: SubscriptionPeriod
    use_balance: bool
    renewal_pending: bool
    windows: UsageWindows
    slots: UsageSlots


# ── SSH Keys ─────────────────────────────────────────────────────────────────


class SSHKey(BaseModel):
    ssh_key: str
    key_name: str
    key_id: str
    user_id: str | None = None


# ── Scripts ──────────────────────────────────────────────────────────────────


class StartupScript(BaseModel):
    script_id: int
    script_name: str | None = None


# ── Filesystems ──────────────────────────────────────────────────────────────


class Filesystem(BaseModel):
    fs_id: int
    fs_name: str | None = None
    storage: int | None = None
    region: OptionalRegionCode = None


# ── VPCs ─────────────────────────────────────────────────────────────────────


class Vpc(BaseModel):
    vpc_id: str
    name: str | None = None
    region: RegionCode
    cidr: str | None = None
    gateway_ip: str | None = None
    is_default: bool = False
    status: str | None = None


class VpcIP(BaseModel):
    """A private IP allocated in a VPC, and the machine holding it."""

    private_ip: str
    machine_id: int | None = None
    status: str | None = None
    mac_address: str | None = None
    lsp_name: str | None = None


# ── Templates ─────────────────────────────────────────────────────────────────


class Template(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    title: str
    description: str | None = None
    category: str | None = None
    versions: str | None = None


# ── Server Meta (for auto-routing + GPU availability) ─────────────────────────


class ServerMetaGPU(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gpu_type: str
    region: RegionCode
    num_free_devices: int = 0
    effective_num_free_devices: int | None = None
    spot_num_free_devices: int | None = None
    price_per_hour: float | None = None
    spot_price: float | None = None
    vram: str | None = None
    arc: str | None = None
    cpus_per_gpu: int | None = None
    ram_per_gpu: int | None = None
    workload_type: str | None = None

    @field_validator("num_free_devices", "effective_num_free_devices", mode="before")
    @classmethod
    def _coerce(cls, v: Any) -> int:
        if v is None or (isinstance(v, str) and not v.strip()):
            return 0
        return int(v)

    @property
    def spot_free_devices(self) -> int:
        """Free GPUs a spot instance can use; falls back to num_free_devices if the backend omits the spot count."""
        return self.num_free_devices if self.spot_num_free_devices is None else self.spot_num_free_devices


class ServerMetaResponse(BaseModel):
    server_meta: list[ServerMetaGPU] = Field(default_factory=list)
    cpu_meta: dict[str, Any] = Field(default_factory=dict)


# ── Instance ─────────────────────────────────────────────────────────────────


class Instance(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    machine_id: int
    cost: float = 0.0
    runtime: str | int = Field(default=0, validation_alias=AliasChoices("runtime", "duration"))
    gpu_type: str | None = None
    ram: int | None = None
    storage_gb: int | None = Field(default=None, validation_alias=AliasChoices("storage_gb", "hdd"))
    cores: int | None = None
    template: str = Field(validation_alias=AliasChoices("template", "framework"))
    framework_id: str | None = None
    version: str | None = None
    fs_id: int | None = None
    num_gpus: int | None = None
    url: str | None = None
    ssh_command: str | None = Field(default=None, validation_alias=AliasChoices("ssh_command", "ssh_str"))
    status: str
    paused_image_size: float | None = Field(default=None, validation_alias=AliasChoices("paused_image_size", "v_size"))
    endpoints: list[str] | None = None
    name: str | None = Field(default=None, validation_alias=AliasChoices("name", "instance_name"))
    is_reserved: bool | None = None
    is_spot: bool | None = None
    committed_resource_id: int | None = None
    reservation_info: dict[str, Any] | None = None
    billing_frequency: str | None = Field(default=None, validation_alias=AliasChoices("billing_frequency", "frequency"))
    vs_url: str | None = None
    deployment_id: str | None = None
    user_id: str | None = None
    disk_type: str | None = None
    public_ip: str | None = None
    private_ip: str | None = None
    vpc_id: str | None = None
    http_ports: str | None = None
    region: OptionalRegionCode = None

    @field_validator("ram", "storage_gb", "cores", mode="before")
    @classmethod
    def _coerce_int(cls, v: Any) -> int | None:
        if v is None:
            return None
        if isinstance(v, str):
            v = re.sub(r"[a-zA-Z]+$", "", v.strip())
            return int(v) if v else None
        return int(v)

    @model_validator(mode="after")
    def normalize_connection_details(self) -> Instance:
        if self.status == "Running":
            return self

        # Hide connection details while the instance is unavailable.
        self.public_ip = None
        self.ssh_command = None
        self.url = None
        self.vs_url = None
        self.endpoints = None

        # VPC private IPs remain reserved across pause/resume. Private IPs for
        # containers and non-VPC VMs are released, so their stale values are hidden.
        if not self.vpc_id:
            self.private_ip = None

        return self


# ── Response wrappers ────────────────────────────────────────────────────────


class InstanceListResponse(BaseModel):
    success: str | bool
    instances: list[Instance] = Field(default_factory=list)


class StatusResponse(BaseModel):
    status: str
    error: str | None = None
    code: int | str | None = None

    @field_validator("error", mode="before")
    @classmethod
    def _coerce_error_none(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, str):
            raw = v.strip()
            if raw.lower() in {"none", "null", ""}:
                return None
            return raw
        return str(v)

    @field_validator("code", mode="before")
    @classmethod
    def _coerce_code_none(cls, v: Any) -> int | str | None:
        if v is None:
            return None
        if isinstance(v, str):
            raw = v.strip()
            if raw.lower() in {"none", "null", ""}:
                return None
            return raw
        return v


# ── Deployments ──────────────────────────────────────────────────────────────


class WorkerInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # worker_id can be null for a worker that is still being set up.
    worker_id: int | None = None
    status: str
    last_used: str | None = None


class Workers(BaseModel):
    model_config = ConfigDict(extra="ignore")

    total: int = 0
    healthy: int = 0
    provisioning: int = 0
    # Named `list` to match the API; use builtins.list for the annotation.
    list: builtins.list[WorkerInfo] = Field(default_factory=list)


class Deployment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    deployment_id: str
    name: str | None = None
    status: str
    error_message: str | None = None
    region: OptionalRegionCode = None
    framework: str | None = None
    gpus_to_use: dict | None = None
    gpus_per_worker: int | None = None
    min_workers: int | None = None
    max_workers: int | None = None
    concurrent_requests: int | None = None
    idle_timeout: int | None = None
    wait_time: int | None = None
    storage: int | None = None
    args: dict | None = None
    env: dict | None = None
    # Set for managed deployments.
    managed_model: ManagedModelSummary | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    workers: Workers = Field(default_factory=Workers)
    queue_depth: int | None = None
    # Total accrued cost of the deployment so far (compute + storage), in `currency`.
    total_cost: float = 0.0
    currency: str = "USD"

    @property
    def model(self) -> str | None:
        return (self.args or {}).get("model")

    @property
    def served_model(self) -> str | None:
        """Model name inference clients must use: the managed model's API name, else the
        served-model-name arg when set, else the model id."""
        if self.managed_model:
            return self.managed_model.api_model_name
        return (self.args or {}).get("served-model-name") or self.model


class DeploymentSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    deployment_id: str
    name: str | None = None
    status: str
    region: OptionalRegionCode = None
    start_time: datetime | None = None
    framework: str | None = None
    min_workers: int | None = None
    max_workers: int | None = None
    gpus_to_use: dict | None = None
    concurrent_requests: int | None = None
    gpus_per_worker: int | None = None
    error_message: str | None = None
    managed_model: ManagedModelSummary | None = None
    total_cost: float = 0.0
    currency: str = "USD"


class DeploymentStatus(BaseModel):
    model_config = ConfigDict(extra="ignore")

    deployment_id: str
    region: OptionalRegionCode = None
    status: str


class RegionError(BaseModel):
    region: RegionCode
    error: str


class DeploymentListResult(BaseModel):
    deployments: list[DeploymentSummary] = Field(default_factory=list)
    region_errors: list[RegionError] = Field(default_factory=list)


class _ManagedModelFacts(BaseModel):
    """Facts about a managed model, shared by catalog entries and deployment records."""

    model_config = ConfigDict(extra="ignore")
    model_slug: str
    display_name: str | None = None
    description: str | None = None
    context_length: int | None = None
    model_card_url: str | None = None


class ManagedModelSummary(_ManagedModelFacts):
    """The managed model behind a deployment (``managed_model`` in a deployment record)."""

    # The model name to send in requests to the deployment.
    api_model_name: str | None = None


class ManagedModel(_ManagedModelFacts):
    """A catalog entry: a managed model offered in one region."""

    region: OptionalRegionCode = None
    gpu_type: str | None = None
    gpus_per_worker: int | None = None
    concurrent_requests: int | None = None
    # Cost of one running worker per hour (GPU price x GPUs per worker); filled in by the CLI catalog.
    price_per_hour: float | None = None


class ManagedModelListResult(BaseModel):
    models: list[ManagedModel] = Field(default_factory=list)
    region_errors: list[RegionError] = Field(default_factory=list)
    currency: str | None = None


class ModelPricing(BaseModel):
    """A model's prices per million tokens, in ``currency``.

    ``usd`` holds the same rates in US dollars, for a row priced in another currency;
    None when the API sends none.
    """

    currency: str
    input_per_million_tokens: float
    cached_input_per_million_tokens: float | None = None
    output_per_million_tokens: float
    usd: ModelPricing | None = None


class ModelThinking(BaseModel):
    """How a model's thinking is controlled.

    Attributes:
        supported: Whether the model can think.
        supports_toggle: Whether a request can switch thinking on and off.
        supported_efforts: The effort levels a request can choose.
        default_effort: The effort applied when a request sends none, or None.
        template_parameters: Keys to set in ``chat_template_kwargs``: "toggle" and/or
            "effort", mapped to the model's parameter name.
    """

    supported: bool = False
    supports_toggle: bool = False
    supported_efforts: list[str] = Field(default_factory=list)
    # The effort the Models API applies to the model when a request sends none.
    default_effort: str | None = None
    # Keys to set in chat_template_kwargs: "toggle" and/or "effort" -> the model's parameter name.
    template_parameters: dict[str, str] = Field(default_factory=dict)


class ModelCapabilities(BaseModel):
    """What a model supports beyond plain chat; None means the catalog doesn't say."""

    tool_calling: bool | None = None


class ModelInfo(BaseModel):
    """One model in the Models API catalog.

    Null fields in the API response take their defaults.

    Attributes:
        context_length: The model's context window in tokens, or None.
        max_output_tokens: The most tokens the model writes in one reply, or None.
        capabilities: What the model supports beyond plain chat.
        pricing: The model's prices, or None when the API sends none.
    """

    id: str
    name: str | None = None
    description: str | None = None
    context_length: int | None = None
    max_output_tokens: int | None = None
    model_card_url: str | None = None
    input_modalities: list[str] = Field(default_factory=list)
    output_modalities: list[str] = Field(default_factory=list)
    thinking: ModelThinking = Field(default_factory=ModelThinking)
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    pricing: ModelPricing | None = None

    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data: Any) -> Any:
        # The API sends null for metadata a model doesn't have yet; use the defaults instead.
        return {key: value for key, value in data.items() if value is not None} if isinstance(data, dict) else data
