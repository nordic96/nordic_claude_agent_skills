# Backend Developer - Global Skills

Universal patterns and best practices for Python/FastAPI backend development.

## FastAPI Patterns

### Application Factory Pattern

```python
def create_app() -> FastAPI:
    app = FastAPI(
        title="API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.include_router(api_router, prefix="/api/v1")
    return app
```

### Dependency Injection

```python
# Use Depends() for reusable dependencies
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    # Verify token and return user
    pass
```

### Response Models

```python
# Always use response_model for type safety and documentation
@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    pass
```

## Python Async Gotchas

### Context Manager for Database Sessions

```python
# WRONG - session may close before operation completes
async def bad_example():
    async with get_db() as db:
        return db.query(User).all()  # Session closed!

# CORRECT - await inside context
async def good_example():
    async with get_db() as db:
        result = await db.execute(select(User))
        return result.scalars().all()
```

### Avoiding Blocking Calls

```python
import asyncio
from functools import partial

# CPU-bound work should use executor
async def process_heavy_task(data):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,  # Default executor
        partial(heavy_cpu_function, data)
    )
    return result
```

## SQLAlchemy 2.0 Patterns

### Async Session Setup

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    echo=True,  # SQL logging (disable in production)
)

async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)
```

### Select Queries (2.0 Style)

```python
from sqlalchemy import select

# CORRECT 2.0 style
stmt = select(User).where(User.id == user_id)
result = await db.execute(stmt)
user = result.scalar_one_or_none()

# With relationships (avoid N+1)
stmt = select(User).options(selectinload(User.posts)).where(User.id == user_id)
```

### Transactions

```python
async def transfer_funds(db: AsyncSession, from_id: int, to_id: int, amount: float):
    async with db.begin():  # Automatic commit/rollback
        from_account = await db.get(Account, from_id)
        to_account = await db.get(Account, to_id)
        from_account.balance -= amount
        to_account.balance += amount
```

## Pydantic Patterns

### Schema Inheritance

```python
from pydantic import BaseModel, ConfigDict

class UserBase(BaseModel):
    email: str
    name: str

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    model_config = ConfigDict(from_attributes=True)  # Pydantic v2
```

### Field Validation

```python
from pydantic import BaseModel, field_validator, EmailStr

class UserCreate(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v
```

### Model-Level Validation (Cross-Field Dependencies)

**Problem:** Need to validate one field based on another field's value (e.g., different rules for production vs development).

**Solution:**

```python
from pydantic import BaseModel, model_validator

class Settings(BaseModel):
    environment: str  # "production" or "development"
    api_key: str
    debug: bool = False

    @model_validator(mode="after")
    def validate_production_config(self) -> "Settings":
        """Enforce strict rules in production mode."""
        if self.environment == "production":
            if len(self.api_key) < 32:
                raise ValueError("API key must be at least 32 characters in production")
            if self.debug:
                raise ValueError("Debug mode not allowed in production")
        return self
```

**Key Insight:** Use `@model_validator(mode="after")` for cross-field validation after individual fields are validated. Use `mode="before"` to validate/transform raw input data before field validation.

### API Documentation with json_schema_extra

**Problem:** Pydantic v2 doesn't auto-generate OpenAPI examples like v1 did.

**Solution:**
```python
from pydantic import BaseModel, ConfigDict, Field

class ChatMessage(BaseModel):
    role: str = Field(description="Message author role")
    content: str = Field(description="Message text")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }
    )
```

**Key Insight:** Use `json_schema_extra` dict with "example" (singular) or "examples" (plural) key. FastAPI automatically picks this up for OpenAPI schema generation and Swagger UI documentation.

### Literal Types for API Objects

**Problem:** Need to enforce specific string values for API object types (OpenAI compatibility).

**Solution:**
```python
from typing import Literal
from pydantic import BaseModel, Field

class ChatCompletionResponse(BaseModel):
    object: Literal["chat.completion"] = Field(
        default="chat.completion",
        description="Object type identifier"
    )
    # This field can only be "chat.completion"
```

**Key Insight:** `Literal` types are validated by Pydantic and show up correctly in OpenAPI schemas. Always set a `default` for API response fields to ensure consistency.

### Field Constraints for Ranges

**Problem:** Need to validate numeric ranges (temperature, probabilities, token counts).

**Solution:**
```python
from pydantic import BaseModel, Field

class CompletionRequest(BaseModel):
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=None, ge=1)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
```

**Key Insight:** Use `ge` (greater than or equal), `le` (less than or equal), `min_length`, `max_length`. These are documented in the OpenAPI schema and enforced at validation time.

## Security Patterns

### Timing-Attack-Safe String Comparison

**Problem:** Using `==` or `!=` to compare API keys/tokens is vulnerable to timing attacks - attackers can infer key characters by measuring response time differences.

**Solution:**

```python
import secrets

# WRONG - timing attack vulnerability
if api_key != settings.api_key:
    raise HTTPException(status_code=401, detail="Invalid API key")

# CORRECT - constant-time comparison
if not secrets.compare_digest(api_key, settings.api_key):
    raise HTTPException(status_code=401, detail="Invalid API key")
```

**Key Insight:** `secrets.compare_digest()` compares strings in constant time, preventing attackers from using timing variations to brute-force keys. Always use for authentication credentials.

### Preventing Sensitive Data in Logs/Repr

**Problem:** Pydantic models with sensitive fields (API keys, passwords) can accidentally get logged via `repr()` or `str()`, exposing secrets.

**Solution:**

```python
from pydantic import BaseModel, Field

class Settings(BaseModel):
    api_key: str = Field(..., repr=False)  # Won't appear in repr/logs
    database_password: str = Field(..., repr=False)
    app_name: str  # OK to show in logs
```

**Effect:** When settings object is logged or repr'd, api_key will show as `api_key='***'` instead of the actual value.

**Key Insight:** Use `repr=False` for any field containing secrets, credentials, or PII that should never appear in logs.

### Password Hashing

```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)
```

### JWT Authentication

```python
from datetime import datetime, timedelta
from jose import jwt, JWTError

SECRET_KEY = "your-secret-key"  # Use env var!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
```

### CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Error Handling

### Custom Exception Handler

```python
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

class AppException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "type": "app_error"},
    )
```

### Structured Error Responses

```python
# Consistent error format
{
    "detail": "User not found",
    "type": "not_found",
    "field": "user_id"  # Optional
}
```

## Testing Patterns

### Test Client Setup

```python
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_create_user(client: AsyncClient):
    response = await client.post(
        "/api/v1/users",
        json={"email": "test@example.com", "password": "testpass123"}
    )
    assert response.status_code == 201
```

### Database Fixtures

```python
@pytest.fixture
async def db_session():
    async with async_session() as session:
        await session.begin()
        yield session
        await session.rollback()  # Clean up after test
```

## Performance Patterns

### Connection Pooling

```python
engine = create_async_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,  # Verify connections
)
```

### Caching with Redis

```python
import aioredis

redis = aioredis.from_url("redis://localhost")

async def get_cached_user(user_id: int) -> User | None:
    cached = await redis.get(f"user:{user_id}")
    if cached:
        return User.parse_raw(cached)
    return None
```

## Alembic Migration Patterns

### Creating Migrations

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "Add users table"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1
```

### Migration Best Practices

- Always review auto-generated migrations
- Add data migrations separately from schema migrations
- Test migrations on a copy of production data
- Include downgrade paths for reversibility

## Logging

```python
import logging
from fastapi import Request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Status: {response.status_code}")
    return response
```

## Environment Configuration

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    secret_key: str
    debug: bool = False

    class Config:
        env_file = ".env"

settings = Settings()
```

## Circular Import Resolution

### Problem: Circular Imports in Route Registration

**Scenario:** Route endpoint files need to import dependencies from `main.py`, but `main.py` imports the routes to register them.

```python
# main.py
from src.api.endpoints import chat  # Creates circular import

# endpoints/chat.py
from src.main import get_inference_client  # Tries to import from main.py
```

### Solution: Create `deps.py` for Shared Dependencies

Instead of keeping dependencies in `main.py`, create a dedicated `src/deps.py` file:

```python
# src/deps.py
from src.services.inference import InferenceClient

def get_inference_client() -> InferenceClient:
    return InferenceClient()
```

Then import from `deps.py` in routes and `main.py`:

```python
# src/main.py
from src.api.endpoints import chat
from src.core import settings

# endpoints/chat.py
from src.deps import get_inference_client
```

**Key Insight:** Dependency definitions should be in their own module (`deps.py`), separate from app initialization (`main.py`). This breaks the circular dependency chain while maintaining a single source of truth for dependency definitions.

---

## Health Check Patterns for Kubernetes

### Three-Tier Health Check Hierarchy

FastAPI services should implement three health check endpoints for different purposes:

```python
# GET /api/v1/health/ - Comprehensive health (monitoring/debugging)
@router.get("/health/", response_model=HealthCheckResponse)
async def health_check():
    """Detailed health status with component information."""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "uptime": app.state.uptime,
        "components": {
            "inference": "operational",
            "cache": "operational"
        }
    }

# GET /api/v1/health/live - Liveness probe (container restart decision)
@router.get("/health/live", response_model=LivenessResponse)
async def liveness():
    """Simple alive indicator - container should be restarted if this fails."""
    return {"alive": True}

# GET /api/v1/health/ready - Readiness probe (load balancer removal)
@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness(inference: InferenceClient = Depends(get_inference_client)):
    """Check if service can accept traffic (remove from LB if false)."""
    try:
        await inference.health_check()
        return {"ready": True}
    except Exception:
        return {"ready": False, "message": "Inference service unavailable"}
```

### Kubernetes Semantics

- **Liveness (`/live`)**: No dependency checks. If this fails, Kubernetes restarts the container.
- **Readiness (`/ready`)**: Checks dependencies. If false, Kubernetes removes pod from load balancer but doesn't restart.
- **Comprehensive (`/health/`)**: For human/monitoring tools. Can be verbose with component details.

**Key Insight:** Separate concerns by endpoint. Liveness = container is alive. Readiness = container can serve traffic. Comprehensive = for ops/debugging.

---

## Error Sanitization in Public APIs

### Problem: Exposing Internal Details in Error Messages

Returning detailed error messages to clients reveals system architecture and can aid attackers.

### Solution: Sanitize Public Responses, Log Full Details

```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,  # Logs full traceback internally
        extra={"path": request.url.path}
    )

    # Return generic message to client
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )
```

**Key Rules:**
1. Log full error + traceback internally with `exc_info=True`
2. Return only generic error messages to clients
3. Store request context (path, method, user_id) for debugging
4. For validation errors, return field names but not internals

**Key Insight:** Never trust error details to clients. Log comprehensively server-side, expose minimally to clients.

---

## Pydantic JSON Serialization

### Omitting None/Null Values

**Problem:** API responses include explicit `null` values for optional fields, making responses verbose and harder to parse.

**Solution:**

```python
from pydantic import BaseModel

class ChatCompletionChunk(BaseModel):
    choices: list["ChunkChoice"]
    model: str
    finish_reason: str | None = None  # May be None

# Omit null fields from JSON output
chunk_json = chunk.model_dump_json(exclude_none=True)
# Output: {"choices": [...], "model": "..."} (no finish_reason if None)
```

**Key Insight:** Use `exclude_none=True` on `model_dump_json()` to create cleaner API responses that exclude null fields, reducing bandwidth and making parsing more straightforward for clients.

## Python Package Imports

### Relative Imports vs Absolute Imports

**Problem:** Mixing absolute imports (`from api.services import ...`) and relative imports (`from .services import ...`) makes code less maintainable and harder to refactor.

**Solution:**

```python
# AVOID - absolute imports within same package
# api/src/services/inference.py
from api.src.core.exceptions import InferenceException

# PREFER - relative imports within same package
# api/src/services/inference.py
from ..core.exceptions import InferenceException

# Absolute imports OK for external packages
import httpx
import structlog
```

**Key Insight:** Use relative imports (`.module`, `..module`) within your package to make code less dependent on the package name. This makes moving/renaming packages easier and more idiomatic Python. Reserve absolute imports for external packages and cross-package imports.

## OpenAI-Compatible Streaming Format

### Server-Sent Events (SSE) Chat Completion Streaming

**Problem:** Implementing streaming chat completions that follow OpenAI's spec for client compatibility.

**Solution:**

```python
# Initial chunk with model and role
{"id": "...", "object": "chat.completion.chunk", "choices": [{"delta": {"role": "assistant"}, "index": 0}]}

# Content chunks
{"id": "...", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "Hello "}, "index": 0}]}
{"id": "...", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "world"}, "index": 0}]}

# Final chunk with finish_reason
{"id": "...", "object": "chat.completion.chunk", "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}

# Terminator
[DONE]
```

**Key Insight:** OpenAI format has specific semantics: role only in first chunk, content in delta chunks, finish_reason in final chunk, then `[DONE]` string. Clients expect this exact sequence via SSE (`text/event-stream` with `data: {json}\n\n` format).

## Rate Limiting with slowapi

### Full Setup Pattern

**Problem:** Using slowapi for rate limiting but endpoint returns 429 without proper error response.

**Solution:**

```python
# 1. Initialize limiter in main app setup
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# 2. Register RateLimitExceeded exception handler
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded"},
    )

# 3. Apply limiter decorator to endpoint
@router.post("/chat/completions")
@limiter.limit("60/minute")
async def create_completion(request: Request, body: ChatCompletionRequest):
    # Handler receives request parameter automatically from limiter
    pass
```

**Key Insight:** Both steps are required: setting `app.state.limiter` AND registering the `RateLimitExceeded` handler. Without both, rate-limited requests will fail ungracefully.

## Error Information Disclosure Prevention

### Problem: Exposing Internal Lists in Error Responses

Some endpoints (like failed chat completions) might want to return details about available options. However, this can expose system architecture to attackers.

### Solution:

```python
# WRONG - exposes internal list
try:
    if model not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Available: {available_models}"  # Leaks info!
        )
except Exception as exc:
    logger.error(f"Error: {exc}")
    raise HTTPException(
        status_code=500,
        detail={
            "error": "Internal error",
            "available_models": available_models  # DEFINITELY leaks info
        }
    )

# CORRECT - generic error to client, log details internally
try:
    if model not in available_models:
        logger.error(f"Invalid model: {model}. Available: {available_models}")
        raise HTTPException(
            status_code=400,
            detail="Invalid model specified"  # No list exposed
        )
except Exception as exc:
    logger.error(f"Chat completion error: {exc}", exc_info=True)
    raise HTTPException(
        status_code=500,
        detail="Internal server error"  # Generic message
    )
```

**Key Insight:** Internal details (available models, component list, API keys, database structure) should only appear in server logs, never in client error responses. Always return generic messages to clients.

## OpenAI API Compatibility: Optional Message Content

### Problem: Strict Type Validation Breaks OpenAI Compatibility

OpenAI's chat completion messages can have `content: null` for certain message types (e.g., function calls, tool use). Strict `content: str` validation rejects valid messages.

### Solution:

```python
from typing import Optional
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None  # Nullable per OpenAI spec
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

# Valid OpenAI messages:
# 1. Regular message with content
{"role": "user", "content": "Hello"}

# 2. Function/tool call without content
{"role": "assistant", "content": None, "tool_calls": [...]}

# 3. Tool response
{"role": "tool", "content": "Result", "tool_call_id": "123"}
```

**Key Insight:** Follow the OpenAI spec exactly for maximum compatibility. Message content is optional (`Optional[str]`) because different message types use different fields. Always validate against actual OpenAI API documentation.

## Type Hint Consistency: Optional vs Union Syntax

### Problem: Inconsistent Type Hint Syntax Across Codebase

Modern Python supports both `Optional[T]` and `T | None`, but mixing them in the same codebase reduces readability.

### Solution:

```python
from typing import Optional

# PREFER - for Python 3.9 compatibility and consistency
class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None

# AVOID - union syntax, less compatible
class BadExample(BaseModel):
    role: str
    content: str | None = None  # Works in 3.10+, inconsistent style
```

**Key Insight:** Use `Optional[T]` from `typing` module consistently across FastAPI/Pydantic projects. This:
1. Works on Python 3.9+
2. Matches common FastAPI examples and documentation
3. Improves readability when whole codebase uses same style
4. Pydantic internally converts both forms, but consistent style helps debugging

## Pydantic AI Integration with HuggingFace

### HuggingFace Inference API Endpoints

**Problem:** HuggingFace deprecated the old `api-inference.huggingface.co` endpoint. Code using this endpoint will receive 400 Bad Request errors.

**Solution:**

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# OLD ENDPOINT (DEPRECATED - returns 400)
base_url = "https://api-inference.huggingface.co/v1"

# NEW ENDPOINT (CURRENT - works with OpenAI-compatible models)
base_url = f"https://router.huggingface.co/hf-inference/models/{model_id}/v1"

# Create provider with HuggingFace OpenAI-compatible endpoint
provider = OpenAIProvider(
    base_url=base_url,
    api_key=hf_token,
)

# Use OpenAIChatModel, NOT HuggingFaceModel (which uses deprecated endpoint)
model = OpenAIChatModel(model_id, provider=provider)
agent = Agent(model=model, system_prompt="Your prompt here")
```

**Key Insight:** Use `OpenAIChatModel` + `OpenAIProvider` for HuggingFace models because HF provides an OpenAI-compatible endpoint. The `HuggingFaceModel` provider uses the deprecated endpoint. Always use the `router.huggingface.co/hf-inference` path for current compatibility.

**Note:** For project-specific HuggingFace configuration patterns, see your project's CLAUDE.md file.

### Pydantic AI Streaming with Message History

**Problem:** Need to handle multi-turn conversations and streaming responses with Pydantic AI while maintaining parameter control (temperature, max_tokens).

**Solution:**

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.settings import ModelSettings

# Build message history from OpenAI-format messages
def build_message_history(messages: list[dict]) -> tuple[str, list[ModelMessage]]:
    history: list[ModelMessage] = []
    last_user_message = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        if role == "user":
            last_user_message = content
            # Add to history as ModelRequest with UserPromptPart
            history.append(ModelRequest(parts=[UserPromptPart(content=content)]))

    # Exclude last user message (it's the current prompt)
    return last_user_message, history[:-1] if history else []

# Use model_settings to control LLM parameters
model_settings = ModelSettings(
    max_tokens=512,  # Controls output length
    temperature=0.7,  # Controls randomness
)

# Stream with message history for multi-turn support
async with agent.run_stream(
    user_message,
    message_history=message_history,
    model_settings=model_settings,
) as response:
    async for chunk in response.stream_text(delta=True):
        if chunk:
            # Process text chunk
            print(chunk, end="")
```

**Key Insight:**
- `message_history` parameter enables multi-turn conversations by providing context to the agent
- `model_settings` controls LLM parameters (max_tokens, temperature) - NOT the `output_type` parameter (which controls structured output)
- `response.stream_text(delta=True)` returns incremental text chunks, ideal for SSE streaming
- Message conversion must handle role ("user", "assistant") mapping correctly

## Pydantic AI Streaming Adapters

### VercelAI Adapter Protocol

**Problem:** Pydantic AI provides `VercelAIAdapter` for streaming responses compatible with Vercel's AI SDK, but the request format and integration quirks are not well documented.

**Solution:**

```python
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/api/v1/vercel-chat")
async def vercel_chat(request: Request):
    """
    VercelAI-compatible streaming endpoint.

    Important: Do NOT add Pydantic model validation to this endpoint
    because FastAPI would consume the request body before VercelAIAdapter
    can read it, causing "stream consumed" errors.
    """
    adapter = VercelAIAdapter(tools=[])

    # VercelAIAdapter.dispatch_request() handles parsing internally
    async for chunk in adapter.dispatch_request(
        request=request,
        model=your_model,
        system_prompt="Your system prompt"
    ):
        yield chunk

    return StreamingResponse(...)
```

**Key Protocol Details:**

1. **Trigger Field** (CRITICAL): Request must include `trigger` field with specific values:
   - `trigger: "submit-message"` - New user message
   - `trigger: "regenerate-message"` - Regenerate last assistant response
   - **WRONG**: `trigger: "run"` (OpenAI format) - Will fail silently

2. **Body Consumption**: Never add request model validation:
   ```python
   # WRONG - FastAPI consumes body, VercelAIAdapter gets empty stream
   @app.post("/chat")
   async def chat(request: Request, body: ChatCompletionRequest):
       adapter.dispatch_request(request, ...)

   # CORRECT - Let VercelAIAdapter handle body parsing
   @app.post("/chat")
   async def chat(request: Request):
       adapter.dispatch_request(request, ...)
   ```

3. **Message Format**: Compatible with OpenAI format but trigger-based routing:
   ```json
   {
     "trigger": "submit-message",
     "messages": [
       {"role": "user", "content": "Hello"},
       {"role": "assistant", "content": "Hi"}
     ],
     "systemPrompt": "Optional override"
   }
   ```

**Key Insight:** The VercelAIAdapter is designed for seamless Vercel AI SDK integration but requires careful request handling. The adapter expects full control over request body parsing, so never add FastAPI route validation that consumes the body first. The trigger field routing (submit vs regenerate) is the core difference from OpenAI's API.

**Note:** For project-specific VercelAI endpoint implementation examples, see your project's CLAUDE.md documentation.

---

## Common Gotchas

1. **Forgetting `await`**: Always await async functions
2. **Session lifecycle**: Don't return database objects after session closes
3. **N+1 queries**: Use `selectinload`/`joinedload` for relationships
4. **Circular imports**: Use `deps.py` to break import chains (not just `TYPE_CHECKING`)
5. **Pydantic v1 vs v2**: Use `model_config` not `Config` class in v2
6. **UTC timestamps**: Always store timestamps in UTC
7. **Environment variables**: Never hardcode secrets
8. **json_schema_extra placement**: Goes inside ConfigDict, not as separate class attribute
9. **default_factory for timestamps**: Use `default_factory=lambda: int(time.time())` for server-generated timestamps
10. **Error sanitization**: Log full errors internally, return generic messages to clients
11. **slowapi setup**: Must register BOTH `app.state.limiter` AND `RateLimitExceeded` handler
12. **Type hints**: Use `Optional[T]` not `T | None` for Python 3.9 compatibility and consistency
13. **VercelAIAdapter body consumption**: Never validate request body when using VercelAIAdapter - let it handle parsing
14. **VercelAI trigger values**: Use `"submit-message"` or `"regenerate-message"`, NOT `"run"`
