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

### Validation

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
