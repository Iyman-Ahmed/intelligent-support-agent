"""
Modal deployment configuration.
Deploy with: modal deploy deploy/modal_app.py
"""

import modal

# ---------------------------------------------------------------------------
# Modal image — build the container
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "libpq-dev")
    .pip_install_from_requirements("requirements.txt")
    .copy_local_dir("app", "/app/app")
)

# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

secrets = [
    modal.Secret.from_name("anthropic-api-key"),      # ANTHROPIC_API_KEY
    modal.Secret.from_name("support-agent-secrets"),  # API_KEY, REDIS_URL, etc.
]

# ---------------------------------------------------------------------------
# Modal app
# ---------------------------------------------------------------------------

app = modal.App("intelligent-support-agent", image=image, secrets=secrets)


@app.function(
    cpu=1,
    memory=512,
    timeout=120,
    allow_concurrent_inputs=100,    # Handle 100 concurrent requests per container
    min_containers=1,               # Keep 1 warm container to avoid cold starts
)
@modal.asgi_app()
def fastapi_app():
    """Main ASGI app served via Modal web endpoint."""
    import sys
    sys.path.insert(0, "/app")
    from app.main import app as fastapi_application
    return fastapi_application


# ---------------------------------------------------------------------------
# Email worker (async batch processing)
# ---------------------------------------------------------------------------

@app.function(
    cpu=0.5,
    memory=256,
    timeout=60,
    schedule=modal.Period(minutes=1),   # Poll for new emails every minute
)
async def email_worker():
    """
    Background worker for processing inbound email queues.
    In production: poll an email queue (SQS, Redis list, etc.)
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Email worker tick — checking for queued emails...")
    # TODO: Implement email queue polling here
    pass


# ---------------------------------------------------------------------------
# Local testing
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Test the deployment locally."""
    import httpx
    # Start a Modal tunnel for local testing
    with modal.forward(8000) as tunnel:
        print(f"🚀 Tunnel URL: {tunnel.url}")
        print("Test with:")
        print(f"  curl -X POST {tunnel.url}/chat/message \\")
        print(f"    -H 'Content-Type: application/json' \\")
        print(f"    -H 'X-API-Key: your-key' \\")
        print(f"    -d '{{\"message\": \"Hello, I need help with my order\"}}'")
        input("Press Enter to stop...")
