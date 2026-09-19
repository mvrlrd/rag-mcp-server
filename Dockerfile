FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY sample_docs/ ./sample_docs/

# MCP-сервер (streamable-http)
EXPOSE 8000

CMD ["python", "-m", "rag_mcp.server"]
