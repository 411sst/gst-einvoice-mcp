# Tesseract is a system binary, so the image installs it with apt rather than pip.
# Without it every scanned page fails; gst_einvoice.ocr raises naming both the places
# it looked and the install command, rather than returning empty text.
FROM python:3.13-slim

# tesseract-ocr-eng is the language data. The base tesseract-ocr package alone
# installs the engine with no traineddata, and every OCR call then fails at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency metadata first, so a source-only change does not reinstall the world.
COPY pyproject.toml ./
COPY gst_einvoice ./gst_einvoice
RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1 \
    GST_MCP_TRANSPORT=streamable-http

EXPOSE 10000

CMD ["python", "-m", "gst_einvoice.server"]
