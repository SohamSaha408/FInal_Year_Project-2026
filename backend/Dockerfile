FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
ENV VEDAS_BASE_URL=https://vedas.sac.gov.in/vconsole
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
