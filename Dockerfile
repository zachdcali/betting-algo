# Use latest Python 3.13 slim image
FROM python:3.13-slim

# Prevent Python from writing .pyc files and enable real-time logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libxss1 \
    libgtk-3-0 \
    libgbm-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip, install Python dependencies, then install Playwright browsers
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    playwright install chromium && \
    playwright install-deps chromium

# Copy the rest of the application
COPY . /app

# Set the default command to open a bash shell
CMD ["bash"]