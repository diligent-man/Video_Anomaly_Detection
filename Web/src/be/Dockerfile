FROM python:3.11.10

RUN adduser nonroot
RUN apt-get update && apt-get install -y curl && apt-get clean

RUN apt-get update && apt-get install -y \
    curl \
#    ffmpeg \
#    libsm6 \
#    libxext6 \
#    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /Web/src/be && chown -R nonroot:nonroot /Web/src/be

USER nonroot
COPY --chown=nonroot:nonroot . /Web/src/be

# Directly install requirements without venv
ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3.11 -m venv $VIRTUAL_ENV
RUN pip install --no-cache-dir -r /Web/src/be/requirements.txt

EXPOSE 6967
CMD ["python3", "/Web/src/be/main.py"]
