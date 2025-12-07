import os
import uvicorn

# Set environment variable for Keras compatibility with transformers
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def main():
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
