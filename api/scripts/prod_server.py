import os
import gunicorn.app.base
import uvicorn

# Set environment variable for Keras compatibility with transformers
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


class GunicornApp(gunicorn.app.base.BaseApplication):
    def __init__(self, app, options=None):
        self.app = app
        self.options = options or {}

    def load(self):
        return self.app

    def load_config(self):
        for key, value in self.options.items():
            self.cfg.set(key, value)


def main():
    options = {
        "bind": "0.0.0.0:8000",
        "workers": 4,
        "worker_class": "uvicorn.workers.UvicornWorker",
    }
    app = uvicorn.run("app.main:app", app_dir=None)  # Point to your FastAPI app
    GunicornApp(app, options).run()
