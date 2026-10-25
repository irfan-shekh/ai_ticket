import os
import sys

# Ensure root directory is on Python path so all modules, models, and assets can be imported
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from werkzeug.middleware.proxy_fix import ProxyFix
from app import app

class VercelPathFixMiddleware:
    """
    Vercel CLI 65+ rewrites incoming paths to the destination path (/api/index).
    This middleware restores the original requested client path from Vercel's
    headers (x-matched-path or x-forwarded-uri) or strips the /api/index prefix,
    ensuring Flask routes (/, /login, /dashboard, etc.) match correctly.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        matched = environ.get('HTTP_X_MATCHED_PATH') or environ.get('HTTP_X_FORWARDED_URI')
        if matched:
            environ['PATH_INFO'] = matched.split('?')[0]
        else:
            path_info = environ.get('PATH_INFO', '')
            if path_info.startswith('/api/index'):
                remainder = path_info[len('/api/index'):]
                environ['PATH_INFO'] = remainder if remainder.startswith('/') else ('/' + remainder)
            elif path_info == '/app.py':
                environ['PATH_INFO'] = '/'

        return self.wsgi_app(environ, start_response)

# Apply middlewares
app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
