import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 8080
PWA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pwa')

class PWAHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PWA_DIR, **kwargs)

    def end_headers(self):
        # Enable Service Worker headers and caching controls
        if self.path.endswith('sw.js'):
            self.send_header('Service-Worker-Allowed', '/')
            self.send_header('Cache-Control', 'no-cache')
        elif self.path.endswith('manifest.json'):
            self.send_header('Content-Type', 'application/manifest+json')
            self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

def run_server(port=PORT):
    server_address = ('', port)
    httpd = HTTPServer(server_address, PWAHandler)
    print(f"============================================================")
    print(f"  AlphaEdge PWA Server running at: http://localhost:{port}")
    print(f"  PWA Directory: {PWA_DIR}")
    print(f"============================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping PWA Server...")
        httpd.server_close()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    run_server(port)
