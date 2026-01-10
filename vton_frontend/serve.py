#!/usr/bin/env python3
"""
Simple HTTP server for the VTON Frontend.
Serves static files and handles CORS for local development.
"""

import http.server
import socketserver
import os
import sys

PORT = int(os.environ.get('PORT', 3000))

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with CORS support."""
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    handler = CORSHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print("=" * 50)
        print(f"🌐 VTON Frontend Server")
        print("=" * 50)
        print(f"   Serving at: http://localhost:{PORT}")
        print(f"   Directory:  {os.getcwd()}")
        print("")
        print("   Press Ctrl+C to stop")
        print("=" * 50)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")
            sys.exit(0)


if __name__ == "__main__":
    main()
