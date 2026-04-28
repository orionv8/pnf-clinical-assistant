import http.server
import socketserver
import os

PORT = 8501
os.chdir('/home/ec2-user/.hermes/profiles/hermes-main/home/projects/pnf-clinical-assistant/pnf-clinical-assistant-main/')

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()
