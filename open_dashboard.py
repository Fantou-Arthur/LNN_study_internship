import http.server
import socketserver
import webbrowser
import os
import threading

PORT = 8000
DIRECTORY = os.getcwd()

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def start_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serveur actif sur http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    # Lancer le serveur dans un thread séparé
    thread = threading.Thread(target=start_server, daemon=True)
    thread.start()
    
    # Ouvrir le navigateur
    url = f"http://localhost:{PORT}/dashboard_oil_comprehensive.html"
    print(f"Ouverture de {url}...")
    webbrowser.open(url)
    
    # Maintenir le script en vie
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServeur arrêté.")
