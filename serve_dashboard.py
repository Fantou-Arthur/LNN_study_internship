import http.server
import socketserver
import webbrowser
import os

PORT = 8000

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Désactiver le cache pour voir les nouvelles images immédiatement
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
        print(f"\nDashboard disponible sur : http://localhost:{PORT}/dashboard.html")
        print("Appuyez sur Ctrl+C pour arrêter le serveur.")
        
        # Ouvrir automatiquement le navigateur
        webbrowser.open(f"http://localhost:{PORT}/dashboard.html")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServeur arrêté.")

if __name__ == "__main__":
    main()
