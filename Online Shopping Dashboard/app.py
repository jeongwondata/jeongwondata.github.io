# =============================================================
# DATS6401 Final Term Project — Phase II: Run Dashboard
# Usage: python app.py
# Then open: http://127.0.0.1:8050
# =============================================================

from dashboard import app
import callbacks  # registers all callbacks

if __name__ == '__main__':
    app.run(debug=True, port=8050)
