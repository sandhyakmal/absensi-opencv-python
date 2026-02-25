Running Aplikasi, download terlebih dahulu Python 3.9 atau 3.11

1.  python3 -m venv venv (mac), python -m venv venv / py -m venv venv (windows)
2.  source venv/bin/activate
3.  python -m pip install --upgrade pip
4.  pip install -r requirements.txt
5.  python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
