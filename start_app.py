import subprocess
import os
import sys
import time
import socket
import threading
import webbrowser

HOST = "localhost"
PORT = 8000
URL  = "http://{}:{}".format(HOST, PORT)

DEPS_STAMP_FILE = ".deps_stamp"

def kill_port(port):
    try:
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True
        )
        pids = set()
        for line in result.stdout.splitlines():
            if ":{}".format(port) in line and "LISTENING" in line:
                parts = line.strip().split()
                if parts:
                    pids.add(parts[-1])
        for pid in pids:
            subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
            print("  - Killed old server (PID {})".format(pid))
        if pids:
            time.sleep(1.5)
    except Exception as e:
        print("  - WARNING: Could not kill old process: {}".format(e))

def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0

def wait_and_open_browser(timeout=30):
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(URL, timeout=1)
            print("\n  Server is ready!  Opening {} ...\n".format(URL))
            webbrowser.open(URL)
            return
        except Exception:
            time.sleep(0.5)
    print("  WARNING: Server did not respond in {}s. Open manually: {}".format(timeout, URL))

def requirements_changed():
    req_path = "requirements.txt"
    if not os.path.exists(req_path):
        return False

    req_mtime = str(os.path.getmtime(req_path))
    if not os.path.exists(DEPS_STAMP_FILE):
        return True

    try:
        with open(DEPS_STAMP_FILE, "r", encoding="utf-8") as f:
            last = f.read().strip()
        return last != req_mtime
    except Exception:
        return True

def write_deps_stamp():
    req_path = "requirements.txt"
    if not os.path.exists(req_path):
        return
    try:
        with open(DEPS_STAMP_FILE, "w", encoding="utf-8") as f:
            f.write(str(os.path.getmtime(req_path)))
    except Exception:
        pass

def run():
    print("--------------------------------------------------")
    print("  ASL-Connect  |  Booting System")
    print("--------------------------------------------------")

    # 1. Ollama check
    print("\n[1/4] Checking Ollama Service...")
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1) as res:
            status = getattr(res, "status", 200)
        if status == 200:
            print("  OK: Ollama is online.")
    except Exception:
        print("  WARNING: Ollama not running - translation features may not work.")
        print("    -> https://ollama.com  then run:  ollama run llama3.2")

    # 2. Install dependencies (only when needed)
    print("\n[2/4] Checking Python Dependencies...")
    try:
        force_install = os.environ.get("FORCE_PIP_INSTALL", "").strip() == "1"
        if force_install or requirements_changed():
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            write_deps_stamp()
            print("  OK: Dependencies installed/updated.")
        else:
            print("  OK: Dependencies unchanged (skipping pip install).")
    except Exception as e:
        print("  ERROR: Failed to install requirements: {}".format(e))
        return

    # 2.5 Database: apply schema migrations + load sign data (Postgres).
    print("\n[DB] Applying migrations and ensuring sign data...")
    if not os.environ.get("DATABASE_URL"):
        print("  - DATABASE_URL not set; using default "
              "postgresql+psycopg://together:together@localhost:5432/together")
    try:
        subprocess.check_call([sys.executable, "-m", "alembic", "upgrade", "head"])
        subprocess.check_call([sys.executable, "scripts/migrate_to_postgres.py"])
        print("  OK: Database ready.")
    except Exception as e:
        print("  WARNING: DB setup step failed ({}). Is Postgres running? "
              "See README (docker-compose up brings it up automatically).".format(e))



    # 3. Free port 8000 if occupied
    print("\n[3/4] Checking Port {}...".format(PORT))
    if port_in_use(PORT):
        print("  - Port {} occupied - restarting old server...".format(PORT))
        kill_port(PORT)
    print("  OK: Port {} is free.".format(PORT))

    # 4. Launch server
    print("\n[4/4] Launching ASL-Connect Server at {}".format(URL))
    print("      (Browser will open automatically)\n")

    # Set PYTHONPATH so uvicorn can import app/server modules
    env = os.environ.copy()
    server_dir = os.path.join(os.path.abspath("."), "app", "server")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (server_dir + os.pathsep + existing) if existing else server_dir

    # Open browser in background once server responds
    threading.Thread(target=wait_and_open_browser, daemon=True).start()

    # Run uvicorn directly - blocks until Ctrl+C
    try:
        subprocess.run(
            [
                sys.executable, "-m", "uvicorn",
                "main:socket_app",
                "--host", "0.0.0.0",
                "--port", str(PORT),
                "--app-dir", server_dir,
            ],
            env=env,
        )
    except KeyboardInterrupt:
        pass
    finally:
        print("\n--------------------------------------------------")
        print("  ASL-Connect shut down. Goodbye!")
        print("--------------------------------------------------")

if __name__ == "__main__":
    run()
