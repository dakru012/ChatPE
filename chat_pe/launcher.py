import os
import sys
import subprocess
import argparse
import webbrowser
import time
import threading

def open_browser(url):
    """Wait a second for the server to start, then open the browser."""
    time.sleep(5.0)
    print(f"🌐 Opening chat interface at {url}...")
    webbrowser.open(url)

def launch():
    parser = argparse.ArgumentParser(description="Experiment Chat Launcher")
    parser.add_argument("--config", type=str, help="Path to the PyExperimenter configuration file")
    parser.add_argument("--db", type=str, help="Path to the database file")
    parser.add_argument("--table", type=str, help="Main table name in the database (optional)")
    parser.add_argument("--db_provider", type=str, help="Database provider (sqlite or mysql)")
    parser.add_argument("--db_host", type=str, help="Database host")
    parser.add_argument("--db_user", type=str, help="Database user")
    parser.add_argument("--db_password", type=str, help="Database password")
    parser.add_argument("--logtables", type=str, help="Comma-separated list of additional logtables to include")
    parser.add_argument("--port", type=int, default=5000, help="Port for the Flask server (default: 5000)")
    parser.add_argument("--arch", type=str, default="react", help="Agent architecture: 'react' or 'plan_execute'")
    args = parser.parse_args()
    
    agent_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(agent_dir, "backend", "app.py")
    
    # If neither config nor db is provided, check for default config
    if not args.config and not args.db:
        default_config = os.path.join(agent_dir, "config", "example_general_usage.yml")
        if os.path.exists(default_config):
            args.config = default_config

    # Ensure config path is absolute and valid if provided
    config_abs = None
    if args.config:
        config_abs = os.path.abspath(args.config)
        if not os.path.exists(config_abs):
            # Try looking in example_project relative to ML_PROJECT root
            root_dir = os.path.dirname(agent_dir)
            alt_path = os.path.join(root_dir, "example_project", "config", os.path.basename(args.config))
            if os.path.exists(alt_path):
                config_abs = alt_path
            else:
                print(f"❌ Error: Config file not found at {args.config}")
                sys.exit(1)
    
    print(f"🚀 Starting Experiment Chat Agent ({args.arch})...")
    if config_abs:
        print(f"📁 Project Config: {config_abs}")
    if args.db:
        print(f"🗄️ Database: {args.db}")
    
    url = f"http://127.0.0.1:{args.port}"
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()
    
    # Use the current Python interpreter for the subprocess
    python_exe = sys.executable
    
    cmd = [python_exe, app_path, "--arch", args.arch]
    if config_abs:
        cmd.extend(["--config", config_abs])
    if args.db:
        cmd.extend(["--db", args.db])
    if args.table:
        cmd.extend(["--table", args.table])
    if args.db_provider:
        cmd.extend(["--db_provider", args.db_provider])
    if args.db_host:
        cmd.extend(["--db_host", args.db_host])
    if args.db_user:
        cmd.extend(["--db_user", args.db_user])
    if args.db_password:
        cmd.extend(["--db_password", args.db_password])
    if args.logtables:
        cmd.extend(["--logtables", args.logtables])
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Stopping agent.")

if __name__ == "__main__":
    launch()
