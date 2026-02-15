import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from chat_pe.agent import ReActAgent, PlanAndExecuteAgent
from chat_pe.services.experiment_service import ExperimentService
from chat_pe.services.llm import LLMService

import argparse

parser = argparse.ArgumentParser(description="Start the Experiment Chat Agent")
parser.add_argument("--config", type=str, help="Path to the PyExperimenter configuration file")
args, unknown = parser.parse_known_args()

def create_app(agent_instance=None):
    frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
    app = Flask(__name__, static_folder=frontend_path, static_url_path='')
    CORS(app)

    @app.route('/')
    def serve_index():
        return app.send_static_file('index.html')

    if agent_instance:
        app.agent = agent_instance
    else:
        parser = argparse.ArgumentParser(description="Start the Experiment Chat Agent")
        parser.add_argument("--config", type=str, help="Path to the PyExperimenter configuration file")
        parser.add_argument("--db", type=str, help="Path to the database file")
        parser.add_argument("--table", type=str, help="Main table name in the database")
        parser.add_argument("--db_provider", type=str, help="Database provider (sqlite or mysql)")
        parser.add_argument("--db_host", type=str, help="Database host (for mysql)")
        parser.add_argument("--db_user", type=str, help="Database user (for mysql)")
        parser.add_argument("--db_password", type=str, help="Database password (for mysql)")
        parser.add_argument("--logtables", type=str, help="Comma-separated list of additional logtables")
        parser.add_argument("--arch", type=str, default="react", help="Agent architecture (react or plan_execute)")
        args, unknown = parser.parse_known_args()
        
        logtables_list = args.logtables.split(",") if args.logtables else []
        
        exp_service = ExperimentService(
            config_path=args.config, 
            db_path=args.db, 
            table_name=args.table,
            db_provider=args.db_provider,
            db_host=args.db_host,
            db_user=args.db_user,
            db_password=args.db_password,
            logtables=logtables_list
        )
        llm_service = LLMService()
        
        if args.arch == "plan_execute":
            app.agent = PlanAndExecuteAgent(exp_service, llm_service)
        else:
            app.agent = ReActAgent(exp_service, llm_service)

    @app.route("/chat", methods=["POST"])
    def chat():
        user_message = request.json.get("message", "")
        if not user_message:
            return jsonify({"response": "I didn't receive a message."}), 400
            
        response = app.agent.run(user_message)
        return jsonify({"response": response})
        
    return app

def run_app(agent_instance=None, port=5000, debug=True):
    app = create_app(agent_instance)
    app.run(debug=debug, port=port, use_reloader=False)

if __name__ == "__main__":
    run_app()

