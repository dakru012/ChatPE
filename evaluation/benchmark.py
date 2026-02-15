import os
import sys
import json
import argparse
import time
from datetime import datetime
import base64

from chat_pe.agent import ReActAgent, PlanAndExecuteAgent
from chat_pe.services.experiment_service import ExperimentService
from chat_pe.services.llm import LLMService

def parse_questions(file_path):
    """Parser for JSON question files or legacy markdown/text files."""
    if not os.path.exists(file_path):
        print(f"❌ Question file not found: {file_path}")
        return None, []
        
    if file_path.endswith('.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                dataset_name = data.get("dataset", "unknown_dataset")
                questions = data.get("questions", [])
                return dataset_name, questions
        except Exception as e:
            print(f"❌ Error parsing JSON questions: {e}")
            return None, []
    
    # Legacy parser
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line.startswith(('-', '*')):
                q = line[1:].strip()
                if q.startswith('"') and q.endswith('"'): q = q[1:-1]
                if q: questions.append({"id": f"Q{i+1}", "question": q})
            elif not line.startswith('#') and line:
                q = line.strip()
                if q.startswith('"') and q.endswith('"'): q = q[1:-1]
                questions.append({"id": f"Q{i+1}", "question": q})
    return "legacy_run", questions

def run_benchmark():
    parser = argparse.ArgumentParser(description="Agent Benchmark Suite")
    # Connection args
    parser.add_argument("--config", type=str, help="Path to PyExperimenter config")
    parser.add_argument("--db", type=str, help="Path to database file (direct mode)")
    parser.add_argument("--table", type=str, help="Table name (direct mode)")
    parser.add_argument("--db_provider", type=str, default="sqlite", help="DB provider (direct mode)")
    parser.add_argument("--db_host", type=str, help="DB host (direct mode)")
    parser.add_argument("--db_user", type=str, help="DB user (direct mode)")
    parser.add_argument("--db_password", type=str, help="DB password (direct mode)")
    parser.add_argument("--logtables", type=str, help="Comma-separated list of additional logtables")
    
    # Run args
    parser.add_argument("--questions", type=str, default="evaluation/questions/yahpogym_experiments_ds_tunability.json", help="Path to questions file")
    parser.add_argument("--arch", type=str, default="react", choices=["react", "plan_execute"], help="Agent architecture")
    parser.add_argument("--output_base", type=str, default="evaluation/results", help="Base directory for results")
    args = parser.parse_args()

    if not args.config and not args.db:
        print("❌ Error: Must provide either --config or --db")
        sys.exit(1)

    logtables_list = args.logtables.split(",") if args.logtables else []

    # Load questions
    dataset_name, questions = parse_questions(args.questions)
    if not questions:
        print("❌ No questions found. Exiting.")
        sys.exit(1)
        
    print(f"🚀 Initializing Benchmark for architecture: {args.arch} | Dataset: {dataset_name}")
    
    # Initialize services
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
        agent = PlanAndExecuteAgent(exp_service, llm_service)
    else:
        agent = ReActAgent(exp_service, llm_service)

    # Prepare output directory: evaluation/results/<dataset>/<arch>_<timestamp>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_base, dataset_name, f"{args.arch}_{timestamp}")
    img_dir = os.path.join(run_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    results = []

    for i, q_item in enumerate(questions):
        q_id = q_item.get("id", f"Q{i+1}")
        q_text = q_item.get("question", "")
        
        print(f"\n[{i+1}/{len(questions)}] ({q_id}) Processing: {q_text}")
        start_time = time.time()
        
        try:
            # ReActAgent.run() is the core entry point
            response = agent.run(q_text)
            duration = time.time() - start_time
            
            text_result = ""
            images = []
            
            if isinstance(response, dict):
                text_result = response.get("text", "")
                images = response.get("images", [])
            else:
                text_result = response

            # Save images: <id>_<arch>_img<n>.png
            image_paths = []
            for j, img_data in enumerate(images):
                img_filename = f"{q_id}_{args.arch}_img{j+1}.png"
                img_path = os.path.join(img_dir, img_filename)
                
                if "," in img_data:
                    img_data = img_data.split(",")[1]
                    
                with open(img_path, "wb") as f:
                    f.write(base64.b64decode(img_data))
                
                image_paths.append(os.path.join("images", img_filename))

            results.append({
                "id": q_id,
                "question": q_text,
                "response": text_result,
                "images": image_paths,
                "duration_seconds": round(duration, 2),
                "status": "success"
            })
            print(f"✅ Completed in {round(duration, 2)}s")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({
                "id": q_id,
                "question": q_text,
                "error": str(e),
                "status": "error"
            })

    # Save JSON report
    summary_path = os.path.join(run_dir, f"report_{args.arch}.json")
    with open(summary_path, "w", encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "dataset": dataset_name,
                "timestamp": timestamp,
                "architecture": args.arch,
                "config": args.config or args.db,
                "total_questions": len(questions),
                "successful": len([r for r in results if r.get("status") == "success"]),
                "failed": len([r for r in results if r.get("status") == "error"])
            },
            "results": results
        }, f, indent=2)

    # Generate Markdown report
    md_report_path = os.path.join(run_dir, f"report_{args.arch}.md")
    with open(md_report_path, "w", encoding='utf-8') as f:
        f.write(f"# Benchmark Report: {dataset_name} ({args.arch})\n")
        f.write(f"- **Timestamp:** {timestamp}\n")
        f.write(f"- **Agent:** {args.arch}\n")
        f.write(f"- **Status:** {len([r for r in results if r.get('status') == 'success'])}/{len(questions)} successful\n\n")
        
        for r in results:
            f.write(f"### {r['id']}: {r['question']}\n")
            if r.get("status") == "success":
                f.write(f"{r['response']}\n\n")
                if r["images"]:
                    for img in r["images"]:
                        f.write(f"![{r['id']}]({img})\n\n")
                f.write(f"*Duration: {r['duration_seconds']}s*\n\n")
            else:
                f.write(f"**❌ Error:** {r.get('error')}\n\n")
            f.write("---\n")

    print(f"\n🏁 Benchmark finished! Results: {run_dir}")

if __name__ == "__main__":
    run_benchmark()
