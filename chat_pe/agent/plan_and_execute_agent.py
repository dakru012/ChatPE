from chat_pe.agent.base_agent import BaseAgent
from chat_pe.agent.react_agent import ReActAgent
import json

class PlanAndExecuteAgent(BaseAgent):
    """
    Implements the Plan-and-Execute architecture.
    Separates planning from multi-step execution.
    """
    def __init__(self, experiment_service, llm_service):
        super().__init__(experiment_service, llm_service)
        self.executor = ReActAgent(experiment_service, llm_service)
        self.executor.memory = self.memory

    def run(self, user_input: str):
        print(f"🗺️ [Plan-and-Execute] Planning for: {user_input}")
        self.add_to_memory("user", user_input)
        
        # 1. Plan
        plan_prompt = self._build_planner_prompt(user_input)
        plan_json = self.llm_service.chat_completion([{"role": "user", "content": plan_prompt}], format="json")
        self.executor.clear_scratchpad()
        
        try:
            plan_data = json.loads(plan_json)
            if isinstance(plan_data, dict):
                for v in plan_data.values():
                    if isinstance(v, list):
                        plan = v
                        break
                else:
                    plan = [plan_data]
            else:
                plan = plan_data
            print(f"📋 Generated Plan: {json.dumps(plan, indent=2)}")
        except Exception as e:
            return f"Failed to generate a valid plan: {str(e)}"
        
        # 2. Execute
        results = []
        all_images = []
        for step in plan:
            print(f"🚶 Executing Step {step.get('step', '?')}: {step.get('description', '')}")
            res = self.executor.run(
                f"Task: {step.get('description', '')}. Context: {str(results)}", 
                add_to_history=False
            )
            
            if isinstance(res, dict):
                text_res = res.get("text", "")
                all_images.extend(res.get("images", []))
            else:
                text_res = res
                
            results.append({"step": step.get('step', '?'), "result": text_res})
            
        # 3. Final Synthesis
        summary_prompt = (
            "Based on the following execution steps and results, provide a concise final answer to the user.\n\n"
            "--- EXECUTION RESULTS ---\n"
            f"{json.dumps(results, indent=2)}\n\n"
            "--- RULES ---\n"
            "1. Output ONLY the answer. Do NOT use meta-talk like 'Based on the results...' or 'Here is your answer'.\n"
            "2. If the user asked a question, provide the direct finding or value.\n"
            "3. If a plot was created, simply state that the visualization has been generated or refer to what it shows.\n"
            f"User Question: {user_input}"
        )
        final_answer = self.llm_service.chat_completion([{"role": "user", "content": summary_prompt}])
        
        self.executor.clear_scratchpad()
        
        self.add_to_memory("agent", final_answer)
        
        if all_images:
            return {
                "text": final_answer,
                "images": all_images
            }
        return final_answer

    def _build_planner_prompt(self, user_input):
        history_str = ""
        relevant_memory = [m for m in self.memory if m["role"] in ["user", "agent"]]
        if len(relevant_memory) > 1:
            history_str = "\n--- CHAT HISTORY ---\n"
            for msg in relevant_memory[:-1]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_str += f"{role}: {msg['content']}\n"

        all_tables_info = self.experiment_service.get_table_info()
        schema_desc = []
        for table_name_cur, columns_info_cur in all_tables_info.items():
            schema_desc.append(f"Table: {table_name_cur}")
            col_list = [f"  - {col} (Example: {info['example']})" for col, info in columns_info_cur.items()]
            schema_desc.append("\n".join(col_list))
        
        schema_desc_str = "\n\n".join(schema_desc)
        
        is_direct = "sql_query" in self.tools

        return (
            "You are a strategic planner for Machine Learning experiment analysis.\n"
            "Break down the user question into a sequence of logical steps.\n\n"
            "--- SYSTEM CONTEXT ---\n"
            "The database contains the following tables and schemas:\n"
            f"{schema_desc_str}\n\n"
            f"{history_str}\n"
            "--- AVAILABLE TOOLS ---\n"
            "[" + ", ".join(self.tools.keys()) + "]\n"
            "NOTE: For any visualization, simply use 'python_tool' with matplotlib or seaborn. Do NOT call `plt.show()`.\n"
            "CRITICAL: A plot ONLY exists after a `python_tool` step. `sql_query` only fetches data. Ensure the plan explicitly separates 'Fetch Data' from 'Create Plot'.\n"
            "DATA INTEGRITY: All analysis data MUST come from tools. Do NOT assume local files (CSVs/SQL files) exist. Your steps must retrieve data first.\n"
            "The analyzer will conclude immediately once a final result or plot is provided.\n"
            f"{'In direct SQL mode. Use sql_query to fetch data (you can specify a custom variable_name).' if is_direct else 'In PyExperimenter mode. Use query_dataframe to fetch data.'}\n\n"
            "Output MUST be a JSON list of objects: [{\"step\": 1, \"action\": \"tool_name\", \"description\": \"task\"}]\n\n"
            f"Question: {user_input}"
        )
