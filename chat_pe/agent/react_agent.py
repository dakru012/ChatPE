from chat_pe.agent.base_agent import BaseAgent

import json
import re

class ReActAgent(BaseAgent):
    """
    Implements the ReAct (Reason + Act) architecture.
    Interleaves reasoning and tool execution in a loop.
    """
    def __init__(self, experiment_service, llm_service):
        super().__init__(experiment_service, llm_service)
        self.scratchpad = ""

    def run(self, user_input: str, max_steps: int = 10, add_to_history: bool = True):
        print(f"🎬 [ReAct] Processing query: {user_input}")
        if add_to_history:
            self.add_to_memory("user", user_input)
            
        # Reset image storage for the new turn
        self._produced_images = []
        
        action_history = []
        
        for step in range(max_steps):
            # 1. Reason
            prompt = self._build_react_prompt(user_input, self.scratchpad)
            response = self.llm_service.chat_completion(
                [{"role": "user", "content": prompt}],
                stop=["Observation:"]
            )
            
            # Ensure the response is properly prefixed for the history (since we nudge with 'Thought:')
            clean_response = response.strip()
            if not clean_response.startswith("Thought:"):
                formatted_response = "Thought: " + clean_response
            else:
                formatted_response = clean_response

            # Parse Action
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            
            # Print Thought for visibility (strip the tool call from it)
            thought_part = formatted_response.split("```json")[0].strip()
            if thought_part.startswith("Thought:"):
                thought_part = thought_part[len("Thought:"):].strip()
            
            if thought_part:
                print(f"💭 Thought: {thought_part}")
            
            self.scratchpad += f"\n{formatted_response}"
            
            # 2. Handle Action or Tool-based Final Answer
            if json_match:
                try:
                    action_data = json.loads(json_match.group(1))
                    action_name = action_data.get("action")
                    parameters = action_data.get("parameters", {})
                    
                    # Loop detection
                    action_sig = f"{action_name}:{json.dumps(parameters, sort_keys=True)}"
                    action_history.append(action_sig)
                    if len(action_history) >= 3 and action_history[-1] == action_history[-2] == action_history[-3]:
                        warning = f"Observation: You have called '{action_name}' with the same parameters 3 times in a row. This approach doesn't seem to be working. Please try a different query, a different tool, or re-examine the data you already have."
                        print(f"⚠️ Loop detected: {action_name}")
                        self.scratchpad += f"\n{warning}"
                        continue

                    # 2a. Check if it's the Final Answer tool
                    if action_name == "final_answer":
                        final_answer = parameters.get("answer", "No answer provided.")
                        print(f"🏁 Final Answer: {final_answer}")
                        
                        if add_to_history:
                            self.add_to_memory("agent", final_answer)
                        
                        if hasattr(self, '_produced_images') and self._produced_images:
                            result = {
                                "text": final_answer,
                                "images": self._produced_images
                            }
                        else:
                            result = final_answer

                        if add_to_history:
                            self.clear_scratchpad()

                        return result
                    
                    # 2b. Execute other tools
                    print(f"🛠️ Executing {action_name}...")
                    observation = self._execute_tool(action_name, parameters)
                     
                    obs_str = str(observation)
                    if len(obs_str) > 2000:
                        obs_str = obs_str[:1000] + "\n... (truncated) ...\n" + obs_str[-1000:]
                    
                    print(f"👁️ Observation: {obs_str[:150]}...")
                    self.scratchpad += f"\nObservation: {obs_str}"
                    continue # Go to next reasoning step
                except Exception as e:
                    error_msg = f"Error parsing/executing action: {str(e)}"
                    print(f"⚠️ {error_msg}")
                    self.scratchpad += f"\nObservation: {error_msg}"
                    continue

            # 3. Fallback: Nudge the agent if it produced raw text without a tool call
            nudge = "Observation: Your response did not contain a tool call (```json). Use 'final_answer' to conclude, or another tool to get data."
            print("⚠️ No tool call found. Nudging agent...")
            self.scratchpad += f"\n{nudge}"
            
        return "Reached maximum steps without a final answer."

    def clear_scratchpad(self):
        self.scratchpad = ""

    def _execute_tool(self, name, parameters):
        if name not in self.tools:
            return f"Error: Tool '{name}' not found. Available tools: {list(self.tools.keys())}"
        
        try:
            # For python_tool, the parameter is usually 'code'
            if name == "python_tool" and "code" in parameters:
                return self.tools[name]["func"](parameters["code"])

            result = self.tools[name]["func"](**parameters)
            return str(result)
        except Exception as e:
            return f"Tool execution failed: {str(e)}"

    def _build_react_prompt(self, question, scratchpad):
        tools_desc = "\n".join([f"- {name}: {info['description']}" for name, info in self.tools.items()])
        all_tables_info = self.experiment_service.get_table_info()
        schema_desc = []
        for table_name_cur, columns_info_cur in all_tables_info.items():
            schema_desc.append(f"Table: {table_name_cur}")
            col_list = [f"  - {col} (Type: {info['type']}, Example: {info['example']})" for col, info in columns_info_cur.items()]
            schema_desc.append("\n".join(col_list))
        
        schema_desc_str = "\n\n".join(schema_desc)
        example_table = list(all_tables_info.keys())[0] if all_tables_info else "experiments"

        is_direct = "sql_query" in self.tools
        example_tool = "sql_query" if is_direct else "query_dataframe"
        example_params = "{\"query\": \"SELECT * FROM experiments WHERE test_accuracy > 0.9\"}" if is_direct else "{\"query\": \"test_accuracy > 0.9\"}"
        python_hint = f"In direct SQL mode, use results from 'sql_query' (default variable: 'df_sql', or your custom name) in 'python_tool'." if is_direct else "In PyExperimenter mode, data is in 'df'. Use 'df.query()' inside 'python_tool' for filtering."
        
        # Include chat history (only user and agent messages)
        history_str = ""
        relevant_memory = [m for m in self.memory if m["role"] in ["user", "agent"]]
        if len(relevant_memory) > 1:
            history_str = "\n--- CHAT HISTORY ---\n"
            for msg in relevant_memory[:-1]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_str += f"{role}: {msg['content']}\n"
        
        return (
            "You are a Machine Learning Research Assistant. Your goal is to answer the user's question by analyzing experiment data through a structured, step-by-step reasoning process.\n\n"
            "--- SYSTEM CONTEXT ---\n"
            "The database contains the following tables and schemas:\n"
            f"{schema_desc_str}\n\n"
            f"{history_str}\n"
            "--- TOOLS AVAILABLE ---\n"
            f"{tools_desc}\n\n"
            "--- VISUALIZATION DISCIPLINE ---\n"
            "1. If you state that you will create a plot, your NEXT action MUST be calling `python_tool`.\n"
            "2. If the user requested a plot, you MUST use `python_tool` to create it and verify capture before answering.\n"
            "3. You are FORBIDDEN from describing, referencing, or claiming a plot exists until you see the `📸 Captured X plot(s).` confirmation in your Observation.\n"
            "4. If the `python_tool` returns `📸 Captured 0 plot(s).`, it means your code ran but did not produce a visible figure. You must fix your code or check your data before trying again.\n\n"
            "ORDER OF OPERATIONS: If a plot is requested, you must FIRST call `python_tool` to generate it. Once you have successfully generated all required plots, you should provide the Final Answer immediately using the `final_answer` tool. Do NOT reason 'The plot shows...' until after you see the capture confirmation.\n\n"
            "Variables you create in 'python_tool' (e.g., 'df_best = ...') PERSIST in session memory.\n"
            f"{python_hint}\n\n"
            "--- INSTRUCTIONS (ReAct Loop) ---\n"
            "You must strictly follow this iterative loop until you have the final answer:\n\n"
            "1. Thought: Reason about what you need to do next.\n"
            "2. Action: Call a tool using a JSON code block. This includes your FINAL ANSWER.\n"
            "   - Use the EXACT parameter names listed in the tool's description.\n"
            "   - If using 'python_tool' and you need to see a variable's value, you MUST either print it or put the variable name on the very last line.\n"
            "3. Observation: The system will provide the output of the tool.\n\n"
            "--- FINAL ANSWER INTEGRITY ---\n"
            "- NEVER provide a 'meta-answer' like 'I have calculated the average' or 'The results are ready'.\n"
            "- ALWAYS include the actual values, percentages, labels, or findings in your Final Answer.\n"
            "- If the data is not in your current turn's Observation or visible in the Chat History, YOU MUST CALL A TOOL TO GET IT.\n"
            "- Claiming to have performed a calculation or created a plot without a preceding 'Observation' of that event is a HALLUCINATION and is strictly forbidden.\n\n"
            "CRITICAL: Once you have the information needed to answer the user's request, YOU MUST STOP and provide the answer using the `final_answer` tool. Do NOT output raw text as your final response.\n\n"
            "FORMAT FOR TOOL CALLS:\n"
            "Thought: [your reasoning]\n"
            "```json\n"
            "{\"action\": \"tool_name\", \"parameters\": {\"param1\": \"val1\"}}\n"
            "```\n\n"
            "FORMAT FOR THE FINAL RESPONSE:\n"
            "Thought: I have finished my analysis.\n"
            "```json\n"
            "{\"action\": \"final_answer\", \"parameters\": {\"answer\": \"[your concise response to the user]\"}}\n"
            "```\n\n"
            "--- EXAMPLES ---\n"
            f"Thought: I need to find all experiments with accuracy > 0.9.\n"
            "```json\n"
            f"{{\"action\": \"{example_tool}\", \"parameters\": {example_params}}}\n"
            "```\n"
            "Observation: [List of experiments]\n\n"
            "Thought: I'll perform a custom query and save it for comparison.\n"
            "```json\n"
            f"{{\"action\": \"sql_query\", \"parameters\": {{\"query\": \"SELECT kernel, test_accuracy FROM {example_table} WHERE status = 'done'\", \"variable_name\": \"df_done\"}}}}\n"
            "```\n"
            "Observation: Result (stored in 'df_done'): ...\n\n"
            "Thought: I will now create a bar plot to visualize the performance per kernel.\n"
            "```json\n"
            "{\"action\": \"python_tool\", \"parameters\": {\"code\": \"import seaborn as sns\\nimport matplotlib.pyplot as plt\\nsns.barplot(data=df_done, x='kernel', y='test_accuracy')\\nplt.title('Accuracy per Kernel')\"}}\n"
            "```\n"
            "Observation: 📸 Captured 1 plot(s).\n\n"
            "Thought: I see the plot was captured. Based on the data and the generated plot, 'rbf' is the most accurate kernel.\n"
            "```json\n"
            "{\"action\": \"final_answer\", \"parameters\": {\"answer\": \"The 'rbf' kernel shows the best performance in the visualized data.\"}}\n"
            "```\n\n"
            "--- CURRENT TASK ---\n"
            f"Question: {question}\n\n"
            "--- WORKINGS (Reasoning History) ---\n"
            f"{self.scratchpad}\n"
            "Thought: "
        )
