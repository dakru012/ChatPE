import abc
import json

class BaseAgent(abc.ABC):
    """
    Abstract base class for all agents. 
    Provides tool registration and shared state management.
    """
    def __init__(self, experiment_service, llm_service):
        self.experiment_service = experiment_service
        self.llm_service = llm_service
        self.tools = {}
        self.memory = []
        self.python_context = {}  # Session memory for code execution
        self._setup_plotting()
        self._register_default_tools()

    def register_tool(self, name, func, description):
        self.tools[name] = {
            "func": func,
            "description": description
        }

    def _register_default_tools(self):
        """Registers the core tools defined in the requirements."""
        # Check if we are in direct DB mode (no experimenter but has direct_conn)
        is_direct = self.experiment_service.experimenter is None and \
                    hasattr(self.experiment_service, 'direct_conn') and \
                    self.experiment_service.direct_conn is not None

        if is_direct:
            self.register_tool(
                "sql_query",
                self._execute_sql_and_store,
                "Execute a read-only SQL SELECT query on the database. "
                "The result is returned as text AND stored as a pandas DataFrame in session memory. "
                "Parameters: {'query': str, 'variable_name': str (optional, default: 'df_sql')}"
            )
        else:
            self.register_tool(
                "query_dataframe", 
                lambda query: self._format_tool_result(self.experiment_service.query_dataframe(query)),
                "Filter experiments using pandas .query() syntax (e.g., 'status == \"done\"' or 'test_accuracy > 0.8'). "
                "Parameters: {'query': str}"
            )

        self.register_tool(
            "restart_runs",
            lambda status_list: self.experiment_service.restart_experiments(*status_list),
            "Reset experiments by status (e.g., ['error', 'done']). "
            "Note: This tool is only available when using a PyExperimenter configuration file."
            "Parameters: {'status_list': list of strings}"
        )
        self.register_tool(
            "python_tool",
            self.python_interpreter,
            "Run Python code to analyze experiment data. "
            "In PyExperimenter mode, data is in 'df'. In direct SQL mode, use 'df_sql' after running 'sql_query'. "
            "Variables created here are stored in session memory. "
            "CRITICAL: Do NOT attempt to load local CSV/SQL files (e.g., pd.read_csv). All data must come from previous tool outputs. "
            "IMPORTANT: If you need to see a result, print() it or ensure it's on the last line. "
            "Parameters: {'code': str}"
        )
        self.register_tool(
            "list_variables",
            self.list_working_memory_vars,
            "List all variables currently stored in session memory. "
            "Parameters: {}"
        )
        self.register_tool(
            "final_answer",
            lambda answer: answer,
            "Submit your final answer to the user. Use this only when you have completed all analysis. "
            "Parameters: {'answer': str}"
        )

    def _execute_sql_and_store(self, query, variable_name="df_sql"):
        """Executes SQL and stores result in python_context."""
        try:
            df = self.experiment_service.execute_sql(query)
            self.python_context[variable_name] = df
            result_str = self._format_tool_result(df)
            return f"💾 Saved SQL result to context variable '{variable_name}'. Use this variable in python_tool if needed.\n\nResult:\n{result_str}"
        except Exception as e:
            return f"Error executing SQL: {str(e)}"

    def _setup_plotting(self):
        """Forces non-interactive backend for matplotlib."""
        import matplotlib
        matplotlib.use('Agg')

    def python_interpreter(self, code: str):
        """Executes python code, capturing both stdout and the final expression value."""
        import pandas as pd
        import io
        import ast
        from contextlib import redirect_stdout
        
        import matplotlib.pyplot as plt
        
        # Always ensure latest 'df' and 'pd' are in context
        is_direct = self.experiment_service.experimenter is None and \
                    hasattr(self.experiment_service, 'direct_conn') and \
                    self.experiment_service.direct_conn is not None
        
        self.python_context["df"] = self.experiment_service.get_experiments()
        self.python_context["pd"] = pd
        self.python_context["plt"] = plt
        
        # Mock plt.show to prevent "non-interactive" warnings
        # Our auto-capture system handles the showing/saving
        self.python_context["plt"].show = lambda *args, **kwargs: None
        
        if is_direct and "df_sql" not in self.python_context:
            self.python_context["df_sql"] = pd.DataFrame()
        
        if code.startswith("```"):
            lines = code.splitlines()
            if lines[0].startswith("```"): lines = lines[1:]
            if lines and lines[-1].startswith("```"): lines = lines[:-1]
            code = "\n".join(lines).strip()
        
        # Strip redundant tool name if present at the start
        if code.lower().startswith("python_tool"):
            code = code[len("python_tool"):].strip()
        
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                # Use AST to split statements and the final expression
                try:
                    tree = ast.parse(code)
                except SyntaxError as se:
                     return f"Syntax Error at line {se.lineno}, offset {se.offset}: {se.msg}\nCode snippet: {se.text}"

                eval_node = None
                
                # If the last node is an expression, we want to eval it
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    eval_node = tree.body.pop()
                
                # Compile and execute the statements
                if tree.body:
                    exec(compile(tree, filename="<agent_code>", mode="exec"), self.python_context)
                
                # Prepare result from the final expression if it exists
                eval_result = None
                if eval_node:
                    eval_result = eval(compile(ast.Expression(eval_node.value), filename="<agent_code>", mode="eval"), self.python_context)
                
                # Combine results
                results = []

                # Auto-capture matplotlib plots
                captured_count = self._capture_plots()
                if captured_count > 0:
                    results.append(f"📸 Captured {captured_count} plot(s).")
                
            stdout_val = output.getvalue().strip()
            
            if stdout_val:
                results.append(stdout_val)
            if eval_result is not None:
                results.append(self._format_tool_result(eval_result))
            
            if not results:
                if "result" in self.python_context:
                    return self._format_tool_result(self.python_context["result"])
                return "Code executed successfully (no output)."
            
            return "\n".join(results)
            
        except Exception as e:
            import traceback
            # Clear any partial plots on failure to prevent leakage
            plt.close('all')
            error_msg = f"Error executing code: {str(e)}\n{traceback.format_exc()}"
            return error_msg

    def _format_tool_result(self, result):
        import pandas as pd
        if isinstance(result, (pd.DataFrame, pd.Series)):
            if result.empty:
                return "Empty result set."
            
            # Smart summarization for large DataFrames
            if isinstance(result, pd.DataFrame) and len(result) > 20:
                summary = [
                    f"DataFrame Summary (Size: {result.shape})",
                    "Columns: " + ", ".join(result.columns.tolist()),
                    "\nFirst 10 rows:",
                    result.head(10).to_string(),
                    "\n...",
                    "Last 10 rows:",
                    result.tail(10).to_string(),
                    f"\nTotal rows: {len(result)}"
                ]
                return "\n".join(summary)
            
            return result.to_string()
        return str(result)

    def _capture_plots(self):
        """Checks for active matplotlib figures and captures them as images. Returns count."""
        import matplotlib.pyplot as plt
        import io
        import base64
        
        fig_nums = plt.get_fignums()
        if not fig_nums:
            return 0

        if not hasattr(self, '_produced_images'):
            self._produced_images = []

        count = 0

        for num in fig_nums:
            try:
                fig = plt.figure(num)
                
                # Check for content to avoid capturing empty/blank plots
                has_content = False
                if fig.axes:
                    for ax in fig.axes:
                        if (ax.get_lines() or ax.get_images() or ax.patches or 
                            ax.get_children() and any(not hasattr(c, 'get_visible') or c.get_visible() for c in ax.get_children() if type(c).__name__ not in ['Spine', 'XAxis', 'YAxis', 'Text'])):
                            if (len(ax.get_lines()) > 0 or len(ax.get_images()) > 0 or 
                                len(ax.patches) > 0 or len(ax.collections) > 0 or
                                len(ax.containers) > 0):
                                has_content = True
                                break
                                
                if not has_content:
                    print(f"⚠️ Figure {num} appears empty. Skipping capture.")
                    plt.close(fig)
                    continue

                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                self._produced_images.append(f"data:image/png;base64,{img_base64}")
                print(f"📸 Captured plot from figure {num}")
                count += 1
            except Exception as e:
                print(f"⚠️ Failed to capture figure {num}: {e}")
                try: plt.close(fig)
                except: pass
        
        return count

    def list_working_memory_vars(self, *args):
        """Lists user-defined variables in the python context."""
        # Filter out builtins and common modules/functions
        ignored = {'__builtins__', 'pd', 'df', 'capture_args', '_capture_args', 'plt', 'io', 'base64', 'ast', 'redirect_stdout'}
        vars_list = [v for v in self.python_context.keys() if v not in ignored and not v.startswith('_')]
        if not vars_list:
            return "No variables currently in session memory."
        
        results = []
        for v in vars_list:
            val = self.python_context[v]
            type_str = type(val).__name__
            summary = ""
            if hasattr(val, 'shape'):
                summary = f" (shape: {val.shape})"
            elif isinstance(val, (list, dict)):
                summary = f" (len: {len(val)})"
            results.append(f"- {v}: {type_str}{summary}")
            
        return "Variables in session memory:\n" + "\n".join(results)

    @abc.abstractmethod
    def run(self, user_input: str):
        pass

    def add_to_memory(self, role, content):
        self.memory.append({"role": role, "content": content})
