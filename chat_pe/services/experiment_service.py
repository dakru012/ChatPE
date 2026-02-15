import os
import pandas as pd
from py_experimenter.experimenter import PyExperimenter

class ExperimentService:
    def __init__(self, config_path: str = None, db_path: str = None, table_name: str = None, 
                 db_provider: str = None, db_host: str = None, db_user: str = None, db_password: str = None,
                 experimenter: PyExperimenter = None, logtables: list = None):
        # Set pandas options for better LLM readability
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 100)

        self.experimenter = experimenter
        self.direct_conn = None
        self.db_path = db_path
        self.table_name = table_name
        self.logtables = logtables or []
        self.provider = db_provider or 'sqlite'
        
        if self.experimenter:
            self.config_path = "Injected from instance" 
            return

        self.config_path = config_path or os.environ.get('PY_EXPERIMENTER_CONFIG')
        
        if self.config_path:
            # Robust path resolution
            self.config_path = os.path.abspath(self.config_path)
            if not os.path.exists(self.config_path):
                potential_paths = [
                    os.path.abspath(os.path.join(os.getcwd(), 'example_project', 'config', os.path.basename(self.config_path))),
                    os.path.abspath(os.path.join(os.getcwd(), 'config', os.path.basename(self.config_path))),
                ]
                for p in potential_paths:
                    if os.path.exists(p):
                        self.config_path = p
                        break
            
            # Discover logtables from YAML if not provided
            if not self.logtables and os.path.exists(self.config_path):
                try:
                    import yaml
                    with open(self.config_path, 'r') as f:
                        config = yaml.safe_load(f)
                        db_config = config.get('PY_EXPERIMENTER', {}).get('Database', {})
                        if not self.table_name:
                            self.table_name = db_config.get('table', {}).get('name')
                        
                        logtables_config = db_config.get('logtables', {})
                        if logtables_config:
                            self.logtables.extend(list(logtables_config.keys()))
                except Exception as e:
                    print(f"⚠️ Warning: Could not parse logtables from YAML: {e}")

            try:
                self.experimenter = PyExperimenter(
                    experiment_configuration_file_path=self.config_path, 
                    name='chat_agent_service'
                )
            except Exception as e:
                print(f"Error initializing PyExperimenter: {e}")
        elif db_path:
            # Direct DB mode
            self._init_direct_connection(db_path, table_name, db_provider, db_host, db_user, db_password)

    def _init_direct_connection(self, db_path, table_name, provider, host, user, password):
        self.provider = provider or 'sqlite'
        self.db_path = os.path.abspath(db_path)
        self.table_name = table_name

        if self.provider == 'sqlite':
            import sqlite3
            self.direct_conn = sqlite3.connect(self.db_path, check_same_thread=False)
            print(f"✅ Connected directly to SQLite database: {self.db_path}")
        elif self.provider == 'mysql':
            try:
                from sqlalchemy import create_engine
                # Format: mysql+mysqlconnector://user:password@host/dbname
                conn_str = f"mysql+mysqlconnector://{user or 'root'}:{password or ''}@{host or 'localhost'}/{db_path}"
                self.engine = create_engine(conn_str)
                self.direct_conn = self.engine.connect()
                print(f"✅ Connected directly to MySQL (SQLAlchemy): {db_path} at {host}")
            except Exception as e:
                print(f"❌ Failed to connect to MySQL: {e}")
                self.direct_conn = None # Ensure it's None on failure

    def get_experiments(self) -> pd.DataFrame:
        try:
            if self.experimenter:
                return self.experimenter.get_table()
            elif self.direct_conn and self.table_name:
                return pd.read_sql(f"SELECT * FROM {self.table_name}", self.direct_conn)
        except Exception as e:
            print(f"⚠️ Error fetching experiments: {e}")
        return pd.DataFrame()

    def execute_sql(self, query: str) -> pd.DataFrame:
        """Executes a raw SQL query and returns a DataFrame. Raises exception on error."""
        if not self.direct_conn:
            raise ConnectionError("No direct database connection available. Use query_dataframe for PyExperimenter configs.")
        
        # Basic read-only check (heuristic)
        forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
        if any(cmd in query.upper() for cmd in forbidden):
            raise PermissionError("Only SELECT queries are allowed for safety.")
        
        return pd.read_sql(query, self.direct_conn)

    def query_dataframe(self, query: str) -> pd.DataFrame:
        """Executes a pandas query on the experiments table."""
        df = self.get_experiments()
        if df.empty:
            return df
        try:
            return df.query(query)
        except Exception as e:
            return pd.DataFrame({"error": [str(e)]})

    def restart_experiments(self, *statuses):
        """Restarts experiments based on status using PyExperimenter's reset_experiments."""
        if not self.experimenter:
            return "Error: Restarting runs requires a PyExperimenter configuration file. This feature is not available in direct DB mode."
        try:
            self.experimenter.reset_experiments(*statuses)
            return f"Successfully reset experiments with status: {statuses}"
        except Exception as e:
            return f"Failed to reset experiments: {str(e)}"

    def get_table_info(self):
        """Returns a dictionary mapping table names to their column metadata: {name: {col: {example, type}}}."""
        tables_to_query = [self.table_name or "experiments"] + self.logtables
        all_tables_info = {}
        
        for table in tables_to_query:
            if not table: continue
            columns_info = {}
            try:
                # Try to fetch schema by reading 1 row
                query = f"SELECT * FROM {table} LIMIT 1"
                if self.experimenter:
                    # PyExperimenter doesn't have a direct 'get logtable' by name easily without internal access
                    # but we can try to use its connection or just rely on direct_conn if we initialized it
                    # Actually PyExperimenter's connection is internal. 
                    # For now, let's use pd.read_sql if we have direct_conn or find another way.
                    # If we have self.experimenter, we can try to get the table if it's the main one
                    if table == self.table_name or table == "experiments":
                        df = self.get_experiments()
                    else:
                        # For logtables in PyExperimenter, we might need to use its database connection
                        # PyExperimenter stores it in self.experimenter.database_handler.connection
                        try:
                            df = pd.read_sql(f"SELECT * FROM {table} LIMIT 1", self.experimenter.database_handler.connection)
                        except:
                            df = pd.DataFrame()
                elif self.direct_conn:
                    df = pd.read_sql(f"SELECT * FROM {table} LIMIT 1", self.direct_conn)
                else:
                    df = pd.DataFrame()

                if not df.empty:
                    row = df.iloc[0]
                    columns_info = {
                        col: {
                            "example": str(row[col])[:50],
                            "type": str(df[col].dtype)
                        } for col in df.columns
                    }
                else:
                    # Fallback for empty tables
                    if self.direct_conn and self.provider == 'sqlite':
                        cursor = self.direct_conn.execute(f"PRAGMA table_info({table})")
                        columns_info = {row[1]: {"example": "No data", "type": "unknown"} for row in cursor.fetchall()}
                    elif self.experimenter and hasattr(self.experimenter, 'database_handler'):
                        # Try to get columns for PyExperimenter
                        conn = self.experimenter.database_handler.connection
                        if self.experimenter.database_handler.db_type == 'sqlite':
                            cursor = conn.execute(f"PRAGMA table_info({table})")
                            columns_info = {row[1]: {"example": "No data", "type": "unknown"} for row in cursor.fetchall()}
                        else:
                            # MySQL fallback
                            columns_info = {"status": {"example": "unknown", "type": "unknown"}}
                
                if columns_info:
                    all_tables_info[table] = columns_info
                    
            except Exception as e:
                print(f"⚠️ Error fetching info for table {table}: {e}")
                
        return all_tables_info
