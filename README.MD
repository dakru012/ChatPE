# Chat-PE: Machine Learning Research Assistant

Chat-PE (Chat PyExperimenter) is a chat-based assistant designed to analyze machine learning experiment data through natural language. It can automatically query your databases, generate plots, and synthesize findings using structured reasoning.

## Installation

To install Chat-PE in development mode (recommended), follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dakru012/ChatPE
   cd Chat_PE
   ```

2. **Install the package**:
   ```bash
   pip install -e .
   ```
   This will install the `chat-pe` command-line tool.

## Usage Guide

You can launch the agent's web interface using the `chat-pe` command. There are three main ways to connect to your data:

### 1. Connecting with a PyExperimenter Config
If you already use [PyExperimenter](https://github.com/lukasbeierle/py-experimenter), you can point the agent to your configuration file:
```bash
chat-pe --config path/to/config.yml
```

### 2. Connecting to a SQLite Database File
To connect directly to a local SQLite database without a PyExperimenter config:
```bash
chat-pe --db path/to/results.db --table my_experiments_table
```

### 3. Connecting to a MySQL/MariaDB Server
To connect to a remote MySQL database:
```bash
chat-pe --db_provider mysql --db database_name --db_host localhost --db_user root --db_password your_password --table experiments_table
```

### Optional Arguments
- `--arch`: Choose the agent's reasoning architecture: `react` (default) or `plan_execute`.
- `--port`: Specify the port for the web interface (default: `5000`).

---

## Evaluation and Benchmarking

Chat-PE includes a benchmarking suite to test the agent's performance on specific datasets and questions.

### Running the Benchmark
The benchmark script supports the same connection arguments as the main launcher:

**Example: Benchmark on a local SQLite file**
```bash
python evaluation/benchmark.py --db results.db --table experiments --questions evaluation/questions/my_questions.json --arch react
```

**Example: Benchmark on a MySQL server**
```bash
python evaluation/benchmark.py --db_provider mysql --db mysql_db --db_user root --table experiments --questions evaluation/questions/my_questions.json --arch plan_execute
```

### Question File Format
The benchmark expects a JSON file structured as follows:
```json
{
  "dataset": "my_dataset_name",
  "questions": [
    {
      "id": "L1_01",
      "question": "How many experiments were executed in total?"
    }
  ]
}
```

### Results
Benchmark results, including reasoning traces and generated plots, are saved to the `evaluation/results/` directory, organized by dataset and timestamp.
