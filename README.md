# epidemIA

Multi-agent AI system for epidemiological simulation and data analysis.

## Requirements

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama](https://ollama.com/)

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/MatteoFalcioni/epidemIA.git
cd epidemIA
uv sync
```

`uv sync` creates and manages the project virtual environment in `.venv`.

### 2. Download the default model

Install Ollama for your operating system from [ollama.com](https://ollama.com/), then download the default model:

```bash
ollama pull qwen3.8:27b
```

The default model needs about 18 GB of storage. See the [Ollama model page](https://ollama.com/library/qwen3.8) for available model variants.

### 3. Configure the model (optional)

The application uses `qwen3.8:27b` for both agents by default. To use other installed models, copy the environment template:

```bash
cp .env.template .env
```

Then set the model names:

```bash
SUPERVISOR_MODEL=<model-name>
ANALYST_MODEL=<model-name>
```

### 4. Run the command-line application

```bash
uv run python src/main.py
```

Enter `/exit` to close the application.

### 5. Run the web interface

Run this command from the `src` directory:

```bash
cd src
uv run python epidemia_gui.py
```

The web interface starts on port 2223.

## Development

Synchronize dependencies after a change to `pyproject.toml`:

```bash
uv sync
```

Check that the lock file is current:

```bash
uv lock --check
```
