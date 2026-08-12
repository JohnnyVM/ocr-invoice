# Ollama vision server

Build and start the local Ollama server from the repository root:

```bash
docker compose -f docker/ollama/compose.yaml up --build
```

The container listens on `http://127.0.0.1:11434` and pulls `qwen3-vl:4b-instruct` on first startup. The model is stored in the named `ollama_models` volume.
