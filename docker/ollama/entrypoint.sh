#!/bin/sh
set -eu

ollama serve &
server_pid=$!

cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

until ollama list >/dev/null 2>&1; do
  sleep 1
done

ollama pull "${OLLAMA_MODEL:-qwen3-vl:4b-instruct}"
wait "$server_pid"
