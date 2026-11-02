#!/bin/sh
set -eu
cd "$(dirname "$0")"
runtime_python="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
if [ -d .packages ] && [ -x "$runtime_python" ]; then
  export PYTHONPATH="$PWD/.packages:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
else
  runtime_python=python3
  export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
fi
case "${1:-}" in
  --layout-evaluate) shift; exec "$runtime_python" tools/evaluate_layouts.py "$@" ;;
  --drive-evaluate) shift; exec "$runtime_python" tools/evaluate_drive.py "$@" ;;
  --gripper-evaluate) shift; exec "$runtime_python" tools/evaluate_gripper.py "$@" ;;
  --evaluate) shift; exec "$runtime_python" evaluate.py "$@" ;;
  --llm-evaluate) shift; exec "$runtime_python" tools/evaluate_language.py "$@" ;;
  --test) shift; exec "$runtime_python" -m unittest discover -s tests -v "$@" ;;
  *) exec "$runtime_python" run.py "$@" ;;
esac
