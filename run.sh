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
  --evaluate) shift; exec "$runtime_python" evaluate.py "$@" ;;
  --test) shift; exec "$runtime_python" -m unittest discover -s tests -v "$@" ;;
  *) exec "$runtime_python" run.py "$@" ;;
esac
