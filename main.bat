del main.py.lprof
uv run kernprof main.py -- -c config.toml
uv run python -m line_profiler -rmzt main.py.lprof > main.py.lines
echo Done. Read results in main.py.lines
