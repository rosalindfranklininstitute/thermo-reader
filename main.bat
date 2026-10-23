rem SPDX-FileCopyrightText: 2026 RFI
rem
rem SPDX-License-Identifier: Apache-2.0
del main.py.lprof
REM uv run kernprof main.py -- -c config.toml
uv run kernprof main.py -- -c "C:\Workspace\data\Away day\20260904_MouseTesticle__Y200um_20ums\conf.toml"
uv run python -m line_profiler -rmzt main.py.lprof > main.py.lines
echo Done. Read results in main.py.lines
