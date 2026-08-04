#!/usr/bin/env bash
# Every mechanical check, in one command.
#
#   ruff     — formatting and import order. Notebooks are linted but not
#              formatted; `[tool.ruff.format]` in pyproject.toml says why.
#   lanorme  — code quality, Agent Skills spec compliance, and the harness's own
#              plugins (tensors, skill_portability, provenance). PYTHONPATH=. is
#              what lets lanorme import them from lanorme_plugins/.
#   pytest   — the checks' own tests, including a false-positive suite. A checker
#              that cries wolf gets bypassed, so quiet-on-clean-input is tested
#              as carefully as fires-on-bad-input.
#   check_local_paths.py — no machine-local absolute path in committed code,
#              evidence or notebooks. A direct stage rather than a lanorme
#              plugin: lanorme's config excludes results/ and notebooks/
#              globally, which would filter the findings away.
#   validate_research.py — the research-integrity gate. Deliberately standalone
#              and dependency-free: a project that never installs lanorme must
#              still have every integrity guarantee. Skipped in the harness repo
#              itself, which is not a research project and has no TREE.md.
#
# EVERY stage runs, every time, and the failures are summarised at the end.
# Two ways this script used to lie about the repository, both found by a
# cold-clone audit on 2026-07-30:
#
#   1. `ruff format --check . && ruff check ...` discarded the formatter's
#      verdict, because `set -e` ignores a failure in any element of an AND-list
#      except the last. It hid 18 unformatted files and 12 lint errors.
#   2. `set -e` aborted the whole script at the first failing stage. lanorme has
#      long-standing violations, so pytest and the research-integrity gate never
#      ran at all — while the README advertised that this one command runs them.
#
# A gate that stops at the first problem cannot tell you whether the others are
# fine, which is the only question worth asking before a commit.
set -uo pipefail
cd "$(dirname "$0")"

# Pinned, because `uvx ruff` and `uvx lanorme` resolve to whatever is newest at
# the moment you run them: the gate's verdict could change with no commit to
# this repository, and ruff is a tool that REWRITES files. Bump these
# deliberately, and run ./check.sh in the same commit that bumps them.
RUFF="ruff@0.16.1"
LANORME="lanorme@0.16.0"
# `uvx` takes tool@version; `uv run --with` takes a PEP 508 requirement.
LANORME_REQ="lanorme==0.16.0"

FAILED_STAGES=""

# Run a stage under its own name, record the failure, and keep going.
run_stage() {
    local label="$1"
    shift
    printf '\n=== %s ===\n' "$label"
    if "$@"; then
        printf -- '--- %s: ok\n' "$label"
    else
        printf -- '--- %s: FAILED (exit %d)\n' "$label" "$?"
        FAILED_STAGES="${FAILED_STAGES}${label}"$'\n'
    fi
}

run_stage "ruff format" uvx "$RUFF" format --check .
run_stage "ruff imports" uvx "$RUFF" check --select I .
run_stage "lanorme" env PYTHONPATH=. uvx "$LANORME" check "${1:-.}"

if [[ -d tests ]]; then
    run_stage "pytest" uv run --with pytest --with "$LANORME_REQ" pytest tests -q
fi

run_stage "local paths" uv run scripts/check_local_paths.py

# The write-up's charts are plain JS inlined into docs/index.html. A syntax
# error there does not fail the build: the page is written, every figure
# silently renders empty, and nothing else here notices. That happened on
# 2026-08-03, from an edit that left a dangling `+`. Skipped rather than
# required, so a clone without node still passes the rest.
if [[ -f docs/charts.js ]]; then
    if command -v node >/dev/null 2>&1; then
        run_stage "docs js syntax" node --check docs/charts.js
    else
        printf '\n=== docs js syntax ===\nnode not installed, skipping\n'
    fi
fi

if [[ -f TREE.md ]]; then
    run_stage "research integrity" uv run scripts/validate_research.py
else
    printf '\nNo TREE.md — skipping research-integrity gate (not a research project).\n'
fi

printf '\n========================================\n'
if [[ -z "$FAILED_STAGES" ]]; then
    printf 'All stages passed.\n'
    exit 0
fi
printf 'FAILED stages:\n'
printf '%s' "$FAILED_STAGES" | sed 's/^/  - /'
exit 1
