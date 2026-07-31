"""A git-free environment for tests that shell out to git.

Git exports GIT_DIR, GIT_INDEX_FILE, GIT_WORK_TREE and GIT_PREFIX to every
process it spawns, and those beat `cwd=`. These tests run under check.sh, which
runs under the pre-commit hook, so a fixture that runs `git init` in a tmp
directory without clearing them retargets the REAL repository instead.

That is not hypothetical. On 2026-07-31 it set `core.bare = true` on this repo
and overwrote user.name and user.email with the fixtures' placeholders, and
`git status` stopped working until the config was repaired by hand. Any test
that spawns git must pass this environment.
"""

from __future__ import annotations

import os


def git_free_env() -> dict[str, str]:
    """The ambient environment with every GIT_* variable removed."""
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
