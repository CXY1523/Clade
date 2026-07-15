# Phase 2C-1 Security Verification

**Verification date:** 2026-07-15 (Asia/Shanghai)

**Branch:** `phase-2c-outbound-url-security`

**Design base:** `89eec7b`

**Implementation tip before this report:** `996f6e7`

**Verified implementation range:** `89eec7b..996f6e7`

**Full handoff range after the documentation commit:** `89eec7b..HEAD`

## Outcome

Phase 2C-1 configuration-probe and persistent local-AI-control gates pass, with one documented correction to a stale verification command. The planned permanent fresh-process test file does not exist in this repository or its history; the previously approved canonical fresh-process inline guard was run instead and passed. No production code or tests were changed for Task 7.

This is not a Phase 2C completion or release approval.

```text
Phase 2C-1 protects configuration probes and the persistent local-AI control.
Phase 2C is not complete: actual AI, load-balancing, streaming, and embedding requests remain for Phase 2C-2.
This branch must not be merged or released before Phase 2C-2 and final review.
```

## Backend gates

All commands used the existing Python environment at `..\phase-2b2-save-path-boundary\backend\.venv\Scripts\python.exe`; no dependency installation or update was performed.

| Command | Exit | Evidence |
| --- | ---: | --- |
| `& $PYTHON -m pytest backend/app --collect-only -q` | 0 | 666 tests collected |
| `& $PYTHON -m pytest backend/app -q` | 0 | 664 passed, 2 skipped, 45 warnings; 0 failures and 0 collection errors |
| `& $PYTHON -m pytest backend/app/core/tests/test_plugin_registry_fresh_process.py -q` | 1 | Initial planned command ran 0 tests because the referenced file is absent |
| Canonical fresh-process inline plugin guard from the backend directory | 0 | Loaded and registered exactly 6 built-in plugins |

The two skips are the existing Windows symlink skips. The 45 warnings equal the allowed ceiling and do not exceed it.

### Fresh-process guard substitution

The Phase 2C-1 plan names `backend/app/core/tests/test_plugin_registry_fresh_process.py`, but that file is absent from `HEAD`, all searched repository history, branches and local worktrees. This was a stale plan reference, not an assertion failure or implementation regression. The first failure was retained above and was not hidden by a rerun.

The canonical guard comes from `docs/superpowers/plans/2026-07-14-backend-baseline-recovery.md`, which explicitly specifies an inline new-process import and says that no additional permanent test is added. It was run after changing to `backend` so `app` had the same import context as the approved baseline:

```powershell
$Python = (Resolve-Path "..\phase-2b2-save-path-boundary\backend\.venv\Scripts\python.exe").Path
Push-Location backend
& $Python -c "from app.services.embedding_plugins import PluginRegistry, load_all_plugins; expected={'ancestry','behavior_strategy','evolution_space','food_web','prompt_optimizer','tile_biome'}; loaded=set(load_all_plugins()); registered=set(PluginRegistry.list_plugins()); assert loaded == expected; assert registered == expected; print(sorted(registered))"
Pop-Location
```

It exited 0 and printed the six expected plugin names. This substitutes the same fresh-process registration invariant without adding or modifying tests.

## Frontend gates

Commands ran from `frontend` using the already linked dependencies.

| Command | Exit | Evidence |
| --- | ---: | --- |
| `npm run test:run` | 0 | 12 test files passed; 84 tests passed; 0 failed |
| `npm run lint` | 0 | 0 errors, 162 warnings |
| `npx tsc --noEmit` | 0 | TypeScript check completed with no output |
| `npm run build` | 0 | TypeScript and Vite production build completed; 4,399 modules transformed |

The lint warning count equals the allowed ceiling and does not exceed it. The build emitted the existing informational warning that one settings module is both dynamically and statically imported; it did not fail the build.

## Repository hygiene and redaction scan

| Command | Exit | Evidence |
| --- | ---: | --- |
| `git diff --check` | 0 | No whitespace errors |
| `rg -n "admin-secret\|sk-unsaved\|resolver sentinel\|upstream echoed" backend/app frontend/src docs/api-guides -g '!**/test_*.py' -g '!**/*.test.ts' -g '!**/*.test.tsx'` | 1 | No production or API-guide matches; exit 1 is ripgrep's no-match result |
| `git status --short` before report creation | 0 | Only the two intended API-guide files were modified |

No API key, administrator token, sensitive query URL, upstream response body or raw exception text is reproduced in this report.

## Commit-range review

Before the Task 7 documentation commit, `git log --oneline 89eec7b..996f6e7` contained 11 physical commits: one Phase 2C-1 plan commit, six Task 1–6 main commits and four retained review-hardening commits. After Task 7 there are seven task checkpoints, but the full physical range contains the additional plan and review commits; the history is reported as-is rather than rounded to seven commits.

The range review found no changes to:

- `backend/app/ai/model_router.py`
- `backend/app/services/system/embedding.py`

The implementation stays within configuration probes and configuration control. It does not broaden the loopback allowlist, follow redirects, use environment proxies, perform an unrestricted second DNS lookup, read an unbounded model-list body, store the administrator token, or change runtime AI/Embedding request paths.

## Remaining review note

The Task 4 reviewer recorded one non-blocking test-hardening note: two route-level redaction tests assert that an upstream-body marker is absent without injecting that marker. The concrete safe-client test does inject a body marker and verifies zero non-200 body reads, so production behavior is covered. Task 7 intentionally did not modify tests; the route-level assertions can be strengthened during final review if desired.

## Handoff

Proceed to a separate Phase 2C-2 design review. Keep this branch isolated and do not merge or release it at this checkpoint.
