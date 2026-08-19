# AI navigation quickstart

Use this file as a lightweight navigation guide for AI agents. Do not duplicate the general project README.

- First: read `graphify-out/GRAPH_REPORT.md` and use `graphify query`, `graphify path`, and `graphify explain` before raw file browsing.
- If `graphify-out/wiki/index.md` exists, use that as the wider map before drilling into source files.
- Keep the first read targeted: a graph summary, then the exact file, then the immediate dependency/context file.

## High-value entry points
- Backend app boot: `backend/app/main.py`
- Analysis entry: `backend/app/analysis_router.py`
- Report generation: `backend/app/report/generator.py`
- Demographic extraction: `backend/app/nlp/demographic_extraction.py`
- AI attribute extraction: `backend/app/nlp/ai_attribute_extraction.py`
- Scoring / privacy-risk logic: `backend/app/scoring/k_anonymity.py`, `backend/app/scoring/privacy_score.py`
- Spanish reference tables: `backend/app/data/ine_reference.py`
- Frontend analysis screen: `webapp/src/pages/Dashboard.tsx`
- Frontend landing/auth flow: `webapp/src/pages/Landing.tsx`
- Frontend API contract: `webapp/src/api.ts`
- Shared report schemas: `backend/app/models/schemas.py`

## Minimal workflow for fixes
- One targeted graph search or lookup.
- Read the exact file and the immediate dependency/context file.
- Patch the smallest relevant scope.
- Validate with the smallest existing test/build command for the changed behavior.

