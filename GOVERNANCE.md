# Governance

This document describes how the repository is maintained and how decisions are made.  
Repository: https://github.com/cesaragostino/DOFT-Delayed-Oscillator-Field-Theory/

## 1) Roles

**Maintainers**
- **Cesar Agostino** (GitHub: [@cesaragostino](https://github.com/cesaragostino)) — Repo Owner

> Contact: open an Issue for general matters. For sensitive topics (e.g., security), use GitHub “Private vulnerability reporting” (Security tab) to contact the owner privately.

## 2) Decision-making

- **Default**: *lazy consensus*. If a change receives ≥1 maintainer approval, CI is green, and no reasoned objections are raised within **72 hours**, it may be merged.
- **Major changes** (public API, license, repo transfer, long-term roadmap): require **2 approvals** if there are ≥2 maintainers, or explicit sign-off by the Repo Owner.
- **Tie-breaker**: Repo Owner decides.

## 3) Branching model

- Active development occurs on **`development`**. When a milestone is complete and tests are green, changes are merged into **`main`**.  
- Feature work should use topic branches: `feat/<short-name>` or `fix/<short-name>`.  
- `main` must remain releasable.

## 4) Releases & versioning

- **SemVer**: `MAJOR.MINOR.PATCH`.
- Release steps:
  1. Update `CHANGELOG.md`.
  2. Bump version.
  3. Tag `vX.Y.Z` and create a GitHub Release (attach artifacts if any).
- Backports only for critical fixes; no formal LTS unless later announced.

## 5) Reviews, CI, and quality

- Every PR must pass CI and, when practical, include tests or validation notebooks/plots.  
- Small, focused PRs are preferred. Maintainers may request changes or push “suggested edits” to contributor branches (when allowed by GitHub).

## 6) AI-assisted workflow (OpenAI & Gemini)

This project uses AI agents to accelerate research and development. Human review is **mandatory** before merge.

- **Agents and roles**
  - **OpenAI GPT-5 Thinking**: physics/mathematics discussions; QA of implementations.
  - **Gemini Pro**: simulation design/evaluation; development assistance.
  - **OpenAI GPT-5 Thinking** (additional): development QA.

- **Provenance requirement**
  - If AI assistance influenced a PR, include a brief note in the PR description:
    - *“AI assistance: Gemini Pro (design), GPT-5 Thinking (QA). Prompts and key diffs attached in the PR description or linked artifact.”*
  - Cite model names/versions and summarize what was generated vs. what was human-written.

- **Safety & licensing**
  - Do **not** paste secrets or private data into external tools.
  - Ensure third-party code suggested by AI respects the project license and attribution.
  - All AI-generated code/text is reviewed and edited by a maintainer before merge.

## 7) Security

- Report vulnerabilities via GitHub **Private vulnerability reporting** (Security → “Report a vulnerability”).  
- Coordinated disclosure: fixes are prepared privately; public disclosure after a patched release unless risk dictates otherwise.

## 8) Maintainer changes

- **Adding**: after ~3 meaningful contributions over ~6 months (guideline), propose via PR updating this file; requires Repo Owner approval (or 2 approvals if ≥2 maintainers).
- **Emeritus/Removal**: inactivity ~6 months, request by the maintainer, or for cause (owner decision).

## 9) Amending this document

- Propose changes via PR. Minor edits (typos/links) may be merged by the Repo Owner.  
- Substantive changes follow the same rule as “Major changes” in §2.

---

_Last updated: 2025-08-26_
