# Video Taxonomy

Code lives in src/video_taxonomy. This is a code-only repository: real taxonomy definitions, business constraints and private source documents must stay outside Git. Supply configuration via --config or VIDEO_TAXONOMY_CONFIG. Read SKILL.md and docs/usage.md before changes.

Preserve full-path leaf matching with ` / `, hierarchical voting and consistent rollups. Keep content and production form separate. Run `python -m unittest discover -s tests -v`; use only generated fixtures. Never restore or copy real category data into this repository, including as JSON, Markdown, tests, caches or build artifacts.
