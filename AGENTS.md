# Video Taxonomy

Code lives in src/video_taxonomy. This is a code-only repository: real taxonomy definitions, business constraints and private source documents must stay outside Git. Supply configuration via --config or VIDEO_TAXONOMY_CONFIG. Read SKILL.md and docs/usage.md before changes.

For execution, first locate the caller's external configuration, metadata and (unless using a prepared lookup) local model weights. Run `validate-config` before encoding. Do not invent missing production inputs. Synthetic tests only prove the software flow, not semantic model accuracy. Never commit run outputs or private configuration. Before publishing, check the Git author/committer identity; do not expose a personal email or machine hostname.

Preserve scheme 3: L1 plus all L1/L2 paths plus all L1/terminal paths, with ` / ` separators and one center per node. Use full 1024-dimensional embeddings and L1-majority voting; preserve direct L1/L2 endpoints instead of forcing leaves. Keep hierarchical rollups consistent, tags-only evidence as the default, and content separate from production form. Run `python -m unittest discover -s tests -v`; use only generated fixtures. Never restore or copy real category data into this repository, including as JSON, Markdown, tests, caches or build artifacts.
