# Agent Rules (GFM Safety)

Hard rules:
1) Never guess node/text alignment. If mapping is not confirmed, mark as BLOCKED or EXPERIMENTAL.
2) Never invent metrics. All numbers must be extracted from logs under logs/**.log.
3) Any code change must include:
   - diff summary
   - one smoke command
   - expected key output lines
4) Keep patches minimal. Preserve existing behavior when new flags are not provided.
5) Prefer reproducibility: log every run and record command lines.
