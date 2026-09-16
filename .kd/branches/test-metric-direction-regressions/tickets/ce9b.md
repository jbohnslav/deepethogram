---
id: "ce9b"
status: closed
deps: []
links: []
created: 2026-09-16T13:37:59Z
type: task
priority: 2
closed_at: 2026-09-16T13:48:07Z
resolution: completed
closed_context: codex:335ea742ef847594
assignee: codex:335ea742ef847594
---
# Correct early stopping direction for classification metrics

Audit 5540: get_stopper hardcodes is_error=True while StopperCallback supplies latest_key, which is F1 in classification. Reproduced using actual Stopper/EarlyStopping class definitions: F1 0.6 -> 0.5 marked best, 0.7 marked non-improvement. Default stopping_type=learning_rate avoids this. Initialize the missing counter in EarlyStopping and derive is_error from the active Metrics.key_metric_mode in StopperCallback.

## Acceptance Criteria

- [x] Early stopping uses the selected metric direction and handles the first non-improvement safely
- [x] Regression tests cover increasing F1 and decreasing loss

## Worklog

- [2026-09-16 09:48] [codex:335ea742] — Fixed EarlyStopping counter initialization and wired existing is_error flag to active Metrics.key_metric_mode through StopperCallback. Both F1 variants continue improving, loss/SSIM minimize, first worsening score no longer crashes. Full default suite: 81 passed, 5 GPU deselected; lint passes.

## Lifecycle

- 2026-09-16T13:48:07Z [codex:335ea742ef847594] — closed (completed)
