---
created: 2026-03-04T09:11:09.794Z
title: Add collapsible right-side analysis settings panel
area: ui
files:
  - app.py
  - src/agent.py
  - src/agent_tools.py
  - src/statistics/analyzer.py
---

## Problem

The current workflow does not provide a dedicated settings UI for analysis defaults. Users must repeatedly specify statistical thresholds and column mappings in chat, and there is no discoverable place to configure advanced metrics options. This slows iteration and increases misconfiguration risk when switching datasets.

Requested UX behavior is a click-to-open settings surface on the right side that can be closed, with controls for p-value, statistical power, default column mappings, and additional metrics/model options.

## Solution

Implement a collapsible right-side settings panel in the UI that opens from a settings trigger and closes via explicit close action (and optional outside click). Include:

1. Statistical controls:
- Significance level (p-value / alpha)
- Statistical power threshold

2. Default column mappings:
- Group
- Effect value
- Pre effect
- Post effect
- Segment
- Customer ID
- Duration

3. Advanced metrics/model options:
- Metric type and model strategy defaults
- Multiple-testing behavior visibility
- Sequential testing controls (if enabled)

4. Wiring and behavior:
- Persist settings per session
- Apply settings to analyzer defaults/tool execution paths
- Reset to sensible defaults
- Surface active settings summary in analysis responses

5. Validation:
- Add tests for settings persistence and config propagation
- Verify panel open/close behavior and no regression in existing analysis flow
