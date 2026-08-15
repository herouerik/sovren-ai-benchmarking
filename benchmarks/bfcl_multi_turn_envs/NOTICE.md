# Vendored source

The Python files in this directory (`gorilla_file_system.py`, `math_api.py`,
`message_api.py`, `posting_api.py`, `ticket_api.py`, `trading_bot.py`,
`travel_booking.py`, `vehicle_control.py`, `long_context.py`) are vendored
from the [Gorilla project](https://github.com/ShishirPatil/gorilla)
(`berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval/func_source_code/`),
licensed under Apache License 2.0. Only the internal cross-import in four
files was changed (absolute `bfcl_eval.eval_checker...` import path rewritten
to a relative `.long_context` import) so the modules work standalone here.

Vendored 2026-08-15 instead of depending on the `bfcl-eval` PyPI package,
which pulls in sglang/vllm/cuda-bindings (multiple GB) as hard dependencies
just to reach these otherwise-lightweight, dependency-free simulator classes
(only `mpmath`, used by `math_api.py`).

See the repository's `LICENSE` file at
https://github.com/ShishirPatil/gorilla/blob/main/LICENSE for the full
Apache 2.0 text.
