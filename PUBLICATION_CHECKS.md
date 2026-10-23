# Publication checks

Date: 2026-10-24.

## root

Status: partial: original USD asset absent.

Test selection:

```text
tests/test_visual_guidance_contract.py
tests/test_visual_recurrent_ppo.py
tests/test_visual_task_progress.py
tests/test_visual_training_cycles.py
```

Observed: 1 failed, 12 passed, 1 warning.

Environment: pre-existing IsaacLab Python with USD and OSQP, CPU tests only.

One scene-generation test failed because original astronaut USD sublayers are not included. The other 12 selected tests passed. This remains an unresolved external-asset requirement; no mock scene or substituted geometry is presented as a passing integration test.

All distributed Python source passed static compilation. Selected text was checked for common credential/private-key formats. These checks do not rerun historical research training, simulator acceptance or physical hardware commissioning.
