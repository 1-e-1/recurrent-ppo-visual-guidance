# Recurrent PPO Visual Guidance

Recurrent RGB-D policy and guarded online PPO integration for dual-arm visual guidance.

Primary method: visual_guidance_lab/. Classical geometry and task-state-machine dependencies remain explicit. Historical pilot failures and mixed results are retained in the original notes.

## Setup

Use a dedicated Python environment and install a compatible PyTorch/torchvision pair for your platform, then:

```sh
python -m pip install -r requirements.txt
```

Run from the repository root. See [method scope](METHOD_SCOPE.md), [publication notes](PUBLICATION_NOTES.md) and the retained original README for entry points and historical status.

## Distribution limits

This release contains source, selected configuration, original tests and research documentation. It does not bundle datasets, pretrained weights, robot scenes or third-party simulation libraries. Historical success rates and paper figures are not new measurements from this release. Simulator and hardware claims retain their original limitations. Do not launch full training queues just to test an installation.

## License

Original material is licensed under [MIT](LICENSE). Third-party libraries, assets and datasets retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).

Selected USD/QP integration tests were run using the pre-existing IsaacLab Python environment. Basic NumPy/PyTorch-only Python installations do not provide pxr (USD); use your original compatible Isaac environment for those tests.

One retained scene-generation test requires original astronaut USD sublayers and failed in this source-only export. The other 12 selected checks passed; see PUBLICATION_CHECKS.md.
