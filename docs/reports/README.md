# Test reports from other machines

Runs made by others than the project, on hardware, drivers or hosts that `LIMITS.md` lists as not
tried or that the project does not have. Each row is what the report says, with a link to it; the
project did not repeat it. How to send one: `CONTRIBUTING.md`, "Test reports from other machines".

| Date | VF | GPUs | Driver | Host | What was run | Result | Report |
|---|---|---|---|---|---|---|---|
| 2026-11-24 | 0.1.4 (`2c93595`) | 2 x V100-SXM2-16GB, NVLink NV6 | 580.196.07 | glibc 2.43 | `tests/gpu/run.sh 0`; `tests/gpu/peer-copy.c` | 165 of 165 PASS; peer copies 128.69 to 128.73 GB/s (0 to 1) and 130.71 to 130.72 GB/s (1 to 0), as without the layer | [issue #3](https://github.com/nbald/volta-forever/issues/3) |
| 2026-11-24 | 0.1.4 release | 2 x V100-SXM2-32GB, NVLink NV6 | 580.196.07 | Ubuntu 24.04.5, glibc 2.39 (the issue's first report) | Strata 0.2.20, CUDA 13 engine 0.1.66, layers split over both GPUs | first request answered with empty caches and the default watchdog; every prompt of 4,096 tokens or more ended the engine (`native_expert_i8/down: unspecified launch failure`, 8 of 8), not with `STRATA_EXPERT_I8=0` | [issue #3](https://github.com/nbald/volta-forever/issues/3) |
