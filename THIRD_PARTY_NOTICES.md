# Third-Party Notices

This repository is licensed under the Apache License 2.0 (see LICENSE); that
license covers the original contributions here. Parts of the code derive from
the upstream projects below and remain under their terms. The
`third_party/vllm` submodule is a fork of
[vLLM](https://github.com/vllm-project/vllm) and keeps vLLM's Apache
License 2.0, recorded in the submodule itself.

Provenance in brief: this harness was built from
[SeLaReasoning](https://github.com/Parker-rfu/SeLaReasoning)'s codebase, which
its authors state "builds upon" [SwiReasoning](https://github.com/sdc17/SwiReasoning).
Most of what survives here traces through SeLaReasoning back to SwiReasoning's
BSD-3-Clause-licensed originals; the `selar` method itself is SeLaReasoning's
own, and their repository publishes no license.

## SwiReasoning — BSD 3-Clause

The batched decoding-loop structure in `src/generation_utils.py`, the `swir`
method, `src/hybrid_cache_compat.py`, and the originals behind `src/grader.py`
and `scripts/merge.py` come from
[SwiReasoning](https://github.com/sdc17/SwiReasoning).

```
BSD 3-Clause License

Copyright (c) 2025, Dachuan Shi

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Soft-Thinking — MIT

The `soft` method (`generate_soft` in `src/generation_utils.py`, and the
engine-side implementation in the vLLM fork) follows the reference
implementation of [Soft-Thinking](https://github.com/UCSB-AI/Soft-Thinking)
([arXiv:2505.15778](https://arxiv.org/abs/2505.15778)).

```
MIT License

Copyright (c) 2025 UC ERIC Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## SeLaReasoning — no published license

The `selar` method (`generate_selar` in `src/generation_utils.py`) and the
trimmed forms of `src/grader.py` and `scripts/merge.py` come from
[SeLaReasoning](https://github.com/Parker-rfu/SeLaReasoning)
([arXiv:2604.08299](https://arxiv.org/abs/2604.08299)), which at the time of
writing publishes no license. Those portions are reproduced here for academic
research, with full credit to the SeLaR authors; all rights in them remain
with those authors, and this repository's Apache-2.0 grant does not extend to
them. If you are a SeLaR author and would like anything here changed or
removed, please open an issue.
