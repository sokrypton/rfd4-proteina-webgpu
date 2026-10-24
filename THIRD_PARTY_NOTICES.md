# Third-party notices

This site redistributes the following third-party software and data.

## AtomWorks Studio

`assets/` (except Mol*, below) is the released browser bundle of
[AtomWorks Studio](https://github.com/RosettaCommons/atomworks-studio), downloaded unmodified at the version pinned in
`studio.json` and composed into `index.html` by `tools/build-studio.mjs`. This repository carries no AtomWorks Studio
source; earlier revisions vendored it under `studio-src/` and patched it locally. `web/studio/`, `guide/`,
`annotation-studio.svg`, `atomworks_logo_dark.svg` and `manifest.webmanifest` also come from AtomWorks Studio.

BSD 3-Clause License

Copyright (c) 2026, Institute for Protein Design, University of Washington

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

Third-party dependencies and the Patinae patch series retain their respective
licenses. See `docs/PROVENANCE.md` and `vendor/patinae/UPSTREAM.toml`.

## Mol*

`assets/molstar-*.js` and `assets/molstar-*.css` bundle [Mol*](https://github.com/molstar/molstar) (MIT License),
copyright (c) 2017 - now, Mol* contributors.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## py2Dmol

`web/vendor/py2dmol-ss.js` is extracted from [py2Dmol](https://github.com/sokrypton/py2Dmol)
(`src/cartoon/geom.js`, by `tools/vendor-py2dmol-ss.mjs`) under its Beer-Ware License (Revision 42).

## fzstd

`shared/rfd4p/vendor/fzstd.js` is [fzstd](https://github.com/101arrowz/fzstd) 0.1.1, unmodified, under the MIT License
(`shared/rfd4p/vendor/LICENSE-fzstd.txt`).

## RFD4-Proteina weights

`weights/int8/` holds a block-64 INT8 quantization (by `tools/export_rfd4p_model.py`) of the
[nvidia/RFD4-Proteina](https://huggingface.co/nvidia/RFD4-Proteina) checkpoint, which its model card releases under the
BSD-3-Clause license. See the model card for the authors' terms and citation.
