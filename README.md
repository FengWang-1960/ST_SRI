# ST-SRI BSPC revision artifact

Scientific evidence for manuscript BSPC-D-26-13810 is provided in the fixed [closeout-v4p1-b16 release](https://github.com/FengWang-1960/ST_SRI/releases/tag/closeout-v4p1-b16-20261010) (10 October 2026).

Download `BSPC_Revision_closeout_v4p1_b16_20261010.zip` and `SHA256SUMS.txt` from that release, verify the archive identity, extract it, and begin with its top-level `README.md`. The archive contains code, frozen configurations, saved scientific outputs, numerical verification scripts, source/input/model identities, historical execution logs and an executable S01 example with a released checkpoint.

The source files shown in this repository are a byte-identical browsing snapshot of selected verifier and S01 Python files. Download the complete release ZIP for the tested execution entry points, dependencies, weights and saved-result inputs.

Archive identity:

- Size: 115,174,077 bytes
- SHA-256: `77f368bf779472d00165c69494dd95965d54ff5d3879b1f683283d49f705ba5f`
- Internal manifest: `MANIFEST_SHA256.csv`, 6,046 entries; manifest excluded from its own list

The active saved-result verifier is `verifier_v4p1/verify_saved_results.py`. Independent-directory macOS portable and strict runs each passed 27,396 checks over 3,507 scientific sources. The preserved Linux portable pass and strict failure remain explicitly documented. The 39 numerical fault tests and 20 provenance tests passed. These results concern saved-result verification, not all model inference or training.

Three author-local S01 scans used a licensed raw recording and matched the archived arrays and identities. A later independent review established saved-evidence consistency but did not repeat raw-signal inference. Raw NinaPro recordings are not distributed; use the [official DB2 access route](https://ninapro.hevs.ch/instructions/DB2.html) to run the example.

Historical sources remain distinct:

- [Zenodo code DOI](https://doi.org/10.5281/zenodo.22907764): the previous two code ZIPs, not the October revision evidence.
- [v2026.09.18 checkpoints/results release](https://github.com/Caiheng-Yu/ST-SRI-BSPC-artifact/releases/tag/v2026.09.18): other replacement checkpoints are linked by pinned asset/member identities in `external_assets/REPLACEMENT_CHECKPOINTS_FIXED_HASHES.json`.

The evidence ZIP is byte-frozen. Its `documents/` directory contains historical source snapshots rather than current submission documents. Internal access fields retain the archive-freeze state; this release page and external access receipts record later publication. Journal upload, reviewer notification and a new DOI are separate from this public GitHub release.
