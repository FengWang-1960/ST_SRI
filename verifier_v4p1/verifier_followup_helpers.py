"""Independent saved-score checks; no project imports and no output writes.

Public entry points accept an artifact root, check(name, passed, detail=None),
and a results dictionary. NumPy and SciPy are the only external dependencies.
The original inference is not rerun: these functions reconstruct statistics
from archived four-state scores, using the recorded float32 arithmetic order.
"""
from pathlib import Path
import csv
import json
import hashlib
import numpy as np
from scipy.ndimage import gaussian_filter1d


def _followup_dir(base, name):
    base = Path(base)
    archived = base / "evidence" / "followup" / name
    if not archived.is_dir():
        raise FileNotFoundError(archived)
    return archived


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _close(left, right):
    return bool(np.allclose(left, right, atol=1e-12, rtol=0, equal_nan=True))


def _array_equal(left, right):
    return bool(np.array_equal(left, right, equal_nan=True))


def verify_m3(base, check, results, recomputed=None, invariant_counts=None):
    """Recompute all 320 M3 cells and all 24 cell / 30 paired summaries.

    D is 0.5 ms times the sum of sigma=2-bin, reflect-smoothed positive
    fixed-target probability difference, cropped to the observed 5--145 ms
    common lag support before smoothing. The four-state order is q11 - q01
    - q10 + q00 in stored array precision. Two event values are averaged
    inside each participant before the 40-participant bootstrap; 10,000
    common paired draws use seed 20261004 and percentile 2.5/97.5 endpoints.
    """
    compare_recomputed = recomputed or (lambda name, left, right: _array_equal(left, right))
    folder = _followup_dir(base, "m3_phase")
    design = _json(folder / "frozen_design.json")
    saved_summary = _json(folder / "results_summary.json")
    cases = design["cases"]
    cell_csv = {row["id"]: row for row in _csv(folder / "all_cells.csv")}
    metrics = ["positive_mass_probability_ms", "max_height_probability",
               "signed_mass_probability_ms", "absolute_mass_probability_ms",
               "logit_signed_mass", "logit_positive_mass"]
    computed = {}
    metadata = {}
    anchor_argmax_changes = 0
    for case in cases:
        subject = int(case["subject"])
        event = case["anchor_window_id"].rsplit("_ph", 1)[0]
        for endpoint in (100, 150):
            for current in (0, 100):
                cid = f"{event}_read{endpoint}_current{current}"
                path = folder / "scans" / f"S{subject:02d}" / f"{cid}.npz"
                rec = _json(path.with_suffix(".json"))
                with np.load(path, allow_pickle=False) as npz:
                    arrays = {name: npz[name] for name in npz.files}
                prefix = "M3/" + cid
                check(prefix + "/score_shape", arrays["prob_scores"].shape == (4, 281))
                check(prefix + "/eligible_axis", _array_equal(arrays["lags_ms"], np.arange(5, 145.1, .5)))
                check(prefix + "/finite_scores", bool(np.isfinite(arrays["prob_scores"]).all()))
                check(prefix + "/score_probability_range", bool(((arrays["prob_scores"] >= 0) & (arrays["prob_scores"] <= 1)).all()))
                prob = arrays["prob_scores"][0] - arrays["prob_scores"][1] - arrays["prob_scores"][2] + arrays["prob_scores"][3]
                logit = arrays["logit_scores"][0] - arrays["logit_scores"][1] - arrays["logit_scores"][2] + arrays["logit_scores"][3]
                pos = gaussian_filter1d(np.maximum(prob, 0).astype(float), sigma=2, mode="reflect")
                log_pos = gaussian_filter1d(np.maximum(logit, 0).astype(float), sigma=2, mode="reflect")
                check(prefix + "/four_state_probability_identity", _array_equal(prob, arrays["prob_signed"]))
                check(prefix + "/four_state_logit_identity", _array_equal(logit, arrays["logit_signed"]))
                check(prefix + "/positive_smoothed_recomputed", compare_recomputed(prefix + "/positive_smoothed_recomputed", pos, arrays["positive_smoothed"]))
                check(prefix + "/strict_maximum_index", int(pos.argmax()) == int(arrays["positive_smoothed"].argmax()))
                if invariant_counts is not None:
                    invariant_counts['M3_maximum_indices'] += 1
                vals = {
                    "positive_mass_probability_ms": float(.5 * pos.sum()),
                    "max_height_probability": float(pos.max()),
                    "signed_mass_probability_ms": float(.5 * prob.astype(float).sum()),
                    "absolute_mass_probability_ms": float(.5 * np.abs(prob.astype(float)).sum()),
                    "logit_signed_mass": float(.5 * logit.astype(float).sum()),
                    "logit_positive_mass": float(.5 * log_pos.sum()),
                }
                check(prefix + "/all_cells_csv_identity", cid in cell_csv)
                for metric, value in vals.items():
                    check(prefix + "/json/" + metric, _close(value, rec[metric]))
                    check(prefix + "/csv/" + metric, _close(value, float(cell_csv[cid][metric])))
                raw_max_lag = float(arrays["lags_ms"][int(pos.argmax())])
                check(prefix + "/ungated_maximum", raw_max_lag == rec["ungated_positive_max_lag_ms"])
                check(prefix + "/target_retained", int(rec["target"]) == int(case["target"]))
                check(prefix + "/npz_hash", hashlib.sha256(path.read_bytes()).hexdigest() == rec["arrays_sha256"])
                computed[(subject, event, endpoint, current)] = vals
                metadata[(subject, event, endpoint, current)] = (rec, arrays["original_probs"])
                if endpoint == 100 and current == 0:
                    anchor_argmax_changes += int(not rec["anchor_target_matches_current_argmax"])
        for endpoint in (100, 150):
            rec0, orig0 = metadata[(subject, event, endpoint, 0)]
            rec1, orig1 = metadata[(subject, event, endpoint, 100)]
            check(f"M3/{event}/endpoint{endpoint}/fixed_input", rec0["input_sha256"] == rec1["input_sha256"])
            check(f"M3/{event}/endpoint{endpoint}/fixed_baseline", rec0["template_sha256"] == rec1["template_sha256"])
            check(f"M3/{event}/endpoint{endpoint}/fixed_original_scores", _array_equal(orig0, orig1))
    check("M3/complete_320_cells", len(computed) == 320 and len(cell_csv) == 320)
    check("M3/40_participants_2_events", len(cases) == 80 and all(sum(int(c["subject"]) == s for c in cases) == 2 for s in range(1, 41)))
    draw = np.random.default_rng(20261004).integers(0, 40, size=(10000, 40))
    check("M3/bootstrap_saved_draws", _array_equal(draw, np.load(folder / "participant_bootstrap_indices.npy", allow_pickle=False)))
    cell_rows = _csv(folder / "cell_summary.csv")
    paired_rows = _csv(folder / "paired_contrasts.csv")
    participant_rows = {(int(r["subject"]), r["metric"], int(r["input_endpoint_phase_ms"]), int(r["current_block_phase_ms"])): r for r in _csv(folder / "participant_cell_means.csv")}
    cells = []
    contrasts = []
    def statistic(values):
        boot = values[draw].mean(axis=1)
        interval = np.quantile(boot, [.025, .975])
        return {"estimate": float(values.mean()), "ci95_low": float(interval[0]), "ci95_high": float(interval[1])}
    def check_stat(name, recalc, row):
        for field in ("estimate", "ci95_low", "ci95_high"):
            check(name + "/" + field, _close(recalc[field], float(row[field])), {"recomputed": recalc[field], "saved": float(row[field])})
    for metric in metrics:
        values = np.empty((40, 2, 2), dtype=float)
        for subject in range(1, 41):
            events = [case["anchor_window_id"].rsplit("_ph", 1)[0] for case in cases if int(case["subject"]) == subject]
            for ei, endpoint in enumerate((100, 150)):
                for ci, current in enumerate((0, 100)):
                    mean = float(np.mean([computed[(subject, event, endpoint, current)][metric] for event in events]))
                    values[subject - 1, ei, ci] = mean
                    saved = participant_rows[(subject, metric, endpoint, current)]
                    check(f"M3/participant{subject}/{metric}/{endpoint}/{current}", _close(mean, float(saved["participant_mean"])))
        for ei, endpoint in enumerate((100, 150)):
            for ci, current in enumerate((0, 100)):
                stat = statistic(values[:, ei, ci])
                row = next(r for r in cell_rows if r["metric"] == metric and int(r["input_endpoint_phase_ms"]) == endpoint and int(r["current_block_phase_ms"]) == current)
                check_stat(f"M3/cell/{metric}/{endpoint}/{current}", stat, row)
                cells.append({"metric": metric, "input_endpoint_phase_ms": endpoint, "current_block_phase_ms": current, **stat})
        differences = [
            ("current100_minus0_at_endpoint100", values[:, 0, 1] - values[:, 0, 0]),
            ("current100_minus0_at_endpoint150", values[:, 1, 1] - values[:, 1, 0]),
            ("endpoint150_minus100_at_current0", values[:, 1, 0] - values[:, 0, 0]),
            ("endpoint150_minus100_at_current100", values[:, 1, 1] - values[:, 0, 1]),
            ("difference_in_differences", values[:, 1, 1] - values[:, 1, 0] - values[:, 0, 1] + values[:, 0, 0]),
        ]
        for label, diff in differences:
            stat = statistic(diff)
            stat.update({"n_participant_positive": int((diff > 0).sum()), "n_participant_negative": int((diff < 0).sum()), "n_participant_zero": int((diff == 0).sum())})
            row = next(r for r in paired_rows if r["metric"] == metric and r["contrast"] == label)
            check_stat(f"M3/contrast/{metric}/{label}", stat, row)
            for field in ("n_participant_positive", "n_participant_negative", "n_participant_zero"):
                check(f"M3/contrast/{metric}/{label}/{field}", stat[field] == int(row[field]))
            contrasts.append({"metric": metric, "contrast": label, **stat})
    for recalculated, section in ((cells, "cell_summary"), (contrasts, "paired_contrasts")):
        check("M3/results_summary/" + section + "/row_count", len(recalculated) == len(saved_summary[section]))
        for expected, saved in zip(recalculated, saved_summary[section]):
            for field, value in expected.items():
                check("M3/results_summary/" + section + "/" + str(expected.get("contrast", expected.get("input_endpoint_phase_ms"))) + "/" + expected["metric"] + "/" + field,
                      _close(value, saved[field]) if isinstance(value, (int, float)) else value == saved[field])
    check("M3/changed_anchor_argmax_count", anchor_argmax_changes == saved_summary["n_argmax_changed_from_anchor_at_endpoint100"])
    check("M3/rest_target_count", sum(int(c["target"]) == 0 for c in cases) == saved_summary["n_anchor_target_rest"])
    results["M3"] = {
        "scope": "Saved four-state probability/logit recomputation; inference was not rerun.",
        "definition": "q11-q01-q10+q00 in stored precision; crop 5-145 ms; positive part then Gaussian sigma2 reflect; D=sum*0.5 ms; equal mean of 2 events within each of 40 participants; paired 10,000-participant-bootstrap seed20261004.",
        "n_cells": len(computed), "cell_summary": cells, "paired_contrasts": contrasts,
        "no_transferred_gate": True,
    }


def verify_m5(base, check, results, recomputed=None, invariant_counts=None):
    """Reconstruct hard/taper profiles and weighted participant statistics.

    q11-q01-q10+q00 is evaluated in the recorded float32 order. Eligible
    support is 5--150 ms with 291 half-ms bins. Positive/negative magnitudes
    are Gaussian sigma2 reflect smoothed; physical mass is their sum*0.5ms.
    Positive profiles normalize the smoothed positive part. A pair requires
    both profiles defined. We retain each subject's archived agreement and
    disagreement population weights; ratios over defined stratum mass use
    10,000 paired participant bootstrap draws, seed2026100415.
    """
    compare_recomputed = recomputed or (lambda name, left, right: _array_equal(left, right))
    folder = _followup_dir(base, "m5_seams")
    design = _json(folder / "frozen_design.json")
    saved_summary = _json(folder / "summary.json")
    windows = design["windows"]
    window_csv = {(r["window_id"], r["operator"]): r for r in _csv(folder / "window_operator_results.csv")}
    paired_csv = {r["window_id"]: r for r in _csv(folder / "paired_window_results.csv")}
    participant_csv = {(int(r["subject"]), r["metric"], r["operator"]): r for r in _csv(folder / "participant_results.csv")}
    records = []
    for entry in windows:
        wid = entry["window_id"]
        rec = _json(folder / "formal" / f"{wid}.json")
        with np.load(folder / "formal" / f"{wid}.npz", allow_pickle=False) as npz:
            arrays = {name: npz[name] for name in npz.files}
        profiles = {}
        operators = {}
        for op in ("hard", "taper"):
            prefix = "M5/" + wid + "/" + op
            scores = [arrays[op + "_" + name] for name in ("q11", "q01", "q10", "q00")]
            signed = np.float32(scores[0]) - np.float32(scores[1]) - scores[2] + scores[3]
            positive = np.maximum(signed, 0).astype(float)
            negative = np.maximum(-signed, 0).astype(float)
            smooth = gaussian_filter1d(positive, sigma=2, mode="reflect")
            smooth_neg = gaussian_filter1d(negative, sigma=2, mode="reflect")
            axis = arrays[op + "_lags_ms"]
            profile = smooth / smooth.sum() if positive.sum() > 0 else np.full(smooth.shape, np.nan)
            profiles[op] = profile
            check(prefix + "/eligible_axis", _array_equal(axis, np.arange(5, 150.1, .5)))
            check(prefix + "/finite_probability_scores", bool(all(np.isfinite(score).all() and (score >= 0).all() and (score <= 1).all() for score in scores)))
            check(prefix + "/four_state_identity", _array_equal(signed, arrays[op + "_signed"]))
            check(prefix + "/positive_identity", _array_equal(positive, arrays[op + "_positive"]))
            check(prefix + "/negative_identity", _array_equal(negative, arrays[op + "_negative_magnitude"]))
            check(prefix + "/profile_recomputed", compare_recomputed(prefix + "/profile_recomputed", profile, arrays[op + "_profile"]))
            saved_profile = arrays[op + "_profile"]
            if positive.sum() > 0:
                check(prefix + "/strict_profile_maximum_index", int(profile.argmax()) == int(saved_profile.argmax()))
                check(prefix + "/strict_smoothed_maximum_lag", float(axis[int(smooth.argmax())]) == rec['operators'][op]['smoothed_positive_argmax_ms'])
                check(prefix + "/strict_raw_maximum_lag", float(axis[int(positive.argmax())]) == rec['operators'][op]['raw_positive_argmax_ms'])
                if invariant_counts is not None:
                    invariant_counts['M5_maximum_indices'] += 1
            else:
                check(prefix + "/strict_undefined_profile", bool(np.isnan(saved_profile).all()))
            metrics = {
                "positive_mass_probability_ms": float(.5 * smooth.sum()),
                "negative_mass_probability_ms": float(.5 * smooth_neg.sum()),
                "positive_mass": float(positive.sum()),
                "negative_mass": float(negative.sum()),
                "shortlag_share": float(profile[axis <= 20].sum()) if positive.sum() > 0 else np.nan,
                "raw_positive_argmax_ms": float(axis[np.argmax(positive)]) if positive.sum() > 0 else np.nan,
                "smoothed_positive_argmax_ms": float(axis[np.argmax(smooth)]) if positive.sum() > 0 else np.nan,
                "current_original_boundary_l2": float(arrays[op + "_current_original_boundary"]),
                "current_replaced_boundary_l2": float(arrays[op + "_current_replaced_boundary"]),
                "history_original_boundary_l2": float(arrays[op + "_seams"][:, :2].mean()),
                "history_replaced_boundary_l2": float(arrays[op + "_seams"][:, 2:].mean()),
                "history_internal_jump_mean_l2": float(arrays[op + "_history_internal_jump_mean"].mean()),
                "current_replacement_l1": float(arrays[op + "_current_replacement_l1"]),
                "history_replacement_l1": float(arrays[op + "_history_replacement_l1"].mean()),
            }
            for metric, value in metrics.items():
                saved = window_csv[(wid, op)][metric]
                check(prefix + "/csv/" + metric, _close(value, float(saved) if saved else np.nan))
                # physical-unit masses were subsequently added only to CSV;
                # all other per-scan quantities are also checked against JSON.
                if metric in rec["operators"][op]:
                    saved_json = rec["operators"][op][metric]
                    check(prefix + "/json/" + metric, _close(value, np.nan if saved_json is None else saved_json))
            operators[op] = metrics
        check("M5/" + wid + "/fixed_original_output", _array_equal(arrays["hard_q11"], arrays["taper_q11"]))
        check("M5/" + wid + "/taper_history_seams_exact", _array_equal(arrays["taper_seams"][:, :2], arrays["taper_seams"][:, 2:]))
        check("M5/" + wid + "/taper_current_seam_exact", _array_equal(arrays["taper_current_original_boundary"], arrays["taper_current_replaced_boundary"]))
        defined = bool(np.isfinite(profiles["hard"]).all() and np.isfinite(profiles["taper"]).all())
        paired = {
            "paired_profile_l1": float(np.abs(profiles["hard"] - profiles["taper"]).sum()) if defined else np.nan,
            "raw_argmax_abs_shift_ms": abs(operators["hard"]["raw_positive_argmax_ms"] - operators["taper"]["raw_positive_argmax_ms"]) if defined else np.nan,
            "history_replacement_strength_ratio": operators["taper"]["history_replacement_l1"] / operators["hard"]["history_replacement_l1"],
            "current_replacement_strength_ratio": operators["taper"]["current_replacement_l1"] / operators["hard"]["current_replacement_l1"],
        }
        for metric, value in paired.items():
            check("M5/" + wid + "/paired_csv/" + metric, _close(value, float(paired_csv[wid][metric]) if paired_csv[wid][metric] else np.nan))
            check("M5/" + wid + "/paired_json/" + metric, _close(value, np.nan if rec[metric] is None else rec[metric]))
        records.append({"subject": int(entry["subject"]), "weight": float(entry["population_weight_within_subject"]), "stratum": entry["sampling_stratum"], "operators": operators, "paired": paired})
    persons = np.array([r["subject"] for r in records])
    weights = np.array([r["weight"] for r in records])
    check("M5/80_windows_160_scans", len(records) == 80 and len(window_csv) == 160)
    check("M5/40_two_stratum_participants", all(sum(persons == s) == 2 and {r["stratum"] for r in records if r["subject"] == s} == {"active_agree", "active_disagree"} for s in range(1, 41)))
    check("M5/weights_sum_one_per_subject", all(_close(weights[persons == s].sum(), 1) for s in range(1, 41)))
    draw = np.random.default_rng(2026100415).integers(0, 40, size=(10000, 40))
    def stat(values, selected_weights=None):
        arr = np.asarray(values, dtype=float)
        sw = weights if selected_weights is None else selected_weights
        finite = np.isfinite(arr)
        # Keep the original ratio estimand when some profiles are undefined.
        numerator = np.array([np.sum(arr[(persons == s) & finite] * sw[(persons == s) & finite]) for s in range(1, 41)])
        denominator = np.array([np.sum(sw[(persons == s) & finite]) for s in range(1, 41)])
        boot_den = denominator[draw].sum(axis=1)
        boot_num = numerator[draw].sum(axis=1)
        bootstrap = boot_num[boot_den > 0] / boot_den[boot_den > 0]
        return {"estimate": float(numerator.sum() / denominator.sum()) if denominator.sum() > 0 else None,
                "ci95": np.percentile(bootstrap, [2.5, 97.5]).tolist() if len(bootstrap) else None,
                "eligible_windows": int((finite & (sw > 0)).sum()),
                "weighted_eligible_fraction": float(denominator.sum() / 40)}
    def compare(label, computed, saved):
        for field, value in computed.items():
            if value is None:
                check(label + "/" + field, saved[field] is None)
            else:
                check(label + "/" + field, _close(value, saved[field]), {"recomputed": value, "saved": saved[field]})
    metrics = list(records[0]["operators"]["hard"])
    operators_summary = {}
    delta_summary = {}
    for op in ("hard", "taper"):
        operators_summary[op] = {}
        for metric in metrics:
            vals = np.array([r["operators"][op][metric] for r in records])
            operators_summary[op][metric] = stat(vals)
            compare(f"M5/summary/{op}/{metric}", operators_summary[op][metric], saved_summary["operators"][op][metric])
            for subject in range(1, 41):
                ok = (persons == subject) & np.isfinite(vals)
                den = float(weights[ok].sum())
                mean = float((vals[ok] * weights[ok]).sum() / den) if den else np.nan
                saved = participant_csv[(subject, metric, op)]
                check(f"M5/participant{subject}/{op}/{metric}/mean", _close(mean, float(saved["weighted_mean_among_defined"]) if saved["weighted_mean_among_defined"] else np.nan))
                check(f"M5/participant{subject}/{op}/{metric}/defined_mass", _close(den, float(saved["defined_stratum_mass"])))
    for metric in metrics:
        vals = np.array([r["operators"]["taper"][metric] - r["operators"]["hard"][metric] for r in records])
        delta_summary[metric] = stat(vals)
        compare("M5/paired_delta/" + metric, delta_summary[metric], saved_summary["paired_delta_taper_minus_hard"][metric])
    paired_summary = {}
    for metric in records[0]["paired"]:
        paired_summary[metric] = stat([r["paired"][metric] for r in records])
        compare("M5/paired_other/" + metric, paired_summary[metric], saved_summary["paired_other"][metric])
    for stratum, saved in saved_summary["stratum_summaries"].items():
        sw = weights * np.array([r["stratum"] == stratum for r in records])
        for metric, target in saved.items():
            vals = np.array([r["operators"]["taper"][metric] - r["operators"]["hard"][metric] for r in records])
            compare(f"M5/stratum/{stratum}/{metric}", stat(vals, sw), target)
    changed = sum(r["paired"]["raw_argmax_abs_shift_ms"] > 0 for r in records)
    check("M5/71_raw_maxima_changed", changed == saved_summary["argmax_changed_windows"])
    for op in ("hard", "taper"):
        zeros = sum(r["operators"][op]["positive_mass"] == 0 for r in records)
        check("M5/zero_positive_mass/" + op, zeros == saved_summary["zero_positive_mass"][op])
    results["M5"] = {
        "scope": "Saved four-state probability recomputation; boundary and intervention diagnostics read from archived arrays; inference was not rerun.",
        "definition": "Stored float32 q11-q01-q10+q00; eligible 5-150ms; positive/negative part then sigma2 reflect smoothing; D=sum*0.5ms; normalized positive profile; archived stratum population weights within each of 40 participants; 10,000 paired-participant draws seed2026100415; defined-profile ratios retain denominators.",
        "operators": operators_summary, "paired_delta_taper_minus_hard": delta_summary,
        "paired_other": paired_summary, "argmax_changed_windows": changed,
        "no_transferred_gate": True,
    }
