#!/usr/bin/env python3
"""Independently recalculate packaged saved results; no training or inference.

Run from any current directory:
    python /path/to/reviewer_artifact/verify_saved_results.py
    python /path/to/verify_saved_results.py --root /path/to/reviewer_artifact

Patch v4p1: pass --root pointing to the unchanged closeout-v4 directory.
Requires NumPy and SciPy. Reads only the packaged evidence directory, never an
original workstation path stored inside a receipt. Writes verifier_*.json into
the artifact root. Evidence and scientific outputs are never modified.

Numerical protocols are deliberately preserved: DB2 calibration statistics use
float64, crop-before-smoothing, sigma=2/reflect, sum*0.5, higher 95th quantiles and
strict AND gates. The archived S39 sensitivity explicitly uses its documented
float32 smoothing. AOPC preserves the registered float32 subtraction. These are
different recorded protocols and must not be substituted for one another.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
from verifier_followup_helpers import verify_m3, verify_m5
from numerical_acceptance import NumericalAcceptance, same_higher_source, same_rank, same_decisions, exact_identity
from quantile_provenance import controlled_source_witness


class Audit:
    def __init__(self, root, mode="strict"):
        self.root = root.resolve()
        self.evidence = (self.root / 'evidence').resolve()
        self.checks, self.sources, self.results = [], {}, {}
        self.numerical = NumericalAcceptance(mode)
        self.invariant_counts = {'DB2_binary_decisions': 0, 'controlled_binary_decisions': 0,
                                 'M3_maximum_indices': 0, 'M5_maximum_indices': 0,
                                 'permutation_rank_items': 0, 'higher_order_source_witnesses': 0,
                                 'controlled_mass_exact_value_equivalent_ties': 0}
        # The v3 scientific evidence is frozen. Future packaging does not create
        # a new baseline from possibly altered inputs.
        with (self.root / 'verification/legacy_v3/MANIFEST_SHA256.csv').open(newline='') as stream:
            self.expected_evidence = {r['path']: r['sha256'] for r in csv.DictReader(stream)
                                      if r['path'].startswith('evidence/')}

    def source(self, path):
        path = Path(path).resolve()
        try:
            name = path.relative_to(self.evidence)
        except ValueError:
            raise ValueError('Input is outside the packaged evidence directory: ' + str(path))
        key = 'evidence/' + str(name)
        if key not in self.sources:
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            self.sources[key] = {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}
            self.check('Frozen v3 evidence SHA-256: ' + key,
                       self.expected_evidence.get(key) == h.hexdigest())
        return path

    def json(self, path):
        return json.loads(self.source(path).read_text(encoding='utf-8-sig'))

    def csv(self, path):
        with self.source(path).open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))

    def npz(self, path):
        # Materialize arrays so the zip handle closes before another file is read.
        with np.load(self.source(path), allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def recomputed(self, name, left, right):
        return self.numerical.compare(name, left, right)

    def decisions(self, name, recomputed, saved):
        a, b = np.asarray(recomputed), np.asarray(saved)
        self.check(name, same_decisions(a, b), {'decisions': int(a.size), 'recomputed_dtype': str(a.dtype), 'saved_dtype': str(b.dtype)})
        self.invariant_counts['DB2_binary_decisions'] += int(a.size)

    def higher_source(self, name, recomputed, saved):
        controlled = re.fullmatch(r"\d+/(original_model_argmax|fixed_class1)/(window|simulated_group_mean): null calibration gates/higher95 selected source indices", name)
        if self.numerical.mode == 'portable' and controlled:
            accepted, detail = controlled_source_witness(recomputed, saved)
            self.check(name, accepted, detail)
            self.invariant_counts['controlled_mass_exact_value_equivalent_ties'] += detail['equivalent_ties']
        else:
            self.check(name, same_higher_source(recomputed, saved))
        self.invariant_counts['higher_order_source_witnesses'] += 2

    def check(self, name, passed, detail=None):
        row = {'check': name, 'passed': bool(passed)}
        if detail is not None:
            row['detail'] = detail
        self.checks.append(row)
        if not passed:
            raise AssertionError(name + (': ' + str(detail) if detail is not None else ''))


def stats(curves):
    """The recorded DB2/M4 scalar detector, vectorized over leading axes."""
    y = gaussian_filter1d(np.asarray(curves, dtype=np.float64)[..., 9:],
                          sigma=2, axis=-1, mode='reflect', truncate=4)
    return np.stack((np.maximum(y, 0).sum(-1) * 0.5, y.max(-1)), axis=-1)


def threshold(values):
    """Sorted order statistic: NumPy's .95 quantile with method='higher'."""
    rows = np.asarray(values).reshape(-1, 2)
    return np.sort(rows, axis=0)[math.ceil((len(rows) - 1) * 0.95)]


def support(values, gates):
    return np.all(values > gates, axis=-1)


def close(a, b, atol=1e-12):
    return bool(np.allclose(a, b, atol=atol, rtol=0, equal_nan=True))


def verify_db2(audit):
    base = audit.evidence / 'db2_source'
    cal = audit.evidence / 'followup/m4_calibration'
    labels = audit.json(cal / 'frozen_design.json')['db2']['reference_variants']
    spec = [('r006_curves', 'reinit'), ('r006_curves', 'param_scramble'),
            ('r006_curves_shuffled', 'trained'), ('r006_curves_shuffled', 'reinit'),
            ('r006_curves_shuffled', 'param_scramble')]
    raw = np.empty((40, 5, 68, 300))
    trained = np.empty((40, 68, 300))
    ids, metadata = [], []
    for si in range(40):
        sid = f'S{si+1:02d}'
        t = audit.npz(base / 'r006_curves' / (sid + '_trained.npz'))
        audit.check(sid + ': trained 68x300 curves', t['curves'].shape == (68, 300))
        audit.check(sid + ': original positive clipping',
                    np.array_equal(t['curves'], np.maximum(t['interactions'], 0)))
        trained[si] = t['curves']
        ids.append(t['ids'].tolist())
        for vi, (directory, variant) in enumerate(spec):
            z = audit.npz(base / directory / f'{sid}_{variant}.npz')
            raw[si, vi] = z['curves']
            meta = json.loads(str(z['meta_json']))
            audit.check(f'{sid}/{labels[vi]}: metadata subject', meta['subject'] == si + 1)
            metadata.append(dict(meta, subject=si+1, reference_family=labels[vi]))
            audit.check(f'{sid}/{labels[vi]}: aligned clipped arrays',
                        z['curves'].shape == (68, 300)
                        and np.array_equal(z['ids'], t['ids'])
                        and np.array_equal(z['curves'], np.maximum(z['interactions'], 0)))
    report = audit.json(base / 'r007_detector_report.json')
    old_gates = np.array([report['thresholds']['thresholds'][k]
                          for k in ['positive_mass', 'peak_height']])
    rs, ts = stats(raw), stats(trained)
    rsm, tsm = stats(raw.mean(2)), stats(trained.mean(1))
    pooled_gates = threshold(rs)
    audit.check('DB2 original pooled gates reconstructed', audit.recomputed('DB2 original pooled gates reconstructed', pooled_gates, old_gates))
    audit.check('DB2 original reference exceedances 509/2720',
                int(support(ts, pooled_gates).sum()) == 509)
    saved_arrays = audit.npz(cal / 'db2_statistics_bootstrap.npz')
    for key, values in [('window_reference_statistics', rs), ('window_trained_statistics', ts),
                        ('participant_reference_statistics', rsm), ('participant_trained_statistics', tsm)]:
        audit.check('DB2 raw-curve statistics match saved ' + key,
                    audit.recomputed('DB2 raw-curve statistics match saved ' + key, values, saved_arrays[key]))
    saved_rs = saved_arrays['window_reference_statistics']
    saved_ts = saved_arrays['window_trained_statistics']
    saved_rsm = saved_arrays['participant_reference_statistics']
    saved_tsm = saved_arrays['participant_trained_statistics']
    audit.higher_source('DB2 original higher95 selected source indices', rs, saved_rs)
    audit.decisions('DB2 original: every trained window decision unchanged', support(ts, pooled_gates), support(saved_ts, old_gates))
    audit.decisions('DB2 original: every reference window decision unchanged', support(rs, pooled_gates), support(saved_rs, old_gates))
    mean_gates, saved_mean_gates = threshold(rsm), threshold(saved_rsm)
    audit.higher_source('DB2 original participant means higher95 selected source indices', rsm, saved_rsm)
    audit.decisions('DB2 original: every trained participant-mean decision unchanged', support(tsm, mean_gates), support(saved_tsm, saved_mean_gates))
    audit.decisions('DB2 original: every reference participant-mean decision unchanged', support(rsm, mean_gates), support(saved_rsm, saved_mean_gates))
    folds = np.arange(40) % 4
    answer = {'definition': 'float64, positive clipping per window, crop 5–150ms before sigma2/reflect smoothing; D=sum*0.5, H=max; strict AND',
              'original': {'trained_exceed': 509, 'trained_n': 2720,
                           'mass_gate': float(pooled_gates[0]), 'height_gate': float(pooled_gates[1])},
              'policies': []}
    for filename, rule in [('db2_results.json', 'pooled'), ('envelope_results.json', 'max_family_post_hoc')]:
        saved = audit.json(cal / filename)
        for level, r, t in [('window', rs, ts), ('participant_mean', rsm, tsm)]:
            rp, tp = np.zeros(r.shape[:-1], bool), np.zeros(t.shape[:-1], bool)
            fold_rows = []
            for fold in range(4):
                ca, te = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
                th = threshold(r[ca]) if rule == 'pooled' else np.max(
                    np.array([threshold(r[ca, v]) for v in range(5)]), axis=0)
                rp[te], tp[te] = support(r[te], th), support(t[te], th)
                row = next(v for v in saved['folds'] if v['level'] == level and v['fold'] == fold)
                gate_name = f'DB2/{rule}/{level}/fold{fold}: independently sorted gates'
                saved_th = np.array([row['mass_threshold'], row['height_threshold']], dtype=np.float64)
                audit.check(gate_name, audit.recomputed(gate_name, th, saved_th))
                saved_r, saved_t = (saved_rs, saved_ts) if level == 'window' else (saved_rsm, saved_tsm)
                if rule == 'pooled':
                    audit.higher_source(gate_name + '/higher95 source indices', r[ca], saved_r[ca])
                else:
                    for family_index in range(5):
                        audit.higher_source(gate_name + f'/family{family_index} higher95 source indices', r[ca, family_index], saved_r[ca, family_index])
                    family_gates = np.array([threshold(r[ca, v]) for v in range(5)])
                    saved_family_gates = np.array([threshold(saved_r[ca, v]) for v in range(5)])
                    audit.check(gate_name + '/envelope attaining family indices', np.array_equal(family_gates.argmax(0), saved_family_gates.argmax(0)))
                audit.decisions(gate_name + '/every reference decision unchanged', rp[te], support(saved_r[te], saved_th))
                audit.decisions(gate_name + '/every trained decision unchanged', tp[te], support(saved_t[te], saved_th))
                audit.check(gate_name + '/saved counts',
                            int(rp[te].sum()) == row['reference_exceed']
                            and int(tp[te].sum()) == row.get('trained_support', row.get('trained_supported')))
                caids = {ident for si in ca for ident in ids[si]}
                teids = {ident for si in te for ident in ids[si]}
                caseeds = {m['variant_seed'] for m in metadata if m['subject'] - 1 in ca}
                teseeds = {m['variant_seed'] for m in metadata if m['subject'] - 1 in te}
                audit.check(f'DB2/{rule}/{level}/fold{fold}: saved participant/window/seed identities disjoint',
                            not set(ca) & set(te) and not caids & teids and not caseeds & teseeds)
                fold_rows.append({'fold': fold, 'mass_gate': float(th[0]), 'height_gate': float(th[1]),
                                  'reference_exceed': int(rp[te].sum()), 'trained_exceed': int(tp[te].sum())})
            expected_trained = (519 if rule == 'pooled' else 22) if level == 'window' else 0
            expected_reference = (630 if rule == 'pooled' else 96) if level == 'window' else (9 if rule == 'pooled' else 2)
            audit.check(f'DB2/{rule}/{level}: aggregate saved-result claim',
                        int(tp.sum()) == expected_trained and int(rp.sum()) == expected_reference)
            by_family = {labels[v]: int(rp[:, v].sum()) for v in range(5)}
            if level == 'window':
                audit.check(f'DB2/{rule}: all window reference exceedances are label-shuffled trained',
                            int(rp[:, 2].sum()) == int(rp.sum()))
            answer['policies'].append({'policy': rule, 'level': level, 'trained_exceed': int(tp.sum()),
                                       'trained_n': int(tp.size), 'reference_exceed': int(rp.sum()),
                                       'reference_n': int(rp.size), 'by_reference_family': by_family,
                                       'folds': fold_rows})
    omissions = audit.csv(cal / 'db2_reference_composition_sensitivity.csv')
    for level, r, t in [('window', rs, ts), ('participant_mean', rsm, tsm)]:
        for vi, label in enumerate(labels):
            th = threshold(r[:, [j for j in range(5) if j != vi]])
            row = next(v for v in omissions if v['level'] == level and v['omitted_family'] == label)
            audit.check(f'DB2/{level}/omit {label}: empirical family-composition sensitivity',
                        close(th, [float(row['mass_threshold']), float(row['height_threshold'])])
                        and int(support(t, th).sum()) == int(row['trained_support'])
                        and int(support(r[:, vi], th).sum()) == int(row['omitted_reference_exceed']))
    for row in audit.csv(cal / 'db2_numeric_floor_sensitivity.csv'):
        eps = float(row['floor'])
        rr, tr = stats(np.where(raw <= eps, 0, raw)), stats(np.where(trained <= eps, 0, trained))
        th = threshold(rr)
        audit.check('DB2 numeric floor ' + str(eps),
                    close(th, [float(row['mass_threshold']), float(row['height_threshold'])])
                    and int(support(tr, th).sum()) == int(row['trained_support']))
    answer['interpretation_limits'] = [
        'The max-family policy is explicitly post hoc, not a corrected primary prevalence estimate.',
        'Reference exceedance rates are empirical benchmarks, not a known-zero physiological false-positive rate.',
        'Participant means mix phases and targets; 0/40 does not prove no participant-level interaction.',
        'DB2 bootstrap confidence intervals are not rerun in this verifier.']
    audit.results['DB2_calibration'] = answer
    return raw, trained, ids, old_gates


def verify_models(audit):
    base = audit.evidence / 'db2_source/r005b_balanced_full_seed20260815'
    comp = audit.evidence / 'completion/comparators/formal'
    closeout = audit.evidence / 'closeout/model_comparison'
    binding = {r['participant']: r for r in audit.csv(
        audit.evidence / 'followup/history/replacement_checkpoint_training_binding.csv')}
    saved_rows = {r['participant']: r for r in audit.csv(closeout / 'complete_audit_per_participant.csv')}
    rows, cms = [], []
    for si in range(1, 41):
        sid = f'S{si:02d}'
        result = audit.json(base / 'subjects' / (sid + '_result.json'))
        cm = np.array(result['audit_metrics']['confusion_matrix'], dtype=np.int64)
        audit.check(sid + ': original 18x18 confusion matrix', cm.shape == (18, 18) and np.all(cm >= 0))
        counts = cm.sum(1)
        audit.check(sid + ': all18 audit classes represented', np.all(counts > 0))
        diag = np.diag(cm)
        values = {'accuracy': float(diag.sum() / cm.sum()),
                  'macro_f1': float((2 * diag / (counts + cm.sum(0))).mean()),
                  'balanced_accuracy': float((diag / counts).mean())}
        membership = audit.csv(base / 'window_manifests' / (sid + '.csv'))
        selected = [r for r in membership if r['split'] == 'audit']
        classes = Counter(int(r['modal_label']) for r in selected)
        audit.check(sid + ': audit membership equals confusion-matrix support',
                    len(selected) == int(cm.sum()) == result['selected_window_counts']['audit']
                    and all(classes[i] == int(counts[i]) for i in range(18)))
        comp_source = audit.json(comp / (sid + '_source_manifest.json'))
        audit.check(sid + ': saved checkpoint hash association agrees across three records',
                    result['checkpoint_sha256'] == comp_source['checkpoint_sha256'] == binding[sid]['checkpoint_sha256'])
        for name, value in values.items():
            audit.check(f'{sid}/{name}: rederived complete-audit metric',
                        abs(value - result['audit_metrics'][name]) < 1e-12
                        and abs(value - float(saved_rows[sid][name])) < 1e-12)
        rows.append(dict(participant=sid, audit_windows=int(cm.sum()), **values))
        cms.append(cm)
    audit.check('40 replacement models: 111697 complete-audit windows',
                len(rows) == 40 and sum(r['audit_windows'] for r in rows) == 111697)
    summaries = []
    saved_summary = audit.csv(closeout / 'complete_audit_metric_summary.csv')
    for name in ['accuracy', 'macro_f1', 'balanced_accuracy']:
        values = np.array([r[name] for r in rows])
        summary = {'metric': name, 'median': float(np.median(values)),
                   'q25': float(np.quantile(values, 0.25)), 'q75': float(np.quantile(values, 0.75)),
                   'mean': float(values.mean()), 'minimum': float(values.min()), 'maximum': float(values.max())}
        saved = next(r for r in saved_summary if r['metric'] == name)
        audit.check('40 replacement models: participant summary ' + name,
                    all(abs(summary[k] - float(saved[k])) < 1e-12 for k in summary if k != 'metric'))
        summaries.append(summary)
    pooled = np.stack(cms).sum(0)
    saved_pooled = audit.csv(closeout / 'complete_audit_pooled_confusion_matrix.csv')
    audit.check('40 replacement models: pooled matrix reconstructed',
                np.array_equal(pooled, [[int(r[f'predicted_{i}']) for i in range(18)] for r in saved_pooled]))
    audit.results['replacement_model_metrics'] = {
        'participants': 40, 'audit_windows': 111697, 'participant_summaries': summaries,
        'pooled_accuracy_descriptive': float(np.trace(pooled) / pooled.sum()),
        'limits': 'Metrics and class counts rederived from saved confusion matrices and memberships. Only saved hash strings are cross-linked; this package does not contain all40 checkpoint bytes, and no new training/inference is run.'}


def verify_comparators(audit):
    base = audit.evidence / 'completion/comparators'
    design, saved = audit.json(base / 'frozen_design.json'), audit.json(base / 'summary.json')
    ids = [r['window_id'] for r in design['windows']]
    subjects = np.array([r['subject'] for r in design['windows']])
    weights = np.array([r['population_weight_within_subject'] for r in design['windows']])
    audit.check('Comparator original80window/40participant design', len(ids) == 80 and len(set(subjects)) == 40)
    audit.check('Comparator original stratum weights sum1/person',
                all(abs(weights[subjects == s].sum() - 1) < 1e-12 for s in range(1, 41)))
    methods, passing = {}, []
    for method in ['ST_SRI', 'TimeSHAP', 'DynaMask', 'DirectFactorial']:
        values = []
        for ident in ids:
            row = audit.json(base / 'formal' / f'{ident}_{method}_primary.json')
            ev = row['evaluation']
            remaining = np.array(ev['remaining_probability'], dtype=np.float32)
            value = float(np.mean(100 * (ev['original_probability'] - remaining)))
            audit.check(f'{ident}/{method}: AOPC from original and6 deletion probabilities',
                        len(remaining) == 6 and abs(value - ev['aopc_pp']) < 1e-12)
            values.append(value)
            if method == 'DynaMask':
                mse = row.get('final_returned_mask_prediction_mse', float('nan'))
                passed = bool(np.isfinite(mse) and mse <= 1e-4)
                audit.check(ident + ': final Dynamask preservation criterion',
                            passed == row['converged'] and row['convergence_threshold'] == 1e-4)
                passing.append(passed)
        methods[method] = np.array(values)
    passing = np.array(passing, bool)
    audit.check('Dynamask final returned masks:32passing windows,26participants',
                int(passing.sum()) == 32 and len(set(subjects[passing])) == 26)
    draws = np.random.default_rng(design['seed'] + 999).integers(0, 40, (5000, 40))

    def estimate(values, eligible):
        mask = np.asarray(eligible, bool) & np.isfinite(values)
        nums = np.array([np.sum(values[(subjects == s) & mask] * weights[(subjects == s) & mask])
                         for s in range(1, 41)])
        dens = np.array([np.sum(weights[(subjects == s) & mask]) for s in range(1, 41)])
        den = dens[draws].sum(1)
        boot = nums[draws].sum(1)[den > 0] / den[den > 0]
        return {'mean': float(nums.sum() / dens.sum()), 'ci95': np.quantile(boot, [0.025, 0.975]).tolist(),
                'windows': int(mask.sum()), 'participants': int((dens > 0).sum()),
                'original_stratum_weight_sum': float(dens.sum())}

    rows = []
    for subset, eligible in [('all80windows', np.ones(80, bool)), ('Dynamask32passingwindows', passing)]:
        for name, values in methods.items():
            result = estimate(values, eligible)
            expected = (saved['methods'][name]['aopc_pp_overall'] if subset == 'all80windows'
                        else saved['same_converged_subset']['methods'][name])
            audit.check(f'Comparator/{subset}/{name}: rederived weighted AOPC and participant-bootstrap CI',
                        close(result['mean'], expected['mean']) and close(result['ci95'], expected['ci95']))
            rows.append(dict(subset=subset, method_or_difference=name, **result))
        for other in ['TimeSHAP', 'DynaMask', 'DirectFactorial']:
            name = 'ST_SRI_minus_' + other
            result = estimate(methods['ST_SRI'] - methods[other], eligible)
            expected = saved['paired_differences_pp'][name][
                'all_windows' if subset == 'all80windows' else 'same_converged_windows']
            audit.check(f'Comparator/{subset}/{name}: within-window paired difference and CI',
                        close(result['mean'], expected['mean']) and close(result['ci95'], expected['ci95']))
            rows.append(dict(subset=subset, method_or_difference=name, **result))
    saved_passing = audit.csv(audit.evidence / 'closeout/model_comparison/comparator_Dynamask32passing_window_ids.csv')
    audit.check('Dynamask passing window identities equal independent threshold selection',
                set(np.array(ids)[passing]) == {r['window_id'] for r in saved_passing})
    audit.results['comparator_AOPC'] = {
        'definition': 'Mean of six percentage-point drops; serialized remaining probabilities cast to float32 before original-probability subtraction. Original stratum weights renormalized within subset;5000 participant-cluster resamples.',
        'bootstrap_seed': int(design['seed'] + 999), 'statistics': rows,
        'limits': 'Saved deletion probabilities are recalculated; explainer optimization and model inference are not rerun. Passing subset is conditional and paired confidence intervals do not support an ST-SRI superiority claim.'}


def verify_synthetic(audit):
    base = audit.evidence / 'completion/m4_validation'
    follow = audit.evidence / 'followup/m4_calibration'
    config, saved = audit.json(base / 'frozen_config.json'), audit.json(base / 'run_results.json')
    models = saved['models']
    audit.check('Controlled validation32models', len(models) == 32)
    fit = []
    cache = {}
    for model in models:
        mid = model['model_id']
        z = audit.npz(base / f'validation_{mid}.npz')
        p, y, hard = z['prediction'], z['teacher_probability'], z['hard_labels']
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        acc = float(np.mean((p >= 0.5) == hard))
        bayes = float(np.mean(np.maximum(y, 1 - y)))
        passed = rmse <= config['acceptance']['heldout_probability_rmse_max'] and rmse < model['constant_rmse']
        audit.check(mid + ': independent saved-prediction fit/accuracy',
                    rmse == model['fit_rmse'] and acc == model['hard_label_accuracy']
                    and bayes == model['expected_Bayes_accuracy'] and bool(passed) == model['fit_acceptance_passed'])
        fit.append({'model_id': mid, 'fit_rmse': rmse, 'hard_label_accuracy': acc,
                    'expected_Bayes_accuracy': bayes, 'fit_passed': bool(passed)})
        c = audit.npz(base / f'curves_{mid}.npz')
        for split in ['calibration', 'test']:
            audit.check(mid + '/' + split + ': target is saved unmasked argmax',
                        np.array_equal(c['original_argmax_' + split], c['unmasked_probabilities_' + split].argmax(1)))
        cache[mid] = c
    audit.check('All32 controlled-model fit acceptance gates pass', all(r['fit_passed'] for r in fit))
    csv_rows = audit.csv(base / 'all_decisions.csv')
    decisions = {(r['model_id'], r['target_mode'], r['level'], r['split'], int(r['index'])): r for r in csv_rows}
    checked, raw_null_test, raw_null_groups = 0, 0, 0
    floor_rows = []
    for seed in config['seeds']:
        for mode in ['original_model_argmax', 'fixed_class1']:
            gates = {}
            for family in ['A', 'B']:
                for mechanism in ['null_dgp', 'interaction_dgp']:
                    mid = f'{seed}_{family}_{mechanism}'
                    data = cache[mid]
                    curves = {}
                    for split in ['calibration', 'test']:
                        signed = data['signed_two_classes_' + split]
                        target = data['original_argmax_' + split]
                        selected = signed[np.arange(256), :, target] if mode == 'original_model_argmax' else signed[:, :, 1]
                        curves[split] = np.maximum(selected, 0)
                    for level in ['window', 'simulated_group_mean']:
                        cur = curves if level == 'window' else {s: c.reshape(32, 8, 300).mean(1) for s, c in curves.items()}
                        if family == 'A' and mechanism == 'null_dgp':
                            gates[level] = threshold(stats(cur['calibration']))
                            threcord = next(r for r in saved['thresholds'] if r['seed'] == seed and r['target_mode'] == mode)
                            sg = threcord['window' if level == 'window' else 'group']
                            gate_name = f'{seed}/{mode}/{level}: null calibration gates'
                            audit.check(gate_name, audit.recomputed(gate_name, gates[level], np.array([sg['positive_mass'], sg['peak_height']], dtype=np.float64)))
                            saved_calibration = np.array([[float(decisions[mid, mode, level, 'calibration', i]['mass']), float(decisions[mid, mode, level, 'calibration', i]['height'])] for i in range(len(cur['calibration']))])
                            audit.higher_source(gate_name + '/higher95 selected source indices', stats(cur['calibration']), saved_calibration)
                        for split in ['calibration', 'test']:
                            st = stats(cur[split])
                            ps = support(st, gates[level])
                            rows = [decisions[mid, mode, level, split, i] for i in range(len(st))]
                            audit.check(f'{mid}/{mode}/{level}/{split}: all saved mass/height/gate decisions',
                                        close(st[:, 0], [float(r['mass']) for r in rows])
                                        and close(st[:, 1], [float(r['height']) for r in rows])
                                        and np.array_equal(ps, [r['supported'].lower() == 'true' for r in rows]))
                            checked += len(st)
                            if family == 'A' and mechanism == 'null_dgp' and mode == 'original_model_argmax' and split == 'test':
                                if level == 'window': raw_null_test += int(ps.sum())
                                else: raw_null_groups += int(ps.sum())
    audit.invariant_counts['controlled_binary_decisions'] = checked
    audit.check('All36864 synthetic saved detector decisions rederived', checked == len(csv_rows) == 36864)
    audit.check('Original strict-additive float32 result86/2048windows,11/256means',
                raw_null_test == 86 and raw_null_groups == 11)
    saved_floor = audit.csv(follow / 'synthetic_numeric_floor_summary.csv')
    for eps in config.get('zero_floor_sensitivity', [0, 1e-7, 1e-6]):
        pass  # Floors are read from the separately frozen follow-up table below.
    floors = sorted({float(r['floor']) for r in saved_floor})
    for eps in floors:
        for level in ['window', 'group_mean']:
            for family in ['A', 'B']:
                for mechanism in ['null_dgp', 'interaction_dgp']:
                    kk = nn = 0
                    for seed in config['seeds']:
                        def selected(mid, split):
                            data = cache[mid]
                            z = data['signed_two_classes_' + split]
                            target = data['original_argmax_' + split]
                            c = z[np.arange(256), :, target].astype(float)
                            c = np.maximum(np.where(np.abs(c) <= eps, 0, c), 0)
                            return c if level == 'window' else c.reshape(32, 8, 300).mean(1)
                        gates = threshold(stats(selected(f'{seed}_A_null_dgp', 'calibration')))
                        pred = support(stats(selected(f'{seed}_{family}_{mechanism}', 'test')), gates)
                        kk += int(pred.sum()); nn += pred.size
                    row = next(r for r in saved_floor if float(r['floor']) == eps and r['level'] == level
                               and r['family'] == family and r['mechanism'] == mechanism)
                    audit.check(f'Synthetic floor{eps}/{level}/{family}/{mechanism}: saved decisions from raw scores',
                                kk == int(row['supported']) and nn == int(row['n']))
                    floor_rows.append({'floor': eps, 'level': level, 'family': family,
                                       'mechanism': mechanism, 'supported': kk, 'n': int(nn)})
    audit.results['synthetic_validation'] = {'fit_metrics': fit, 'saved_detector_decisions_rederived': checked,
                                            'original_strict_null': {'windows': [86, 2048], 'group_means': [11, 256]},
                                            'numeric_floor_sensitivity': floor_rows,
                                            'limits': 'Stored validation predictions and scan scores are reanalyzed. The stored constant-training-mean baseline is used for the acceptance comparison; it is not regenerated here. No weights are loaded and no training or inference is rerun. A-family additive learned probability is the exact-null condition; B null_dgp does not certify a learned-model null.'}


def verify_precision(audit):
    base = audit.evidence / 'followup/m4_calibration'
    original = audit.evidence / 'completion/m4_validation'
    saved = audit.json(base / 'precision_results.json')
    recomputed, decisions = [], []
    for seed in range(202610041, 202610049):
        for mechanism in ['null_dgp', 'interaction_dgp']:
            z = audit.npz(base / f'precision64_{seed}_A_{mechanism}.npz')
            old = audit.npz(original / f'curves_{seed}_A_{mechanism}.npz')
            null_curves = {}
            for split in ['calibration', 'test']:
                signed = z['signed_two_classes_' + split]
                n = len(signed)
                targets = old['original_argmax_' + split][:n]
                row = {'seed': seed, 'mechanism': mechanism, 'split': split, 'n': n,
                       'max_float32_abs_eligible': float(np.max(np.abs(old['signed_two_classes_' + split][:n, 9:]))),
                       'max_float64_abs_eligible': float(np.max(np.abs(signed[:, 9:]))),
                       'max_float64_oracle_error': float(np.max(np.abs(signed[:, 9:, 1] - z['oracle_' + split][:, 9:]))),
                       'max_float64_float32_difference': float(np.max(np.abs(signed[:, 9:] - old['signed_two_classes_' + split][:n, 9:]))),
                       'argmax_changed': int(np.sum(z['probabilities_' + split].argmax(1) != targets))}
                expected = next(r for r in saved['rows'] if r['seed'] == seed and r['mechanism'] == mechanism and r['split'] == split)
                audit.check(f'Precision/{seed}/{mechanism}/{split}: raw-array numerical comparison',
                            all((row[k] == expected[k]) if isinstance(row[k], str) else close(row[k], expected[k]) for k in row))
                recomputed.append(row)
                if mechanism == 'null_dgp':
                    null_curves[split] = np.maximum(signed[np.arange(n), :, targets], 0)
            if mechanism == 'null_dgp':
                for level in ['window', 'group_mean']:
                    cur = null_curves if level == 'window' else {s: c.reshape(32, 8, 300).mean(1) for s, c in null_curves.items()}
                    gates = threshold(stats(cur['calibration']))
                    ps = support(stats(cur['test']), gates)
                    row = {'seed': seed, 'level': level, 'n': int(ps.size), 'supported': int(ps.sum()),
                           'mass_threshold': float(gates[0]), 'height_threshold': float(gates[1])}
                    expected = next(r for r in saved['null_decisions'] if r['seed'] == seed and r['level'] == level)
                    audit.check(f'Precision/{seed}/{level}: double-precision residual gates',
                                all((row[k] == expected[k]) if isinstance(row[k], str) else close(row[k], expected[k]) for k in row))
                    decisions.append(row)
    null_rows = [r for r in recomputed if r['mechanism'] == 'null_dgp']
    inter_rows = [r for r in recomputed if r['mechanism'] == 'interaction_dgp']
    audit.check('Precision4096null+512interaction saved scans',
                sum(r['n'] for r in null_rows) == 4096 and sum(r['n'] for r in inter_rows) == 512)
    audit.check('Precision residual windows64/2048 and groupmeans7/256',
                sum(r['supported'] for r in decisions if r['level'] == 'window') == 64
                and sum(r['supported'] for r in decisions if r['level'] == 'group_mean') == 7)
    audit.check('Precision no original target changes', sum(r['argmax_changed'] for r in recomputed) == 0)
    audit.results['precision_saved_output'] = {
        'null_scans': 4096, 'interaction_scans': 512,
        'null_max_float32_abs_eligible': max(r['max_float32_abs_eligible'] for r in null_rows),
        'null_max_float64_abs_eligible': max(r['max_float64_abs_eligible'] for r in null_rows),
        'max_interaction_oracle_error': max(r['max_float64_oracle_error'] for r in inter_rows),
        'recalibrated_residual_exceedances': {'windows': [64, 2048], 'group_means': [7, 256]},
        'limits': 'Only saved double-precision output arrays are independently recalculated. No claim of having rerun the double-precision scans. Residual-gate exceedances rank floating-point roundoff and are not a DB2 false-positive validation.'}


def verify_s39(audit, old_gates):
    base = audit.evidence / 'db2_source'
    summary = audit.csv(audit.evidence / 'followup/history/S39_influence_summary.csv')
    rows = []
    for dataset, relative in [('primary', 'r006_merged/trained.npz'),
                              ('R010_rest_mean', 'r010_analysis/merged_rest_mean.npz')]:
        z = audit.npz(base / relative)
        keep = (z['lags_ms'] >= 5) & (z['lags_ms'] <= 150)
        y = gaussian_filter1d(z['curves'][:, keep].astype(np.float32), 2, axis=1, mode='reflect')
        # This historical influence receipt specifies float32 smoothing, not DB2's float64 follow-up.
        ps = (y.sum(1) * 0.5 > old_gates[0]) & (y.max(1) > old_gates[1])
        ix = np.flatnonzero(z['ids'] == 'S39_onset_seg173_ph0')
        audit.check(dataset + ': one predesignated unresolved S39 record', len(ix) == 1)
        for scenario, selected in [('all', ps), ('exclude_unresolved', np.delete(ps, ix[0]))]:
            saved = next(r for r in summary if r['dataset'] == dataset and r['scenario'] == scenario)
            audit.check(dataset + '/' + scenario + ': stored-rate influence recomputed',
                        int(selected.sum()) == int(saved['supported']) and len(selected) == int(saved['n_windows']))
            rows.append({'dataset': dataset, 'scenario': scenario, 'reference_exceed': int(selected.sum()),
                         'n': len(selected), 'percent': float(100 * selected.mean())})
    audit.results['S39_influence'] = {'saved_array_reanalysis': rows, 'single_status_flip_max_pp': 100 / 2720,
                                      'limits': 'This rate-influence bound does not resolve the historical CUDA/CPU/MPS discrepancy.'}


def verify_permutation_and_figure(audit, raw, trained, ids, gates):
    base = audit.evidence / 'completion/controlled_replays'
    rep = audit.evidence / 'closeout/representative'
    design = audit.json(base / 'frozen_design.json')
    cases = {r['window_id']: r for r in design['cases'] if r['representative']}
    selection = audit.json(rep / 'spectrum_example_selection.json')
    manifest = {r['run_id']: r for r in audit.csv(rep / '800_permutation_selection_manifest.csv')}
    candidates = []
    for path in sorted((base / 'random/mps').glob('S*/*_perm_*_primary.npz')):
        z, meta = audit.npz(path), audit.json(path.with_suffix('.json'))
        if meta['variant'] != 'perm':
            continue
        case, ident = cases[meta['window_id']], path.stem
        audit.check(ident + ': permutation array hash matches metadata',
                    audit.sources[str(path.relative_to(audit.root))]['sha256'] == meta['arrays_sha256'])
        audit.check(ident + ': unchanged target/primary background and four-state arithmetic',
                    meta['target'] == case['target'] and meta['background'] == 'primary'
                    and np.array_equal(z['prob_signed'], z['prob_scores'][0] - z['prob_scores'][1]
                                       - z['prob_scores'][2] + z['prob_scores'][3]))
        st = stats(np.maximum(z['prob_signed'], 0))
        passed = bool(support(st, gates))
        row = manifest[ident]
        audit.check(ident + ': recomputed permutation mass/height/gate',
                    close(st, [meta['statistics']['positive_mass'], meta['statistics']['peak_height']])
                    and close(st, [float(row['eligible_mass_probability_ms']), float(row['peak_height_probability'])])
                    and passed == meta['supported'])
        rank = (meta['seed'] - (20261004 + 1009 * meta['subject'])) // 17
        audit.check(ident + ': frozen ten-seed schedule',
                    rank in range(10) and meta['seed'] == 20261004 + 1009 * meta['subject'] + 17 * rank)
        candidates.append({'run_id': ident, 'window_id': meta['window_id'], 'subject': meta['subject'],
                           'positive_mass': float(st[0]), 'peak_height': float(st[1]),
                           'reference_exceed': passed, 'positive_curve': np.maximum(z['prob_signed'], 0).astype(float)})
    audit.check('Permutation saved scans800,80windows,40participants,10seeds/window',
                len(candidates) == len(manifest) == 800 and len({r['window_id'] for r in candidates}) == 80
                and len({r['subject'] for r in candidates}) == 40
                and all(v == 10 for v in Counter(r['window_id'] for r in candidates).values()))
    audit.check('Permutation saved800 curves: zero original reference-gate exceedances',
                sum(r['reference_exceed'] for r in candidates) == 0)
    run_ids = np.array([r['run_id'] for r in candidates])
    recomputed_mass = np.array([r['positive_mass'] for r in candidates])
    saved_mass = np.array([float(manifest[r['run_id']]['eligible_mass_probability_ms']) for r in candidates])
    audit.check('Permutation all800 deterministic mass ranks unchanged (exact ties by run_id)',
                same_rank(-recomputed_mass, -saved_mass, run_ids))
    audit.invariant_counts['permutation_rank_items'] = len(candidates)
    chosen = sorted(candidates, key=lambda r: (-r['positive_mass'], r['run_id']))[0]
    audit.check('RepresentativeD is declared post-hoc largest mass among all800',
                selection['selection_is_post_hoc'] and chosen['run_id'] == selection['selected_parameter_permutation']['run_id'])
    plotted = audit.npz(rep / 'plotted_curves.npz')
    flat_ids = np.array([ident for group in ids for ident in group])
    flat_train = trained.reshape(-1, 300)
    flat_ref = raw.reshape(-1, 300)
    ts, rs = stats(flat_train), stats(flat_ref)
    audit.check('RepresentativeC is maximum-mass reference family example',
                close(stats(plotted['C_raw_positive'])[0], rs[:, 0].max()))
    for panel in ['A', 'B']:
        record = next(r for r in selection['examples'] if r['panel'] == panel)
        ix = np.flatnonzero(flat_ids == record['window_id'])
        audit.check('Representative' + panel + ': saved raw curve equals original trained source',
                    len(ix) == 1 and np.array_equal(plotted[panel + '_raw_positive'], flat_train[ix[0]]))
        audit.check('Representative' + panel + ': gate status agrees with original reference gates',
                    bool(support(ts[ix[0]], gates)) == (panel == 'A'))
    audit.check('RepresentativeD raw curve matches independently selected saved run',
                np.array_equal(plotted['D_raw_positive'], chosen['positive_curve']))
    for panel in ['A', 'B', 'C', 'D']:
        expected = gaussian_filter1d(plotted[panel + '_raw_positive'][9:], 2, mode='reflect')
        audit.check('Representative' + panel + ': plotted processing independently reconstructed',
                    audit.recomputed('Representative' + panel + ': plotted processing independently reconstructed', expected, plotted[panel + '_eligible_smoothed']))
        audit.check('Representative' + panel + ': smoothed maximum index unchanged', int(expected.argmax()) == int(plotted[panel + '_eligible_smoothed'].argmax()))
    audit.results['permutation_and_representative'] = {
        'saved_curves': 800, 'windows': 80, 'participants': 40, 'reference_exceedances': 0,
        'post_hoc_selected_D': {k: v for k, v in chosen.items() if k != 'positive_curve'},
        'limits': 'Finite-seed saved-array result, not a universal dependence-on-learned-parameters guarantee. Representative selection is explicitly post hoc.'}


def verify_boundary_lineage(audit):
    base = audit.evidence / 'closeout/boundary_lineage'
    rows = audit.csv(base / 'smoothing_lineage_summary.csv')
    provenance = {r['participant']: r for r in audit.csv(base / 'lineage_per_participant.csv')}
    revised, historical = [], []
    hist_n = 0
    for subject in range(1, 41):
        sid = f'S{subject:02d}'
        r = audit.npz(audit.evidence / 'db2_source/r006_curves' / (sid + '_trained.npz'))
        hpath = audit.evidence / 'historical_cache' / f'S{subject}_tc30_stride1_block10_max150.npz'
        h = audit.npz(hpath)
        audit.check(sid + ': lineage cache hashes bound to saved source table',
                    audit.sources[str(hpath.relative_to(audit.root))]['sha256'] == provenance[sid]['historical_cache_sha256'])
        lags = r['lags_ms']; keep = (lags >= 5) & (lags <= 150)
        audit.check(sid + ': lineage uses common lag grid', np.array_equal(lags, h['lags_ms']))
        revised.append(np.maximum(r['curves'], 0).astype(float).mean(0))
        historical.append(np.maximum(h['spectra'].mean(0), 0))
        hist_n += len(h['spectra'])
    audit.check('Boundary lineage: all839historical selected windows retained', hist_n == 839)
    mapped = {}
    for lineage, arr in [('historical_839', np.stack(historical)), ('revised_2720', np.stack(revised))]:
        for row in [r for r in rows if r['lineage'] == lineage]:
            sigma = float(row['sigma_bins']); order = row['order']
            if order == 'smooth_then_crop':
                prepared = gaussian_filter1d(arr, sigma, axis=1, mode='reflect')[:, keep]
            elif order == 'crop_then_smooth':
                prepared = gaussian_filter1d(arr[:, keep], sigma, axis=1, mode='reflect')
            else:
                work = arr.copy(); work[:, ~keep] = 0
                prepared = gaussian_filter1d(work, sigma, axis=1, mode='reflect')[:, keep]
            peak = lags[keep][prepared.argmax(1)]
            mapped[lineage, sigma, order] = peak
            audit.check(f'{lineage}/{sigma}/{order}: raw maxima counts and median',
                        int((peak == 5).sum()) == int(row['n_floor']) and int((peak < 30).sum()) == int(row['n_short_lt30'])
                        and float(np.median(peak)) == float(row['median_peak_ms']))
    audit.check('Revised smoothing order changes8of40maxima',
                int((mapped['revised_2720', 2., 'smooth_then_crop'] != mapped['revised_2720', 2., 'crop_then_smooth']).sum()) == 8)
    audit.results['boundary_lineage'] = {'historical_windows': 839, 'revised_windows': 2720,
        'summary_rows': rows, 'changed_revised_maxima': 8,
        'limits': 'Local historical caches reproduce Table S8 counts; original receipt documents byte discrepancies with later public manifest. Cache recomputation does not certify that later byte binding. Ungated means are not reference-exceedance decisions.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent,
                        help='Artifact root containing evidence/; defaults to this script directory')
    parser.add_argument('--mode', choices=['strict', 'portable'], default='strict',
                        help='strict: original exact-value regression; portable: explicit epsilon-scaled float64 allowlist plus mandatory exact invariants')
    parser.add_argument('--output-dir', type=Path, help='Receipt directory; defaults to artifact root. Mode-specific filenames never overwrite v3 receipts.')
    args = parser.parse_args()
    output_dir = (args.output_dir or args.root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = Audit(args.root, args.mode)
    if not audit.evidence.is_dir():
        parser.error('Artifact root must contain evidence/: ' + str(audit.root))
    sections = []
    db2 = None
    tasks = [('DB2', lambda: verify_db2(audit)), ('models', lambda: verify_models(audit)),
             ('comparators', lambda: verify_comparators(audit)), ('synthetic', lambda: verify_synthetic(audit)),
             ('precision', lambda: verify_precision(audit)), ('boundary_lineage', lambda: verify_boundary_lineage(audit))]
    for name, run in tasks:
        try:
            returned = run()
            if name == 'DB2': db2 = returned
            sections.append({'section': name, 'passed': True})
        except Exception as error:
            audit.checks.append({'check': name + ': section completed', 'passed': False,
                                 'detail': type(error).__name__ + ': ' + str(error)})
            sections.append({'section': name, 'passed': False, 'error': str(error)})
        print(name + ': ' + ('PASS' if sections[-1]['passed'] else 'FAIL'), flush=True)
    if db2 is not None:
        for name, run in [('S39', lambda: verify_s39(audit, db2[3])),
                          ('permutation_figure', lambda: verify_permutation_and_figure(audit, *db2))]:
            try:
                run(); sections.append({'section': name, 'passed': True})
            except Exception as error:
                audit.checks.append({'check': name + ': section completed', 'passed': False,
                                     'detail': type(error).__name__ + ': ' + str(error)})
                sections.append({'section': name, 'passed': False, 'error': str(error)})
            print(name + ': ' + ('PASS' if sections[-1]['passed'] else 'FAIL'), flush=True)
    # Package-relative M3/M5 saved-score checks use the explicit companion adapter.
    for name, function in [('M3', verify_m3), ('M5', verify_m5)]:
        if function is not None:
            try:
                function(audit.root, audit.check, audit.results, recomputed=audit.recomputed, invariant_counts=audit.invariant_counts)
                sections.append({'section': name, 'passed': True})
                # Register only files the follow-up functions document as consumed.
                for path in (audit.evidence / 'followup' / {'M3': 'm3_phase', 'M5': 'm5_seams'}[name]).rglob('*'):
                    if path.is_file() and path.suffix in ['.npz', '.csv', '.json', '.npy']:
                        audit.source(path)
            except Exception as error:
                audit.checks.append({'check': name + ': section completed', 'passed': False,
                                     'detail': type(error).__name__ + ': ' + str(error)})
                sections.append({'section': name, 'passed': False, 'error': str(error)})
            print(name + ': ' + ('PASS' if sections[-1]['passed'] else 'FAIL'), flush=True)
    success = all(r['passed'] for r in sections) and all(r['passed'] for r in audit.checks)
    meta = {'verifier_revision': 'v4p1-controlled-mass-provenance',
            'source_witness_policy': 'Canonical archived source; controlled mass may use an exactly equal recalculated tie only. See PROVENANCE_PATCH.md. Other indices remain strict.',
            'all_passed': success, 'number_of_checks': len(audit.checks), 'sections': sections,
            'acceptance_mode': args.mode, 'numerical_acceptance': audit.numerical.summary(),
            'strict_invariants': audit.invariant_counts,
            'source_witness_summary': {'checked_coordinates': audit.invariant_counts['higher_order_source_witnesses'],
              'same_native_source_indices': audit.invariant_counts['higher_order_source_witnesses'] - audit.invariant_counts['controlled_mass_exact_value_equivalent_ties'],
              'exact_value_equivalent_mass_ties': audit.invariant_counts['controlled_mass_exact_value_equivalent_ties']},
            'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'companion_script_sha256': {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['numerical_acceptance.py','verifier_followup_helpers.py','quantile_provenance.py']},
            'scope': 'Independent saved-result recalculation only; no training, model inference, or remote action.',
            'root_resolution': '__file__.resolve().parent unless --root supplied; all input paths relative to packaged evidence/',
            'required_core': ['original DB2 509/2720', 'fourfold pooled519/2720', 'post-hoc max-family22/2720',
                              'participant-mean0/40 under both policies', '40model classification metrics',
                              'Dynamask32passing same-window AOPC and paired participant-bootstrap CIs']}
    for filename, obj in [(f'verifier_{args.mode}_results.json', dict(meta, results=audit.results)),
                          (f'verifier_{args.mode}_checks.json', dict(meta, checks=audit.checks)),
                          (f'verifier_{args.mode}_sources.json', {'inputs': audit.sources,
                                                    'note': 'Only packaged evidence files read; embedded legacy absolute paths never resolved.'})]:
        (output_dir / filename).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    (output_dir / f'verifier_{args.mode}_numerical_comparisons.json').write_text(json.dumps({'summary': audit.numerical.summary(), 'comparisons': audit.numerical.records}, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'all_passed': success, 'acceptance_mode': args.mode, 'checks': len(audit.checks), 'source_files': len(audit.sources),
                      'numerical_acceptance': audit.numerical.summary(), 'strict_invariants': audit.invariant_counts,
                      'sections': sections}, indent=2), flush=True)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
