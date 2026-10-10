"""Restricted float64 recomputation policy; identity and decisions stay exact.

No global replacement of numpy predicates is performed. Callers must opt in
at named float64 smoothing/statistic/profile operations. Strict is the default.
"""
import re
import numpy as np

EPS64 = np.finfo(np.float64).eps
# Exact semantic allowlist, including finite operations that happened to agree
# in the external run. Budgets are fixed operation counts, not fitted errors.
RULES = (
    (re.compile(r'^DB2 original pooled gates reconstructed$'), 512, False, 'nonnegative Gaussian statistic and higher order statistic'),
    (re.compile(r'^DB2 raw-curve statistics match saved (window_reference_statistics|window_trained_statistics|participant_reference_statistics|participant_trained_statistics)$'), 512, False, 'nonnegative Gaussian, sum291 and at most mean68'),
    (re.compile(r'^DB2/(pooled|max_family_post_hoc)/(window|participant_mean)/fold[0-3]: independently sorted gates$'), 512, False, 'reordered previously computed nonnegative statistics'),
    (re.compile(r'^20261004[1-8]/(original_model_argmax|fixed_class1)/(window|simulated_group_mean): null calibration gates$'), 512, False, 'nonnegative Gaussian, sum291 and optional mean8'),
    (re.compile(r'^M3/.+/positive_smoothed_recomputed$'), 64, False, '17-tap positive Gaussian convolution'),
    (re.compile(r'^M5/.+/(hard|taper)/profile_recomputed$'), 512, True, '17-tap nonnegative Gaussian, sum291, normalized division'),
    (re.compile(r'^Representative[ABCD]: plotted processing independently reconstructed$'), 64, False, '17-tap nonnegative Gaussian convolution'),
)


def exact_identity(left, right, equal_nan=False):
    """The original strict array predicate, unchanged."""
    return bool(np.array_equal(left, right, equal_nan=equal_nan))


def same_decisions(left, right):
    a, b = np.asarray(left), np.asarray(right)
    return a.dtype == np.dtype(bool) and b.dtype == np.dtype(bool) and exact_identity(a, b)


def same_argmax(left, right):
    a, b = np.asarray(left), np.asarray(right)
    return a.shape == b.shape and int(np.argmax(a)) == int(np.argmax(b))


def stable_rank(values, identities=None):
    """Ascending full order; exact ties resolve by frozen identity/index."""
    values = np.asarray(values)
    ident = np.arange(values.size) if identities is None else np.asarray(identities)
    return np.lexsort((ident, values))


def same_rank(left, right, identities=None):
    a, b = np.asarray(left), np.asarray(right)
    return a.shape == b.shape and exact_identity(stable_rank(a, identities), stable_rank(b, identities))


def same_higher_source(left, right, q=.95):
    """Strict source-index witness for each higher-order-statistic coordinate.

    Uses frozen flattened input index as the secondary key for exact ties.
    This compares the selected source index, not every rank in every DB2 row.
    """
    a, b = np.asarray(left).reshape(-1, 2), np.asarray(right).reshape(-1, 2)
    if a.shape != b.shape:
        return False
    index = int(np.ceil((len(a)-1)*q))
    return all(int(stable_rank(a[:, column])[index]) == int(stable_rank(b[:, column])[index]) for column in range(2))


class NumericalAcceptance:
    def __init__(self, mode='strict'):
        if mode not in ('strict', 'portable'):
            raise ValueError('Unknown acceptance mode: ' + mode)
        self.mode = mode
        self.records = []

    def compare(self, name, left, right):
        rule = next((item for item in RULES if item[0].fullmatch(name)), None)
        if rule is None:
            raise ValueError('Recomputation operation is not whitelisted: ' + name)
        _, budget, allow_empty_profile, reason = rule
        a, b = np.asarray(left), np.asarray(right)
        exact_value_equal = exact_identity(a, b, equal_nan=allow_empty_profile)
        bitwise_equal = bool(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes())
        row = {'check': name, 'mode': self.mode, 'dtype_recomputed': str(a.dtype),
               'dtype_saved': str(b.dtype), 'shape_recomputed': list(a.shape),
               'shape_saved': list(b.shape), 'operation_budget': budget,
               'gamma': float(budget*EPS64/(1-budget*EPS64)), 'reason': reason,
               'bitwise_equal': bitwise_equal, 'original_array_equal': exact_value_equal, 'max_absolute_error': None,
               'maximum_element_bound': None, 'maximum_error_to_bound_ratio': None,
               'finite_elements': 0, 'accepted': False, 'failure_reason': None}
        compatible = a.shape == b.shape and a.dtype == np.dtype('float64') and b.dtype == np.dtype('float64')
        if not compatible:
            row['failure_reason'] = 'shape or float64 dtype mismatch'
        else:
            nan_a, nan_b = np.isnan(a), np.isnan(b)
            inf_a, inf_b = np.isinf(a), np.isinf(b)
            # Only a whole undefined M5 normalized profile may contain NaNs.
            nonfinite_ok = not np.any(inf_a | inf_b) and exact_identity(nan_a, nan_b)
            if np.any(nan_a):
                nonfinite_ok = bool(nonfinite_ok and allow_empty_profile and nan_a.all())
            if not nonfinite_ok:
                row['failure_reason'] = 'unexpected nonfinite values or changed NaN/Inf pattern'
            else:
                finite = np.isfinite(a) & np.isfinite(b)
                av, bv = a[finite], b[finite]
                with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                    difference = np.abs(av-bv)
                    scale = np.maximum(np.abs(av), np.abs(bv))
                    bound = row['gamma'] * scale
                finite_differences = bool(np.isfinite(difference).all() and np.isfinite(bound).all() and (av >= 0).all() and (bv >= 0).all())
                zeros_agree = bool(np.array_equal(av == 0, bv == 0))
                within = bool(finite_differences and zeros_agree and np.all(difference <= bound))
                ratio = np.divide(difference, bound, out=np.zeros_like(difference), where=bound > 0)
                if np.any((bound == 0) & (difference != 0)):
                    ratio[(bound == 0) & (difference != 0)] = np.inf
                row.update(finite_elements=int(finite.sum()),
                           max_absolute_error=float(difference.max()) if difference.size else 0.0,
                           maximum_element_bound=float(bound.max()) if bound.size else 0.0,
                           maximum_error_to_bound_ratio=float(ratio.max()) if ratio.size and np.isfinite(ratio.max()) else (None if ratio.size else 0.0))
                row['accepted'] = bool(exact_value_equal if self.mode == 'strict' else within)
                if not row['accepted']:
                    row['failure_reason'] = 'strict equality failed' if self.mode == 'strict' else 'epsilon budget exceeded, zero changed, or nonfinite difference'
        self.records.append(row)
        return row['accepted']

    def summary(self):
        rows = self.records
        return {'mode': self.mode, 'comparisons': len(rows),
                'bitwise_equal': sum(r['bitwise_equal'] for r in rows),
                'original_array_equal': sum(r['original_array_equal'] for r in rows),
                'accepted_nonbitwise': sum(r['accepted'] and not r['bitwise_equal'] for r in rows),
                'accepted_nonexact_value': sum(r['accepted'] and not r['original_array_equal'] for r in rows),
                'rejected': sum(not r['accepted'] for r in rows),
                'max_absolute_error': max((r['max_absolute_error'] or 0 for r in rows), default=0),
                'policy': 'Explicit float64 allowlist; gamma_n times elementwise max magnitude; no absolute error floor; strict discrete invariants are separate mandatory checks.'}
