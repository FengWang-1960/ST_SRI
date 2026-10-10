"""v4p1 controlled-mass source witness: exact ties, canonical provenance.

No changed lag, smoothing, statistic, threshold, saved data, or gate predicate.
This is explicitly a revision of source-witness semantics, not original v4 PASS.
The canonical source is ALWAYS rederived from the hash-verified stored rows.
Only a native mass-source difference that selects exactly the same archived
threshold value from an exact recomputed tie is admissible. Height stays exact.
"""
import math
import numpy as np
from numerical_acceptance import stable_rank


def controlled_source_witness(recomputed, saved):
    """Return (accepted, detail) for aligned M4 calibration mass/height rows.

    Alignment/identity comes from the caller's hash-verified frozen inputs and
    keyed calibration CSV. The new check adds a gamma_512 bound for EVERY
    calibration scalar; a source exception requires exact, not near, equality.
    """
    a, b = np.asarray(recomputed), np.asarray(saved)
    detail = {'policy': 'controlled-mass-exact-value-tie-v1',
              'accepted': False, 'coordinates': [], 'equivalent_ties': 0,
              'canonical_source_authority': 'hash-verified archived calibration rows'}
    if (a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or len(a) == 0
            or a.dtype != np.dtype('float64') or b.dtype != np.dtype('float64')):
        detail['reason'] = 'shape, empty data, or float64 dtype mismatch'
        return False, detail
    if not (np.isfinite(a).all() and np.isfinite(b).all() and (a >= 0).all() and (b >= 0).all()):
        detail['reason'] = 'nonfinite or negative calibration statistic'
        return False, detail
    eps = np.finfo(np.float64).eps
    gamma = (512 * eps) / (1 - 512 * eps)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        error = np.abs(a-b)
        bound = gamma*np.maximum(a,b)
    if (not np.isfinite(error).all() or not np.isfinite(bound).all()
            or not np.array_equal(a == 0, b == 0) or not np.all(error <= bound)):
        detail['reason'] = 'calibration scalar changed beyond existing operation budget or zero mask'
        return False, detail
    k = math.ceil((len(a)-1)*.95)
    detail.update(rank_index=k, calibration_rows=len(a), maximum_absolute_error=float(error.max()))
    accepted = True
    for column, name in enumerate(('mass','height')):
        ia = int(stable_rank(a[:,column])[k])
        ib = int(stable_rank(b[:,column])[k])
        same = ia == ib
        # Strong equality: both recomputed sources AND the archived threshold
        # scalar must be identical. No tolerance is applied to this predicate.
        exact_tie = bool(column == 0 and not same
                         and a[ia,column] == a[ib,column] == b[ib,column])
        passed = same or exact_tie
        detail['coordinates'].append({'coordinate':name, 'canonical_source_index':ib,
            'native_recomputed_source_index':ia, 'exact_same_source':same,
            'exact_value_equivalent_tie':exact_tie, 'accepted':bool(passed),
            'canonical_threshold':float(b[ib,column]), 'recomputed_threshold':float(a[ia,column]),
            'recomputed_at_canonical_source':float(a[ib,column])})
        detail['equivalent_ties'] += int(exact_tie)
        accepted = accepted and passed
    detail['accepted'] = bool(accepted)
    return bool(accepted), detail
