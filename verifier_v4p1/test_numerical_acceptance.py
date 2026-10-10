#!/usr/bin/env python3
"""Actual isolated fault injection; never changes packaged scientific evidence."""
import argparse
import csv
import hashlib
import json
import platform
import tempfile
from pathlib import Path
import numpy as np
import scipy
from numerical_acceptance import (NumericalAcceptance, EPS64, exact_identity,
                                  same_decisions, same_argmax, same_rank, same_higher_source)
from verify_saved_results import Audit, support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'verification/numerical_fault_test_receipt.json')
    args = parser.parse_args()
    tests = []
    def check(name, passed):
        tests.append({'test': name, 'passed': bool(passed)})
        if not passed:
            raise AssertionError(name)
    def rejected(name, function):
        try:
            function()
        except (AssertionError, ValueError):
            check(name, True)
        else:
            check(name, False)
    smooth = 'M3/fault/positive_smoothed_recomputed'
    profile = 'M5/fault/hard/profile_recomputed'
    stat = 'DB2 raw-curve statistics match saved window_trained_statistics'
    a = np.array([.125, 1., 12.], dtype=np.float64)
    ulp = np.nextafter(a, np.inf)
    p = NumericalAcceptance('portable')
    check('last-bit positive smoothing perturbation accepted', p.compare(smooth, ulp, a))
    check('strict original exact-value predicate rejects same perturbation', not NumericalAcceptance('strict').compare(smooth, ulp, a))
    check('tiny-valued last-bit perturbation accepted without absolute floor', p.compare(smooth, np.nextafter(np.array([1e-20]), np.inf), np.array([1e-20])))
    check('record reports actual dtype shape maxerror bound byte identity', all(key in p.records[-1] for key in ['dtype_recomputed','dtype_saved','shape_recomputed','shape_saved','max_absolute_error','maximum_element_bound','bitwise_equal','original_array_equal']))
    check('budget-exceeding relative perturbation rejected', not p.compare(smooth, a*(1+512*EPS64), a))
    check('zero to minimum positive subnormal rejected', not p.compare(smooth, np.array([np.nextafter(0., 1.)]), np.array([0.])))
    check('float32 recomputation rejected', not p.compare(smooth, a.astype(np.float32), a.astype(np.float32)))
    check('different dtype rejected', not p.compare(smooth, a, a.astype(np.float32)))
    check('different shape rejected', not p.compare(smooth, a.reshape(3,1), a))
    check('changed NaN pattern rejected', not p.compare(profile, np.array([np.nan, np.nan]), np.array([np.nan, 1.])))
    check('partial undefined M5 profile rejected even with matching NaNs', not p.compare(profile, np.array([np.nan, 1.]), np.array([np.nan, 1.])))
    check('whole undefined M5 profile retained', p.compare(profile, np.full(3,np.nan), np.full(3,np.nan)))
    check('unexpected M3 NaN rejected', not p.compare(smooth, np.full(3,np.nan), np.full(3,np.nan)))
    check('matching infinity rejected', not p.compare(smooth, np.array([np.inf]), np.array([np.inf])))
    check('overflowing difference rejected', not p.compare(smooth, np.array([np.finfo(float).max]), np.array([-np.finfo(float).max])))
    check('negative output outside nonnegative operation rejected', not p.compare(smooth, np.array([-1.]), np.array([-1.])))
    rejected('unknown operation cannot opt in to numeric tolerance', lambda: p.compare('unknown_scope', a, a))
    check('original ±zero predicate semantics preserved', NumericalAcceptance('strict').compare(smooth,np.array([-0.]),np.array([0.])))
    z = NumericalAcceptance('strict');z.compare(smooth,np.array([-0.]),np.array([0.]))
    check('byte identity distinguished from original numeric predicate', z.records[0]['original_array_equal'] and not z.records[0]['bitwise_equal'])
    # A perturbation that passes the numeric budget must still fail if it moves
    # any scientific threshold decision, maximum index or deterministic rank.
    saved_st = np.array([[1.,1.]], dtype=float)
    changed_st = np.nextafter(saved_st, np.inf)
    check('small threshold-sensitive statistic meets numeric budget', p.compare(stat, changed_st, saved_st))
    check('threshold decision change rejected independently', not same_decisions(support(changed_st,np.array([1.,1.])), support(saved_st,np.array([1.,1.]))))
    saved_peak = np.array([1.,1.]);changed_peak=np.array([1.,np.nextafter(1.,np.inf)])
    check('near-tie maximum perturbation meets numeric budget', p.compare(smooth,changed_peak,saved_peak))
    check('argmax change rejected independently', not same_argmax(changed_peak,saved_peak))
    changed_rank=np.array([np.nextafter(1.,np.inf),1.])
    check('near-tie rank perturbation meets numeric budget', p.compare(smooth,changed_rank,saved_peak))
    check('deterministic rank change rejected independently', not same_rank(changed_rank,saved_peak,np.array(['A','B'])))
    saved_cal=np.ones((3,2));changed_cal=saved_cal.copy();changed_cal[0]=np.nextafter(changed_cal[0],np.inf)
    check('higher95 selected source index change rejected', not same_higher_source(changed_cal,saved_cal))
    check('changed participant/window identity rejected', not exact_identity(np.array(['S01_windowA']),np.array(['S01_windowB'])))
    check('changed bootstrap draw rejected', not exact_identity(np.array([[1,2,3]]),np.array([[1,2,4]])))
    check('changed label rejected', not exact_identity(np.array([0,1]),np.array([0,2])))
    check('changed lag-axis rejected', not exact_identity(np.array([5.,5.5]),np.array([5.,5.6])))
    # Exercise the actual Audit assertion and frozen-hash path, not a mock of
    # its predicate. The isolated minimal package lives only in a temp folder.
    with tempfile.TemporaryDirectory(prefix='bspc_v4_fault_') as directory:
        root=Path(directory); evidence=root/'evidence'; evidence.mkdir()
        legacy=root/'verification/legacy_v3'; legacy.mkdir(parents=True)
        specimen=evidence/'specimen.bin'; specimen.write_bytes(b'unchanged-scientific-input')
        digest=hashlib.sha256(specimen.read_bytes()).hexdigest()
        with (legacy/'MANIFEST_SHA256.csv').open('w',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=['path','bytes','sha256']);writer.writeheader();writer.writerow({'path':'evidence/specimen.bin','bytes':specimen.stat().st_size,'sha256':digest})
        audit=Audit(root,'portable')
        check('actual Audit frozen evidence hash accepts unchanged source', audit.source(specimen) == specimen.resolve())
        rejected('actual Audit rejects changed gate vector', lambda: audit.decisions('fault threshold decision',np.array([True]),np.array([False])))
        rejected('actual Audit rejects exchanged predicates despite identical total count',lambda: audit.decisions('fault count cancellation',np.array([True,False]),np.array([False,True])))
        rejected('actual Audit rejects nonbool discrete decision vector',lambda: audit.decisions('fault dtype',np.array([1],dtype=np.int64),np.array([True])))
        rejected('actual Audit rejects changed higher95 calibration source index',lambda: audit.higher_source('fault higher95',changed_cal,saved_cal))
        specimen.write_bytes(b'changed-scientific-input')
        rejected('actual Audit rejects changed input bytes despite portable mode', lambda: Audit(root,'portable').source(specimen))
        rejected('actual Audit rejects changed ID',lambda: audit.check('fault ID',exact_identity(np.array(['S01']),np.array(['S02']))))
        rejected('actual Audit rejects changed maximum index',lambda: audit.check('fault argmax',same_argmax(changed_peak,saved_peak)))
        rejected('actual Audit rejects changed rank',lambda: audit.check('fault rank',same_rank(changed_rank,saved_peak)))
    receipt={'all_passed':all(t['passed'] for t in tests),'number_of_tests':len(tests),'tests':tests,
             'environment':{'platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
             'scope':'Actual isolated fault injection of bounded comparison and mandatory strict guards; not training/inference and not a Linux portability certificate.',
             'policy_sha256':hashlib.sha256((Path(__file__).parent/'numerical_acceptance.py').read_bytes()).hexdigest(),
             'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'all_passed':receipt['all_passed'],'number_of_tests':len(tests),'receipt':str(args.output.resolve())},indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
