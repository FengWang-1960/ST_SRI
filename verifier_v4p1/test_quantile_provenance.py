"""Tests for a versioned exact-value tie witness, not a tolerance on IDs."""
from pathlib import Path
import sys,json,unittest
import numpy as np
import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
W=Path(__file__).resolve().parent;P=W;sys.path.insert(0,str(P))
from numerical_acceptance import same_higher_source
try:
 from quantile_provenance import controlled_source_witness
except ImportError:
 def controlled_source_witness(a,b): return same_higher_source(a,b),{'policy':'original-exact-index'}
ROOT=W/'fixtures'
class Tests(unittest.TestCase):
 def load(self,n):
  with np.load(ROOT/f'witness_{n}.npz',allow_pickle=False) as z:return z['recomputed'].copy(),z['saved'].copy()
 def test_first_record_exact_value_tie(self):
  a,b=self.load(1);self.assertFalse(same_higher_source(a,b));self.assertTrue(controlled_source_witness(a,b)[0])
 def test_second_record_exact_value_tie(self):
  a,b=self.load(2);self.assertFalse(same_higher_source(a,b));self.assertTrue(controlled_source_witness(a,b)[0])
 def test_unchanged_records(self):
  a,b=self.load(1);self.assertTrue(controlled_source_witness(b,b)[0])
 def test_wrong_shape(self):
  a,b=self.load(1);self.assertFalse(controlled_source_witness(a[:-1],b)[0])
 def test_wrong_dtype(self):
  a,b=self.load(1);self.assertFalse(controlled_source_witness(a.astype('float32'),b)[0])
 def test_unrecorded_threshold_change(self):
  a,b=self.load(1);a[[86,102],0]=np.nextafter(a[[86,102],0],-np.inf);self.assertFalse(controlled_source_witness(a,b)[0])
 def test_archived_source_outside_exact_tie(self):
  a,b=self.load(1);a[102,0]=np.nextafter(a[102,0],np.inf);self.assertFalse(controlled_source_witness(a,b)[0])
 def test_height_source_exchange_not_authorized(self):
  a,b=self.load(1);k=int(np.ceil((len(a)-1)*.95));from numerical_acceptance import stable_rank
  j=int(stable_rank(a[:,1])[k]);a[j,1]=0;self.assertFalse(controlled_source_witness(a,b)[0])
 def test_overbudget_other_record(self):
  a,b=self.load(1);idx=int(np.flatnonzero(a[:,0]>0)[0]);a[idx,0]*=1.001;self.assertFalse(controlled_source_witness(a,b)[0])
 def test_zero_to_nonzero(self):
  a,b=self.load(1);a[0,0]=np.nextafter(0.,1.);b[0,0]=0.;self.assertFalse(controlled_source_witness(a,b)[0])
 def test_nonfinite(self):
  a,b=self.load(1);a[0,0]=np.nan;self.assertFalse(controlled_source_witness(a,b)[0])
 def test_inputs_not_modified(self):
  a,b=self.load(1);x=a.copy();y=b.copy();controlled_source_witness(a,b);np.testing.assert_array_equal(a,x);np.testing.assert_array_equal(b,y)
 def test_negative(self):
  a,b=self.load(1);a[0,0]=-1;self.assertFalse(controlled_source_witness(a,b)[0])
 def test_global_old_predicate_unchanged(self):
  a,b=self.load(1);self.assertFalse(same_higher_source(a,b))
 def test_ordinary_distinct_source_not_accepted(self):
  a=np.arange(40,dtype=float).reshape(20,2)+1;b=a.copy();a[[18,19]]=a[[19,18]];self.assertFalse(controlled_source_witness(a,b)[0])
 def test_canonical_id_preserved(self):
  a,b=self.load(1);ok,detail=controlled_source_witness(a,b);self.assertTrue(ok);self.assertEqual(detail['coordinates'][0]['canonical_source_index'],102);self.assertEqual(detail['coordinates'][0]['native_recomputed_source_index'],86)
 def test_production_controlled_scope(self):
  from verify_saved_results import Audit
  x=Audit(args.root,'portable');a,b=self.load(1);x.higher_source('202610043/original_model_argmax/window: null calibration gates/higher95 selected source indices',a,b);self.assertEqual(x.invariant_counts['controlled_mass_exact_value_equivalent_ties'],1)
 def test_production_strict_still_rejects(self):
  from verify_saved_results import Audit
  x=Audit(args.root,'strict');a,b=self.load(1)
  with self.assertRaises(AssertionError):x.higher_source('202610043/original_model_argmax/window: null calibration gates/higher95 selected source indices',a,b)
 def test_production_db2_still_rejects(self):
  from verify_saved_results import Audit
  x=Audit(args.root,'portable');a,b=self.load(1)
  with self.assertRaises(AssertionError):x.higher_source('DB2 original higher95 selected source indices',a,b)
 def test_production_unknown_scope_still_rejects(self):
  from verify_saved_results import Audit
  x=Audit(args.root,'portable');a,b=self.load(1)
  with self.assertRaises(AssertionError):x.higher_source('other higher95 selected source indices',a,b)
if __name__=='__main__':
 out=args.output;out.parent.mkdir(parents=True,exist_ok=True);sys.argv=sys.argv[:1]
 class Result(unittest.TextTestResult):
  def __init__(self,*a,**k):super().__init__(*a,**k);self.rows=[]
  def addSuccess(self,test):super().addSuccess(test);self.rows.append({'test':test.id(),'passed':True})
  def addFailure(self,test,err):super().addFailure(test,err);self.rows.append({'test':test.id(),'passed':False,'error':self._exc_info_to_string(err,test)})
  def addError(self,test,err):super().addError(test,err);self.rows.append({'test':test.id(),'passed':False,'error':self._exc_info_to_string(err,test)})
 res=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));out.write_text(json.dumps({'tests':res.testsRun,'passed':res.wasSuccessful(),'results':res.rows},indent=2));sys.exit(not res.wasSuccessful())
