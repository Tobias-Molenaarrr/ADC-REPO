"""
Team 6 (WG2) check: does LinearPlattSVM (model.py) reproduce scikit-learn's own
predict_proba? We train a fresh linear SVC with probability=True on random
sparse binary data (the same kind of input the politeness model uses), wrap its
fitted arrays the way the old pickles store them, and compare the outputs.

    python scripts/check_linear_platt.py        (run from the repository root)
"""
import os
import sys
import types

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import LinearPlattSVM                      # noqa: E402

rng = np.random.RandomState(0)
X = csr_matrix((rng.rand(400, 60) < 0.15).astype(float))
y = (np.asarray(X[:, :5].sum(axis=1)).ravel() + rng.rand(400) * 1.5 > 1.0).astype(int)

svc = SVC(C=0.02, kernel="linear", probability=True, random_state=0).fit(X, y)

# Present the fitted state in the layout of an old pickle (attribute names
# 'probA_' / 'probB_' instead of the private '_probA' / '_probB').
old_style = types.SimpleNamespace(
    classes_=svc.classes_, _dual_coef_=svc._dual_coef_,
    support_vectors_=svc.support_vectors_, _intercept_=svc._intercept_,
    probA_=svc._probA, probB_=svc._probB)
ours = LinearPlattSVM(old_style)

expected = svc.predict_proba(X)
actual = ours.predict_proba(X)
print("max |difference| over %d rows: %.2e" % (len(y), np.abs(expected - actual).max()))
assert np.allclose(expected, actual, atol=1e-6), "LinearPlattSVM disagrees with sklearn"
print("OK: LinearPlattSVM matches sklearn %s predict_proba" %
      __import__("sklearn").__version__)
