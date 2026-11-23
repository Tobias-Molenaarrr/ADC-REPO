import sys
import os
import pickle
import warnings

# This file provides an interface to a pre-trained politeness SVM.
#
# [Team 6 changes, Nov 2026] Original code by Danescu-Niculescu-Mizil, Sudhof,
# Jurafsky, Leskovec and Potts (Apache 2.0); Python 3 port by B. Meyers.
#   1. sys.stderr.write() was called with TWO arguments (a comma instead of
#      string concatenation), so the "package not found" message itself raised
#      a TypeError. Messages are now single strings.
#   2. nltk is optional: features/vectorizer.py has a small fallback tokenizer.
#   3. The pickled SVM was saved with scikit-learn 0.15-0.18 and cannot be
#      used by scikit-learn 1.x (module moved, private attributes changed).
#      Because the kernel is linear we read the stored weights and apply the
#      stored Platt-scaling parameters ourselves (see LinearPlattSVM).

#### CHECK DEPENDENCIES ########################################################
def _require(module_name, pip_name):
    """Import a module or exit with a readable message."""
    try:
        return __import__(module_name)
    except ImportError:
        sys.stderr.write("Package not found: Politeness model requires "
                         "python package %s\n" % pip_name)
        sys.exit(2)

np = _require("numpy", "numpy")
scipy = _require("scipy", "scipy")
sklearn = _require("sklearn", "scikit-learn")
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.special import expit

#### CHECK DEPENDENCY VERSIONS #################################################
# Versions are compared as integer tuples: the original string comparison
# ("1.9" < "1.10" is False) gives wrong answers for two-digit version parts.
def _version_tuple(v):
    return tuple(int(p) for p in v.split(".")[:3] if p.isdigit())

for _name, _package, _expected in [("scikit-learn", sklearn, "0.18.1"),
                                   ("numpy", np, "1.12.0"),
                                   ("scipy", scipy, "0.18.1")]:
    if _version_tuple(_package.__version__) < _version_tuple(_expected):
        sys.stderr.write("Warning: package '%s', expected version >= %s, "
                         "detected %s. Code functionality not guaranteed.\n"
                         % (_name, _expected, _package.__version__))

#### PACKAGE IMPORTS ###########################################################
from features.vectorizer import PolitenessFeatureVectorizer

#### LOADING THE PRE-TRAINED SVM ###############################################
def _dense(matrix):
    """Return a numpy array whether the stored matrix is sparse or dense."""
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


class _CompatUnpickler(pickle.Unpickler):
    """Unpickle an SVC saved by an old scikit-learn: the class used to live in
    sklearn.svm.classes, which newer versions renamed to sklearn.svm._classes."""
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ImportError:
            if module.startswith("sklearn.svm."):
                return super().find_class("sklearn.svm._classes", name)
            raise


class LinearPlattSVM(object):
    """Minimal stand-in for sklearn.svm.SVC(kernel='linear', probability=True).

    Only the fitted arrays are used from the pickle, never SVC's methods, so it
    does not depend on the scikit-learn version. libsvm's decision value is
        f(x) = x . w + b,   w = sum_i alpha_i y_i x_i   (positive -> classes_[0])
    and its binary probability is Platt scaling
        P(classes_[0]) = 1 / (1 + exp(A*f + B)).
    Here classes_ = [0, 1] = [impolite, polite], so predict_proba returns
    [[P(impolite), P(polite)]] like the original model.
    """
    def __init__(self, svc):
        d = vars(svc)
        # '_dual_coef_' / '_intercept_' hold libsvm's raw (unflipped) values;
        # older pickles only have 'dual_coef_' in the raw convention.
        dual = _dense(d['_dual_coef_'] if '_dual_coef_' in d else d['dual_coef_'])
        support_vectors = _dense(d['support_vectors_'])
        assert list(d['classes_']) == [0, 1], "expected classes [impolite, polite]"
        self.w = (dual @ support_vectors).ravel()
        self.b = float(np.ravel(d['_intercept_'])[0])
        self.A = float(np.ravel(d['probA_'])[0])
        self.B = float(np.ravel(d['probB_'])[0])
        self.n_features = self.w.shape[0]

    def predict_proba(self, X):
        f = np.ravel(X @ self.w + self.b)
        r01 = expit(-(self.A * f + self.B))            # Platt scaling
        r01 = np.clip(r01, 1e-7, 1 - 1e-7)             # libsvm's min_prob
        p_impolite = np.array([_libsvm_two_class_coupling(r) for r in r01])
        return np.column_stack([p_impolite, 1.0 - p_impolite])


def _libsvm_two_class_coupling(r01):
    """Port of libsvm's multiclass_probability() for k = 2 classes.

    libsvm does not return the raw Platt probability for two classes: it runs a
    short iterative refinement that stops once the error is below 0.005 / k, so
    the result differs from the plain sigmoid by up to about 0.004. We repeat
    it so that scores match what scikit-learn's SVC.predict_proba returned.
    """
    k, eps, max_iter = 2, 0.005 / 2, 100
    r = [[0.0, r01], [1.0 - r01, 0.0]]
    Q = [[r[1][0] ** 2, -r[1][0] * r[0][1]],
         [-r[1][0] * r[0][1], r[0][1] ** 2]]
    p = [1.0 / k, 1.0 / k]
    for _ in range(max_iter):
        Qp = [Q[t][0] * p[0] + Q[t][1] * p[1] for t in range(k)]
        pQp = p[0] * Qp[0] + p[1] * Qp[1]
        if max(abs(Qp[t] - pQp) for t in range(k)) < eps:
            break
        for t in range(k):
            diff = (-Qp[t] + pQp) / Q[t][t]
            p[t] += diff
            pQp = (pQp + diff * (diff * Q[t][t] + 2 * Qp[t])) / (1 + diff) / (1 + diff)
            for j in range(k):
                Qp[j] = (Qp[j] + diff * Q[t][j]) / (1 + diff)
                p[j] /= (1 + diff)
    return p[0]


def load_svm(path):
    """Load a pickled politeness SVM (Python 2 era pickles need latin1)."""
    with warnings.catch_warnings():
        # scikit-learn warns that the pickle is from another version. We only
        # read arrays from it, so the warning does not apply.
        warnings.simplefilter("ignore")
        with open(path, 'rb') as handle:
            svc = _CompatUnpickler(handle, encoding='latin1', fix_imports=True).load()
    return LinearPlattSVM(svc)


# Serialized model filename
MODEL_FILENAME = os.path.join(os.path.split(__file__)[0], 'politeness-svm.p')

# Load model, initialize vectorizer
clf = load_svm(MODEL_FILENAME)
vectorizer = PolitenessFeatureVectorizer()

def score(request):
    """
    :param request - The request document to score
    :type request - dict with 'sentences' and 'parses' field
        sample (taken from test_documents.py)--
        {
            'sentences': [
                "Have you found the answer for your question?",
                "If yes would you please share it?"
            ],
            'parses': [
                ["csubj(found-3, Have-1)", "dobj(Have-1, you-2)",
                 "root(ROOT-0, found-3)", "det(answer-5, the-4)",
                 "dobj(found-3, answer-5)", "poss(question-8, your-7)",
                 "prep_for(found-3, question-8)"],
                ["prep_if(would-3, yes-2)", "root(ROOT-0, would-3)",
                 "nsubj(would-3, you-4)", "ccomp(would-3, please-5)",
                 "nsubj(it-7, share-6)", "xcomp(please-5, it-7)"]
            ]
        }

    returns class probabilities as a dict
        { 'polite': float, 'impolite': float }
    """
    # Vectorizer returns {feature-name: value} dict
    features = vectorizer.features(request)
    fv = [features[f] for f in sorted(features.keys())]
    # Single-row sparse matrix
    X = csr_matrix(np.asarray([fv]))
    probs = clf.predict_proba(X)
    # Massage return format
    probs = {"polite": probs[0][1], "impolite": probs[0][0]}
    return probs


if __name__ == "__main__":
    # Sample classification of requests
    args = sys.argv[1:]
    if len(args) != 1:
        sys.exit(1)
    else:
        from scripts.format_input import format_doc
        doc_text = ""
        with open(args[0], "r") as doc:
            doc_text = doc.read()
        parsed_docs = format_doc(doc_text)
        polite = []
        impolite = []
        for i, doc in enumerate(parsed_docs):
            probs = score(doc)
            polite.append(probs['polite'])
            impolite.append(probs['impolite'])
            print("\n====\nSentence " + str(i) + ":\n" + str(doc['sentences'][0]))
            print("\tP(polite) = %.3f" % probs['polite'])
            print("\tP(impolite) = %.3f" % probs['impolite'])

        print("\n====\nDocument:")
        print("\tP(polite) = %.3f" % np.mean(polite))
        print("\tP(impolite) = %.3f" % np.mean(impolite))

#    from test_documents import TEST_DOCUMENTS

#    for doc in TEST_DOCUMENTS:

#        probs = score(doc)

#        print("====================")
#        print("Text: ", doc['text'])
#        print("\tP(polite) = %.3f" % probs['polite'])
#        print("\tP(impolite) = %.3f" % probs['impolite'])
#        print("\n")


