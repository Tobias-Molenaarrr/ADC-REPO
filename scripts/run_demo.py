"""
Team 6 (WG2) demo: score the four sample requests in test_documents.py with the
two sets of pre-trained model files and print what the code finds.

    python scripts/run_demo.py            (run from the repository root)

The sample requests already contain dependency parses, so no Stanford CoreNLP
server is needed. The two model sets are:
  models/master          the model files from the original master branch
  models/python3_branch  the model files shipped on the python3 branch
"""
import os
import sys
import _pickle

import numpy as np
from scipy.sparse import csr_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import model                                         # noqa: E402  (our fixed loader)
from features.vectorizer import PolitenessFeatureVectorizer   # noqa: E402
from request_utils import check_is_request           # noqa: E402
from test_documents import TEST_DOCUMENTS            # noqa: E402


def load_list(path):
    with open(path, "rb") as handle:
        return _pickle.load(handle, encoding="latin1", fix_imports=True)


def build(model_name):
    """Return (vectorizer, svm) for one set of model files."""
    folder = os.path.join(ROOT, "models", model_name)
    vectorizer = PolitenessFeatureVectorizer()
    vectorizer.unigrams = load_list(os.path.join(folder, "featunigrams.p"))
    vectorizer.bigrams = load_list(os.path.join(folder, "featbigrams.p"))
    svm = model.load_svm(os.path.join(folder, "politeness-svm.p"))
    return vectorizer, svm


def p_polite(vectorizer, svm, doc):
    features = vectorizer.features(dict(doc))        # copy: features() adds keys
    vector = [features[name] for name in sorted(features)]
    assert len(vector) == svm.n_features, (len(vector), svm.n_features)
    return svm.predict_proba(csr_matrix(np.asarray([vector])))[0][1], features


def main():
    models = {name: build(name) for name in ("master", "python3_branch")}
    print("Feature counts: " + ", ".join(
        "%s = %d unigrams + %d bigrams + 21 strategies = %d (SVM expects %d)" % (
            n, len(v.unigrams), len(v.bigrams),
            len(v.unigrams) + len(v.bigrams) + 21, s.n_features)
        for n, (v, s) in models.items()))
    print()
    print("%-4s %-13s %-8s %-18s %-18s" % (
        "Doc", "Human score", "Request?", "P(polite) master", "P(polite) python3"))
    details = []
    for i, doc in enumerate(TEST_DOCUMENTS, 1):
        row, feats = [], None
        for name, (vectorizer, svm) in models.items():
            p, f = p_polite(vectorizer, svm, doc)
            row.append(p)
            if name == "master":
                feats = f
        print("%-4d %+-13.1f %-8s %-18.3f %-18.3f" % (
            i, doc["score"], check_is_request(doc), row[0], row[1]))
        details.append((i, doc["text"], feats))
    print()
    for i, text, feats in details:
        fired = sorted(k.replace("feature_politeness_==", "").rstrip("=")
                       for k, v in feats.items()
                       if v and k.startswith("feature_politeness_"))
        n_ngrams = sum(1 for k, v in feats.items()
                       if v and (k.startswith("UNIGRAM_") or k.startswith("BIGRAM_")))
        print("Doc %d: %s" % (i, text[:70] + ("..." if len(text) > 70 else "")))
        print("   strategies detected: %s" % (", ".join(fired) or "none"))
        print("   unigram/bigram features present: %d" % n_ngrams)


if __name__ == "__main__":
    main()
