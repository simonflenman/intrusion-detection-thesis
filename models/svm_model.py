from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC

def build_svm(
    kernel: str = "rbf",
    gamma: float = 0.1,
    n_components: int = 300,
    C: float = 1.0,
    max_iter: int = 10000,
    random_state: int = 42,
) -> Pipeline:
    """
    Returns a Pipeline that maps the features into an approximate RBF space
    via Nystroem, then fits a *quiet* LinearSVC on top.
    """
    return Pipeline([
        ("nystroem", Nystroem(
            kernel=kernel,
            gamma=gamma,
            n_components=n_components,
            random_state=random_state
        )),
        ("svc", LinearSVC(
            C=C,
            dual=False,
            max_iter=max_iter,
            verbose=False,      # <-- turn off solver logs
            random_state=random_state
        ))
    ])
