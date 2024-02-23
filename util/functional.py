import torch

def instance_norm(X, dim, Y=None):
    mu = X.mean(dim, keepdims=True)
    X = X - mu
    std = ((X ** 2).mean(dim, keepdims=True) + 1e-8) ** 0.5
    if Y is None:
        return X / std
    else:
        return X / std, (Y - mu) / std


def ridge_regression(X, Y, lamda=0, bias=True):
    if bias:
        X = torch.cat([torch.ones(*X.shape[:-1], 1, device=X.device), X], dim=-1)
    X_t = X.transpose(-2, -1)

    n_samples, n_dim = X.shape[-2:]
    try:
        if n_samples >= n_dim:
            # standard
            A = X_t @ X
            A.diagonal(dim1=-2, dim2=-1).add_(lamda)
            B = X_t @ Y
            weights = torch.linalg.solve(A, B)
        else:
            # Woodbury
            A = X @ X_t
            A.diagonal(dim1=-2, dim2=-1).add_(lamda)
            weights = X_t @ torch.linalg.solve(A, Y)
    except torch.linalg.LinAlgError:
        print("The matrix is singular. Using pseudoinverse to solve.")
        assert lamda == 0
        weights = torch.linalg.lstsq(X, Y).solution
    return weights


def get_concept(X, Y, norm, penalty, bias):
    with torch.no_grad():
        mean = X.mean(-1, keepdims=True)
        var = ((X - mean) ** 2).mean(-1, keepdims=True)
        mask = (var > 1e-5).reshape(-1, 1)
        std = (var + 1e-5) ** 0.5
        if norm == 'last':
            X, Y = X - X[..., [-1]], Y - X[..., [-1]]
        elif norm == 'instance':
            X, Y = (X - mean) / std, (Y - mean) / std
            # X, Y = instance_norm(X, -1, Y)

        X, Y = X.reshape(-1, X.shape[-1]) * mask, Y.reshape(-1, Y.shape[-1]) * mask
        concept_YX = ridge_regression(X, Y, lamda=penalty, bias=bias).transpose(-1, -2)
    return concept_YX

