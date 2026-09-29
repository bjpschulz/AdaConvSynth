import torch


def inner_product(t1, t2):

    if torch.is_complex(t1):

        innerp = torch.sum(t1.flatten() * t2.flatten().conj())
    else:
        innerp = torch.sum(t1.flatten() * t2.flatten())
    return innerp


def power_iteration(operator, q0, niter=128):
    """
    Power iteration to estimate the operator norm (largest singular value) of a linear operator A.
    It estimates sqrt(lambda_max(A^H A)).

    Args:
        operator: Function that represents application of A.
        q0: Initial vector (torch tensor) for iteration, should be normalized.
        niter: Number of power iterations.

    Returns:
        Approximate largest singular value of operator.
    """
    # Ensure no gradients are tracked
    with torch.no_grad():
        qk = q0 / torch.linalg.norm(q0)  # Normalize the initial vector
        for _ in range(niter):
            AtAqk = operator(qk)
            qk = AtAqk / torch.linalg.norm(AtAqk)  # Re-normalize

        # Compute the final norm estimation
        op_norm = torch.sqrt(inner_product(operator(qk), operator(qk)).real).cpu()

    return op_norm


def conj_grad(H, b, x, niter=4):
    r = H(x)
    r = b - r
    p = r.clone()
    sqnorm_r_old = inner_product(r, r)

    for _ in range(niter):
        d = H(p)
        inner_p_d = inner_product(p, d)
        alpha = sqnorm_r_old / inner_p_d
        x = x + alpha * p
        r = r - alpha * d
        sqnorm_r_new = inner_product(r, r)
        beta = sqnorm_r_new / sqnorm_r_old
        sqnorm_r_old = sqnorm_r_new
        p = r + beta * p
    return x
