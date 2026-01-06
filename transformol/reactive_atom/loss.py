"""
A set of loss functions

Updates:
    18.10.2025 Initial script [Rangsiman Ketkaew]
"""

import torch


def localization_surrogate_J_sur(P):

    return (P**2).sum()


def orthogonality_penalty(P):
    # This is to encourage different orbitals to concentrate on different atoms

    K = P @ P.t()
    offdiag = K - torch.diag(torch.diag(K))

    return (offdiag**2).sum()
