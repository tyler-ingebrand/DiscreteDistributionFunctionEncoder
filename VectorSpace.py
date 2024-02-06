from typing import Union

import torch

# see this:
# Isometric Logratio Transformations for Compositional Data Analysis
# J. J. Egozcue, V. Pawlowsky-Glahn,  G. Mateu-Figueras, and C. Barcel ́o-Vidal
# particularly pages 3 and 4


# note it is about 20x faster to operate in logit space, and its more numerically stable
# this is because logits space does addition and multiplication, rather than multiplication and exponentiation, respectively.
# these operations are much faster.

def allclose(p_1:torch.tensor, p_2:torch.tensor) -> bool:
    if len(p_1.shape) == 0:
        return torch.allclose(p_1, p_2)
    else:    # convert to probs from logits
        p_1 = torch.exp(p_1) / torch.sum(torch.exp(p_1), dim=-1, keepdim=True)
        p_2 = torch.exp(p_2) / torch.sum(torch.exp(p_2), dim=-1, keepdim=True)
        return torch.allclose(p_1, p_2)


# discrete probability distributions dpd_addition
# think of it as how likely are the two distributions to agree on a particular outcome
def dpd_add(p_1:torch.tensor, p_2:torch.tensor, epsilon=1e-5) -> torch.tensor:
    # logit space
    return p_1 + p_2

    # prob space
    # p_3 = p_1 * p_2
    # return p_3 / (torch.sum(p_3, dim=-1, keepdim=True) + epsilon)

# discrete probability distributions scalar dpd_multiplyplication
# multiplying by values >1 increases the likelihood of the most likely outcome
# whereas multiplying by values <1 increases the likelihood of the least likely outcome
def dpd_multiply(p_1:torch.tensor, scalar:Union[int, float, torch.tensor], epsilon=1e-5) -> torch.tensor:
    # logit space
    return p_1 * scalar

    # prob space
    # p_2 = p_1 ** scalar
    # return p_2 / (torch.sum(p_2, dim=-1, keepdim=True) + epsilon)

# discrete probability distributions inner product
# checks if the underlying distributions are close
def dpd_inner_product(p_1:torch.tensor, p_2:torch.tensor) -> torch.tensor:
    # logit space
    mean_a = torch.mean(p_1, dim=-1, keepdim=True)
    mean_b = torch.mean(p_2, dim=-1, keepdim=True)
    return torch.sum((p_1 - mean_a) * (p_2 - mean_b), dim=-1)

    # can either convert to prob distributions, then do the normal method. Or do it in logit space (above)
    # p_1 = torch.exp(p_1) / torch.sum(torch.exp(p_1), dim=-1, keepdim=True)
    # p_2 = torch.exp(p_2) / torch.sum(torch.exp(p_2), dim=-1, keepdim=True)

    # prob space
    # n_categories = p_1.shape[-1]
    # g_p_1 = torch.prod(p_1, dim=-1) ** (1 / n_categories)  # geometric mean of p_1
    # g_p_2 = torch.prod(p_2, dim=-1) ** (1 / n_categories)  # geometric mean of p_2
    # return torch.sum(torch.log(p_1 / g_p_1.unsqueeze(-1)) * torch.log(p_2 / g_p_2.unsqueeze(-1)), dim=-1)

if __name__ == "__main__":
    n_categories = 4  # number of categories

    # zero = torch.ones(n_categories) / n_categories  # the zero vector
    zero = torch.zeros(n_categories, dtype=torch.float32)
    # demonstrates that only c-1 basis vectors are needed, using n_categories = 4
    # -a -b - c
    # a = torch.tensor([0.5, 1 / 6, 1 / 6, 1 / 6])
    # b = torch.tensor([1 / 6, 0.5, 1 / 6, 1 / 6])
    # c = torch.tensor([1 / 6, 1 / 6, 0.5, 1 / 6])
    # print(dpd_add(dpd_add(dpd_multiply(a, -1), dpd_multiply(b, -1)), dpd_multiply(c, -1)))
    a = torch.tensor([1, 0, 0, 0])
    b = torch.tensor([0, 1, 0, 0])
    c = torch.tensor([0, 0, 1, 0])
    print(dpd_add(dpd_add(dpd_multiply(a, -1), dpd_multiply(b, -1)), dpd_multiply(c, -1)))

    min, max = -10, 10
    for i in range(10):
        # create distributions
        a = torch.rand(n_categories) * (max - min) + min
        b = torch.rand(n_categories) * (max - min) + min
        c = torch.rand(n_categories) * (max - min) + min

        # make them prob distributions. No need if we stay in logit form
        # a = a / torch.sum(a)
        # b = b / torch.sum(b)
        # c = c / torch.sum(c)

        # check all vector properties
        # associative of dpd_addition
        assert allclose(dpd_add(dpd_add(a, b), c), dpd_add(a, dpd_add(b, c)))
        assert allclose(dpd_add(dpd_add(a, c), b), dpd_add(a, dpd_add(b, c)))

        # commutative of dpd_addition
        assert allclose(dpd_add(a, b), dpd_add(b, a))
        assert allclose(dpd_add(a, c), dpd_add(c, a))
        assert allclose(dpd_add(b, c), dpd_add(c, b))

        # identity of dpd_addition
        assert allclose(dpd_add(a, zero), a)
        assert allclose(dpd_add(b, zero), b)
        assert allclose(dpd_add(c, zero), c)

        # inverse dpd_addition
        assert allclose(dpd_add(a, dpd_multiply(a, -1)), zero)
        assert allclose(dpd_add(b, dpd_multiply(b, -1)), zero)
        assert allclose(dpd_add(c, dpd_multiply(c, -1)), zero)

        # scalar dpd_multiplyplication, field dpd_multiplyplication
        assert allclose(dpd_multiply(dpd_multiply(a, 2), 3), dpd_multiply(a, 6))
        assert allclose(dpd_multiply(dpd_multiply(a, 3), 2), dpd_multiply(a, 6))

        # identity dpd_multiplyplication
        assert allclose(dpd_multiply(a, 1), a)
        assert allclose(dpd_multiply(b, 1), b)
        assert allclose(dpd_multiply(c, 1), c)

        # distributive scalar dpd_multiplyplication wrt vector dpd_addition
        assert allclose(dpd_multiply(dpd_add(a, b), 2), dpd_add(dpd_multiply(a, 2), dpd_multiply(b, 2)))
        assert allclose(dpd_multiply(dpd_add(a, c), 2), dpd_add(dpd_multiply(a, 2), dpd_multiply(c, 2)))
        assert allclose(dpd_multiply(dpd_add(b, c), 2), dpd_add(dpd_multiply(b, 2), dpd_multiply(c, 2)))

        # distributive scalar dpd_multiplyplication wrt field dpd_addition
        assert allclose(dpd_multiply(a, 2 + 3), dpd_add(dpd_multiply(a, 2), dpd_multiply(a, 3)))
        assert allclose(dpd_multiply(b, 2 + 3), dpd_add(dpd_multiply(b, 2), dpd_multiply(b, 3)))
        assert allclose(dpd_multiply(c, 2 + 3), dpd_add(dpd_multiply(c, 2), dpd_multiply(c, 3)))

        # now check inner product properties
        # Symmetry
        assert allclose(dpd_inner_product(a, b), dpd_inner_product(b, a)), "first is {}, second is {}".format(dpd_inner_product(a, b), dpd_inner_product(b, a))
        assert allclose(dpd_inner_product(a, c), dpd_inner_product(c, a)), "first is {}, second is {}".format(dpd_inner_product(a, c), dpd_inner_product(c, a))
        assert allclose(dpd_inner_product(b, c), dpd_inner_product(c, b)), "first is {}, second is {}".format(dpd_inner_product(b, c), dpd_inner_product(c, b))

        # 0 vector is orthogonal to all vectors
        assert allclose(dpd_inner_product(a, zero), torch.tensor(0.0)), "first is {}, second is {}".format(dpd_inner_product(a, zero), torch.tensor(0.0))
        assert allclose(dpd_inner_product(b, zero), torch.tensor(0.0)), "first is {}, second is {}".format(dpd_inner_product(b, zero), torch.tensor(0.0))
        assert allclose(dpd_inner_product(c, zero), torch.tensor(0.0)), "first is {}, second is {}".format(dpd_inner_product(c, zero), torch.tensor(0.0))

        # positive definite
        assert dpd_inner_product(a, a) >= 0, "a is {}, ip is {}".format(a, dpd_inner_product(a, a))
        assert dpd_inner_product(b, b) >= 0, "b is {}, ip is {}".format(b, dpd_inner_product(b, b))
        assert dpd_inner_product(c, c) >= 0, "c is {}, ip is {}".format(c, dpd_inner_product(c, c))

        # linearity in first argument
        first = dpd_inner_product(dpd_add(dpd_multiply(a, 3), dpd_multiply(b, 2)), c)
        second = 3 * dpd_inner_product(a, c) + 2 * dpd_inner_product(b, c)
        assert allclose(first, second), "first is {}, second is {}".format(first, second)
        assert allclose(dpd_inner_product(dpd_add(dpd_multiply(a, 3), dpd_multiply(c, 2)), b),3 * dpd_inner_product(a, b) + 2 * dpd_inner_product(c, b))
        assert allclose(dpd_inner_product(dpd_add(dpd_multiply(b, 3), dpd_multiply(c, 2)), a),3 * dpd_inner_product(b, a) + 2 * dpd_inner_product(c, a))

    print("All tests passed!")


