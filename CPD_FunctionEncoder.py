import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np

device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")


n_dims = 2
X_low = torch.ones(n_dims, device=device) * -1
X_high = torch.ones(n_dims, device=device) * 1
distribution_type = "circle" # or "line"

n_distributions = 10
n_basis = 100
hidden_dim = 250
descent_steps = 1000
n_samples = 100
std_dev = 0.1

positive_logit_value = 3
negative_logit_value = 0

method = "inner_product" # "inner_product" # or "norm"



# each distribution consists of a gaussian along a line
# first sample a line (2 endpoints in X), then sample a point uniformly from line, then sample a gaussian around that point
# use rejection sampling to ensure its within the bounds of X
def get_distributions(n_distributions):
    if distribution_type == "circle":
        return get_distributions_circle(n_distributions)
    elif distribution_type == "line":
        return get_distributions_lines(n_distributions)
    else:
        raise ValueError(f"Unknown distribution type {distribution_type}")

def get_distributions_lines(n_distributions):
    endpoints = torch.rand(n_distributions, 2, 2, device=device) * (X_high - X_low) + X_low

    def sample(n_points):
        theta = torch.rand(n_distributions, n_points, device=device)
        means = endpoints[:, 0].unsqueeze(1) + (endpoints[:, 1] - endpoints[:, 0]).unsqueeze(1) * theta.unsqueeze(-1)
        # sample from gaussian
        samples = torch.normal(means, std_dev)
        return samples
    return sample, endpoints


def set_distributions_lines(endpoints):
    def sample(n_points):
        theta = torch.rand(endpoints.shape[0], n_points, device=device)
        means = endpoints[:, 0].unsqueeze(1) + (endpoints[:, 1] - endpoints[:, 0]).unsqueeze(1) * theta.unsqueeze(-1)
        # sample from gaussian
        samples = torch.normal(means, std_dev)
        return samples
    return sample

def get_distributions_circle(n_distributions):
    # center is always 0,0
    radii = torch.rand(n_distributions, device=device) * (X_high[0] - X_low[0]) / 2
    def sample(n_points):
        theta = torch.rand(n_distributions, n_points, 1, device=device) * 2 * np.pi
        means = torch.concat([radii.reshape(-1, 1, 1) * torch.cos(theta), radii.reshape(-1, 1, 1) * torch.sin(theta)], dim=2)
        samples = torch.normal(means, std_dev)
        return samples
    return sample, radii.unsqueeze(1)



def get_random_xs(n_samples, Xs):
    if Xs is not None:
        random_Xs = torch.zeros(Xs.shape[0], n_samples, n_dims)
        for f in range(Xs.shape[0]):
            for i in range(n_samples):
                while True:
                    x = torch.rand(n_dims, device=device) * (X_high - X_low) + X_low
                    if torch.all(torch.norm(x - Xs[f]) > 0.1):
                        random_Xs[f, i] = x
                        break
        return random_Xs.to(device)
    else:
        return torch.rand(n_distributions, n_samples, n_dims, device=device) * (X_high - X_low) + X_low
G = torch.nn.Sequential(
        torch.nn.Linear(n_dims, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, n_basis),
).to(device)
opt = torch.optim.Adam(G.parameters(), lr=1e-3)

# logging
date_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logdir = f"continuous_logs/{date_time_str}"
logger = SummaryWriter(logdir)


for descent_step in trange(descent_steps):
    # sample distributions, approximate logits
    sampler, endpoints = get_distributions(n_distributions)
    true_Xs = sampler(n_samples)
    assert true_Xs.shape == (n_distributions, n_samples, n_dims)
    true_Xs_approximate_logits = torch.ones(n_distributions, n_samples, device=device) * positive_logit_value
    assert true_Xs_approximate_logits.shape == (n_distributions, n_samples)
    

    
    # sample random points, far from sampled points 
    random_Xs = get_random_xs(n_samples, true_Xs)
    assert random_Xs.shape == (n_distributions, n_samples, n_dims), f"{random_Xs.shape} != {(n_distributions, n_samples, n_dims)}"
    random_Xs_approximate_logits = torch.ones(n_distributions, n_samples, device=device) * negative_logit_value
    assert random_Xs_approximate_logits.shape == (n_distributions, n_samples)
    
    # combine
    Xs = torch.cat([true_Xs, random_Xs], dim=1)
    approximate_logits = torch.cat([true_Xs_approximate_logits, random_Xs_approximate_logits], dim=1)
    assert Xs.shape == (n_distributions, 2*n_samples, n_dims)
    assert approximate_logits.shape == (n_distributions, 2*n_samples)

    # compute the matrix for inner product
    approximate_logits_matrix = approximate_logits.unsqueeze(1) - approximate_logits.unsqueeze(2)
    assert approximate_logits_matrix.shape == (n_distributions, 2*n_samples, 2*n_samples)


    # compute representation
    # first get basis function logits
    individual_logits = G(Xs)
    assert individual_logits.shape == (n_distributions, 2*n_samples, n_basis)
    individual_logits_matrix = individual_logits.unsqueeze(1) - individual_logits.unsqueeze(2)
    assert individual_logits_matrix.shape == (n_distributions, 2*n_samples, 2*n_samples, n_basis)

    # next multiply matrices element wise
    individual_encodings = approximate_logits_matrix.unsqueeze(-1) * individual_logits_matrix
    assert individual_encodings.shape == (n_distributions, 2*n_samples, 2*n_samples, n_basis)
    representations = torch.mean(individual_encodings, dim=(1, 2)) * (X_high[0] - X_low[0]) * (X_high[1] - X_low[1])

    if method == "inner_product":
        # approximate function
        logits = torch.einsum("fek, fk -> fe", individual_logits, representations)
        assert logits.shape == (n_distributions, 2*n_samples)

        # compute loss as inner product
        error_vector = logits - approximate_logits
        error_matrix = error_vector.unsqueeze(1) - error_vector.unsqueeze(2)
        assert error_matrix.shape == (n_distributions, 2*n_samples, 2*n_samples)


        error_inner_products = torch.mean(error_matrix ** 2, dim=(1, 2))
        loss = torch.mean(error_inner_products)
    else:
        true_xs_logits = individual_logits[:, :n_samples, :]
        logits = torch.einsum("fek, fk -> fe", true_xs_logits, representations)
        loss = -torch.mean(logits)

    # back prop
    opt.zero_grad()
    loss.backward()
    opt.step()

    # write to log
    logger.add_scalar("loss", loss.item(), descent_step)


# now do some plotting
n_cols, n_rows = 4, 2
fig = plt.figure(figsize=(n_cols*4 + 1, n_rows*3.8))
gs = plt.GridSpec(n_rows, n_cols + 1, width_ratios=[4, 4, 4, 4, 1])# , height_ratios=[4,4]) # the last axe is for color bar
axes = [fig.add_subplot(gs[i//n_cols, i%n_cols], aspect='equal') for i in range(n_cols*n_rows)]


# compute pdf for all distributions
# first create a dense grid over X
n_grid = 101
Xs = torch.linspace(X_low[0], X_high[0], n_grid, device=device)
Ys = torch.linspace(X_low[1], X_high[1], n_grid, device=device)
X, Y = torch.meshgrid(Xs, Ys)
Xs = torch.stack([X, Y], dim=0).transpose(0, 2).to(device)

# compute pdfs
with torch.no_grad():
    G_Xs = G(Xs)
    pdfs = torch.einsum("ijk, fk -> ijf", G_Xs, representations)
    e_pdfs = torch.exp(pdfs)
    # int_X p(x) dx dy \approx V/N sum_x p(x)
    sums = torch.sum(e_pdfs, dim=(0, 1)) * (X_high[0] - X_low[0]) * (X_high[1] - X_low[1]) / n_grid ** 2
    # sums = torch.sum(e_pdfs, dim=(0, 1)) 
    probs = e_pdfs / sums


# plot using the last distribution from training
for i in range(n_cols*n_rows):
    ax = axes[i]

    # plot pdfs
    prob = probs[:, :, i]
    ax.contourf(Xs[:, :, 0].cpu().numpy(), Xs[:, :, 1].cpu().numpy(), prob.cpu().numpy(), levels=20, cmap="Reds")

    # plot groundturth data
    ax.scatter(true_Xs[i, :, 0].cpu().numpy(), true_Xs[i, :, 1].cpu().numpy(), c="black", label="Samples", s=3)
    if distribution_type == "line":
        ax.plot(endpoints[i, :, 0].cpu().numpy(), endpoints[i, :, 1].cpu().numpy(), c="blue", linewidth=4)
    elif distribution_type == "circle":
        pass
        # ax.add_patch(plt.Circle((0, 0), endpoints[i, 0].item(), fill=False, color="blue", linewidth=4))
    else:
        raise ValueError(f"Unknown distribution type {distribution_type}")


    # move axis
    ax.set_xlim(X_low[0].cpu().numpy(), X_high[0].cpu().numpy())
    ax.set_ylim(X_low[1].cpu().numpy(), X_high[1].cpu().numpy())

    # remove x axis labels for all but bottom row
    if i < n_cols * (n_rows - 1):
        ax.set_xticks([])
    else:
        ax.set_xticks([-1, 0, 1])

    if i % n_cols != 0:
        ax.set_yticks([])
    else:
        ax.set_yticks([-1, 0, 1])



# add a color bar to last col
cax = fig.add_subplot(gs[:, -1])
cbar = plt.colorbar(ax.collections[0], cax=cax, orientation="vertical", fraction=0.1)


# save the plot
fig.tight_layout()
plt.savefig(f"{logdir}/continuous_plot.png")


# now create a synthetic example with a smiley face
with torch.no_grad():
    p1 = [-0.5, 0.75]
    p2 = [-0.5, 0.25]
    p3 = [0.5, 0.75]
    p4 = [0.5, 0.25]
    p5 = [-0.5, -0.5]
    p6 = [0.5, -0.5]
    endpoints = torch.tensor([[p1, p2], [p3, p4], [p5, p6]], device=device)
    samplers = set_distributions_lines(endpoints)
    true_Xs = samplers(n_samples)

    # squish the dimensions so its one distribution
    true_Xs = true_Xs.view(-1, n_dims).unsqueeze(0)
    assert true_Xs.shape == (1, n_samples * 3, n_dims)

    true_Xs_approximate_logits = torch.ones(1, n_samples* 3, device=device) * positive_logit_value
    assert true_Xs_approximate_logits.shape == (1, n_samples * 3)

    # sample random points, far from sampled points
    random_Xs = get_random_xs(3 * n_samples, true_Xs)
    assert random_Xs.shape == (1, 3 *n_samples, n_dims), f"{random_Xs.shape} != {(n_distributions, 3 *n_samples, n_dims)}"
    random_Xs_approximate_logits = torch.ones(1, 3 * n_samples, device=device) * negative_logit_value
    assert random_Xs_approximate_logits.shape == (1, 3 *n_samples)


    # combine
    Xs = torch.cat([true_Xs, random_Xs], dim=1)
    approximate_logits = torch.cat([true_Xs_approximate_logits, random_Xs_approximate_logits], dim=1)
    assert Xs.shape == (1, 6*n_samples, n_dims)
    assert approximate_logits.shape == (1, 6*n_samples)

    # compute the matrix for inner product
    approximate_logits_matrix = approximate_logits.unsqueeze(1) - approximate_logits.unsqueeze(2)
    assert approximate_logits_matrix.shape == (1, 6*n_samples, 6*n_samples)




    # compute representation
    # first get basis function logits
    individual_logits = G(Xs)
    assert individual_logits.shape == (1, 6*n_samples, n_basis)
    individual_logits_matrix = individual_logits.unsqueeze(1) - individual_logits.unsqueeze(2)
    assert individual_logits_matrix.shape == (1, 6*n_samples, 6*n_samples, n_basis)

    # next multiply matrices element wise
    individual_encodings = approximate_logits_matrix.unsqueeze(-1) * individual_logits_matrix
    assert individual_encodings.shape == (1, 6*n_samples, 6*n_samples, n_basis)
    representations = torch.mean(individual_encodings, dim=(1, 2))

    # generate grid over xs
    n_grid = 101
    Xs = torch.linspace(X_low[0], X_high[0], n_grid)
    Ys = torch.linspace(X_low[1], X_high[1], n_grid)
    X, Y = torch.meshgrid(Xs, Ys)
    Xs = torch.stack([X, Y], dim=0).transpose(0, 2).to(device)

    # compute pdfs
    G_Xs = G(Xs)
    pdfs = torch.einsum("ijk, fk -> ijf", G_Xs, representations)

    # int_X p(x) dx dy \approx V/N sum_x p(x)
    sums = torch.sum(torch.exp(pdfs), dim=(0, 1)) * (X_high[0] - X_low[0]) * (X_high[1] - X_low[1]) / n_grid ** 2
    probs = torch.exp(pdfs) / sums

    # plot
    fig, ax = plt.subplots()
    ax.contourf(Xs[:, :, 0].cpu().numpy(), Xs[:, :, 1].cpu().numpy(), probs[:, :, 0].cpu().numpy(), levels=20, cmap="Reds")
    ax.scatter(true_Xs[0, :, 0].cpu().numpy(), true_Xs[0, :, 1].cpu().numpy(), c="black", label="Samples", s=3)
    ax.plot(endpoints[0, :, 0].cpu().numpy(), endpoints[0, :, 1].cpu().numpy(), c="blue", linewidth=4)
    ax.plot(endpoints[1, :, 0].cpu().numpy(), endpoints[1, :, 1].cpu().numpy(), c="blue", linewidth=4)
    ax.plot(endpoints[2, :, 0].cpu().numpy(), endpoints[2, :, 1].cpu().numpy(), c="blue", linewidth=4)
    ax.set_xlim(X_low[0].cpu().numpy(), X_high[0].cpu().numpy())
    ax.set_ylim(X_low[1].cpu().numpy(), X_high[1].cpu().numpy())
    plt.savefig(f"{logdir}/smiley_face.png")






# now try maximum liklihood
# compute representation by doing gradient descent on maximum liklihood problem
n_inner_steps = 10000
with torch.no_grad():
    samplers, endpoints = get_distributions(n_distributions)
    true_Xs = samplers(n_samples)
    g_true_Xs = G(true_Xs)

representations = torch.randn(n_distributions, n_basis, device=true_Xs.device) 
representations *= 0.1
representations.requires_grad = True
inner_opt = torch.optim.Adam([representations], lr=1e-3)
for inner_gradient_step in trange(n_inner_steps):
    with torch.no_grad():
        # sample random points, far from sampled points
        random_Xs = get_random_xs(n_samples, Xs=None)
        g_random_Xs = G(random_Xs)
        assert random_Xs.shape == (n_distributions, n_samples, n_dims), f"{random_Xs.shape} != {(n_distributions, n_samples, n_dims)}"

    if torch.isnan(representations).any():
        print("NAN")
        done = True
        break
    ct_g_random_Xs = torch.einsum("fnk, fk -> fn", g_random_Xs, representations)
    # ct_g_random_Xs = torch.clip(ct_g_random_Xs, 1e-4)
    z_c = torch.mean(torch.exp(ct_g_random_Xs), dim=1) * (X_high[0] - X_low[0]) * (X_high[1] - X_low[1])

    if torch.isnan(z_c).any():
        print("NAN")
        done = True
        break
    log_prob = -n_samples * torch.log(z_c + 1e-7) + torch.einsum("fnk, fk -> f", g_true_Xs, representations)
    if torch.isnan(log_prob).any():
        print("NAN")
        done = True
        break
    inner_loss = -torch.mean(log_prob)
    # print(z_c.min().item())
    if torch.isnan(inner_loss):
        print("NAN")
        done = True
        break
    inner_loss.backward()
    # grad clip
    norm = torch.nn.utils.clip_grad_norm_(representations, 1)
    if torch.isnan(norm):
        print("NAN")
        done = True
        break
    if torch.isnan(representations.grad).any():
        print("NAN")
        done = True
        break
    inner_opt.step()
    inner_opt.zero_grad()
    logger.add_scalar("norm", norm.item(), inner_gradient_step + n_inner_steps * descent_step)

representations = representations.detach() # no need to track gradients anymore

    
# now do some plotting
n_cols, n_rows = 4, 2
fig = plt.figure(figsize=(n_cols*4 + 1, n_rows*3.8))
gs = plt.GridSpec(n_rows, n_cols + 1, width_ratios=[4, 4, 4, 4, 1])# , height_ratios=[4,4]) # the last axe is for color bar
axes = [fig.add_subplot(gs[i//n_cols, i%n_cols], aspect='equal') for i in range(n_cols*n_rows)]


# compute pdf for all distributions
# first create a dense grid over X
n_grid = 101
Xs = torch.linspace(X_low[0], X_high[0], n_grid).to(device)
Ys = torch.linspace(X_low[1], X_high[1], n_grid).to(device)
X, Y = torch.meshgrid(Xs, Ys)
Xs = torch.stack([X, Y], dim=0).transpose(0, 2).to(device)

# compute pdfs
with torch.no_grad():
    G_Xs = G(Xs)
    pdfs = torch.einsum("ijk, fk -> ijf", G_Xs, representations)
    e_pdfs = torch.exp(pdfs)
    # int_X p(x) dx dy \approx V/N sum_x p(x)
    sums = torch.sum(e_pdfs, dim=(0, 1)) * (X_high[0] - X_low[0]) * (X_high[1] - X_low[1]) / n_grid ** 2
    # sums = torch.sum(e_pdfs, dim=(0, 1)) 
    probs = e_pdfs / sums


# plot using the last distribution from training
for i in range(n_cols*n_rows):
    ax = axes[i]

    # plot pdfs
    prob = probs[:, :, i]
    ax.contourf(Xs[:, :, 0].cpu().numpy(), Xs[:, :, 1].cpu().numpy(), prob.cpu().numpy(), levels=20, cmap="Reds")

    # plot groundturth data
    ax.scatter(true_Xs[i, :, 0].cpu().numpy(), true_Xs[i, :, 1].cpu().numpy(), c="black", label="Samples", s=3)
    if distribution_type == "line":
        ax.plot(endpoints[i, :, 0].cpu().numpy(), endpoints[i, :, 1].cpu().numpy(), c="blue", linewidth=4)
    elif distribution_type == "circle":
        pass
        # ax.add_patch(plt.Circle((0, 0), endpoints[i, 0].item(), fill=False, color="blue", linewidth=4))
    else:
        raise ValueError(f"Unknown distribution type {distribution_type}")


    # move axis
    ax.set_xlim(X_low[0].cpu().numpy(), X_high[0].cpu().numpy())
    ax.set_ylim(X_low[1].cpu().numpy(), X_high[1].cpu().numpy())

    # remove x axis labels for all but bottom row
    if i < n_cols * (n_rows - 1):
        ax.set_xticks([])
    else:
        ax.set_xticks([-1, 0, 1])

    if i % n_cols != 0:
        ax.set_yticks([])
    else:
        ax.set_yticks([-1, 0, 1])



# add a color bar to last col
cax = fig.add_subplot(gs[:, -1])
cbar = plt.colorbar(ax.collections[0], cax=cax, orientation="vertical", fraction=0.1)


# save the plot
fig.tight_layout()
plt.savefig(f"{logdir}/maximum_liklihood_plot.png")





