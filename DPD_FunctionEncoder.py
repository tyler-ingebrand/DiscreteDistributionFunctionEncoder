import datetime
import os

import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from VectorSpace import dpd_inner_product, dpd_multiply, dpd_add

# suppose we have a set of functions mapping from [0,1] to K, where K is a category set
# for simplicity, assume the functions are defined by the ranges for which each is dominant.
input_space = [0.0, 1.0]
n_categories = 3
n_functions = 10
seed = 0
n_basis = 110
descent_steps = 100_000
n_datapoints = 1000
device = "cuda:0"

# example data confidence. How much we trust the example data, IE 95% confidence
example_data_confidence = 0.95
example_data_logit = torch.log(torch.tensor((n_categories - 1) * example_data_confidence / (1.0 - example_data_confidence)))

# seed everything
torch.manual_seed(seed)

# logs
logdir = "logs"
date_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logdir = f"{logdir}/{date_time_str}"
os.makedirs(logdir, exist_ok=True)
logger = SummaryWriter(logdir)

# Sample a function which maps from input to a category for different ranges of inputs
def sample_functions(n_functions, n_categories, input_space):
    barriers = torch.rand(n_functions, n_categories-1, device=device) * (input_space[1] - input_space[0]) + input_space[0]
    barriers = torch.sort(barriers, dim=-1).values
    chosen_categories = torch.stack([torch.randperm(n_categories, device=device) for _ in range(n_functions)])

    def function(x):
        # efficiently compute the category for each x
        indexes = torch.stack([torch.searchsorted(b, x) for b in barriers])
        # the output for the first function should be chosen_categories[0][indexes[0]]
        return chosen_categories[torch.arange(n_functions).unsqueeze(-1), indexes]
    return function, barriers, chosen_categories

# create a function encoder which maps an input value to a distribution over categories for each basis function
class DPD_FunctionEncoder(torch.nn.Module):
    def __init__(self, n_basis, n_categories, input_size=1, hidden_size=100):
        super(DPD_FunctionEncoder, self).__init__()
        self.n_basis = n_basis
        self.n_categories = n_categories
        self.input_size = input_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, n_basis * n_categories),
        )

    def forward(self, x):
        logits = self.model(x).view(-1, self.n_basis, self.n_categories)
        return logits

model = DPD_FunctionEncoder(n_basis, n_categories).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# train
for descent_step in trange(descent_steps):
    # generate data
    functions, _, __ = sample_functions(n_functions, n_categories, input_space)
    example_xs = torch.rand(n_datapoints, device=device) * (input_space[1] - input_space[0]) + input_space[0]
    example_ys = functions(example_xs)
    xs = torch.rand(n_datapoints, device=device) * (input_space[1] - input_space[0]) + input_space[0]
    ys = functions(xs)

    # compute representation using example data
    example_basis_data = model(example_xs.unsqueeze(-1)).squeeze()

    # create ohe of the chosen value, where the max is 0.95 and the min are 0.05 / n_categories-1
    example_ys_ohe = torch.nn.functional.one_hot(example_ys, n_categories).float()
    smoothed_example_ys_ohe = example_ys_ohe * example_data_logit

    individual_example_encodings = dpd_inner_product(example_basis_data.unsqueeze(0), smoothed_example_ys_ohe.unsqueeze(2))
    encodings = torch.mean(individual_example_encodings, dim=1)

    # approximate function
    basis_data = model(xs.unsqueeze(-1)).squeeze()
    individual_encodings = dpd_multiply(basis_data.unsqueeze(0), encodings.unsqueeze(1).unsqueeze(-1))
    # y_hat = individual_encodings[:, :, 0, :]
    # for i in range(1, n_basis):
    #     y_hat = dpd_add(y_hat, individual_encodings[:, :, i, :])
    # y hats are logits
    logits = torch.sum(individual_encodings, dim=-2)

    if torch.isnan(logits).any():
        print("NaN detected")

    # compute loss
    loss = torch.nn.CrossEntropyLoss()(logits.view(-1, n_categories), ys.view(-1))


    # backprop
    opt.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    opt.step()

    # log
    logger.add_scalar("loss", loss.item(), descent_step)
    logger.add_scalar("grad_norm", norm.item(), descent_step)



# Graphs!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# now see if it works for 9 functions (graph them)
n_sample_functions = 9
functions, barriers, chosen_categories = sample_functions(n_sample_functions, n_categories, input_space)

# generate data
n_datapoints = 1000
xs = torch.linspace(input_space[0], input_space[1], n_datapoints, device=device)
ys = functions(xs)
ys_ohe = torch.nn.functional.one_hot(ys, n_categories).float()
smoothed_ys_ohe = torch.where(ys_ohe == 1, example_data_confidence, (1.0-example_data_confidence) / (n_categories-1))

# compute representation using example data
basis_data = model(xs.unsqueeze(-1)).squeeze()
individual_encodings = dpd_inner_product(basis_data.unsqueeze(0), smoothed_ys_ohe.unsqueeze(2))
encodings = torch.mean(individual_encodings, dim=1)

# approximate function
basis_data = model(xs.unsqueeze(-1)).squeeze()
individual_encodings = dpd_multiply(basis_data.unsqueeze(0), encodings.unsqueeze(1).unsqueeze(-1))
logits = torch.sum(individual_encodings, dim=-2)
y_hat = torch.nn.functional.softmax(logits, dim=-1)


def to_cpu(tensor):
    return tensor.detach().cpu().numpy()

# plot the functions along with the probability distributions
fig, ax = plt.subplots(3, 3, figsize=(10, 10))
colors = "r", "g", "b", "c", "m", "y", "k", "orange", "purple"
labels = list([chr(ord('A') + i) for i in range(n_categories)])

for i in range(n_sample_functions):
    ax[i//3][i%3].plot(to_cpu(xs), to_cpu(ys[i]), label="true", color="black")
    for b, color, l in zip(reversed(range(n_categories)), colors, labels):
        ax[i//3][i%3].plot(to_cpu(xs), to_cpu(2 * y_hat[i][:, b]), label=l, color=color)
    ax[i//3][i%3].set_yticks([i for i in range(n_categories)])
    ax[i//3][i%3].set_yticklabels(reversed([chr(ord('A') + i) for i in range(n_categories)]))
ax[2][2].legend()
fig.tight_layout()

# save the plot
plt.savefig(f"{logdir}/prob.png")
plt.clf()


# plot the functions along with the max approximation
y_hat = torch.max(y_hat, dim=-1).indices
fig, ax = plt.subplots(3, 3, figsize=(10, 10))
for i in range(n_sample_functions):
    ax[i//3][i%3].plot(to_cpu(xs), to_cpu(ys[i]), label="true", color="black")
    ax[i//3][i%3].plot(to_cpu(xs), to_cpu(y_hat[i]), label="approx", color="blue")
    ax[i//3][i%3].set_yticks([i for i in range(n_categories)])
    ax[i//3][i%3].set_yticklabels(reversed([chr(ord('A') + i) for i in range(n_categories)]))
ax[2][2].legend()
fig.tight_layout()

# save the plot
plt.savefig(f"{logdir}/max.png")
