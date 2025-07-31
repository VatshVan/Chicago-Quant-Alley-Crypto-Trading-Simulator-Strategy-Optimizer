from algorithm.epsilon_greedy import EpsilonGreedy
from algorithm.etc import ExploreThenCommit
from algorithm.Upper_Confidence_Bound import UCB1
from algorithm.KL_UCB import KLUCB
from algorithm.Thompson_Sampling import ThompsonSampling
from algorithm.Weighted_Majority import WeightedMajority
from algorithm.Exp3_exponential_weight_Algorithm import Exp3
from algorithm.Linear_UCB import LinUCB
from algorithm.Halving_Algorithm import Halving
from algorithm.Lower_Upper_Confidence_Bound import LUCB
from algorithm.KL_LUCB import KLLUCB
from algorithm.lil_UCB import LilUCB

import numpy as np
import matplotlib.pyplot as plt
import os

def simulate_stochastic(Algorithm, T, true_means, dist="bernoulli", sigma=1.0, **kwargs):
    K = len(true_means)
    print(f"\nEvaluating: {Algorithm.__name__} | Distribution: {dist}")
    if Algorithm.__name__ == "ThompsonSampling":
        algo = Algorithm(n_arms=K, reward_dist=dist, sigma2=sigma)
    else:
        try:
            algo = Algorithm(n_arms=K, **kwargs)
        except TypeError:
            algo = Algorithm(K, **kwargs)

    regret = np.zeros(T)
    pulls_best = 0
    best_mean = np.max(true_means)
    optimal_arm = np.argmax(true_means)
    cum_regret = 0.0

    ucb_log = []
    chosen_arms_log = []
    rewards_log = []

    for t in range(T):
        arm = algo.select_arm(t) if isinstance(algo, ExploreThenCommit) else algo.select_arm()
        if hasattr(algo, 'ucb_history'):
            if hasattr(algo, 'ucb_history') and len(algo.ucb_history) > 0:
                ucb_log.append(algo.ucb_history[-1])
        elif hasattr(algo, 'ucb'):
            ucb_values = [algo.ucb(i, t) for i in range(K)]
            ucb_log.append(ucb_values)
        else:
            ucb_log.append([np.nan]*K)
        chosen_arms_log.append(arm)

        r = float(np.random.rand() < true_means[arm]) if dist == "bernoulli" else np.random.normal(true_means[arm], sigma)
        rewards_log.append(r)
        algo.update(arm, r)
        cum_regret += best_mean - r
        regret[t] = cum_regret
        if arm == optimal_arm:
            pulls_best += 1

    freq_best = pulls_best / T
    print(f"  → Final Regret: {cum_regret:.2f} | Best Arm Freq: {freq_best:.3f}")
    return regret, freq_best, np.array(ucb_log), chosen_arms_log, np.array(rewards_log), optimal_arm

def simulate_contextual_linucb(T, K, d, true_thetas, alpha=1.0):
    print("\nEvaluating: LinUCB (Contextual Bandit)")
    algo = LinUCB(n_arms=K, d=d, alpha=alpha)
    regret = np.zeros(T)
    cum_regret = 0.0
    ucb_log = []
    chosen_arms_log = []
    rewards_log = []

    for t in range(T):
        contexts = [np.random.rand(d) for _ in range(K)]
        expected = [contexts[i] @ true_thetas[i] for i in range(K)]
        optimal = np.argmax(expected)
        chosen = algo.select_arm(contexts)
        ucb_log.append(algo.ucb_history[-1])
        chosen_arms_log.append(chosen)
        r = contexts[chosen] @ true_thetas[chosen] + np.random.normal(0, 0.1)
        rewards_log.append(r)
        algo.update(chosen, r, contexts[chosen])
        cum_regret += expected[optimal] - r
        regret[t] = cum_regret

    print(f"  → Final Regret: {cum_regret:.2f}")
    return regret, np.array(ucb_log), chosen_arms_log, np.array(rewards_log)

K = 10
T = 10_000
np.random.seed(0)
os.makedirs("plots", exist_ok=True)

true_means_bern = np.random.rand(K)
true_means_gauss = np.random.randn(K) * 0.5 + 0.5

algorithms = [
    (EpsilonGreedy,    {"epsilon":0.1}),
    (ExploreThenCommit,{"n_explore":10}),
    (UCB1,             {}),
    (KLUCB,            {}),
    (ThompsonSampling, {}),
]

# Bernoulli Regret/CB Plots
plt.figure(figsize=(8,5))
for Algo, params in algorithms:
    r, f, ucb_log, chosen_arms, rewards_log, optimal_arm = simulate_stochastic(
        Algo, T, true_means_bern, dist="bernoulli", sigma=1.0, **params)
    plt.plot(r, label=f"{Algo.__name__} (freq best {f:.2f})")
    if Algo.__name__ in ['UCB1', 'KLUCB']:
        plt.figure(figsize=(10,5))
        timeline = np.arange(len(ucb_log))
        ucb_opt = np.array([ucb_log[t][optimal_arm] if ucb_log[t][optimal_arm]==ucb_log[t][optimal_arm] else np.nan for t in range(T)])
        plt.plot(timeline, ucb_opt, label=f"UCB (opt arm={optimal_arm})", alpha=0.7)
        chosen_rewards = np.array([rewards_log[t] for t in range(T) if chosen_arms[t]==optimal_arm])
        chosen_steps = np.array([t for t in range(T) if chosen_arms[t]==optimal_arm])
        if len(chosen_steps) > 0:
            plt.scatter(chosen_steps, chosen_rewards, c='orange', s=8, label="Observed Reward (opt arm)", alpha=0.6)
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title(f"{Algo.__name__}: UCB vs. Observed Reward (Opt. Arm)")
        plt.legend(); plt.grid()
        plt.tight_layout()
        plt.savefig(f"plots/{Algo.__name__}_cb_vs_reward_bernoulli.png")
plt.title("Cumulative Regret (Bernoulli)")
plt.xlabel("t"); plt.ylabel("Regret"); plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig("plots/regret_bernoulli.png")

# Gaussian Regret Plots
plt.figure(figsize=(8,5))
for Algo, params in algorithms:
    r, f, _, _, _, _ = simulate_stochastic(Algo, T, true_means_gauss, dist="gaussian", sigma=1.0, **params)
    plt.plot(r, label=f"{Algo.__name__} (freq best {f:.2f})")
plt.title("Cumulative Regret (Gaussian)")
plt.xlabel("t"); plt.ylabel("Regret"); plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig("plots/regret_gaussian.png")

# Best-arm selection frequency (Bernoulli)
plt.figure(figsize=(8,5))
for Algo, params in algorithms:
    print(f"\nCalculating best-arm frequency for: {Algo.__name__}")
    if Algo.__name__ == "ThompsonSampling":
        algo = Algo(n_arms=K, reward_dist="bernoulli", sigma2=1.0)
    else:
        try:
            algo = Algo(n_arms=K, **params)
        except TypeError:
            algo = Algo(K, **params)

    best = np.argmax(true_means_bern)
    freq = np.zeros(T)
    count = 0
    for t in range(T):
        arm = algo.select_arm(t) if isinstance(algo, ExploreThenCommit) else algo.select_arm()
        r = float(np.random.rand() < true_means_bern[arm])
        algo.update(arm, r)
        if arm == best:
            count += 1
        freq[t] = count / (t+1)
    plt.plot(freq, label=Algo.__name__)
plt.title("Best-Arm Selection Frequency (Bernoulli)")
plt.xlabel("t"); plt.ylabel("Frequency"); plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig("plots/best_arm_frequency_bernoulli.png")

# Contextual LinUCB
d = 5
true_thetas = [np.random.rand(d) for _ in range(K)]
r_ctx, ucb_ctx_log, ctx_chosen_arms, ctx_rewards_log = simulate_contextual_linucb(T, K, d, true_thetas, alpha=1.0)
plt.figure(figsize=(8,5))
plt.plot(r_ctx)
plt.title("Cumulative Regret: Contextual LinUCB")
plt.xlabel("t"); plt.ylabel("Regret"); plt.grid(); plt.tight_layout()
plt.savefig("plots/contextual_linucb_regret.png")

plt.figure(figsize=(10,5))
timeline = np.arange(len(ctx_chosen_arms))
for a_plot in [np.random.randint(0, K), np.argmax([tt.mean() for tt in true_thetas])]:
    ucb_vals = [ucb_ctx_log[t][a_plot] for t in range(len(ucb_ctx_log))]
    plt.plot(timeline, ucb_vals, label=f"UCB (arm {a_plot})")
chosen_rewards = [ctx_rewards_log[t] for t in range(T) if ctx_chosen_arms[t]==a_plot]
chosen_steps = [t for t in range(T) if ctx_chosen_arms[t]==a_plot]
if len(chosen_steps) > 0:
    plt.scatter(chosen_steps, chosen_rewards, c='orange', s=8, label="Observed Reward (ctx arm)", alpha=0.6)
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("LinUCB: UCB vs. Observed Reward (Select Arms)")
plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig("plots/linucb_cb_vs_reward.png")
