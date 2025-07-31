import numpy as np
import matplotlib.pyplot as plt
import os

class ThompsonSampling:
    def __init__(self, n_arms, reward_dist='bernoulli', sigma2=1.0, mu0=0.0, var0=1.0):
        """
        n_arms       : number of arms
        reward_dist  : 'bernoulli' or 'gaussian'
        sigma2       : known noise variance for Gaussian rewards
        mu0, var0    : prior mean and variance for Gaussian rewards
        """
        self.n_arms = n_arms
        self.reward_dist = reward_dist

        if reward_dist == 'bernoulli':
            self.alpha = np.ones(n_arms)
            self.beta  = np.ones(n_arms)
        elif reward_dist == 'gaussian':
            self.sigma2 = sigma2
            self.mu0 = mu0
            self.tau0 = 1.0 / var0
            self.counts = np.zeros(n_arms)
            self.sums = np.zeros(n_arms)
        else:
            raise ValueError("reward_dist must be 'bernoulli' or 'gaussian'")

        self.chosen_arms = []
        self.ucb_history = []
        self.sample_history = []

    def select_arm(self):
        if self.reward_dist == 'bernoulli':
            samples = np.random.beta(self.alpha, self.beta)
            self.sample_history.append(list(samples))
            chosen = np.argmax(samples)
        else:
            tau_n = self.tau0 + self.counts
            mu_n = (self.tau0 * self.mu0 + self.sums) / tau_n
            std_n = np.sqrt(1.0 / tau_n)
            samples = np.random.normal(mu_n, std_n)
            self.sample_history.append(list(samples))
            chosen = np.argmax(samples)
        self.chosen_arms.append(chosen)
        return chosen

    def update(self, chosen_arm, reward):
        if self.reward_dist == 'bernoulli':
            self.alpha[chosen_arm] += reward
            self.beta[chosen_arm] += 1 - reward
        else:
            self.counts[chosen_arm] += 1
            self.sums[chosen_arm] += reward

def plot_evaluation(rewards, cumulative_regret, best_arm_freq, freq_over_time,
                   sample_history, chosen_arms, true_params, dist):
    os.makedirs("plots", exist_ok=True)
    print(f"Reward dist: {dist}")
    print(f"Best arm selection frequency: {best_arm_freq:.4f}")
    print(f"True params: {np.round(true_params,2)}")

    plt.figure(figsize=(16,8))

    plt.subplot(2,2,1)
    plt.plot(np.cumsum(rewards), label="Cumulative Reward")
    plt.plot(cumulative_regret, label="Cumulative Regret")
    plt.title("Thompson Sampling: Reward & Regret")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()

    plt.subplot(2,2,2)
    plt.plot(freq_over_time, label="Best Arm Frequency")
    plt.title("Best Arm Selection Frequency")
    plt.xlabel("Time Steps")
    plt.ylabel("Fraction")
    plt.legend()
    plt.grid()

    timeline = np.arange(len(sample_history))
    k = len(true_params)
    arms_to_plot = [np.argmax(true_params), np.argmin(true_params), np.random.randint(0, k)]
    plt.subplot(2,2,3)
    for i in arms_to_plot:
        sample_curve = [step[i] for step in sample_history]
        plt.plot(timeline, sample_curve, label=f"Arm {i} Sampled Mean")
    observed = [true_params[a] for a in chosen_arms]
    plt.scatter(timeline, observed, s=3, c='k', alpha=0.3, label="True Mean of Pulled Arm")
    plt.title("Posterior Samples (Selected Arms)")
    plt.xlabel("Time Steps")
    plt.ylabel("Sampled Mean / True Mean")
    plt.legend(fontsize="small")
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"plots/thompson_sampling_evaluation_{dist}.png")
    plt.close()

def simulate_thompson_sampling(T=10000, K=10, reward_dist='bernoulli'):
    if reward_dist == 'bernoulli':
        true_params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9])
    else:
        true_params = np.random.normal(0.5, 0.2, size=K)
    best_arm = np.argmax(true_params)
    algo = ThompsonSampling(n_arms=K, reward_dist=reward_dist)

    rewards = []
    best_count = 0
    freq_over_time = []
    cumulative_regret = []
    total_regret = 0

    for t in range(T):
        arm = algo.select_arm()
        if reward_dist == 'bernoulli':
            r = float(np.random.rand() < true_params[arm])
        else:
            r = np.random.normal(true_params[arm], 1.0)
        algo.update(arm, r)
        rewards.append(r)
        regret = true_params[best_arm] - r
        total_regret += regret
        cumulative_regret.append(total_regret)
        if arm == best_arm:
            best_count += 1
        freq_over_time.append(best_count / (t + 1))

    avg_reward = np.mean(rewards)
    best_freq = best_count / T
    return (rewards, avg_reward, best_freq, true_params,
            cumulative_regret, freq_over_time, algo.sample_history, algo.chosen_arms)

results = simulate_thompson_sampling(
    T=10000, K=10, reward_dist='gaussian'
)
(rewards, avg_r, freq, params,
 cumulative_regret, freq_over_time, sample_history, chosen_arms) = results

plot_evaluation(
    rewards, cumulative_regret, freq, freq_over_time,
    sample_history, chosen_arms, params, dist='gaussian'
)
