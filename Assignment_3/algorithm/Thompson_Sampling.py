import numpy as np
import matplotlib.pyplot as plt

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
            # Beta(1,1) prior
            self.alpha = np.ones(n_arms)
            self.beta  = np.ones(n_arms)

        elif reward_dist == 'gaussian':
            # Normal(mu0, var0) prior, known noise variance sigma2
            self.sigma2 = sigma2
            self.mu0    = mu0
            self.tau0   = 1.0 / var0            # prior precision = 1/var0
            # sufficient stats for each arm
            self.counts = np.zeros(n_arms)
            self.sums   = np.zeros(n_arms)
        else:
            raise ValueError("reward_dist must be 'bernoulli' or 'gaussian'")

    def select_arm(self):
        if self.reward_dist == 'bernoulli':
            # sample from Beta posterior
            samples = np.random.beta(self.alpha, self.beta)
            return np.argmax(samples)

        # Gaussian case: sample from Normal posterior N(μ_n, 1/τ_n)
        # posterior precision τ_n = τ0 + n
        tau_n = self.tau0 + self.counts
        # posterior mean μ_n = (τ0·μ0 + sum_rewards) / τ_n
        mu_n  = (self.tau0 * self.mu0 + self.sums) / tau_n
        # draw one sample per arm
        std_n = np.sqrt(1.0 / tau_n)
        samples = np.random.normal(mu_n, std_n)
        return np.argmax(samples)

    def update(self, chosen_arm, reward):
        if self.reward_dist == 'bernoulli':
            # reward must be 0 or 1
            self.alpha[chosen_arm] += reward
            self.beta [chosen_arm] += 1 - reward
        else:
            # reward is real-valued
            self.counts[chosen_arm] += 1
            self.sums  [chosen_arm] += reward

def plot_rewards(rewards, avg_reward, best_arm_freq, true_params, dist):
    print(f"Reward dist: {dist}")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Best arm selection frequency: {best_arm_freq:.4f}")
    print(f"True params: {np.round(true_params,2)}")
    plt.plot(np.cumsum(rewards), label="Cumulative Reward")
    plt.title(f"Thompson Sampling ({dist.capitalize()})")
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid()
    plt.savefig(f"plots/thompson_sampling_{dist}.png")

def simulate_thompson_sampling(T=10000, K=10, reward_dist='bernoulli'):
    if reward_dist == 'bernoulli':
        true_params = np.array([0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.7,0.8,0.9])
    else:
        true_params = np.random.normal(0.5, 0.2, size=K)  # example Gaussian means

    best_arm = np.argmax(true_params)
    algo = ThompsonSampling(n_arms=K, reward_dist=reward_dist)

    rewards = []
    best_count = 0

    for _ in range(T):
        arm = algo.select_arm()
        if reward_dist == 'bernoulli':
            r = float(np.random.rand() < true_params[arm])
        else:
            # Gaussian reward with known noise variance 1
            r = np.random.normal(true_params[arm], 1.0)
        algo.update(arm, r)
        rewards.append(r)
        if arm == best_arm:
            best_count += 1

    avg_reward   = np.mean(rewards)
    best_freq    = best_count / T
    return rewards, avg_reward, best_freq, true_params

# Example usage:
rewards, avg_r, freq, params = simulate_thompson_sampling(
    T=10000, K=10, reward_dist='gaussian'
)
plot_rewards(rewards, avg_r, freq, params, dist='gaussian')