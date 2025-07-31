import numpy as np
import matplotlib.pyplot as plt
import os

class ExploreThenCommit:
    def __init__(self, n_arms, n_explore):
        self.n_arms = n_arms
        self.n_explore = n_explore
        self.total_explore_steps = n_arms * n_explore
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.best_arm = None
        self.values_history = []
        self.counts_history = []
        self.selected_arms = []

    def select_arm(self, t=None):
        if t is None:
            if self.best_arm is None:
                self.best_arm = np.argmax(self.values)
            return self.best_arm

        if t < self.total_explore_steps:
            return t % self.n_arms
        else:
            if self.best_arm is None:
                self.best_arm = np.argmax(self.values)
            return self.best_arm

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = value + (reward - value) / n
        self.values_history.append(self.values.copy())
        self.counts_history.append(self.counts.copy())

def plot_evaluation(rewards, cumulative_regret, best_arm_freq, freq_over_time, true_probs):
    os.makedirs("plots", exist_ok=True)
    print(f"Best arm selection frequency: {best_arm_freq:.4f}")
    print(f"True arm means: {np.round(true_probs, 2)}")
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(rewards), label="Cumulative Reward")
    plt.plot(cumulative_regret, label="Cumulative Regret")
    plt.title("Explore-Then-Commit Performance")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(freq_over_time, label="Best Arm Frequency")
    plt.title("Best Arm Selection Frequency")
    plt.xlabel("Time Steps")
    plt.ylabel("Fraction")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("plots/explore_then_commit_evaluation.png")
    plt.close()

def simulate_etc(n_explore=10, T=10000, K=10):
    true_probs = np.random.rand(K)
    best_arm = np.argmax(true_probs)
    best_mean = true_probs[best_arm]
    cumulative_regret = []
    total_regret = 0
    freq_over_time = []

    algo = ExploreThenCommit(n_arms=K, n_explore=n_explore)
    rewards = []
    best_arm_counts = 0

    for t in range(T):
        arm = algo.select_arm(t)
        reward = np.random.rand() < true_probs[arm]
        algo.update(arm, reward)
        rewards.append(reward)
        regret = best_mean - reward
        total_regret += regret
        cumulative_regret.append(total_regret)
        if arm == best_arm:
            best_arm_counts += 1
        freq_over_time.append(best_arm_counts / (t + 1))

    avg_reward = np.mean(rewards)
    return rewards, avg_reward, best_arm_counts / T, true_probs, cumulative_regret, freq_over_time

rewards, avg_reward, best_arm_freq, true_probs, cumulative_regret, freq_over_time = simulate_etc(n_explore=10)
plot_evaluation(rewards, cumulative_regret, best_arm_freq, freq_over_time, true_probs)