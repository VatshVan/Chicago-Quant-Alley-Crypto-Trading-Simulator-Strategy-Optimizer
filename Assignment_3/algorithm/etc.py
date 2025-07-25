import numpy as np
import matplotlib.pyplot as plt

import numpy as np

class ExploreThenCommit:
    def __init__(self, n_arms, n_explore):
        self.n_arms = n_arms
        self.n_explore = n_explore
        self.total_explore_steps = n_arms * n_explore
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.best_arm = None

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

def plot_rewards(rewards, avg_reward, best_arm_freq, true_probs):
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Best arm selection frequency: {best_arm_freq:.4f}")
    print(f"True arm means: {np.round(true_probs, 2)}")
    plt.plot(np.cumsum(rewards), label="Cumulative Reward")
    plt.title("Explore-Then-Commit Bandit Performance")
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid()
    plt.savefig("plots/explore_then_commit_performance.png")

def simulate_etc(n_explore=10, T=10000, K=10):
    true_probs = np.random.rand(K)
    best_arm = np.argmax(true_probs)
    
    algo = ExploreThenCommit(n_arms=K, n_explore=n_explore)
    
    rewards = []
    best_arm_counts = 0

    for t in range(T):
        arm = algo.select_arm(t)
        reward = np.random.rand() < true_probs[arm]
        algo.update(arm, reward)
        rewards.append(reward)
        if arm == best_arm:
            best_arm_counts += 1
    
    avg_reward = np.mean(rewards)
    return rewards, avg_reward, best_arm_counts / T, true_probs

rewards, avg_reward, best_arm_freq, true_probs = simulate_etc(n_explore=10)

plot_rewards(rewards, avg_reward, best_arm_freq, true_probs)