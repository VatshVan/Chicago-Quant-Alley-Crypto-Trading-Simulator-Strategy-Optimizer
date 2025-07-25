import numpy as np
import matplotlib.pyplot as plt

class UCB1:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_count = 0

    def select_arm(self):
        if self.total_count < self.n_arms:
            # Pull each arm once in the beginning
            return self.total_count
        else:
            ucb_values = self.values + np.sqrt(2 * np.log(self.total_count) / self.counts)
            return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.total_count += 1
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = value + (reward - value) / n

def plot_rewards(rewards, avg_reward, best_arm_freq, true_probs):
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Best arm selection frequency: {best_arm_freq:.4f}")
    print(f"True arm means: {np.round(true_probs, 2)}")
    plt.plot(np.cumsum(rewards), label="Cumulative Reward")
    plt.title("UCB1 Bandit Performance")
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid()
    plt.savefig("plots/ucb1_bandit_performance.png")

def simulate_ucb1(T=10000, K=10):
    true_probs = np.random.rand(K)
    best_arm = np.argmax(true_probs)
    
    algo = UCB1(n_arms=K)
    
    rewards = []
    best_arm_counts = 0

    for t in range(T):
        arm = algo.select_arm()
        reward = np.random.rand() < true_probs[arm]
        algo.update(arm, reward)
        rewards.append(reward)
        if arm == best_arm:
            best_arm_counts += 1
    
    avg_reward = np.mean(rewards)
    return rewards, avg_reward, best_arm_counts / T, true_probs

rewards, avg_reward, best_arm_freq, true_probs = simulate_ucb1()
plot_rewards(rewards, avg_reward, best_arm_freq, true_probs)
