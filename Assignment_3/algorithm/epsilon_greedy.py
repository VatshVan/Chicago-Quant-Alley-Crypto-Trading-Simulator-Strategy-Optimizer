import numpy as np
import matplotlib.pyplot as plt
import os

class EpsilonGreedy:
    def __init__(self, n_arms, epsilon):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)  # Number of times each arm was played
        self.values = np.zeros(n_arms)  # Average reward for each arm

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            # Explore
            return np.random.randint(self.n_arms)
        else:
            # Exploit
            return np.argmax(self.values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # Update average reward using incremental formula
        self.values[chosen_arm] = value + (reward - value) / n

def plot_rewards(rewards, avg_reward, best_arm_freq, true_probs):
    os.makedirs("plots", exist_ok=True)
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Best arm selection frequency: {best_arm_freq:.4f}")
    print(f"True arm means: {np.round(true_probs, 2)}")
    plt.plot(np.cumsum(rewards), label="Cumulative Reward")
    plt.title("Epsilon-Greedy Bandit Performance (ϵ=0.1)")
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid()
    plt.savefig("plots/epsilon_greedy_performance.png")
    plt.close()

def simulate_epsilon_greedy(epsilon, T=10000, K=10):
    # True means of each arm (Bernoulli bandit)
    true_probs = np.random.rand(K)
    best_arm = np.argmax(true_probs)
    
    algo = EpsilonGreedy(n_arms=K, epsilon=epsilon)
    
    rewards = []
    best_arm_counts = 0
    
    for t in range(T):
        arm = algo.select_arm()
        reward = np.random.rand() < true_probs[arm]  # Bernoulli reward
        algo.update(arm, reward)
        rewards.append(reward)
        if arm == best_arm:
            best_arm_counts += 1
    
    avg_reward = np.mean(rewards)
    return rewards, avg_reward, best_arm_counts / T, true_probs

rewards, avg_reward, best_arm_freq, true_probs = simulate_epsilon_greedy(epsilon=0.1)

plot_rewards(rewards, avg_reward, best_arm_freq, true_probs)