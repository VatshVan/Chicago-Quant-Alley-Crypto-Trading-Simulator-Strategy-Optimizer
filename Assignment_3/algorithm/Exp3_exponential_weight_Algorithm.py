import numpy as np
import matplotlib.pyplot as plt
import os

class Exp3:
    def __init__(self, n_arms, gamma=0.07):
        self.n_arms = n_arms
        self.gamma = gamma
        self.weights = np.ones(n_arms)
        self.last_probs = np.ones(n_arms) / n_arms

    def select_arm(self):
        total_weight = np.sum(self.weights)
        probs = (1 - self.gamma) * (self.weights / total_weight) + self.gamma / self.n_arms
        self.last_probs = probs
        return np.random.choice(self.n_arms, p=probs)

    def update(self, chosen_arm, reward):
        probs = self.last_probs
        x_hat = reward / probs[chosen_arm] if probs[chosen_arm] > 0 else 0
        self.weights[chosen_arm] *= np.exp((self.gamma * x_hat) / self.n_arms)

def generate_adversarial_rewards(T, K, rotation=1000):
    rewards = np.zeros((T, K))
    for t in range(T):
        best_arm = (t // rotation) % K
        rewards[t, best_arm] = 1
    return rewards

def plot_evaluation(rewards, cumulative_regret, best_arm_freq, freq_over_time):
    os.makedirs("plots", exist_ok=True)
    print(f"Best arm selection frequency: {best_arm_freq:.4f}")
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(rewards), label="Cumulative Reward")
    plt.plot(cumulative_regret, label="Cumulative Regret")
    plt.title("Exp3 (Adversarial Bandit Setting)")
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
    plt.savefig("plots/exp3_adversarial_evaluation.png")
    plt.close()

def simulate_exp3_adversarial(T=10000, K=10, rotation=1000):
    reward_matrix = generate_adversarial_rewards(T, K, rotation)
    best_arm_sequence = np.argmax(reward_matrix, axis=1)
    algo = Exp3(n_arms=K, gamma=0.07)

    rewards = []
    cumulative_regret = []
    total_regret = 0
    freq_over_time = []
    best_arm_counts = 0

    for t in range(T):
        arm = algo.select_arm()
        reward = reward_matrix[t, arm]
        algo.update(arm, reward)
        rewards.append(reward)
        regret = 1 - reward
        total_regret += regret
        cumulative_regret.append(total_regret)
        if arm == best_arm_sequence[t]:
            best_arm_counts += 1
        freq_over_time.append(best_arm_counts / (t + 1))

    avg_reward = np.mean(rewards)
    return rewards, avg_reward, best_arm_counts / T, cumulative_regret, freq_over_time

rewards, avg_reward, best_arm_freq, cumulative_regret, freq_over_time = simulate_exp3_adversarial()
plot_evaluation(rewards, cumulative_regret, best_arm_freq, freq_over_time)
