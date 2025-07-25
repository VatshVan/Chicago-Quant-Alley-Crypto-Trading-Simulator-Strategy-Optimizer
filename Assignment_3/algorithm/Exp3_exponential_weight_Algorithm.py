import numpy as np
import matplotlib.pyplot as plt

class Exp3:
    def __init__(self, n_arms, gamma=0.07):
        self.n_arms = n_arms
        self.gamma = gamma
        self.weights = np.ones(n_arms)

    def select_arm(self):
        total_weight = np.sum(self.weights)
        probs = (1 - self.gamma) * (self.weights / total_weight) + self.gamma / self.n_arms
        self.last_probs = probs
        return np.random.choice(self.n_arms, p=probs)

    def update(self, chosen_arm, reward):
        probs = self.last_probs
        x_hat = reward / probs[chosen_arm]
        self.weights[chosen_arm] *= np.exp((self.gamma * x_hat) / self.n_arms)

def generate_adversarial_rewards(T, K, rotation=1000):
    rewards = np.zeros((T, K))
    for t in range(T):
        best_arm = (t // rotation) % K
        rewards[t, best_arm] = 1  # Only best arm gives reward 1
    return rewards

def plot_rewards(rewards, avg_reward, best_arm_freq):
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Best arm selection frequency: {best_arm_freq:.4f}")
    plt.plot(np.cumsum(rewards), label="Cumulative Reward")
    plt.title("Exp3 (Adversarial Bandit Setting)")
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid()
    plt.savefig("plots/exp3_adversarial_performance.png")

def simulate_exp3_adversarial(T=10000, K=10, rotation=1000):
    reward_matrix = generate_adversarial_rewards(T, K, rotation)
    best_arm_sequence = np.argmax(reward_matrix, axis=1)

    algo = Exp3(n_arms=K, gamma=0.07)

    rewards = []
    best_arm_counts = 0

    for t in range(T):
        arm = algo.select_arm()
        reward = reward_matrix[t, arm]
        algo.update(arm, reward)
        rewards.append(reward)
        if arm == best_arm_sequence[t]:
            best_arm_counts += 1

    avg_reward = np.mean(rewards)
    return rewards, avg_reward, best_arm_counts / T

# Run the adversarial simulation
rewards, avg_reward, best_arm_freq = simulate_exp3_adversarial()
plot_rewards(rewards, avg_reward, best_arm_freq)