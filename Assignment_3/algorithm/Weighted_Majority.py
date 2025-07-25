import numpy as np
import matplotlib.pyplot as plt

class WeightedMajority:
    def __init__(self, n_arms, eta=0.2):
        self.n_arms = n_arms
        self.weights = np.ones(n_arms)
        self.eta = eta

    def select_arm(self):
        probs = self.weights / np.sum(self.weights)
        return np.random.choice(self.n_arms, p=probs)

    def update(self, loss_vector):
        # Full information: update all weights
        for i in range(self.n_arms):
            self.weights[i] *= (1 - self.eta) ** loss_vector[i]

def generate_adversarial_loss_matrix(T, K, rotation=1000):
    loss_matrix = np.ones((T, K))  # All losses = 1
    for t in range(T):
        best_arm = (t // rotation) % K
        loss_matrix[t, best_arm] = 0  # Best arm gets loss 0 (i.e., reward 1)
    return loss_matrix

def plot_rewards(rewards, avg_reward, best_arm_freq, true_best_arm_seq):
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Best arm selection frequency: {best_arm_freq:.4f}")
    plt.plot(np.cumsum(rewards), label="Cumulative Reward")
    plt.title("Weighted Majority (Adversarial Full Info)")
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid()
    plt.savefig("plots/weighted_majority_adversarial.png")

def simulate_weighted_majority_adversarial(T=10000, K=10, rotation=1000):
    loss_matrix = generate_adversarial_loss_matrix(T, K, rotation)
    best_arm_sequence = np.argmin(loss_matrix, axis=1)

    algo = WeightedMajority(n_arms=K, eta=0.2)

    rewards = []
    best_arm_counts = 0

    for t in range(T):
        arm = algo.select_arm()
        loss_vector = loss_matrix[t]
        reward = 1 - loss_vector[arm]  # Convert back to reward
        algo.update(loss_vector)
        rewards.append(reward)
        if arm == best_arm_sequence[t]:
            best_arm_counts += 1

    avg_reward = np.mean(rewards)
    return rewards, avg_reward, best_arm_counts / T, best_arm_sequence

# Run the simulation
rewards, avg_reward, best_arm_freq, best_arm_seq = simulate_weighted_majority_adversarial()
plot_rewards(rewards, avg_reward, best_arm_freq, best_arm_seq)
