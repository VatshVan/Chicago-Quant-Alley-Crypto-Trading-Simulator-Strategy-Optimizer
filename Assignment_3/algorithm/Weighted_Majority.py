import numpy as np
import matplotlib.pyplot as plt
import os

class WeightedMajority:
    def __init__(self, n_arms, eta=0.2):
        self.n_arms = n_arms
        self.weights = np.ones(n_arms)
        self.eta = eta
        self.probs_history = []
        self.chosen_arms = []

    def select_arm(self):
        probs = self.weights / np.sum(self.weights)
        self.probs_history.append(probs.copy())
        chosen = np.random.choice(self.n_arms, p=probs)
        self.chosen_arms.append(chosen)
        return chosen

    def update(self, loss_vector):
        for i in range(self.n_arms):
            self.weights[i] *= (1 - self.eta) ** loss_vector[i]

def generate_adversarial_loss_matrix(T, K, rotation=1000):
    loss_matrix = np.ones((T, K))
    for t in range(T):
        best_arm = (t // rotation) % K
        loss_matrix[t, best_arm] = 0
    return loss_matrix

def plot_evaluation(rewards, cumulative_regret, best_arm_freq, freq_over_time, probs_history, chosen_arms, best_arm_seq):
    os.makedirs("plots", exist_ok=True)
    print(f"Best arm selection frequency: {best_arm_freq:.4f}")

    plt.figure(figsize=(16,8))

    plt.subplot(2,2,1)
    plt.plot(np.cumsum(rewards), label="Cumulative Reward")
    plt.plot(cumulative_regret, label="Cumulative Regret")
    plt.title("Weighted Majority: Reward & Regret")
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

    timeline = np.arange(len(probs_history))
    k = probs_history[0].shape[0]
    arms_to_plot = [np.bincount(best_arm_seq).argmax(), np.bincount(best_arm_seq).argmin(), np.random.randint(0, k)]
    plt.subplot(2,2,3)
    for i in arms_to_plot:
        prob_curve = [p[i] for p in probs_history]
        plt.plot(timeline, prob_curve, label=f"Arm {i} Prob")
    plt.title("Probability for Representative Arms")
    plt.xlabel("Time Steps")
    plt.ylabel("Probability")
    plt.legend(fontsize="small")
    plt.grid()

    plt.tight_layout()
    plt.savefig("plots/weighted_majority_evaluation.png")
    plt.close()

def simulate_weighted_majority_adversarial(T=10000, K=10, rotation=1000):
    loss_matrix = generate_adversarial_loss_matrix(T, K, rotation)
    best_arm_sequence = np.argmin(loss_matrix, axis=1)

    algo = WeightedMajority(n_arms=K, eta=0.2)

    rewards = []
    best_arm_counts = 0
    cumulative_regret = []
    freq_over_time = []
    total_regret = 0

    for t in range(T):
        arm = algo.select_arm()
        loss_vector = loss_matrix[t]
        reward = 1 - loss_vector[arm]
        algo.update(loss_vector)
        rewards.append(reward)
        regret = 1 - reward
        total_regret += regret
        cumulative_regret.append(total_regret)
        if arm == best_arm_sequence[t]:
            best_arm_counts += 1
        freq_over_time.append(best_arm_counts / (t+1))

    avg_reward = np.mean(rewards)
    probs_history = algo.probs_history
    chosen_arms = algo.chosen_arms
    return rewards, avg_reward, best_arm_counts / T, best_arm_sequence, cumulative_regret, freq_over_time, probs_history, chosen_arms

rewards, avg_reward, best_arm_freq, best_arm_seq, cumulative_regret, freq_over_time, probs_history, chosen_arms = simulate_weighted_majority_adversarial()
plot_evaluation(rewards, cumulative_regret, best_arm_freq, freq_over_time, probs_history, chosen_arms, best_arm_seq)
