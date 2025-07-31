import numpy as np
import matplotlib.pyplot as plt
import os

class UCB1:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_count = 0
        self.ucb_history = []
        self.chosen_arms = []

    def select_arm(self):
        if self.total_count < self.n_arms:
            chosen = self.total_count
            self.ucb_history.append(np.ones(self.n_arms))
            self.chosen_arms.append(chosen)
            return chosen
        else:
            ucb_values = self.values + np.sqrt(2 * np.log(self.total_count) / self.counts)
            self.ucb_history.append(ucb_values.copy())
            chosen = np.argmax(ucb_values)
            self.chosen_arms.append(chosen)
            return chosen

    def update(self, chosen_arm, reward):
        self.total_count += 1
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = value + (reward - value) / n

def plot_evaluation(rewards, cumulative_regret, best_arm_freq, freq_over_time, ucb_history, chosen_arms, true_probs):
    os.makedirs("plots", exist_ok=True)
    print(f"Best arm selection frequency: {best_arm_freq:.4f}")
    print(f"True arm means: {np.round(true_probs, 2)}")
    
    plt.figure(figsize=(16,8))

    plt.subplot(2,2,1)
    plt.plot(np.cumsum(rewards), label="Cumulative Reward")
    plt.plot(cumulative_regret, label="Cumulative Regret")
    plt.title("UCB1: Reward & Regret")
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

    timeline = np.arange(len(ucb_history))
    k = len(true_probs)
    arms_to_plot = [np.argmax(true_probs), np.argmin(true_probs), np.random.randint(0, k)]
    plt.subplot(2,2,3)
    for i in arms_to_plot:
        ucb_curve = [u[i] for u in ucb_history]
        plt.plot(timeline, ucb_curve, label=f"Arm {i} UCB")
    observed = [true_probs[a] for a in chosen_arms]
    plt.scatter(timeline, observed, s=5, c='k', alpha=0.3, label="True Mean of Pulled Arm")
    plt.title("UCB for Representative Arms")
    plt.xlabel("Time Steps")
    plt.ylabel("UCB / Reward Mean")
    plt.legend(fontsize="small")
    plt.grid()

    plt.tight_layout()
    plt.savefig("plots/ucb1_evaluation.png")
    plt.close()

def simulate_ucb1(T=10000, K=10):
    true_probs = np.random.rand(K)
    best_arm = np.argmax(true_probs)
    best_mean = true_probs[best_arm]

    algo = UCB1(n_arms=K)
    
    rewards = []
    best_arm_counts = 0
    cumulative_regret = []
    freq_over_time = []
    total_regret = 0

    for t in range(T):
        arm = algo.select_arm()
        reward = np.random.rand() < true_probs[arm]
        algo.update(arm, reward)
        rewards.append(reward)
        regret = best_mean - reward
        total_regret += regret
        cumulative_regret.append(total_regret)
        if arm == best_arm:
            best_arm_counts += 1
        freq_over_time.append(best_arm_counts / (t+1))

    avg_reward = np.mean(rewards)
    ucb_history = algo.ucb_history
    chosen_arms = algo.chosen_arms
    return rewards, avg_reward, best_arm_counts / T, true_probs, cumulative_regret, freq_over_time, ucb_history, chosen_arms

rewards, avg_reward, best_arm_freq, true_probs, cumulative_regret, freq_over_time, ucb_history, chosen_arms = simulate_ucb1()
plot_evaluation(rewards, cumulative_regret, best_arm_freq, freq_over_time, ucb_history, chosen_arms, true_probs)
