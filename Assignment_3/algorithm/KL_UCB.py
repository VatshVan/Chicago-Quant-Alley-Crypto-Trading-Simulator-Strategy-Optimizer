import numpy as np
import matplotlib.pyplot as plt
import os

class KLUCB:
    def __init__(self, n_arms, c=3):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_count = 0
        self.c = c
        self.ucb_history = []
        self.selected_arms = []

    def select_arm(self):
        self.total_count += 1
        if self.total_count <= self.n_arms:
            chosen = self.total_count - 1
            self.ucb_history.append(np.ones(self.n_arms))
            self.selected_arms.append(chosen)
            return chosen

        ucbs = np.zeros(self.n_arms)
        threshold = np.log(self.total_count) + self.c * np.log(np.log(self.total_count))
        for i in range(self.n_arms):
            if self.counts[i] > 0:
                ucbs[i] = solve_kl_ucb(self.values[i], self.counts[i], threshold)
            else:
                ucbs[i] = 1.0
        self.ucb_history.append(ucbs.copy())
        chosen = np.argmax(ucbs)
        self.selected_arms.append(chosen)
        return chosen

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = value + (reward - value) / n

    def get_ucb_history(self):
        return self.ucb_history

    def get_selected_arms(self):
        return self.selected_arms

def kl_divergence(p, q):
    eps = 1e-15
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def solve_kl_ucb(p_hat, n, threshold):
    lower, upper = p_hat, 1.0
    for _ in range(25):
        q = (lower + upper) / 2
        if n * kl_divergence(p_hat, q) > threshold:
            upper = q
        else:
            lower = q
    return (lower + upper) / 2

def plot_evaluation(rewards, cumulative_regret, best_arm_freq, freq_over_time, ucbs, selected_arms, true_probs):
    os.makedirs("plots", exist_ok=True)
    print(f"Best arm selection frequency: {best_arm_freq:.4f}")
    print(f"True arm means: {np.round(true_probs, 2)}")

    plt.figure(figsize=(16, 8))

    plt.subplot(2, 2, 1)
    plt.plot(np.cumsum(rewards), label="Cumulative Reward")
    plt.plot(cumulative_regret, label="Cumulative Regret")
    plt.title("KLUCB: Reward & Regret")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(freq_over_time, label="Best Arm Frequency")
    plt.title("Best Arm Selection Frequency")
    plt.xlabel("Time Steps")
    plt.ylabel("Fraction")
    plt.legend()
    plt.grid()

    timeline = np.arange(len(ucbs))
    k = len(true_probs)
    arms_to_plot = [np.argmax(true_probs), np.argmin(true_probs), np.random.randint(0, k)]
    for i in arms_to_plot:
        plt.subplot(2, 2, 3)
        ucb_curve = [u[i] for u in ucbs]
        plt.plot(timeline, ucb_curve, label=f"Arm {i} UCB")
    observed = [true_probs[a] for a in selected_arms]
    plt.scatter(timeline, observed, s=5, c='k', alpha=0.3, label="Observed Reward Mean")
    plt.title("UCB Bounds (Selected Arms)")
    plt.xlabel("Time Steps")
    plt.ylabel("UCB Value / True Mean")
    plt.legend(fontsize="small")
    plt.grid()

    plt.tight_layout()
    plt.savefig("plots/kl_ucb_full_evaluation.png")
    plt.close()

def simulate_kl_ucb(T=10000, K=10):
    true_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9])
    best_arm = np.argmax(true_probs)
    best_mean = true_probs[best_arm]
    algo = KLUCB(n_arms=K)

    rewards = []
    best_arm_counts = 0
    cumulative_regret = []
    total_regret = 0
    freq_over_time = []

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
        freq_over_time.append(best_arm_counts / (t + 1))

    avg_reward = np.mean(rewards)
    ucbs = algo.get_ucb_history()
    selected_arms = algo.get_selected_arms()
    return rewards, avg_reward, best_arm_counts / T, true_probs, cumulative_regret, freq_over_time, ucbs, selected_arms

rewards, avg_reward, best_arm_freq, true_probs, cumulative_regret, freq_over_time, ucbs, selected_arms = simulate_kl_ucb()
plot_evaluation(rewards, cumulative_regret, best_arm_freq, freq_over_time, ucbs, selected_arms, true_probs)
