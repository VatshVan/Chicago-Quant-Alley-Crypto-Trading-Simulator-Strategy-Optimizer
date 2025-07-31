import numpy as np
import matplotlib.pyplot as plt
import os

class LinUCB:
    def __init__(self, n_arms, d, alpha=1.0):
        self.n_arms = n_arms
        self.d = d
        self.alpha = alpha
        self.A = [np.identity(d) for _ in range(n_arms)]
        self.b = [np.zeros(d) for _ in range(n_arms)]
        self.ucb_history = []
        self.chosen_arms = []

    def select_arm(self, contexts):
        p_values = []
        for i in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            x = contexts[i]
            p = x.T @ theta + self.alpha * np.sqrt(x.T @ A_inv @ x)
            p_values.append(p)
        chosen = np.argmax(p_values)
        self.chosen_arms.append(chosen)
        self.ucb_history.append(p_values.copy())
        return chosen

    def update(self, chosen_arm, reward, context):
        x = context
        self.A[chosen_arm] += np.outer(x, x)
        self.b[chosen_arm] += reward * x

def plot_evaluation(rewards, cumulative_regret, best_arm_freq, freq_over_time, ucb_history, chosen_arms):
    os.makedirs("plots", exist_ok=True)
    print(f"Best arm selection frequency: {best_arm_freq:.4f}")

    plt.figure(figsize=(16,8))

    plt.subplot(2,2,1)
    plt.plot(np.cumsum(rewards), label="Cumulative Reward")
    plt.plot(cumulative_regret, label="Cumulative Regret")
    plt.title("LinUCB: Reward & Regret")
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
    n_arms_to_plot = min(3, len(ucb_history[0]))  # Plot up to 3 arms
    for i in range(n_arms_to_plot):
        ucb_curve = [step[i] for step in ucb_history]
        plt.subplot(2,2,3)
        plt.plot(timeline, ucb_curve, label=f"Arm {i} UCB")
    moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
    plt.subplot(2,2,3)
    plt.plot(np.arange(100-1, len(rewards)), moving_avg, c='k', linestyle='--', label="Observed Mean (window=100)")
    plt.title("UCB Values & Observed Reward")
    plt.xlabel("Time Steps")
    plt.ylabel("UCB Value / Reward")
    plt.legend(fontsize="small")
    plt.grid()

    plt.tight_layout()
    plt.savefig("plots/linucb_evaluation.png")
    plt.close()

def simulate_linucb(T=10000, K=10, d=5, alpha=1.0):
    algo = LinUCB(n_arms=K, d=d, alpha=alpha)
    true_theta = [np.random.rand(d) for _ in range(K)]

    rewards = []
    best_arm_counts = 0
    freq_over_time = []
    cumulative_regret = []
    total_regret = 0

    for t in range(T):
        contexts = [np.random.rand(d) for _ in range(K)]
        expected_rewards = [context @ true_theta[i] for i, context in enumerate(contexts)]
        best_arm = np.argmax(expected_rewards)
        best_mean = expected_rewards[best_arm]

        arm = algo.select_arm(contexts)
        noise = np.random.normal(0, 0.1)
        reward = contexts[arm] @ true_theta[arm] + noise
        reward = np.clip(reward, 0, 1)

        algo.update(arm, reward, contexts[arm])
        rewards.append(reward)

        regret = best_mean - reward
        total_regret += regret
        cumulative_regret.append(total_regret)
        if arm == best_arm:
            best_arm_counts += 1
        freq_over_time.append(best_arm_counts / (t + 1))

    avg_reward = np.mean(rewards)
    ucb_history = algo.ucb_history
    chosen_arms = algo.chosen_arms
    return rewards, avg_reward, best_arm_counts / T, cumulative_regret, freq_over_time, ucb_history, chosen_arms

rewards, avg_reward, best_arm_freq, cumulative_regret, freq_over_time, ucb_history, chosen_arms = simulate_linucb()
plot_evaluation(rewards, cumulative_regret, best_arm_freq, freq_over_time, ucb_history, chosen_arms)
