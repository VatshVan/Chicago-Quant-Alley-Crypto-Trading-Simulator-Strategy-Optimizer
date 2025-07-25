import numpy as np
import matplotlib.pyplot as plt

class LinUCB:
    def __init__(self, n_arms, d, alpha=1.0):
        self.n_arms = n_arms
        self.d = d
        self.alpha = alpha
        self.A = [np.identity(d) for _ in range(n_arms)]
        self.b = [np.zeros(d) for _ in range(n_arms)]

    def select_arm(self, contexts):
        p_values = []
        for i in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            x = contexts[i]
            p = x.T @ theta + self.alpha * np.sqrt(x.T @ A_inv @ x)
            p_values.append(p)
        return np.argmax(p_values)

    def update(self, chosen_arm, reward, context):
        x = context
        self.A[chosen_arm] += np.outer(x, x)
        self.b[chosen_arm] += reward * x

def plot_rewards(rewards, avg_reward, best_arm_freq, true_probs):
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Best arm selection frequency: {best_arm_freq:.4f}")
    print(f"True arm means: {np.round(true_probs, 2)}")
    plt.plot(np.cumsum(rewards), label="Cumulative Reward")
    plt.title("LinUCB Bandit Performance")
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid()
    plt.show()


def simulate_linucb(T=10000, K=10, d=5, alpha=1.0):
    # Random true parameter for each arm
    algo = LinUCB(n_arms=K, d=d, alpha=alpha)
    true_theta = [np.random.rand(d) for _ in range(K)]

    rewards = []
    best_arm_counts = 0

    for t in range(T):
        contexts = [np.random.rand(d) for _ in range(K)]
        expected_rewards = [context @ true_theta[i] for i, context in enumerate(contexts)]
        best_arm = np.argmax(expected_rewards)

        arm = algo.select_arm(contexts)
        noise = np.random.normal(0, 0.1)
        reward = contexts[arm] @ true_theta[arm] + noise
        reward = np.clip(reward, 0, 1)

        algo.update(arm, reward, contexts[arm])
        rewards.append(reward)
        if arm == best_arm:
            best_arm_counts += 1
    
    avg_reward = np.mean(rewards)
    # plot_rewards(rewards, avg_reward, best_arm_freq, [theta.mean() for theta in true_theta])
    return rewards, avg_reward, best_arm_counts / T

rewards, avg_reward, best_arm_freq = simulate_linucb()