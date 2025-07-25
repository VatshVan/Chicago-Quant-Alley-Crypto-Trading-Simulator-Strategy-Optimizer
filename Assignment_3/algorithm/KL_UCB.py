import numpy as np
import matplotlib.pyplot as plt

class KLUCB:
    def __init__(self, n_arms, c=3):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_count = 0
        self.c = c

    def select_arm(self):
        self.total_count += 1
        if self.total_count <= self.n_arms:
            return self.total_count - 1  # Pull each arm once
        else:
            ucbs = np.zeros(self.n_arms)
            threshold = np.log(self.total_count) + self.c * np.log(np.log(self.total_count))
            for i in range(self.n_arms):
                if self.counts[i] > 0:
                    ucbs[i] = solve_kl_ucb(self.values[i], self.counts[i], threshold)
                else:
                    ucbs[i] = 1.0  # force exploration
            return np.argmax(ucbs)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = value + (reward - value) / n

def kl_divergence(p, q):
    eps = 1e-15  # to avoid log(0)
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def solve_kl_ucb(p_hat, n, threshold):
    # Binary search to find q in [p_hat, 1] s.t. n * KL(p_hat || q) <= threshold
    lower, upper = p_hat, 1.0
    for _ in range(25):  # 25 iterations is typically sufficient
        q = (lower + upper) / 2
        if n * kl_divergence(p_hat, q) > threshold:
            upper = q
        else:
            lower = q
    return (lower + upper) / 2

def plot_rewards(rewards, avg_reward, best_arm_freq, true_probs):
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Best arm selection frequency: {best_arm_freq:.4f}")
    print(f"True arm means: {np.round(true_probs, 2)}")
    plt.plot(np.cumsum(rewards), label="Cumulative Reward")
    plt.title("KLUCB Bandit Performance")
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Reward / Regret")
    plt.legend()
    plt.grid()
    plt.savefig("plots/kl_ucb_results.png")

def simulate_kl_ucb(T=10000, K=10):
    true_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9])
    best_arm = np.argmax(true_probs)

    algo = KLUCB(n_arms=K)
    
    rewards = []
    best_arm_counts = 0
    cumulative_reward = 0

    for t in range(T):
        arm = algo.select_arm()
        reward = np.random.rand() < true_probs[arm]
        algo.update(arm, reward)
        rewards.append(reward)
        cumulative_reward += reward
        if arm == best_arm:
            best_arm_counts += 1

    avg_reward = np.mean(rewards)
    return rewards, avg_reward, best_arm_counts / T, true_probs

rewards, avg_reward, best_arm_freq, true_probs = simulate_kl_ucb()
plot_rewards(rewards, avg_reward, best_arm_freq, true_probs)