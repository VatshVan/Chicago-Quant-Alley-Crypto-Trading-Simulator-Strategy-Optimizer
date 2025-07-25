import numpy as np
import matplotlib.pyplot as plt
import math

class KLLUCB:
    def __init__(self, K=10, delta=0.05, true_probs=None):
        self.K = K
        self.delta = delta
        self.true_probs = true_probs if true_probs is not None else np.random.rand(K)
        self.best_arm = np.argmax(self.true_probs)
        self.counts = np.zeros(K)
        self.sums = np.zeros(K)
        self.reward_history = []

    def beta(self, t):
        return np.log(3 * t / self.delta)

    def run(self, max_total_pulls=10000):
        for i in range(self.K):
            reward = np.random.rand() < self.true_probs[i]
            self.sums[i] += reward
            self.counts[i] += 1
            self.reward_history.append(reward)

        t = self.K
        while np.sum(self.counts) < max_total_pulls:
            means = self.sums / self.counts
            threshold = self.beta(t)

            ucb = np.array([kl_ucb_bound(means[i], self.counts[i], threshold, upper=True) for i in range(self.K)])
            lcb = np.array([kl_ucb_bound(means[i], self.counts[i], threshold, upper=False) for i in range(self.K)])

            i_best = np.argmax(means)
            competitors = [i for i in range(self.K) if i != i_best]
            j = max(competitors, key=lambda i: ucb[i])

            if lcb[i_best] >= ucb[j]:
                break

            for i in [i_best, j]:
                reward = np.random.rand() < self.true_probs[i]
                self.sums[i] += reward
                self.counts[i] += 1
                self.reward_history.append(reward)
                t += 1

        return i_best

    def report(self):
        avg_reward = np.mean(self.reward_history)
        identified = np.argmax(self.sums / self.counts)
        is_correct = identified == self.best_arm
        return {
            "identified_best": identified,
            "true_best": self.best_arm,
            "is_correct": is_correct,
            "avg_reward": avg_reward,
            "cumulative_rewards": np.cumsum(self.reward_history)
        }

def kl(p, q):
    eps = 1e-12
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def kl_ucb_bound(p_hat, n, threshold, upper=True):
    lower, upper_bound = (p_hat, 1.0) if upper else (0.0, p_hat)
    for _ in range(25):
        q = (lower + upper_bound) / 2
        divergence = n * kl(p_hat, q)
        if divergence > threshold:
            if upper:
                upper_bound = q
            else:
                lower = q
        else:
            if upper:
                lower = q
            else:
                upper_bound = q
    return (lower + upper_bound) / 2

def plot_rewards(result):
    print(f"Identified best arm: {result['identified_best']}")
    print(f"True best arm: {result['true_best']}")
    print(f"Correct identification: {result['is_correct']}")
    print(f"Average reward collected: {result['avg_reward']:.4f}")

    plt.plot(result["cumulative_rewards"], label="Cumulative Reward")
    plt.title("KL-LUCB (Best Arm Identification)")
    plt.xlabel("Total Pulls")
    plt.ylabel("Cumulative Reward")
    plt.grid()
    plt.legend()
    plt.savefig("plots/kl_lucb_results.png")
def simulate_kl_lucb():
    kl_lucb = KLLUCB(K=10, delta=0.05, true_probs=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]))
    kl_lucb.run(max_total_pulls=10000)
    result = kl_lucb.report()
    return result

result = simulate_kl_lucb()
plot_rewards(result)