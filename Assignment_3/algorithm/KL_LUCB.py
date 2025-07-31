import numpy as np
import matplotlib.pyplot as plt
import math
import os

class KLLUCB:
    def __init__(self, K=10, delta=0.05, true_probs=None):
        self.K = K
        self.delta = delta
        self.true_probs = true_probs if true_probs is not None else np.random.rand(K)
        self.best_arm = np.argmax(self.true_probs)
        self.counts = np.zeros(K)
        self.sums = np.zeros(K)
        self.reward_history = []
        self.mean_history = []
        self.ucb_history = []
        self.lcb_history = []
        self.pull_timeline = []
        self.chosen_arms = []

    def beta(self, t):
        return np.log(3 * t / self.delta)

    def run(self, max_total_pulls=10000):
        for i in range(self.K):
            reward = np.random.rand() < self.true_probs[i]
            self.sums[i] += reward
            self.counts[i] += 1
            self.reward_history.append(reward)
            self.chosen_arms.append(i)

        t = self.K
        while np.sum(self.counts) < max_total_pulls:
            means = self.sums / self.counts
            threshold = self.beta(t)

            ucb = np.array([kl_ucb_bound(means[i], self.counts[i], threshold, upper=True) for i in range(self.K)])
            lcb = np.array([kl_ucb_bound(means[i], self.counts[i], threshold, upper=False) for i in range(self.K)])
            self.mean_history.append(means.copy())
            self.ucb_history.append(ucb.copy())
            self.lcb_history.append(lcb.copy())
            self.pull_timeline.append(t)

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
                self.chosen_arms.append(i)
                t += 1

        return i_best

    def report(self):
        avg_reward = np.mean(self.reward_history)
        identified = np.argmax(self.sums / self.counts)
        is_correct = identified == self.best_arm
        best_mean = self.true_probs[self.best_arm]
        regret_history = np.array([best_mean - r for r in self.reward_history])
        cumulative_regret = np.cumsum(regret_history)
        freq_over_time = []
        best_count = 0
        for idx, a in enumerate(self.chosen_arms):
            if a == self.best_arm:
                best_count += 1
            freq_over_time.append(best_count / (idx + 1))
        return {
            "identified_best": identified,
            "true_best": self.best_arm,
            "is_correct": is_correct,
            "avg_reward": avg_reward,
            "cumulative_rewards": np.cumsum(self.reward_history),
            "cumulative_regret": cumulative_regret,
            "freq_over_time": freq_over_time,
            "ucb_history": self.ucb_history,
            "lcb_history": self.lcb_history,
            "mean_history": self.mean_history,
            "pull_timeline": self.pull_timeline,
            "chosen_arms": self.chosen_arms
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

def plot_evaluation(results):
    os.makedirs("plots", exist_ok=True)
    print(f"Identified best arm: {results['identified_best']}")
    print(f"True best arm: {results['true_best']}")
    print(f"Correct identification: {results['is_correct']}")
    print(f"Average reward collected: {results['avg_reward']:.4f}")

    plt.figure(figsize=(16, 8))

    plt.subplot(2, 2, 1)
    plt.plot(results["cumulative_rewards"], label="Cumulative Reward")
    plt.plot(results["cumulative_regret"], label="Cumulative Regret")
    plt.title("KL-LUCB: Reward & Regret")
    plt.xlabel("Total Pulls")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(results["freq_over_time"], label="Best Arm Frequency")
    plt.title("Best Arm Selection Frequency")
    plt.xlabel("Total Pulls")
    plt.ylabel("Fraction")
    plt.legend()
    plt.grid()
    timeline = np.array(results["pull_timeline"])
    for i in range(len(results["ucb_history"][0])):
        plt.subplot(2, 2, 3)
        ucb_curve = [ucb[i] for ucb in results["ucb_history"]]
        lcb_curve = [lcb[i] for lcb in results["lcb_history"]]
        plt.plot(timeline, ucb_curve, label=f"Arm {i} UCB", alpha=0.6)
        plt.plot(timeline, lcb_curve, label=f"Arm {i} LCB", alpha=0.6, linestyle="--")
    plt.title("Confidence Bounds per Arm (KL-LUCB)")
    plt.xlabel("Step (t)")
    plt.ylabel("Bound Value")
    plt.grid()
    plt.legend(fontsize='x-small', ncol=2)

    plt.subplot(2, 2, 4)
    for i in range(len(results["mean_history"][0])):
        mean_curve = [m[i] for m in results["mean_history"]]
        plt.plot(timeline, mean_curve, label=f"Arm {i} Mean", alpha=0.6)
    plt.title("Empirical Means per Arm")
    plt.xlabel("Step (t)")
    plt.ylabel("Empirical Mean")
    plt.grid()
    plt.legend(fontsize="x-small", ncol=2)

    plt.suptitle("KL-LUCB Bandit: Full Evaluation")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("plots/kl_lucb_evaluation.png")
    plt.close()

def simulate_kl_lucb():
    kl_lucb = KLLUCB(K=10, delta=0.05, true_probs=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]))
    kl_lucb.run(max_total_pulls=10000)
    result = kl_lucb.report()
    return result

result = simulate_kl_lucb()
plot_evaluation(result)
