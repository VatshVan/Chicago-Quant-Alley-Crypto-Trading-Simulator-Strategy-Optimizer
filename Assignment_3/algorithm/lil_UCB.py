import numpy as np
import matplotlib.pyplot as plt

class LilUCB:
    def __init__(self, K=10, delta=0.05, epsilon=0.1, true_probs=None):
        self.K = K
        self.delta = delta
        self.epsilon = epsilon
        self.true_probs = true_probs if true_probs is not None else np.random.rand(K)
        self.best_arm = np.argmax(self.true_probs)
        self.counts = np.zeros(K)
        self.sums = np.zeros(K)
        self.reward_history = []

    def ucb(self, i, t):
        if self.counts[i] == 0:
            return float('inf')
        mu_hat = self.sums[i] / self.counts[i]
        beta = (1 + np.sqrt(self.epsilon)) * np.sqrt(
            (2 * (1 + self.epsilon) * np.log(np.log((1 + self.epsilon) * self.counts[i] + 2) / self.delta))
            / self.counts[i]
        )
        return mu_hat + beta

    def run(self, max_total_pulls=10000):
        t = 0

        # Pull each arm once
        for i in range(self.K):
            reward = np.random.rand() < self.true_probs[i]
            self.sums[i] += reward
            self.counts[i] += 1
            self.reward_history.append(reward)
            t += 1

        while np.sum(self.counts) < max_total_pulls:
            ucbs = [self.ucb(i, t) for i in range(self.K)]
            i_best = np.argmax(ucbs)

            reward = np.random.rand() < self.true_probs[i_best]
            self.sums[i_best] += reward
            self.counts[i_best] += 1
            self.reward_history.append(reward)
            t += 1

        return np.argmax(self.sums / self.counts)

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

def plot_results(result):
    print(f"Identified best arm: {result['identified_best']}")
    print(f"True best arm: {result['true_best']}")
    print(f"Correct identification: {result['is_correct']}")
    print(f"Average reward collected: {result['avg_reward']:.4f}")

    plt.plot(result["cumulative_rewards"], label="Cumulative Reward")
    plt.title("lil'UCB (Best Arm Identification)")
    plt.xlabel("Total Pulls")
    plt.ylabel("Cumulative Reward")
    plt.grid()
    plt.legend()
    plt.savefig("plots/lilucb_results.png")

def simulate_lilucb():
    lilucb = LilUCB(K=10, delta=0.05, epsilon=0.1, true_probs=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]))
    lilucb.run(max_total_pulls=10000)
    result = lilucb.report()
    return result

plot_results(simulate_lilucb())