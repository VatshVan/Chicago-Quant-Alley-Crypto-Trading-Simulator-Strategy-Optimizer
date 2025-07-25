import numpy as np
import matplotlib.pyplot as plt

class LUCB:
    def __init__(self, K=10, delta=0.05, true_probs=None):
        self.K = K
        self.delta = delta
        self.true_probs = true_probs if true_probs is not None else np.random.rand(K)
        self.best_arm = np.argmax(self.true_probs)
        self.counts = np.zeros(K)
        self.sums = np.zeros(K)
        self.reward_history = []

    def confidence_radius(self, n):
        if n == 0:
            return np.inf
        return np.sqrt(np.log(3 / self.delta) / (2 * n))

    def run(self, max_total_pulls=10000):
        # Pull each arm once initially
        for i in range(self.K):
            reward = np.random.rand() < self.true_probs[i]
            self.sums[i] += reward
            self.counts[i] += 1
            self.reward_history.append(reward)

        while np.sum(self.counts) < max_total_pulls:
            means = self.sums / self.counts
            ucbs = means + np.array([self.confidence_radius(n) for n in self.counts])
            lcbs = means - np.array([self.confidence_radius(n) for n in self.counts])

            i_best = np.argmax(means)
            # Exclude best arm from competitor search
            competitors = [i for i in range(self.K) if i != i_best]
            j = max(competitors, key=lambda i: ucbs[i])

            # Stopping condition
            if lcbs[i_best] >= ucbs[j]:
                break

            # Sample both i_best and j
            for i in [i_best, j]:
                reward = np.random.rand() < self.true_probs[i]
                self.sums[i] += reward
                self.counts[i] += 1
                self.reward_history.append(reward)

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
def plot_results(result):
    print(f"Identified best arm: {result['identified_best']}")
    print(f"True best arm: {result['true_best']}")
    print(f"Correct identification: {result['is_correct']}")
    print(f"Average reward collected: {result['avg_reward']:.4f}")

    plt.plot(result["cumulative_rewards"], label="Cumulative Reward")
    plt.title("LUCB (Best Arm Identification)")
    plt.xlabel("Total Pulls")
    plt.ylabel("Cumulative Reward")
    plt.grid()
    plt.legend()
    plt.savefig("plots/lucb_algorithm.png")

def simulate_lucb():
    lucb = LUCB(K=10, delta=0.05, true_probs=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]))
    lucb.run(max_total_pulls=10000)
    result = lucb.report()
    return result

simulate_lucb()
