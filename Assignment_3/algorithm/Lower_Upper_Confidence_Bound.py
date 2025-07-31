import numpy as np
import matplotlib.pyplot as plt
import os

class LUCB:
    def __init__(self, K=10, delta=0.05, true_probs=None):
        self.K = K
        self.delta = delta
        self.true_probs = true_probs if true_probs is not None else np.random.rand(K)
        self.best_arm = np.argmax(self.true_probs)
        self.counts = np.zeros(K)
        self.sums = np.zeros(K)
        self.reward_history = []
        self.ucb_history = []
        self.lcb_history = []
        self.chosen_arms = []

    def confidence_radius(self, n):
        if n == 0:
            return np.inf
        return np.sqrt(np.log(3 / self.delta) / (2 * n))

    def run(self, max_total_pulls=10000):
        for i in range(self.K):
            reward = np.random.rand() < self.true_probs[i]
            self.sums[i] += reward
            self.counts[i] += 1
            self.reward_history.append(reward)
            self.chosen_arms.append(i)
            means = self.sums / self.counts
            ucbs = means + np.array([self.confidence_radius(n) for n in self.counts])
            lcbs = means - np.array([self.confidence_radius(n) for n in self.counts])
            self.ucb_history.append(ucbs.copy())
            self.lcb_history.append(lcbs.copy())

        while np.sum(self.counts) < max_total_pulls:
            means = self.sums / self.counts
            ucbs = means + np.array([self.confidence_radius(n) for n in self.counts])
            lcbs = means - np.array([self.confidence_radius(n) for n in self.counts])
            self.ucb_history.append(ucbs.copy())
            self.lcb_history.append(lcbs.copy())

            i_best = np.argmax(means)
            competitors = [i for i in range(self.K) if i != i_best]
            j = max(competitors, key=lambda i: ucbs[i])

            if lcbs[i_best] >= ucbs[j]:
                break

            for i in [i_best, j]:
                reward = np.random.rand() < self.true_probs[i]
                self.sums[i] += reward
                self.counts[i] += 1
                self.reward_history.append(reward)
                self.chosen_arms.append(i)

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
        for idx, arm in enumerate(self.chosen_arms):
            if arm == self.best_arm:
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
            "chosen_arms": self.chosen_arms,
            "true_probs": self.true_probs,
        }

def plot_evaluation(result):
    os.makedirs("plots", exist_ok=True)
    print(f"Identified best arm: {result['identified_best']}")
    print(f"True best arm: {result['true_best']}")
    print(f"Correct identification: {result['is_correct']}")
    print(f"Average reward collected: {result['avg_reward']:.4f}")
    print(f"True arm means: {np.round(result['true_probs'], 2)}")

    plt.figure(figsize=(16, 8))

    plt.subplot(2,2,1)
    plt.plot(result["cumulative_rewards"], label="Cumulative Reward")
    plt.plot(result["cumulative_regret"], label="Cumulative Regret")
    plt.title("LUCB: Reward & Regret")
    plt.xlabel("Total Pulls")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()

    plt.subplot(2,2,2)
    plt.plot(result["freq_over_time"], label="Best Arm Frequency")
    plt.title("Best Arm Selection Frequency")
    plt.xlabel("Total Pulls")
    plt.ylabel("Fraction")
    plt.legend()
    plt.grid()

    timeline = np.arange(len(result["ucb_history"]))
    k = len(result["true_probs"])
    arms_to_plot = [np.argmax(result["true_probs"]), np.argmin(result["true_probs"]), np.random.randint(0, k)]
    plt.subplot(2,2,3)
    for i in arms_to_plot:
        plt.plot(timeline, [u[i] for u in result["ucb_history"]], label=f"Arm {i} UCB")
        plt.plot(timeline, [l[i] for l in result["lcb_history"]], linestyle='--')
    plt.title("UCB & LCB for Representative Arms")
    plt.xlabel("Step")
    plt.ylabel("Bound Value")
    plt.legend(fontsize='small')
    plt.grid()

    plt.tight_layout()
    plt.savefig("plots/lucb_evaluation.png")
    plt.close()

def simulate_lucb():
    lucb = LUCB(K=10, delta=0.05, true_probs=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]))
    lucb.run(max_total_pulls=10000)
    return lucb.report()

result = simulate_lucb()
plot_evaluation(result)
