import numpy as np
import matplotlib.pyplot as plt
import os

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
        self.ucb_history = []
        self.chosen_arms = []

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
        for i in range(self.K):
            reward = np.random.rand() < self.true_probs[i]
            self.sums[i] += reward
            self.counts[i] += 1
            self.reward_history.append(reward)
            self.chosen_arms.append(i)
            t += 1
            self.ucb_history.append([self.ucb(j, t) for j in range(self.K)])

        while np.sum(self.counts) < max_total_pulls:
            ucbs = [self.ucb(i, t) for i in range(self.K)]
            i_best = np.argmax(ucbs)
            reward = np.random.rand() < self.true_probs[i_best]
            self.sums[i_best] += reward
            self.counts[i_best] += 1
            self.reward_history.append(reward)
            self.chosen_arms.append(i_best)
            t += 1
            self.ucb_history.append([self.ucb(j, t) for j in range(self.K)])
        return np.argmax(self.sums / self.counts)

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
            "chosen_arms": self.chosen_arms,
            "true_probs": self.true_probs
        }

def plot_evaluation(result):
    os.makedirs("plots", exist_ok=True)
    print(f"Identified best arm: {result['identified_best']}")
    print(f"True best arm: {result['true_best']}")
    print(f"Correct identification: {result['is_correct']}")
    print(f"Average reward collected: {result['avg_reward']:.4f}")
    print(f"True arm means: {np.round(result['true_probs'],2)}")

    plt.figure(figsize=(16,8))

    plt.subplot(2,2,1)
    plt.plot(result["cumulative_rewards"], label="Cumulative Reward")
    plt.plot(result["cumulative_regret"], label="Cumulative Regret")
    plt.title("lil'UCB: Reward & Regret")
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
    arms_to_plot = [np.argmax(result["true_probs"]), np.argmin(result["true_probs"]), np.random.randint(0,k)]
    for i in arms_to_plot:
        plt.subplot(2,2,3)
        ucb_curve = [u[i] for u in result["ucb_history"]]
        plt.plot(timeline, ucb_curve, label=f"Arm {i} UCB")
    plt.title("UCB Bounds (Selected Arms)")
    plt.xlabel("Time Steps")
    plt.ylabel("UCB Value")
    plt.legend(fontsize="small")
    plt.grid()

    plt.tight_layout()
    plt.savefig("plots/lilucb_evaluation.png")
    plt.close()

def simulate_lilucb():
    lilucb = LilUCB(K=10, delta=0.05, epsilon=0.1,
                    true_probs=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]))
    lilucb.run(max_total_pulls=10000)
    result = lilucb.report()
    return result

plot_evaluation(simulate_lilucb())
