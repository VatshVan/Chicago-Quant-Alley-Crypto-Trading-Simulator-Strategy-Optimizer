import numpy as np
import matplotlib.pyplot as plt
import os

class Halving:
    def __init__(self, K=10, budget=10000, true_probs=None):
        self.K = K
        self.budget = budget
        self.true_probs = true_probs if true_probs is not None else np.random.rand(K)
        self.best_arm = np.argmax(self.true_probs)
        self.rounds = int(np.ceil(np.log2(K)))
        self.identified_best = None
        self.total_pulls = 0
        self.reward_history = []
        self.chosen_arms_history = []

    def run(self):
        arms = list(range(self.K))
        pulls_per_round = self.budget // sum([len(arms) for _ in range(self.rounds)])  # Estimate simple allocation

        for _ in range(self.rounds):
            mean_rewards = []
            for arm in arms:
                pulls = np.random.rand(pulls_per_round) < self.true_probs[arm]
                self.reward_history.extend(pulls)
                self.chosen_arms_history.extend([arm] * pulls_per_round)
                self.total_pulls += pulls_per_round
                mean_rewards.append(np.mean(pulls))
            top_indices = np.argsort(mean_rewards)[-len(arms) // 2:]
            arms = [arms[i] for i in top_indices]

            if len(arms) == 1:
                break

        self.identified_best = arms[0]
        return self.identified_best

    def report(self):
        avg_reward = np.mean(self.reward_history)
        correct = self.identified_best == self.best_arm
        best_mean = self.true_probs[self.best_arm]
        regret_per_step = [best_mean - reward for reward in self.reward_history]
        cumulative_regret = np.cumsum(regret_per_step)
        freq_over_time = []
        best_arm_count = 0
        for i, a in enumerate(self.chosen_arms_history):
            if a == self.best_arm:
                best_arm_count += 1
            freq_over_time.append(best_arm_count / (i + 1))

        return {
            "identified_best": self.identified_best,
            "true_best": self.best_arm,
            "is_correct": correct,
            "avg_reward": avg_reward,
            "cumulative_rewards": np.cumsum(self.reward_history),
            "cumulative_regret": cumulative_regret,
            "freq_over_time": freq_over_time
        }

def plot_evaluation(results):
    os.makedirs("plots", exist_ok=True)
    print(f"Identified best arm: {results['identified_best']}")
    print(f"True best arm: {results['true_best']}")
    print(f"Correct identification: {results['is_correct']}")
    print(f"Average reward collected: {results['avg_reward']:.4f}")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(results["cumulative_rewards"], label="Cumulative Reward")
    plt.plot(results["cumulative_regret"], label="Cumulative Regret")
    plt.title("Halving Algorithm: Reward & Regret")
    plt.xlabel("Total Pulls")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(results["freq_over_time"], label="Best Arm Frequency")
    plt.title("Best Arm Selection Frequency")
    plt.xlabel("Total Pulls")
    plt.ylabel("Fraction")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("plots/halving_algorithm_class_evaluation.png")
    plt.close()

def simulate_halving_class():
    halving = Halving(K=10, budget=10000, true_probs=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]))
    halving.run()
    results = halving.report()
    return results

plot_evaluation(simulate_halving_class())
