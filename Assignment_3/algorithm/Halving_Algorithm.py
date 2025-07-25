import numpy as np
import matplotlib.pyplot as plt

class Halving:
    def __init__(self, K=10, budget=10000, true_probs=None):
        self.K = K
        self.budget = budget
        self.true_probs = true_probs if true_probs is not None else np.random.rand(K)
        self.best_arm = np.argmax(self.true_probs)
        self.rounds = int(np.ceil(np.log2(K)))  # Number of halving rounds
        self.identified_best = None
        self.total_pulls = 0
        self.reward_history = []

    def run(self):
        arms = list(range(self.K))

        # Estimate how many pulls per arm per round (simple allocation)
        pulls_per_round = self.budget // sum([len(arms) for _ in range(self.rounds)])

        for _ in range(self.rounds):
            mean_rewards = []
            for arm in arms:
                pulls = np.random.rand(pulls_per_round) < self.true_probs[arm]
                self.reward_history.extend(pulls)
                self.total_pulls += pulls_per_round
                mean_rewards.append(np.mean(pulls))

            # Keep top half of the arms
            top_indices = np.argsort(mean_rewards)[-len(arms) // 2:]
            arms = [arms[i] for i in top_indices]

            if len(arms) == 1:
                break

        self.identified_best = arms[0]
        return self.identified_best

    def report(self):
        avg_reward = np.mean(self.reward_history)
        correct = self.identified_best == self.best_arm
        return {
            "identified_best": self.identified_best,
            "true_best": self.best_arm,
            "is_correct": correct,
            "avg_reward": avg_reward,
            "cumulative_rewards": np.cumsum(self.reward_history)
        }

def plot_results(results):
    print(f"Identified best arm: {results['identified_best']}")
    print(f"True best arm: {results['true_best']}")
    print(f"Correct identification: {results['is_correct']}")
    print(f"Average reward collected: {results['avg_reward']:.4f}")

    plt.plot(results["cumulative_rewards"], label="Cumulative Reward")
    plt.title("Halving Algorithm (Class Version)")
    plt.xlabel("Total Pulls")
    plt.ylabel("Cumulative Reward")
    plt.grid()
    plt.legend()
    plt.savefig("plots/halving_algorithm_class.png")

def simulate_halving_class():
    halving = Halving(K=10, budget=10000, true_probs=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]))
    halving.run()
    results = halving.report()
    return results

plot_results(simulate_halving_class())
