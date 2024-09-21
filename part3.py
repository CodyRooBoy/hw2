import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Simulating Drifting Bandits and Sudden Shifts

# Functions to Generate and Update Probabilities
def get_probabilities():
    probs = [
        np.random.normal(0, 5),
        np.random.normal(-0.5, 12),
        np.random.normal(2, 3.9),
        np.random.normal(-0.5, 7),
        np.random.normal(-1.2, 8),
        np.random.normal(-3, 7),
        np.random.normal(-10, 20),
        np.random.normal(-0.5, 1),
        np.random.normal(-1, 2),
        np.random.normal(1, 6),
        np.random.normal(0.7, 4),
        np.random.normal(-6, 11),
        np.random.normal(-7, 1),
        np.random.normal(-0.5, 2),
        np.random.normal(-6.5, 1),
        np.random.normal(-3, 6),
        np.random.normal(0, 8),
        np.random.normal(2, 3.9),
        np.random.normal(-9, 12),
        np.random.normal(-1, 6),
        np.random.normal(-4.5, 8)
    ]
    return probs

def apply_drift(probabilities, drift_rate=-0.001):
    return [p + drift_rate for p in probabilities]

def apply_sudden_shifts(probabilities, step):
    if step == 3000:
        probabilities[0] += 7
        probabilities[2] += 3
        probabilities[7] += 1
        probabilities[18] += 2
    return probabilities

# Function to Update Score History
def update_score_history(score, score_history, machine):
    score_history[machine].append(score)

def get_current_preds(score_history):
    averages = []
    for scores in score_history:
        if len(scores) > 0:
            avg = sum(scores) / len(scores)
        else:
            avg = 0
        averages.append(avg)
    return averages

# Epsilon-Greedy Action Selection
def get_action(epsilon, score_history):
    if random.random() > epsilon:
        current_preds = get_current_preds(score_history)
        best_machine = max(range(len(current_preds)), key=lambda i: current_preds[i])
        return best_machine
    else:
        return random.choice(range(len(score_history)))

# Thompson Sampling Functions
def update_record_history(outcome, record_history, machine):
    alpha, beta_val = record_history[machine]
    if outcome == 'win':
        alpha += 1
    elif outcome == 'loss':
        beta_val += 1
    record_history[machine] = (alpha, beta_val)

def sample_machine(record_history, machine):
    alpha_param, beta_param = record_history[machine]
    return beta.rvs(alpha_param, beta_param)

def get_action_thompson(record_history):
    samples = []
    for i in range(20):
        samples.append(sample_machine(record_history, i))
    return np.argmax(samples)

# Simulation Functions
def run_sim_epsilon_greedy(steps, epsilon):
    score_history = [[] for _ in range(20)]
    probabilities = get_probabilities()
    individual_score_history = []
    
    for step in range(steps):
        probabilities = apply_drift(probabilities)
        probabilities = apply_sudden_shifts(probabilities, step)
        machine = get_action(epsilon, score_history)
        value = probabilities[machine]
        update_score_history(value, score_history, machine)
        individual_score_history.append(value)
        
    return score_history, compute_running_average(individual_score_history)

def run_sim_thompson_sampling(steps):
    record_history = [(1, 1) for _ in range(20)]
    score_history = [[] for _ in range(20)]
    probabilities = get_probabilities()
    individual_score_history = []
    
    for step in range(steps):
        probabilities = apply_drift(probabilities)
        probabilities = apply_sudden_shifts(probabilities, step)
        machine = get_action_thompson(record_history)
        value = probabilities[machine]
        if value > 0:
            reward = 1
            update_record_history('win', record_history, machine)
        else:
            reward = 0
            update_record_history('loss', record_history, machine)
        update_score_history(value, score_history, machine)
        individual_score_history.append(value)
        
    return score_history, compute_running_average(individual_score_history)

# Helper Functions
def compute_running_average(score_history):
    running_avg = []
    total = 0
    for i, score in enumerate(score_history, 1):
        total += score
        running_avg.append(total / i)
    return running_avg

# Plotting Functions
def plot_running_average_combined(full_results, epsilon_values, thompson_results, filename="combined_plot.png"):
    plt.figure(figsize=(20, 10))
    for i, running_average_history in enumerate(full_results):
        plt.plot(running_average_history, label=f'Epsilon = {epsilon_values[i]}')
    
    plt.plot(thompson_results, label='Thompson Sampling', linestyle='--', color='black', linewidth=2)
    
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Running Average of Rewards for Epsilon-Greedy and Thompson Sampling')
    plt.legend()
    plt.ylim(-4, 4)
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig(filename)
    plt.close()  # Close the figure to prevent it from being displayed

# Compare Strategies
def compare_strategies(steps=10_000, epsilon=0.1, filename="epsilon_vs_thompson.png"):
    full_results = []
    for epsilon in [epsilon]:  # Modify to test multiple epsilon values if needed
        results_epsilon, avg_scores_epsilon = run_sim_epsilon_greedy(steps, epsilon)
        full_results.append(avg_scores_epsilon)

    results_thompson, avg_scores_thompson = run_sim_thompson_sampling(steps)

    plot_running_average_combined(full_results, [epsilon], avg_scores_thompson, filename=filename)

# Restart Thompson Sampling after Sudden Shift
def restart_thompson_sampling(steps=10_000, sudden_shift_step=3_000, filename="thompson_restart.png"):
    full_results = []
    for epsilon in [0.1]:
        results_epsilon, avg_scores_epsilon = run_sim_epsilon_greedy(steps, epsilon)
        full_results.append(avg_scores_epsilon)
    
    results_thompson_before, avg_scores_thompson_before = run_sim_thompson_sampling(sudden_shift_step)
    results_thompson_after, avg_scores_thompson_after = run_sim_thompson_sampling(steps - sudden_shift_step)
    
    avg_scores_thompson_combined = avg_scores_thompson_before + avg_scores_thompson_after

    plot_running_average_combined(full_results, [0.1], avg_scores_thompson_combined, filename=filename)

# Running Part 3 Simulations
if __name__ == "__main__":
    compare_strategies(steps=10_000, epsilon=0.1, filename="epsilon_vs_thompson.png")
    restart_thompson_sampling(steps=10_000, sudden_shift_step=3_000, filename="thompson_restart.png")