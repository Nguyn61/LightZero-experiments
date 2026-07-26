import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

base_dir1 = os.path.expanduser("~/lightzero-full/pilot_results")
base_dir2 = os.path.expanduser("~/lightzero-full/pilot_results/sum_to_three_curriculum_reward")
arms = ["A_sparse_fixed", "B_dense_fixed", "C_sparse_curriculum", "D_dense_curriculum", "E_dense_broad"]
colors = ['red', 'blue', 'green', 'orange', 'purple']

plt.figure(figsize=(10, 6))

for arm, color in zip(arms, colors):
    # Try dir2 first, then dir1
    events_dir = os.path.join(base_dir2, arm, "seed-0", "attempt-01", "log", "serial")
    if not os.path.exists(events_dir):
        events_dir = os.path.join(base_dir1, arm, "seed-0", "attempt-01", "log", "serial")
        
    if not os.path.exists(events_dir):
        print(f"Directory not found: {events_dir}")
        continue
            
    event_files = glob.glob(os.path.join(events_dir, "events.out.tfevents.*"))
    if not event_files:
        print(f"No event files found for {arm}")
        continue
    
    event_file = event_files[0]
    try:
        ea = EventAccumulator(event_file)
        ea.Reload()
        tag = 'evaluator_step/reward_mean'
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            rewards = [e.value for e in events]
            plt.plot(steps, rewards, label=arm, color=color, linewidth=2)
            print(f"{arm}: Final Reward = {rewards[-1]:.2f}")
    except Exception as e:
        print(f"Error parsing {arm}: {e}")

plt.xlabel("Environment Steps")
plt.ylabel("Eval Reward Mean")
plt.title("Sum-to-Three Curriculum Pilot Learning Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_dir1, "learning_curves.png"))
print("Plot saved to learning_curves.png")
