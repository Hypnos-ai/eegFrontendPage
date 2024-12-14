import time
import random

# Define the tasks with specific counts
tasks = {
    "Right_Hand": ["Imagine or perform right hand movement", 21],
    "Left_Hand": ["Imagine or perform left hand movement", 21],
    "Blinking": ["Blink your eyes at a comfortable pace", 21],
    "Jaw_Clenching": ["Clench your jaw with moderate force", 21],
    "Relax": ["Relax your body and mind, do nothing", 21]
}

# Create task sequence with balanced distribution
task_sequence = []
for task_name, (instruction, count) in tasks.items():
    task_sequence.extend([(task_name, instruction)] * count)

# Randomize the order of tasks
random.shuffle(task_sequence)

# Define parameters for task duration and rest time
task_duration = 5  # Duration to perform each task in seconds
rest_duration = 3  # Rest period between tasks in seconds

# Start the task loop
print("\nStarting recording session...")
print(f"Each task is {task_duration} seconds long with {rest_duration} second breaks")
print(f"Total tasks: {len(task_sequence)} ({len(task_sequence) * (task_duration + rest_duration) / 60:.1f} minutes)\n")

# Get session number and write header
with open('collected_data/annotations.txt', "a+") as f:
    session_num = input("Enter session number (e.g., 1): ")
    f.write(f"session{session_num}\n")

# Initialize counters
task_counts = {task: 0 for task in tasks.keys()}

try:
    for task_name, instruction in task_sequence:
        # Update and display progress
        task_counts[task_name] += 1
        print("\n" + "="*50)
        print(f"Task: {task_name} (Trial {task_counts[task_name]}/21)")
        print(f"Instruction: {instruction}")
        
        # Write task to annotations file
        with open('collected_data/annotations.txt', "a+") as f:
            f.write(f"{task_name.split('_')[0]}\n")
        
        # Countdown before task
        print("\nPreparing...")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(0.7)
        
        # Task execution
        print(f"\nSTART NOW! Continue for {task_duration} seconds...")
        time.sleep(task_duration)
        
        # Rest period
        print("\nRest... relax for a few seconds.")
        time.sleep(rest_duration)
        
    # Write session end marker
    with open('collected_data/annotations.txt', "a+") as f:
        f.write('_____________________________________________\n')
    
    # Display completion summary
    print("\n" + "="*50)
    print("Recording session completed!")
    print("\nTrials completed per task:")
    for task, count in task_counts.items():
        print(f"- {task}: {count}/21")

except KeyboardInterrupt:
    print("\n\nRecording interrupted by user.")
    print("Note: Partial session data has been saved.")
    with open('collected_data/annotations.txt', "a+") as f:
        f.write('_____________________________________________\n')
    
    # Display partial completion summary
    print("\nTrials completed before interruption:")
    for task, count in task_counts.items():
        print(f"- {task}: {count}/21")

finally:
    print("\nYou can close this window.")
