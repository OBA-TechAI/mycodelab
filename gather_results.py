import glob
import csv

# Collect all matching files
result_files = sorted(glob.glob("res_*.txt"))

output_rows = []

for filename in result_files:
	try:
		with open(filename, "r") as f:
			lines = f.readlines()
			if len(lines) < 2:
				print(f"Skipping {filename}: not enough lines.")
				continue

			# Parse accuracy and random state
			accuracy_line = lines[0].strip()
			state_line = lines[1].strip()

			# Example: "Test Accuracy: 88.89%"
			acc_str = accuracy_line.split(":", 1)[1].strip().rstrip("%")
			# Example: "Random State: 15"
			state_str = state_line.split(":", 1)[1].strip()

			accuracy = float(acc_str)
			random_state = int(state_str)

			output_rows.append((accuracy, random_state))

	except Exception as e:
		print(f"Error processing {filename}: {e}")

# Write to CSV
with open("results_gather.csv", "w", newline="") as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(["Accuracy", "RandomState"])
	writer.writerows(output_rows)

print(f"Done. {len(output_rows)} entries written to results_gather.csv.")
