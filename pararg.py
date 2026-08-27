import csv
import statistics

input_file = "results_gather.csv"
output_file = "results_stats.csv"

accuracies = []

# Read accuracies from the CSV
with open(input_file, newline="") as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		try:
			acc = float(row["Accuracy"])
			accuracies.append(acc)
		except (KeyError, TypeError, ValueError):
			print(f"Skipping row with invalid accuracy: {row}")

# Compute statistics
if accuracies:
	mean_acc = statistics.mean(accuracies)
	sd_acc = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
else:
	mean_acc = 0.0
	sd_acc = 0.0

# Write to CSV
with open(output_file, "w", newline="") as out_csv:
	writer = csv.writer(out_csv)
	writer.writerow(["MeanAccuracy", "StdDevAccuracy"])
	writer.writerow([f"{mean_acc:.4f}", f"{sd_acc:.4f}"])

print(f"Summary written to {output_file}")
