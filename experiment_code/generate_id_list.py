import csv
import random

def generate_subject_ids(n):
    # Generate n unique 6-digit numbers
    ids = random.sample(range(100000, 1000000), n)
    
    # Write to CSV
    with open('participant_ID_list.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subject_id', 'used'])
        for id in ids:
            writer.writerow([id, 0])


generate_subject_ids(n = 120)

if __name__ == "__main__":
    while True:
        try:
            n = int(input("Enter number of subjects needed: "))
            if n <= 0:
                print("Please enter a positive integer.")
                continue
            if n > 900000:
                print("Maximum 900,000 unique IDs possible.")
                continue
            generate_subject_ids(n)
            print(f"Generated {n} IDs in participant_ID_list.csv")
            break
        except ValueError:
            print("Please enter a valid integer.")