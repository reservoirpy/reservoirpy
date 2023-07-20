#!/bin/bash

# Define the CSV file path
csv_file="output.csv"

# Write the CSV header
echo "tts,r2" > "$csv_file"

# Loop through tts values from 28 to 140 with a step size of 28
for tts in $(seq 28 28 56); do
    # Run your Python script with the current tts value
    python -m testing.phds --mode simple --only_v 1 --forecast 1 --tts "$tts"  --err 0 > output.txt

    # Extract the root mean square error from the output
    r2=$(grep "root mean square error" output.txt | awk '{print $NF}')

    # Append tts and r2 values to the CSV file
    echo "$tts,$r2" >> "$csv_file"
done

# Remove the temporary output file
rm output.txt
