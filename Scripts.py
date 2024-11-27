import os
import json
import shutil

def convert_jsonfile_sign_language_words_to_output_txt():
    # Path to the folder containing JSON files
    folder_path = "D:/Downloads/names/01"

    # Output text file
    # Saved to output.txt
    output_file = "output.txt"

    # Set to track processed IDs
    processed_ids = set()

    # Open the output file for writing
    with open(output_file, "w", encoding="utf-8") as outfile:
        # Loop through sorted files in the folder
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)

                # Extract the ID from the filename
                try:
                    file_id = filename.split('_')[2][4:8]  # Extract the ID like "2207"
                except IndexError:
                    print(f"Skipping malformed file name: {filename}")
                    continue

                # Process the JSON file
                try:
                    with open(file_path, "r", encoding="utf-8") as json_file:
                        data = json.load(json_file)

                        # Extract the "name" attribute
                        name = data["data"][0]["attributes"][0]["name"]
                        if file_id not in processed_ids:
                            processed_ids.add(file_id)
                            # Write to the output file
                            outfile.write(f"{file_id} = \"{name}\"\n")
                        else:
                            print(f"Duplicate ID {file_id}, skipping.")
                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    print(f"Error processing file {filename}: {e}")

    print(f"Processing complete. Results saved to {output_file}")

def create_empty_folder_per_word_for_training_and_validation_video_data():
    # Path to the output.txt file
    input_file = "output.txt"

    # Path to the directory where folders will be created
    #output_dir = "D:/collected_data/training_data"
    #output_dir = "D:/collected_data/validation_data"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the input file and create folders
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            # Extract the ID and name from each line
            if '=' in line:
                file_id, name = line.strip().split(' = ')
                name = name.strip('"')  # Remove quotes around the name

                # Construct folder name and path
                folder_name = f"{file_id}_{name}"
                folder_path = os.path.join(output_dir, folder_name)

                # Create the folder
                os.makedirs(folder_path, exist_ok=True)
                print(f"Created folder: {folder_name}")

def rename_move_videos_to_training_and_validation_folders():
    # Path to the training_data directory
    training_data_dir = "D:/collected_data/training_data"

    # Path to the directory containing mp4 files
    mp4_files_dir = "D:/collected_data/raw"

    # Mapping for direction abbreviations
    direction_mapping = {
        'F': 'Front',
        'L': 'Left',
        'R': 'Right',
        'U': 'Up',
        'D': 'Down'
    }

    # Loop through all mp4 files
    for filename in os.listdir(mp4_files_dir):
        if filename.endswith('.mp4'):
            # Parse the filename
            parts = filename.split('_')
            word_id = parts[2][4:]  # Extract WORD ID (e.g., 1549)
            person = parts[3]  # Extract person identifier (e.g., REAL01)
            direction = parts[4].split('.')[0]  # Extract direction (e.g., F)

            # Determine the person number and direction
            person_number = int(person[4:])  # Extract number from REAL01, REAL02, etc.
            direction_name = direction_mapping.get(direction, direction.lower())

            # Construct the new file name
            new_filename = f"person{person_number}_{direction_name}.mp4"

            # Find the target folder by matching WORD ID
            target_folder = None
            for folder in os.listdir(training_data_dir):
                if folder.startswith(word_id + "_"):  # Match the folder starting with the WORD ID
                    target_folder = os.path.join(training_data_dir, folder)
                    break

            # Ensure the target folder exists
            if not target_folder or not os.path.exists(target_folder):
                print(f"Target folder for {word_id} does not exist. Skipping {filename}.")
                continue

            # Determine the target file path
            target_file_path = os.path.join(target_folder, new_filename)

            # Move and rename the file
            source_file_path = os.path.join(mp4_files_dir, filename)
            shutil.move(source_file_path, target_file_path)
            print(f"Moved {filename} to {target_file_path}")

if __name__ == '__main__':
    #convert_jsonfile_sign_language_words_to_output_txt()
    #create_empty_folder_per_word_for_training_and_validation_video_data()
    rename_move_videos_to_training_and_validation_folders()