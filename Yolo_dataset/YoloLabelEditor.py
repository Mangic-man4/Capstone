import os

def bulk_edit_labels(folder_path, replacement_map, default_label="Flip"):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue  # Skip empty lines

                try:
                    # Check if the first part is a number (no label)
                    float(parts[0])
                    parts.insert(0, default_label)
                except ValueError:
                    old_label = parts[0]
                    if old_label in replacement_map:
                        parts[0] = replacement_map[old_label]
                
                updated_lines.append(" ".join(parts) + "\n")

            with open(file_path, 'w') as file:
                file.writelines(updated_lines)

# Example usage
replacement_map = {
    "test1": "Flip"
    
}

folder_path = r"C:\Users\alexa\Documents\School docs\Capstone\Yolo_dataset\Dataset 1\labels\val"
bulk_edit_labels(folder_path, replacement_map)
