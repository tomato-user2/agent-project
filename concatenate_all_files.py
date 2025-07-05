import os

def concatenate_all_files(output_file):
    # Get the current directory
    current_directory = os.getcwd()
    
    # List all files in the current directory
    all_files = [f for f in os.listdir(current_directory) if os.path.isfile(f)]
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file in all_files:
            # Write the file name as a header
            outfile.write(f"# {file}\n")
            # Write the content of the file
            with open(file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
            # Add a separator line
            outfile.write("\n" + "-"*40 + "\n\n")

if __name__ == "__main__":
    output_file_name = "concatenated_output.txt"
    concatenate_all_files(output_file_name)
    print(f"All files have been concatenated into {output_file_name}.")
