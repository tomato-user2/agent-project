import os

def concatenate_py_files(output_file):
    # Get the current directory
    current_directory = os.getcwd()
    
    # List all .py files in the current directory
    py_files = [f for f in os.listdir(current_directory) if f.endswith('.py')]
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for py_file in py_files:
            # Write the file name as a header
            outfile.write(f"# {py_file}\n")
            # Write the content of the file
            with open(py_file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
            # Add a separator line
            outfile.write("\n" + "-"*40 + "\n\n")

if __name__ == "__main__":
    output_file_name = "concatenated_output.txt"
    concatenate_py_files(output_file_name)
    print(f"All .py files have been concatenated into {output_file_name}.")
