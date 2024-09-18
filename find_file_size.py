import os


def find_large_files(directory, size_limit_mb=50):
    # Convert size limit from MB to bytes
    size_limit_bytes = size_limit_mb * 1024 * 1024

    # Walk through all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                # Get the size of the file
                file_size = os.path.getsize(file_path)

                # Check if the file size is greater than the size limit
                if file_size > size_limit_bytes:
                    size_in_mb = file_size / (1024 * 1024)
                    print(f"{file_path} - {size_in_mb:.2f} MB")
            except OSError as e:
                print(f"Could not access {file_path}: {e}")


if __name__ == "__main__":
    # Replace 'your_directory_path_here' with the directory you want to search
    head_directory = "."
    find_large_files(head_directory)
