import os

current_directory = './gt/'
dirs = os.listdir(current_directory)

# Iterate over each item in the list
for item in dirs:
    folder_path = os.path.join(current_directory, item)
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        for fn in files:
            file_path = os.path.join(folder_path, fn).removesuffix('.jpg')
            with open(f'{file_path}.txt', 'a') as file:
                file.close()
