import os

# Specify the directory path you want to check
directory_path = r'C:\Users\UTKARSH\Desktop\data science\dl\har2'

# Check if the directory has write permission
if os.access(directory_path, os.W_OK):
    print("Write permission is granted for the directory.")
else:
    print("Write permission is not granted for the directory.")

