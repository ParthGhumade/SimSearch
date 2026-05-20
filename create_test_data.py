import os

def create_test_structure(root_dir):
    structure = {
        "": ["img_root.jpg", "garbage.txt", "readme.md"],
        "nested_1": ["photo1.png", "data.json"],
        "nested_1/deep_folder": ["deep_img.webp", "old_log.log"],
        "nested_2": ["vacation.jpeg", "budget.xlsx"],
        "nested_2/another_one": ["profile.bmp", "script.py"],
        "empty_folder": []
    }

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        print(f"Created root: {root_dir}")

    for folder, files in structure.items():
        path = os.path.join(root_dir, folder)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created folder: {path}")
        
        for file in files:
            file_path = os.path.join(path, file)
            with open(file_path, 'w') as f:
                f.write("test content")
            print(f"Created file: {file_path}")

if __name__ == "__main__":
    test_path = os.path.join(os.getcwd(), "test_media")
    create_test_structure(test_path)
    print("\nTest structure created successfully.")
