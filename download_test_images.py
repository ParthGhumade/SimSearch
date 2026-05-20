import urllib.request
import os

def download_real_images(root_dir):
    # Map of relative paths to image URLs (using picsum.photos for random images)
    images_to_replace = [
        "img_root.jpg",
        "nested_1/photo1.png",
        "nested_1/deep_folder/deep_img.webp",
        "nested_2/vacation.jpeg",
        "nested_2/another_one/profile.bmp"
    ]
    
    headers = {'User-Agent': 'Mozilla/5.0'}

    for rel_path in images_to_replace:
        full_path = os.path.join(root_dir, rel_path)
        if not os.path.exists(full_path):
            # Create directory if it doesn't exist (though it should from previous step)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
        print(f"Downloading real image for: {rel_path}...")
        try:
            # picsum.photos/width/height provides a random image
            url = "https://picsum.photos/400/300"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response, open(full_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Successfully replaced {rel_path}")
        except Exception as e:
            print(f"Failed to download {rel_path}: {e}")

if __name__ == "__main__":
    test_path = os.path.join(os.getcwd(), "test_media")
    download_real_images(test_path)
    print("\nAll dummy images have been replaced with real downloads.")
