import os
from icrawler.builtin import GoogleImageCrawler

def download_image(query, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    crawler = GoogleImageCrawler(storage={'root_dir': output_dir})

    crawler.crawl(keyword=query, max_num=1)
    files = os.listdir(output_dir)
    if files:
        old_path = os.path.join(output_dir, files[0])
        new_path = os.path.join(output_dir, filename)
        os.rename(old_path, new_path)
        return new_path
    return None

def main():
    animal_names = [
        "cat", "dog", "bird", "elephant", "tiger",
        "lion", "horse", "bear", "monkey", "giraffe"
    ]

    dataset_dir = "dataset-for-testing-pipeline"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    for animal in animal_names:
        print(f"Downloading image for {animal}...")
        filename = f"{animal}.jpg"
        image_path = download_image(f"A photo of a {animal}", dataset_dir, filename)

        if image_path:
            print(f"Image for {animal} saved at: {image_path}")
        else:
            print(f"Failed to download image for {animal}.")

if __name__ == "__main__":
    main()