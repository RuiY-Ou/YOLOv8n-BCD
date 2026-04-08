import json
import os
import random
import shutil
from pathlib import Path
import cv2
from tqdm import tqdm
from collections import defaultdict, Counter


class BalancedNightDatasetSelector:
    def __init__(self, base_path, target_total=3500):
        """
        Initialize the selector
        Args:
            base_path: Dataset base path (F:\\deeplearning\\ultralytics-8.3.163\\datasets\\BDD100K)
            target_total: Total number of images to select
        """
        self.base_path = Path(base_path)
        self.target_total = target_total

        # Input paths (your previously extracted night dataset)
        self.night_images_train = self.base_path / "bdd100k_night" / "images" / "train"
        self.night_labels_train = self.base_path / "bdd100k_night" / "labels" / "train"

        # Output paths (new selected subset)
        self.selected_images = self.base_path / "bdd100k_night_selected" / "images" / "train"
        self.selected_labels = self.base_path / "bdd100k_night_selected" / "labels" / "train"

        # Create output directories
        for path in [self.selected_images, self.selected_labels]:
            path.mkdir(parents=True, exist_ok=True)

        # Target categories (mapped to YOLO class IDs)
        # Note: uses the YOLO class IDs after conversion
        self.target_categories = {
            0: 'person',  # includes rider
            1: 'car',
            2: 'bike',
            3: 'motor',
            4: 'traffic light'
        }

        # Minimum target counts per category (adjustable based on your paper requirements)
        # Here a relatively balanced ratio is set
        self.category_min_targets = {
            0: 700,  # person
            1: 1500,  # car (largest, but we control the ratio)
            2: 600,  # bike
            3: 400,  # motor
            4: 800  # traffic light
        }

        # Store information for all images
        self.image_info = []  # each element: (image_name, set of categories, category counts, total objects)

        # Statistics
        self.stats = {
            'total_images': 0,
            'total_objects': 0,
            'category_counts': defaultdict(int),
            'selected_images': 0,
            'selected_objects': 0,
            'selected_category_counts': defaultdict(int)
        }

    def analyze_dataset(self):
        """Analyze the dataset and collect category information for each image"""
        print("Analyzing dataset...")

        # Get all image files
        image_files = list(self.night_images_train.glob("*.jpg"))
        if not image_files:
            # Try other formats
            image_files = list(self.night_images_train.glob("*.jpeg")) + \
                          list(self.night_images_train.glob("*.png"))

        self.stats['total_images'] = len(image_files)
        print(f"Found {self.stats['total_images']} night images")

        # Analyze each image
        for img_path in tqdm(image_files, desc="Analyzing images"):
            img_name = img_path.stem
            label_path = self.night_labels_train / f"{img_name}.txt"

            if not label_path.exists():
                continue

            # Read YOLO format labels
            categories_in_image = set()
            category_counter = Counter()
            total_objects = 0

            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id in self.target_categories:
                            categories_in_image.add(class_id)
                            category_counter[class_id] += 1
                            total_objects += 1

            # Update global statistics
            for cat_id, count in category_counter.items():
                self.stats['category_counts'][cat_id] += count
            self.stats['total_objects'] += total_objects

            # Store image info (only images with at least one target category)
            if categories_in_image:
                self.image_info.append({
                    'name': img_name,
                    'categories': categories_in_image,
                    'category_counts': dict(category_counter),
                    'total_objects': total_objects,
                    'unique_categories': len(categories_in_image)
                })

        print("\nDataset analysis complete!")
        print("=" * 50)
        print("Original dataset statistics:")
        print(f"  Total images: {self.stats['total_images']}")
        print(f"  Total objects: {self.stats['total_objects']}")
        print("  Object counts per category:")
        for cat_id, cat_name in sorted(self.target_categories.items()):
            count = self.stats['category_counts'][cat_id]
            percentage = (count / self.stats['total_objects'] * 100) if self.stats['total_objects'] > 0 else 0
            print(f"    {cat_name} (ID:{cat_id}): {count} ({percentage:.1f}%)")
        print("=" * 50)

    def select_balanced_subset(self):
        """Intelligently select a balanced subset"""
        print("\nStarting intelligent image selection...")

        # Step 1: Sort by "information density" (images with more categories first)
        print("1. Sorting images by information density...")
        self.image_info.sort(key=lambda x: (
            -x['unique_categories'],  # more categories first
            -x['total_objects']  # more objects first
        ))

        # Step 2: Intelligent selection
        selected = []
        selected_names = set()
        current_category_counts = defaultdict(int)

        # First ensure basic coverage for each category
        print("2. Ensuring basic coverage for each category...")
        for cat_id in self.target_categories:
            cat_images = [info for info in self.image_info
                          if cat_id in info['categories'] and info['name'] not in selected_names]

            # Select some representative images for each category
            for info in cat_images[:max(50, self.category_min_targets[cat_id] // 20)]:
                if info['name'] not in selected_names:
                    selected.append(info)
                    selected_names.add(info['name'])
                    # Update counts
                    for cid, count in info['category_counts'].items():
                        current_category_counts[cid] += count

        # Step 3: Fill to target number, prioritizing images that supplement under-represented categories
        print("3. Intelligently filling to target count...")
        pbar = tqdm(total=self.target_total, desc="Selection progress")
        pbar.update(len(selected))

        # Calculate current satisfaction for each category
        def get_category_satisfaction():
            satisfaction = {}
            for cat_id, min_target in self.category_min_targets.items():
                current = current_category_counts[cat_id]
                satisfaction[cat_id] = min(current / min_target, 1.0) if min_target > 0 else 1.0
            return satisfaction

        # Continue selection until target is reached
        while len(selected) < self.target_total and len(selected) < len(self.image_info):
            # Calculate current satisfaction
            satisfaction = get_category_satisfaction()

            # Find the most unsatisfied category
            most_unsatisfied = min(satisfaction.items(), key=lambda x: x[1])[0]

            # Find the best unselected image containing this category
            best_image = None
            best_score = -1

            for info in self.image_info:
                if info['name'] in selected_names:
                    continue

                if most_unsatisfied in info['categories']:
                    # Calculate a "comprehensive value" score for this image
                    score = 0

                    # 1. Contribution to the most unsatisfied category
                    score += info['category_counts'].get(most_unsatisfied, 0) * 5

                    # 2. Contribution to other under-satisfied categories
                    for cat_id, sat in satisfaction.items():
                        if sat < 0.8 and cat_id in info['categories']:
                            score += info['category_counts'].get(cat_id, 0) * 3

                    # 3. Information density bonus
                    score += info['unique_categories'] * 2

                    if score > best_score:
                        best_score = score
                        best_image = info

            # If no image containing the most unsatisfied category is found, select the best overall
            if not best_image:
                for info in self.image_info:
                    if info['name'] not in selected_names:
                        score = info['unique_categories'] * 3 + info['total_objects']
                        if score > best_score:
                            best_score = score
                            best_image = info

            if best_image:
                selected.append(best_image)
                selected_names.add(best_image['name'])
                # Update counts
                for cat_id, count in best_image['category_counts'].items():
                    current_category_counts[cat_id] += count

                pbar.update(1)
            else:
                break  # No more images available

        pbar.close()

        # Step 4: Final fine-tuning to ensure approximate balance
        print("4. Final fine-tuning...")
        final_selected = selected[:self.target_total]

        # Recompute final statistics
        self.stats['selected_images'] = len(final_selected)
        self.stats['selected_category_counts'] = defaultdict(int)

        for info in final_selected:
            for cat_id, count in info['category_counts'].items():
                self.stats['selected_category_counts'][cat_id] += count
            self.stats['selected_objects'] += info['total_objects']

        return final_selected

    def copy_selected_data(self, selected_images):
        """Copy selected images and labels to the new directory"""
        print(f"\nCopying {len(selected_images)} selected images and labels...")

        for info in tqdm(selected_images, desc="Copying files"):
            img_name = info['name']

            # Source paths
            src_img = self.night_images_train / f"{img_name}.jpg"
            src_label = self.night_labels_train / f"{img_name}.txt"

            # Destination paths
            dst_img = self.selected_images / f"{img_name}.jpg"
            dst_label = self.selected_labels / f"{img_name}.txt"

            # Copy files
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            if src_label.exists():
                shutil.copy2(src_label, dst_label)

        print("File copying completed!")

    def print_selection_statistics(self):
        """Print statistics of the selected subset"""
        print("\n" + "=" * 60)
        print("Dataset selection completed! Detailed statistics:")
        print("=" * 60)

        print(f"\n📊 Scale comparison:")
        print(f"  Original dataset: {self.stats['total_images']} images, {self.stats['total_objects']} objects")
        print(f"  Selected subset: {self.stats['selected_images']} images, {self.stats['selected_objects']} objects")

        print(f"\n🎯 Category distribution comparison:")
        print("  " + "Category".ljust(15) + "Original".ljust(12) + "Selected".ljust(12) + "Selected %".ljust(12))
        print("  " + "-" * 50)

        total_original = self.stats['total_objects']
        total_selected = self.stats['selected_objects']

        for cat_id, cat_name in sorted(self.target_categories.items()):
            orig_count = self.stats['category_counts'][cat_id]
            sel_count = self.stats['selected_category_counts'][cat_id]

            orig_pct = (orig_count / total_original * 100) if total_original > 0 else 0
            sel_pct = (sel_count / total_selected * 100) if total_selected > 0 else 0

            print(f"  {cat_name.ljust(15)}"
                  f"{str(orig_count).ljust(12)}"
                  f"{str(sel_count).ljust(12)}"
                  f"{sel_pct:.1f}%")

        print(f"\n📈 Selected subset key metrics:")

        # Average objects per image
        avg_objects = self.stats['selected_objects'] / self.stats['selected_images'] if self.stats[
                                                                                            'selected_images'] > 0 else 0
        print(f"  • Average objects per image: {avg_objects:.1f}")

        # Category coverage
        categories_covered = sum(1 for cat_id in self.target_categories
                                 if self.stats['selected_category_counts'][cat_id] > 0)
        print(f"  • Categories covered: {categories_covered}/{len(self.target_categories)}")

        # Check if minimum targets are met
        print(f"  • Category object count satisfaction:")
        for cat_id, cat_name in self.target_categories.items():
            current = self.stats['selected_category_counts'][cat_id]
            target = self.category_min_targets.get(cat_id, 0)
            satisfied = "✓" if current >= target else "⚠"
            print(f"    {satisfied} {cat_name}: {current}/{target}")

        print(f"\n💾 Data saved to:")
        print(f"  Images: {self.selected_images}")
        print(f"  Labels: {self.selected_labels}")
        print("=" * 60)

        # Generate a simple configuration info file
        config_content = f"""# Selected Night Object Detection Dataset Configuration

Dataset size:
  Number of images: {self.stats['selected_images']}
  Total objects: {self.stats['selected_objects']}
  Average objects per image: {avg_objects:.2f}

Category distribution:
"""
        for cat_id, cat_name in sorted(self.target_categories.items()):
            count = self.stats['selected_category_counts'][cat_id]
            pct = (count / self.stats['selected_objects'] * 100) if self.stats['selected_objects'] > 0 else 0
            config_content += f"  {cat_name}: {count} ({pct:.1f}%)\n"

        config_path = self.base_path / "bdd100k_night_selected" / "dataset_info.txt"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)

        print(f"\n📝 Dataset info saved to: {config_path}")


def main():
    """Main function"""
    import time

    print("=" * 60)
    print("Autonomous Driving Night Object Detection Dataset - Intelligent Selection Tool")
    print("=" * 60)

    # Base path - adjust according to your actual path
    base_path = r"F:\deeplearning\ultralytics-8.3.163\datasets\BDD100K"

    # Initialize selector
    selector = BalancedNightDatasetSelector(base_path, target_total=3500)

    # 1. Analyze dataset
    selector.analyze_dataset()

    # 2. Intelligent selection
    selected = selector.select_balanced_subset()

    # 3. Copy selected data
    selector.copy_selected_data(selected)

    # 4. Print statistics
    selector.print_selection_statistics()

    print("\n🎉 Dataset selection completed! You now have a high-quality, category-balanced")
    print(f"   night object detection dataset containing ~{len(selected)} images.")
    print("\n💡 Next steps:")
    print("   1. Check the generated dataset_info.txt file")
    print("   2. Split the selected dataset into train/val/test (suggest 7:2:1)")
    print("   3. Start your experiments with a pre-trained model!")


if __name__ == "__main__":
    main()
