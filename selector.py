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
        Initialize selector.
        Args:
            base_path: root path of dataset
            target_total: target number of images to select
        """
        self.base_path = Path(base_path)
        self.target_total = target_total

        self.night_images_train = self.base_path / "bdd100k_night" / "images" / "train"
        self.night_labels_train = self.base_path / "bdd100k_night" / "labels" / "train"

        self.selected_images = self.base_path / "bdd100k_night_selected" / "images" / "train"
        self.selected_labels = self.base_path / "bdd100k_night_selected" / "labels" / "train"

        for path in [self.selected_images, self.selected_labels]:
            path.mkdir(parents=True, exist_ok=True)

        self.target_categories = {
            0: 'person',
            1: 'car',
            2: 'bike',
            3: 'motor',
            4: 'traffic light'
        }

        self.category_min_targets = {
            0: 700,
            1: 1500,
            2: 600,
            3: 400,
            4: 800
        }

        self.image_info = []

        self.stats = {
            'total_images': 0,
            'total_objects': 0,
            'category_counts': defaultdict(int),
            'selected_images': 0,
            'selected_objects': 0,
            'selected_category_counts': defaultdict(int)
        }

    def analyze_dataset(self):
        """Analyze dataset and collect category info per image."""
        print("Analyzing dataset...")

        image_files = list(self.night_images_train.glob("*.jpg"))
        if not image_files:
            image_files = list(self.night_images_train.glob("*.jpeg")) + \
                          list(self.night_images_train.glob("*.png"))

        self.stats['total_images'] = len(image_files)
        print(f"Found {self.stats['total_images']} night images")

        for img_path in tqdm(image_files, desc="Analyzing"):
            img_name = img_path.stem
            label_path = self.night_labels_train / f"{img_name}.txt"

            if not label_path.exists():
                continue

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

            for cat_id, count in category_counter.items():
                self.stats['category_counts'][cat_id] += count
            self.stats['total_objects'] += total_objects

            if categories_in_image:
                self.image_info.append({
                    'name': img_name,
                    'categories': categories_in_image,
                    'category_counts': dict(category_counter),
                    'total_objects': total_objects,
                    'unique_categories': len(categories_in_image)
                })

        print("Dataset analysis completed.")
        print("=" * 50)
        print("Original dataset statistics:")
        print(f"  Total images: {self.stats['total_images']}")
        print(f"  Total objects: {self.stats['total_objects']}")
        print("  Category counts:")
        for cat_id, cat_name in sorted(self.target_categories.items()):
            count = self.stats['category_counts'][cat_id]
            percentage = (count / self.stats['total_objects'] * 100) if self.stats['total_objects'] > 0 else 0
            print(f"    {cat_name} (ID:{cat_id}): {count} ({percentage:.1f}%)")
        print("=" * 50)

    def select_balanced_subset(self):
        """Intelligently select a balanced subset."""
        print("\nStarting intelligent selection...")

        print("1. Sorting by information density...")
        self.image_info.sort(key=lambda x: (
            -x['unique_categories'],
            -x['total_objects']
        ))

        selected = []
        selected_names = set()
        current_category_counts = defaultdict(int)

        print("2. Ensuring basic coverage for each category...")
        for cat_id in self.target_categories:
            cat_images = [info for info in self.image_info
                          if cat_id in info['categories'] and info['name'] not in selected_names]

            for info in cat_images[:max(50, self.category_min_targets[cat_id] // 20)]:
                if info['name'] not in selected_names:
                    selected.append(info)
                    selected_names.add(info['name'])
                    for cid, count in info['category_counts'].items():
                        current_category_counts[cid] += count

        print("3. Filling to target size...")
        pbar = tqdm(total=self.target_total, desc="Selection progress")
        pbar.update(len(selected))

        def get_category_satisfaction():
            satisfaction = {}
            for cat_id, min_target in self.category_min_targets.items():
                current = current_category_counts[cat_id]
                satisfaction[cat_id] = min(current / min_target, 1.0) if min_target > 0 else 1.0
            return satisfaction

        while len(selected) < self.target_total and len(selected) < len(self.image_info):
            satisfaction = get_category_satisfaction()
            most_unsatisfied = min(satisfaction.items(), key=lambda x: x[1])[0]

            best_image = None
            best_score = -1

            for info in self.image_info:
                if info['name'] in selected_names:
                    continue
                if most_unsatisfied in info['categories']:
                    score = 0
                    score += info['category_counts'].get(most_unsatisfied, 0) * 5
                    for cat_id, sat in satisfaction.items():
                        if sat < 0.8 and cat_id in info['categories']:
                            score += info['category_counts'].get(cat_id, 0) * 3
                    score += info['unique_categories'] * 2

                    if score > best_score:
                        best_score = score
                        best_image = info

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
                for cat_id, count in best_image['category_counts'].items():
                    current_category_counts[cat_id] += count
                pbar.update(1)
            else:
                break

        pbar.close()

        print("4. Final trimming...")
        final_selected = selected[:self.target_total]

        self.stats['selected_images'] = len(final_selected)
        self.stats['selected_category_counts'] = defaultdict(int)

        for info in final_selected:
            for cat_id, count in info['category_counts'].items():
                self.stats['selected_category_counts'][cat_id] += count
            self.stats['selected_objects'] += info['total_objects']

        return final_selected

    def copy_selected_data(self, selected_images):
        """Copy selected images and labels to output directory."""
        print(f"\nCopying {len(selected_images)} selected images and labels...")

        for info in tqdm(selected_images, desc="Copying"):
            img_name = info['name']
            src_img = self.night_images_train / f"{img_name}.jpg"
            src_label = self.night_labels_train / f"{img_name}.txt"
            dst_img = self.selected_images / f"{img_name}.jpg"
            dst_label = self.selected_labels / f"{img_name}.txt"

            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            if src_label.exists():
                shutil.copy2(src_label, dst_label)

        print("Copy completed.")

    def print_selection_statistics(self):
        """Print statistics of the selected subset."""
        print("\n" + "=" * 60)
        print("Selection completed. Detailed statistics:")
        print("=" * 60)

        print(f"\nSize comparison:")
        print(f"  Original: {self.stats['total_images']} images, {self.stats['total_objects']} objects")
        print(f"  Selected: {self.stats['selected_images']} images, {self.stats['selected_objects']} objects")

        print(f"\nCategory distribution:")
        print("  " + "Category".ljust(15) + "Original".ljust(12) + "Selected".ljust(12) + "Percent".ljust(12))
        print("  " + "-" * 50)

        total_original = self.stats['total_objects']
        total_selected = self.stats['selected_objects']

        for cat_id, cat_name in sorted(self.target_categories.items()):
            orig_count = self.stats['category_counts'][cat_id]
            sel_count = self.stats['selected_category_counts'][cat_id]
            sel_pct = (sel_count / total_selected * 100) if total_selected > 0 else 0
            print(f"  {cat_name.ljust(15)}"
                  f"{str(orig_count).ljust(12)}"
                  f"{str(sel_count).ljust(12)}"
                  f"{sel_pct:.1f}%")

        avg_objects = self.stats['selected_objects'] / self.stats['selected_images'] if self.stats['selected_images'] > 0 else 0
        print(f"\nKey metrics:")
        print(f"  Average objects per image: {avg_objects:.1f}")
        categories_covered = sum(1 for cat_id in self.target_categories if self.stats['selected_category_counts'][cat_id] > 0)
        print(f"  Covered categories: {categories_covered}/{len(self.target_categories)}")
        print(f"  Min target satisfaction:")
        for cat_id, cat_name in self.target_categories.items():
            current = self.stats['selected_category_counts'][cat_id]
            target = self.category_min_targets.get(cat_id, 0)
            satisfied = "OK" if current >= target else "LOW"
            print(f"    {satisfied}: {cat_name} = {current} (min {target})")

        print(f"\nOutput paths:")
        print(f"  Images: {self.selected_images}")
        print(f"  Labels: {self.selected_labels}")
        print("=" * 60)

        config_content = f"""# Balanced night detection dataset configuration

Dataset size:
  Images: {self.stats['selected_images']}
  Total objects: {self.stats['selected_objects']}
  Avg objects per image: {avg_objects:.2f}

Category distribution:
"""
        for cat_id, cat_name in sorted(self.target_categories.items()):
            count = self.stats['selected_category_counts'][cat_id]
            pct = (count / self.stats['selected_objects'] * 100) if self.stats['selected_objects'] > 0 else 0
            config_content += f"  {cat_name}: {count} objects ({pct:.1f}%)\n"

        config_path = self.base_path / "bdd100k_night_selected" / "dataset_info.txt"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print(f"\nDataset info saved to: {config_path}")


def main():
    print("=" * 60)
    print("Nighttime Object Detection Dataset - Balanced Selector")
    print("=" * 60)

    base_path = r"F:\deeplearning\ultralytics-8.3.163\datasets\BDD100K"

    selector = BalancedNightDatasetSelector(base_path, target_total=3500)

    selector.analyze_dataset()

    selected = selector.select_balanced_subset()

    selector.copy_selected_data(selected)

    selector.print_selection_statistics()

    print("\nSelection completed successfully!")


if __name__ == "__main__":
    main()
