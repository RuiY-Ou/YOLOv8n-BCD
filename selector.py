import json
import os
import random
import shutil
from pathlib import Path
import cv2
from collections import defaultdict, Counter


class BalancedNightDatasetSelector:
    def __init__(self, base_path, target_total=3500):
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
        image_files = list(self.night_images_train.glob("*.jpg"))
        if not image_files:
            image_files = list(self.night_images_train.glob("*.jpeg")) + \
                          list(self.night_images_train.glob("*.png"))

        self.stats['total_images'] = len(image_files)

        for img_path in image_files:
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

    def select_balanced_subset(self):
        self.image_info.sort(key=lambda x: (
            -x['unique_categories'],
            -x['total_objects']
        ))

        selected = []
        selected_names = set()
        current_category_counts = defaultdict(int)

        for cat_id in self.target_categories:
            cat_images = [info for info in self.image_info
                          if cat_id in info['categories'] and info['name'] not in selected_names]

            for info in cat_images[:max(50, self.category_min_targets[cat_id] // 20)]:
                if info['name'] not in selected_names:
                    selected.append(info)
                    selected_names.add(info['name'])
                    for cid, count in info['category_counts'].items():
                        current_category_counts[cid] += count

        while len(selected) < self.target_total and len(selected) < len(self.image_info):
            def get_category_satisfaction():
                satisfaction = {}
                for cat_id, min_target in self.category_min_targets.items():
                    current = current_category_counts[cat_id]
                    satisfaction[cat_id] = min(current / min_target, 1.0) if min_target > 0 else 1.0
                return satisfaction

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
            else:
                break

        final_selected = selected[:self.target_total]

        self.stats['selected_images'] = len(final_selected)
        self.stats['selected_category_counts'] = defaultdict(int)

        for info in final_selected:
            for cat_id, count in info['category_counts'].items():
                self.stats['selected_category_counts'][cat_id] += count
            self.stats['selected_objects'] += info['total_objects']

        return final_selected

    def copy_selected_data(self, selected_images):
        for info in selected_images:
            img_name = info['name']
            src_img = self.night_images_train / f"{img_name}.jpg"
            src_label = self.night_labels_train / f"{img_name}.txt"
            dst_img = self.selected_images / f"{img_name}.jpg"
            dst_label = self.selected_labels / f"{img_name}.txt"
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            if src_label.exists():
                shutil.copy2(src_label, dst_label)

    def print_selection_statistics(self):
        avg_objects = self.stats['selected_objects'] / self.stats['selected_images'] if self.stats['selected_images'] > 0 else 0

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


def main():
    base_path = r"F:\deeplearning\ultralytics-8.3.163\datasets\BDD100K"
    selector = BalancedNightDatasetSelector(base_path, target_total=3500)
    selector.analyze_dataset()
    selected = selector.select_balanced_subset()
    selector.copy_selected_data(selected)
    selector.print_selection_statistics()


if __name__ == "__main__":
    main()
