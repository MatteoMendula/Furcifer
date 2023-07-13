import os

file_with_coco_names = "category_names.txt"
class_names = open("category_names.txt").readlines()
class_names = [c.strip() for c in class_names]
print(class_names)