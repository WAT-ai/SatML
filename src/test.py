from src.data_loader import data_generator, bbox_data_generator

test_dir = './data/raw_data/STARCOP_train_easy'

# Test data_generator function
print("Testing data_generator...")
for i, (images, labels, directory) in enumerate(data_generator(test_dir)):
    print(f"[data_generator] Sample {i+1}:")
    print(f"  Image shape: {images.shape}")
    print(f"  Label shape: {labels.shape}")
    print(f"  Directory: {directory}")
    break  # Tests only 1 entry

# Test bbox_data_generator function
print("\nTesting bbox_data_generator...")
for i, (images, bboxes, directory) in enumerate(bbox_data_generator(test_dir, max_boxes=10)):
    print(f"[bbox_data_generator] Sample {i+1}:")
    print(f"  Image shape: {images.shape}")
    print(f"  BBoxes shape: {bboxes.shape}")
    print(f"  Directory: {directory}")
    break  # Tests only 1 entry