import os
from pathlib import Path

import numpy as np
import cv2

from torchvision import transforms
from torch.utils import data
from utils.kitti_utils import get_kitti_directories, load_label
from utils.class_dimensions import get_class_dimensions, compute_class_dimensions_from_kitti
from utils.angles import get_bin, generate_bins
import glob

class Dataset(data.Dataset):
    def __init__(self, kitti_base_dir, output_dimensions_path, bins=2, overlap=0.1):
        # Get KITTI directories
        paths = get_kitti_directories(kitti_base_dir)
        self.kitti_images_dir = paths['images']
        self.kitti_labels_dir = paths['labels']
        self.kitti_calib_dir = paths['calib']

        # Get images
        self.image_files = sorted(glob.glob(os.path.join(self.kitti_images_dir, "*.png")))
        self.num_images = len(self.image_files)

        # Create angle bins
        self.bins = bins
        self.angle_bins = generate_bins(self.bins)
        self.interval = 2 * np.pi / self.bins
        self.overlap = overlap

        # Ranges for confidence
        # [(min angle in bin, max angle in bin), ... ]
        self.bin_ranges = []
        for i in range(0,bins):
            self.bin_ranges.append(( (i*self.interval - overlap) % (2*np.pi), \
                                (i*self.interval + self.interval + overlap) % (2*np.pi)) )

        # List of objects
        self.object_list = []
        for img_file in self.image_files:
            img_name = Path(img_file).stem
            label_file = os.path.join(self.kitti_labels_dir, f"{img_name}.txt")
            if not os.path.exists(label_file):
                continue
            
            objects = load_label(label_file)

            # Add image file path to each object
            for obj in objects:
                obj['image_path'] = img_file

            self.object_list.extend(objects)

        # Load class average
        compute_class_dimensions_from_kitti(kitti_base_dir, save_file=output_dimensions_path)

    def _add_orientation_confidence(self, label):
        """
        Add orientation confidence to the label based on angle bins
        """
        angle = label['alpha'] + np.pi
        bin_idxs = get_bin(angle, self.bins, self.overlap)
        orientation = np.zeros((self.bins, 2))
        confidence = np.zeros(self.bins)
        for bin_idx in bin_idxs:
            angle_diff = angle - self.angle_bins[bin_idx]

            orientation[bin_idx,:] = np.array([np.cos(angle_diff), np.sin(angle_diff)])
            confidence[bin_idx] = 1
        label['orientation'] = orientation
        label['confidence'] = confidence
        return label
    
    def _update_dimensions(self, label):
        """
        Update dimensions of the label based on class
        """
        # Convert dimensions to numpy array if it's a tuple
        dimensions = np.array(label['dimensions'])
        class_dims = np.array(get_class_dimensions(label['class_name']))
        
        # Subtract class dimensions
        dimensions = dimensions - class_dims
        
        label['dimensions'] = dimensions
        return label

    def __getitem__(self, index):
        label = self.object_list[index]
        label = self._add_orientation_confidence(label)
        label = self._update_dimensions(label)

        img = format_img(cv2.imread(label['image_path']), label['bbox_2d'])

        return img, label

    def __len__(self):
        return len(self.object_list)
    
def format_img(img, bbox_2d):
    # transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    process = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    x1, y1, x2, y2 = bbox_2d
    height, width = img.shape[:2]

    # Convert to integers and clamp to image boundaries
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))
    
    # Crop the image
    crop = img[y1:y2, x1:x2]

    # Crop image
    crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)

    # apply transform for batch
    batch = process(crop)

    return batch
