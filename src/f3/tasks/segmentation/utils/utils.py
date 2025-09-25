import numpy as np

# https://github.com/open-mmlab/mmsegmentation/blob/00790766aff22bd6470dbbd9e89ea40685008395/mmseg/utils/class_names.py#L249C1-L249C1
def cityscapes_palette(num_classes=19):
    """Cityscapes palette for external use."""
    palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
                [0, 0, 230], [119, 11, 32]]
    palette = np.array(palette)
    if num_classes == 11:
        palette = palette[np.array([
            10, 2, 4, 11, 5, 0, 1, 8, 13, 3, 7 
        ])]
    return np.vstack((palette, np.array([0, 0, 0])))


def classes19_to_classes11():
    """
        Convert the 19 class cityscapes labels to 11 class labels.
        
        Cityscapes has 19 classes, but DSEC and most event evaluate suites have 11 classes.
        So this function maps the 19 classes to 11 classes.
    """
    mapping = np.zeros(256, dtype=np.uint8)
    mapping[255] = 255
    mapping[:19] = np.array([
        5, 6, 1, 9, 2, 4, 10, 10, 7, 7, 0, 3, 3, 8, 8, 8, 8, 8, 8 # based on the mapping given in labels.py of DSEC-semantic
    ])
    return mapping
