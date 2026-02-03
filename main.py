
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from src.preprocessing import DrywallQADatasetCustom
import matplotlib.pyplot as plt


def main():
    d = DrywallQADatasetCustom("/home/saitama/Documents/Prompted Segmentation for Drywall QA/datasets/Drywall_Join_Detect_v1/train/")
    # print(len(d))
    print(d[0])    
    image = d[0]['mask']
    # print(annotations)
    plt.imshow(image)
    plt.show()
    # d.data_info()


if __name__ == "__main__":
    main()