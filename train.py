
import torch
from PIL import Image

from transformers import Sam3Processor, Sam3Model
from src.preprocessing import DrywallQADatasetCustom
import matplotlib.pyplot as plt

device = "cuda:0" if torch.cuda.is_available() else "cpu"


d2_train = DrywallQADatasetCustom("/home/saitama/Documents/Prompted Segmentation for Drywall QA/datasets/cracks_v1/train")


model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

input = processor(d2_train[5]['image'], text="Segment wall cracks", return_tensors="pt").to(device)


with torch.inference_mode():
    output = model(**input)

# print(outputs)

result = processor.post_process_instance_segmentation(
                            output,
                            threshold=0.25,
                            mask_threshold=0.5,
                            target_sizes=[d2_train[0]['image'].size[::-1]]
)

masks = result[0]["masks"]

if masks.shape[0] > 0:
    print("cracks found")

fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax = ax.flatten()
ax[0].imshow(d2_train[5]['image'])
ax[1].imshow(result[0]["masks"][0].cpu().numpy())
# ax[2].imshow(mask1)
# ax[3].imshow(mask2)
plt.show()

