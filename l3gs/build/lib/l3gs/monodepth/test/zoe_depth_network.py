import time
from matplotlib import pyplot as plt
import torch
import os

repo = "isl-org/ZoeDepth"
# Zoe_N
# model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)

# Zoe_K
# model_zoe_k = torch.hub.load(repo, "ZoeD_K", pretrained=True)

# Zoe_NK
model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)

##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_nk.to(DEVICE)

# f = open("depths/times.txt", "a")


# Local file
from PIL import Image
import cv2
image = cv2.imread(os.path.join("test","testimages","test1.jpg"))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
depth_numpy = zoe.infer_pil(image)  # as numpy

from torchvision.transforms import ToTensor
image = ToTensor()(image).to(DEVICE).unsqueeze(0)
depth_tensor = zoe.infer(image)  # as torch tensor
# torch.from_numpy(image).permute(2,0,1).unsqueeze(0).to(DEVICE)
startT = time.time()

# depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image

# depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor

zoe.infer(image)

endTime = time.time() - startT
print(str(endTime) + "\n")
# f.write("zoed_nk" + " " + str(endTime) + "\n")



# # Colorize output
# from zoedepth.utils.misc import colorize

# colored = colorize(depth)

# # save colored output
# fpath_colored = "/path/to/output_colored.png"
# Image.fromarray(colored).save(fpath_colored)

# plt.imshow(depth_numpy)
# plt.savefig(f"depths/zoe_nk_depth.png")
