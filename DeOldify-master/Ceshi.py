import torch.backends.cudnn

from deoldify import device
from deoldify.device_id import DeviceId

device.set(device= DeviceId.CPU)
from deoldify.visualize import *


plt.style.use('dark_background')
torch.backends.cudnn.benchmark = True # 加速训练
ceshi1 = Path('./')
print(ceshi1)
colorizer = get_image_colorizer(artistic=False)
render_factor = 30

source_path = "H:/Data/ceshi.png"



#colorizer.plot_transformed_image(path = source_path,render_factor = render_factor,compare=True)

