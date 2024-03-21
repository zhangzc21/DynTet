import json
from torch.utils.tensorboard import SummaryWriter
import os
# 读取包含不同损失的JSON文件

path = "losses/3dmm_loss.json"

with open(path, 'r') as json_file:
    loss_data = json.load(json_file)

# 初始化TensorBoard的SummaryWriter
log_dir = os.path.join('tensorboard', os.path.basename(os.path.basename(os.path.dirname(path))))  # 保存TensorBoard日志的目录
writer = SummaryWriter(log_dir)

# 遍历不同损失并将它们写入TensorBoard日志
for loss_name, loss_values in loss_data.items():
    for step, loss_value in enumerate(loss_values):
        writer.add_scalar(loss_name, loss_value, global_step=step)

# 关闭SummaryWriter
writer.close()

os.system(f'tensorboard --logdir tensorboard')
