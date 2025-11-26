from os import makedirs
from env import root_dir


logs_path = f"{root_dir}/logs"
chkpts_path = f"{root_dir}/chkpts"
exports_path = f"{root_dir}/exports"
videos_path = f"{root_dir}/videos"
scene_path = f"{root_dir}/scene/scene.xml"

# Create directories
makedirs(logs_path, exist_ok=True)
makedirs(chkpts_path, exist_ok=True)
makedirs(exports_path, exist_ok=True)
makedirs(videos_path, exist_ok=True)
