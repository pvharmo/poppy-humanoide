import time

import mujoco
from mujoco import viewer

xml = open("../scene/scene.xml", "r").read()
scene = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(scene)

joints = [
    "abs_x",
    "abs_y",
    "abs_z",
    "bust_x",
    "bust_y",
    "floating_base",
    "head_y",
    "head_z",
    "l_ankle_y",
    "l_arm_z",
    "l_elbow_y",
    "l_hip_x",
    "l_hip_y",
    "l_hip_z",
    "l_knee_y",
    "l_shoulder_x",
    "l_shoulder_y",
    "r_ankle_y",
    "r_arm_z",
    "r_elbow_y",
    "r_hip_x",
    "r_hip_y",
    "r_hip_z",
    "r_knee_y",
    "r_shoulder_x",
    "r_shoulder_y",
]

bodies = [
    "abdomen",
    "abs_motors",
    "bust_motors",
    "chest",
    "head",
    "l_foot",
    "l_forearm",
    "l_hip",
    "l_hip_motor",
    "l_shin",
    "l_shoulder",
    "l_shoulder_motor",
    "l_thigh",
    "l_upper_arm",
    "neck",
    "pelvis",
    "r_foot",
    "r_forearm",
    "r_hip",
    "r_hip_motor",
    "r_shin",
    "r_shoulder",
    "r_shoulder_motor",
    "r_thigh",
    "r_upper_arm",
    "spine",
    "world",
]

print(scene.sensor("pelvis_site_pos"))

# for body in bodies:
#     scene.body(body).mass = scene.body(body).mass
#     print(scene.body(body).mass)

# for joint in joints:
#     print(scene.joint(joint))

# print(scene.body("r_hip_x"))

# viewer.launch(scene)

# viewer = viewer.launch_passive(scene, data)

# start = time.time()
# while viewer.is_running() and time.time() - start < 30:
#     step_start = time.time()

#     # mj_step can be replaced with code that also evaluates
#     # a policy and applies a control signal before stepping the physics.
#     mujoco.mj_step(scene, data)

#     # Pick up changes to the physics state, apply perturbations, update options from GUI.
#     viewer.sync()
