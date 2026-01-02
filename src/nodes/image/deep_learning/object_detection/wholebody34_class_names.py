# Wholebody34 class names for DEIMv2
# Class IDs:
# 0: Body, 1: Adult, 2: Child, 3: Male, 4: Female
# 5: Body-With-Wheelchair, 6: Body-With-Crutches, 7: Head
# 8: Front, 9: Right-Front, 10: Right-Side, 11: Right-Back
# 12: Back, 13: Left-Back, 14: Left-Side, 15: Left-Front
# 16: Face, 17: Eye, 18: Nose, 19: Mouth, 20: Ear
# 21: Collarbone, 22: Shoulder, 23: Solar_plexus, 24: Elbow, 25: Wrist
# 26: Hand, 27: Left-Hand, 28: Right-Hand, 29: Abdomen
# 30: Hip_joint, 31: Knee, 32: Ankle, 33: Foot

wholebody34_class_names = {
    0: "Body",
    1: "Adult",
    2: "Child",
    3: "Male",
    4: "Female",
    5: "Body-With-Wheelchair",
    6: "Body-With-Crutches",
    7: "Head",
    8: "Front",
    9: "Right-Front",
    10: "Right-Side",
    11: "Right-Back",
    12: "Back",
    13: "Left-Back",
    14: "Left-Side",
    15: "Left-Front",
    16: "Face",
    17: "Eye",
    18: "Nose",
    19: "Mouth",
    20: "Ear",
    21: "Collarbone",
    22: "Shoulder",
    23: "Solar_plexus",
    24: "Elbow",
    25: "Wrist",
    26: "Hand",
    27: "Left-Hand",
    28: "Right-Hand",
    29: "Abdomen",
    30: "Hip_joint",
    31: "Knee",
    32: "Ankle",
    33: "Foot",
}

# Head pose colors (BGR format)
HEAD_POSE_COLORS = {
    0: (216, 67, 21),   # Front
    1: (255, 87, 34),   # Right-Front
    2: (123, 31, 162),  # Right-Side
    3: (255, 193, 7),   # Right-Back
    4: (76, 175, 80),   # Back
    5: (33, 150, 243),  # Left-Back
    6: (156, 39, 176),  # Left-Side
    7: (0, 188, 212),   # Left-Front
}

# Head pose names
HEAD_POSE_NAMES = {
    0: "Front",
    1: "Right-Front",
    2: "Right-Side",
    3: "Right-Back",
    4: "Back",
    5: "Left-Back",
    6: "Left-Side",
    7: "Left-Front",
}

# Edge connections for skeleton
WHOLEBODY34_EDGES = [
    (21, 22), (21, 22),  # collarbone -> shoulder (left and right)
    (21, 23),            # collarbone -> solar_plexus
    (22, 24), (22, 24),  # shoulder -> elbow (left and right)
    (22, 30), (22, 30),  # shoulder -> hip_joint (left and right)
    (24, 25), (24, 25),  # elbow -> wrist (left and right)
    (23, 29),            # solar_plexus -> abdomen
    (29, 30), (29, 30),  # abdomen -> hip_joint (left and right)
    (30, 31), (30, 31),  # hip_joint -> knee (left and right)
    (31, 32), (31, 32),  # knee -> ankle (left and right)
]
