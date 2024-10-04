import numpy as np
import math

PI = 3.14159265359
ARM_LOWER = [-2*PI, -2*PI, -2*PI, -2*PI, -2*PI, -2*PI]
ARM_UPPER = [2*PI, 2*PI, 2*PI, 2*PI, 2*PI, 2*PI]
ARM_RANGE = [4*PI, 4*PI, 4*PI, 4*PI, 4*PI, 4*PI]
ARM_HOME = [PI/4, -PI/2, -PI/2, -PI/2, PI/2, 0]
ARM_DAMP = [0.01, 0.01, 0.001, 0.001, 0.001, 0.001]

GRIPPER_LOWER = [0, -0.8, 0, 0, -0.8, 0]
GRIPPER_UPPER = [0.8, 0, 0.8, 0.8, 0, 0.8]
GRIPPER_RANGE = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
GRIPPER_HOME = [0, 0, 0, 0, 0, 0]
GRIPPER_DAMP = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]

IDX_ARM = [1, 2, 3, 4, 5, 6]
IDX_GRIPPER = [10, 12, 14, 15, 17, 19]
IDX_TCP = 20
IDX_CAMERA = 23
IDX_CAMERA2 = 26

# WORKSPACE_LIMITS = np.asarray([[0.276, 0.724], [-0.224, 0.224], [-0.0001, 0.4]])
WORKSPACE_LIMITS = [0.25, 0.65, -0.22, 0.22, 0.85, 1.25]

# task
LANG_TEMPLATES = ["give me the {keyword}", # label
                "I need a {keyword}", # general label
                "grasp a {keyword} object", # shape or color
                "I want a {keyword} object", # shape or color
                # "get something to {keyword}", # function
                ]

# LABEL = ["banana", "red mug", "scissors", "strawberry", "apple", "lemon", 
#         "peach", "pear", "orange", "knife", "flat screwdriver", "racquetball", "cup", "toy airplane", "screw"
#         "dabao sod", "toothpaste", "darlie box", "dabao facewash", "pantene", "head shoulders", "tape"]

GENERAL_LABEL = ["fruit", "container", "toy", "cup"]
COLOR_SHAPE = ["yellow", "red", "round"]
FUNCTION = ["eat", "drink", "play", "hold other things"]
# LABEL_DIR_MAP = ["005", "007", "009", "011", "012", "013",
#                 "014", "015", "016", "018", "020", "021", "022", "024", "027",
#                 "038", "041", "058", "061", "062", "066", "070"]
LABEL = ["banana", "red mug", "scissors", 
         "strawberry", "apple", "lemon", 
         "peach", "pear", "orange", 
         "knife", "flat screwdriver", "racquetball", 
         "cup", "toy airplane", "screw"
         "dabao sod", "toothpaste", "darlie box", 
         "dabao facewash", "pantene", "head shoulders", 
         "tape", "washbasin", "plum",
         "flat-head screwdriver bits", "three-blade fan", "Hexagonal socket",
         "phillips screwdriver head", "Clamping device", "lock",
         "water emulsion", "mouthwash",
         "box", "laundry detergent",
         "facial cleanser", "marker pen", "toothpaste pressed in bottle",
         "e-sports mouse", "zebra", "rhinoceros",
         "elephant", "giraffe", "yogurt bottle",
         "soap", "toothpaste pressed in tube", "lion",
         "box", "adhesive tape", "grater",
         "film wiper", "ice cube tray"]


LABEL_DIR_MAP = ["005", "007", "009", 
                 "011", "012", "013",
                 "014", "015", "016", 
                 "018", "020", "021", 
                 "022", "024", "027",
                 "038", "041", "058", 
                 "061", "062", "066", 
                 "070", "006", "017",
                 "026", "028", "029",
                 "030", "031", "032",
                 "034", "037", 
                 "039", "040", 
                 "042", "043", "044",
                 "047", "050", "052",
                 "053", "055", "057",
                 "059", "064", "067",
                 "068", "070", "072",
                 "073", "074"
                 ]


KEYWORD_DIR_MAP = {"fruit": ["005", "011", "012", "013", "014", "015", "016", "017"],
                    "container": ["006", "007", "022"],
                #     "toy": ["024", "026", "027", "028", "029", "030", "031",
                #             "075", "076", "077", "078", "079", "080", "081", "082", "083", 
                #             "084", "085", "086", "087"],
                    "toy": ["024", "026", "027", "028", "029", "030", "031"],
                    "cup": ["022"],
                    "yellow": ["005", "013", "028", "031"],
                    "red": ["011", "012"],
                    "round": ["016", "017", "021"],
                    "box": ["039"],
                    "eat": ["005", "011", "012", "013", "014", "015", "016", "017"], 
                    "drink": ["057"],
                    "play": ["024", "026", "027", "028", "029", "030", "031"],
                    "hold other things": ["006", "007", "022"]}

UNSEEN_LABEL = ["black marker", "bleach cleanser", "blue moon", "gelatin box", "magic clean", "pink tea box", "red marker", 
                "remote controller", "repellent", "shampoo", "small clamp", "soap dish", "suger", "suger", "two color hammer",
                "yellow bowl", "yellow cup"]

UNSEEN_LABEL_DIR_MAP = ["black_marker", "bleach_cleanser", "blue_moon", "gelatin_box", "magic_clean", "pink_tea_box", "red_marker", 
                "remote_controller_1", "repellent", "shampoo", "small_clamp", "soap_dish", "suger_1", "suger_2", "two_color_hammer",
                "yellow_bowl", "yellow_cup"]

UNSEEN_GENERAL_LABEL = ["suger", "container"]
UNSEEN_COLOR_SHAPE = ["yellow", "red"]
UNSEEN_FUNCTION = ["clean"]
UNSEEN_KEYWORD_DIR_MAP = {"suger": ["suger_1", "suger_2"],
                    "container": ["soap_dish", "yellow_cup"],
                    "yellow": ["yellow_bowl", "yellow_cup"],
                    "red": ["red_marker"],
                    "clean": ["bleach_cleanser", "blue_moon", "magic_clean", "shampoo"]}

# image
PIXEL_SIZE = 0.002
IMAGE_SIZE = 224