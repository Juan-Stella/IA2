# Capture area used for dataset screenshots.
# Pygame rects use (x, y, width, height), while Keras target_size uses
# (height, width).
CROP_LEFT = 55
CROP_TOP = 100
CROP_WIDTH = 520
CROP_HEIGHT = 330
CROP_RIGHT = CROP_LEFT + CROP_WIDTH

MODEL_IMAGE_SIZE = (120, 190)
LIVE_IMAGE_PATH = "./images/live/temp.png"
