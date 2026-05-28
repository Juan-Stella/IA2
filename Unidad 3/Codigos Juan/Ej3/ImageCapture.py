import uuid
from pathlib import Path

import pygame

from CaptureConfig import CROP_HEIGHT, CROP_LEFT, CROP_TOP, CROP_WIDTH, LIVE_IMAGE_PATH


class ImageCapture():
    def __init__(self, screen_spawn_position):
        self.count = 0

        # Prepare the directories in which the images are stored
        Path("./images/").mkdir(parents=True, exist_ok=True)
        Path("./images/up/").mkdir(parents=True, exist_ok=True)
        Path("./images/down/").mkdir(parents=True, exist_ok=True)
        Path("./images/right/").mkdir(parents=True, exist_ok=True)
        Path("./images/live/").mkdir(parents=True, exist_ok=True)
        self.ss_id = uuid.uuid4()

    def capture_rect(self, screen):
        screen_rect = screen.get_rect()
        return pygame.Rect(CROP_LEFT, CROP_TOP, CROP_WIDTH, CROP_HEIGHT).clip(screen_rect)

    def save_screen_crop(self, screen, path):
        screenshot = screen.subsurface(self.capture_rect(screen)).copy()
        pygame.image.save(screenshot, path)

    def take_screenshot(self, key, screen):
        self.count += 1
        self.save_screen_crop(screen, "./images/{}/{}.png".format(key, self.count))

    def capture(self, userInput, screen):
        # Take a screenshot on command and tag it on the pressed button folder
        if userInput[pygame.K_UP]:
            self.take_screenshot("up", screen)

        elif userInput[pygame.K_DOWN]:
            self.take_screenshot("down", screen)

        else:
            self.take_screenshot("right", screen)

    def capture_live(self, screen):
        # Automatically take a screenshot for the Tensorflow model to work
        self.save_screen_crop(screen, LIVE_IMAGE_PATH)
