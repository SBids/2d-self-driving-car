import pygame
import math
import os
from utils import rot_center

class Car(pygame.sprite.Sprite):
    def __init__(self,src, start_dir, start_x, start_y):
        self.start_x = start_x
        self.start_y = start_y
        self.x = self.start_x
        self.y = self.start_y
        self.dir = start_dir
        self.speed = 0
        self.wheel = 0  # -1 to 1
        image = pygame.image.load(src)
        self.image = pygame.transform.scale(image, (20,20))
        # pygame.transform.scale(image, DEFAULT_IMAGE_SIZE)
        self.img = None
        self.img_mask = None
        self.img_rect = None
        self.max_wheel_pos = 1
        pygame.sprite.Sprite.__init__(self)

    def reset(self):
        self.dir = 0
        self.x = self.start_x
        self.y = self.start_y

    def update_pos(self, dt):
        if (self.wheel > self.max_wheel_pos):
            self.wheel = self.max_wheel_pos
        elif (self.wheel < -self.max_wheel_pos):
            self.wheel = -self.max_wheel_pos


        self.dir -= self.wheel * self.speed / 33 * dt
        self.x += math.cos(math.radians(self.dir)) * self.speed / 33 * dt
        self.y -= math.sin(math.radians(self.dir)) * self.speed / 33 * dt

    def blit(self, screen, dt):
        self.img = rot_center(self.image, self.dir)
        self.img_mask = pygame.mask.from_surface(self.img)
        self.img_rect = self.img.get_rect()
        screen.blit(self.img, (self.x, self.y))
        self.update_pos(dt)