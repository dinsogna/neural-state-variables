import os
import math
import random
from PIL import Image, ImageDraw
import numpy as np

class CircularMotion():
   
   def __init__(self, canvas_height, canvas_width, canvas_radius, point_radius, point_omega, canvas_color='gray', point_color='blue'):
      self.canvas_height = canvas_height
      self.canvas_width = canvas_width
      self.canvas_radius = canvas_radius
      self.point_radius = point_radius
      self.point_omega = point_omega
      self.canvas_color = canvas_color
      self.point_color = point_color
      self.angle = random.random() * 2 * math.pi

   def next_frame(self):
      self.angle = self.angle + self.point_omega / 60
      self.center = (int(self.canvas_width / 2 + self.canvas_radius * math.cos(self.angle)),
                     int(self.canvas_height / 2 - self.canvas_radius * math.sin(self.angle)))
      image = Image.new('RGB', (self.canvas_width, self.canvas_height))
      draw = ImageDraw.Draw(image)
      draw.rectangle((0, 0, 128, 128), fill=self.canvas_color)
      draw.ellipse((self.center[0] - self.point_radius,
                    self.center[1] - self.point_radius,
                    self.center[0] + self.point_radius,
                    self.center[1] + self.point_radius), fill=self.point_color, outline=self.point_color)
      return image, self.angle

def main():

   num_videos = int(input('Enter number of videos to generate: '))
   if num_videos < 1:
      raise ValueError('The number of videos to generate must be at least 1')
   point_radius_min = int(input('Enter minimum point radius between 1 and 24: '))
   if point_radius_min < 1 or point_radius_min > 24:
      raise ValueError('The minimum radius must be between 1 and 24')  
   point_radius_max = int(input('Enter maximum point radius between 1 and 24: '))
   if point_radius_max < 1 or point_radius_max > 24:
      raise ValueError('The maximum radius must be at most 24')
   if point_radius_max < point_radius_min:
      raise ValueError('The maximum radius must be at least the minimum radius')
      
   point_omega_min = float(input('Enter minimum angular velocity: '))
   point_omega_max = float(input('Enter maximum angular velocity: '))
   if point_omega_max < point_omega_min:
      raise ValueError('The maximum angular velocity must be at least the minimum angular velocity')
    
   if not os.path.exists('circular_motion'):
      os.mkdir('circular_motion')
   
   all_angles = np.zeros((num_videos,60), dtype=float)

   for video in range(num_videos):
      if not os.path.exists(f'circular_motion/{video}'):
         os.mkdir(f'circular_motion/{video}')
      point_radius = random.randint(point_radius_min, point_radius_max)
      point_omega = random.random() * (point_omega_max - point_omega_min) + point_omega_min
      cm = CircularMotion(128, 128, 40, point_radius, point_omega)
      
      for dt in range(60):
         frame, all_angles[video,dt] = cm.next_frame()
         frame.save(f'circular_motion/{video}/{dt}.png')
   
   # print(all_angles)
   np.save('all_angles', all_angles)

if __name__ == '__main__':
   main()
