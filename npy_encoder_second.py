# -*- coding: utf-8 -*-
"""npy_encoder.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CwEuXIBy2l91-H0H5BiXS66Q78p5b77C
"""

import PIL.Image
import argparse
import os
import sys
import bz2
import pickle
from tqdm import tqdm
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel, load_images
from keras.models import load_model
from keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl

def split_to_batches(l, n):
  for i in range(0, len(l), n):
    yield l[i:i + n]

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

def unpack_bz2(src_path):
  data = bz2.BZ2File(src_path).read()
  dst_path = src_path[:-4]
  with open(dst_path, 'wb') as fp:
    fp.write(data)
  return dst_path

def align_images(raw_dir, aligned_dir):
  landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2', LANDMARKS_MODEL_URL, cache_subdir='temp'))
  RAW_IMAGES_DIR = raw_dir
  ALIGNED_IMAGES_DIR = aligned_dir
  
  landmarks_detector = LandmarksDetector(landmarks_model_path)
  for img_name in os.listdir(RAW_IMAGES_DIR):
    raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
    for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
      face_img_name = '%s_%d.png'%(os.path.splitext(img_name)[0], i)
      aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
      
      image_align(raw_img_path, aligned_face_path, face_landmarks)

def main():
  parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
  parser.add_argument('name', help='Name of a combined image')
  parser.add_argument('raw_dir', help='Directory with a raw image for encoding')
  parser.add_argument('aligned_dir', help='Directory with a aligned image')
  parser.add_argument('generated_images_dir', help='Directory for storing generated images')
  parser.add_argument('dlatent_dir', help='Directory for storing dlatent representations')
  parser.add_argument('--data_dir', default='data', help='Directory for storing optional models')
  parser.add_argument('--mask_dir', default='masks', help='Directory for storing optional masks')
  parser.add_argument('--model_res', default=1024, help='The dimension of images in the StyleGAN model', type=int)
  parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)

  #Perceptual model params
  parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
  parser.add_argument('--lr', default=0.01, help='Learning rate for perceptual model', type=float)
  parser.add_argument('--iterations', default=500, help='Number of optimization steps for each batch', type=int)

  parser.add_argument('--use_lpips_loss', default=100, help='Use LPIPS perceptual loss; 0 to disable, > 0 to scale.', type=float)

  #Generator params
  parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=bool)
  parser.add_argument('--tile_dlatents', default=False, help='Tile dlatents to use a single vector at each scale', type=bool)
  parser.add_argument('--clipping_threshold', default=2.0, help='Stochastic clipping of gradient values outside of this threshold', type=float)
  args, other_args = parser.parse_known_args()

  #encoder_main
  os.makedirs(args.raw_dir, exist_ok=True)
  src_dir=args.raw_dir+args.name
  img = PIL.Image.open(src_dir)
  wpercent = (256/float(img.size[0]))
  hsize = int((float(img.size[1])*float(wpercent)))
  img = img.resize((256, hsize), PIL.Image.LANCZOS)
  #align_images
  os.makedirs(args.aligned_dir, exist_ok=True)
  align_images(args.raw_dir, args.aligned_dir)
  #encode_images
  ref_images = [os.path.join(args.aligned_dir, x) for x in os.listdir(args.aligned_dir)]
  ref_images = list(filter(os.path.isfile, ref_images))
  
  if len(ref_images) == 0:
    raise Exception('%s is empty' % args.aligned_dir)

  os.makedirs(args.data_dir, exist_ok=True)
  os.makedirs(args.mask_dir, exist_ok=True)
  os.makedirs(args.generated_images_dir, exist_ok=True)
  os.makedirs(args.dlatent_dir, exist_ok = True)
  
  tflib.init_tf()
  with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)

  perc_model = None
  if (args.use_lpips_loss > 0.00000001):
      with dnnlib.util.open_url('https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2', cache_dir=config.cache_dir) as f:
          perc_model =  pickle.load(f)
  generator = Generator(Gs_network, args.batch_size, clipping_threshold=args.clipping_threshold, tiled_dlatent=args.tile_dlatents, model_res=args.model_res, randomize_noise=args.randomize_noise)
  perceptual_model = PerceptualModel(args, perc_model=perc_model, batch_size=args.batch_size)
  perceptual_model.build_perceptual_model(generator)#.generated_image

  for images_batch in tqdm(split_to_batches(ref_images, args.batch_size), total=len(ref_images)//args.batch_size):
    names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
    perceptual_model.set_reference_images(images_batch)
    op = perceptual_model.optimize(generator.dlatent_variable, iterations=args.iterations)
    pbar = tqdm(op, leave=False, total=args.iterations)

    for loss_dict in pbar:
        pbar.set_description(" ".join(names) + "Loss: %.2f" %loss)
    print(" ".join(names), " Loss : ", loss)

    # Generate images from found dlatents and save them
    generated_images = generator.generate_images()
    generated_dlatents = generator.get_dlatents()
    for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
      img = PIL.Image.fromarray(img_array, 'RGB')
      img.save(os.path.join(args.generated_images_dir, f'{img_name}.png'), 'PNG')
      np.save(os.path.join(args.dlatent_dir, f'{img_name}.npy'), dlatent)
      
    generator.reset_dlatents()

if __name__ == "__main__":
  main()
