import os
import numpy as np
import tensorflow as tf
from PIL import Image

from models.lpips_tensorflow import learned_perceptual_metric_model, perceptual_loss


def load_image(fn):
    image = Image.open(fn).convert('RGB')
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)

    image = tf.constant(image, dtype=tf.dtypes.float32)
    return image


image_size = 224
model_dir = './models'
weights = 'vggface'
if weights == 'imagenet':
    vgg_ckpt_fn = os.path.join(model_dir, 'vgg', 'exported')
elif weights == 'vggface':
    vgg_ckpt_fn = os.path.join('keras_vggface', 'rcmalli_vggface_tf_notop_vgg16.h5')
lin_ckpt_fn = os.path.join(model_dir, 'lin', 'exported')
lpips = learned_perceptual_metric_model(image_size, weights, vgg_ckpt_fn, lin_ckpt_fn)
pl = perceptual_loss(image_size)

# official pytorch model value:
# Distance: ex_ref.png <-> ex_p0.png = 0.569
# Distance: ex_ref.png <-> ex_p1.png = 0.422
image_fn1 = './imgs/bns/real/img1.png'
image_fn2 = './imgs/bns/gen/img1.png'
# image_fn2 = './imgs/ex_p1.png'

image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
lpips01 = lpips([image1, image2])
print('LPIPS: {:.3f}'.format(lpips01))
pl01 = pl([image1, image2])
print('Perceptual loss: {:.3f}'.format(pl01))