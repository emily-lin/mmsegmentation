from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import mmcv

config_file = 'configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
checkpoint_file = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

model = init_segmentor(config_file, checkpoint_file, device = 'cuda:0')
img = 'demo/demo.png'
result = inference_segmentor(model, img)
model.show_result(img, result, out_file = 'result.jpg', opacity = 0.5)


# ======================
out_dir = '/tmp'
test_dir = '.../images/test'
for x in os.listdir(test_dir):
  img = os.path.join(test_dir, x)
  out_path = os.path.join(out_dir, x)
  result = inference_segmentor(model, img) # 0, 1.
  # Add the groundtruth label to the result. # 2
  # load the gt image
  model.show_result(img, result, out_file = out_path, opacity = 0.5)

