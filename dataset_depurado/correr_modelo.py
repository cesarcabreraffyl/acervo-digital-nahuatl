from kraken import blla
from kraken.lib import vgsl
from kraken import blla
from kraken import serialization
from PIL import Image

# can be any supported image format and mode
im = Image.open('foo.png')

model_path = 'dataset_representativo/model_27.mlmodel'
model = vgsl.TorchVGSLModel.load_model(model_path)

baseline_seg = blla.segment(im, model=model)

xml = serialization.serialize(baseline_seg,
                                   image_size=im.size,
                                   template='page')
with open('segmentation_output.xml', 'w') as fp:
        fp.write(xml)