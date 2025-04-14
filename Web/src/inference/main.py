"""
Temporary code for demo. Reimplement later on !
Ref: https://github.com/craston/MARS?tab=readme-ov-file#models
     https://github.com/seominseok0429/Real-world-Anomaly-Detection-in-Surveillance-Videos-pytorch/blob/main/vis.py
"""
import os
import time
import torch
import glob

from model import generate_model
from opts import parse_opts
import torch.nn.functional as F

import numpy as np
from PIL import Image

try:
    import accimage
except ImportError:
    accimage = None

import cv2
from matplotlib import pyplot as plt
plt.switch_backend("tkagg")

from learner import Learner

class ToTensor(object):

    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass





def main() -> None:
    pretrain = '../weights/RGB_Kinetics_16f.pth'

    opts = parse_opts()
    opts.pretrain_path = pretrain
    opts.arch = "resnext-101"
    opts.sample_size = 240
    opts.sample_duration = 15
    opts.output_layers = ["avgpool"]

    save_path = "./out"
    classifier = Learner().cuda()  # classifier
    classifier.eval()

    model = generate_model(opts)
    # model.load_state_dict(torch.load(pretrain)['state_dict'])
    model = model.to("cuda")

    images = glob.glob("./images/*")
    images.sort()
    images = images[:100]

    segment = len(images) // 16

    y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    x_value = [i for i in range(segment)]

    inputs = torch.Tensor(1, 3, 16, 240, 240)
    x_time = [jj for jj in range(len(images))]

    for num, i in enumerate(images):
        if num < 16:
            inputs[:, :, num, :, :] = ToTensor(1)(Image.open(i)).permute(0, -1, 1)[..., :240]
            cv_img = cv2.imread(i)
            h, w, _ = cv_img.shape
            cv_img = cv2.putText(cv_img, 'FPS : 0.0, Pred : 0.0', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,                                 (255, 200, 240), 2)
        else:
            inputs[:, :, :15, :, :] = inputs[:, :, 1:, :, :]
            inputs[:, :, 15, :, :] = ToTensor(1)(Image.open(i)).permute(0, -1, 1)[..., :240]
            inputs = inputs.cuda()
            start = time.time()
            output, feature = model(inputs)
            feature = F.normalize(feature, p=2, dim=1)

            out = classifier(feature)
            y_pred.append(out.item())
            end = time.time()
            FPS = str(1 / (end - start))[:5]
            out_str = str(out.item())[:5]

            print(len(x_value) / len(y_pred), num)

            cv_img = cv2.imread(i)
            cv_img = cv2.putText(cv_img, 'FPS :' + FPS + ' Pred :' + out_str, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                 (255, 200, 240), 2)
            if out.item() > 0.4:
                cv_img = cv2.rectangle(cv_img, (0, 0), (w, h), (0, 0, 255), 3)

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        path = save_path + '/' + os.path.basename(i)
        cv2.imwrite(path, cv_img)

        if num == 100:
            break
    os.system('ffmpeg -y -pattern_type glob -i "%s" "%s"' % (save_path + '/*.png', save_path + '.mp4'))
    plt.plot(x_time, y_pred)
    plt.savefig(save_path + '.png', dpi=300)
    plt.cla()
    return None


if __name__ == '__main__':
    main()