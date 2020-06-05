import sys
import logging
import torch
import utils
import numpy as np

import models.crnn as crnn
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable


class CRNNRecognizer:
    def __init__(self, alphabet, model_name):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            map_location = lambda storage, loc: storage.cuda()
        else:
            self.device = torch.device('cpu')
            map_location = 'cpu'

        logging.info('Using device [%s]' % self.device)

        fr = open(alphabet)
        dictionary = fr.readline()
        fr.close()
        alphabet = unicode(dictionary, 'utf8')
        self.__model = crnn.CRNN(32, 1, len(alphabet) + 1, 256).to(self.device)
        load_model = torch.load(model_name, map_location=map_location)
        model_dict = self.__model.state_dict()
        for k, v in model_dict.items():
            if k in load_model:
                model_dict[k] = load_model[k]
        self.__model.load_state_dict(model_dict)
        self.__converter = utils.strLabelConverter(alphabet)

    def recognize(self, img_url, text_im_array):
        result = []
        for crop_img in text_im_array:
            text_img = self.resize_normalize(crop_img).to(self.device)
            image = Variable(text_img.view(1, *text_img.size()))
            self.__model.eval()
            preds = self.__model(image)
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            preds_size = Variable(torch.IntTensor([preds.size(0)]))
            sim_pred = self.__converter.decode(preds.data, preds_size.data, raw=False)
            predictions = sim_pred.encode('utf8')
            result.append(predictions)
        return result

    def resize_normalize(self, img):
        (w, h) = img.size
        ratio = h / 32.0
        new_w = int(w / ratio)
        new_w = new_w if new_w % 4 == 0 else (new_w // 4 + 1) * 4
        if ratio != 1.0:
            img = img.resize((new_w, 32), Image.BILINEAR)
        transform = transforms.ToTensor()
        img = transform(img)
        img.sub_(0.5).div_(0.5)
        return img
