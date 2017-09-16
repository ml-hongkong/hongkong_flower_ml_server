from os.path import join, basename
from glob import glob
from model.resnet50 import model, load_imgs, remap

from config import BASE_DIR

data_path = join(BASE_DIR, 'data/sorted')

for dir in ['valid', 'test', 'train']:
    total = 0
    correct = 0
    for path in glob(join(data_path, dir, '*')):
        class_ix = basename(path)
        print('predicting class', class_ix, end='\r')

        imgs = glob(join(path, '*'))
        res = model.predict(load_imgs(imgs))
        for x in res.argsort(1)[:, -1]:
            total += 1
            if remap[x] == class_ix:
                correct += 1

    print(f'{type}: {correct / total}')
