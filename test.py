from model.resnet50 import predict

if __name__ == '__main__' :
    print( predict([
        '/tmp/image'
        #'/data/flower/keras-transfer-learning-for-oxford102/data/jpg/image_02727.jpg'
    ]))
