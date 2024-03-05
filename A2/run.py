from trainMVShapeClassifier import trainMVShapeClassifier
from testMVImageClassifier import testMVImageClassifier
import pickle as p
import torch

train_path = '/Users/abhim/Downloads/python_starter_code/dataset/train'
test_path = '/Users/abhim/Downloads/python_starter_code/dataset/test'

# TRAIN
model, info = trainMVShapeClassifier(train_path, cuda=False, verbose=True)
#
# # TO SAVE TIME for just testing code, uncomment the following 2 lines to load your pre-trained model
# model = torch.load('model/model_epoch_19.pth', map_location=lambda storage, location: storage)["model"]
# info = p.load( open( "info.p", "rb" ) )

# TEST
#testMVImageClassifier(test_path, model, info, pooling='mean', cuda=False, verbose=True)
testMVImageClassifier(test_path, model, info, pooling='max', cuda=False, verbose=True)