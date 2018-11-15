import mxnet as mx
import gluoncv

# you can change it to your image filename
filename = 'dataset/frame_616_patch_1.jpg'
# you may modify it to switch to another model. The name is case-insensitive
model_name = 'yolo3_darknet53_voc'
# download and load the pre-trained model
net = gluoncv.model_zoo.get_model(model_name, pretrained=True)
# load image
img = mx.image.imread(filename)
# apply default data preprocessing
transformed_img = gluoncv.data.transforms.presets.imagenet.transform_eval(img)
# run forward pass to obtain the predicted score for each class
pred = net(transformed_img)
# map predicted values to probability by softmax
prob = mx.nd.softmax(pred)[0].asnumpy()
# find the 5 class indices with the highest score
ind = mx.nd.topk(pred, k=5)[0].astype('int').asnumpy().tolist()
# print the class name and predicted probability
print('The input picture is classified to be')
for i in range(5):
    print('- [%s], with probability %.3f.'%(net.classes[ind[i]], prob[ind[i]]))