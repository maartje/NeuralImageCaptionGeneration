from torchvision import models
import torch

class ImageEncoder:

    def __init__(self, model_name, layer_name, encoder_models = {}):
        self.model_name = model_name
        self.layer_name = layer_name
        self.model = None
        self.layer = None
        self.encoder_models = {
            'resnet18': models.resnet18,
            'resnet152': models.resnet152,
            'vgg16': models.vgg16
        }
        self.encoder_models.update(encoder_models)


    def load_model(self):
        self.model = self.encoder_models[self.model_name](pretrained=True) 
        self.layer = self.model._modules.get(self.layer_name)
        self.model.eval()

    def encode(self, img):
        embedding = [] # stores the output for the given layer
        def copy_data(m, i, o):
            embedding.append(o.clone())
        h = self.layer.register_forward_hook(copy_data)
        with torch.no_grad():
            self.model(img)
        h.remove()    
        return embedding[0].squeeze()


