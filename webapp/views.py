from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import default_storage
from django.core.mail import send_mail


# classification imports
import torch
import torchvision
from PIL import Image
import numpy as np

import os
from pathlib import Path

# Create your views here.

class_names = ['normal', 'viral', 'covid']

resnet18 = torchvision.models.resnet18(pretrained=True)

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.fc = torch.nn.Linear(in_features=512, out_features=3)

# resnet18.load_state_dict(torch.load('covid_classifier.pt'))
BASE_DIR = Path(__file__).resolve().parent.parent

resnet18.load_state_dict(torch.load(
    os.path.join(BASE_DIR, 'webapp\\covid_classifier.pt')))
resnet18.eval()


def predict_image_class(image_path):
    image = Image.open(image_path).convert('RGB')
    image = test_transform(image)
    image = image.unsqueeze(0)
    output = resnet18(image)[0]
    probabilities = torch.nn.Softmax(dim=0)(output)
    probabilities = probabilities.cpu().detach().numpy()
    predicted_class_index = np.argmax(probabilities)

    prob_ar = probabilities.tolist()
    formatted_prob_ar = ['%.5f' % elem for elem in prob_ar]

    predicted_class_name = class_names[predicted_class_index]
    return probabilities, predicted_class_index, predicted_class_name, formatted_prob_ar


def index(request):
    if(request.method == "POST"):
        email = request.POST.get('email')
        file = request.FILES['imageFile']
        file_name = default_storage.save(file.name, file)
        file_url = default_storage.path(file_name)
        image_path = file_url

        probabilities, predicted_class_index, predicted_class_name, prob_ar = predict_image_class(
            image_path)

        if(predicted_class_index == 2):
            positive = True
            str = "You are positive"
            send_mail('You are Positive!',
                      str,
                      'Team Covid', [email],
                      fail_silently=False,
                      html_message=None)
        else:
            positive = False
            str = "You are negative"
            send_mail('You are negative!',
                      str,
                      'Team Covid', [email],
                      fail_silently=False,
                      html_message=None)

        image_src = "../../../media/" + file_name

        context = {
            'probabilities': prob_ar,
            'predicted_class_index': predicted_class_index,
            'predicted_class_name': predicted_class_name,
            'file_name': file_name,
            'file_url': file_url,
            'image_src': image_src,
            'positive': positive,
        }

        return render(request, 'webapp/main.html', {'context': context})

    elif request.method == 'GET':
        return render(request, 'webapp/main.html')

    return render(request, 'webapp/main.html')
