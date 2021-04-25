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


def cov_pos_msg(name, prob_ar):
    return f'Hello {name}, I am sorry to tell you this but your X-Ray has been classified as covid positive. You would need to get tested from a covid facility to get ensured about the results where you would be informed about the further steps.\
            \
            \
            \n\nYour X-Rays were classified with following probablities:\
            \nNormal: {prob_ar[0]}\
            \nViral: {prob_ar[1]}\
            \nCovid: {prob_ar[2]}\
            \
            \n\nMentioned below are some important ways to help keep you and your family safe.\
            \n1. Stay home except to get medical care.\
            \n2. Separate yourself from other people.\
            \n3. Monitor your symptoms.\
            \n4. Call ahead before visiting your doctor.\
            \n5. Cover your coughs and sneezes.\
            \n6. Clean your hands often.\
            \n7. Avoid sharing personal household items.\
            \n8. Clean all "high-touch" surfaces everyday.\
            \
            \n\nPlease go through some of links to get more important information.\
            \nCDC Guidelines on \"What to Do If You Are Sick\" - https://www.cdc.gov/coronavirus/2019-ncov/if-you-are-sick/steps-when-sick.html\
            \nCDC Guidelines on \"Caring for Someone Sick at Home\" - https://www.cdc.gov/coronavirus/2019-ncov/if-you-are-sick/care-for-someone.html\
            \
            \n\nStay safe and healthy...!\
            \
            \
            \n\n\nRegards,\
            \nMiraaj, Neeraj, Paras\
            \nGroup 4'


def cov_neg_msg(name, prob_ar):
    return f'Hello {name}, You are covid negative, but being negative doesn\'t ensures that you will stay negative.\
            \
            \
            \n\nYour X-Rays were classified with following probablities:\
            \nNormal: {prob_ar[0]}\
            \nViral: {prob_ar[1]}\
            \nCovid: {prob_ar[2]}\
            \
            \n\nMentioned below are some important ways to help keep you and your family safe.\
            \n1. Wear a mask that covers your nose and mouth to help protect yourself and others.\
            \n2. Stay 6 feet apart from others who don’t live with you.\
            \n3. Get a COVID-19 vaccine when it is available to you.\
            \n4. Avoid crowds and poorly ventilated indoor spaces.\
            \n5. Wash your hands often with soap and water. Use hand sanitizer if soap and water aren’t available.\
            \
            \n\nPlease go through some of links to get more important information.\
            \nCDC Guidelines on \"What to Do If You Are Sick\" - https://www.cdc.gov/coronavirus/2019-ncov/if-you-are-sick/steps-when-sick.html\
            \nCDC Guidelines on \"Caring for Someone Sick at Home\" - https://www.cdc.gov/coronavirus/2019-ncov/if-you-are-sick/care-for-someone.html\
            \
            \n\nStay safe and healthy...!\
            \
            \
            \n\n\nRegards,\
            \nMiraaj, Neeraj, Paras\
            \nGroup 4'


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
        name = request.POST.get('name')
        file = request.FILES['imageFile']
        file_name = default_storage.save(file.name, file)
        file_url = default_storage.path(file_name)
        image_path = file_url

        probabilities, predicted_class_index, predicted_class_name, prob_ar = predict_image_class(
            image_path)

        if(predicted_class_index == 2):
            positive = True
            message = cov_pos_msg(name, prob_ar)
            send_mail('You are Positive!',
                      message,
                      'Team Covid', [email],
                      fail_silently=False,
                      html_message=None)
        else:
            positive = False
            message = cov_neg_msg(name, prob_ar)

            send_mail('You are negative!',
                      message,
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
