from django.shortcuts import render,redirect
from django import template
from django.contrib.sessions.models import Session
import string
from datetime import date
import datetime
from datetime import datetime
import cv2

import operator
from django.conf import settings

import datetime
from datetime import date
import tensorflow as tf
import os
from django.contrib import messages


from .models import *
from django.shortcuts import render,redirect

import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model


from tensorflow.keras.preprocessing import image
import os
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import User_Details
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import operator


def home(request):
    return render(request,'home.html',{})


def about(request):
    return render(request,"about.html",{})
def base(request):
    return render(request,"base.html",{})


def Admin_login(request):
    if request.method == 'POST':
        Username = request.POST['Username']
        password = request.POST['password']
        
        if Admin_Details.objects.filter(Username=Username, Password=password).exists():
                user = Admin_Details.objects.get(Username=Username, Password=password)
                request.session['type_id'] = 'Admin'
                request.session['username'] = Username
                request.session['login'] = 'Yes'
                return redirect('/')
        else:
            messages.info(request,'Invalid Credentials')
            return redirect('/Admin_login/')
    else:
        return render(request, 'Admin_login.html', {})


def brain(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        age = request.POST.get('age')
        contact = request.POST.get('contact')
        emailid = request.POST.get('email')
        username = request.POST.get('Username')
        image_file = request.FILES.get('Image1')

        # Check if username or email already exists
        if User_Details.objects.filter(Username=username).exists():
            messages.error(request, 'Username already taken.')
            return redirect('/brain/')
        elif User_Details.objects.filter(Email=emailid).exists():
            messages.error(request, 'Email already taken.')
            return redirect('/brain/')

        # Save image file to media folder
        if image_file:
            image_path = os.path.join(settings.MEDIA_ROOT, image_file.name)
            with open(image_path, 'wb') as f:
                for chunk in image_file.chunks():
                    f.write(chunk)

            # Load pre-trained model
            loaded_model = load_model('brain_model_new.h5', compile=False)

            # Load and preprocess the image
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            # Make prediction
            predictions = loaded_model.predict(x)[0]
            class_names = ['glioma tumor', 'meningioma tumor', 'no tumor', 'pituitary tumor']
            prediction = dict(zip(class_names, predictions))


            # Sort prediction by probability
            sorted_prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            predicted_class = sorted_prediction[0][0]
            print(predicted_class)

            # Determine symptoms and treatment based on predicted class
            symptoms, treatment = get_symptoms_and_treatment(predicted_class)

            # Save user details to the database
            user = User_Details(Name=name, Contact=contact, Age=age, Email=emailid,
                                Username=username, Image1=image_file, Symptoms=symptoms, Treatment=treatment,Class_detected = predicted_class )
            user.save()

            messages.success(request, 'Details submitted successfully! Results available in the admin page.')
            
            # Pass detected tumor type, symptoms, and treatment to the template
            context = {
                'predicted_class': predicted_class,
                'symptoms': symptoms,
                'treatment': treatment
            }
            print(context)
            return render(request, 'brain.html', {'context':context})
        else:
            messages.error(request, 'No image uploaded.')
            return redirect('/brain/')
    else:
        return render(request, 'brain.html', {})







def get_symptoms_and_treatment(predicted_class):
    if predicted_class == "glioma tumor":
        symptoms = "Headache, Nausea or vomiting, Confusion or a decline in brain function, Memory loss, Personality changes or irritability."
        treatment = "Chemotherapy drugs can be taken in pill form (orally) or injected into a vein (intravenously)."
    elif predicted_class == "meningioma tumor":
        symptoms = "Hearing loss or ringing in the ears, Memory loss, Loss of smell, Seizures."
        treatment = "The first treatment for a malignant meningioma is surgery, if possible. The goal of surgery is to obtain tissue to determine the tumor type and to remove as much tumor as possible without causing more symptoms for the person."
    elif predicted_class == "no tumor":
        symptoms = "No problems detected."
        treatment = "No problems detected."
    elif predicted_class == "pituitary tumor":
        symptoms = "Vision problems, Unexplained tiredness, Mood changes, Irritability, Unexplained changes in menstrual cycles, Erectile dysfunction."
        treatment = "Surgery, Radiation therapy, Medications, Replacement of pituitary hormones."
    elif predicted_class == "MildDementia":
        symptoms = "Forgetfulness, Trouble with problem-solving, Difficulty completing familiar tasks, Confusion with time or place."
        treatment = "Medications, Cognitive therapy, Lifestyle changes."
    elif predicted_class == "ModerateDementia":
        symptoms = "Worsening memory, Difficulty recognizing family and friends, Increased confusion, Difficulty speaking, Anxiety or aggression."
        treatment = "Medications, Supportive therapies, Supervision and assistance."
    elif predicted_class == "NonDementia":
        symptoms = "No significant cognitive impairment observed."
        treatment = "No specific treatment required."
    elif predicted_class == "VeryMildDementia":
        symptoms = "Subtle cognitive decline, May not be noticeable to others."
        treatment = "Lifestyle changes, Regular monitoring."
    else:
        # Default values if the predicted class is unknown
        symptoms = "Symptoms not available."
        treatment = "Treatment not available."
    
    return symptoms, treatment



def Alzhimers(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        age = request.POST.get('age')
        contact = request.POST.get('contact')
        emailid = request.POST.get('email')
        username = request.POST.get('Username')
        image_file = request.FILES.get('Image1')

        # Check if username or email already exists
        if User_Details.objects.filter(Username=username).exists():
            messages.error(request, 'Username already taken.')
            return redirect('/Alzhimers/')
        elif User_Details.objects.filter(Email=emailid).exists():
            messages.error(request, 'Email already taken.')
            return redirect('/Alzhimers/')

        # Save image file to media folder
        if image_file:
            image_path = os.path.join(settings.MEDIA_ROOT, image_file.name)
            with open(image_path, 'wb') as f:
                for chunk in image_file.chunks():
                    f.write(chunk)

            # Load pre-trained model
            loaded_model = load_model('alz_model_new.h5', compile=False)

            # Load and preprocess the image
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            # Make prediction
            predictions = loaded_model.predict(x)[0]
            class_names = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']
            prediction = dict(zip(class_names, predictions))


            # Sort prediction by probability
            sorted_prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            predicted_class = sorted_prediction[0][0]
            print(predicted_class)

            # Determine symptoms and treatment based on predicted class
            symptoms, treatment = get_symptoms_and_treatment(predicted_class)

            # Save user details to the database
            user = User_Details(Name=name, Contact=contact, Age=age, Email=emailid,
                                Username=username, Image1=image_file, Symptoms=symptoms, Treatment=treatment,Class_detected = predicted_class )
            user.save()

            messages.success(request, 'Details submitted successfully! Results available in the admin page.')
            
            # Pass detected tumor type, symptoms, and treatment to the template
            context = {
                'predicted_class': predicted_class,
                'symptoms': symptoms,
                'treatment': treatment
            }
            print(context)
            return render(request, 'Alzhimers.html', {'context':context})
        else:
            messages.error(request, 'No image uploaded.')
            return redirect('/Alzhimers/')
    else:
        return render(request, 'Alzhimers.html', {})





def logout(request):
    Session.objects.all().delete()
    return redirect('/')






def View_Users(request):
    if request.method == 'POST':
        return redirect('/View_Users/')
    else:
        sty = User_Details.objects.all()
        return render(request, 'View_Users.html', {'sty':sty})

