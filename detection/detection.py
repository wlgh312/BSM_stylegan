import cv2
import numpy as np
import dlib
import face_recognition
import os
import argparse


MODEL_MEAN_VALUES = (78.42633776603, 87.7689143744, 114.895847746)

age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

comp=0

def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt','age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return age_net, gender_net

def video_detector(age_net, gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX

def compare(gender1, gender2, age1, age2, comp):
    if gender1==gender2 and age1==age2:
        comp=0
        print("Same! : {}".format(comp))
    else:
        comp=1
        print("Different! : {}".format(comp))
    return comp
def main():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    parser.add_argument('dst_image', help='Name of a original image')
    parser.add_argument('mix_image', help='Name of a mixing result image')
    args, other_args = parser.parse_known_args()

    known_image = face_recognition.load_image_file(args.dst_image)
    unknown_image = face_recognition.load_image_file(args.mix_image)
    age_net, gender_net = load_caffe_models()
    video_detector(age_net, gender_net)
    known_image = cv2.resize(known_image, dsize=(0, 0), fx=1, fy=1, interpolation=cv2.INTER_AREA)
    unknown_image = cv2.resize(unknown_image, dsize=(0, 0), fx=1, fy=1, interpolation=cv2.INTER_AREA)
    biden_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    known_gray = cv2.cvtColor(known_image, cv2.COLOR_BGR2GRAY)
    unknown_gray = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2GRAY)
    known_faces = face_cascade.detectMultiScale(known_gray, 1.1, 5)
    unknown_faces = face_cascade.detectMultiScale(unknown_gray, 1.1, 5)
    for (x, y, w, h) in known_faces:
        #Get Face
        cv2.rectangle(known_image, (x, y), (x+2, y+h), (255, 255, 0), 2)
        face_img = known_image[y:y+h, h:h+w].copy()
        blob1 = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        #Predict Gender
        gender_net.setInput(blob1)
        gender_preds = gender_net.forward()
        gender1 = gender_list[gender_preds[0].argmax()]
        print("Original Gender : " + gender1)
        #Predict Age
        age_net.setInput(blob1)
        age_preds = age_net.forward()
        age1 = age_list[age_preds[0].argmax()]
        print("Original Age Range : " + age1)

    for (x, y, w, h) in unknown_faces:
        #Get Face
        cv2.rectangle(unknown_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        face_img = unknown_image[y:y + h, h:h + w].copy()
        blob2 = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        #Predict Gender
        gender_net.setInput(blob2)
        gender_preds = gender_net.forward()
        gender2 = gender_list[gender_preds[0].argmax()]
        print("Mixing Result Gender : " + gender2)
        #Predict Age
        age_net.setInput(blob2)
        age_preds = age_net.forward()
        age2 = age_list[age_preds[0].argmax()]
        print("Mixing Result Age Range : " + age2)
    comp=compare(gender1, gender2, age1, age2, comp)

if __name__ == "__main__":
    main()

