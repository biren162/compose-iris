import os
import pickle
from sklearn.ensemble import RandomForestClassifier

# define the class encodings and reverse encodings
classes = {0: "class_0", 1: "class_1", 2: "class_2"}
r_classes = {y: x for x, y in classes.items()}

# function to process data and return it in correct format
def process_data(data):
    processed = [
        {
    "p1": d.p1,
    "p2": d.p2,
    "p3": d.p3,
    "p4": d.p4,
    "p5": d.p5,
    "p6": d.p6,
    "p7": d.p7,
    "p8": d.p8,
    "p9": d.p9,
    "p10": d.p10,
    "p11": d.p11,
    "p12": d.p12,
    "wine_class": d.wine_class
        }
        for d in data
    ]

    return processed
