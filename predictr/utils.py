import pickle
from sklearn.ensemble import RandomForestClassifier

# define a Random forest classifier
clf = RandomForestClassifier()

# define the class encodings and reverse encodings
classes = {0: "class_0", 1: "class_1", 2: "class_2"}
r_classes = {y: x for x, y in classes.items()}


# function to load the model
def load_model():
    global clf
    clf = pickle.load(open("models/wine_rf.pkl", "rb"))


# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = clf.predict([x])[0]
    return classes[prediction]
