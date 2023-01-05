# # lancer un training, sotcker dans un fichier temporaire, faire un predict dessus


# from predict.predict.run import TextPredictionModel
# import tempfile
# import os
# import json
# import unittest
# import tensorflow as tf
# from tensorflow.keras.models import load_model


# class TestPredict(unittest.TestCase):
#     def test_from_artefacts(self):
#         # check that the 3 parameters model, params and labels_index are correctly loaded and that it returns a TextPredictionModel object
#         # we create a temporary file to store artefacts
#         with tempfile.TemporaryDirectory() as tmpdirname:
            
#             # Creating a dummy keras model
#             model_fake = tf.keras.Sequential()
#             model_fake.add(tf.keras.layers.Dense(1, input_shape=(10,)))
#             model_fake.add(tf.keras.layers.Dense(2, activation="softmax"))
#             model_fake.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#             # save the model in the temporary directory
#             model_fake.save(os.path.join(tmpdirname, "model_fake.h5"))

#             # create a fake object TextPredictionModel with fake model, params and labels_index
#             object_fake = TextPredictionModel(model_fake, {"max_len": 10}, {"label1": 0, "label2": 1})

#             # # we create fake model
#             # model_fake = object_fake.model
#             # with open(os.path.join(tmpdirname, "model_fake.json"), "w") as f:
#             #     json.dump(model_fake, f)

#             # we create fake params
#             params_fake = object_fake.params
#             with open(os.path.join(tmpdirname, "params_fake.json"), "w") as f:
#                 json.dump(params_fake, f)
#             # we create fake labels_index
#             labels_index_fake = object_fake.labels_index
#             with open(os.path.join(tmpdirname, "labels_index_fake.json"), "w") as f:
#                 json.dump(labels_index_fake, f)

#             # we load the model from the artefacts
#             model_artefacts = load_model(os.path.join(tmpdirname, "model_fake.h5"))          
#             # we load the params from the artefacts
#             with open(os.path.join(tmpdirname, "params_fake.json"), "r") as f:
#                 params_artefacts = json.load(f)
#             # we load the labels_index from the artefacts
#             with open(os.path.join(tmpdirname, "labels_index_fake.json"), "r") as f:
#                 labels_index_artefacts = json.load(f)


#             # we check that the model is correctly loaded with debugging.assert_equal
#             # tf.debugging.assert_equal(model_artefacts, model_fake)

#             # we check that the model is correctly loaded
#             self.assertEqual(model_artefacts, model_fake)
#             # we check that the params are correctly loaded
#             self.assertEqual(params_artefacts, params_fake)
#             # we check that the labels_index are correctly loaded
#             self.assertEqual(labels_index_artefacts, labels_index_fake)



#     def test_predict(self):
#         # check that the predict method returns the correct top_k tags for a list of texts
#         # create a fake text_list, list of text to predict with questions
#         text_list = ["How to create a list in Python?", "How to remove an element from a table in mysql?"]

#         # create a fake prediction of the 5 tags
#         predictions = ["python", "mysql", "list", "remove", "table"]

#             # we check that the predict method returns the correct top_k tags for a list of texts
#             text_list = ["hello world"]
#             top_k = 5
#             predictions = object_fake.predict(text_list, top_k)
#             self.assertEqual(predictions, predictions)

import unittest
from unittest.mock import MagicMock
import tempfile

import pandas as pd

from predict.predict import run
from train.train import run as train_run
from preprocessing.preprocessing import utils


def load_dataset_mock():
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags
    })

class TestPredict(unittest.TestCase):

    # use the function defined on test_model_train as a mock for utils.LocalTextCategorizationDataset.load_dataset
    utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())
    dataset = utils.LocalTextCategorizationDataset.load_dataset

    def test_predict(self):
        # TODO: CODE HERE
        # create a dictionary params for train conf
        params = {
            "batch_size": 1,
            "epochs": 1,
            "dense_dim": 64,
            "min_samples_per_label": 4,
            "verbose": 1
        }

        # we create a temporary file to store artefacts
        with tempfile.TemporaryDirectory() as model_dir:
            # run a training
            accuracy, _ = train_run.train("fake_path", params, model_dir, False)

            # instance a TextPredictModel class
            textpredictmodel = run.TextPredictionModel.from_artefacts(model_dir)

            # run a prediction
            predictions_obtained = textpredictmodel.predict({"0": "python", "1": "php"})
            # predictions_obtained = textpredictmodel.predict(['toto'], 0)



        # TODO: CODE HERE
        # assert that predictions obtained are equals to expected ones
        self.assertEqual(accuracy, 0.5)

