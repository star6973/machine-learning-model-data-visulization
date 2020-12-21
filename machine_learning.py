import model
import numpy as np
import tensorflow as tf
from joblib import dump, load


def pre_train_machine_learning(embedding_model_name, machine_type, X_train, X_test, y_train, y_test):

    if machine_type == 'Logistic':

        # load the machine_model from disk
        filename = 'C:/Users/battl/PycharmProjects/cse_project/project list/Machine Learning Classification Model Visualization Web Service/machine_model/' + embedding_model_name.lower() + '_logistic.pkl'
        log_clf = load(filename)

        train_y_pred = log_clf.predict(X_train)
        test_y_pred = log_clf.predict(X_test)

    elif machine_type == 'SVM':

        # load the machine_model from disk
        filename = 'C:/Users/battl/PycharmProjects/cse_project/project list/Machine Learning Classification Model Visualization Web Service/machine_model/' + embedding_model_name.lower() + '_svm.pkl'
        svm_clf = load(filename)

        train_y_pred = svm_clf.predict(X_train)
        test_y_pred = svm_clf.predict(X_test)

    elif machine_type == 'RandomForest':

        # load the machine_model from disk
        filename = 'C:/Users/battl/PycharmProjects/cse_project/project list/Machine Learning Classification Model Visualization Web Service/machine_model/' + embedding_model_name.lower() + '_randomforest.pkl'
        rnd_clf = load(filename)

        train_y_pred = rnd_clf.predict(X_train)
        test_y_pred = rnd_clf.predict(X_test)

    elif machine_type == 'FNN':

        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        train_label = le.fit_transform(y_train)

        # load the machine_model from disk
        fnn_clf = tf.keras.models.load_model(
            'C:/Users/battl/PycharmProjects/cse_project/project list/Machine Learning Classification Model Visualization Web Service/machine_model/' + embedding_model_name.lower() + '_fnn.h5')

        train_prediction = fnn_clf.predict(X_train.toarray())
        test_prediction = fnn_clf.predict(X_test.toarray())

        tr_y_pred = []
        for i in range(len(train_prediction)):
            tr_y_pred.append(np.argmax(train_prediction[i]))

        te_y_pred = []
        for i in range(len(test_prediction)):
            te_y_pred.append(np.argmax(test_prediction[i]))

        train_y_pred = le.inverse_transform(tr_y_pred)
        test_y_pred = le.inverse_transform(te_y_pred)

    elif machine_type == 'user_defined_machine_learning':
        pass

    return train_y_pred, test_y_pred


def machine_learning(embedding_model_name, machine_type, X_train, X_test, y_train, y_test, params):

    target_names = list(set(y_train))

    if machine_type == 'Logistic':

        log_clf = model.myLogisticRegression(params)
        log_clf.fit(X_train, y_train)

        # save the machine_model to disk
        filename = 'C:/Users/battl/PycharmProjects/cse_project/project list/Machine Learning Classification Model Visualization Web Service/machine_model/' + embedding_model_name.lower() + '_logistic.pkl'
        dump(log_clf, filename)

        train_y_pred = log_clf.predict(X_train)
        test_y_pred = log_clf.predict(X_test)

    elif machine_type == 'SVM':

        svm_clf = model.mySVM(params)
        svm_clf.fit(X_train, y_train)

        # save the machine_model to disk
        filename = 'C:/Users/battl/PycharmProjects/cse_project/project list/Machine Learning Classification Model Visualization Web Service/achine_model/' + embedding_model_name.lower() + '_svm.pkl'
        dump(svm_clf, filename)

        train_y_pred = svm_clf.predict(X_train)
        test_y_pred = svm_clf.predict(X_test)

    elif machine_type == 'RandomForest':

        rnd_clf = model.myRandomForestClassifier(params)
        rnd_clf.fit(X_train, y_train)

        # save the machine_model to disk
        filename = 'C:/Users/battl/PycharmProjects/cse_project/project list/Machine Learning Classification Model Visualization Web Service/machine_model/' + embedding_model_name.lower() + '_randomforest.pkl'
        dump(rnd_clf, filename)

        train_y_pred = rnd_clf.predict(X_train)
        test_y_pred = rnd_clf.predict(X_test)

    elif machine_type == 'FNN':

        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        train_label = le.fit_transform(y_train)

        input_layer_units = int(params['input_layer_units'][0])
        hidden_layer_units = int(params['hidden_layer_units'][0])
        output_layer_units = len(target_names)

        hidden_layer_count = int(params['hidden_layer_count'][0])

        input_layer_activation = params['input_layer_activation'][0]
        hidden_layer_activation = params['hidden_layer_activation'][0]
        output_layer_activation = params['output_layer_activation'][0]

        optimizer = params['optimizer'][0]
        epochs = int(params['epochs'][0])
        batch_size = int(params['batch_size'][0])

        fnn_clf = tf.keras.Sequential()
        fnn_clf.add(tf.keras.layers.Dense(input_layer_units, activation=input_layer_activation, input_shape=(len(X_train.toarray()[0]), )))

        for i in range(hidden_layer_count):
            fnn_clf.add(tf.keras.layers.Dense(hidden_layer_units, activation=hidden_layer_activation))

        fnn_clf.add(tf.keras.layers.Dense(output_layer_units, activation=output_layer_activation))

        fnn_clf.compile(optimizer=optimizer,
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

        fnn_clf.fit(X_train.toarray(), train_label, epochs=epochs, batch_size=batch_size)

        # save the machine_model to disk
        fnn_clf.save(
            'C:/Users/battl/PycharmProjects/cse_project/project list/Machine Learning Classification Model Visualization Web Service/machine_model/' + embedding_model_name.lower() + '_fnn.h5')

        train_prediction = fnn_clf.predict(X_train.toarray())
        test_prediction = fnn_clf.predict(X_test.toarray())

        tr_y_pred = []
        for i in range(len(train_prediction)):
            tr_y_pred.append(np.argmax(train_prediction[i]))

        te_y_pred = []
        for i in range(len(test_prediction)):
            te_y_pred.append(np.argmax(test_prediction[i]))

        train_y_pred = le.inverse_transform(tr_y_pred)
        test_y_pred = le.inverse_transform(te_y_pred)

    elif machine_type == 'user_defined_machine_learning':
        pass

    return train_y_pred, test_y_pred