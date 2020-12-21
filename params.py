def get_embed_params(embed_type, response_data):

    if embed_type == 'CounterVector':

        embed_params = {
            "min_df": int(response_data[0]),
            "max_df": float(response_data[1]),
            "max_features": None if response_data[2] == 'None' else int(response_data[2]),
        }
        return embed_params

    elif embed_type == 'TF-IDF':

        embed_params = {
            "min_df": int(response_data[0]),
            "max_df": float(response_data[1]),
            "max_features": None if response_data[2] == 'None' else int(response_data[2]),
        }
        return embed_params

    elif embed_type == 'Doc2Vec':

        embed_params = {
            "dm": int(response_data[0]),
            "vector_size": int(response_data[1]),
            "window": int(response_data[2]),
            "alpha": float(response_data[3]),
            "epochs": int(response_data[4]),
            "negative": int(response_data[5]),
        }
        return embed_params

    elif embed_type == 'user_defined_embedding':

        embed_params = {}
        return embed_params


def get_machine_params(machine_type, response_data):

    if machine_type == 'Logistic':
        for i in range(len(response_data)):
            response_data[i] = response_data[i].split(" ")

        machine_params = {
            "penalty": [str(i) for i in response_data[0]],
            "C": [float(i) for i in response_data[1]],
            "random_state": [int(i) for i in response_data[2]],
            "max_iter": [int(i) for i in response_data[3]],
            "l1_ratio": [None] if response_data[4][0] == 'None' else [float(i) for i in response_data[4]],
        }
        return machine_params

    # 8개 파라미터
    elif machine_type == 'SVM':
        for i in range(len(response_data)):
            response_data[i] = response_data[i].split(" ")

        machine_params = {
            "loss": [str(i) for i in response_data[0]],
            "penalty": [str(i) for i in response_data[1]],
            "alpha": [float(i) for i in response_data[2]],
            "l1_ratio": [float(i) for i in response_data[3]],
            "max_iter": [int(i) for i in response_data[4]],
            "random_state": [int(i) for i in response_data[5]],
            "learning_rate": [str(i) for i in response_data[6]],
            "eta0": [float(i) for i in response_data[7]],
        }
        return machine_params

    # 3개 파라미터
    elif machine_type == 'RandomForest':
        for i in range(len(response_data)):
            response_data[i] = response_data[i].split(" ")

        machine_params = {
            "n_estimators": [int(i) for i in response_data[0]],
            "max_depth": [None] if response_data[1][0] == 'None' else [int(i) for i in response_data[1]],
            "random_state": [int(i) for i in response_data[2]],
        }
        return machine_params

    # 8개 파라미터
    elif machine_type == 'FNN':
        for i in range(len(response_data)):
            response_data[i] = response_data[i].split(" ")

        machine_params = {
            "input_layer_units": [int(i) for i in response_data[0]],
            "hidden_layer_units": [int(i) for i in response_data[1]],
            "hidden_layer_count": [int(i) for i in response_data[2]],
            "input_layer_activation": [str(i) for i in response_data[3]],
            "hidden_layer_activation": [str(i) for i in response_data[4]],
            "output_layer_activation": [str(i) for i in response_data[5]],
            "optimizer": [str(i) for i in response_data[6]],
            "epochs": [int(i) for i in response_data[7]],
            "batch_size": [int(i) for i in response_data[8]],
        }
        return machine_params

    elif machine_type == 'user_defined_machine_learning':

        machine_params = {}
        return machine_params

