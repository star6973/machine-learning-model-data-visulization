import os
import json
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from embedding import embedding, pre_train_embedding
from machine_learning import machine_learning, pre_train_machine_learning
from dimension_reduction import dimension_reduction
from params import get_embed_params, get_machine_params
import pandas as pd

train_file_path = '/home/ubuntu/project2/train_list/'
test_file_path = '/home/ubuntu/project2/test_list/'
embed_model_path = '/home/ubuntu/project2/embedding_model/'
machine_model_path = '/home/ubuntu/project2/machine_model/'
path = '/home/ubuntu/project2/csv_files/'

app = Flask(__name__)

app.secret_key = "secret key"
app.config['train_file_path'] = train_file_path
app.config['test_file_path'] = test_file_path
app.config['embed_model_path'] = embed_model_path
app.config['machine_model_path'] = machine_model_path


@app.route("/", methods=["GET", "POST"])
@app.route("/introduction", methods=["GET", "POST"])
def page1():
    return render_template('introduction.html')


@app.route('/fileUpload', methods=["GET", "POST"])
def page2():

    train_file_list = os.listdir(train_file_path)
    test_file_list = os.listdir(test_file_path)
    embed_model_list = os.listdir(embed_model_path)
    machine_model_list = os.listdir(machine_model_path)

    if request.method == "POST":

        train_file = request.files["train_file"]
        test_file = request.files["test_file"]

        if train_file:
            f1 = secure_filename(train_file.filename)
            train_file.save(os.path.join(app.config['train_file_path'], f1))

        if test_file:
            f2 = secure_filename(test_file.filename)
            test_file.save(os.path.join(app.config['test_file_path'], f2))

        return render_template('fileUpload.html', filename1=f1, filename2=f2,
                               train_file_lis=train_file_list,
                               test_file_list=test_file_list,
                               embed_model_list=embed_model_list,
                               machine_model_list=machine_model_list)

    else:
        return render_template('fileUpload.html',
                               train_file_list=train_file_list,
                               test_file_list=test_file_list,
                               embed_model_list=embed_model_list,
                               machine_model_list=machine_model_list)


@app.route('/machineLearning', methods=["GET", "POST"])
def page3():

    train_file_list = os.listdir(train_file_path)
    test_file_list = os.listdir(test_file_path)
    embed_model_list = os.listdir(embed_model_path)
    machine_model_list = os.listdir(machine_model_path)

    if request.method == "POST":

        # 시각화 버튼을 눌렀을 경우
        if request.form.get("visual_button"):

            response_data = request.form.get("visual_button")
            response_data = json.loads(response_data)
            print(response_data)

            trainFile = response_data['trainData']
            testFile = response_data['testData']

            # 데이터 읽기
            train = pd.read_csv(train_file_path + trainFile)
            test = pd.read_csv(test_file_path + testFile)

            # 결측치가 있는지 확인하기(우선은 제거하는 방식)
            if pd.isnull(train['x']).sum() > 0 or pd.isnull(train['y']).sum() > 0:
                train = train.dropna()
            if pd.isnull(test['x']).sum() > 0 or pd.isnull(test['y']).sum() > 0:
                test = test.dropna()

            train = train.sample(frac=1).reset_index(drop=True)
            test = test.sample(frac=1).reset_index(drop=True)

            # 1) 처음 임베딩 및 시각화인 경우 -> 임베딩 파라미터만 받아오면 됨
            # is_pre_embed 없음, is_pre_train 없음, machine_value []
            if 'is_pre_embed' not in response_data and 'is_pre_machine' not in response_data and response_data['machine_value'] == []:

                print('first-embed, no-machine')

                embed_type = response_data['embed_type']
                embed_params = get_embed_params(embed_type, response_data['embed_value'])

                # 임베딩
                X_train, X_test, y_train, y_test = embedding(trainFile.split(".")[0], embed_type, train, test, embed_params)

                # 차원축소
                dimension_type = response_data['dimension_type']
                dimension_reduction(dimension_type, X_train, X_test, y_train, y_test)

                return render_template('visualization.html', visualization="embedding_and_visualization")

            # 2) pre 임베딩 및 시각화인 경우 -> 어떠한 파라미터도 받을 필요 없음
            # is_pre_embed 있음, is_pre_train 없음, embed_value [], machine_value []
            elif 'is_pre_embed' in response_data and 'is_pre_machine' not in response_data and response_data['embed_value'] == [] and response_data['machine_value'] == []:

                print('pre-embed, no-machine')

                embed_type = response_data['embed_type']
                pre_embed_model = response_data['pre_embed_model']

                # 임베딩
                X_train, X_test, y_train, y_test = pre_train_embedding(embed_type, pre_embed_model, train, test)

                # 차원축소
                dimension_type = response_data['dimension_type']
                dimension_reduction(dimension_type, X_train, X_test, y_train, y_test)

                return render_template('visualization.html', visualization="embedding_and_visualization")

            # 3) 처음 임베딩 및 처음 머신러닝 및 시각화인 경우 -> 임베딩, 머신러닝 파라미터 모두 받아오면 됨
            # is_pre_embed 없음, is_pre_train 없음, machine_value 있음
            elif 'is_pre_embed' not in response_data and 'is_pre_machine' not in response_data and response_data['machine_value'] != []:

                print('first-embed, first-machine')

                embed_type = response_data['embed_type']
                embed_params = get_embed_params(embed_type, response_data['embed_value'])

                machine_type = response_data['machine_type']
                machine_params = get_machine_params(machine_type, response_data['machine_value'])

                # 임베딩
                X_train, X_test, y_train, y_test = embedding(trainFile.split(".")[0], embed_type, train, test, embed_params)

                # 차원축소
                dimension_type = response_data['dimension_type']
                dimension_reduction(dimension_type, X_train, X_test, y_train, y_test)

                # 머신러닝
                train_y_pred, test_y_pred = machine_learning(embed_type, machine_type, X_train, X_test, y_train, y_test, machine_params)

            # 4) pre 임베딩 및 처음 머신러닝 및 시각화인 경우 -> 머신러닝 파라미터만 받아오면 됨
            # is_pre_embed 있음, is_pre_train 없음, embed_value [], machine_value 있음
            elif 'is_pre_embed' in response_data and 'is_pre_machine' not in response_data and response_data['embed_value'] == [] and response_data['machine_value'] != []:

                print('pre-embed, first-machine')

                embed_type = response_data['embed_type']
                pre_embed_model = response_data['pre_embed_model']

                machine_type = response_data['machine_type']
                machine_params = get_machine_params(machine_type, response_data['machine_value'])

                # 임베딩
                X_train, X_test, y_train, y_test = pre_train_embedding(embed_type, pre_embed_model, train, test)

                # 차원축소
                dimension_type = response_data['dimension_type']
                dimension_reduction(dimension_type, X_train, X_test, y_train, y_test)

                # 머신러닝
                train_y_pred, test_y_pred = machine_learning(embed_type, machine_type, X_train, X_test, y_train, y_test, machine_params)

            # 5) pre 임베딩 및 pre 머신러닝 및 시각화인 경우 -> 어떠한 파라미터도 받을 필요 없음
            # is_pre_embed 있음, is_pre_train 있음
            elif 'is_pre_embed' in response_data and 'is_pre_machine' in response_data:

                print('pre-embed, pre-machine')

                embed_type = response_data['embed_type']
                machine_type = response_data['machine_type']

                pre_embed_model = response_data['pre_embed_model']

                # 임베딩
                X_train, X_test, y_train, y_test = pre_train_embedding(embed_type, pre_embed_model, train, test)

                # 차원축소
                dimension_type = response_data['dimension_type']
                dimension_reduction(dimension_type, X_train, X_test, y_train, y_test)

                # 머신러닝
                train_y_pred, test_y_pred = pre_train_machine_learning(embed_type, machine_type, X_train, X_test, y_train, y_test)

            # 훈련 종료 후 머신러닝 결과
            from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

            target_names = list(set(y_train))
            train_df = pd.DataFrame(confusion_matrix(y_train, train_y_pred), index=target_names, columns=target_names)
            test_df = pd.DataFrame(confusion_matrix(y_test, test_y_pred), index=target_names, columns=target_names)

            path = r'/home/ubuntu/project2/csv_files/'

            train_df.to_csv(path + 'confusion_matrix_train.csv', index=False)
            test_df.to_csv(path + 'confusion_matrix_test.csv', index=False)

            # 분류 평가 지표
            train_accuracy = accuracy_score(y_train, train_y_pred)
            train_precision = precision_score(y_train, train_y_pred, average='macro')
            train_recall = recall_score(y_train, train_y_pred, average='macro')
            train_f1 = f1_score(y_train, train_y_pred, average='macro')

            test_accuracy = accuracy_score(y_test, test_y_pred)
            test_precision = precision_score(y_test, test_y_pred, average='macro')
            test_recall = recall_score(y_test, test_y_pred, average='macro')
            test_f1 = f1_score(y_test, test_y_pred, average='macro')

            print('train accuracy: {}, test accuracy: {}'.format(train_accuracy, test_accuracy))
            print('train precision: {}, test precision: {}'.format(train_precision, test_precision))
            print('train recall: {}, test recall: {}'.format(train_recall, test_recall))
            print('train f1: {}, test f1: {}'.format(train_f1, test_f1))

            train_score_df = pd.DataFrame(columns=['Metrics', 'Score'])
            train_score_df['Metrics'] = ['accuracy', 'precision', 'recall', 'f1']
            train_score_df['Score'] = [round(train_accuracy, 2), round(train_precision, 2), round(train_recall, 2), round(train_f1, 2)]
            train_score_df.to_csv(path + 'metrics_score_train.csv', index=False)

            test_score_df = pd.DataFrame(columns=['Metrics', 'Score'])
            test_score_df['Metrics'] = ['accuracy', 'precision', 'recall', 'f1']
            test_score_df['Score'] = [round(test_accuracy, 2), round(test_precision, 2), round(test_recall, 2), round(test_f1, 2)]
            test_score_df.to_csv(path + 'metrics_score_test.csv', index=False)

            train_df = pd.read_csv(path + 'embedding_and_visualization_train.csv')
            test_df = pd.read_csv(path + 'embedding_and_visualization_test.csv')

            train_df['pred'] = train_y_pred
            train_df['success'] = train_df['pred'] == train_df['target']
            train_df['success'] = train_df['success'].astype(int)

            test_df['pred'] = test_y_pred
            test_df['success'] = test_df['pred'] == test_df['target']
            test_df['success'] = test_df['success'].astype(int)

            success_mapping_table = {0: "실패", 1: "성공"}
            train_df['success'] = train_df['success'].map(success_mapping_table)
            test_df['success'] = test_df['success'].map(success_mapping_table)

            train_df.to_csv(path + 'embedding_and_machinelearning_visualization_train.csv', index=False)
            test_df.to_csv(path + 'embedding_and_machinelearning_visualization_test.csv', index=False)

            return render_template('visualization.html', visualization="embedding_and_machineLearning_visualization")

        return render_template('machineLearning.html', train_file_list=train_file_list,
                                                       test_file_list=test_file_list,
                                                       embed_model_list=embed_model_list,
                                                       machine_model_list=machine_model_list)
    else:
        return render_template("machineLearning.html", train_file_list=train_file_list,
                                                       test_file_list=test_file_list,
                                                       embed_model_list=embed_model_list,
                                                       machine_model_list=machine_model_list)


@app.route('/visualization', methods=["GET", "POST"])
def page4():
    if os.path.isfile('/home/ubuntu/project2/csv_files/metrics_score_train.csv'):
        return render_template('visualization.html', visualization="embedding_and_machineLearning_visualization")
    else:
        return render_template('visualization.html', visualization="embedding_and_visualization")


# 훈련 데이터 평가 지표 값을 받는 라우터
@app.route('/metrics_score_train')
def data1_1():
    df = pd.read_csv(path + 'metrics_score_train.csv')
    return df.to_csv()


# 테스트 데이터 평가 지표 값을 받는 라우터
@app.route('/metrics_score_test')
def data1_2():
    df = pd.read_csv(path + 'metrics_score_test.csv')
    return df.to_csv()


# 훈련 데이터 오차 행렬 값을 받는 라우터
@app.route('/confusion_matrix_train')
def data2_1():
    df = pd.read_csv(path + 'confusion_matrix_train.csv')
    values = df.values
    columns = df.columns

    # 그래프를 그리기 위한 데이터 정렬
    new_df = pd.DataFrame(columns=['group', 'variable', 'value'])

    cnt = 0
    for idx in range(len(values)):
        for jdx in range(len(values[idx]) - 1, -1, -1):
            new_df.loc[cnt, ['group']] = columns[idx]
            new_df.loc[cnt, ['variable']] = columns[jdx]
            new_df.loc[cnt, ['value']] = values[jdx][idx]
            cnt += 1

    return new_df.to_csv()


# 테스트 데이터 오차 행렬 값을 받는 라우터
@app.route('/confusion_matrix_test')
def data2_2():
    df = pd.read_csv(path + 'confusion_matrix_test.csv')
    values = df.values
    columns = df.columns

    # 그래프를 그리기 위한 데이터 정렬
    new_df = pd.DataFrame(columns=['group', 'variable', 'value'])

    cnt = 0
    for idx in range(len(values)):
        for jdx in range(len(values[idx])- 1, -1, -1):
            new_df.loc[cnt, ['group']] = columns[idx]
            new_df.loc[cnt, ['variable']] = columns[jdx]
            new_df.loc[cnt, ['value']] = values[jdx][idx]
            cnt += 1

    return new_df.to_csv()


# 훈련 데이터 차원 축소 값을 받는 라우터
@app.route('/embedding_and_visualization_train')
def data3_1():
    print('훈련 csv 파일 생성 완료')
    df = pd.read_csv(path + 'embedding_and_visualization_train.csv')
    return df.to_csv()


# 테스트 데이터 차원 축소 값을 받는 라우터
@app.route('/embedding_and_visualization_test')
def data3_2():
    print('테스트 csv 파일 생성 완료')
    df = pd.read_csv(path + 'embedding_and_visualization_test.csv')
    return df.to_csv()


# 훈련 데이터 머신러닝 후의 차원 축소 값을 받는 라우터
@app.route('/embedding_and_machinelearning_visualization_train')
def data4_1():
    df = pd.read_csv(path + 'embedding_and_machinelearning_visualization_train.csv')
    return df.to_csv()


# 테스트 데이터 머신러닝 후의 차원 축소 값을 받는 라우터
@app.route('/embedding_and_machinelearning_visualization_test')
def data4_2():
    df = pd.read_csv(path + 'embedding_and_machinelearning_visualization_test.csv')
    return df.to_csv()


# 훈련 csv 파일을 확인할 수 있는 라우터
@app.route('/data/train_csv/<file>', methods=["GET", "POST"])
def train_csv_file(file):
    file_name = pd.read_csv(train_file_path + file)
    return file_name.to_html()


# 테스트 csv 파일을 확인할 수 있는 라우터
@app.route('/data/test_csv/<file>', methods=["GET", "POST"])
def test_csv_file(file):
    file_name = pd.read_csv(test_file_path + file)
    return file_name.to_html()


# 임베딩 모델을 다운로드할 수 있는 라우터
@app.route('/data/embed/<model>', methods=["GET", "POST"])
def downloadable_data1(model):
    print('임베딩 모델 다운 받기')
    file_name = embed_model_path + model
    return send_file(file_name,
                     mimetype='pkl',
                     attachment_filename=model,
                     as_attachment=True)


# 머신러닝 모델을 다운로드할 수 있는 라우터
@app.route('/data/machine/<model>', methods=["GET", "POST"])
def downloadable_data2(model):
    print('머신러닝 모델 다운 받기')
    file_name = machine_model_path + model
    return send_file(file_name,
                     mimetype='pkl',
                     attachment_filename=model,
                     as_attachment=True)


if __name__ == "__main__":

    port = 8080
    os.system("open http://localhost:{0}".format(port))

    app.debug = True
    app.run(port=port)