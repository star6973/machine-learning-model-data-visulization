import logging
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class Model(object):
    """상속 받는 클래스는 아래의 4개의 함수를 모두 필히 구현해야함"""

    def __init__(self):
        logging.debug("loading {}".format(self.__class__.__name__))

    # 모델 구현 함수(밖으로 보이지 않는 함수)
    def _build_model(self):
        raise Exception("_build_model() method not implemented")

    # 학습 단계(밖으로 보여지는 함수)
    def fit(self, x, y):
        raise Exception("fit() method not implemented")

    # 예측 단계
    def predict(self, x):
        raise Exception("predict() method not implemented")

    # 예측 확률 출력 함수
    def predict_proba(self, x):
        raise Exception("predict_proba() method not implemented")


class myLogisticRegression(Model):

    def __init__(self, params, cv=4):
        self.params = params
        self.cv = cv # 교차검증
        self.model = self._build_model()

    # 하이퍼파라미터 조정
    def _build_model(self):
        return GridSearchCV(estimator=LogisticRegression(), param_grid=self.params, cv=self.cv)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)


class mySVM(Model):

    def __init__(self, params, cv=4):
        self.params = params
        self.cv = cv
        self.model = self._build_model()

    # 하이퍼파라미터 조정
    def _build_model(self):
        return GridSearchCV(estimator=SGDClassifier(), param_grid=self.params, cv=self.cv)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)


class myRandomForestClassifier(Model):

    def __init__(self, params, cv=4):
        self.params = params
        self.cv = cv
        self.model = self._build_model()

    # 하이퍼파라미터 조정
    def _build_model(self):
        return GridSearchCV(estimator=RandomForestClassifier(), param_grid=self.params, cv=self.cv)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)


class user_defined_machine_learning(Model):
    pass