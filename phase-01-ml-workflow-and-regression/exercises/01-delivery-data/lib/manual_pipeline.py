class ManualPipeline:
    def __init__(self, preprocessor, estimator):
        self.preprocessor = preprocessor
        self.estimator = estimator

    def fit(self, X, y):
        transformed_data = self.preprocessor.fit_transform(X)
        self.estimator.fit(transformed_data, y)
        return self

    def predict(self, X):
        transformed_data = self.preprocessor.transform(X)
        return self.estimator.predict(transformed_data)
