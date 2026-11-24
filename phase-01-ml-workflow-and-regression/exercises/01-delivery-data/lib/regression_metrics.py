class RegressionMetrics:
    @staticmethod
    def residuals(y_true, y_pred):
        if len(y_true) != len(y_pred):
            raise ValueError("length must match")

        if not y_true or not y_pred:
            return []

        return [actual_val - pred_val for actual_val, pred_val in zip(y_true, y_pred)]

    @staticmethod
    def squared_errors(y_true, y_pred):
        return [err**2 for err in RegressionMetrics.residuals(y_true, y_pred)]

    @staticmethod
    def mae(y_true, y_pred):
        if len(y_true) != len(y_pred):
            raise ValueError("length must match")
        if not y_true:
            return 0.0

        abs_errors = [abs(err) for err in RegressionMetrics.residuals(y_true, y_pred)]
        return sum(abs_errors) / len(abs_errors)

    @staticmethod
    def mse(y_true, y_pred):
        if len(y_true) != len(y_pred):
            raise ValueError("length must match")
        if not y_true:
            return 0.0
        sq_errors = RegressionMetrics.squared_errors(y_true, y_pred)
        return sum(sq_errors) / len(sq_errors)

    @staticmethod
    def rmse(y_true, y_pred):
        return RegressionMetrics.mse(y_true, y_pred) ** 0.5

    @staticmethod
    def r_square(y_true, y_pred):
        if len(y_true) != len(y_pred):
            raise ValueError("length must match")
        if not y_true:
            return 0.0

        y_mean = sum(y_true) / len(y_true)
        ss_tot = sum((val - y_mean) ** 2 for val in y_true)
        if ss_tot == 0:
            raise ValueError(
                "R^2 is undefined when y_true has zero variance (all values are equal)"
            )

        ss_res = sum(RegressionMetrics.squared_errors(y_true, y_pred))
        return 1 - (ss_res / ss_tot)


if __name__ == "__main__":
    actual = [20, 30, 40]
    predicted = [22, 27, 40]

    r_metrics = RegressionMetrics()
    print(r_metrics.residuals(actual, predicted))
    print(r_metrics.squared_errors(actual, predicted))
    print(r_metrics.mae(actual, predicted))
    print(r_metrics.mse(actual, predicted))
    print(r_metrics.rmse(actual, predicted))
    print(r_metrics.r_square(actual, predicted))
