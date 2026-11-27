import numpy as np

from lib.manual_imputer import ManualSimpleImputer


class ManualColumnTransformer:
    def __init__(self, transformers):
        self.transformers = transformers

    def fit(self, data):
        for transform in self.transformers:
            _, steps, columns = transform

            # Select the columns assigned to this branch.
            current_data = data[columns].values.tolist()

            for step in steps:
                # The imputer expects 2D data.
                if isinstance(step, ManualSimpleImputer):
                    step_data = current_data
                # The scaler and encoder expect 1D data.
                else:
                    step_data = [row[0] for row in current_data]

                # Learn from the current data.
                step.fit(step_data)

                # Transform the data before fitting the next step.
                result = step.transform(step_data)

                # Keep the data in 2D form for the next iteration.
                if result and not isinstance(result[0], list):
                    current_data = [[value] for value in result]
                else:
                    current_data = result

        return self

    def transform(self, data):
        transformed_branches = []

        for transform in self.transformers:
            _, steps, columns = transform

            current_data = data[columns].values.tolist()

            for step in steps:
                if isinstance(step, ManualSimpleImputer):
                    step_data = current_data
                else:
                    step_data = [row[0] for row in current_data]

                result = step.transform(step_data)

                if result and not isinstance(result[0], list):
                    current_data = [[value] for value in result]
                else:
                    current_data = result

            transformed_branches.append(current_data)

        return np.hstack(transformed_branches)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
