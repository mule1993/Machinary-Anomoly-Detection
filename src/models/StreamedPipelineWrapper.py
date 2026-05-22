import mlflow


class StreamedPipelineWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, preprocessor, model):
        """Pass the in-memory fitted objects directly to the instance."""
        self.preprocessor = preprocessor
        self.model = model
        self.expected_features = list(preprocessor.feature_names_in_)

    def predict(self, context, model_input):
        """Unified inference gateway."""
        # Execute your structural transform pipeline sequentially
        preprocessed_data = self.preprocessor.transform(model_input)
        return self.model.predict(preprocessed_data)
