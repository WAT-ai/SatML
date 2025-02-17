
class BaseProcessor:

    def __init__(self, config):
        """
        Initialize the preprocessor with a config specifying which steps to apply.

        :param config: Dictionary where keys are preprocessing step names 
                       and values are booleans (True = Apply, False = Skip).
        """
        self.config = config
        self.pipeline = self.build_pipeline()

    def build_pipeline(self):
        """
        Constructs the preprocessing pipeline dynamically based on the config.
        """
        pipeline = []
        for step_name in self.config:
            if self.config[step_name] and hasattr(self, step_name):
                pipeline.append(getattr(self, step_name))
        return pipeline

    def preprocess(self, data):
        """
        Applies the dynamically built preprocessing pipeline to the dataset.
        """
        for step in self.pipeline:
            data = step(data)
        return data
    
