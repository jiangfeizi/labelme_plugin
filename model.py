import onnxruntime


class Model:
    def __init__(self, weights) -> None:
        self.session = onnxruntime.InferenceSession(weights, providers=['CPUExecutionProvider'])

    def __call__(self, output_names, input_feed): # format of image is bgr or gray.
        output = self.session.run(output_names, input_feed)
        return output
