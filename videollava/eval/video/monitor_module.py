import time
class MonitoringModule:

    def __init__(self, model, show_inference=True, show_token_count=True):
        self.show_inference = show_inference
        # self.show_token_count = show_token_count
        self.inference_results = {}
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def generate(self, *args, **inputs):
        ## --- Disabled token count, doesn't work on this version of the model. ---

        # if self.show_token_count:
        #     image_token_count = int((inputs["input_ids"] == DEFAULT_IMAGE_TOKEN).sum())
        #
        #     if image_token_count > 0:
        #         self.inference_results["image_token_count"] = image_token_count

        ## --- Disabled token count, doesn't work on this version of the model. ---

        start = time.time()
        out = self.model.generate(*args, **inputs)
        end = time.time()

        if self.show_inference:
            self.inference_results["inference_latency"] = end - start

        return out
