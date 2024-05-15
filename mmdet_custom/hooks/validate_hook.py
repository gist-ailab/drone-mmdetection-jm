from mmengine.runner import Hook
import torch
import os
from mmdet.apis import inference_detector, show_result_pyplot


class ValidateHook(Hook):
    def __init__(self, interval, save_path):
        self.interval = interval
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return
        # Ensure the model is in eval mode
        runner.model.eval()
        results = []
        dataloader = runner.data_loader['val']
        dataset = dataloader.dataset

        # Iterate over the validation dataset
        for i, data in enumerate(dataloader):
            with torch.no_grad():
                result = inference_detector(runner.model, data['img'][0])
            results.append(result)
            # Save inference images
            img_path = os.path.join(self.save_path, f'iter_{runner.iter}_img_{i}.jpg')
            show_result_pyplot(runner.model, data['img'][0], result, out_file=img_path)

        runner.model.train()  # Return model to training mode

    def every_n_iters(self, runner, n):
        """ Check if runner.iter is a multiple of n. """
        return runner.iter % n == 0 if n > 0 else False