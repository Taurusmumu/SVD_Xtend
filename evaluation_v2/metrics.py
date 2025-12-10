import torch
from skimage.metrics import structural_similarity as ssim
import numpy as np
import lpips
import PIL.Image as Image
import math
import cv2
import torchvision.transforms as transforms


class MetricsCalculator:
    """A class to manage and calculate different evaluation metrics."""

    def __init__(self, metric_names, device):
        """
        Initializes the metric calculator.
        Args:
            metric_names (list): A list of metric names to calculate (e.g., ['ssim', 'lpips']).
            device (torch.device): The device to run calculations on.
        """
        self.metric_names = metric_names
        self.device = device
        self.lpips_fn = lpips.LPIPS(net='vgg').to(device)
        self.lpips_fn.eval()
        print("LPIPS model initialized.")

    def _calculate_flicker(self, frames):

        def calculate_mae(img1, img2):
            """Computing the mean absolute error (MAE) between two images."""
            if img1.shape != img2.shape:
                print("Images don't have the same shape.")
                return
            return np.mean(cv2.absdiff(np.array(img1, dtype=np.float32), np.array(img2, dtype=np.float32)))

        """Calculates SSIM between an image and each frame of a video."""
        score_seq = []
        for i in range(len(frames) - 1):
            score_seq.append(calculate_mae(frames[i], frames[i + 1]))
        return (255.0 - np.mean(score_seq).item()) / 255.0


    def _calculate_ssim(self, gt_np, pred_np):
        """Calculates SSIM between an image and each frame of a video."""
        num_frames = pred_np.shape[0]
        ssim_scores = []
        for i in range(num_frames):
            score = ssim(gt_np[i], pred_np[i], multichannel=True, channel_axis=2, data_range=1.0)
            ssim_scores.append(score)

        ssim_scores = np.mean(ssim_scores)
        return ssim_scores

    def _calculate_ssim_m(self, gt_np, pred_np):
        """Calculates SSIM between an image and each frame of a video."""
        num_frames = pred_np.shape[0]
        gt_middle = gt_np[num_frames // 2]
        ssim_scores = []
        for i in range(num_frames):
            score = ssim(gt_middle, pred_np[i], multichannel=True, channel_axis=2, data_range=1.0)
            ssim_scores.append(score)

        return ssim_scores

    def _calculate_ssim_ver(self, gt_np, pred_np):
        """Calculates SSIM between an image and each frame of a video."""
        # num_frames = pred_np.shape[0]
        # pred_gt = gt_np[num_frames // 2]
        # pred_pred = pred_np[num_frames // 2]
        # gt_scores = []
        # for i in range(num_frames):
        #     score = ssim(pred_gt, gt_np[i], multichannel=True, channel_axis=2, data_range=1.0)
        #     gt_scores.append(score)
        #
        # pred_scores = []
        # for i in range(num_frames):
        #     score = ssim(pred_pred, pred_np[i], multichannel=True, channel_axis=2, data_range=1.0)
        #     pred_scores.append(score)
        num_frames = pred_np.shape[0]
        pred_gt = gt_np[0]
        pred_pred = pred_np[0]
        gt_scores = []
        for i in range(1, num_frames):
            score = ssim(pred_gt, gt_np[i], multichannel=True, channel_axis=2, data_range=1.0)
            gt_scores.append(score)
            pred_gt = gt_np[i]

        pred_scores = []
        for i in range(1, num_frames):
            score = ssim(pred_pred, pred_np[i], multichannel=True, channel_axis=2, data_range=1.0)
            pred_scores.append(score)
            pred_pred = pred_np[i]

        compare_scores = np.mean(np.abs(np.array(gt_scores) - np.array(pred_scores)))
        return compare_scores

    def _psnr(self, img1, img2):

        mse = np.mean((img1 - img2) ** 2)
        if mse < 1e-10:
            return 100
        return 20 * math.log10(1.0 / math.sqrt(mse))

    def _calculate_psnr(self, gt_np, pred_np):
        """Calculates PSNR between an image and each frame of a video."""
        # [0,1]
        num_frames = pred_np.shape[0]
        psnr_scores = []
        for i in range(num_frames):
            psnr = self._psnr(gt_np[i], pred_np[i])
            psnr_scores.append(psnr)

        psnr_scores = np.mean(psnr_scores)
        return psnr_scores

    def _calculate_psnr_m(self, gt_np, pred_np):
        """Calculates PSNR between an image and each frame of a video."""
        # [0,1]
        num_frames = pred_np.shape[0]
        psnr_scores = []
        gt_middle = gt_np[num_frames // 2]
        for i in range(num_frames):
            psnr = self._psnr(gt_middle, pred_np[i])
            psnr_scores.append(psnr)

        return psnr_scores

    def _calculate_psnr_ver(self, gt_np, pred_np):
        """Calculates SSIM between an image and each frame of a video."""
        num_frames = pred_np.shape[0]
        # mid_gt = gt_np[num_frames // 2]
        # mid_pred = pred_np[num_frames // 2]
        # gt_scores = []
        # for i in range(num_frames):
        #     score = self._psnr(mid_gt, gt_np[i])
        #     gt_scores.append(score)
        #
        # pred_scores = []
        # for i in range(num_frames):
        #     score = self._psnr(mid_pred, pred_np[i])
        #     pred_scores.append(score)
        pred_gt = gt_np[0]
        pred_pred = pred_np[0]
        gt_scores = []
        for i in range(1, num_frames):
            score = self._psnr(pred_gt, gt_np[i])
            gt_scores.append(score)
            pred_gt = gt_np[i]

        pred_scores = []
        for i in range(1, num_frames):
            score = self._psnr(pred_pred, pred_np[i])
            pred_scores.append(score)
            pred_pred = pred_np[i]

        compare_scores = np.mean(np.abs(np.array(gt_scores) - np.array(pred_scores)))
        return compare_scores

    def _calculate_lpips(self, gt_tensor, pred_tensor):
        """Calculates LPIPS between an image and each frame of a video."""
        num_frames = gt_tensor.shape[0]
        lpips_scores = []
        with torch.no_grad():
            for i in range(num_frames):
                # lpips_fn expects tensors in range [-1, 1]
                score = self.lpips_fn(gt_tensor[i], pred_tensor[i])
                lpips_scores.append(score.item())

        lpips_scores = np.mean(lpips_scores)
        return lpips_scores

    def _calculate_lpips_m(self, gt_tensor, pred_tensor):
        """Calculates LPIPS between an image and each frame of a video."""
        num_frames = gt_tensor.shape[0]
        lpips_scores = []
        gt_middle = gt_tensor[num_frames // 2]
        with torch.no_grad():
            for i in range(num_frames):
                # lpips_fn expects tensors in range [-1, 1]
                score = self.lpips_fn(gt_middle, pred_tensor[i])
                lpips_scores.append(score.item())

        return lpips_scores

    def _calculate_lpips_ver(self, gt_tensor, pred_tensor):
        """Calculates SSIM between an image and each frame of a video."""
        num_frames = pred_tensor.shape[0]
        # mid_gt = gt_tensor[num_frames // 2]
        # mid_pred = pred_tensor[num_frames // 2]
        # gt_scores = []
        # for i in range(num_frames):
        #     score = self.lpips_fn(mid_gt, gt_tensor[i]).item()
        #     gt_scores.append(score)
        #
        # pred_scores = []
        # for i in range(num_frames):
        #     score = self.lpips_fn(mid_pred, pred_tensor[i]).item()
        #     pred_scores.append(score)
        pred_gt = gt_tensor[0]
        pred_pred = pred_tensor[0]
        gt_scores = []
        for i in range(1, num_frames):
            score = self.lpips_fn(pred_gt, gt_tensor[i]).item()
            gt_scores.append(score)
            pred_gt = gt_tensor[i]

        pred_scores = []
        for i in range(1, num_frames):
            score = self.lpips_fn(pred_pred, pred_tensor[i]).item()
            pred_scores.append(score)
            pred_pred = pred_tensor[i]

        compare_scores = np.mean(np.abs(np.array(gt_scores) - np.array(pred_scores)))
        return compare_scores

    def evaluate(self, original_video, recon_video, blur_mask=None):

        results = {}

        # Prepare tensors for different libraries
        # For scikit-image (ssim): convert to numpy, HWC, range [0, 1]
        original_video_np = np.array(original_video, dtype=np.float32) / 255.0
        recon_video_np = np.array(recon_video, dtype=np.float32) / 255.0
        if blur_mask is not None:
            gt_np = original_video_np[blur_mask]
            pred_np = recon_video_np[blur_mask]

        # For LPIPS: keep as torch tensor, BCHW, range [-1, 1]
        transform_pipeline = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL [0, 255] to Tensor [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizes Tensor [0, 1] to [-1, 1]
        ])
        original_video_tensor = torch.empty((len(original_video), 3, original_video[0].size[0], original_video[0].size[1]), device=self.device)
        recon_video_tensor = torch.empty((len(recon_video), 3, recon_video[0].size[0], recon_video[0].size[1]), device=self.device)
        for i in range(len(original_video)):
            original_video_tensor[i] = transform_pipeline(original_video[i]).to(self.device)
            recon_video_tensor[i] = transform_pipeline(recon_video[i]).to(self.device)

        if blur_mask is not None:
            gt_tensor = original_video_tensor[torch.tensor(blur_mask)]
            pred_tensor = recon_video_tensor[torch.tensor(blur_mask)]

        if 'ssim' in self.metric_names:
            results['ssim'] = self._calculate_ssim(gt_np, pred_np)
        if 'ssim_ver' in self.metric_names:
            results['ssim_ver'] = self._calculate_ssim_ver(gt_np, pred_np)
        if 'ssim_m' in self.metric_names:
            results['ssim_m'] = self._calculate_ssim_m(original_video_np, recon_video_np)

        if 'lpips' in self.metric_names and self.lpips_fn:
            results['lpips'] = self._calculate_lpips(gt_tensor, pred_tensor)
        if 'lpips_ver' in self.metric_names and self.lpips_fn:
            results['lpips_ver'] = self._calculate_lpips_ver(gt_tensor, pred_tensor)
        if 'lpips_m' in self.metric_names and self.lpips_fn:
            results['lpips_m'] = self._calculate_lpips_m(original_video_tensor, recon_video_tensor)

        if 'psnr' in self.metric_names:
            results['psnr'] = self._calculate_psnr(gt_np, pred_np)
        if 'psnr_ver' in self.metric_names:
            results['psnr_ver'] = self._calculate_psnr_ver(gt_np, pred_np)
        if 'psnr_m' in self.metric_names:
            results['psnr_m'] = self._calculate_psnr_m(original_video_np, recon_video_np)

        return results


class VBench:
    """A class to manage and calculate different evaluation metrics."""

    def __init__(self, metric_names, device):
        """
        Initializes the metric calculator.
        Args:
            metric_names (list): A list of metric names to calculate (e.g., ['ssim', 'lpips']).
            device (torch.device): The device to run calculations on.
        """
        self.metric_names = metric_names
        self.device = device
        # self.lpips_fn = None
        # if 'lpips' in self.metric_names:
        #     print("Initializing LPIPS model...")
        #     self.lpips_fn = lpips.LPIPS(net='vgg').to(device)
        #     self.lpips_fn.eval()
        #     print("LPIPS model initialized.")

    def _calculate_flicker(self, frames):

        def calculate_mae(img1, img2):
            """Computing the mean absolute error (MAE) between two images."""
            if img1.shape != img2.shape:
                print("Images don't have the same shape.")
                return
            return np.mean(cv2.absdiff(np.array(img1, dtype=np.float32), np.array(img2, dtype=np.float32)))

        """Calculates SSIM between an image and each frame of a video."""
        score_seq = []
        for i in range(len(frames) - 1):
            score_seq.append(calculate_mae(frames[i], frames[i + 1]))
        return (255.0 - np.mean(score_seq).item()) / 255.0

    def evaluate(self, video_list, config):

        result = {}
        if "temporal_flickering" in self.metric_names:
            temp_flicker_score = []
            for video in video_list:
                score = self._calculate_flicker(video)
                temp_flicker_score.append(score)
            result["temporal_flickering"] = np.mean(temp_flicker_score)
        print(f"Temporal Flickering Score: {np.mean(temp_flicker_score)}")
        return result
