import time, os, sys, subprocess
import numpy as np
import cv2
import torch
from torchvision import transforms
from pathlib import Path

from .nets import S3FDNet
from .box_utils import nms_


img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')

class S3FD():

    def __init__(self, device='cuda'):

        tstamp = time.time()
        self.device = device

        PATH_WEIGHT = Path(__file__).parent / "sfd_face.pth"
        if os.path.isfile(PATH_WEIGHT) == False:
            Link = "1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt"
            cmd = "gdown --id %s -O %s"%(Link, PATH_WEIGHT)
            subprocess.call(cmd, shell=True, stdout=None)
        

        # print('[S3FD] loading with', self.device)
        self.net = S3FDNet(device=self.device).to(self.device)
        PATH = os.path.join(os.getcwd(), PATH_WEIGHT)
        state_dict = torch.load(PATH, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        # print('[S3FD] finished loading (%.4f sec)' % (time.time() - tstamp))
    
    def detect_faces(self, image, conf_th=0.8, scales=[1]):

        w, h = image.shape[1], image.shape[0]

        bboxes = np.empty(shape=(0, 5))

        with torch.no_grad():
            for s in scales:
                scaled_img = cv2.resize(image, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

                scaled_img = np.swapaxes(scaled_img, 1, 2)
                scaled_img = np.swapaxes(scaled_img, 1, 0)
                scaled_img = scaled_img[[2, 1, 0], :, :]
                scaled_img = scaled_img.astype('float32')
                scaled_img -= img_mean
                scaled_img = scaled_img[[2, 1, 0], :, :]
                x = torch.from_numpy(scaled_img).unsqueeze(0).to(self.device)
                y = self.net(x)

                detections = y.data
                scale = torch.Tensor([w, h, w, h])

                for i in range(detections.size(1)):
                    j = 0
                    while detections[0, i, j, 0] > conf_th:
                        score = detections[0, i, j, 0]
                        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                        bbox = (pt[0], pt[1], pt[2], pt[3], score)
                        bboxes = np.vstack((bboxes, bbox))
                        j += 1

            keep = nms_(bboxes, 0.1)
            bboxes = bboxes[keep]

        return bboxes

    # def _preprocess_single(self, img, target_size=None):
    #     # img: H, W, 3 (BGR cv2 image)
    #     # target_size: (W, H) to which this img will be resized before sending to net
    #     if target_size is not None:
    #         img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_LINEAR)
    #     # follow original preprocessing sequence but vectorized
    #     img = img.astype('float32')
    #     # convert to C,H,W
    #     img = img.transpose(2, 0, 1)  # (3,H,W)
    #     # original code did some channel reorders; to stay consistent:
    #     img = img[[2, 1, 0], :, :]    # swap BGR <-> RGB path used originally
    #     img -= img_mean               # img_mean is (3,1,1)
    #     img = img[[2, 1, 0], :, :]    # swap back (keeps original behavior)
    #     return img  # (3,H,W), float32

    # def detect_faces_batch(self, images, conf_th=0.8, scales=[1], resize_to=None, nms_thresh=0.1):
    #     """
    #     images: either a single numpy array (H,W,3) or a list/np.ndarray of images (H,W,3).
    #     resize_to: if provided, tuple (W, H) - all images will be resized to this shape before net.
    #             If None and images have different sizes, all images will be resized to images[0]'s size.
    #     returns: list of bboxes arrays, one per input image. Each bboxes array shape (N,5): x1,y1,x2,y2,score
    #     """
    #     # Normalize input list
    #     if isinstance(images, np.ndarray) and images.ndim == 3:
    #         images = [images]
    #     elif isinstance(images, np.ndarray) and images.ndim == 4:
    #         images = [images[i] for i in range(images.shape[0])]
    #     else:
    #         images = list(images)

    #     B = len(images)
    #     if B == 0:
    #         return []

    #     # remember original sizes for rescaling boxes back
    #     orig_sizes = [(img.shape[1], img.shape[0]) for img in images]  # (w,h)
    #     # if resize_to not given, ensure all inputs have same size; if not, resize to first image size
    #     if resize_to is None:
    #         first_w, first_h = orig_sizes[0]
    #         same_size = all((w == first_w and h == first_h) for (w, h) in orig_sizes)
    #         if not same_size:
    #             target_size = (first_w, first_h)
    #         else:
    #             target_size = (first_w, first_h)
    #     else:
    #         target_size = resize_to  # (W, H)

    #     all_bboxes_per_image = [[] for _ in range(B)]

    #     with torch.no_grad():
    #         self.net.eval()

    #         # process per scale but batch across images for that scale
    #         for s in scales:
    #             # prepare batch of scaled images (same spatial dims required)
    #             preproc_imgs = []
    #             scaled_sizes = []
    #             for img in images:
    #                 # resize original image to common base size first
    #                 # then scale by s
    #                 base_resized = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_LINEAR)
    #                 if s != 1:
    #                     scaled_w = int(round(target_size[0] * s))
    #                     scaled_h = int(round(target_size[1] * s))
    #                     scaled_img = cv2.resize(base_resized, dsize=(scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    #                 else:
    #                     scaled_img = base_resized
    #                     scaled_w, scaled_h = target_size

    #                 scaled_sizes.append((scaled_w, scaled_h))
    #                 preproc = self._preprocess_single(scaled_img, target_size=None)
    #                 preproc_imgs.append(preproc)

    #             # stack into batch (B,3,H,W)
    #             x = torch.from_numpy(np.stack(preproc_imgs, axis=0)).to(self.device)

    #             # forward
    #             y = self.net(x)  # expect shape (B, num_classes, top_k, 5) as before
    #             detections = y.data  # tensor

    #             # compute and cache priors for this scaled size if needed
    #             # PriorBox expects size as (width,height)
    #             feat_maps = []
    #             # build features_maps in same way original wrapper did
    #             # we need to reconstruct features_maps given loc layers outputs shapes
    #             # easiest: run a dummy forward of loc heads on one of source feature maps is not trivial here.
    #             # BUT your model internally computes PriorBox earlier (in original forward). The original wrapper
    #             # computed PriorBox inside the net's forward using 'size' and features_maps collected from loc outputs.
    #             # We don't have access to those shapes here easily; Instead rely on the model's output 'y' which is detections.
    #             # So we avoid recomputing prior here. If your net requires explicit PriorBox usage outside, compute/priorbox here.
    #             # (In your original wrapper you used priorbox AFTER computing loc/conf; keep same pattern.)

    #             # decode per-image in batch
    #             for b in range(B):
    #                 w_orig, h_orig = orig_sizes[b]
    #                 w_scaled, h_scaled = scaled_sizes[b]
    #                 scale = torch.tensor([w_orig, h_orig, w_orig, h_orig], device=self.device)

    #                 # detections shape: (B, num_classes, top_k, 5)
    #                 # guard length of top_k dimension
    #                 num_classes = detections.size(1)
    #                 top_k = detections.size(2)

    #                 bboxes = []
    #                 for i in range(num_classes):
    #                     j = 0
    #                     # iterate while j < top_k and score > conf_th
    #                     while j < top_k and detections[b, i, j, 0] > conf_th:
    #                         score = float(detections[b, i, j, 0].cpu().item())
    #                         pt = (detections[b, i, j, 1:] * scale).cpu().numpy()  # x1,y1,x2,y2 scaled to orig image
    #                         bbox = (pt[0], pt[1], pt[2], pt[3], score)
    #                         bboxes.append(bbox)
    #                         j += 1

    #                 if len(bboxes) > 0:
    #                     bboxes = np.array(bboxes, dtype=np.float32)
    #                     keep = nms_(bboxes, nms_thresh)
    #                     bboxes = bboxes[keep]
    #                 else:
    #                     bboxes = np.empty((0, 5), dtype=np.float32)

    #                 # append detections found at this scale to the result for this image
    #                 if bboxes.size != 0:
    #                     # stack with existing detections from previous scales
    #                     if len(all_bboxes_per_image[b]) == 0:
    #                         all_bboxes_per_image[b] = bboxes
    #                     else:
    #                         all_bboxes_per_image[b] = np.vstack((all_bboxes_per_image[b], bboxes))

    #         # After all scales processed, run one more NMS per image to merge across scales
    #         final_results = []
    #         for b in range(B):
    #             if isinstance(all_bboxes_per_image[b], np.ndarray) and all_bboxes_per_image[b].size != 0:
    #                 keep = nms_(all_bboxes_per_image[b], nms_thresh)
    #                 res = all_bboxes_per_image[b][keep]
    #             else:
    #                 res = np.empty((0, 5), dtype=np.float32)
    #             final_results.append(res)

    #     return final_results
