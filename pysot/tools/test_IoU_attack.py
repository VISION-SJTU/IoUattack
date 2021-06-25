# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str,
        help='datasets')
parser.add_argument('--config', default='', type=str,
        help='config file')
parser.add_argument('--snapshot', default='', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)

def main():
    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(model)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0
    perturb_max = 10000
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            l2_normes = []
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == 0:
                    last_preturb = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                    save_bbox = gt_bbox_
                elif idx > frame_counter:
                    # black-box IoU attack
                    image = img
                    last_gt = save_bbox
                    # heavy noise image
                    heavy_noise = np.random.randint(-1, 2, (image.shape[0], image.shape[1], image.shape[2])) * 128
                    image_noise = image + heavy_noise
                    image_noise = np.clip(image_noise, 0, 255)

                    noise_sample = image_noise - 128
                    clean_sample_init = image.astype(np.float) - 128
                    image_noise = image_noise.astype(np.uint8)
                    # query
                    outputs_orig = tracker.track_fixed(image)
                    outputs_target = tracker.track_fixed(image_noise)
                    target_score = overlap_ratio(np.array(outputs_orig['bbox']), np.array(outputs_target['bbox']))
                    adversarial_sample = image.astype(np.float) - 128

                    if target_score < 0.8:
                        # parameters
                        n_steps = 0
                        epsilon = 0.05
                        delta = 0.05
                        weight = 0.5
                        para_rate = 0.9
                        # Move a small step
                        while True:
                            # Initialize with previous perturbations
                            clean_sample = clean_sample_init + weight * last_preturb
                            trial_sample = clean_sample + forward_perturbation(
                                epsilon * get_diff(clean_sample, noise_sample), adversarial_sample, noise_sample)
                            trial_sample = np.clip(trial_sample, -128, 127)
                            outputs_adv = tracker.track_fixed((trial_sample + 128).astype(np.uint8))
                            # IoU score
                            threshold_1 = overlap_ratio(np.array(outputs_orig['bbox']), np.array(outputs_adv['bbox']))
                            threshold_2 = overlap_ratio(np.array(last_gt), np.array(outputs_adv['bbox']))
                            threshold = para_rate * threshold_1 + (1 - para_rate) * threshold_2
                            adversarial_sample = trial_sample
                            break

                        while True:
                            # Tangential direction
                            d_step = 0
                            while True:
                                d_step += 1
                                # print("\t#{}".format(d_step))
                                trial_samples = []
                                score_sum = []
                                for i in np.arange(10):
                                    trial_sample = adversarial_sample + orthogonal_perturbation(delta,
                                                                                                adversarial_sample,
                                                                                                noise_sample)
                                    trial_sample = np.clip(trial_sample, -128, 127)
                                    # query
                                    outputs_adv = tracker.track_fixed((trial_sample + 128).astype(np.uint8))
                                    # IoU score
                                    score_1 = overlap_ratio(np.array(outputs_orig['bbox']),
                                                            np.array(outputs_adv['bbox']))
                                    score_2 = overlap_ratio(np.array(last_gt), np.array(outputs_adv['bbox']))
                                    score = para_rate * score_1 + (1 - para_rate) * score_2
                                    score_sum = np.hstack((score_sum, score))
                                    trial_samples.append(trial_sample)
                                d_score = np.mean(score_sum <= threshold)
                                if d_score > 0.0:
                                    if d_score < 0.3:
                                        delta /= 0.9
                                    elif d_score > 0.7:
                                        delta *= 0.9
                                    adversarial_sample = np.array(trial_samples)[np.argmin(np.array(score_sum))]
                                    threshold = score_sum[np.argmin(np.array(score_sum))]
                                    break
                                elif d_step >= 5 or delta > 0.3:
                                    break
                                else:
                                    delta /= 0.9
                            # Normal direction
                            e_step = 0
                            while True:
                                trial_sample = adversarial_sample + forward_perturbation(
                                    epsilon * get_diff(adversarial_sample, noise_sample), adversarial_sample,
                                    noise_sample)
                                trial_sample = np.clip(trial_sample, -128, 127)
                                # query
                                outputs_adv = tracker.track_fixed((trial_sample + 128).astype(np.uint8))
                                l2_norm = np.mean(get_diff(clean_sample_init, trial_sample))
                                # IoU score
                                threshold_1 = overlap_ratio(np.array(outputs_orig['bbox']),
                                                            np.array(outputs_adv['bbox']))
                                threshold_2 = overlap_ratio(np.array(last_gt), np.array(outputs_adv['bbox']))
                                threshold_sum = para_rate * threshold_1 + (1 - para_rate) * threshold_2

                                if threshold_sum <= threshold:
                                    adversarial_sample = trial_sample
                                    epsilon *= 0.9
                                    threshold = threshold_sum
                                    break
                                elif e_step >= 30 or l2_norm > perturb_max:
                                    break
                                else:
                                    epsilon /= 0.9
                            n_steps += 1

                            if threshold <= target_score or l2_norm > perturb_max:
                                adversarial_sample = np.clip(adversarial_sample, -128, 127)
                                l2_norm = np.mean(get_diff(clean_sample_init, adversarial_sample))
                                l2_normes.append(l2_norm)
                                break

                        last_preturb = adversarial_sample - clean_sample
                        img = (adversarial_sample + 128).astype(np.uint8)
                    else:
                        image = img
                        adversarial_sample = image + last_preturb
                        adversarial_sample = np.clip(adversarial_sample, 0, 255)
                        img = adversarial_sample.astype(np.uint8)
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            model_name = 'SiamRPN++(IoU_attack)'
            # save results
            video_path = os.path.join('results', args.dataset, model_name,
                    'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))


def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''
    rect1 = np.transpose(rect1)

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]
        
    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou

def orthogonal_perturbation(delta, prev_sample, target_sample):
    size = int(max(prev_sample.shape[0]/4, prev_sample.shape[1]/4, 224))
    prev_sample_temp = np.resize(prev_sample, (size, size, 3))
    target_sample_temp = np.resize(target_sample, (size, size, 3))
    # Generate perturbation
    perturb = np.random.randn(size, size, 3)
    perturb /= get_diff(perturb, np.zeros_like(perturb))
    perturb *= delta * np.mean(get_diff(target_sample_temp, prev_sample_temp))
    # Project perturbation onto sphere around target
    diff = (target_sample_temp - prev_sample_temp).astype(np.float32)
    diff /= get_diff(target_sample_temp, prev_sample_temp)
    diff = diff.reshape(3, size, size)
    perturb = perturb.reshape(3, size, size)
    for i, channel in enumerate(diff):
        perturb[i] -= np.dot(perturb[i], channel) * channel
    perturb = perturb.reshape(size, size, 3)
    perturb_temp = np.resize(perturb, (prev_sample.shape[0], prev_sample.shape[1], 3))
    return perturb_temp

def forward_perturbation(epsilon, prev_sample, target_sample):
    perturb = (target_sample - prev_sample).astype(np.float32)
    perturb /= get_diff(target_sample, prev_sample)
    perturb *= epsilon
    return perturb

def get_diff(sample_1, sample_2):
    sample_1 = sample_1.reshape(3, sample_1.shape[0], sample_1.shape[1])
    sample_2 = sample_2.reshape(3, sample_2.shape[0], sample_2.shape[1])
    sample_1 = np.resize(sample_1, (3, 271, 271))
    sample_2 = np.resize(sample_2, (3, 271, 271))

    diff = []
    for i, channel in enumerate(sample_1):
        diff.append(np.linalg.norm((channel - sample_2[i]).astype(np.float32)))
    return np.array(diff)


if __name__ == '__main__':
    main()
