# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable

from mmseg.registry import METRICS
from sklearn import metrics


@METRICS.register_module()
class IoUMetric(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(
                    pred_label)
                self.results.append(
                    self.intersect_and_union(pred_label, label, num_classes,
                                             self.ignore_index))
            # format_result
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                # The index range of official ADE20k dataset is from 0 to 150.
                # But the index range of output is from 0 to 149.
                # That is because we set reduce_zero_label=True.
                if data_sample.get('reduce_zero_label', False):
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)

    def compute_metrics(self, results: list, exclude_background=False,
                        print_metrics=True) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
            exclude_background: Exclude background in mean metrics. Default
              to False to keep ADE20k settings.
            print_metrics: Print metrics.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)

        class_names = self.dataset_meta['classes']

        # summary table
        if exclude_background:
          mean_metrics = {}
          for k, v in ret_metrics.items():
            if v.ndim < 1:
              mean_metrics[k] = np.nanmean(v)
            elif v.ndim == 1:
              mean_metrics[k] = np.nanmean(v[1:])
            else:
              raise ValueError('Invalid dimension of values', k, v.shape)
          ret_metrics_summary = OrderedDict({
            ret_metric: np.round(mean_metric_value * 100, 2)
              for ret_metric, mean_metric_value in mean_metrics.items()
          })
        else:
          ret_metrics_summary = OrderedDict({
              ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
              for ret_metric, ret_metric_value in ret_metrics.items()
          })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        # each class table
        ret_metrics.pop('aAcc', None)
        if print_metrics:
          ret_metrics_class = OrderedDict({
              ret_metric: np.round(ret_metric_value * 100, 2)
              for ret_metric, ret_metric_value in ret_metrics.items()
          })
          ret_metrics_class.update({'Class': class_names})
          ret_metrics_class.move_to_end('Class', last=False)
          class_table_data = PrettyTable()
          for key, val in ret_metrics_class.items():
              class_table_data.add_column(key, val)

          print_log('per class results:', logger)
          print_log('\n' + class_table_data.get_string(), logger=logger)
          return metrics
        else:
          return metrics, ret_metrics

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta**2) * (precision * recall) / (
                (beta**2 * precision) + recall)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall

        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics


@METRICS.register_module()
class IoUROCMetric(IoUMetric):
    """Add stack-level ROC evaluation."""

    def get_exam_name(self, img_path: str):
        """Get the exam ID from image path."""
        stack_id = osp.basename(img_path).split('_')[-4]  # Example: 04_1112-080735__ObjId5e09a068d3779b9c61fe4b46_1_022_Im.png
        assert 'ObjId' in stack_id, "ObjID must be in stack id."
        return stack_id

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(
                    pred_label)
                result = {
                  'segm':  self.intersect_and_union(pred_label, label, num_classes, self.ignore_index),
                  'exam_id': self.get_exam_name(data_sample['img_path'])
                }
                self.results.append(result)

    def compute_classify_metrics(self, results: Dict[str, list]) -> Dict[str, float]:
        """Compute classification metric."""
        exam_pred_label = {}
        for res in self.results:
          if res['exam_id'] not in exam_pred_label:
            # Set pred: area_pred_label, and label: area_label > 0
            exam_pred_label[res['exam_id']] =  {'pred': res['segm'][2].numpy(), 'label': res['segm'][3].numpy() > 0}
          else:
            exam_pred_label[res['exam_id']]['pred'] += res['segm'][2].numpy()
            exam_pred_label[res['exam_id']]['label'] = np.logical_or(res['segm'][3].numpy() > 0,  exam_pred_label[res['exam_id']]['label'])

        # Get one element to get num_classes.
        num_classes = next(iter(exam_pred_label.values()))['pred'].shape[0]
        # Exclude background ROC-AUC and set it to NaN.
        roc_classes = [np.nan]
        for c in range(1, num_classes):
          stack_scores = np.array([v['pred'][c] for v in exam_pred_label.values()])
          stack_labels = np.array([v['label'][c] for v in exam_pred_label.values()])
          fpr, tpr, thresh = metrics.roc_curve(stack_labels, stack_scores, pos_label=1)
          roc_auc = metrics.auc(fpr,tpr)
          roc_classes.append(roc_auc)
        ret_metrics = {'ROC-AUC': np.array(roc_classes)}
        summary_metrics = {'ROC-AUC': np.round(np.nanmean(roc_classes) * 100, 2)}
        return summary_metrics, ret_metrics
    
    def print_metrics(self, ret_metrics: Dict[str, float]):
        """Print the metrics."""
        # Print metrics.
        ret_metrics.pop('aAcc', None)
        logger: MMLogger = MMLogger.get_current_instance()
        class_names = self.dataset_meta['classes']
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('ROC-AUC per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

    def compute_metrics(self, results: Dict[str, list]) -> Dict[str, float]:
        """Compute the segmentation and classification metrics."""
        segm_results = [x['segm'] for x in self.results]
        segm_metrics, segm_ret_metrics = super().compute_metrics(segm_results, exclude_background=True, print_metrics=False)
        class_metrics, class_ret_metrics = self.compute_classify_metrics(self.results)
        segm_metrics.update(class_metrics)
        segm_ret_metrics.update(class_ret_metrics)
        self.print_metrics(segm_ret_metrics)
        return segm_metrics

