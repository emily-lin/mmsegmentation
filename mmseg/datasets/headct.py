from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class HeadCTDataset(BaseSegDataset):
    """
    Head CT multiclass dataset.

    The segmentation map of Head CT multiclass dataset. 0 is set to background. We use the other 6 categories for evaluation.

    """
    METAINFO = dict(
      classes = ('Background', 'Contusion/ICH', 'TAI/Petechial', 'Epidural',
                 'Subdural', 'Subarachnoid', 'Intraventricular'),
      palette = [[0, 0, 0], [0, 128, 0], [128, 0, 0], [0, 0, 128], [31, 255, 0],
                 [255, 31, 0], [0, 61, 255]],
    )

    def __init__(self, img_suffix='_Im.png', seg_map_suffix='_Gt.png', reduce_zero_label=False, **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, reduce_zero_label=reduce_zero_label, **kwargs)
