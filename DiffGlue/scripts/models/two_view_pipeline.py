"""
A two-view sparse feature matching pipeline.

This model contains sub-models for each step:
    feature extraction, feature matching, outlier filtering, pose estimation.
Each step is optional, and the features or matches can be provided as input.
Default: SuperPoint with nearest neighbor matching.

Convention for the matches: m0[i] is the index of the keypoint in image 1
that corresponds to the keypoint i in image 0. m0[i] = -1 if i is unmatched.
"""

from hydra.utils import instantiate
from omegaconf import OmegaConf

from . import image_feature_extractor
from . import get_model
from .base_model import BaseModel

to_ctr = OmegaConf.to_container  # convert DictConfig to dict


class TwoViewPipeline(BaseModel):
    default_conf = {
        "encoder": {},
        "extractor": {
            "name": None,
            "trainable": False,
        },
        "diffuser": {"name": None},
        "matcher": {"name": None},
        "filter": {"name": None},
        "solver": {"name": None},
        "ground_truth": {"name": None},
        "allow_no_extract": False,
        "run_gt_in_forward": False,
    }
    required_data_keys = ["view0", "view1"]
    strict_conf = False  # need to pass new confs to children models
    components = [
        "extractor",
        "diffuser",
        "matcher",
        "filter",
        "solver",
        "ground_truth",
    ]

    def _init(self, conf):
        if conf.encoder._target_:
            print(f"Using encoder {conf.encoder._target_}")
            self.encoder = instantiate(
                conf.encoder, _recursive_=False
            )
        else:
            print("No encoder")

        if conf.extractor.name:
            self.extractor = get_model(conf.extractor.name)(
                to_ctr(conf.extractor)
            )

        if conf.diffuser.name:
            self.diffuser = get_model(conf.diffuser.name)(
                to_ctr(conf.diffuser)
            )
            self.eval_diffuser = get_model(conf.diffuser.name)(
                {
                    **to_ctr(conf.diffuser),
                    **{"timestep_respacing": str(conf.diffuser.ddim_steps)},
                }
            )

        if conf.matcher.name:
            self.matcher = get_model(conf.matcher.name)(to_ctr(conf.matcher))

        if conf.filter.name:
            self.filter = get_model(conf.filter.name)(to_ctr(conf.filter))

        if conf.solver.name:
            self.solver = get_model(conf.solver.name)(to_ctr(conf.solver))

        if conf.ground_truth.name:
            self.ground_truth = get_model(conf.ground_truth.name)(
                to_ctr(conf.ground_truth)
            )

    def extract_view(self, data, i):
        data_i = data[f"view{i}"]
        pred_i = data_i.get("cache", {})
        skip_extract = len(pred_i) > 0 and self.conf.allow_no_extract
        if self.conf.extractor.name and not skip_extract:
            pred_i = {**pred_i, **self.extractor(data_i)}
        elif self.conf.extractor.name and not self.conf.allow_no_extract:
            pred_i = {**pred_i, **self.extractor({**data_i, **pred_i})}
        return pred_i

    def _forward(self, data):
        pred0 = self.extract_view(data, "0")
        pred1 = self.extract_view(data, "1")
        pred = {
            **{k + "0": v for k, v in pred0.items()},
            **{k + "1": v for k, v in pred1.items()},
        }

        if self.conf.ground_truth.name and self.training:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})

        if self.conf.encoder._target_:
            pred = {**pred, **self.encoder({**data, **pred})}

        if self.conf.diffuser.name and self.conf.matcher.name:
            # print(list(pred.keys())) # ['keypoints0', 'keypoint_scores0', 'descriptors0', 'keypoints1', 'keypoint_scores1', 'descriptors1']
            # print(list(data.keys())) # ['H_0to1', 'scene', 'idx', 'is_illu', 'name', 'view0', 'view1']

            if self.training:
                pred = {
                    **pred,
                    **self.diffuser(self.matcher, {**data, **pred}),
                }
            else:
                pred = {
                    **pred,
                    **self.eval_diffuser(self.matcher, {**data, **pred}),
                }
            # print(list(pred.keys())) # ['keypoints0', 'keypoint_scores0', 'descriptors0', 'keypoints1', 'keypoint_scores1', 'descriptors1', 'matches0', 'matches1', 'matching_scores0', 'matching_scores1', 'ref_descriptors0', 'ref_descriptors1', 'log_assignment', 'mean', 'variance', 'log_variance', 'pred_xstart', 'sample']
            # print(list(data.keys())) # ['H_0to1', 'scene', 'idx', 'is_illu', 'name', 'view0', 'view1']
        elif self.conf.matcher.name:
            pred = {**pred, **self.matcher({**data, **pred})}
        if self.conf.filter.name:
            pred = {**pred, **self.filter({**data, **pred})}
        if self.conf.solver.name:
            pred = {**pred, **self.solver({**data, **pred})}
        return pred

    def loss(self, pred, data):
        losses = {}
        metrics = {}
        total = 0

        # get labels
        if self.conf.ground_truth.name and not self.training:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})

        for k in self.components:
            apply = True
            if "apply_loss" in self.conf[k].keys():
                apply = self.conf[k].apply_loss
            if self.conf[k].name and apply:
                try:
                    if k == "diffuser" and not self.training:
                        k = "eval_diffuser"
                    losses_, metrics_ = getattr(self, k).loss(
                        pred, {**pred, **data}
                    )
                except NotImplementedError:
                    continue
                if "matcher_total" in losses_.keys():
                    losses_["total"] = losses_["matcher_total"]
                elif "diffuser_total" in losses_.keys():
                    losses_["total"] = (
                        losses_["diffuser_total"]
                        * self.conf.diffuser.diffuser_loss_weight
                    )
                losses = {**losses, **losses_}
                metrics = {**metrics, **metrics_}
                total = losses_["total"] + total
        return {**losses, "total": total}, metrics
