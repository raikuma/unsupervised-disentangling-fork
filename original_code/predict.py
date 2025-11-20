import os
from dataloading import (
    load_and_preprocess_image,
    dataset_map_train,
    dataset_map_test,
    keypoint_files_map,
)
from transformations import tps_parameters
from dotmap import DotMap
import numpy as np
from config import parse_args, write_hyperparameters
from model import Model
from utils import (
    save_python_files,
    transformation_parameters,
    find_ckpt,
    batch_colour_map,
    save,
    initialize_uninitialized,
)
import tensorflow as tf
import numpy as np
from typing import *
import sklearn
import seaborn
from sklearn import *
import json


def pck(distances: np.ndarray, tolerance_pixels: int, image_size: int):
    # 6 pixel tolerance, normalized to 256 image distance
    pck = distances < (tolerance_pixels / image_size)
    return np.mean(pck)


def main(arg):
    model_save_dir = os.path.join("experiments", arg.name)
    with tf.variable_scope("Data_prep"):
        raw_dataset = dataset_map_test[arg.dataset](arg)

        dataset = raw_dataset.map(
            load_and_preprocess_image, num_parallel_calls=arg.data_parallel_calls
        )
        # allow last smaller batch but pad it up to arg.bn and return valid count
        dataset = dataset.batch(arg["bn"], drop_remainder=False)

        def _pad_batch(batch):
            m = tf.shape(batch)[0]
            valid = tf.identity(m)

            def _pad():
                repeats = arg["bn"] - m
                last = batch[-1:]
                padding = tf.tile(last, [repeats, 1, 1, 1])
                batch_padded = tf.concat([batch, padding], axis=0)
                return batch_padded, valid

            def _nopad():
                return batch, valid

            return tf.cond(tf.less(m, arg["bn"]), _pad, _nopad)

        dataset = dataset.map(_pad_batch)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        b_images, valid_count = next_element

        orig_images = tf.tile(b_images, [2, 1, 1, 1])

        scal = tf.placeholder(dtype=tf.float32, shape=(), name="scal_placeholder")
        tps_scal = tf.placeholder(dtype=tf.float32, shape=(), name="tps_placeholder")
        rot_scal = tf.placeholder(
            dtype=tf.float32, shape=(), name="rot_scal_placeholder"
        )
        off_scal = tf.placeholder(
            dtype=tf.float32, shape=(), name="off_scal_placeholder"
        )
        scal_var = tf.placeholder(
            dtype=tf.float32, shape=(), name="scal_var_placeholder"
        )
        augm_scal = tf.placeholder(
            dtype=tf.float32, shape=(), name="augm_scal_placeholder"
        )
        coord_jitter = tf.placeholder(dtype=tf.float32, shape=(), name="coord_jitter_placeholder")

        tps_param_dic = tps_parameters(
            2 * arg.bn, scal, tps_scal, rot_scal, off_scal, scal_var, rescal=1, coord_jitter=coord_jitter
        )
        tps_param_dic.augm_scal = augm_scal

    ctr = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    if "infer" in arg.mode:
        with tf.Session(config=config) as sess:

            model = Model(orig_images, arg, tps_param_dic)
            tvar = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            saver = tf.train.Saver(var_list=tvar)
            merged = tf.summary.merge_all()

            ckpt, ctr = find_ckpt(os.path.join(model_save_dir, "saved_model"))
            saver.restore(sess, ckpt)

            initialize_uninitialized(sess)
            mu_list = []
            while True:
                try:
                    feed = transformation_parameters(
                        arg, ctr, no_transform=True
                    )  # no transform if arg.visualize
                    # When no_transform=True we still used the configured `scal`
                    # (default 0.8) which applies a zoom in the TPS transform
                    # and results in a cropped visualization. Force scal=1.0
                    # here so the visualized image is the full original.
                    feed.scal = 1.0
                    trf = {
                        scal: feed.scal,
                        tps_scal: feed.tps_scal,
                        coord_jitter: 0.0,
                        scal_var: feed.scal_var,
                        rot_scal: feed.rot_scal,
                        off_scal: feed.off_scal,
                        augm_scal: feed.augm_scal,
                    }
                    ctr += 1

                    img, img_rec, mu, heat_raw, valid = sess.run(
                        [
                            model.image_in,
                            model.reconstruct_same_id,
                            model.mu,
                            batch_colour_map(model.part_maps),
                            valid_count,
                        ],
                        feed_dict=trf,
                    )
                    v = int(valid)
                    save(img[:v, ...], mu[:v, ...], ctr, model_save_dir)
                    mu_list.append(mu[:v, ...])
                except tf.errors.OutOfRangeError:
                    print("End of Prediction")
                    break
            print("Saving outputs")
            mu_list = np.concatenate(mu_list, axis=0)
            np.savez_compressed(
                os.path.join(model_save_dir, "keypoints_predicted_test.npz"),
                keypoints_predicted=mu_list,
            )

    if "eval" in arg.mode:
        # TODO: extract method
        if arg.dataset in ["deepfashion"]:
            # regress to keypoints
            with np.load(
                os.path.join(model_save_dir, "keypoints_predicted.npz")
            ) as data:
                keypoints_predicted = data["keypoints_predicted"]

            with open(keypoint_files_map[arg.dataset], "rb",) as f:
                gt_keypoint_data = json.load(f)
                gt_keypoints = (
                    np.stack([d["keypoints"] for d in gt_keypoint_data], axis=0) / 256.0
                )
                joint_order = gt_keypoint_data[0]["joint_order"]

            N = keypoints_predicted.shape[0]

            X_train = keypoints_predicted[:N, ...].reshape(N, -1)
            y_train = gt_keypoints[:N, ...].reshape(N, -1)
            X_test = X_train

            regr = sklearn.linear_model.Ridge(alpha=0.0, fit_intercept=False)
            _ = regr.fit(X_train, y_train)
            y_predict = regr.predict(X_test)
            regressed_keypoints = y_predict.reshape(N, -1, 2)
            joint_order = gt_keypoint_data[0]["joint_order"]
            landmarks_gt = gt_keypoints[:N, ...]
            landmarks_regressed = y_predict.reshape((N, -1, 2))
            distances = np.linalg.norm(landmarks_gt - landmarks_regressed, axis=-1)
            np.savez_compressed(
                os.path.join(model_save_dir, "keypoints_regressed.npz"),
                regressed_keypoints=landmarks_regressed,
                distances=distances,
            )

            import seaborn
            from matplotlib import pyplot as plt
            import pandas as pd

            table = pd.DataFrame(distances, columns=joint_order.values())

            plt.style.use("seaborn-whitegrid")
            NB_RC_PARAMS = {
                "figure.figsize": [8, 5],
                "figure.dpi": 220,
                "figure.autolayout": True,
                "legend.frameon": True,
            }
            with plt.rc_context(NB_RC_PARAMS):
                ax = table.boxplot(rot=45, showfliers=False, fontsize=12)
                ax.set_ylabel(r"$||e||$")
                ax.set_ylim([0, 0.1])
                plt.savefig(os.path.join(model_save_dir, "keypoint_distances.png"))

            pck_value = pck(distances, arg.pck_tolerance, arg.in_dim)
            with open(os.path.join(model_save_dir, "metrics.txt"), "w") as f:
                print(f"pck : {100 * pck_value: .0f}%", file=f)

        if arg.dataset in ["cub"]:
            # regress to keypoints
            with np.load(
                os.path.join(model_save_dir, "keypoints_predicted.npz")
            ) as data:
                keypoints_predicted = data["keypoints_predicted"] # (N', 10, 2)
            gt_keypoints_data = np.load(keypoint_files_map[arg.dataset]) # (N, 15, 3)
            
            # TODO: filter failed detections
            train_X = keypoints_predicted * 0.5 + 0.5
            train_y = gt_keypoints_data[:, :, :2][:train_X.shape[0], ...]
            visibility = gt_keypoints_data[:, :, -1][:train_X.shape[0], ...]
            # train_X = torch.cat([batch['det_keypoints'] for batch in batch_list]) * 0.5 + 0.5
            # train_y = torch.cat([batch['keypoints'] for batch in batch_list])
            # visibility = torch.cat([batch['visibility'] for batch in batch_list])

            scores = []
            num_gnd_kp = 15
            betas = []
            for i in range(num_gnd_kp):
                # index = visibility[:, i].bool()
                index = visibility[:, i].astype(bool)
                if index.sum() == 0:
                    betas.append(np.zeros(2*train_X.shape[1], 2))
                    continue
                features = train_X[index]
                features = features.reshape(features.shape[0], -1)
                label = train_y[index, i]
                try:
                    # beta = (features.T @ features).inverse() @ features.T @ label
                    beta = np.linalg.inv(features.T @ features) @ features.T @ label
                except:
                    # beta = (features.T @ features + np.eye(features.shape[-1]).to(features)).inverse() @ features.T @ label
                    beta = np.linalg.inv(features.T @ features + np.eye(features.shape[-1])) @ features.T @ label
                betas.append(beta)

                pred_label = features @ beta
                # score = (pred_label - label).norm(dim=-1).sum()
                score = np.linalg.norm(pred_label - label, axis=-1).sum()
                scores.append(score.item())

            print('val_loss', np.sum(scores) / visibility.sum().item())

            landmarks_gt = train_y
            landmarks_regressed = np.zeros_like(landmarks_gt)
            for i in range(num_gnd_kp):
                features = train_X.reshape(train_X.shape[0], -1)
                beta = betas[i]
                pred_label = features @ beta
                landmarks_regressed[:, i, :] = pred_label
            distances = np.linalg.norm(landmarks_gt - landmarks_regressed, axis=-1)

            # apply visibility mask
            distances = distances * visibility

            np.savez_compressed(
                os.path.join(model_save_dir, "keypoints_regressed.npz"),
                regressed_keypoints=landmarks_regressed,
                distances=distances,
            )

            pck_value = pck(distances, arg.pck_tolerance, arg.in_dim)
            with open(os.path.join(model_save_dir, "metrics.txt"), "w") as f:
                print(f"pck : {100 * pck_value: .0f}%", file=f)
                print(f"pck : {100 * pck_value: .0f}%")


if __name__ == "__main__":
    arg = DotMap(vars(parse_args()))
    if arg.decoder == "standard":
        if arg.reconstr_dim == 256:
            arg.rec_stages = [
                [256, 256],
                [128, 128],
                [64, 64],
                [32, 32],
                [16, 16],
                [8, 8],
                [4, 4],
            ]
            arg.feat_slices = [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [4, arg.n_parts],
                [2, 4],
                [0, 2],
            ]
            arg.part_depths = [
                arg.n_parts,
                arg.n_parts,
                arg.n_parts,
                arg.n_parts,
                arg.n_parts,
                4,
                2,
            ]

        if arg.reconstr_dim == 128:
            arg.rec_stages = [[128, 128], [64, 64], [32, 32], [16, 16], [8, 8], [4, 4]]
            arg.feat_slices = [[0, 0], [0, 0], [0, 0], [4, arg.n_parts], [2, 4], [0, 2]]
            arg.part_depths = [arg.n_parts, arg.n_parts, arg.n_parts, arg.n_parts, 4, 2]
    main(arg)
