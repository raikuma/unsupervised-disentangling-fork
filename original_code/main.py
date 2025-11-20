import os
from dataloading import load_and_preprocess_image, dataset_map_train, dataset_map_test
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
import time


def main(arg):
    model_save_dir = os.path.join("experiments", arg.name)

    with tf.variable_scope("Data_prep"):
        if arg.mode == "train":
            raw_dataset = dataset_map_train[arg.dataset](arg)

        elif arg.mode == "predict":
            # run prediction over the full train set (not test)
            raw_dataset = dataset_map_train[arg.dataset](arg)

        dataset = raw_dataset.map(
            load_and_preprocess_image, num_parallel_calls=arg.data_parallel_calls
        )
        # For training we repeat for epochs; for predict we want a single pass
        if arg.mode == "train":
            dataset = dataset.batch(arg["bn"], drop_remainder=True).repeat(arg.epochs)
        else:
            dataset = dataset.batch(arg["bn"], drop_remainder=True)
        # Batch handling: keep valid count and pad last batch up to `arg.bn` if needed
        if arg.mode == "train":
            dataset = dataset.batch(arg["bn"], drop_remainder=True)
            dataset = dataset.map(lambda batch: (batch, tf.constant(arg["bn"], dtype=tf.int32)))
        else:
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
        pad_size = arg.pad_size
        b_images = tf.pad(
            b_images,
            tf.constant([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]]),
            constant_values=1,
        )

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
    with tf.Session(config=config) as sess:

        model = Model(orig_images, arg, tps_param_dic)
        tvar = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        saver = tf.train.Saver(var_list=tvar)
        merged = tf.summary.merge_all()

        if arg.mode == "train":
            if arg.load:
                ckpt, ctr = find_ckpt(os.path.join(model_save_dir, "saved_model"))
                saver.restore(sess, ckpt)
            else:
                save_python_files(save_dir=os.path.join(model_save_dir, "bin"))
                write_hyperparameters(arg.toDict(), model_save_dir)
                sess.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter(
                os.path.join("summaries", arg.name), graph=sess.graph
            )

        elif arg.mode == "predict":
            ckpt, ctr = find_ckpt(os.path.join(model_save_dir, "saved_model"))
            saver.restore(sess, ckpt)

        initialize_uninitialized(sess)
        # start timer for CLI progress
        start_time = time.time()
        # If predicting over the train set, collect predicted mus and save once
        if arg.mode == "predict":
            mu_list = []

        if arg.num_steps == -1:
            while True:
                try:
                    feed = transformation_parameters(
                        arg, ctr, no_transform=(arg.mode == "predict")
                    )  # no transform if arg.visualize

                    # Ensure no geometric/appearance augmentation during prediction
                    if arg.mode == "predict":
                        feed.scal = 1.0
                        feed.tps_scal = 0.0
                        feed.rot_scal = 0.0
                        feed.off_scal = 0.0
                        feed.scal_var = 0.0
                        feed.augm_scal = 0.0

                    trf = {
                        scal: feed.scal,
                        tps_scal: feed.tps_scal,
                        coord_jitter: arg.coord_jitter if arg.mode == "train" else 0.0,
                        scal_var: feed.scal_var,
                        rot_scal: feed.rot_scal,
                        off_scal: feed.off_scal,
                        augm_scal: feed.augm_scal,
                    }

                    ctr += 1

                    if arg.mode == "train":
                        if np.mod(ctr, arg.summary_interval) == 0:
                            merged_summary = sess.run(merged, feed_dict=trf)
                            writer.add_summary(merged_summary, global_step=ctr)
                        # run optimization and fetch loss
                        _, loss = sess.run([model.optimize, model.loss], feed_dict=trf)
                        # CLI progress print
                        if np.mod(ctr, arg.print_interval) == 0:
                            elapsed = time.time() - start_time
                            steps_done = max(1, ctr)
                            steps_per_sec = steps_done / elapsed if elapsed > 0 else float("inf")
                            if arg.num_steps > 0:
                                remaining = max(0, arg.num_steps - ctr)
                                eta = remaining / steps_per_sec if steps_per_sec > 0 else float("inf")
                                eta_str = "ETA: {:.1f}s".format(eta)
                            else:
                                eta_str = ""
                            try:
                                loss_val = float(np.asarray(loss))
                                loss_str = "loss={:.6f}".format(loss_val)
                            except Exception:
                                loss_str = "loss={}".format(loss)
                            print(
                                "Step {} | {} | {:.2f} step/s {}".format(
                                    ctr, loss_str, steps_per_sec, eta_str
                                )
                            )
                        if np.mod(ctr, arg.save_interval) == 0:
                            saver.save(
                                sess,
                                os.path.join(
                                    model_save_dir, "saved_model", "save_net.ckpt"
                                ),
                                global_step=ctr,
                            )

                    elif arg.mode == "predict":
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

                        # save visualization images and collect only valid keypoints
                        v = int(valid)
                        save(img[:v, ...], mu[:v, ...], ctr, model_save_dir)
                        mu_list.append(mu[:v, ...])

                except tf.errors.OutOfRangeError:
                    print("End of training.")
                    break

        else:
            for i in range(arg.num_steps):
                try:
                    feed = transformation_parameters(
                        arg, ctr, no_transform=(arg.mode == "predict")
                    )  # no transform if arg.visualize

                    # Ensure no geometric/appearance augmentation during prediction
                    if arg.mode == "predict":
                        feed.scal = 1.0
                        feed.tps_scal = 0.0
                        feed.rot_scal = 0.0
                        feed.off_scal = 0.0
                        feed.scal_var = 0.0
                        feed.augm_scal = 0.0

                    trf = {
                        scal: feed.scal,
                        tps_scal: feed.tps_scal,
                        scal_var: feed.scal_var,
                        rot_scal: feed.rot_scal,
                        off_scal: feed.off_scal,
                        augm_scal: feed.augm_scal,
                    }

                    ctr += 1
                    if arg.mode == "train":
                        if np.mod(ctr, arg.summary_interval) == 0:
                            merged_summary = sess.run(merged, feed_dict=trf)
                            writer.add_summary(merged_summary, global_step=ctr)
                        # run optimization and fetch loss
                        _, loss = sess.run([model.optimize, model.loss], feed_dict=trf)
                        # CLI progress print
                        if np.mod(ctr, arg.print_interval) == 0:
                            elapsed = time.time() - start_time
                            steps_done = max(1, ctr)
                            steps_per_sec = steps_done / elapsed if elapsed > 0 else float("inf")
                            if arg.num_steps > 0:
                                remaining = max(0, arg.num_steps - ctr)
                                eta = remaining / steps_per_sec if steps_per_sec > 0 else float("inf")
                                eta_str = "ETA: {:.1f}s".format(eta)
                            else:
                                eta_str = ""
                            try:
                                loss_val = float(np.asarray(loss))
                                loss_str = "loss={:.6f}".format(loss_val)
                            except Exception:
                                loss_str = "loss={}".format(loss)
                            print(
                                "Step {} | {} | {:.2f} step/s {}".format(
                                    ctr, loss_str, steps_per_sec, eta_str
                                )
                            )
                        if np.mod(ctr, arg.save_interval) == 0:
                            saver.save(
                                sess,
                                os.path.join(
                                    model_save_dir, "saved_model", "save_net.ckpt"
                                ),
                                global_step=ctr,
                            )

                    elif arg.mode == "predict":
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
                    print("End of training.")
                    break
        if arg.mode == "predict":
            print("Saving predicted keypoints")
            if len(mu_list) > 0:
                mu_all = np.concatenate(mu_list, axis=0)
                np.savez_compressed(
                    os.path.join(model_save_dir, "keypoints_predicted_train.npz"),
                    keypoints_predicted=mu_all,
                )
            else:
                print("No keypoints collected.")

        print("Done with Training.")


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
