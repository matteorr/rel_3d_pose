"""
Loads MTurk results and runs crowdsourcing code from here to combine them.

Change PATH_TO_CROWDSOURCING_CODE to it points to:
https://github.com/gvanhorn38/crowdsourcing

Treats each keypoint as independent, and creates a string: "im_id_kp0,kp1"

NOTE: a minor modification has been made to crowdsourcing.py in load()

    if data is None and fname is None:
        print("Error: Must specify a file name to load or a datset dictionary.")
        return False

    if fname is not None:
        self.fname = fname
        with open(fname) as f:
            data = json.load(f)

"""

import pickle
import json
from collections import Counter
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import user_study_utils as utils

PATH_TO_CROWDSOURCING_CODE = '../../../'
sys.path.append(PATH_TO_CROWDSOURCING_CODE)
from crowdsourcing.annotations.classification.binary import CrowdDatasetBinaryClassification


def resolve_labels(mturk_results):

    # convert into crowdsourcing format
    annos = []
    for res in mturk_results:
        ann = {}
        label = res['ann']
        if label == -1:
            label = 0

        ann['anno'] = {"label":label}
        ann['worker_id'] = res['w_id']
        ann['image_id'] = str(res['im_id']) + '_'+ str(res['kp0']) +','+ str(res['kp1'])
        annos.append(ann)

    raw_dataset = {"dataset":{}, "images":{}, "workers":{}, "annos":annos}


    #
    # Code from validate raw dataset
    # Compute some stats on the images, workers and annotations
    image_ids = [anno['image_id'] for anno in raw_dataset['annos']]
    worker_ids = [anno['worker_id'] for anno in raw_dataset['annos']]

    num_annos_per_image = Counter(image_ids)
    num_images = len(num_annos_per_image)
    avg_num_annos_per_image = np.mean(num_annos_per_image.values())
    median_num_annos_per_image = np.median(num_annos_per_image.values())

    num_annos_per_worker = Counter(worker_ids)
    num_workers = len(num_annos_per_worker)
    avg_num_annos_per_worker = np.mean(num_annos_per_worker.values())
    median_num_annos_per_worker = np.median(num_annos_per_worker.values())

    num_annotations = len(raw_dataset['annos'])
    anno_labels = [anno['anno']['label'] for anno in raw_dataset['annos']]
    num_yes_labels = sum(anno_labels)
    num_no_labels = len(anno_labels) - num_yes_labels

    # Print out the stats
    print "%d images" % (num_images,)
    print "%0.3f average annotations per image" % (avg_num_annos_per_image,)
    print "%d median annotations per image" % (median_num_annos_per_image,)
    print
    print "%d workers" % (num_workers,)
    print "%0.3f average annotations per worker" % (avg_num_annos_per_worker,)
    print "%d median annotations per worker" % (median_num_annos_per_worker,)
    print
    print "%d annotations" % (num_annotations,)
    print "%d annotations == 1" % (num_yes_labels,)
    print "%d annotations == 0" % (num_no_labels,)

    # Check to see if a worker provided multiple annotations on the same image
    image_id_worker_id_pairs = [(anno['image_id'], anno['worker_id']) for anno in raw_dataset['annos']]
    if len(set(image_id_worker_id_pairs)) != len(image_id_worker_id_pairs):
        print "\nWARNING: at least one worker labeled an image multiple times. These duplicate annotations should be removed.\n"
        image_worker_counts = Counter(image_id_worker_id_pairs)
        for ((image_id, worker_id), c) in image_worker_counts.most_common():
            if c > 1:
                print "Worker %s annotated image %s %d times" % (worker_id, image_id, c)


    # run merger
    full_dataset = CrowdDatasetBinaryClassification(
        computer_vision_predictor=None, # No computer vision
        estimate_priors_automatically=True, # Estimate pooled worker priors? Or leave them fixed?
        min_risk = 0.02, # The minimum risk for an image to be considered finished (i.e. 98% confident)
    )

    # Load in the worker annotations
    # NOTE I have changed the code from the repo so it accepts a dictionary
    try:
        full_dataset.load(data=raw_dataset)
    except:
        print("READ: >>>> Must change the crowdsourcing.py file as specified at the top of this file.")
        raise

    # Estimate the binary label for each image
    full_dataset.estimate_parameters(avoid_if_finished=False)

    # Get the finished annotations
    image_id_to_finished = full_dataset.check_finished_annotations(set_finished=True)

    num_finished = sum(image_id_to_finished.values())
    print "%d / %d (%0.2f%%) images are finished" % (num_finished, len(image_id_to_finished), 100. * float(num_finished) / len(image_id_to_finished))

    results = convert_results(full_dataset)
    return results


def convert_results(full_dataset):
    # get the labels and worker values
    combined = {}
    combined['images'] = {}
    combined['workers'] = {}
    for im_id in full_dataset.images:
        combined['images'][im_id] = full_dataset.images[im_id].encode()
        combined['images'][im_id]['label'] = full_dataset.images[im_id].y.encode()

        # split the file name up so we know the keypoints
        combined['images'][im_id]['raw_id'] = im_id.split('_')[0]
        combined['images'][im_id]['kp0'] = int(im_id.split('_')[1].split(',')[0])
        combined['images'][im_id]['kp1'] = int(im_id.split('_')[1].split(',')[1])

    for w in full_dataset.workers:
        combined['workers'][w] = full_dataset.workers[w].encode()

    return combined
