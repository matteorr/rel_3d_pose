from flask import Flask, render_template, send_file, make_response, request, current_app, url_for
from jinja2 import Environment, PackageLoader

import json

from pymongo import MongoClient

########################################################################
# MONGO CLIENT DATABASE SETUP
_mongo_client = MongoClient()

##########################################################################################
#  FLASK INSTRUCTIONS
app = Flask(__name__)


@app.route("/cocoa_depth/lsp/<hit_id>", methods=['GET'])
def lsp_turker_task(hit_id):
    # http://www.vision.caltech.edu/~mronchi/data/LSP/images/
    _mongo_db = _mongo_client.cocoa_depth_lsp
    _mongo_coll_2 = _mongo_db.keypoint_labels
    _mongo_coll_3 = _mongo_db.depth_hit_id2lsp_subj_id

    res_kpts = _mongo_coll_2.find_one()
    res_kpts = json.dumps(res_kpts["_keypoint_labels"])
    res = _mongo_coll_3.find_one({"_amt_hit_id": int(hit_id)})
    print "You just requested the mturk page -> " + str(hit_id) + " " + str(res)

    return render_template(
        'TurkAnnotationGUI_v8.html',
        subj_id_list=str(res["_lsp_subjs_ids"]),
        keypoint_labels=res_kpts,
        dataset="lsp")


@app.route("/cocoa_depth/coco/<hit_id>", methods=['GET'])
def coco_turker_task(hit_id):
    # httt://www.vision.caltech.edu/~mronchi/data/LSP/images/
    _mongo_db = _mongo_client.cocoa_depth
    _mongo_coll_2 = _mongo_db.keypoint_labels
    _mongo_coll_3 = _mongo_db.depth_hit_id2coco_subj_id

    res_kpts = _mongo_coll_2.find_one()
    res_kpts = json.dumps(res_kpts["_keypoint_labels"])
    res = _mongo_coll_3.find_one({"_amt_hit_id": int(hit_id)})
    print "You just requested the mturk page -> " + str(hit_id) + " " + str(res)

    return render_template(
        'TurkAnnotationGUI_v6.html',
        subj_id_list=str(res["_coco_subjs_ids"]),
        keypoint_labels=res_kpts,
        dataset="coco")


@app.route("/cocoa_depth/human/<hit_id>", methods=['GET'])
def human_turker_task(hit_id):
    _mongo_db = _mongo_client.cocoa_depth_human36m
    _mongo_coll_2 = _mongo_db.keypoint_labels
    _mongo_coll_3 = _mongo_db.depth_hit_id2human_subj_id

    res_kpts = _mongo_coll_2.find_one()
    res_kpts = json.dumps(res_kpts["_keypoint_labels"])
    res = _mongo_coll_3.find_one({"_amt_hit_id": int(hit_id)})
    print "You just requested the mturk page -> " + str(hit_id) + " " + str(res)

    return render_template(
        'TurkAnnotationGUI_v8.html',
        subj_id_list=str(res["_human_subjs_ids"]),
        keypoint_labels=res_kpts,
        dataset="human")


@app.route('/cocoa_depth/lsp/GetTrialData/', methods=['POST'])
def lsp_GetTrialData():
    _mongo_db = _mongo_client.cocoa_depth_lsp
    _mongo_coll_1 = _mongo_db.depth_amt_gui_data

    _lsp_subj_id = request.json['_lsp_subj_id']
    print "_lsp_subj_id is " + str(_lsp_subj_id)

    res_1 = _mongo_coll_1.find_one({"_lsp_subj_id": _lsp_subj_id})
    return_dict = {}
    print "ok 1"
    return_dict["_image_keypoints"] = res_1["_image_keypoints"]
    print "ok 2"
    return_dict["_lsp_img_src"] = res_1["_lsp_img_src"].split('/')[-1]
    print "ok 3"
    return_dict["_keypoints_bbox"] = res_1[
        "_keypoints_bbox"] if "_keypoints_bbox" in res_1 else None
    print "ok 4"
    return_dict["_comps"] = res_1["_comps"]
    print "ok 5"

    print "-----------------------------"
    print return_dict

    return json.dumps(return_dict)


@app.route('/cocoa_depth/coco/GetTrialData/', methods=['POST'])
def coco_GetTrialData():
    _mongo_db = _mongo_client.cocoa_depth
    _mongo_coll_1 = _mongo_db.depth_amt_gui_data

    _coco_subj_id = request.json['_coco_subj_id']
    print "_coco_subj_id is " + str(_coco_subj_id)

    res_1 = _mongo_coll_1.find_one({"_coco_subj_id": _coco_subj_id})
    return_dict = {}
    print "ok 1"
    return_dict["_image_keypoints"] = res_1["_image_keypoints"]
    print "ok 2"
    return_dict["_coco_img_src"] = res_1["_coco_img_src"]
    print "ok 3"
    return_dict["_keypoints_bbox"] = res_1["_keypoints_bbox"]
    print "ok 4"

    print "-----------------------------"
    print return_dict

    return json.dumps(return_dict)


@app.route('/cocoa_depth/human/GetTrialData/', methods=['POST'])
def human_GetTrialData():
    _mongo_db = _mongo_client.cocoa_depth_human36m
    _mongo_coll_1 = _mongo_db.depth_amt_gui_data

    _human_subj_id = request.json['_human_subj_id']
    print "_human_subj_id is " + str(_human_subj_id)

    res_1 = _mongo_coll_1.find_one({"_human_subj_id": _human_subj_id})
    return_dict = {}
    print res_1
    print "ok 1"
    return_dict["_image_keypoints"] = res_1["_image_keypoints"]
    print "ok 2"
    return_dict["_human_img_src"] = res_1["_human_img_src"]
    print "ok 3"
    return_dict["_keypoints_bbox"] = res_1["_keypoints_bbox"]
    print "ok 4"
    return_dict["_comps"] = res_1["_comps"]
    print "ok 5"

    print "-----------------------------"
    print return_dict

    return json.dumps(return_dict)


@app.route('/cocoa_depth/lsp/PostTrialResults/', methods=['POST'])
def lsp_PostTrialResults():
    _mongo_db = _mongo_client.cocoa_depth_lsp
    _mongo_coll_7 = _mongo_db.depth_amt_gui_trials_results

    _lsp_subj_id = request.json["_lsp_subj_id"]
    print "annotated subject: " + str(_lsp_subj_id)

    _depth_str = request.json["_depth_str"]
    print "depth: " + _depth_str

    _assignment_id = request.json["_assignment_id"]
    print "_assignment_id: " + str(_assignment_id)

    _hit_id = request.json["_hit_id"]
    print "_hit_id: " + str(_hit_id)

    _worker_id = request.json["_worker_id"]
    print "_worker_id: " + str(_worker_id)

    _trial_rt = request.json["_trial_rt"]
    print "_trial_rt: " + str(_trial_rt)

    _trial_num = request.json["_trial_num"]
    print "_trial_num: " + str(_trial_num)

    # create a document in the collection for every annotated subject
    _mongo_hit = {
        "_lsp_subj_id": _lsp_subj_id,
        "_hit_id": _hit_id,
        "_assignment_id": _assignment_id,
        "_worker_id": _worker_id,
        "_trial_rt": _trial_rt,
        "_trial_num": _trial_num,
        "_depth_str": _depth_str,
    }

    # insert the document in the collection
    res = _mongo_coll_7.insert(_mongo_hit)

    print res
    return str(res)


@app.route('/cocoa_depth/coco/PostTrialResults/', methods=['POST'])
def coco_PostTrialResults():
    _mongo_db = _mongo_client.cocoa_depth
    _mongo_coll_7 = _mongo_db.depth_amt_gui_trials_results

    _coco_subj_id = request.json["_coco_subj_id"]
    print "annotated subject: " + str(_coco_subj_id)

    _depth_str = request.json["_depth_str"]
    print "depth: " + _depth_str

    _assignment_id = request.json["_assignment_id"]
    print "_assignment_id: " + str(_assignment_id)

    _hit_id = request.json["_hit_id"]
    print "_hit_id: " + str(_hit_id)

    _worker_id = request.json["_worker_id"]
    print "_worker_id: " + str(_worker_id)

    _trial_rt = request.json["_trial_rt"]
    print "_trial_rt: " + str(_trial_rt)

    _trial_num = request.json["_trial_num"]
    print "_trial_num: " + str(_trial_num)

    # create a document in the collection for every annotated subject
    _mongo_hit = {
        "_coco_subj_id": _coco_subj_id,
        "_hit_id": _hit_id,
        "_assignment_id": _assignment_id,
        "_worker_id": _worker_id,
        "_trial_rt": _trial_rt,
        "_trial_num": _trial_num,
        "_depth_str": _depth_str,
    }

    # insert the document in the collection
    res = _mongo_coll_7.insert(_mongo_hit)

    print res
    return str(res)


@app.route('/cocoa_depth/human/PostTrialResults/', methods=['POST'])
def human_PostTrialResults():
    _mongo_db = _mongo_client.cocoa_depth_human36m
    _mongo_coll_7 = _mongo_db.depth_amt_gui_trials_results

    _human_subj_id = request.json["_human_subj_id"]
    print "annotated subject: " + str(_human_subj_id)

    _depth_str = request.json["_depth_str"]
    print "depth: " + _depth_str

    _assignment_id = request.json["_assignment_id"]
    print "_assignment_id: " + str(_assignment_id)

    _hit_id = request.json["_hit_id"]
    print "_hit_id: " + str(_hit_id)

    _worker_id = request.json["_worker_id"]
    print "_worker_id: " + str(_worker_id)

    _trial_rt = request.json["_trial_rt"]
    print "_trial_rt: " + str(_trial_rt)

    _trial_num = request.json["_trial_num"]
    print "_trial_num: " + str(_trial_num)

    # create a document in the collection for every annotated subject
    _mongo_hit = {
        "_human_subj_id": _human_subj_id,
        "_hit_id": _hit_id,
        "_assignment_id": _assignment_id,
        "_worker_id": _worker_id,
        "_trial_rt": _trial_rt,
        "_trial_num": _trial_num,
        "_depth_str": _depth_str,
    }

    # insert the document in the collection
    res = _mongo_coll_7.insert(_mongo_hit)

    print res
    return str(res)


@app.route('/cocoa_depth/lsp/PostHITResults/', methods=['POST'])
def lsp_PostAssignmentResults():
    _mongo_db = _mongo_client.cocoa_depth_lsp
    _mongo_coll_8 = _mongo_db.depth_amt_gui_hits_results
    return PostAssigmentResults(_mongo_coll_8)


@app.route('/cocoa_depth/coco/PostHITResults/', methods=['POST'])
def coco_PostAssignmentResults():
    _mongo_db = _mongo_client.cocoa_depth
    _mongo_coll_8 = _mongo_db.depth_amt_gui_hits_results
    return PostAssigmentResults(_mongo_coll_8)


@app.route('/cocoa_depth/human/PostHITResults/', methods=['POST'])
def human_PostAssignmentResults():
    _mongo_db = _mongo_client.cocoa_depth_human36m
    _mongo_coll_8 = _mongo_db.depth_amt_gui_hits_results
    return PostAssigmentResults(_mongo_coll_8)


def PostAssigmentResults(_mongo_coll_8):
    _hit_id = request.json["_hit_id"]
    print "_hit_id: " + str(_hit_id)

    _assignment_id = request.json["_assignment_id"]
    print "_assignment_id: " + _assignment_id

    _worker_id = request.json["_worker_id"]
    print "_worker_id: " + _worker_id

    _worker_exp = request.json["_worker_exp"]
    print "_worker_exp: " + str(_worker_exp)

    _hit_it = request.json["_hit_it"]
    print "__hit_it: " + str(_hit_it)

    _hit_flag = request.json["_hit_flag"]
    if _hit_flag == 'Yes':
        _hit_flag = True
    else:
        _hit_flag = False
    print "_hit_flag: " + str(_hit_flag)

    _hit_rt = request.json["_hit_rt"]
    print "_hit_rt: " + str(_hit_rt)

    _gui_rating = request.json["_gui_rating"]
    try:
        _gui_rating = int(_gui_rating)
    except ValueError:
        _gui_rating = -1
    print "_gui_rating: " + str(_gui_rating)

    _hit_comment = request.json["_hit_comment"]
    print "_hit_comment: " + _hit_comment

    _trials_results = request.json["_trials_results"]
    print "_trials_results: " + _trials_results

    _hit_depth_str = request.json["_hit_depth_str"]
    print "_hit_depth_str: " + _hit_depth_str

    _hit_reject_flag = request.json["_hit_reject_flag"]
    print "_hit_reject_flag: " + str(_hit_reject_flag)

    # create a document in the collection for every annotated subject
    _mongo_hit = {
        "_hit_id": _hit_id,
        "_assignment_id": _assignment_id,
        "_worker_id": _worker_id,
        "_worker_exp": _worker_exp,
        "_hit_it": _hit_it,
        "_hit_flag": _hit_flag,
        "_hit_rt": _hit_rt,
        "_gui_rating": _gui_rating,
        "_hit_comment": _hit_comment,
        "_trials_results": _trials_results,
        "_hit_depth_str": _hit_depth_str,
        "_hit_reject_flag": _hit_reject_flag
    }

    # insert the document in the collection
    res = _mongo_coll_8.insert(_mongo_hit)

    print res
    return str(res)


@app.route('/cocoa_depth/lsp/IsNewWorker/', methods=['POST'])
def lsp_IsNewWorker():
    _mongo_db = _mongo_client.cocoa_depth_lsp
    _mongo_coll_5 = _mongo_db.depth_amt_gui_workers
    return IsNewWorker(_mongo_coll_5)


@app.route('/cocoa_depth/coco/IsNewWorker/', methods=['POST'])
def coco_IsNewWorker():
    _mongo_db = _mongo_client.cocoa_depth
    _mongo_coll_5 = _mongo_db.depth_amt_gui_workers
    return IsNewWorker(_mongo_coll_5)


@app.route('/cocoa_depth/human/IsNewWorker/', methods=['POST'])
def human_IsNewWorker():
    _mongo_db = _mongo_client.cocoa_depth_human36m
    _mongo_coll_5 = _mongo_db.depth_amt_gui_workers
    return IsNewWorker(_mongo_coll_5)


def IsNewWorker(_mongo_coll_5):
    worker_exp = 0

    _worker_id = request.json["_worker_id"]
    print "worker id: " + str(_worker_id)

    # look for a document in the collection of workers with that ID, if not present add it
    res = _mongo_coll_5.find_one({'_worker_id': _worker_id})
    if (bool(res) == False):
        # document not in the collection, insert in the collection
        res = _mongo_coll_5.insert({
            '_worker_id': _worker_id,
            '_worker_exp': 0
        })
    else:
        worker_exp = res['_worker_exp'] + 1
        _mongo_coll_5.update({
            '_worker_id': res['_worker_id']
        }, {
            '$set': {
                '_worker_exp': res['_worker_exp'] + 1
            },
            '$currentDate': {
                'lastModified': True
            }
        })

    print str(worker_exp)
    return str(worker_exp)


@app.route('/cocoa_depth/lsp/IsBlockedWorker/', methods=['POST'])
def lsp_IsBlockedWorker():
    _mongo_db = _mongo_client.cocoa_depth_lsp
    _mongo_coll_6 = _mongo_db.depth_amt_gui_blocked_workers
    return IsBlockedWorker(_mongo_coll_6)


@app.route('/cocoa_depth/coco/IsBlockedWorker/', methods=['POST'])
def coco_IsBlockedWorker():
    _mongo_db = _mongo_client.cocoa_depth
    _mongo_coll_6 = _mongo_db.depth_amt_gui_blocked_workers
    return IsBlockedWorker(_mongo_coll_6)


@app.route('/cocoa_depth/human/IsBlockedWorker/', methods=['POST'])
def human_IsBlockedWorker():
    _mongo_db = _mongo_client.cocoa_depth_human36m
    _mongo_coll_6 = _mongo_db.depth_amt_gui_blocked_workers
    return IsBlockedWorker(_mongo_coll_6)


def IsBlockedWorker(_mongo_coll_6):
    return_string = 'REJECT'

    _worker_id = request.json["_worker_id"]
    print "worker id: " + str(_worker_id)

    # look for a document in the collection of workers with that ID
    res = _mongo_coll_6.find_one({'_worker_id': _worker_id})
    if (bool(res) == False):
        return_string = 'OK'

    print return_string
    return return_string


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
