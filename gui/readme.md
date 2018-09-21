# Web Based Relative Depth Annotation Tool
This contains the code to deploy the annotation tool we used to collect data for the paper. 

## Setup
The GUI is run using python Flask backend with MongoDB as the database. Routing is done with nginx.


### MongoDB Setup
MongoDB is used to store the data during development and as a backup to AMT.

Install MongoDB from [here](https://docs.mongodb.com/manual/installation/). Create the folder where you want the db to live.
```
mkdir -p /data/db
```

To run the server, run `mongod --dbpath ./data/db`.

### Remote Setup

For the configuration of routes in `/etc/nginx/sites-enabled/default`, use the `default` file in the repo as an example.

## Starting the Server

### Local Server
Run the server.
```
python pythonServer_v6.py
```

### Remote Server
Start a screen and start the flask server.
```
screen -S cocoa_depth
sudo uwsgi --module pythonServer_v6 --callable app -s /tmp/uwsgi_mturk_cocoa_depth.sock
```

Leave the screen (`Ctrl+A+D`) and change the owner of the server socket.
```
sudo chown -R www-data:www-data /tmp/uwsgi_mturk_cocoa_depth.sock
```


### Querying Mongodb
Here's a couple code snippets to get started in ipython. The first code block here is for querying the Human3.6m data.
```python
import json
import sys
from pymongo import MongoClient
_mongo_client = MongoClient()

_mongo_db = _mongo_client.cocoa_depth_human36m

_mongo_coll_1 = _mongo_db.depth_amt_gui_data
_mongo_coll_2 = _mongo_db.keypoint_labels
_mongo_coll_3 = _mongo_db.depth_hit_id2human_subj_id
_mongo_coll_4 = _mongo_db.human_subj_id2depth_hit_id

_mongo_coll_5 = _mongo_db.depth_amt_gui_workers
_mongo_coll_6 = _mongo_db.depth_amt_gui_blocked_workers

_mongo_coll_7 = _mongo_db.depth_amt_gui_trials_results
_mongo_coll_8 = _mongo_db.depth_amt_gui_hits_results
```

This is querying the MSCOCO data.
```python
import json
import sys
from pymongo import MongoClient
_mongo_client = MongoClient()

_mongo_db = _mongo_client.cocoa_depth

_mongo_coll_1 = _mongo_db.depth_amt_gui_data
_mongo_coll_2 = _mongo_db.keypoint_labels
_mongo_coll_3 = _mongo_db.depth_hit_id2coco_subj_id
_mongo_coll_4 = _mongo_db.coco_subj_id2depth_hit_id

_mongo_coll_5 = _mongo_db.depth_amt_gui_workers
_mongo_coll_6 = _mongo_db.depth_amt_gui_blocked_workers

_mongo_coll_7 = _mongo_db.depth_amt_gui_trials_results
_mongo_coll_8 = _mongo_db.depth_amt_gui_hits_results
```

Useful code tidbits.
```python
# example of a drop
# _mongo_coll_7.drop()

_mongo_coll_8.find_one()

cursor = _mongo_coll_8.find({})
for document in cursor:
    # print document
    if document['_hit_comment'] != "":
        # print document['_hit_id']
        print document['_hit_comment']
```

## Amazon Mechanical Turk


### Mturk Functions
To create hits, run the `mturk_api` method `createHITS()` in ipython.
```python
from mturk_api import mturk_depth_api
mturk_depth_api.createHITs()
```
To delete hits, call the `deleteAllHits()` method.
```python
mturk_depth_api.deleteAllHits()
```

### Deploying to AMT checklist
* Make sure you have money in your account.
* Always test on sandbox first before deploying.
* change MAX_HITS in mturk_depth_api to 800 or 100 or whatever.
* Change the url setup for mechanical turk.
* Change the external_url in createHITs to appropriate route.
* Check reward price, title, and description, keywords. etc. 17 cents is too much. Can you Guess the Closest Thing in the Image? as a title. NUMBER_ASSIGNEMTNS would be 3.
* Uncomment the qualifications. Qualifications is for people, keep it to US or Canada. include a numberhitsapprovedrequirement to like 100. also include number of hits approve to 98%.
* Clear the mongodb coll_7 and coll_8.
* Change the url in the GUI html form between sandbox and normal. 
* Make sure to reload the mturk_api into ipython after making changes.

### Process results
* `processAssignments()` to process and package the results. It generates a pickle file.
* `getReviewableHITs()` gets all the reviewable hits (all assignments are complete).
* `getReviewableAssignments()` gets all the completed assignments.

### Reference
If you find our work useful in your research please cite our paper:  
```
@inproceedings{relativeposeBMVC18,
  title     = {It's all Relative: Monocular 3D Human Pose Estimation from Weakly Supervised Data},
  author    = {Ronchi, Matteo Ruggero and Mac Aodha, Oisin and Eng, Robert and Perona, Pietro},
  booktitle = {BMVC},
  year = {2018}
}
```
