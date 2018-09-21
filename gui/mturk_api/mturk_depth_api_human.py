from boto.mturk.connection import MTurkConnection
from boto.mturk.question import ExternalQuestion
from boto.mturk.qualification import Qualifications, PercentAssignmentsApprovedRequirement, LocaleRequirement, NumberHitsApprovedRequirement
from boto.mturk.connection import MTurkRequestError

import os
import json
import time
import pickle
import datetime
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import skimage.io as io

#######################################################################
# HUMAN ANNOTATIONS SETUP

HUMAN_ANNOTATION_FILE = '/home/ubuntu/datasets/human3.6/annotations/human36m_train_17.json'
HUMAN_IMAGES_FOLDER = '/home/ubuntu/datasets/human3.6/images_17/'
HUMAN_IMAGES_SERVER_FOLDER = '/static/images/human_17/'

#######################################################################
# MECHANICAL TURK SETUP
 
_host = 'mechanicalturk.sandbox.amazonaws.com'
# _host = 'mechanicalturk.amazonaws.com'

#######################################################################
# CONSTANTS

STARTING_HIT = 1 
MAX_HITS = 200

NUMBER_HITS = MAX_HITS - STARTING_HIT + 1
NUMBER_HIT_ASSIGNMENTS = 5

HOST_DOMAIN = 'https://aws-ec2-cit-mronchi.org'
 
########################################################################
# MONGO CLIENT DATABASE SETUP

from pymongo import MongoClient

_mongo_client = MongoClient()
_mongo_db = _mongo_client.cocoa_depth_human36m

_mongo_coll_1 = _mongo_db.depth_amt_gui_data
_mongo_coll_2 = _mongo_db.keypoint_labels
_mongo_coll_3 = _mongo_db.depth_hit_id2human_subj_id
_mongo_coll_4 = _mongo_db.human_subj_id2depth_hit_id

_mongo_coll_5 = _mongo_db.interactions_amt_gui_workers
_mongo_coll_6 = _mongo_db.interactions_amt_gui_blocked_workers

def getHITType():
    # Changing this will add another hit type and might mess up later fetches...
    # Only change if you know what you are doing...

    _mtc = MTurkConnection( host = _host )

    _title = "Guess the Closest Part of a Person!"
    _description="Help us find out which body part of a person is closest to the camera that took the picture."
    _keywords = "person, people, image, images, object, objects, depth, comparisons, human3.6m"
    
    _reward = _mtc.get_price_as_price(0.1)
    _duration = 60 * 15
    _approval_delay = 60 * 60 * 24 * 10
        
    _qualifications = Qualifications()
    _qualifications.add(PercentAssignmentsApprovedRequirement('GreaterThanOrEqualTo', 98, required_to_preview=True))
    _qualifications.add(NumberHitsApprovedRequirement('GreaterThanOrEqualTo', 100, required_to_preview=True))
    _qualifications.add(LocaleRequirement('EqualTo', 'US', required_to_preview=True))

    return _mtc.register_hit_type(title=_title, description=_description, reward=_reward, duration=_duration, keywords=_keywords, approval_delay=_approval_delay, qual_req=_qualifications)


def createHITs( savePath = '/home/ubuntu/amt_guis/cocoa_depth/hits/human/', hit_name = '' ):
    setIds = range( STARTING_HIT, STARTING_HIT + NUMBER_HITS )

    mtc = MTurkConnection( host = _host )
    hits = []

    hitType = getHITType()[0]
    hitLifeTime = 60 * 60 * 24 * 7

    count = 0
    for setId in setIds:
        
        external_url = HOST_DOMAIN + '/cocoa_depth/human/' + str( setId )
        print external_url
        
        q = ExternalQuestion( external_url=external_url, frame_height=1000 )
        hit = mtc.create_hit(hit_type=hitType.HITTypeId,
                             question=q,
                             max_assignments = NUMBER_HIT_ASSIGNMENTS,
                             lifetime=hitLifeTime)

        hits.append( hit[0] )
        
        count += 1
        if count >= MAX_HITS:
            # pass is just a place holder
            pass
            
    if savePath == '':
        if 'MTURK_STORAGE_PATH' in os.environ:
            savePath = os.environ['MTURK_STORAGE_PATH']
        else:
            savePath == './'

    if hit_name == '':
        hit_name = 'cocoa_test_' + str(NUMBER_HITS) + '_DepthHITS'
   
    time_stamp = time.strftime( "%Y-%m-%d_%H-%M-%S" )
        
    filename = os.path.join( savePath, hit_name + '_' + time_stamp + ".pkl")
        
    print "Storing created hit data at %s" % (filename)
    with open(filename, 'wb') as f:
        pickle.dump( hits, f )


def deleteAllHits():
    # this function should probably take an input parameter 
    # of a pickle file with the hits to be disposed...
    mtc = MTurkConnection(host=_host)
    for hit in mtc.get_all_hits():
        mtc.disable_hit( hit.HITId )


def getReviewableHITs( verbose = True ):
    mtc = MTurkConnection( host = _host )
    hitType = getHITType()[0]
    
    page_size = 100
    # this gets the first page and allows to check how many other pages
    hits = mtc.get_reviewable_hits( page_size = page_size )

    total_pages = float( hits.TotalNumResults ) / page_size
    int_total = int( total_pages )
    if( total_pages - int_total > 0 ):
        total_pages = int_total + 1
    else:
        total_pages = int_total
            
    if verbose:
        print "Total Reviewable HITs: [%s]" % hits.TotalNumResults
        print "Total Number of Pages: [%i]" % total_pages
   
    # first page was already retrieved
    pn = 1
    if verbose:
        print " -> request page [%i]" % pn    

    while pn < total_pages:
        pn = pn + 1
        if verbose:
            print " -> request page [%i]" % pn        
        temp_hits = mtc.get_reviewable_hits( hit_type=hitType.HITTypeId, page_size=page_size, page_number=pn )
        # extend the hit list
        hits.extend(temp_hits)
        
    return hits

    
def getReviewableAssignments():
    mtc = MTurkConnection( host = _host )
    # note: if there are more than 100 assignments per hit the function
    # must be modified to retrieve all pages of the assignments
    page_size = 100
    
    _assignments = []

    #_num_hits = sum(1 for _ in mtc.get_all_hits())
    #print "Total Number of HITs: [%d]" %(_num_hits)
    
    _num_reviewable = 0
    _num_hits = 0
    print "Analyzed [%d] HITs" %(_num_hits+1)
    for hit in mtc.get_all_hits():
        _num_hits += 1
	if _num_hits % 500 == 0:
            print "Analyzed [%d] HITs" %_num_hits
        
        tmp_assign = [_assignment for _assignment in mtc.get_assignments( hit.HITId, page_size = page_size )]
        if len( tmp_assign ) == NUMBER_HIT_ASSIGNMENTS:
            _num_reviewable += 1 
        
        _assignments.extend( tmp_assign )
    
    print "Total Number of HITs: [%d]" %( _num_hits )
    print "Total Number of Assignments: [%d]" %( len(_assignments) )
    print "Total Number of Reviewavle HITs: [%d]" %( _num_reviewable )
    
    return _assignments    

       
def processAssignments( save = True, savePath = '/home/ubuntu/amt_guis/cocoa_depth/hits/human/', hit_name = '', verbose = False ):
    _mtc = MTurkConnection( host = _host )
    mtc_assignments = getReviewableAssignments()
    assignments_data = extractAssignmentData( mtc_assignments, verbose )
   

    # Simple filter for only the human3.6 assignments
    new_assignments_data = []
    for ass_data in assignments_data:
         _trials_results_dict = json.loads( ass_data['_trials_results'] )
         print _trials_results_dict[_trials_results_dict.keys()[0]].keys()
         if "_human_subj_id" not in _trials_results_dict[_trials_results_dict.keys()[0]].keys():
             continue
         new_assignments_data.append(ass_data)
    assignments_data = new_assignments_data
        
    num_assignments = len(assignments_data)

    # store assignments info here, for persistence
    # list containing all the assignments
    _all_assignments = []
    # list containing the assignments that were flagged by the turkers
    _flagged_assignments = []
    # list with the assignments that were rejected
    _rejected_assignments = []
    # list with the assignments that are not rejected nor flagged
    _good_assignments = []
    # list with the assignments were something inexpected on my side happened
    _error_assignments = []

    worker_ids = set()
    
    print "===================================================="
    print "Number of Assignments to analyze: [%d]" %(num_assignments)
    print "===================================================="
    
    count = 0
    for ass_data in assignments_data:
        count += 1
        if verbose:
            print " - Assignment [%d/%d]" %(count,num_assigments)
            
        worker_ids.add( ass_data['_worker_id'] )
        
        cleaned_data     = cleanAssignmentData( ass_data )
        _polished_data   = cleaned_data[0]
        _error           = cleaned_data[1]
        _hit_reject_flag = cleaned_data[2]
        _hit_flag        = cleaned_data[3]
        
        _all_assignments.append( _polished_data )
        if _error:
            _error_assignments.append( _polished_data )
        else:
            if _hit_reject_flag:
                _rejected_assignments.append( _polished_data )
            else:
                if _hit_flag:
                    _flagged_assignments.append( _polished_data )
                else:
                    _good_assignments.append( _polished_data )

    # print out some stats
    print "Distinct workers:               [%d]" % (len(worker_ids),)
    print "Total number of assignments:    [%d]" % (len(_all_assignments),)
    print "Rejected assignments:           [%d]" % (len(_rejected_assignments),)
    print "Flagged assignments:            [%d]" % (len(_flagged_assignments),)
    print "Good assignments:               [%d]" % (len(_good_assignments),)
    print "Error assignments:              [%d]" % (len(_error_assignments),)

    return_dict = {
        "_all_assignments":_all_assignments,
        "_rejected_assignments":_rejected_assignments,
        "_flagged_assignments":_flagged_assignments,
        "_good_assignments":_good_assignments,
        "_error_assignments":_error_assignments} 

    if save:
        if savePath == '':
            if 'MTURK_STORAGE_PATH' in os.environ:
                savePath = os.environ['MTURK_STORAGE_PATH']
            else:
                savePath == './'

        if hit_name == '':
            hit_name = 'cocoa_test_completed_' + str(len(_all_assignments)) + '_DepthHITS'
   
        time_stamp = time.strftime( "%Y-%m-%d_%H-%M-%S" )
        
        filename = os.path.join( savePath, hit_name + '_' + time_stamp + ".pkl")
        
        print "Storing created hit data at %s" % (filename)
        with open(filename, 'wb') as f:
            pickle.dump( return_dict, f )

    return return_dict


def cleanAssignmentData( ass_data ):
    _polished_data                  = {}
    _polished_data['worker_id']     = ass_data['_worker_id']
    _polished_data['worker_exp']    = ass_data['_worker_exp']
    _polished_data['assignment_id'] = ass_data['_assignment_id']
    _polished_data['hit_id']        = ass_data['_hit_id']
    _polished_data['response_time'] = ass_data['_hit_rt']
    _polished_data['hit_comment']   = ass_data['_hit_comment']
    _polished_data['hit_it']        = ass_data['_hit_it']
    _polished_data['gui_rating']    = ass_data['_gui_rating']

    _ass_human_subj_ids = _mongo_coll_3.find_one({'_amt_hit_id':_polished_data['hit_id']})['_human_subjs_ids']
    _polished_data['human_subj_ids'] = _ass_human_subj_ids

    _trials_results_dict = json.loads( ass_data['_trials_results'] )

    _ass_trials = []

    _error = False
    for key in _trials_results_dict.keys():
        _trial = _trials_results_dict[key]
	_depth = json.loads(_trial['_depth_str'])

	_trial_info = \
        { "depth": _depth,
         'response_time': _trial['_trial_rt'],
         'human_subj_id': _trial['_human_subj_id'] }
        res_coll_1 = _mongo_coll_1.find_one({ '_human_subj_id':_trial['_human_subj_id'] })
        _trial_info['img_id'] = res_coll_1['_human_img_id']
        _ass_trials.append(_trial_info)

    _polished_data['trials'] = _ass_trials

    return (_polished_data, _error, ass_data['_hit_reject_flag'], ass_data['_hit_flag'])
        

def extractAssignmentData( assignments, verbose = False ):
    all_assignments_data = []
    
    # These are the contents of the assignments for the depth HITs
    _worker_id = ''
    _worker_exp = 0
    _hit_id = 0
    _assignment_id = ''
    _gui_rating = ''
    _hit_comment = ''
    _hit_rt = 0
    _hit_it = 0
    _trials_results = ''
    _hit_depth_str  = ''
    _hit_reject_flag = False
    _hit_flag = False

    for ass in assignments:
        if verbose:
            print "===================================================="
            print "Content of Assignment: [%s]" % ass.AssignmentId
            print "===================================================="
        for question_form_answer in ass.answers[0]:
            key = question_form_answer.qid
            value = question_form_answer.fields

            if key == '_hit_id':
                _hit_id = int(value[0])
                if verbose:
                    print " - HIT ID: [%d]" % (_hit_id)
            elif key == '_assignment_id':
                _assignment_id = value[0]
                if verbose:
                    print " - Assignment ID: [%s]" % (_assignment_id)
            elif key == '_worker_id':
                _worker_id = value[0]
                if verbose:
                    print " - Worker ID: [%s]" % (_worker_id)
            elif key == '_worker_exp':
                _worker_exp = int(value[0])
                if verbose:
                    print " - Worker experience: [%d]" % (_worker_exp)
            elif key == '_gui_rating':
                _gui_rating = value[0]
                try: 
                    _gui_rating = int(_gui_rating)
                except ValueError:
                    _gui_rating = -1
                if verbose:
                    print " - GUI rating: [%d/10]" % (_gui_rating)         
            elif key == '_hit_comment':
                _hit_comment = value[0]
                if verbose:
                    print " - Assignment comment: [%s]" % (_hit_comment)
            elif key == '_hit_rt':
                _hit_rt = int(value[0])
                if verbose:
                    print " - Assignment response time: [%d]" % (_hit_rt)
            elif key == '_hit_it':
                _hit_it = int(value[0])
                if verbose:
                    print " - Assignment instruction time: [%d]" % (_hit_it)
            elif key == '_trials_results':
                _trials_results = value[0]
                if verbose:
                    print " - Assignment results: [%s]" % (_trials_results)    
            elif key == '_hit_depth_str':
                _hit_depth_str = value[0]
                if verbose:
                    print " - Assignment depth string: [%s]" % (_hit_depth_str)    
            elif key == '_hit_reject_flag':
                _hit_reject_flag = value[0]
                if str(_hit_reject_flag) == 'false':
                    _hit_reject_flag = False
                else:
                    _hit_reject_flag = True
                if verbose:
                    print " - Assignment reject flag: [%s]" % (str(_hit_reject_flag))
            elif key == '_hit_flag':
                _hit_flag = value[0]
                if _hit_flag == 'Yes':
                    _hit_flag = True
                else:
                    _hit_flag = False
                if verbose:
                    print " - Assignment information flag: [%s]" % (str(_hit_flag))    
            elif key == "_dataset":
                _dataset = value[0]
                if verbose:
                    print " - Assignment dataset: [%s]" % (_dataset) 
            else:
                print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                print "ERROR: unknown key [%r]" % (key,)
                print "Relevant info:"
                pprint(vars(_assignment))
                pprint(vars(question_form_answer))
                print "Exiting..."
                print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                return

        tmp_ass                          = {}
        tmp_ass['_worker_id']            = _worker_id
        tmp_ass['_worker_exp']           = _worker_exp
        tmp_ass['_hit_id']               = _hit_id
        tmp_ass['_assignment_id']        = _assignment_id
        tmp_ass['_gui_rating']           = _gui_rating
        tmp_ass['_hit_comment']          = _hit_comment
        tmp_ass['_hit_rt']               = _hit_rt
        tmp_ass['_hit_it']               = _hit_it
        tmp_ass['_trials_results']       = _trials_results
        tmp_ass['_hit_depth_str']        = _hit_depth_str
        tmp_ass['_hit_reject_flag']      = _hit_reject_flag
        tmp_ass['_hit_flag']             = _hit_flag

        all_assignments_data.append( tmp_ass )

    return all_assignments_data


def plotHITStatus( savePath = '/home/ubuntu/amt_guis/cocoa_depth/plots/', filename = 'time_info' ):
    pdf = PdfPages( savePath + filename + '.pdf')
    fig = plt.figure()
    plt.clf()

    page_size = 100
    ass_time_info_list = []
    
    mtc = MTurkConnection( host = _host )
    assignments = getReviewableAssignments()

    for ass in assignments:
        time_info = \
        {'AcceptTime':ass.AcceptTime,
        'SubmitTime':ass.SubmitTime,
        'ExecutionTime': [ question_form_answer.fields[0] for question_form_answer in ass.answers[0] if question_form_answer.qid == '_hit_rt' ][0] }
            
        ass_time_info_list.append( time_info )
            
    ass_time_info_list.sort(key=lambda x: datetime.datetime.strptime(x['AcceptTime'],'%Y-%m-%dT%H:%M:%SZ'))
    first_assignment = ass_time_info_list[0]
    ass_time_info_list.sort(key=lambda x: datetime.datetime.strptime(x['SubmitTime'],'%Y-%m-%dT%H:%M:%SZ'))
    last_assignment = ass_time_info_list[-1]

    time_since_beginning = int(( datetime.datetime.strptime(last_assignment['SubmitTime'],'%Y-%m-%dT%H:%M:%SZ') - datetime.datetime.strptime(first_assignment['AcceptTime'],'%Y-%m-%dT%H:%M:%SZ')).total_seconds())
    completed_percentage = []
    # time since beginning in one hour intervals
    time_range = range( 0, time_since_beginning + 3600, 3600 )

    for s in time_range:
        currently_completed = \
            [x for x in ass_time_info_list if datetime.datetime.strptime(x['SubmitTime'],'%Y-%m-%dT%H:%M:%SZ') < datetime.timedelta(seconds=s) + datetime.datetime.strptime(first_assignment['SubmitTime'],'%Y-%m-%dT%H:%M:%SZ')] 
        perc = len( currently_completed ) / float( NUMBER_HITS * NUMBER_HIT_ASSIGNMENTS )
        completed_percentage.append( perc )

    per_hour_completion_rate = len(ass_time_info_list) / float(time_since_beginning / 3600)
    #print per_hour_completion_rate
    
    hours_to_completion = ((NUMBER_HITS * NUMBER_HIT_ASSIGNMENTS) - len(ass_time_info_list)) / per_hour_completion_rate
    #print hours_to_completion

    plt.plot( time_range, completed_percentage )
   
    rows = ['Completed Assignments','Total Assignments','Hour Completion Rate','Hours to Completion']
    data = [["%d"%(len(ass_time_info_list))],["%d"%(NUMBER_HITS * NUMBER_HIT_ASSIGNMENTS)],["%.2f" % per_hour_completion_rate],["%.2f" % hours_to_completion]]
    
    plt.table(cellText=data,rowLabels=rows,loc='center',colWidths = [0.1]*3)
    
    plt.title('Per hour completion percentage')
    
    plt.xticks( time_range[0::10], [str(x/3600) for x in time_range[0::10]] )
    plt.yticks([0,0.2,0.4,0.6,0.8,1],['0%', '20%','40%','60%','80%','100%'])
    
    plt.ylabel('Completion Percentage')
    plt.xlabel('Hours since beginning of task')

    plt.grid()
    pdf.savefig()
    pdf.close()
    plt.close()


def payTurkersAssignments():
    _mtc = MTurkConnection( host = _host )

    rejected = 0
    approved = 0
    failed_rejected = 0
    failed_approved = 0
    
    failed_approved_list = []
    failed_rejected_list = []

    return_dict = processAssignments( save=False )

    # list with the assignments that are not rejected nor flagged
    _good_assignments = return_dict['_good_assignments']
    for ass in _good_assignments:
        try:
            _mtc.approve_assignment( ass['assignment_id'] )
            approved += 1
        except MTurkRequestError:
            failed_approved += 1
            failed_approved_list.append( ass )

    # list containing the assignments that were flagged by the turkers
    _flagged_assignments = return_dict['_flagged_assignments']            
    for ass in _flagged_assignments:
        try:
            _mtc.approve_assignment( ass['assignment_id'] )
            approved += 1
        except MTurkRequestError:
            failed_approved += 1
            failed_approved_list.append( ass )

    # list with the assignments were something inexpected on my side happened
    _error_assignments = return_dict['_error_assignments']
    for ass in _error_assignments:
        try:
            _mtc.approve_assignment( ass['assignment_id'] )
            approved += 1
        except MTurkRequestError:
            failed_approved += 1
            failed_approved_list.append( ass )
                
    # list with the assignments that were rejected
    _rejected_assignments = return_dict['_rejected_assignments']
    for ass in _rejected_assignments:
        try:
            _mtc.reject_assignment( ass['assignment_id'] )
            rejected += 1
        except MTurkRequestError:
            failed_rejected += 1
            failed_rejected_list.append( ass )
            
    print "Approved:        [%d]"%approved
    print "Rejected:        [%d]"%rejected
    print "Not Approved:    [%d]"%failed_approved
    print "Not Rejected:    [%d]"%failed_rejected 
    
    return (failed_approved_list, failed_rejected_list)
