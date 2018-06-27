from flask import jsonify
from flask import Blueprint, request

# michaniki async app
from ...celeryapp import michaniki_celery_app

blueprint = Blueprint('remote_tasks', __name__)

@blueprint.route('/info', methods=['POST'])
def task_info():
    """
    check the progress of the remote tasks by its id
    """
    task_ids = request.get_json()
    task_results = []
    
    for each_id in task_ids:
        this_task = michaniki_celery_app.AsyncResult(each_id)
        this_status = str(this_task.state)
        
        if this_status == "SUCCESS":
            # get the training and validation acc
            this_res = this_task.get()
        
            task_results.append({
                "remote_task_id": each_id,
                "status": this_status,
                "final training accuracy": float(this_res[0]),
                "final validation accuracy": float(this_res[1]),
                })
        else:
            # FAILD or PENDING
            task_results.append({
                "remote_task_id": each_id,
                "status": this_status,
                })
            
    return jsonify(
        {"Tasks Status": task_results}
        )
    
@blueprint.route('/cancel', methods=['GET'])
def cancel_task():
    task_id = request.args.get('remote_task_id')
    michaniki_celery_app.control.revoke(task_id, terminate=True)
    
    return jsonify(
        {
         "Tasks Status": {
             "remote_task_id": task_id,
             "status": "REVOKED"
             }
        })