from bottle import run, post, request
from threading import Thread
from queue import Queue
import datetime
import requests
import shutil
import tempfile
from os.path import join
from model import predict

HOST = '0.0.0.0'
PORT = 22222
ENDPOINT = 'https://127.0.0.1/api/v0/finish'


def log(*arg):
    t = '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    print(t, *arg)


## job queue
jobQueue = Queue()


## worker
def worker():
    log('Running worker')
    while True:
        job = jobQueue.get(block=True)
        log('processing job:', job)
        try:
            process(job)
        except:
            pass
        log('finished job:', job)


def process(job):
    url = job['image_url']
    job_id = job['job_id']

    log('fetching image:', url)
    r = requests.get(url, verify=False, timeout=10, stream=True)
    log('return status:', r.status_code)
    if r.status_code == 200:
        tempdir = tempfile.gettempdir()
        path = join(tempdir, 'image')
        log('saving tmp image to', path)
        with open(path, 'wb') as f:
            r.rawdecode_content = True
            shutil.copyfileobj(r.raw, f)
        result = predict([path])
        log('reply', job_id)
        requests.post(ENDPOINT, json={
            'job_id': job_id,
            'result': result
        })
    else:
        ## post without result = failed
        request.post(ENDPOINT, json={
            'job_id': job_id
        })


## background thread for processing
Thread(target=worker).start()


@post('/')
def index():
    json = request.json
    log('received job:', json)
    jobQueue.put(json)


print(f'Listening on port {PORT}')
run(host=HOST, port=PORT, reloader=False)
