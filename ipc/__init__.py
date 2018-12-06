import zmq as zmq
import threading
import string
import random
import sys


def tprint(msg):
    """like print, but won't get newlines confused with multiple threads"""
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()



def random_string(length):
  return ''.join(random.choice(string.ascii_letters) for m in range(length))




class Frontend:
  def __init__(self, context, url):
    self._socket = context.socket(zmq.ROUTER)
    self._socket.bind(url)
    self.url = url
    print("new frontend listening on {}".format(self.url))


  def get_socket(self):
    return self._socket




class Backend:
  def __init__(self, context, url):
    self._socket = context.socket(zmq.DEALER)
    self._socket.bind(url)
    self.url = url
    print("new backend listening on {}".format(self.url))

  
  def get_socket(self):
    return self._socket



class WorkerPool:
  def __init__(self, frontend_url, backend_url):
    self._context = zmq.Context()
    self._frontend = Frontend(self._context, frontend_url)
    self._backend = Backend(self._context, backend_url)

    self.workers = []


  def append(self, worker_thread):
    worker_thread.set_context(self._context)
    worker_thread.listen(self._backend.url)
    self.workers.append(worker_thread)

  def proxy(self):
    zmq.proxy(self._frontend.get_socket(), self._backend.get_socket())



class WorkerThread(threading.Thread):
  def __init__(self):
    threading.Thread.__init__(self)
    self._id = random_string(8)
    self._context = None
    self._listen_url = ""
    self._socket = None


  def set_context(self, context):
    self._context = context

  
  def listen(self, url):
    self._listen_url = url


  def connect(self):
    self._socket = self._context.socket(zmq.DEALER)
    self._socket.connect(self._listen_url)
    tprint("Thread {} Started".format(self._id))


  def close(self):
    self._socket.close()


  def receive(self):
    ident, text = self._socket.recv_multipart()
    tprint("Thread {} received {} from {}".format(self._id, text, ident))

    return (ident, text)

  
  def send_to(self, ident, data):
    return self._socket.send_multipart([ident, data])

