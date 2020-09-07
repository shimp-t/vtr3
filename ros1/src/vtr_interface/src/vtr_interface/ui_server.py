#!/usr/bin/env python

import io
import os
import logging
osp = os.path

import numpy as np
import flask
import requests
from requests.exceptions import ConnectTimeout, ReadTimeout, RequestException

from vtr_interface import UI_ADDRESS, UI_PORT
from vtr_interface import utils
from vtr_interface.proto import Graph_pb2

import asrl__planning
from asrl__planning.msg import MissionStatus
from asrl__pose_graph.msg import GraphComponent

logging.basicConfig(level=logging.WARNING)

log = logging.getLogger('WebServer')
log.setLevel(logging.INFO)

# Web server config
app = flask.Flask(__name__,
                  static_folder="../../vtr_frontend/vtr-ui/build",
                  template_folder="../../vtr_frontend/vtr-ui/build",
                  static_url_path="")
app.config['DEBUG'] = True

app.config['CACHE'] = False  # map cache
app.config['CACHE_PATH'] = osp.abspath(
    osp.join(osp.dirname(__file__), '../../vtr_frontend/cache'))
app.config['PROTO_PATH'] = osp.abspath(
    osp.join(osp.dirname(__file__), '../../proto'))

app.secret_key = 'asecretekey'

app.logger.setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR)


@app.route("/")
def main_page():
  return flask.redirect("index.html")


@app.route('/proto/<path:proto_file>')
def proto_files(proto_file):
  return flask.send_from_directory(app.config['PROTO_PATH'], proto_file)


@app.route('/cache/tile/<s>/<x>/<y>/<z>')
def tile_cache(s, x, y, z):
  """Load google satellite map"""
  fname = x + '.jpg'
  fdir = osp.join(app.config['CACHE_PATH'], 'tile', z, y)
  fpath = osp.join(fdir, fname)

  if app.config['CACHE'] and osp.isfile(fpath):
    log.debug("Using cached tile {%s,%s,%s}", x, y, z)
    return flask.send_from_directory(fdir,
                                     fname,
                                     cache_timeout=60 * 60 * 24 * 30)
  # elif not app.config['OFFLINE']:
  headers = {
      'Accept': 'image/webp,image/*,*/*;q=0.8',
      'User-Agent': flask.request.user_agent.string
  }
  # url = 'https://khms' + s + '.googleapis.com/kh?v=199&hl=en-GB&x=' + x + '&y=' + y + '&z=' + z
  url = 'http://mt1.google.com/vt/lyrs=y&x=' + x + '&y=' + y + '&z=' + z

  try:
    res = requests.get(url, headers=headers, verify=False)
  except (ConnectTimeout, ReadTimeout) as e:
    log.warning('Tile {%s,%s,%s} timed out: %s', x, y, z, e.message)
    flask.abort(408)
  except ConnectionError as e:
    log.error('Tile {%s,%s,%s} coulnd\'t connect: %s', x, y, z, e.message)
    flask.abort(503)
  except HTTPError as e:
    log.error('Tile {%s,%s,%s} returned HTTP error %d', x, y, z,
              e.response.status_code)
    flask.abort(e.response.status_code)
  except RequestException as e:
    log.error('Something went really sideways on tile {%s,%s,%s}: %s', x, y, z,
              e.message)
    flask.abort(500)

  if res.ok:
    try:
      sio = io.BytesIO(res.content)
      if app.config['CACHE']:
        log.debug("Caching tile: {%s,%s,%s}", x, y, z)
        os.makedirs(fdir, exist_ok=True)
        img = Image.open(sio)
        img.save(fpath)
      else:
        log.debug("Proxying tile: {%s,%s,%s}", x, y, z)

      sio.seek(0)
      return flask.send_file(sio,
                             mimetype='image/jpeg',
                             cache_timeout=60 * 60 * 24 * 30)
    except Exception as e:
      log.error('Something went really sideways on tile {%s,%s,%s}: %s', x, y,
                z, e)
      flask.abort(500)
  else:
    log.warning("Tile {%s,%s,%s} did not exist on server", x, y, z)

  log.debug("Tile {%s,%s,%s} not in offline cache", x, y, z)
  flask.abort(404)


@app.route('/api/map/<seq>')
def get_map(seq):
  """API endpoint to get the full map"""
  graph = utils.get_graph(seq)

  proto_graph = Graph_pb2.Graph()
  proto_graph.seq = graph.seq
  proto_graph.stamp = graph.stamp.to_sec()
  proto_graph.root = graph.rootId

  if graph.seq < 0:
    return proto_graph.SerializeToString()

  if not graph.projected:
    raise RuntimeError(
        "Received an unprojected graph... What do I do with this?")

  for v in graph.vertices:
    vertex = proto_graph.vertices.add()
    vertex.id = v.id
    vertex.lat = v.T_projected.y
    vertex.lng = v.T_projected.x
    vertex.theta = v.T_projected.theta
    vertex.neighbours.extend(v.neighbours)

  for c in graph.components:
    if c.type == GraphComponent.PATH:
      component = proto_graph.paths.add()
    elif c.type == GraphComponent.CYCLE:
      component = proto_graph.cycles.add()
    else:
      raise RuntimeError(
          "Encountered unknown graph component type in UI Server")

    component.vertices.extend(c.vertices)

  proto_graph.branch.vertices.extend(graph.active_branch)
  proto_graph.junctions.extend(graph.junctions)

  gps_coords = np.array(
      [[v.T_projected.y, v.T_projected.x] for v in graph.vertices])

  if len(gps_coords > 0):
    mn = list(np.min(np.array(gps_coords), axis=0))
    mx = list(np.max(np.array(gps_coords), axis=0))
  elif len(graph.center) == 2:
    mn = graph.center
    mx = graph.center
  else:
    mn = [43.781596, -79.467298]
    mx = [43.782806, -79.464608]

  proto_graph.min_bnd.lat = mn[0]
  proto_graph.min_bnd.lng = mn[1]
  proto_graph.max_bnd.lat = mx[0]
  proto_graph.max_bnd.lng = mx[1]
  proto_graph.map_center.lat = (mn[0] + mx[0]) / 2
  proto_graph.map_center.lng = (mn[1] + mx[1]) / 2

  return proto_graph.SerializeToString()


@app.route('/api/init')
def init_state():
  """API endpoint to get the initial state of the robot/localization chain"""
  rclient = asrl__planning.remote_client()
  return flask.jsonify(vertex=rclient.trunk_vertex,
                       seq=rclient.path_seq,
                       tfLeafTrunk=rclient.T_leaf_trunk,
                       tfLeafTarget=rclient.T_leaf_target,
                       path=rclient.path,
                       covLeafTrunk=rclient.cov_leaf_trunk,
                       covLeafTarget=rclient.cov_leaf_target)


@app.route('/api/goal/all')
def get_goals():
  """API endpoint to get all goals"""
  rclient = asrl__planning.remote_client()
  status_text = {
      MissionStatus.PROCESSING: "PROCESSING",
      MissionStatus.PAUSED: "PAUSED",
      MissionStatus.PENDING_PAUSE: "PENDING_PAUSE",
      MissionStatus.EMPTY: "EMPTY"
  }.get(rclient.status, "")

  return flask.jsonify(goals=rclient.goals, status=status_text)


if __name__ == '__main__':
  log.info("Launching UI server.")

  # TODO: Server runs on all interfaces.  Can we assume a trusted network?
  app.run(threaded=True, host=UI_ADDRESS, port=UI_PORT, use_reloader=False)