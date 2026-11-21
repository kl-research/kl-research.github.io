// H2 gait demo core: MuJoCo (WASM) model + observation + MLP policy + 50 Hz control loop.
// No DOM access here, so the same file runs in the page and in Node tests.

function wrapPi(a) {
  a = (a + Math.PI) % (2 * Math.PI);
  if (a < 0) a += 2 * Math.PI;
  return a - Math.PI;
}

// Policy network: observation normalizer followed by an ELU MLP, weights as one float32 buffer.
export class MlpPolicy {
  constructor(meta, buffer) {
    const w = new Float32Array(buffer);
    const blk = (name) => {
      const [off, n] = meta.layout[name];
      return w.subarray(off, off + n);
    };
    this.dims = meta.dims;
    this.mean = blk('mean');
    this.std = blk('std');
    this.layers = [];
    for (let i = 0; i < this.dims.length - 1; i++) {
      this.layers.push({ w: blk('w' + i), b: blk('b' + i), nin: this.dims[i], nout: this.dims[i + 1] });
    }
    this.bufs = this.dims.map((n) => new Float32Array(n));
  }

  // obs: raw (un-normalized) observations. Returns a view that stays valid until the next call.
  forward(obs) {
    let x = this.bufs[0];
    for (let i = 0; i < x.length; i++) x[i] = (obs[i] - this.mean[i]) / this.std[i];
    const last = this.layers.length - 1;
    for (let l = 0; l <= last; l++) {
      const { w, b, nin, nout } = this.layers[l];
      const y = this.bufs[l + 1];
      for (let o = 0, row = 0; o < nout; o++, row += nin) {
        let s = b[o];
        for (let i = 0; i < nin; i++) s += w[row + i] * x[i];
        y[o] = l < last && s < 0 ? Math.expm1(s) : s; // ELU on hidden layers
      }
      x = y;
    }
    return x;
  }
}

const ROOT = '/h2';
function mkdirp(FS, p) {
  let cur = '';
  for (const part of p.split('/').filter(Boolean)) {
    cur += '/' + part;
    try { FS.mkdir(cur); } catch (e) { /* exists */ }
  }
}

// Write the robot meshes into MuJoCo's in-memory file system once; every scene XML refers to them.
export function installMeshes(mujoco, meshIndex, meshBuf) {
  const FS = mujoco.FS;
  mkdirp(FS, ROOT + '/assets');
  const meshes = new Uint8Array(meshBuf);
  for (const [rel, [off, len]] of Object.entries(meshIndex)) {
    const path = ROOT + '/assets/' + rel;
    mkdirp(FS, path.slice(0, path.lastIndexOf('/')));
    FS.writeFile(path, meshes.subarray(off, off + len));
  }
}

export class H2Sim {
  // robot: robot facts shared by all scenes (joint names, default pose, action scale, gait period ...).
  // scene: { name, scene_xml, sim: {decimation, step_dt}, terrain? }.
  constructor(mujoco, robot, scene) {
    this.mj = mujoco;
    this.scene = scene;
    mujoco.FS.writeFile(`${ROOT}/${scene.name}.xml`, scene.scene_xml);
    this.model = mujoco.MjModel.from_xml_path(`${ROOT}/${scene.name}.xml`);
    this.data = new mujoco.MjData(this.model);

    const m = this.model;
    const OBJ = mujoco.mjtObj;
    const id = (type, name) => {
      const i = mujoco.mj_name2id(m, type.value, name);
      if (i < 0) throw new Error('模型里找不到: ' + name);
      return i;
    };
    this.nj = robot.joint_names.length;
    this.qadr = new Int32Array(this.nj);
    this.vadr = new Int32Array(this.nj);
    this.actId = new Int32Array(this.nj);
    const qposadr = m.jnt_qposadr, dofadr = m.jnt_dofadr;
    robot.joint_names.forEach((n, k) => {
      const j = id(OBJ.mjOBJ_JOINT, robot.joint_prefix + n);
      this.qadr[k] = qposadr[j];
      this.vadr[k] = dofadr[j];
      this.actId[k] = id(OBJ.mjOBJ_ACTUATOR, robot.joint_prefix + n);
    });
    this.gyroAdr = m.sensor_adr[id(OBJ.mjOBJ_SENSOR, robot.imu_gyro_sensor)];
    this.rootBody = id(OBJ.mjOBJ_BODY, robot.root_body);
    this.terrainBody = id(OBJ.mjOBJ_BODY, 'terrain');
    this.defaultPos = Float64Array.from(robot.default_joint_pos);
    this.actionScale = robot.action_scale;
    this.decimation = scene.sim.decimation;
    this.stepDt = scene.sim.step_dt;
    this.period = robot.gait_period;
    this.fallAngle = robot.fall_angle;
    this.standZ = m.key_qpos[2];           // base height of the standing keyframe above flat ground

    this.lastAction = new Float32Array(this.nj);
    this.cmd = new Float64Array(3);        // command actually fed to the policy (vx, vy, wz), body frame
    this.spawn = [0, 0];
    this.headingTarget = 0;
    this.stepCount = 0;
    this.groundZ = 0;
    this.policy = null;
    this.scan = null;
    this._gid = new mujoco.IntBuffer(1);
    this._nrm = new mujoco.DoubleBuffer(3);
    this._pnt = [0, 0, 0];
    this._down = [0, 0, -1];
    this._terrainOnly = [1, 0, 0, 0, 0, 0];
  }

  dispose() {
    this._gid.delete(); this._nrm.delete();
    this.data.delete(); this.model.delete();
  }

  // scanCfg: null for policies without a height scan, else { offsets:[x0,y0,x1,y1,...], max_distance, groups, frame_body }.
  setPolicy(policy, scanCfg) {
    this.policy = policy;
    this.obs = new Float32Array(policy.dims[0]);
    this.scan = null;
    if (scanCfg) {
      const b = this.mj.mj_name2id(this.model, this.mj.mjtObj.mjOBJ_BODY.value, scanCfg.frame_body);
      this.scan = {
        offsets: Float64Array.from(scanCfg.offsets), n: scanCfg.offsets.length / 2, max: scanCfg.max_distance, body: b,
        groups: [0, 1, 2, 3, 4, 5].map((g) => (scanCfg.groups.includes(g) ? 1 : 0)),
        heights: new Float32Array(scanCfg.offsets.length / 2),
      };
    }
    this.lastAction.fill(0);
  }

  // Height of the terrain surface under (x, y), ignoring the robot.
  groundHeight(x, y) {
    const p = this._pnt;
    p[0] = x; p[1] = y; p[2] = 60;
    const d = this.mj.mj_ray(this.model, this.data, p, this._down, this._terrainOnly, true, -1, this._gid, this._nrm);
    return d < 0 ? null : 60 - d;
  }

  // Put the robot in its standing keyframe at (x, y), feet on the local terrain.
  reset(x, y) {
    const { mj, model, data } = this;
    if (x !== undefined) { this.spawn[0] = x; this.spawn[1] = y; }
    mj.mj_resetDataKeyframe(model, data, 0);
    mj.mj_forward(model, data);
    let top = null;
    for (const [dx, dy] of [[0, 0], [0.12, 0.12], [0.12, -0.12], [-0.12, 0.12], [-0.12, -0.12], [0.2, 0], [-0.2, 0]]) {
      const h = this.groundHeight(this.spawn[0] + dx, this.spawn[1] + dy);
      if (h !== null && (top === null || h > top)) top = h;
    }
    this.groundZ = top === null ? 0 : top;
    data.qpos[0] = this.spawn[0]; data.qpos[1] = this.spawn[1];
    data.qpos[2] = this.standZ + this.groundZ + (this.groundZ !== 0 ? 0.01 : 0);
    mj.mj_forward(model, data);
    this.lastAction.fill(0);
    this.stepCount = 0;
    this.headingTarget = this.yaw();
    this.fallen = false;
  }

  yaw() {
    const q = this.data.qpos;
    const w = q[3], x = q[4], y = q[5], z = q[6];
    return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  }

  tilt() {
    const q = this.data.qpos;
    const x = q[4], y = q[5];
    return Math.acos(Math.max(-1, Math.min(1, 1 - 2 * (x * x + y * y))));
  }

  // Base linear velocity in the body frame (what the command is tracked against), plus yaw rate.
  bodyVelocity(out) {
    const q = this.data.qpos, v = this.data.qvel;
    const w = q[3], x = q[4], y = q[5], z = q[6];
    const vx = v[0], vy = v[1], vz = v[2];
    out[0] = (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y + w * z) * vy + 2 * (x * z - w * y) * vz;
    out[1] = 2 * (x * y - w * z) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z + w * x) * vz;
    out[2] = this.data.sensordata[this.gyroAdr + 2];
    return out;
  }

  // Height scan as in training: a grid of rays cast straight down from the pelvis height, laid out in the
  // pelvis' yaw frame. Rays may also hit the robot's own legs (the pelvis body itself is excluded).
  _heightScan(o, k) {
    const { mj, model, data } = this, s = this.scan;
    const xp = data.xpos, xm = data.xmat, b = s.body;
    const px = xp[3 * b], py = xp[3 * b + 1], pz = xp[3 * b + 2];
    let c = xm[9 * b], sn = xm[9 * b + 3];
    const n = Math.hypot(c, sn) || 1;
    c /= n; sn /= n;
    const p = this._pnt, off = s.offsets;
    p[2] = pz;
    for (let i = 0; i < s.n; i++) {
      const lx = off[2 * i], ly = off[2 * i + 1];
      p[0] = px + c * lx - sn * ly;
      p[1] = py + sn * lx + c * ly;
      const d = mj.mj_ray(model, data, p, this._down, s.groups, true, b, this._gid, this._nrm);
      const h = d < 0 || d > s.max ? s.max : d;
      s.heights[i] = h;
      o[k++] = h / s.max;
    }
    return k;
  }

  // One 50 Hz control step. user = [vx, vy, wz]; headingHold mirrors the heading command used in training:
  // with no turn input, wz = clip(0.5 * (target_heading - heading)).
  controlStep(user, headingHold) {
    const { mj, model, data } = this;
    const qpos = data.qpos, qvel = data.qvel, sens = data.sensordata, ctrl = data.ctrl;
    const cmd = this.cmd;
    cmd[0] = user[0]; cmd[1] = user[1]; cmd[2] = user[2];
    const yaw = this.yaw();
    if (!headingHold || Math.abs(user[2]) > 1e-3) {
      this.headingTarget = yaw;
    } else {
      cmd[2] = Math.max(-1, Math.min(1, 0.5 * wrapPi(this.headingTarget - yaw)));
    }

    const o = this.obs;
    let k = 0;
    o[k++] = sens[this.gyroAdr]; o[k++] = sens[this.gyroAdr + 1]; o[k++] = sens[this.gyroAdr + 2];
    const w = qpos[3], x = qpos[4], y = qpos[5], z = qpos[6];
    o[k++] = -2 * (x * z - w * y);          // gravity direction in the body frame
    o[k++] = -2 * (y * z + w * x);
    o[k++] = -(1 - 2 * (x * x + y * y));
    o[k++] = cmd[0]; o[k++] = cmd[1]; o[k++] = cmd[2];
    if (Math.hypot(cmd[0], cmd[1], cmd[2]) < 0.1) {
      o[k++] = 0; o[k++] = 0;               // standing: gait clock off
    } else {
      const g = ((this.stepCount * this.stepDt) % this.period) / this.period;
      o[k++] = Math.sin(2 * Math.PI * g); o[k++] = Math.cos(2 * Math.PI * g);
    }
    const nj = this.nj;
    for (let j = 0; j < nj; j++) o[k++] = qpos[this.qadr[j]] - this.defaultPos[j];
    for (let j = 0; j < nj; j++) o[k++] = qvel[this.vadr[j]];
    for (let j = 0; j < nj; j++) o[k++] = this.lastAction[j];
    if (this.scan) k = this._heightScan(o, k);

    const a = this.policy.forward(o);
    for (let j = 0; j < nj; j++) {
      this.lastAction[j] = a[j];
      ctrl[this.actId[j]] = this.defaultPos[j] + this.actionScale * a[j];
    }
    for (let s = 0; s < this.decimation; s++) mj.mj_step(model, data);
    mj.mj_forward(model, data);             // as in training: sensors and poses match the state the policy sees next
    this.stepCount++;
    if (this.tilt() > this.fallAngle || qpos[2] < this.groundZ - 6) this.fallen = true;
    return this.fallen;
  }

  // Velocity kick on the base, like the push disturbance used in training (world-frame m/s).
  push(dvx, dvy) {
    this.data.qvel[0] += dvx;
    this.data.qvel[1] += dvy;
  }

  get time() { return this.stepCount * this.stepDt; }
}
