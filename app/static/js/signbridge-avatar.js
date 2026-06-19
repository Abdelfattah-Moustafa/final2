/* SignBridge 3D avatar renderer.
 *
 * Replaces the 2D neon stick-figure (drawAuraAvatar in app.js) with a
 * Three.js scene that loads a rigged Ready Player Me / Mixamo .glb
 * avatar and drives it from MediaPipe Holistic landmarks via Kalidokit.
 *
 * Dependencies (loaded as plain <script> tags before this file):
 *   - THREE (three.js r128, already loaded by index.html)
 *   - THREE.GLTFLoader
 *   - Kalidokit (UMD)
 *
 * Avatar source: by default we look for
 *   /static/avatars/signbridge.glb
 * Drop any Ready Player Me .glb (free at readyplayer.me) at that path.
 * To override the URL globally, set window.SIGNBRIDGE_AVATAR_URL before
 * this script runs.
 *
 * Public API (matches the old stick-figure renderer so callers don't change):
 *   window.drawAuraAvatar(ctx, flatLandmarks, width, height)
 *   - ctx:   2D canvas context of the original placeholder canvas
 *   - flatLandmarks: 1662-float array (33*4 pose + 468*3 face + 21*3 LH + 21*3 RH)
 *                   or null for the empty state
 *   - width / height: ignored (we follow the canvas's box)
 *
 * The function clears the 2D layer every call so text/progress overlays
 * drawn by the caller after this call still appear on top.
 */
(function () {
  'use strict';

  const AVATAR_URL = window.SIGNBRIDGE_AVATAR_URL || '/static/avatars/signbridge.glb';
  const renderers = new WeakMap();
  const fallback = window.drawAuraAvatar; // original stick figure (from app.js)

  function ready() {
    return typeof THREE !== 'undefined' &&
           typeof THREE.GLTFLoader !== 'undefined' &&
           typeof window.Kalidokit !== 'undefined';
  }

  function drawAvatar(ctx, lm, width, height) {
    if (!ctx || !ctx.canvas) return;
    // Always clear the 2D layer so caller's later fillText() shows clean.
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    if (!ready()) {
      // Libraries still loading — fall back so the user sees something.
      if (fallback) fallback(ctx, lm, width, height);
      return;
    }

    let r = renderers.get(ctx.canvas);
    if (!r) {
      try {
        r = new AvatarRenderer(ctx.canvas);
        renderers.set(ctx.canvas, r);
      } catch (e) {
        console.error('[SignBridge Avatar] init failed:', e);
        if (fallback) fallback(ctx, lm, width, height);
        return;
      }
    }
    r.update(lm);
  }

  // Replace the global. Function declarations from app.js are configurable
  // on window in browsers, so this reassignment wins.
  window.drawAuraAvatar = drawAvatar;
  window.drawSignBridgeAvatar = drawAvatar;

  // ───────────────────────────────────────────────────────────────
  class AvatarRenderer {
    constructor(canvas2d) {
      this.canvas2d = canvas2d;
      const parent = canvas2d.parentElement;
      if (!parent) throw new Error('avatar canvas has no parent');

      // Sibling WebGL canvas, behind the 2D label canvas.
      const glCanvas = document.createElement('canvas');
      glCanvas.className = 'sb-avatar-gl';
      glCanvas.style.cssText =
        'position:absolute;inset:0;width:100%;height:100%;' +
        'pointer-events:none;z-index:0;';
      parent.insertBefore(glCanvas, canvas2d);
      // Make the 2D canvas a transparent overlay so its labels render on top.
      canvas2d.style.background = 'transparent';
      canvas2d.style.zIndex = '2';
      this.glCanvas = glCanvas;

      this.renderer = new THREE.WebGLRenderer({
        canvas: glCanvas, alpha: true, antialias: true,
      });
      this.renderer.setPixelRatio(window.devicePixelRatio || 1);
      this.renderer.outputEncoding = THREE.sRGBEncoding;

      this.scene = new THREE.Scene();
      this.camera = new THREE.PerspectiveCamera(28, 1, 0.1, 100);

      // Soft 3-light setup — flattering for skin / hair.
      const hemi = new THREE.HemisphereLight(0xffffff, 0x222233, 0.85);
      this.scene.add(hemi);
      const key = new THREE.DirectionalLight(0xffffff, 0.85);
      key.position.set(2, 4, 3);
      this.scene.add(key);
      const rim = new THREE.DirectionalLight(0x88aaff, 0.4);
      rim.position.set(-2, 3, -2);
      this.scene.add(rim);

      this.bones = {};
      this.fingers = {};
      this.morphMesh = null;
      this.modelReady = false;
      this.lastRig = null;

      this._resize();
      this._ro = new ResizeObserver(() => this._resize());
      this._ro.observe(parent);

      this._loadModel();
      // Idle render loop — keeps the avatar visible & lerping smoothly
      // between landmark updates (which arrive at ~40fps).
      this._tick = this._tick.bind(this);
      requestAnimationFrame(this._tick);
    }

    _resize() {
      const rect = this.glCanvas.getBoundingClientRect();
      const w = Math.max(2, Math.round(rect.width));
      const h = Math.max(2, Math.round(rect.height));
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }

    _loadModel() {
      const loader = new THREE.GLTFLoader();
      loader.load(
        AVATAR_URL,
        (gltf) => this._onModelLoaded(gltf),
        undefined,
        (err) => {
          console.error('[SignBridge Avatar] glb load failed:', err);
          this._showMissingAvatarHint();
        }
      );
    }

    _onModelLoaded(gltf) {
      const model = gltf.scene;
      model.traverse((o) => {
        if (o.isSkinnedMesh) o.frustumCulled = false;
      });
      // Center the model and stand it on the floor (y=0).
      const bbox = new THREE.Box3().setFromObject(model);
      const size = bbox.getSize(new THREE.Vector3());
      const center = bbox.getCenter(new THREE.Vector3());
      model.position.x -= center.x;
      model.position.z -= center.z;
      model.position.y -= bbox.min.y;

      this.scene.add(model);
      this.model = model;

      // Frame the upper body — sign language lives in head-to-hands.
      const targetY = bbox.min.y + size.y * 0.78;
      const dist = size.y * 1.45;
      this.camera.position.set(0, targetY, dist);
      this.camera.lookAt(0, targetY - size.y * 0.05, 0);

      this._indexBones(model);
      this.modelReady = true;
    }

    _indexBones(root) {
      const dict = {};
      root.traverse((o) => {
        if (o.isBone) {
          // Strip Mixamo prefix and lowercase for forgiving lookup.
          const key = o.name.replace(/^mixamorig:?/i, '').toLowerCase();
          dict[key] = o;
        }
      });
      this.bones = dict;
      const b = (n) => dict[n] || null;

      this.body = {
        Hips:          b('hips'),
        Spine:         b('spine'),
        Chest:         b('spine1') || b('chest'),
        UpperChest:    b('spine2') || b('upperchest'),
        Neck:          b('neck'),
        Head:          b('head'),
        LeftUpperArm:  b('leftarm')      || b('leftupperarm'),
        RightUpperArm: b('rightarm')     || b('rightupperarm'),
        LeftLowerArm:  b('leftforearm')  || b('leftlowerarm'),
        RightLowerArm: b('rightforearm') || b('rightlowerarm'),
        LeftHand:      b('lefthand'),
        RightHand:     b('righthand'),
      };
      const finger = (side, name, n) =>
        b(`${side}hand${name}${n}`) || b(`${side}${name}${n}`);
      const fingerGroup = (side) => ({
        ThumbProximal:     finger(side, 'thumb', 1),
        ThumbIntermediate: finger(side, 'thumb', 2),
        ThumbDistal:       finger(side, 'thumb', 3),
        IndexProximal:     finger(side, 'index', 1),
        IndexIntermediate: finger(side, 'index', 2),
        IndexDistal:       finger(side, 'index', 3),
        MiddleProximal:    finger(side, 'middle', 1),
        MiddleIntermediate:finger(side, 'middle', 2),
        MiddleDistal:      finger(side, 'middle', 3),
        RingProximal:      finger(side, 'ring', 1),
        RingIntermediate:  finger(side, 'ring', 2),
        RingDistal:        finger(side, 'ring', 3),
        LittleProximal:    finger(side, 'pinky', 1) || finger(side, 'little', 1),
        LittleIntermediate:finger(side, 'pinky', 2) || finger(side, 'little', 2),
        LittleDistal:      finger(side, 'pinky', 3) || finger(side, 'little', 3),
      });
      this.fingers = { Left: fingerGroup('left'), Right: fingerGroup('right') };

      // Find the mesh with ARKit blendshapes (RPM heads have these).
      root.traverse((o) => {
        if (o.isMesh && o.morphTargetDictionary && o.morphTargetInfluences) {
          // RPM has multiple morph-enabled meshes; prefer the one with
          // mouthOpen / jawOpen since that's what we drive.
          const d = o.morphTargetDictionary;
          if (d.jawOpen !== undefined || d.mouthOpen !== undefined || d.eyeBlinkLeft !== undefined) {
            this.morphMesh = o;
          } else if (!this.morphMesh) {
            this.morphMesh = o;
          }
        }
      });
    }

    update(flat) {
      if (!this.modelReady) return;
      if (!flat || !flat.length) {
        this.lastRig = null;
        return;
      }
      this.lastRig = this._solve(flat);
    }

    _solve(lm) {
      const K = window.Kalidokit;
      const POSE_OFF = 0, POSE_STRIDE = 4, POSE_N = 33;
      const FACE_OFF = POSE_N * POSE_STRIDE;
      const FACE_N = 468;
      const LH_OFF = FACE_OFF + FACE_N * 3;
      const RH_OFF = LH_OFF + 21 * 3;

      const pose3D = new Array(POSE_N);
      const pose2D = new Array(POSE_N);
      let poseValid = false;
      for (let i = 0; i < POSE_N; i++) {
        const o = POSE_OFF + i * POSE_STRIDE;
        const x = lm[o]     || 0;
        const y = lm[o + 1] || 0;
        const z = lm[o + 2] || 0;
        const v = lm[o + 3];
        const vis = (v === undefined ? 1 : v);
        pose3D[i] = { x, y, z, visibility: vis };
        pose2D[i] = { x, y, visibility: vis };
        if (x || y) poseValid = true;
      }

      const face = new Array(FACE_N);
      let faceValid = false;
      for (let i = 0; i < FACE_N; i++) {
        const o = FACE_OFF + i * 3;
        const x = lm[o]     || 0;
        const y = lm[o + 1] || 0;
        const z = lm[o + 2] || 0;
        face[i] = { x, y, z };
        if (x || y) faceValid = true;
      }

      const unpackHand = (base) => {
        const out = new Array(21);
        let valid = false;
        for (let i = 0; i < 21; i++) {
          const o = base + i * 3;
          const x = lm[o]     || 0;
          const y = lm[o + 1] || 0;
          const z = lm[o + 2] || 0;
          out[i] = { x, y, z };
          if (x || y) valid = true;
        }
        return valid ? out : null;
      };
      const leftHand  = unpackHand(LH_OFF);
      const rightHand = unpackHand(RH_OFF);

      const opts = { runtime: 'mediapipe', video: null };
      const poseRig = poseValid ? K.Pose.solve(pose3D, pose2D, opts) : null;
      const faceRig = faceValid ? K.Face.solve(face, { ...opts, smoothBlink: true }) : null;
      const lhRig   = leftHand  ? K.Hand.solve(leftHand,  'Left')  : null;
      const rhRig   = rightHand ? K.Hand.solve(rightHand, 'Right') : null;
      return { poseRig, faceRig, lhRig, rhRig };
    }

    _tick() {
      requestAnimationFrame(this._tick);
      if (!this.modelReady) {
        this.renderer.render(this.scene, this.camera);
        return;
      }
      this._applyRig(0.4);
      this.renderer.render(this.scene, this.camera);
    }

    _applyRig(lerp) {
      const rig = this.lastRig;
      if (!rig) return;

      const setEuler = (bone, e) => {
        if (!bone || !e) return;
        bone.rotation.x += ((e.x || 0) - bone.rotation.x) * lerp;
        bone.rotation.y += ((e.y || 0) - bone.rotation.y) * lerp;
        bone.rotation.z += ((e.z || 0) - bone.rotation.z) * lerp;
      };

      const p = rig.poseRig;
      const body = this.body;
      if (p) {
        setEuler(body.Spine, p.Spine);
        // Split spine rotation across the three RPM spine bones for a
        // softer curve than slamming it all onto one joint.
        if (body.Chest && p.Spine) {
          body.Chest.rotation.x += ((p.Spine.x || 0) * 0.5 - body.Chest.rotation.x) * lerp;
          body.Chest.rotation.y += ((p.Spine.y || 0) * 0.5 - body.Chest.rotation.y) * lerp;
          body.Chest.rotation.z += ((p.Spine.z || 0) * 0.5 - body.Chest.rotation.z) * lerp;
        }
        setEuler(body.LeftUpperArm,  p.LeftUpperArm);
        setEuler(body.RightUpperArm, p.RightUpperArm);
        setEuler(body.LeftLowerArm,  p.LeftLowerArm);
        setEuler(body.RightLowerArm, p.RightLowerArm);
        setEuler(body.LeftHand,  p.LeftHand);
        setEuler(body.RightHand, p.RightHand);
      }

      // Head tilt from the face solver beats the pose-derived neck.
      if (rig.faceRig && rig.faceRig.head) {
        setEuler(body.Head, rig.faceRig.head);
      }

      const applyHand = (side, hand) => {
        if (!hand) return;
        const f = this.fingers[side];
        const prefix = side; // 'Left' or 'Right'
        const names = [
          'ThumbProximal','ThumbIntermediate','ThumbDistal',
          'IndexProximal','IndexIntermediate','IndexDistal',
          'MiddleProximal','MiddleIntermediate','MiddleDistal',
          'RingProximal','RingIntermediate','RingDistal',
          'LittleProximal','LittleIntermediate','LittleDistal',
        ];
        for (const n of names) setEuler(f[n], hand[prefix + n]);
      };
      applyHand('Left',  rig.lhRig);
      applyHand('Right', rig.rhRig);

      // ARKit-style face morphs (RPM half/full-body avatars expose these).
      if (this.morphMesh && rig.faceRig) {
        const dict = this.morphMesh.morphTargetDictionary;
        const infl = this.morphMesh.morphTargetInfluences;
        const eyeL = 1 - (rig.faceRig.eye?.l ?? 1);
        const eyeR = 1 - (rig.faceRig.eye?.r ?? 1);
        const mouthY = Math.max(0, Math.min(1, rig.faceRig.mouth?.y ?? 0));
        const setMorph = (name, v) => {
          const i = dict[name];
          if (i !== undefined) infl[i] = (infl[i] || 0) + (v - (infl[i] || 0)) * lerp;
        };
        setMorph('eyeBlinkLeft', eyeL);
        setMorph('eyeBlinkRight', eyeR);
        setMorph('mouthOpen', mouthY);
        setMorph('jawOpen', mouthY * 0.85);
      }
    }

    _showMissingAvatarHint() {
      const parent = this.glCanvas.parentElement;
      if (!parent || parent.querySelector('.sb-avatar-missing')) return;
      const div = document.createElement('div');
      div.className = 'sb-avatar-missing';
      div.style.cssText =
        'position:absolute;inset:0;display:grid;place-items:center;' +
        'text-align:center;padding:20px;color:#aaa;z-index:1;' +
        'font:13px/1.5 Inter,system-ui,sans-serif;background:rgba(0,0,0,.35);';
      div.innerHTML =
        'Avatar model missing.<br>' +
        '<span style="font-size:11px;opacity:.75">Drop a Ready Player Me ' +
        '<code>.glb</code> at <code>/static/avatars/signbridge.glb</code></span>';
      parent.appendChild(div);
    }
  }
})();
