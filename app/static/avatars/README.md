# SignBridge avatar

Drop a rigged humanoid `.glb` file here named `signbridge.glb`:

```
app/static/avatars/signbridge.glb
```

## Get a free avatar (30 seconds)

1. Go to <https://readyplayer.me> and click **Create avatar** (no account
   required for the download).
2. Pick the look you want, finish the flow, and on the final screen click
   the **"..."** menu → **View .glb** (or copy the URL ending in `.glb`).
3. Either:
   - **Bundle it**: save the file as `signbridge.glb` in this folder, or
   - **Hot-link it**: in `index.html`, set
     `window.SIGNBRIDGE_AVATAR_URL = '<your-rpm-url>';` before
     `signbridge-avatar.js` loads.

Any rigged humanoid `.glb` with Mixamo-style bone names
(`Hips`, `Spine`, `LeftArm`, `RightForeArm`, `LeftHandIndex1`, …) will
work — RPM matches this out of the box, as does anything exported from
Mixamo.

## How it's driven

`signbridge-avatar.js` overrides `window.drawAuraAvatar` with a Three.js
scene. Each call gets the same MediaPipe Holistic landmark array the old
stick figure used (33×4 pose + 468×3 face + 21×3 left/right hands).
The landmarks are fed through Kalidokit (`Pose.solve`, `Face.solve`,
`Hand.solve`), which produces bone rotations + ARKit face morph values
that are applied to the loaded avatar each frame.

If the `.glb` is missing or any of the libraries (Three.js GLTFLoader,
Kalidokit) fail to load, the renderer falls back to the original 2D
stick figure so SignBridge keeps working.
