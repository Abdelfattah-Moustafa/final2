/**
 * drawAuraAvatar — Canvas-based neon skeleton renderer.
 * Auto-normalizes the skeleton to a consistent size and centers it.
 * @param {CanvasRenderingContext2D} ctx
 * @param {number[]|null} lm  flat landmark array
 * @param {number} width
 * @param {number} height
 */
function drawAuraAvatar(ctx, lm, width, height) {
  ctx.clearRect(0, 0, width, height);

  // Dark grid background
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 1;
  for (let i = 0; i < width; i += 40) {
    ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, height); ctx.stroke();
  }
  for (let i = 0; i < height; i += 40) {
    ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(width, i); ctx.stroke();
  }

  if (!lm || lm.length === 0) return;

  // ── Landmark layout from build_db.py ────────────────────────────────
  // Pose:      33 pts × 4 (x,y,z,vis) = 132 values → offset 0
  // Face:     468 pts × 3              = 1404 values → offset 132
  // LeftHand:  21 pts × 3              =   63 values → offset 1536
  // RightHand: 21 pts × 3              =   63 values → offset 1599
  // Total per frame: 1662 values
  const poseOffset = 0;
  const faceOffset = 33 * 4;             // 132
  const lhOffset   = faceOffset + 468 * 3; // 1536
  const rhOffset   = lhOffset + 21 * 3;    // 1599

  // ── Auto-normalization ──────────────────────────────────────────────
  // Gather all valid upper-body landmark positions (raw normalized coords)
  // to compute a bounding box, then scale/translate to center on canvas.
  const rawPts = [];

  // Collect pose points (key upper body: 0=nose, 11-16=arms, 23-24=hips)
  const keyPose = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24];
  for (const idx of keyPose) {
    const base = poseOffset + idx * 4;
    if (base + 1 >= lm.length) continue;
    const nx = lm[base], ny = lm[base + 1];
    if (nx && ny && !isNaN(nx) && !isNaN(ny) && nx > -0.5 && nx < 1.5 && ny > -0.5 && ny < 1.5) {
      rawPts.push({ x: nx, y: ny });
    }
  }

  // Collect hand wrist points
  for (const off of [lhOffset, rhOffset]) {
    const nx = lm[off], ny = lm[off + 1];
    if (nx && ny && !isNaN(nx) && !isNaN(ny) && nx > -0.5 && nx < 1.5 && ny > -0.5 && ny < 1.5) {
      rawPts.push({ x: nx, y: ny });
    }
  }

  // Collect face center (nose tip)
  {
    const base = faceOffset + 4 * 3; // face point 4 = nose tip
    if (base + 1 < lm.length) {
      const nx = lm[base], ny = lm[base + 1];
      if (nx && ny && !isNaN(nx) && !isNaN(ny)) {
        rawPts.push({ x: nx, y: ny });
      }
    }
  }

  if (rawPts.length < 3) return; // not enough points to render

  // Compute bounding box of collected points
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of rawPts) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }

  const bboxW = maxX - minX || 0.001;
  const bboxH = maxY - minY || 0.001;
  const bboxCx = (minX + maxX) / 2;
  const bboxCy = (minY + maxY) / 2;

  // Target: fill 70% of canvas, centered
  const padding = 0.15;
  const targetW = width * (1 - 2 * padding);
  const targetH = height * (1 - 2 * padding);
  const scale = Math.min(targetW / bboxW, targetH / bboxH);
  const offsetX = width / 2;
  const offsetY = height / 2;

  /**
   * Get screen-space point for a landmark, with auto-normalization.
   * Returns null if the landmark is missing (zero or NaN).
   */
  function getPt(idx, offset, stride = 3) {
    const base = offset + idx * stride;
    if (base + 1 >= lm.length) return null;
    const nx = lm[base];
    const ny = lm[base + 1];
    // Treat 0/NaN as "not tracked"
    if (!nx || !ny || isNaN(nx) || isNaN(ny)) return null;
    // Transform: center on bbox, scale uniformly, center on canvas
    return {
      x: (nx - bboxCx) * scale + offsetX,
      y: (ny - bboxCy) * scale + offsetY
    };
  }

  ctx.lineCap  = 'round';
  ctx.lineJoin = 'round';

  // --- Skeleton arms + torso ---
  ctx.lineWidth   = 5;
  ctx.strokeStyle = '#ffffff';
  [
    [11,13],[13,15],  // left arm
    [12,14],[14,16],  // right arm
    [11,12],          // shoulders
    [11,23],[12,24],  // torso sides
    [23,24],          // hips
  ].forEach(([a, b]) => {
    const p1 = getPt(a, poseOffset, 4);
    const p2 = getPt(b, poseOffset, 4);
    if (p1 && p2) {
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
    }
  });

  // Shoulder / wrist joint dots
  [11, 12, 15, 16, 23, 24].forEach(i => {
    const p = getPt(i, poseOffset, 4);
    if (p) {
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, Math.PI * 2); ctx.fill();
    }
  });

  // --- Hands ---
  function drawHand(offset, color) {
    // Only draw if the wrist is tracked — gives us a reliable anchor
    const wrist = getPt(0, offset);
    if (!wrist) return; // whole hand absent — skip entirely

    ctx.strokeStyle = color;
    ctx.fillStyle   = color;

    [1, 5, 9, 13, 17].forEach(start => {
      let prev = wrist;
      for (let i = start; i < start + 4; i++) {
        const curr = getPt(i, offset);
        if (curr && prev) {
          ctx.lineWidth = 2;
          ctx.beginPath(); ctx.moveTo(prev.x, prev.y); ctx.lineTo(curr.x, curr.y); ctx.stroke();
          ctx.beginPath(); ctx.arc(curr.x, curr.y, 3, 0, Math.PI * 2); ctx.fill();
        }
        prev = curr; // may be null — next segment will also skip
      }
    });
  }
  drawHand(lhOffset, '#ff00ff');  // left  — magenta
  drawHand(rhOffset, '#00ffff');  // right — cyan

  // --- Face ---
  function drawFace() {
    // Face silhouette: MediaPipe face mesh has points 0-468
    // Key outline points for face boundary (approx)
    const faceOutline = [
      10, 338, 297, 332, 284, 251, 389,  // left side
      356, 454, 323, 361, 288, 397, 365, // left to bottom
      379, 378, 400, 377, 152, 148, 176, 149, 150, 136, // bottom
      172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, // right side
      10 // close loop
    ];

    // Face outline in light cyan with lower opacity
    ctx.strokeStyle = 'rgba(86, 243, 255, 0.4)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < faceOutline.length; i++) {
      const pt = getPt(faceOutline[i], faceOffset);
      if (pt) {
        if (i === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      }
    }
    ctx.stroke();

    // Eyes — draw circles for eye regions
    ctx.fillStyle = 'rgba(86, 243, 255, 0.3)';
    
    // Left eye center (approximate)
    const leftEyeCenter = getPt(33, faceOffset);
    if (leftEyeCenter) {
      ctx.beginPath();
      ctx.arc(leftEyeCenter.x, leftEyeCenter.y, 12, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = 'rgba(86, 243, 255, 0.5)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Right eye center (approximate)
    const rightEyeCenter = getPt(263, faceOffset);
    if (rightEyeCenter) {
      ctx.beginPath();
      ctx.arc(rightEyeCenter.x, rightEyeCenter.y, 12, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = 'rgba(86, 243, 255, 0.5)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Nose line (simple line from top to bottom of nose)
    ctx.strokeStyle = 'rgba(122, 162, 255, 0.3)';
    ctx.lineWidth = 1;
    const noseTop = getPt(10, faceOffset);
    const noseTip = getPt(4, faceOffset);
    const noseBottom = getPt(195, faceOffset);
    
    if (noseTop && noseTip) {
      ctx.beginPath();
      ctx.moveTo(noseTop.x, noseTop.y);
      ctx.lineTo(noseTip.x, noseTip.y);
      ctx.stroke();
    }
    if (noseTip && noseBottom) {
      ctx.beginPath();
      ctx.moveTo(noseTip.x, noseTip.y);
      ctx.lineTo(noseBottom.x, noseBottom.y);
      ctx.stroke();
    }

    // Mouth outline (simple)
    ctx.strokeStyle = 'rgba(255, 100, 150, 0.3)';
    ctx.lineWidth = 1;
    const mouthLeft = getPt(61, faceOffset);
    const mouthRight = getPt(291, faceOffset);
    const mouthTop = getPt(164, faceOffset);
    const mouthBottom = getPt(18, faceOffset);
    
    if (mouthLeft && mouthRight && mouthTop && mouthBottom) {
      ctx.beginPath();
      ctx.ellipse(
        (mouthLeft.x + mouthRight.x) / 2,
        (mouthTop.y + mouthBottom.y) / 2,
        Math.abs(mouthRight.x - mouthLeft.x) / 2,
        Math.abs(mouthBottom.y - mouthTop.y) / 2,
        0, 0, Math.PI * 2
      );
      ctx.stroke();
    }
  }
  drawFace();
}

