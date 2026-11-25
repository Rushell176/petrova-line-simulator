struct Camera {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    time: vec4<f32>,
};
@group(0) @binding(0) var<uniform> cam: Camera;

@group(1) @binding(0) var aurora_tex: texture_cube<f32>;
@group(1) @binding(1) var aurora_samp: sampler;

// ---------------- hull ----------------

struct HullVIn {
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) col: vec4<f32>,
};

struct HullVOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) wpos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) col: vec4<f32>,
};

@vertex
fn vs_hull(in: HullVIn) -> HullVOut {
    var out: HullVOut;
    out.clip = cam.view_proj * vec4<f32>(in.pos, 1.0);
    out.wpos = in.pos;
    out.nrm = in.nrm;
    out.col = in.col;
    return out;
}

@fragment
fn fs_hull(in: HullVOut) -> @location(0) vec4<f32> {
    let pj = cam.time.y;
    let n = normalize(in.nrm);
    let v = normalize(cam.cam_pos.xyz - in.wpos);
    // Projector off: the ship is lit only by the green planet ahead.
    // Projector on: the pink light of the line takes over.
    let l = normalize(mix(vec3<f32>(0.10, 0.35, 1.0), vec3<f32>(0.45, 0.65, -0.60), pj));
    let diff = max(dot(n, l), 0.0);
    let rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
    let hv = normalize(l + v);
    let spec = pow(max(dot(n, hv), 0.0), 42.0);

    var col = in.col.rgb * mix(vec3<f32>(0.40, 1.0, 0.32), vec3<f32>(1.0), pj)
        * (mix(0.015, 0.07, pj) + mix(0.30, 0.70, pj) * diff);
    col += mix(vec3<f32>(0.25, 0.85, 0.22), vec3<f32>(1.0, 0.20, 0.30), pj) * rim * mix(0.16, 0.08, pj);
    col += mix(vec3<f32>(0.55, 1.0, 0.45), vec3<f32>(1.0, 0.30, 0.38), pj) * spec * mix(0.08, 0.35, pj);
    col += mix(vec3<f32>(0.004, 0.010, 0.003), vec3<f32>(0.032, 0.0055, 0.011), pj);

    let d = length(cam.cam_pos.xyz - in.wpos);
    let fog = 1.0 - exp(-d * 0.024);
    col = mix(col, mix(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.105, 0.019, 0.038), pj), fog);
    return vec4<f32>(col, 1.0);
}

// ---------------- particles ----------------

struct PVIn {
    @location(0) center: vec3<f32>,
    @location(1) size: f32,
    @location(2) seed: f32,
    @location(3) speed: f32,
    @location(4) tint: vec3<f32>,
    @builtin(vertex_index) vi: u32,
};

struct PVOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) tint: vec3<f32>,
    @location(2) bright: f32,
    @location(3) bokeh: f32,
};

@vertex
fn vs_particles(in: PVIn) -> PVOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>(0.5, -0.5),
        vec2<f32>(0.5, 0.5),
        vec2<f32>(-0.5, -0.5),
        vec2<f32>(0.5, 0.5),
        vec2<f32>(-0.5, 0.5),
    );

    let t = cam.time.x;
    var c = in.center;

    c.z = c.z - in.speed * (t + in.seed * 57.0);
    // Only a slight lateral drift: everything should advance down the line
    // axis instead of wandering side to side.
    c.x += sin(t * 0.25 + in.seed * 19.0) * (0.12 + in.size * 0.22);
    c.y += cos(t * 0.21 + in.seed * 23.0) * (0.10 + in.size * 0.18);

    // Wrap the field around the viewer on every axis so space is full in all
    // directions and there is no visible start (or end) of the stream.
    let half = vec3<f32>(34.0, 26.0, 48.0);
    var rel = c - cam.cam_pos.xyz;
    rel = rel - round(rel / (2.0 * half)) * (2.0 * half);
    let edge = (1.0 - smoothstep(0.82, 1.0, abs(rel.x) / half.x))
        * (1.0 - smoothstep(0.82, 1.0, abs(rel.y) / half.y))
        * (1.0 - smoothstep(0.82, 1.0, abs(rel.z) / half.z));
    c = cam.cam_pos.xyz + rel;

    let vpos = cam.view * vec4<f32>(c, 1.0);
    let corner = corners[in.vi];
    let depth = -vpos.z;
    let is_bokeh = in.size > 0.09;

    // Each smear lives on its own cycle: it fades out to nothing, stays gone a
    // while, swells back in.
    let rate = 1.55 + in.seed * 1.60;
    let wave = 0.5 + 0.5 * sin(t * rate + in.seed * 61.0);
    let life = smoothstep(0.04, 0.55, wave);

    var s: f32;
    var vis: f32;
    if (is_bokeh) {
        // Cap the disc size against its distance so a smear can drift right
        // through the camera and past the eye, keeping a fixed angular size
        // instead of ballooning over the whole screen.
        s = min(in.size, max(depth, 0.12) * 0.38);
        vis = smoothstep(0.12, 0.60, depth);
    } else {
        // Specks also keep a sane angular size when one drifts past the eye.
        s = min(
            max(in.size, max(depth, 0.0) * 0.0011),
            max(depth, 0.02) * 0.30,
        ) * smoothstep(0.03, 0.15, depth);
        vis = 1.0;
    }
    // Bokeh quads carry a margin around the disc for the outer glow.
    let halo = select(1.0, 3.2, is_bokeh);
    let p = vpos + vec4<f32>(corner * s * halo, 0.0, 0.0);

    var out: PVOut;
    out.clip = cam.proj * p;
    out.corner = corner * halo;
    out.tint = in.tint;

    let dist = max(depth, 0.0001);
    let tw = 0.55 + 0.45 * pow(0.5 + 0.5 * sin(t * (0.3 + in.seed * 1.5) + in.seed * 61.0), 2.0);
    let atten = exp(-dist * 0.020);
    var bri: f32;
    if (is_bokeh) {
        let near_dim = 0.55 + 0.45 * smoothstep(0.3, 3.0, depth);
        bri = tw * atten * near_dim * 0.78 * life;
    } else {
        // Specks drift past the eye too; dim them up close so a nearby grain
        // never flares into a pale ball.
        let near_dim = 0.35 + 0.65 * smoothstep(0.5, 3.0, depth);
        bri = tw * atten * near_dim * 0.82;
    }
    out.bright = bri * vis * edge * cam.time.y;
    out.bokeh = select(0.0, 1.0, is_bokeh);
    return out;
}

@fragment
fn fs_particles(in: PVOut) -> @location(0) vec4<f32> {
    let r = length(in.corner);
    if (r > 1.60) {
        discard;
    }

    var col: vec3<f32>;
    if (in.bokeh > 0.5) {
        // Out-of-focus light: a hot core bleeding into a soft body and a wide
        // coloured spill. No hard rim, so each smear reads as glowing light
        // instead of a flat coin.
        let core = exp(-r * r * 24.0);
        let body = exp(-r * r * 4.5);
        let fall = max(r - 0.44, 0.0);
        let halo_a = (0.55 * exp(-fall * fall * 24.0) + 0.50 * exp(-fall * fall * 2.6))
            * (1.0 - smoothstep(1.15, 1.55, r));
        let hot_tint = mix(in.tint, vec3<f32>(1.0, 0.55, 0.62), 0.55);
        let core_tint = mix(in.tint, vec3<f32>(1.0, 0.70, 0.76), 0.40);
        col = (in.tint * body * 0.50
            + core_tint * core * 1.05
            + hot_tint * halo_a * 0.60) * in.bright;
    } else {
        // Soft gaussian speck with a feathered edge, never a solid ball.
        let a = exp(-r * r * 7.0) * (1.0 - smoothstep(0.42, 0.5, r));
        col = in.tint * in.bright * 0.75 * a;
    }
    return vec4<f32>(col, 1.0);
}

// ---------------- sky: stars + aurora ----------------

struct SkyVIn {
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) col: vec4<f32>,
};

struct SkyVOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) dir: vec3<f32>,
};

@vertex
fn vs_sky(in: SkyVIn) -> SkyVOut {
    var out: SkyVOut;
    out.clip = cam.view_proj * vec4<f32>(in.pos, 1.0);
    out.dir = normalize(in.pos - cam.cam_pos.xyz);
    return out;
}

fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

fn vnoise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let w = f * f * (3.0 - 2.0 * f);
    let a = hash31(i);
    let b = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let c = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let d = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let e = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let g = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let h = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let k = hash31(i + vec3<f32>(1.0, 1.0, 1.0));
    let x0 = mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
    let x1 = mix(mix(e, g, w.x), mix(h, k, w.x), w.y);
    return mix(x0, x1, w.z);
}

fn fbm3(p: vec3<f32>) -> f32 {
    var v = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i = 0; i < 4; i++) {
        v += amp * vnoise3(q);
        q = q * 2.07 + vec3<f32>(13.7, 7.3, 3.1);
        amp *= 0.5;
    }
    return v;
}

fn star_layer(d: vec3<f32>, scale: f32, thresh: f32, t: f32) -> vec3<f32> {
    let p = d * scale;
    let cell = floor(p);
    let f = fract(p);
    let h = hash31(cell);
    let lit = step(thresh, h);
    let off = vec3<f32>(
        hash31(cell + 13.1),
        hash31(cell + 57.7),
        hash31(cell + 91.3),
    ) * 0.8 + 0.1;
    let core = smoothstep(0.16, 0.0, length(f - off));
    let tw = 0.55 + 0.45 * sin(t * 0.8 + h * 73.0);
    return vec3<f32>(1.0, 0.85 + 0.1 * h, 0.9) * core * lit * tw * (0.30 + 0.75 * h);
}

// A folded sheet of light: a noisy great-circle band around `axis`, with a
// bright edge, fine fibres run along it and slow breathing folds.
fn curtain(d: vec3<f32>, axis: vec3<f32>, width: f32, t: f32, seed: f32) -> f32 {
    let wob = vec3<f32>(
        fbm3(d * 1.6 + vec3<f32>(seed, 0.0, t * 0.012)),
        fbm3(d * 1.6 + vec3<f32>(5.1, seed + 1.7, 8.3)),
        fbm3(d * 1.6 + vec3<f32>(2.9, 9.4, seed + 4.1)),
    );
    let dw = normalize(d + (wob - 0.5) * 0.55);
    let s = dot(dw, axis);
    let w = width * (0.6 + 0.8 * fbm3(dw * 2.5 + seed));
    let band = exp(-s * s * w);
    let fold = 0.50 + 0.50 * smoothstep(-0.30, 0.20, s);
    let fiber = 0.35 + 0.65 * pow(vnoise3(dw * 24.0 + vec3<f32>(0.0, seed, t * 0.04)), 2.0);
    let breakup = smoothstep(0.38, 0.78, fbm3(dw * 4.0 + seed * 3.1 + 31.0));
    return band * fold * fiber * breakup;
}

@fragment
fn fs_sky(in: SkyVOut) -> @location(0) vec4<f32> {
    let t = cam.time.x;
    let pj = cam.time.y;
    let d = normalize(in.dir);
    let el = d.y;

    var col = vec3<f32>(0.0012, 0.0004, 0.0006);

    // The planet the projected light comes from: a marbled green gas giant with
    // rust storms, hanging in otherwise empty black space.
    let pdir = normalize(vec3<f32>(0.0, 0.24, 1.0));
    let pa = dot(d, pdir);
    let limb = smoothstep(0.72, 0.80, pa);
    var open_sky = 1.0;
    if (limb > 0.0) {
        let q = d * 3.4;
        let w1 = fbm3(q + vec3<f32>(0.0, 0.0, t * 0.006));
        let w2 = fbm3(q * 2.1 + w1 * 2.2 + 7.0);
        let m = fbm3(q * 1.6 + w2 * 2.6 + 3.0);
        let bands = 0.5 + 0.5 * sin(d.y * 9.0 + m * 6.0);
        var pc = mix(vec3<f32>(0.020, 0.115, 0.022), vec3<f32>(0.075, 0.32, 0.060), bands * 0.6 + m * 0.4);
        pc = mix(pc, vec3<f32>(0.16, 0.48, 0.12), smoothstep(0.55, 0.85, m) * 0.6);
        let rust = smoothstep(0.62, 0.80, w2);
        pc = mix(pc, vec3<f32>(0.30, 0.10, 0.025), rust * 0.85);
        let sh = 0.18 + 0.82 * smoothstep(-0.45, 0.55, dot(d, normalize(vec3<f32>(0.45, 0.55, 0.70))));
        pc *= sh;
        pc *= 0.55 + 0.45 * smoothstep(0.72, 0.92, pa);
        // The planet dissolves as the projection takes over, so no green
        // residue sits behind the light field — and its occlusion mask fades
        // with it, letting the aurora fill the disc instead of a black hole.
        // Dissolve runs on the slow channel (time.z), lagging the light field.
        let pfade = 1.0 - smoothstep(0.02, 0.90, cam.time.z);
        col += pc * limb * pfade;
        open_sky = 1.0 - limb * pfade;
    }

    col += (star_layer(d, 150.0, 0.030, t) * 0.75 + star_layer(d, 380.0, 0.016, t) * 0.45)
        * pj * open_sky;

    // Deep space with a faint red haze hugging the horizon line.
    let hor = pow(clamp(1.0 - abs(el), 0.0, 1.0), 10.0);
    col += vec3<f32>(0.024, 0.004, 0.008) * hor * pj * open_sky;
    col += vec3<f32>(0.014, 0.002, 0.005) * smoothstep(0.0, -0.35, el) * pj * open_sky;

    // The aurora curtains are expensive, so they are baked into a cubemap
    // (one face refreshed per couple of frames) and simply sampled here.
    col += textureSample(aurora_tex, aurora_samp, d).rgb * pj * open_sky;

    return vec4<f32>(col, 1.0);
}

// Total aurora radiance for a view direction; shared by the cubemap bake.
fn aurora_radiance(d: vec3<f32>, t: f32) -> vec3<f32> {
    // Aurora curtains: folded sheets of pink-red light scattered all around,
    // each with its own orientation (horizontal, vertical, diagonal, below).
    let ax1 = normalize(vec3<f32>(0.18, 0.92, 0.34));
    let ax2 = normalize(vec3<f32>(0.88, 0.22, 0.42));
    let ax3 = normalize(vec3<f32>(0.40, -0.60, 0.69));
    let ax4 = normalize(vec3<f32>(0.66, 0.58, -0.48));
    var aur = curtain(d, ax1, 6.5, t, 0.0) * 0.90;
    aur += curtain(d, ax2, 9.0, t, 3.7) * 0.80;
    aur += curtain(d, ax3, 11.0, t, 7.9) * 0.70;
    aur += curtain(d, ax4, 14.0, t, 11.3) * 0.55;
    aur = 1.0 - exp(-aur * 2.0);

    let hot = smoothstep(0.08, 0.85, aur);
    var col = mix(vec3<f32>(0.36, 0.02, 0.07), vec3<f32>(1.0, 0.30, 0.38), hot) * aur * 1.25;
    col += vec3<f32>(1.0, 0.55, 0.58) * pow(hot, 3.0) * 0.20;
    return col;
}

struct BakeOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) dir: vec3<f32>,
};

@vertex
fn vs_skybake(@builtin(vertex_index) vi: u32, @builtin(instance_index) face: u32) -> BakeOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let fc = p[vi];
    var dir: vec3<f32>;
    switch face {
        case 0u: { dir = vec3<f32>(1.0, fc.y, -fc.x); }
        case 1u: { dir = vec3<f32>(-1.0, fc.y, fc.x); }
        case 2u: { dir = vec3<f32>(fc.x, 1.0, -fc.y); }
        case 3u: { dir = vec3<f32>(fc.x, -1.0, fc.y); }
        case 4u: { dir = vec3<f32>(fc.x, fc.y, 1.0); }
        default: { dir = vec3<f32>(-fc.x, fc.y, -1.0); }
    }
    var out: BakeOut;
    out.clip = vec4<f32>(fc, 0.0, 1.0);
    out.dir = dir;
    return out;
}

@fragment
fn fs_skybake(in: BakeOut) -> @location(0) vec4<f32> {
    return vec4<f32>(aurora_radiance(normalize(in.dir), cam.time.x), 1.0);
}

// ---------------- tether ----------------

struct LineVIn {
    @location(0) pos: vec3<f32>,
    @location(1) col: vec4<f32>,
};

struct LineVOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) col: vec4<f32>,
};

@vertex
fn vs_line(in: LineVIn) -> LineVOut {
    var out: LineVOut;
    out.clip = cam.view_proj * vec4<f32>(in.pos, 1.0);
    out.col = in.col;
    return out;
}

@fragment
fn fs_line(in: LineVOut) -> @location(0) vec4<f32> {
    let pj = cam.time.y;
    // High-alpha lines are the projector hover stroke: full brightness with a
    // soft pulse, so they glow against either background.
    if (in.col.a > 0.9) {
        let pulse = 0.80 + 0.20 * sin(cam.time.x * 6.0);
        return vec4<f32>(in.col.rgb * pulse, 1.0);
    }
    return vec4<f32>(in.col.rgb * (0.30 + 0.70 * pj), in.col.a * 0.9 * (0.35 + 0.65 * pj));
}
